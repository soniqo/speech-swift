import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// The Nemotron-H language backbone from the VoiceChat checkpoint.
///
/// The architecture is the 56-layer Nemotron-H backbone from `mlx-swift-lm`,
/// adapted locally so VoiceChat can drive it from fused audio/text embeddings
/// as well as token ids.
public final class VoiceChatLanguageModel {
    public let model: VoiceChatNemotronHModel
    public let configuration: NemotronHConfiguration

    /// The separate tool-call output projection. It shares the LM head's shape
    /// but is a distinct channel, and `NemotronHModel` knows nothing about it,
    /// so it is kept aside for the duplex runtime to bind.
    public let functionHead: VoiceChatMatrix?

    init(
        model: VoiceChatNemotronHModel,
        configuration: NemotronHConfiguration,
        functionHead: VoiceChatMatrix?
    ) {
        self.model = model
        self.configuration = configuration
        self.functionHead = functionHead
    }

    /// Logits over the vocabulary for a batch of token ids.
    ///
    /// A cache is always supplied, even for a one-shot forward pass. This is
    /// load-bearing: `NemotronHBackbone` derives its causal attention mask from
    /// the cache, and returns `.none` when there isn't one — which silently
    /// makes attention bidirectional so every position can see the future.
    /// Generation paths pass a cache anyway, so the bug only surfaces on a
    /// teacher-forced call, where it cost ~6% divergence from the reference.
    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        model(tokens, cache: cache ?? newCache())
    }

    /// Advance one duplex frame from the already-fused input embedding.
    public func call(
        embeddings: MLXArray,
        cache: [KVCache]
    ) -> (logits: MLXArray, hidden: MLXArray) {
        model.call(embeddings: embeddings, cache: cache)
    }

    /// Shared embedding table used by the text and function feedback channels.
    public func embed(_ tokenIDs: MLXArray) -> MLXArray {
        model.embed(tokenIDs)
    }

    /// A fresh cache: `MambaCache` for the recurrent layers, `KVCacheSimple`
    /// for the four attention layers, nothing for the MLP blocks.
    ///
    /// This split is the whole reason the model is cheap to run for long
    /// conversations — only 4 of 56 layers grow with sequence length.
    public func newCache() -> [KVCache] {
        model.newCache(parameters: nil)
    }
}

public extension VoiceChatLanguageModel {
    /// Load an exported `llm/` bundle.
    static func load(from directory: URL) throws -> VoiceChatLanguageModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(NemotronHConfiguration.self, from: configData)

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(weightsURL)
        }

        let quantization = try Self.quantizationSpec(configURL:
            directory.appendingPathComponent("config.json"))

        let model = VoiceChatNemotronHModel(configuration)
        var weights = try MLX.loadArrays(url: weightsURL)

        // Carried outside the module tree because it is a second output channel
        // over the same hidden state. Keep it packed: dense fp32 would exceed
        // two gigabytes for this vocabulary.
        let functionHead: VoiceChatMatrix?
        if weights["function_head.weight"] != nil {
            let tuple = quantization.flatMap {
                $0.perTensor["function_head.weight"]
                    ?? (groupSize: $0.groupSize, bits: $0.bits)
            }
            let spec = tuple.map { QuantizationConfig(groupSize: $0.groupSize, bits: $0.bits) }
            functionHead = try VoiceChatMatrix(
                weights: weights, name: "function_head.weight", quantization: spec)
        } else {
            functionHead = nil
        }
        weights = weights.filter { !$0.key.hasPrefix("function_head") }

        // `sanitize` swaps conv1d axes only when the last dimension is not 1.
        // The export already writes (out, kernel, 1), so this is a no-op here —
        // called anyway so the two stay correct if either side changes.
        weights = model.sanitize(weights: weights)

        if let quantization {
            // Per-tensor bit widths matter: the int5 build holds lm_head and
            // function_head at 8 bits, so one uniform width would rebuild those
            // layers wrong and every logit with them.
            //
            // MLX's `quantize` takes a bit width for the whole call and an
            // `apply` closure that never sees the path, so widths are applied in
            // passes — narrowest-scope first. `quantizeSingle` returns nil for a
            // module that is already `Quantized`, so later passes leave earlier
            // ones alone.
            let specials = Set(quantization.perTensor.values.map { "\($0.groupSize)/\($0.bits)" })
            for spec in specials.sorted() {
                let parts = spec.split(separator: "/").compactMap { Int($0) }
                guard parts.count == 2 else { continue }
                let (groupSize, bits) = (parts[0], parts[1])
                quantize(model: model, groupSize: groupSize, bits: bits) { path, module in
                    guard module is Linear || module is Embedding else { return false }
                    guard let perTensor = quantization.perTensor["\(path).weight"] else { return false }
                    return perTensor.groupSize == groupSize && perTensor.bits == bits
                }
            }
            quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits) {
                path, module in
                guard module is Linear || module is Embedding else { return false }
                return weights["\(path).scales"] != nil
            }
        }

        // Diff before updating: a structural mismatch crashes MLX's verified
        // update rather than throwing, so report it here instead.
        let expected = Set(model.parameters().flattened().map { $0.0 })
        let provided = Set(weights.keys)
        let missing = expected.subtracting(provided).sorted()
        let extra = provided.subtracting(expected).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            throw VoiceChatLoadError.unexpectedKeys(missing + extra)
        }

        model.update(parameters: ModuleParameters.unflattened(weights))
        eval(model)
        return VoiceChatLanguageModel(
            model: model, configuration: configuration, functionHead: functionHead)
    }

    private struct QuantizationSpec {
        let groupSize: Int
        let bits: Int
        let perTensor: [String: (groupSize: Int, bits: Int)]
    }

    private static func quantizationSpec(configURL: URL) throws -> QuantizationSpec? {
        let data = try Data(contentsOf: configURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let quantization = root["quantization"] as? [String: Any],
              let groupSize = quantization["group_size"] as? Int,
              let bits = quantization["bits"] as? Int
        else { return nil }

        var perTensor: [String: (groupSize: Int, bits: Int)] = [:]
        if let table = root["quantization_per_tensor"] as? [String: [String: Int]] {
            for (key, spec) in table {
                if let g = spec["group_size"], let b = spec["bits"] {
                    perTensor[key] = (g, b)
                }
            }
        }
        return QuantizationSpec(groupSize: groupSize, bits: bits, perTensor: perTensor)
    }
}
