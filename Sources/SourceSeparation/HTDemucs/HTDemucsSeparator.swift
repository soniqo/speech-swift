import Foundation
import MLX
import MLXNN
import AudioCommon

/// Bag-of-models source separator for Hybrid Transformer Demucs (Demucs v4).
///
/// `htdemucs_ft` ships 4 fine-tuned sub-models with diagonal combine-weights, so
/// each sub-model contributes only its own stem. Loads all sub-models from the
/// exported safetensors (keys prefixed `model_{i}.`).
public final class HTDemucsSeparator {
    public let config: HTDemucsConfig
    let models: [HTDemucs]

    public static let defaultModelId = "aufklarer/HTDemucs-FT-MLX"

    /// Weight precision. `fp16` is the default published bundle; `int8` is a
    /// smaller pre-quantized bundle (transformer Linear layers quantized).
    public enum Precision: String, Sendable, CaseIterable {
        case fp16, int8
        public var modelName: String { self == .int8 ? "htdemucs_ft_int8" : "htdemucs_ft" }
    }

    init(config: HTDemucsConfig, models: [HTDemucs]) {
        self.config = config
        self.models = models
    }

    /// Download the bag from HuggingFace (default `aufklarer/HTDemucs-FT-MLX`) and
    /// load it. `precision` picks the published bundle: `.fp16` (default, ~320 MB)
    /// or `.int8` (pre-quantized, smaller).
    public static func fromPretrained(
        modelId: String = defaultModelId,
        precision: Precision = .fp16,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> HTDemucsSeparator {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let name = precision.modelName
        progressHandler?(0.0, "Downloading \(name)...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: cacheDir,
            additionalFiles: ["\(name).safetensors", "\(name)_config.json"]
        ) { f in progressHandler?(f * 0.9, "Downloading \(name)...") }
        progressHandler?(0.95, "Loading...")
        let sep = try fromLocal(directory: cacheDir, modelName: name)
        progressHandler?(1.0, "Ready")
        return sep
    }

    /// Load the bag from a local export directory containing
    /// `<model>.safetensors` + `<model>_config.json`.
    ///
    /// If the config carries a `quantization` marker the bundle is pre-quantized
    /// (int8): the quantized module structure is built before loading so the
    /// packed `weight`/`scales`/`biases` keys line up. Otherwise float weights are
    /// cast to float32 (parity-friendly); pass `quantizeBits` to quantize an fp16
    /// bundle in memory instead.
    public static func fromLocal(directory: URL, modelName: String = "htdemucs_ft",
                                 quantizeBits: Int? = nil) throws -> HTDemucsSeparator {
        let cfg = try HTDemucsConfig.load(
            from: directory.appendingPathComponent("\(modelName)_config.json"))
        let all = try MLX.loadArrays(
            url: directory.appendingPathComponent("\(modelName).safetensors"))

        var models: [HTDemucs] = []
        for i in 0..<cfg.numModels {
            let prefix = "model_\(i)."
            var sub: [String: MLXArray] = [:]
            sub.reserveCapacity(all.count / cfg.numModels + 1)
            for (k, v) in all where k.hasPrefix(prefix) {
                // Preserve packed quantized weights (uint32); upcast floats to f32.
                let arr = v.dtype == .uint32 ? v : v.asType(.float32)
                sub[String(k.dropFirst(prefix.count))] = arr
            }
            let model = HTDemucs(cfg)
            // .all => fail on any unused/missing/shape-mismatched key, i.e. the
            // Swift module tree must exactly mirror the exported names.
            if let q = cfg.quantization {
                // Build the quantized structure first so packed keys match.
                quantize(model: model, groupSize: q.groupSize, bits: q.bits) { _, m in m is Linear }
                try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
            } else {
                try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
                if let bits = quantizeBits {
                    // Quantize only Linear layers (cross-transformer FFN/out-proj).
                    // Convs and packed-attention in_proj aren't MLX-quantizable.
                    quantize(model: model, groupSize: 64, bits: bits) { _, m in m is Linear }
                }
            }
            models.append(model)
        }
        return HTDemucsSeparator(config: cfg, models: models)
    }

    /// Build a pre-quantized int8 bundle from a local fp16/fp32 export. Quantizes
    /// the same Linear layers as the in-memory path and writes
    /// `<modelName>_int8.safetensors` + `<modelName>_int8_config.json` to `outDir`
    /// (non-packed params stored as fp16). The result loads via `fromLocal`.
    public static func exportQuantizedBundle(
        fromDirectory dir: URL, modelName: String = "htdemucs_ft",
        toDirectory outDir: URL, bits: Int = 8, groupSize: Int = 64
    ) throws {
        let cfgURL = dir.appendingPathComponent("\(modelName)_config.json")
        let cfg = try HTDemucsConfig.load(from: cfgURL)
        let all = try MLX.loadArrays(url: dir.appendingPathComponent("\(modelName).safetensors"))

        var out: [String: MLXArray] = [:]
        for i in 0..<cfg.numModels {
            let prefix = "model_\(i)."
            var sub: [String: MLXArray] = [:]
            for (k, v) in all where k.hasPrefix(prefix) {
                sub[String(k.dropFirst(prefix.count))] = v.asType(.float32)
            }
            let model = HTDemucs(cfg)
            try model.update(parameters: ModuleParameters.unflattened(sub), verify: .all)
            quantize(model: model, groupSize: groupSize, bits: bits) { _, m in m is Linear }
            for (k, v) in model.parameters().flattened() {
                // Keep packed weights (uint32) exact; store other params as fp16.
                out["\(prefix)\(k)"] = v.dtype == .uint32 ? v : v.asType(.float16)
            }
        }
        eval(Array(out.values))
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        try MLX.save(arrays: out, url: outDir.appendingPathComponent("\(modelName)_int8.safetensors"))

        // int8 config = fp16 config + quantization marker.
        var obj = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        obj["dtype"] = "int8"
        obj["quantization"] = ["bits": bits, "group_size": groupSize]
        let data = try JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: outDir.appendingPathComponent("\(modelName)_int8_config.json"))
    }

    /// Triangular cross-fade weight over `length` (ramp up to the middle, down to
    /// the edges), normalized to max 1. Matches demucs' segment transition weight.
    private func triangularWeight(_ length: Int) -> [Float] {
        let half = length / 2
        var w = [Float](repeating: 0, count: length)
        for i in 0..<half { w[i] = Float(i + 1) }          // 1..half
        for i in half..<length { w[i] = Float(length - i) } // (length-half)..1
        let m = w.max() ?? 1
        return w.map { $0 / m }
    }

    /// Apply one sub-model across the full mixture with overlapping
    /// training-length windows + triangular-weighted overlap-add. `mix`:
    /// [1, C, L] → [1, S, C, L]. (shifts=0, deterministic.)
    private func applySplit(_ model: HTDemucs, _ mix: MLXArray, overlap: Float = 0.25) -> MLXArray {
        let length = mix.dim(-1)
        let S = config.sources.count, C = config.audioChannels
        let segLen = config.trainingLength
        let stride = max(1, Int((1 - overlap) * Float(segLen)))
        let wFull = MLXArray(triangularWeight(segLen))

        var out = MLXArray.zeros([1, S, C, length])
        var sumW = MLXArray.zeros([length])
        var offset = 0
        while offset < length {
            let chunkLen = min(segLen, length - offset)
            let chunk = mix[0..., 0..., offset ..< (offset + chunkLen)]   // model pads internally
            let chunkOut = model(chunk)                                   // [1, S, C, chunkLen]
            let w = wFull[0 ..< chunkLen]
            let right = length - offset - chunkLen
            let pads = [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((offset, right))]
            out = out + padded(chunkOut * w.reshaped([1, 1, 1, chunkLen]), widths: pads)
            sumW = sumW + padded(w, widths: [IntOrPair((offset, right))])
            eval(out, sumW)
            offset += stride
        }
        return out / sumW    // [length] broadcasts over [1, S, C, length]
    }

    /// Separate a mixture into stems. `mix`: [1, audioChannels, L] @ 44.1 kHz.
    /// Returns source name → [1, audioChannels, L]. Diagonal bag: each stem comes
    /// from its own fine-tuned sub-model.
    public func separate(_ mix: MLXArray) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (s, name) in config.sources.enumerated() {
            let full = applySplit(models[s], mix)    // [1, S, C, L]
            let stem = full[0..., s, 0..., 0...]      // [1, C, L]
            eval(stem)
            result[name] = stem
        }
        return result
    }
}
