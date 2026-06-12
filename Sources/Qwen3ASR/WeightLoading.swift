import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon

/// Weight loading utilities for Qwen3-ASR
/// Uses direct HuggingFace key paths — model structure must match exactly.
///
/// All loaders stream weights per safetensors shard rather than accumulating
/// every tensor from every file into a single `[String: MLXArray]` dict before
/// applying. This keeps transient load-time peak memory at roughly
/// `model_size + one_shard` instead of `model_size + checkpoint_size` (the
/// shards plus the assembled dict were duplicating the entire model in RAM
/// during load — observed ~1.5–2.0 GB peak on 1.7B). `Module.update(parameters:)`
/// is documented as partial-safe (mlx-swift `Module.swift:423`: "any omitted
/// values will be unchanged"), so applying every component against every shard
/// is correct even when a single layer's tensors are split across files.
public enum WeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try CommonWeightLoader.loadSafetensors(url: url)
    }

    /// Load and apply weights to model using HuggingFace key paths directly.
    /// Streams per shard — see file docstring.
    public static func loadWeights(
        into audioEncoder: Qwen3AudioEncoder,
        from directory: URL
    ) throws {
        let files = try safetensorFiles(in: directory)
        print("Found \(files.count) safetensor files")

        var appliedTotal = 0
        for file in files {
            print("Loading: \(file.lastPathComponent)")
            let raw = try loadSafetensors(url: file)
            let audioTowerWeights = stripPrefix(raw, prefix: "audio_tower.")
            if audioTowerWeights.isEmpty { continue }
            applyAudioEncoderComponents(
                to: audioEncoder, weights: audioTowerWeights,
                transposeConv2dPyTorch: false)
            appliedTotal += audioTowerWeights.count
            // `raw` and `audioTowerWeights` go out of scope here; their
            // MLXArray references release once each call to
            // `update(parameters:)` above has adopted the tensors the
            // model actually needed.
        }
        print("Applied weights to audio encoder (\(audioEncoder.layers.count) layers, \(appliedTotal) tensors)")
    }

    /// Load and apply weights to quantized text decoder. Per-shard streaming.
    public static func loadTextDecoderWeights(
        into textModel: QuantizedTextModel,
        from directory: URL
    ) throws {
        let files = try safetensorFiles(in: directory)
        var appliedTotal = 0
        for file in files {
            let raw = try loadSafetensors(url: file)
            let textWeights = stripPrefix(raw, prefix: "model.")
            if textWeights.isEmpty { continue }
            applyQuantizedTextDecoderComponents(to: textModel, weights: textWeights)
            appliedTotal += textWeights.count
        }
        print("Applied weights to text decoder (\(textModel.layers.count) layers, \(appliedTotal) tensors)")
    }

    // MARK: - Forced Aligner Weight Loading

    /// Load weights for the forced aligner model. Per-shard streaming.
    ///
    /// Weight key structure (under optional `thinker.` prefix):
    ///   - `audio_tower.*` → audio encoder
    ///   - `model.*` → text decoder (quantized or float)
    ///   - `lm_head.*` → classify head (Linear, NOT quantized)
    public static func loadForcedAlignerWeights(
        into model: Qwen3ForcedAligner,
        from directory: URL
    ) throws {
        let files = try safetensorFiles(in: directory)

        var audioApplied = 0
        var textApplied = 0
        var headApplied = 0

        for file in files {
            print("Loading: \(file.lastPathComponent)")
            let raw = try loadSafetensors(url: file)

            // Strip `thinker.` if present so downstream prefix-strip logic
            // sees a uniform key space.
            let normalized = stripPrefix(raw, prefix: "thinker.", keepUnprefixed: true)

            let audioTowerWeights = stripPrefix(normalized, prefix: "audio_tower.")
            if !audioTowerWeights.isEmpty {
                applyAudioEncoderComponents(
                    to: model.audioEncoder, weights: audioTowerWeights,
                    transposeConv2dPyTorch: true)
                audioApplied += audioTowerWeights.count
            }

            let textWeights = stripPrefix(normalized, prefix: "model.")
            if !textWeights.isEmpty {
                if let quantized = model.textDecoder as? QuantizedTextModel {
                    applyQuantizedTextDecoderComponents(to: quantized, weights: textWeights)
                } else if let floatModel = model.textDecoder as? FloatTextModel {
                    applyFloatTextDecoderComponents(to: floatModel, weights: textWeights)
                }
                textApplied += textWeights.count
            }

            let classifyWeights = filterPrefix(normalized, keepingPrefix: "lm_head.")
            if !classifyWeights.isEmpty {
                CommonWeightLoader.applyLinearWeights(
                    to: model.classifyHead, prefix: "lm_head", from: classifyWeights)
                headApplied += classifyWeights.count
            }
        }

        print("Audio tower: \(audioApplied), Text decoder: \(textApplied), Classify head: \(headApplied)")
        print("Applied audio encoder weights (\(model.audioEncoder.layers.count) layers)")
        if let quantized = model.textDecoder as? QuantizedTextModel {
            print("Applied quantized text decoder weights (\(quantized.layers.count) layers)")
        } else if let floatModel = model.textDecoder as? FloatTextModel {
            print("Applied float text decoder weights (\(floatModel.layers.count) layers)")
        }
        print("Applied classify head weights")
    }

    // MARK: - Shard discovery + prefix filter helpers

    private static func safetensorFiles(in directory: URL) throws -> [URL] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let files = contents.filter { $0.pathExtension == "safetensors" }
        guard !files.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }
        // Sort lexicographically so the load order is deterministic across
        // runs — purely cosmetic for the logs, but it makes regression
        // diffs against captured stderr stable.
        return files.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Return a new dict containing only keys with `prefix`, with the prefix
    /// stripped off. `keepUnprefixed: true` retains keys that DON'T have the
    /// prefix (used by the forced aligner where `thinker.` is optional).
    private static func stripPrefix(
        _ weights: [String: MLXArray],
        prefix: String,
        keepUnprefixed: Bool = false
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.hasPrefix(prefix) {
                out[String(key.dropFirst(prefix.count))] = value
            } else if keepUnprefixed {
                out[key] = value
            }
        }
        return out
    }

    /// Return a new dict containing only keys with `keepingPrefix`,
    /// preserving the prefix on the kept keys.
    private static func filterPrefix(
        _ weights: [String: MLXArray],
        keepingPrefix: String
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix(keepingPrefix) {
            out[key] = value
        }
        return out
    }

    // MARK: - Per-shard component application

    /// Apply every audio-encoder component slot against `weights`. Components
    /// whose tensors aren't present in this shard are no-ops (each
    /// `apply…Weights` helper guards its `weights[key]` lookups).
    private static func applyAudioEncoderComponents(
        to audioEncoder: Qwen3AudioEncoder,
        weights: [String: MLXArray],
        transposeConv2dPyTorch: Bool
    ) {
        applyConv2dWeights(to: audioEncoder.conv2d1, prefix: "conv2d1", from: weights, transposePyTorch: transposeConv2dPyTorch)
        applyConv2dWeights(to: audioEncoder.conv2d2, prefix: "conv2d2", from: weights, transposePyTorch: transposeConv2dPyTorch)
        applyConv2dWeights(to: audioEncoder.conv2d3, prefix: "conv2d3", from: weights, transposePyTorch: transposeConv2dPyTorch)
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.convOut, prefix: "conv_out", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: audioEncoder.lnPost, prefix: "ln_post", from: weights)
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.proj1, prefix: "proj1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: audioEncoder.proj2, prefix: "proj2", from: weights)
        for (index, layer) in audioEncoder.layers.enumerated() {
            applyEncoderLayerWeights(to: layer, prefix: "layers.\(index)", from: weights)
        }
    }

    private static func applyQuantizedTextDecoderComponents(
        to textModel: QuantizedTextModel,
        weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyQuantizedEmbeddingWeights(
            to: textModel.embedTokens, prefix: "embed_tokens", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: textModel.norm, prefix: "norm", from: weights)
        for (index, layer) in textModel.layers.enumerated() {
            applyQuantizedDecoderLayerWeights(
                to: layer, prefix: "layers.\(index)", from: weights)
        }
    }

    private static func applyFloatTextDecoderComponents(
        to textModel: FloatTextModel,
        weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyEmbeddingWeights(
            to: textModel.embedTokens, prefix: "embed_tokens", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: textModel.norm, prefix: "norm", from: weights)
        for (index, layer) in textModel.layers.enumerated() {
            applyFloatDecoderLayerWeights(
                to: layer, prefix: "layers.\(index)", from: weights)
        }
    }

    // MARK: - ASR-specific Weight Application Helpers

    private static func applyQuantizedDecoderLayerWeights(
        to layer: QuantizedTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        // Q/K norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

        // Layer norms
        CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)

        // MLP
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyQuantizedLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    private static func applyFloatDecoderLayerWeights(
        to layer: FloatTextDecoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.selfAttn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(prefix).post_attention_layernorm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)
    }

    // MARK: - Audio Encoder Weight Helpers

    private static func applyConv2dWeights(
        to conv: Conv2d,
        prefix: String,
        from weights: [String: MLXArray],
        transposePyTorch: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            if transposePyTorch {
                // PyTorch Conv2d: [outC, inC, kH, kW] -> MLX Conv2d: [outC, kH, kW, inC]
                params["weight"] = .value(weight.transposed(0, 2, 3, 1))
            } else {
                params["weight"] = .value(weight)
            }
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    private static func applyEncoderLayerWeights(
        to layer: AudioEncoderLayer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        // Self attention
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.selfAttn.outProj, prefix: "\(prefix).self_attn.out_proj", from: weights)

        // Layer norms
        CommonWeightLoader.applyLayerNormWeights(to: layer.selfAttnLayerNorm, prefix: "\(prefix).self_attn_layer_norm", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: layer.finalLayerNorm, prefix: "\(prefix).final_layer_norm", from: weights)

        // FFN
        CommonWeightLoader.applyLinearWeights(to: layer.fc1, prefix: "\(prefix).fc1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: layer.fc2, prefix: "\(prefix).fc2", from: weights)
    }
}
