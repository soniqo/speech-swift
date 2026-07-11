import Foundation
import MLX
import MLXCommon

/// Applies remapped checkpoint tensors onto the runtime modules using the
/// shared per-module loaders, failing loudly on any missing tensor.
enum HiggsTTSWeightLoading {
    static func apply(_ weights: [String: MLXArray], to backbone: HiggsTTSBackbone) throws {
        func require(_ key: String) throws {
            guard weights[key] != nil else {
                throw HiggsTTSError.missingRequiredFile("tensor \(key)")
            }
        }

        try require("embed_tokens.weight")
        CommonWeightLoader.applyEmbeddingWeights(
            to: backbone.embedTokens, prefix: "embed_tokens", from: weights)

        for (index, layer) in backbone.layers.enumerated() {
            let prefix = "layers.\(index)"
            let attn = layer.attn
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
                try require("\(prefix).self_attn.\(name).weight")
            }
            CommonWeightLoader.applyLinearWeights(
                to: attn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attn.qNorm, prefix: "\(prefix).self_attn.q_norm", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attn.kNorm, prefix: "\(prefix).self_attn.k_norm", from: weights)

            for name in ["gate_proj", "up_proj", "down_proj"] {
                try require("\(prefix).mlp.\(name).weight")
            }
            CommonWeightLoader.applyLinearWeights(
                to: layer.mlp.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.mlp.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.mlp.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)

            try require("\(prefix).input_layernorm.weight")
            try require("\(prefix).post_attention_layernorm.weight")
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm, prefix: "\(prefix).input_layernorm", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm,
                prefix: "\(prefix).post_attention_layernorm",
                from: weights)
        }

        try require("norm.weight")
        CommonWeightLoader.applyRMSNormWeights(to: backbone.norm, prefix: "norm", from: weights)
    }

    static func apply(_ weights: [String: MLXArray], to fused: HiggsTTSFusedCodebook) throws {
        guard weights["embedding.weight"] != nil else {
            throw HiggsTTSError.missingRequiredFile("tensor embedding.weight")
        }
        CommonWeightLoader.applyEmbeddingWeights(
            to: fused.embedding, prefix: "embedding", from: weights)
    }
}
