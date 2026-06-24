import Foundation
import MLXCommon
import MLX
import MLXNN

/// Loads quantized safetensors weights into `Qwen3DenseModel`.
///
/// Standard Qwen3 key layout (mlx-community / mlx_lm format), `model.` prefix stripped:
///   - `embed_tokens.{weight,scales,biases}`           → embed_tokens (PreQuantizedEmbedding)
///   - `norm.weight`                                    → final RMSNorm
///   - `layers.{i}.input_layernorm.weight`
///   - `layers.{i}.post_attention_layernorm.weight`
///   - `layers.{i}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}`
///   - `layers.{i}.self_attn.{q,k}_norm.weight`
///   - `layers.{i}.mlp.{gate,up,down}_proj.{weight,scales,biases}`
///   - `lm_head.{weight,scales,biases}` (only when embeddings are untied)
public enum Qwen3DenseWeightLoader {
    public static func loadWeights(
        into model: Qwen3DenseModel,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading weight files...")
        let all = try CommonWeightLoader.loadAllSafetensors(from: directory)
        progressHandler?(0.3, "Loaded \(all.count) tensors")

        var w: [String: MLXArray] = [:]
        for (key, value) in all {
            if key.hasPrefix("model.") {
                w[String(key.dropFirst("model.".count))] = value
            } else {
                w[key] = value   // keeps top-level "lm_head.*"
            }
        }

        progressHandler?(0.4, "Embeddings + final norm...")
        CommonWeightLoader.applyQuantizedEmbeddingWeights(to: model.embedTokens, prefix: "embed_tokens", from: w)
        CommonWeightLoader.applyRMSNormWeights(to: model.norm, prefix: "norm", from: w)
        if let lmHead = model.lmHead {
            CommonWeightLoader.applyQuantizedLinearWeights(to: lmHead, prefix: "lm_head", from: w)
        }

        progressHandler?(0.5, "Transformer layers...")
        let n = model.config.numHiddenLayers
        for i in 0..<n {
            let p = "layers.\(i)"
            let layer = model.layers[i]
            CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(p).input_layernorm", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(p).post_attention_layernorm", from: w)

            let attn = layer.attn
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.qProj, prefix: "\(p).self_attn.q_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.kProj, prefix: "\(p).self_attn.k_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.vProj, prefix: "\(p).self_attn.v_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.oProj, prefix: "\(p).self_attn.o_proj", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: attn.qNorm, prefix: "\(p).self_attn.q_norm", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: attn.kNorm, prefix: "\(p).self_attn.k_norm", from: w)

            let mlp = layer.mlp
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(p).mlp.gate_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.upProj, prefix: "\(p).mlp.up_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.downProj, prefix: "\(p).mlp.down_proj", from: w)

            progressHandler?(0.5 + 0.45 * Double(i + 1) / Double(n), "Layer \(i + 1)/\(n)")
        }

        eval(model)
        progressHandler?(1.0, "Weights loaded")
    }
}
