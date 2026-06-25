import Foundation
import MLXCommon
import MLX
import MLXNN

/// Loads quantized safetensors weights into `Gemma4Model`.
///
/// Gemma 4 multimodal MLX checkpoint key layout — keys are prefixed `language_model.model.`
/// (the text tower of a multimodal model). We strip that prefix to bare keys:
///   - `embed_tokens.{weight,scales,biases}`                 → tied embedding / lm_head
///   - `embed_tokens_per_layer.{weight,scales,biases}`       → per-layer embedding table (PLE)
///   - `per_layer_model_projection.{weight,scales,biases}`   → PLE projection
///   - `per_layer_projection_norm.weight`                    → PLE projection RMSNorm
///   - `norm.weight`                                         → final RMSNorm
///   - `layers.{i}.self_attn.{q,k,v,o}_proj.{weight,scales,biases}` (k/v absent on KV-shared layers)
///   - `layers.{i}.self_attn.{q,k}_norm.weight`              (k_norm absent on KV-shared layers; no v_norm weight)
///   - `layers.{i}.mlp.{gate,up,down}_proj.{weight,scales,biases}`
///   - `layers.{i}.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight`
///   - `layers.{i}.per_layer_input_gate.{weight,scales,biases}`
///   - `layers.{i}.per_layer_projection.{weight,scales,biases}`
///   - `layers.{i}.post_per_layer_input_norm.weight`
///   - `layers.{i}.layer_scalar`
///
/// `vision_tower.*` / `audio_tower.*` keys (and the redundant k/v keys some exports emit for KV-shared
/// layers) are simply ignored — the loader only wires the weights the model actually owns.
public enum Gemma4WeightLoader {
    public static func loadWeights(
        into model: Gemma4Model,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading weight files...")
        let all = try CommonWeightLoader.loadAllSafetensors(from: directory)
        progressHandler?(0.3, "Loaded \(all.count) tensors")

        // Strip the `language_model.model.` prefix (multimodal text tower). Also tolerate the plain
        // `model.` layout in case a text-only checkpoint is used.
        var w: [String: MLXArray] = [:]
        for (key, value) in all {
            if key.hasPrefix("language_model.model.") {
                w[String(key.dropFirst("language_model.model.".count))] = value
            } else if key.hasPrefix("model.") {
                w[String(key.dropFirst("model.".count))] = value
            } else {
                w[key] = value
            }
        }

        progressHandler?(0.4, "Embeddings + PLE + final norm...")
        CommonWeightLoader.applyQuantizedEmbeddingWeights(to: model.embedTokens, prefix: "embed_tokens", from: w)
        CommonWeightLoader.applyQuantizedEmbeddingWeights(to: model.embedTokensPerLayer, prefix: "embed_tokens_per_layer", from: w)
        CommonWeightLoader.applyQuantizedLinearWeights(to: model.perLayerModelProjection, prefix: "per_layer_model_projection", from: w)
        CommonWeightLoader.applyRMSNormWeights(to: model.perLayerProjectionNorm, prefix: "per_layer_projection_norm", from: w)
        CommonWeightLoader.applyRMSNormWeights(to: model.norm, prefix: "norm", from: w)

        progressHandler?(0.5, "Transformer layers...")
        let n = model.config.numHiddenLayers
        for i in 0..<n {
            let p = "layers.\(i)"
            let layer = model.layers[i]

            // Sandwich norms.
            CommonWeightLoader.applyRMSNormWeights(to: layer.inputLayerNorm, prefix: "\(p).input_layernorm", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: layer.postAttentionLayerNorm, prefix: "\(p).post_attention_layernorm", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: layer.preFeedforwardLayerNorm, prefix: "\(p).pre_feedforward_layernorm", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: layer.postFeedforwardLayerNorm, prefix: "\(p).post_feedforward_layernorm", from: w)

            // Attention.
            let attn = layer.attn
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.qProj, prefix: "\(p).self_attn.q_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: attn.oProj, prefix: "\(p).self_attn.o_proj", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: attn.qNorm, prefix: "\(p).self_attn.q_norm", from: w)
            if let kProj = attn.kProj { CommonWeightLoader.applyQuantizedLinearWeights(to: kProj, prefix: "\(p).self_attn.k_proj", from: w) }
            if let vProj = attn.vProj { CommonWeightLoader.applyQuantizedLinearWeights(to: vProj, prefix: "\(p).self_attn.v_proj", from: w) }
            if let kNorm = attn.kNorm { CommonWeightLoader.applyRMSNormWeights(to: kNorm, prefix: "\(p).self_attn.k_norm", from: w) }

            // MLP.
            let mlp = layer.mlp
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(p).mlp.gate_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.upProj, prefix: "\(p).mlp.up_proj", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: mlp.downProj, prefix: "\(p).mlp.down_proj", from: w)

            // Per-layer input gating.
            CommonWeightLoader.applyQuantizedLinearWeights(to: layer.perLayerInputGate, prefix: "\(p).per_layer_input_gate", from: w)
            CommonWeightLoader.applyQuantizedLinearWeights(to: layer.perLayerProjection, prefix: "\(p).per_layer_projection", from: w)
            CommonWeightLoader.applyRMSNormWeights(to: layer.postPerLayerInputNorm, prefix: "\(p).post_per_layer_input_norm", from: w)

            // Layer scalar. NB: the `@ParameterInfo(key:)` is lost once reassigned in init(), so the
            // update key must be the Swift property name (`layerScalar`), not the safetensors key.
            if let ls = w["\(p).layer_scalar"] {
                layer.update(parameters: ModuleParameters(values: ["layerScalar": .value(ls)]))
            }

            progressHandler?(0.5 + 0.45 * Double(i + 1) / Double(n), "Layer \(i + 1)/\(n)")
        }

        eval(model)
        progressHandler?(1.0, "Weights loaded")
    }
}
