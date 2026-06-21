import Foundation
import MLX
import MLXNN
import MLXCommon
import AudioCommon

/// Weight loading for the Mimi codec encoder (`SpeechTokenizerEncoder`).
///
/// Reads `encoder.*` keys from the Qwen3-TTS-Tokenizer-12Hz safetensors
/// (HuggingFace `MimiModel` layout). The SEANet conv stack is a flat `layers`
/// list where activations occupy (param-less) indices: 0 = init conv;
/// per stage k∈0..3 the resnet block is at 1+3k and the strided downsample at
/// 3+3k; 14 = final conv.
extension TTSWeightLoader {

    public static func loadSpeechTokenizerEncoderWeights(
        into encoder: SpeechTokenizerEncoder,
        from directory: URL
    ) throws {
        let w = try CommonWeightLoader.loadAllSafetensors(from: directory)
        logLoad("Found \(w.count) speech tokenizer weights total (encoder load)")

        // MARK: SEANet conv encoder
        let seanet = encoder.seanet
        CommonWeightLoader.applyConv1dWeights(
            to: seanet.initConv.conv, prefix: "encoder.encoder.layers.0.conv", from: w, transpose: true)

        let resnetIdx = [1, 4, 7, 10]
        let downIdx = [3, 6, 9, 12]
        for (k, layer) in seanet.layers.enumerated() {
            let rb = layer.residuals[0]
            CommonWeightLoader.applyConv1dWeights(
                to: rb.convA.conv, prefix: "encoder.encoder.layers.\(resnetIdx[k]).block.1.conv", from: w, transpose: true)
            CommonWeightLoader.applyConv1dWeights(
                to: rb.convB.conv, prefix: "encoder.encoder.layers.\(resnetIdx[k]).block.3.conv", from: w, transpose: true)
            CommonWeightLoader.applyConv1dWeights(
                to: layer.downsample.conv, prefix: "encoder.encoder.layers.\(downIdx[k]).conv", from: w, transpose: true)
        }
        CommonWeightLoader.applyConv1dWeights(
            to: seanet.finalConv.conv, prefix: "encoder.encoder.layers.14.conv", from: w, transpose: true)

        // MARK: Encoder transformer (HF Mimi layout)
        for (i, layer) in encoder.transformer.layers.enumerated() {
            let lp = "encoder.encoder_transformer.layers.\(i)."
            CommonWeightLoader.applyLayerNormWeights(to: layer.inputLayernorm, prefix: lp + "input_layernorm", from: w)
            CommonWeightLoader.applyLayerNormWeights(to: layer.postAttnLayernorm, prefix: lp + "post_attention_layernorm", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.qProj, prefix: lp + "self_attn.q_proj", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.kProj, prefix: lp + "self_attn.k_proj", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.vProj, prefix: lp + "self_attn.v_proj", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.oProj, prefix: lp + "self_attn.o_proj", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.fc1, prefix: lp + "mlp.fc1", from: w)
            CommonWeightLoader.applyLinearWeights(to: layer.fc2, prefix: lp + "mlp.fc2", from: w)
            setParam(layer, key: "self_attn_scale", from: w, weightKey: lp + "self_attn_layer_scale.scale")
            setParam(layer, key: "mlp_scale", from: w, weightKey: lp + "mlp_layer_scale.scale")
        }

        // MARK: Extra ×2 downsample
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.downsample.conv, prefix: "encoder.downsample.conv", from: w, transpose: true)

        // MARK: Split RVQ (semantic + first 15 acoustic)
        let q = encoder.quantizer
        CommonWeightLoader.applyConv1dWeights(
            to: q.semanticRVQ.inputProj.conv,
            prefix: "encoder.quantizer.semantic_residual_vector_quantizer.input_proj", from: w, transpose: true)
        setCodebook(q.semanticRVQ.codebooks[0],
            prefix: "encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook", from: w)

        CommonWeightLoader.applyConv1dWeights(
            to: q.acousticRVQ.inputProj.conv,
            prefix: "encoder.quantizer.acoustic_residual_vector_quantizer.input_proj", from: w, transpose: true)
        for (j, cb) in q.acousticRVQ.codebooks.enumerated() {
            setCodebook(cb,
                prefix: "encoder.quantizer.acoustic_residual_vector_quantizer.layers.\(j).codebook", from: w)
        }

        logLoad("Applied weights to Mimi codec encoder")
    }

    private static func setParam(_ module: Module, key: String, from w: [String: MLXArray], weightKey: String) {
        if let v = w[weightKey] {
            module.update(parameters: ModuleParameters(values: [key: .value(v)]))
        }
    }

    private static func setCodebook(_ cb: MimiEuclideanCodebook, prefix: String, from w: [String: MLXArray]) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let e = w["\(prefix).embed_sum"] { params["embed_sum"] = .value(e) }
        if let c = w["\(prefix).cluster_usage"] { params["cluster_usage"] = .value(c) }
        if !params.isEmpty {
            cb.update(parameters: ModuleParameters(values: params))
        }
    }
}
