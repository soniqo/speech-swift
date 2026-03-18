import Foundation
import MLX
import MLXNN
import MLXCommon
import AudioCommon

/// Extension to TTSWeightLoader that adds weight loading for SpeechTokenizerEncoder.
/// Mirrors loadSpeechTokenizerDecoderWeights() using the same safetensors file,
/// but reads from "encoder.*" prefixes instead of "decoder.*".
extension TTSWeightLoader {

    // MARK: - Speech Tokenizer Encoder Weight Loading

    public static func loadSpeechTokenizerEncoderWeights(
        into encoder: SpeechTokenizerEncoder,
        from directory: URL
    ) throws {
        let allWeights = try CommonWeightLoader.loadAllSafetensors(from: directory)

        print("Found \(allWeights.count) speech tokenizer weights total (encoder load)")

        // RVQ codebook weights — same codebooks as decoder, stored under encoder prefix
        loadEncoderRVQWeights(into: encoder.rvq, from: allWeights)

        // Input conv: encoder.encoder.0.conv  (mirrors decoder.decoder.6.conv reversed)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.inputConv.conv,
            prefix: "encoder.encoder.0.conv",
            from: allWeights,
            transpose: true)

        // Encoder blocks: encoder.encoder.{1,2,3,4}
        // Each block: residualUnits (act1/conv1/act2/conv2) + snake + strided conv
        for (i, block) in encoder.encoderBlocks.enumerated() {
            loadEncoderBlockWeights(to: block, blockKey: "encoder.encoder.\(i + 1)", from: allWeights)
        }

        // Channel projection: encoder.encoder.5.conv (mirrors decoder preConv)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.channelProj.conv,
            prefix: "encoder.encoder.5.conv",
            from: allWeights,
            transpose: true)

        // Post-downsample ConvNeXt + strided conv stages: encoder.downsample.{0,1}
        // Stage 0: .0 = ConvNeXt, .1.conv = strided conv
        loadConvNeXtBlockWeights(to: encoder.postConvNeXt1, prefix: "encoder.downsample.0.0", from: allWeights)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.postDownsample1.conv,
            prefix: "encoder.downsample.0.1.conv",
            from: allWeights,
            transpose: true)
        // Stage 1
        loadConvNeXtBlockWeights(to: encoder.postConvNeXt2, prefix: "encoder.downsample.1.0", from: allWeights)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.postDownsample2.conv,
            prefix: "encoder.downsample.1.1.conv",
            from: allWeights,
            transpose: true)

        // Post-conv: encoder.post_conv.conv  (mirrors decoder.pre_conv.conv)
        CommonWeightLoader.applyConv1dWeights(
            to: encoder.postConv.conv,
            prefix: "encoder.post_conv.conv",
            from: allWeights,
            transpose: true)

        // Encoder transformer
        CommonWeightLoader.applyLinearWeights(
            to: encoder.transformer.inputProj,
            prefix: "encoder.pre_transformer.input_proj",
            from: allWeights)
        CommonWeightLoader.applyLinearWeights(
            to: encoder.transformer.outputProj,
            prefix: "encoder.pre_transformer.output_proj",
            from: allWeights)
        for (i, layer) in encoder.transformer.layers.enumerated() {
            loadEncoderTransformerLayerWeights(to: layer, index: i, from: allWeights)
        }
        CommonWeightLoader.applyRMSNormWeights(
            to: encoder.transformer.norm,
            prefix: "encoder.pre_transformer.norm",
            from: allWeights)

        print("Applied weights to Speech Tokenizer Encoder")
    }

    // MARK: - Encoder RVQ

    private static func loadEncoderRVQWeights(
        into rvq: EncoderRVQ,
        from weights: [String: MLXArray]
    ) {
        // rvq_first semantic codebook
        loadEncoderQuantizerCodebook(
            into: rvq.rvqFirst.quantizers[0].embedding,
            prefix: "encoder.quantizer.rvq_first.vq.layers.0._codebook",
            from: weights)
        // rvq_first input_proj (Conv1d 512->256, kernel=1)
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.rvqFirst.outputProj,
            prefix: "encoder.quantizer.rvq_first.input_proj",
            from: weights,
            transpose: true)

        // rvq_rest: 15 acoustic codebooks
        for i in 0..<rvq.rvqRest.numQuantizers {
            loadEncoderQuantizerCodebook(
                into: rvq.rvqRest.quantizers[i].embedding,
                prefix: "encoder.quantizer.rvq_rest.vq.layers.\(i)._codebook",
                from: weights)
        }
        // rvq_rest input_proj
        CommonWeightLoader.applyConv1dWeights(
            to: rvq.rvqRest.outputProj,
            prefix: "encoder.quantizer.rvq_rest.input_proj",
            from: weights,
            transpose: true)
    }

    private static func loadEncoderQuantizerCodebook(
        into embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        if let embed = weights["\(prefix).embed"] {
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(embed)]
            embedding.update(parameters: ModuleParameters(values: params))
            return
        }
        if let usage = weights["\(prefix).cluster_usage"],
           let embSum = weights["\(prefix).embedding_sum"] {
            let eps = MLXArray(Float(1e-7))
            let clampedUsage = maximum(usage, eps).expandedDimensions(axis: -1)
            let computed = embSum / clampedUsage
            let params: [String: NestedItem<String, MLXArray>] = ["weight": .value(computed)]
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    // MARK: - Encoder Block Weights

    /// Load weights for one EncoderBlock.
    /// Safetensors key structure (encoder side, mirror of decoder):
    ///   blockKey.block.{0,1,2} = 3 ResidualUnits (act1/conv1/act2/conv2)
    ///   blockKey.block.3       = SnakeBeta activation
    ///   blockKey.block.4.conv  = strided CausalConv1d (downsample)
    private static func loadEncoderBlockWeights(
        to block: EncoderBlock,
        blockKey: String,
        from weights: [String: MLXArray]
    ) {
        // Residual units: block.{0,1,2}
        for (j, unit) in block.residualUnits.enumerated() {
            let resPrefix = "\(blockKey).block.\(j)"
            loadSnakeBetaWeights(to: unit.snake1, prefix: "\(resPrefix).act1", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv1.conv, prefix: "\(resPrefix).conv1.conv", from: weights, transpose: true)
            loadSnakeBetaWeights(to: unit.snake2, prefix: "\(resPrefix).act2", from: weights)
            CommonWeightLoader.applyConv1dWeights(
                to: unit.conv2.conv, prefix: "\(resPrefix).conv2.conv", from: weights, transpose: true)
        }

        // Snake activation before downsample: block.3
        loadSnakeBetaWeights(to: block.snake, prefix: "\(blockKey).block.3", from: weights)

        // Strided conv (downsample): block.4.conv
        CommonWeightLoader.applyConv1dWeights(
            to: block.downsample.conv, prefix: "\(blockKey).block.4.conv", from: weights, transpose: true)
    }

    // MARK: - Encoder Transformer Layer Weights

    private static func loadEncoderTransformerLayerWeights(
        to layer: DecoderTransformerLayer,
        index: Int,
        from weights: [String: MLXArray]
    ) {
        let prefix = "encoder.pre_transformer.layers.\(index)"

        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.qProj, prefix: "\(prefix).self_attn.q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.kProj, prefix: "\(prefix).self_attn.k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.vProj, prefix: "\(prefix).self_attn.v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.selfAttn.oProj, prefix: "\(prefix).self_attn.o_proj", from: weights)

        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm1, prefix: "\(prefix).input_layernorm", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: layer.norm2, prefix: "\(prefix).post_attention_layernorm", from: weights)

        CommonWeightLoader.applyLinearWeights(
            to: layer.gateProj, prefix: "\(prefix).mlp.gate_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.upProj, prefix: "\(prefix).mlp.up_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: layer.downProj, prefix: "\(prefix).mlp.down_proj", from: weights)

        if let scale = weights["\(prefix).self_attn_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.attnLayerScale.update(parameters: ModuleParameters(values: params))
        }
        if let scale = weights["\(prefix).mlp_layer_scale.scale"] {
            let params: [String: NestedItem<String, MLXArray>] = ["scale": .value(scale.reshaped([1, 1, -1]))]
            layer.mlpLayerScale.update(parameters: ModuleParameters(values: params))
        }
    }
}
