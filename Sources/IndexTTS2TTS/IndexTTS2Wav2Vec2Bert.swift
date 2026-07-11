import Foundation
import MLX
import MLXNN

private enum IndexTTS2Wav2Vec2BertConfig {
    static let featureDim = 160
    static let hiddenSize = 1024
    static let intermediateSize = 4096
    static let numLayers = 24
    static let numHeads = 16
    static let headSize = 64
    static let convKernel = 31
    static let layerNormEps: Float = 1e-5
    static let leftMaxPosition = 64
    static let rightMaxPosition = 8
    static let relativePositionCount = leftMaxPosition + rightMaxPosition + 1
}

final class IndexTTS2Wav2Vec2BertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    override init() {
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: IndexTTS2Wav2Vec2BertConfig.featureDim,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _projection.wrappedValue = Linear(
            IndexTTS2Wav2Vec2BertConfig.featureDim,
            IndexTTS2Wav2Vec2BertConfig.hiddenSize,
            bias: true)
        super.init()
    }

    func callAsFunction(_ inputFeatures: MLXArray) -> MLXArray {
        projection(layerNorm(inputFeatures))
    }
}

final class IndexTTS2Wav2Vec2BertFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    override init() {
        _intermediateDense.wrappedValue = Linear(
            IndexTTS2Wav2Vec2BertConfig.hiddenSize,
            IndexTTS2Wav2Vec2BertConfig.intermediateSize,
            bias: true)
        _outputDense.wrappedValue = Linear(
            IndexTTS2Wav2Vec2BertConfig.intermediateSize,
            IndexTTS2Wav2Vec2BertConfig.hiddenSize,
            bias: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        outputDense(silu(intermediateDense(hiddenStates)))
    }
}

final class IndexTTS2Wav2Vec2BertConvolutionModule: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "depthwise_layer_norm") var depthwiseLayerNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    override init() {
        let hidden = IndexTTS2Wav2Vec2BertConfig.hiddenSize
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: hidden,
            outputChannels: hidden * 2,
            kernelSize: 1,
            bias: false)
        _depthwiseConv.wrappedValue = Conv1d(
            inputChannels: hidden,
            outputChannels: hidden,
            kernelSize: IndexTTS2Wav2Vec2BertConfig.convKernel,
            groups: hidden,
            bias: false)
        _depthwiseLayerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: hidden,
            outputChannels: hidden,
            kernelSize: 1,
            bias: false)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let hidden = IndexTTS2Wav2Vec2BertConfig.hiddenSize
        var h = layerNorm(hiddenStates)
        h = pointwiseConv1(h)
        let gate = h[0..., 0..., 0..<hidden] * sigmoid(h[0..., 0..., hidden..<(hidden * 2)])
        h = MLX.padded(
            gate,
            widths: [
                .init((0, 0)),
                .init((IndexTTS2Wav2Vec2BertConfig.convKernel - 1, 0)),
                .init((0, 0)),
            ])
        h = depthwiseConv(h)
        h = depthwiseLayerNorm(h)
        h = silu(h)
        return pointwiseConv2(h)
    }
}

final class IndexTTS2Wav2Vec2BertSelfAttention: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "distance_embedding") var distanceEmbedding: Embedding

    override init() {
        let hidden = IndexTTS2Wav2Vec2BertConfig.hiddenSize
        _linearQ.wrappedValue = Linear(hidden, hidden, bias: true)
        _linearK.wrappedValue = Linear(hidden, hidden, bias: true)
        _linearV.wrappedValue = Linear(hidden, hidden, bias: true)
        _linearOut.wrappedValue = Linear(hidden, hidden, bias: true)
        _distanceEmbedding.wrappedValue = Embedding(
            embeddingCount: IndexTTS2Wav2Vec2BertConfig.relativePositionCount,
            dimensions: IndexTTS2Wav2Vec2BertConfig.headSize)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let b = hiddenStates.dim(0)
        let t = hiddenStates.dim(1)
        let heads = IndexTTS2Wav2Vec2BertConfig.numHeads
        let headSize = IndexTTS2Wav2Vec2BertConfig.headSize
        let hidden = IndexTTS2Wav2Vec2BertConfig.hiddenSize
        let scale = MLXArray(1.0 / Foundation.sqrt(Float(headSize))).asType(hiddenStates.dtype)

        let q = linearQ(hiddenStates)
            .reshaped([b, t, heads, headSize])
            .transposed(0, 2, 1, 3)
        let k = linearK(hiddenStates)
            .reshaped([b, t, heads, headSize])
            .transposed(0, 2, 1, 3)
        let v = linearV(hiddenStates)
            .reshaped([b, t, heads, headSize])
            .transposed(0, 2, 1, 3)

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
        scores = scores + relativeKeyScores(q, sequenceLength: t) * scale

        let probs = softmax(scores.asType(.float32), axis: -1).asType(hiddenStates.dtype)
        var out = matmul(probs, v)
        out = out.transposed(0, 2, 1, 3).reshaped([b, t, hidden])
        return linearOut(out)
    }

    private func relativeKeyScores(_ query: MLXArray, sequenceLength t: Int) -> MLXArray {
        let left = IndexTTS2Wav2Vec2BertConfig.leftMaxPosition
        let right = IndexTTS2Wav2Vec2BertConfig.rightMaxPosition
        var ids: [Int32] = []
        ids.reserveCapacity(t * t)
        for i in 0..<t {
            for j in 0..<t {
                let distance = min(max(j - i, -left), right) + left
                ids.append(Int32(distance))
            }
        }
        let distance = MLXArray(ids, [t, t])
        let positional = distanceEmbedding(distance).asType(query.dtype)
            .reshaped([1, 1, t, t, IndexTTS2Wav2Vec2BertConfig.headSize])
        return (query.expandedDimensions(axis: 3) * positional).sum(axis: -1)
    }
}

final class IndexTTS2Wav2Vec2BertEncoderLayer: Module {
    @ModuleInfo(key: "ffn1_layer_norm") var ffn1LayerNorm: LayerNorm
    @ModuleInfo(key: "ffn1") var ffn1: IndexTTS2Wav2Vec2BertFeedForward
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: IndexTTS2Wav2Vec2BertSelfAttention
    @ModuleInfo(key: "conv_module") var convolution: IndexTTS2Wav2Vec2BertConvolutionModule
    @ModuleInfo(key: "ffn2_layer_norm") var ffn2LayerNorm: LayerNorm
    @ModuleInfo(key: "ffn2") var ffn2: IndexTTS2Wav2Vec2BertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        let hidden = IndexTTS2Wav2Vec2BertConfig.hiddenSize
        _ffn1LayerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _ffn1.wrappedValue = IndexTTS2Wav2Vec2BertFeedForward()
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _selfAttention.wrappedValue = IndexTTS2Wav2Vec2BertSelfAttention()
        _convolution.wrappedValue = IndexTTS2Wav2Vec2BertConvolutionModule()
        _ffn2LayerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        _ffn2.wrappedValue = IndexTTS2Wav2Vec2BertFeedForward()
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: hidden,
            eps: IndexTTS2Wav2Vec2BertConfig.layerNormEps,
            affine: true)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var residual = hiddenStates
        var h = ffn1(ffn1LayerNorm(hiddenStates)) * 0.5 + residual

        residual = h
        h = selfAttention(selfAttentionLayerNorm(h)) + residual

        residual = h
        h = convolution(h) + residual

        residual = h
        h = ffn2(ffn2LayerNorm(h)) * 0.5 + residual
        return finalLayerNorm(h)
    }
}

final class IndexTTS2Wav2Vec2BertEncoder: Module {
    @ModuleInfo var layers: [IndexTTS2Wav2Vec2BertEncoderLayer]

    override init() {
        _layers.wrappedValue = (0..<IndexTTS2Wav2Vec2BertConfig.numLayers).map { _ in
            IndexTTS2Wav2Vec2BertEncoderLayer()
        }
        super.init()
    }
}

final class IndexTTS2Wav2Vec2Bert: Module {
    @ModuleInfo(key: "feature_projection") var featureProjection: IndexTTS2Wav2Vec2BertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: IndexTTS2Wav2Vec2BertEncoder
    @ParameterInfo(key: "masked_spec_embed") var maskedSpecEmbed: MLXArray

    override init() {
        _featureProjection.wrappedValue = IndexTTS2Wav2Vec2BertFeatureProjection()
        _encoder.wrappedValue = IndexTTS2Wav2Vec2BertEncoder()
        _maskedSpecEmbed.wrappedValue = MLXArray.zeros([IndexTTS2Wav2Vec2BertConfig.hiddenSize])
        super.init()
    }

    static func load(from weights: [String: MLXArray]) throws -> IndexTTS2Wav2Vec2Bert {
        let model = IndexTTS2Wav2Vec2Bert()
        try model.loadWeights(weights)
        model.train(false)
        return model
    }

    func loadWeights(_ weights: [String: MLXArray]) throws {
        var mapped: [String: MLXArray] = [:]
        mapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            if Self.isConv1dWeight(key) {
                mapped[key] = value.asType(.float32).transposed(0, 2, 1)
            } else {
                mapped[key] = value.asType(.float32)
            }
        }
        try update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
    }

    func hiddenState17(inputFeatures: MLXArray) -> MLXArray {
        var h = featureProjection(inputFeatures)
        for layerIndex in 0..<17 {
            h = encoder.layers[layerIndex](h)
        }
        eval(h)
        return h
    }

    private static func isConv1dWeight(_ key: String) -> Bool {
        key.hasSuffix(".conv_module.pointwise_conv1.weight") ||
            key.hasSuffix(".conv_module.pointwise_conv2.weight") ||
            key.hasSuffix(".conv_module.depthwise_conv.weight")
    }
}
