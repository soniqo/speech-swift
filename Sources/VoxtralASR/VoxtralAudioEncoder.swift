import Foundation
import MLX
import MLXCommon
import MLXNN

final class VoxtralEncoderAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(_ config: VoxtralAudioConfig) {
        heads = config.numAttentionHeads
        headDimension = config.hiddenSize / config.numAttentionHeads
        scale = 1 / sqrt(Float(headDimension))
        _queryProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _outputProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let query = queryProjection(input)
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let key = keyProjection(input)
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let value = valueProjection(input)
            .reshaped(batch, length, heads, headDimension)
            .transposed(0, 2, 1, 3)
        let attended = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: key,
            vHeads: value,
            scale: scale)
        return outputProjection(attended)
    }
}

final class VoxtralEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: VoxtralEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var feedForward1: Linear
    @ModuleInfo(key: "fc2") var feedForward2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: VoxtralAudioConfig) {
        _selfAttention.wrappedValue = VoxtralEncoderAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        _feedForward1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
        _feedForward2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = input + selfAttention(selfAttentionLayerNorm(input))
        hidden = hidden + feedForward2(gelu(feedForward1(finalLayerNorm(hidden))))
        return hidden
    }
}

final class VoxtralAudioEncoder: Module {
    let config: VoxtralAudioConfig

    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positionEmbedding: Embedding
    @ModuleInfo var layers: [VoxtralEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: VoxtralAudioConfig) {
        self.config = config
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.numMelBins,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            padding: 1)
        _conv2.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxSourcePositions,
            dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            VoxtralEncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var hidden = features.asType(conv1.weight.dtype)
        hidden = gelu(conv1(hidden))
        hidden = gelu(conv2(hidden))
        precondition(
            hidden.dim(1) == config.maxSourcePositions,
            "Voxtral audio chunks must contain exactly \(config.maxSourcePositions) encoder positions")
        hidden = hidden + positionEmbedding.weight.expandedDimensions(axis: 0).asType(hidden.dtype)
        for layer in layers { hidden = layer(hidden) }
        return layerNorm(hidden)
    }
}

final class VoxtralMultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var inputProjection: Linear
    @ModuleInfo(key: "linear_2") var outputProjection: Linear

    init(_ config: VoxtralConfig) {
        _inputProjection.wrappedValue = Linear(
            config.audioConfig.intermediateSize,
            config.textConfig.hiddenSize,
            bias: false)
        _outputProjection.wrappedValue = Linear(
            config.textConfig.hiddenSize,
            config.textConfig.hiddenSize,
            bias: false)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        outputProjection(gelu(inputProjection(features)))
    }
}
