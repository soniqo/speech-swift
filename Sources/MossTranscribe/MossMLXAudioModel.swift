import Foundation
import MLX
import MLXCommon
import MLXNN

final class MossMLXEncoderAttention: Module {
    let heads: Int
    let headDimension: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(_ config: MossMLXConfiguration.Audio) {
        heads = config.attentionHeads
        headDimension = config.hiddenSize / config.attentionHeads
        scale = 1 / sqrt(Float(headDimension))
        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: true
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: true
        )
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
            scale: scale
        )
        return outputProjection(attended)
    }
}

final class MossMLXEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention:
        MossMLXEncoderAttention
    @ModuleInfo(key: "self_attn_layer_norm") var
        selfAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var feedForward1: Linear
    @ModuleInfo(key: "fc2") var feedForward2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: MossMLXConfiguration.Audio) {
        _selfAttention.wrappedValue = MossMLXEncoderAttention(config)
        _selfAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        _feedForward1.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: true
        )
        _feedForward2.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: true
        )
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden =
            input + selfAttention(selfAttentionLayerNorm(input))
        hidden =
            hidden
            + feedForward2(gelu(feedForward1(finalLayerNorm(hidden))))
        return hidden
    }
}

final class MossMLXWhisperEncoder: Module {
    let config: MossMLXConfiguration.Audio

    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo(key: "embed_positions") var positionEmbedding: Embedding
    @ModuleInfo var layers: [MossMLXEncoderLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(_ config: MossMLXConfiguration.Audio) {
        self.config = config
        _conv1.wrappedValue = Conv1d(
            inputChannels: config.melBins,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            padding: 1
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maximumSourcePositions,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0..<config.layers).map { _ in
            MossMLXEncoderLayer(config)
        }
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEpsilon
        )
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var hidden = features.asType(conv1.weight.dtype)
        hidden = gelu(conv1(hidden))
        hidden = gelu(conv2(hidden))
        precondition(
            hidden.dim(1) == config.maximumSourcePositions,
            "MOSS audio chunks must contain exactly "
                + "\(config.maximumSourcePositions) encoder positions"
        )
        hidden =
            hidden
            + positionEmbedding.weight
                .expandedDimensions(axis: 0)
                .asType(hidden.dtype)
        for layer in layers {
            hidden = layer(hidden)
        }
        return layerNorm(hidden)
    }
}

final class MossMLXVQAdaptor: Module {
    /// Preserve the upstream Sequential indices (Linear, SiLU, Linear,
    /// LayerNorm). The activation has no parameters, so its unflattened
    /// weight-tree entry is `none`.
    @ModuleInfo var layers: [Module]

    init(_ config: MossMLXConfiguration) {
        _layers.wrappedValue = [
            Linear(
                config.audio.hiddenSize * config.audio.mergeSize,
                config.decoder.hiddenSize,
                bias: true
            ),
            Identity(),
            Linear(
                config.decoder.hiddenSize,
                config.decoder.hiddenSize,
                bias: true
            ),
            LayerNorm(
                dimensions: config.decoder.hiddenSize,
                eps: config.decoder.rmsNormEpsilon
            ),
        ]
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let inputProjection = layers[0] as! Linear
        let outputProjection = layers[2] as! Linear
        let outputNorm = layers[3] as! LayerNorm
        return outputNorm(
            outputProjection(silu(inputProjection(input)))
        )
    }
}

final class MossMLXAudioModel: Module {
    let config: MossMLXConfiguration

    @ModuleInfo(key: "whisper_encoder") var encoder:
        MossMLXWhisperEncoder
    @ModuleInfo(key: "vq_adaptor") var adaptor: MossMLXVQAdaptor

    init(_ config: MossMLXConfiguration) {
        self.config = config
        _encoder.wrappedValue = MossMLXWhisperEncoder(config.audio)
        _adaptor.wrappedValue = MossMLXVQAdaptor(config)
        super.init()
    }

    /// Encode one batch of padded 30-second feature chunks.
    ///
    /// `features` is `[batch, 3000, 80]`. Each token count identifies how
    /// many of the 375 merged outputs are real for its corresponding chunk.
    func encode(
        features: MLXArray,
        tokenCounts: [Int]
    ) throws -> MLXArray {
        guard
            features.ndim == 3,
            features.dim(0) == tokenCounts.count,
            features.dim(1) == MossWhisperFeatureExtractor.timeFrames,
            features.dim(2) == config.audio.melBins
        else {
            throw MossTranscribeError.invalidAudio(
                "MLX feature batch has an unexpected shape"
            )
        }
        let encoded = encoder(features)
        var mergedChunks: [MLXArray] = []
        mergedChunks.reserveCapacity(tokenCounts.count)
        for (index, tokenCount) in tokenCounts.enumerated() {
            let encoderPositions = tokenCount * config.audio.mergeSize
            guard
                tokenCount > 0,
                encoderPositions <= encoded.dim(1)
            else {
                throw MossTranscribeError.invalidAudio(
                    "invalid MOSS audio token count \(tokenCount)"
                )
            }
            mergedChunks.append(
                encoded[index, 0..<encoderPositions, 0...]
                    .reshaped(
                        tokenCount,
                        config.audio.hiddenSize
                            * config.audio.mergeSize
                    )
            )
        }
        return adaptor(concatenated(mergedChunks, axis: 0))
    }
}
