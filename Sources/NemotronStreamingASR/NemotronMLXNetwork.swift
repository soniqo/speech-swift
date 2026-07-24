import Foundation
import MLX
import MLXFast
import MLXNN

private func nemotronQuantizedLinear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool = true,
    quantization: NemotronMLXQuantization
) -> Linear {
    QuantizedLinear(
        inputDimensions,
        outputDimensions,
        bias: bias,
        groupSize: quantization.groupSize,
        bits: quantization.bits,
        mode: .affine
    )
}

final class NemotronMLXCausalConv2D: Module {
    @ModuleInfo var conv: Conv2d

    init(
        inputChannels: Int,
        outputChannels: Int,
        groups: Int = 1
    ) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: 3,
            stride: 2,
            padding: 0,
            groups: groups
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let paddedInput = MLX.padded(
            input,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((2, 1)),
                IntOrPair((2, 1)),
                IntOrPair((0, 0)),
            ]
        )
        return conv(paddedInput)
    }
}

final class NemotronMLXSubsampling: Module {
    // This must remain an array, rather than a wrapper with numeric property
    // keys. Python MLX serializes Sequential/list children as array nodes, and
    // MLX-Swift's quantizer/strict loader preserves that tree distinction.
    @ModuleInfo var conv: [Module]
    @ModuleInfo var out: Linear

    init(
        _ configuration: NemotronMLXConfiguration.Encoder,
        quantization: NemotronMLXQuantization
    ) {
        let channels = configuration.subsamplingChannels
        _conv.wrappedValue = [
            NemotronMLXCausalConv2D(
                inputChannels: 1,
                outputChannels: channels
            ),
            ReLU(),
            NemotronMLXCausalConv2D(
                inputChannels: channels,
                outputChannels: channels,
                groups: channels
            ),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 1
            ),
            ReLU(),
            NemotronMLXCausalConv2D(
                inputChannels: channels,
                outputChannels: channels,
                groups: channels
            ),
            Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: 1
            ),
            ReLU(),
        ]
        _out.wrappedValue = nemotronQuantizedLinear(
            configuration.subsamplingChannels * 17,
            configuration.hiddenSize,
            quantization: quantization
        )
        super.init()
    }

    /// `[batch, mel frames, mel bins]` to `[batch, encoder frames, hidden]`.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var hidden = (conv[0] as! NemotronMLXCausalConv2D)(
            input.expandedDimensions(axis: -1)
        )
        hidden = (conv[1] as! ReLU)(hidden)
        hidden = (conv[2] as! NemotronMLXCausalConv2D)(hidden)
        hidden = (conv[3] as! Conv2d)(hidden)
        hidden = (conv[4] as! ReLU)(hidden)
        hidden = (conv[5] as! NemotronMLXCausalConv2D)(hidden)
        hidden = (conv[6] as! Conv2d)(hidden)
        hidden = (conv[7] as! ReLU)(hidden)
        let batch = hidden.dim(0)
        let time = hidden.dim(1)
        let frequency = hidden.dim(2)
        let channels = hidden.dim(3)
        hidden = hidden
            .transposed(0, 1, 3, 2)
            .reshaped(batch, time, channels * frequency)
        return out(hidden)
    }
}

final class NemotronMLXFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(
        hiddenSize: Int,
        expansion: Int,
        quantization: NemotronMLXQuantization
    ) {
        _linear1.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize * expansion,
            bias: false,
            quantization: quantization
        )
        _linear2.wrappedValue = nemotronQuantizedLinear(
            hiddenSize * expansion,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(silu(linear1(input)))
    }
}

final class NemotronMLXLayerCache {
    var key: MLXArray?
    var value: MLXArray?
    var convolution: MLXArray?

    func append(
        key newKey: MLXArray,
        value newValue: MLXArray,
        leftContext: Int
    ) -> (MLXArray, MLXArray) {
        var combinedKey = key.map {
            MLX.concatenated([$0, newKey], axis: 2)
        } ?? newKey
        var combinedValue = value.map {
            MLX.concatenated([$0, newValue], axis: 2)
        } ?? newValue

        let retained = leftContext + newKey.dim(2)
        if combinedKey.dim(2) > retained {
            let start = combinedKey.dim(2) - retained
            combinedKey = combinedKey[0..., 0..., start..., 0...]
            combinedValue = combinedValue[0..., 0..., start..., 0...]
        }
        key = combinedKey
        value = combinedValue
        return (combinedKey, combinedValue)
    }

    func prependConvolution(
        _ input: MLXArray,
        cacheSize: Int
    ) -> MLXArray {
        if convolution == nil {
            convolution = MLXArray.zeros(
                [input.dim(0), cacheSize, input.dim(2)],
                dtype: input.dtype
            )
        }
        let combined = MLX.concatenated([convolution!, input], axis: 1)
        convolution = combined[
            0...,
            (combined.dim(1) - cacheSize)...,
            0...
        ]
        return combined
    }

    var evaluatedArrays: [MLXArray] {
        [key, value, convolution].compactMap { $0 }
    }
}

final class NemotronMLXRelativeAttention: Module {
    let heads: Int
    let headSize: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var query: Linear
    @ModuleInfo(key: "linear_k") var key: Linear
    @ModuleInfo(key: "linear_v") var value: Linear
    @ModuleInfo(key: "linear_out") var output: Linear
    @ModuleInfo(key: "linear_pos") var position: Linear
    @ParameterInfo(key: "pos_bias_u") var positionBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var positionBiasV: MLXArray

    init(
        hiddenSize: Int,
        heads: Int,
        quantization: NemotronMLXQuantization
    ) {
        self.heads = heads
        headSize = hiddenSize / heads
        scale = 1 / sqrt(Float(headSize))
        _query.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        _key.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        _value.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        _output.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        _position.wrappedValue = nemotronQuantizedLinear(
            hiddenSize,
            hiddenSize,
            bias: false,
            quantization: quantization
        )
        _positionBiasU.wrappedValue = MLXArray.zeros([heads, headSize])
        _positionBiasV.wrappedValue = MLXArray.zeros([heads, headSize])
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbedding: MLXArray,
        cache: NemotronMLXLayerCache,
        leftContext: Int
    ) -> MLXArray {
        let batch = input.dim(0)
        let queryLength = input.dim(1)

        let projectedQuery = query(input)
            .reshaped(batch, queryLength, heads, headSize)
        let queryU = (projectedQuery + positionBiasU)
            .transposed(0, 2, 1, 3)
        let queryV = (projectedQuery + positionBiasV)
            .transposed(0, 2, 1, 3)
        let newKey = key(input)
            .reshaped(batch, queryLength, heads, headSize)
            .transposed(0, 2, 1, 3)
        let newValue = value(input)
            .reshaped(batch, queryLength, heads, headSize)
            .transposed(0, 2, 1, 3)
        let (allKey, allValue) = cache.append(
            key: newKey,
            value: newValue,
            leftContext: leftContext
        )

        let positionLength = positionEmbedding.dim(1)
        let projectedPosition = position(positionEmbedding)
            .reshaped(1, positionLength, heads, headSize)
            .transposed(0, 2, 1, 3)
        var relativeScores = MLX.matmul(
            queryV,
            projectedPosition.swappedAxes(-2, -1)
        )
        relativeScores = relativeShift(relativeScores)
        relativeScores = relativeScores[
            0...,
            0...,
            0...,
            0..<allKey.dim(2)
        ] * scale

        let attended = MLXFast.scaledDotProductAttention(
            queries: queryU,
            keys: allKey,
            values: allValue,
            scale: scale,
            mask: relativeScores
        )
        return output(
            attended
                .transposed(0, 2, 1, 3)
                .reshaped(batch, queryLength, heads * headSize)
        )
    }

    private func relativeShift(_ input: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let headCount = input.dim(1)
        let queryLength = input.dim(2)
        let positionLength = input.dim(3)
        var shifted = MLX.padded(
            input,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((1, 0)),
            ]
        )
        shifted = shifted.reshaped(
            batch,
            headCount,
            positionLength + 1,
            queryLength
        )
        shifted = shifted[0..., 0..., 1..., 0...]
        return shifted.reshaped(
            batch,
            headCount,
            queryLength,
            positionLength
        )
    }
}

final class NemotronMLXConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwise1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwise: Conv1d
    @ModuleInfo(key: "batch_norm") var normalization: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwise2: Conv1d

    init(hiddenSize: Int, kernelSize: Int) {
        _pointwise1.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize * 2,
            kernelSize: 1,
            bias: false
        )
        _depthwise.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: kernelSize,
            groups: hiddenSize,
            bias: false
        )
        _normalization.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _pointwise2.wrappedValue = Conv1d(
            inputChannels: hiddenSize,
            outputChannels: hiddenSize,
            kernelSize: 1,
            bias: false
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: NemotronMLXLayerCache,
        cacheSize: Int
    ) -> MLXArray {
        var hidden = MLXNN.glu(pointwise1(input), axis: 2)
        hidden = cache.prependConvolution(hidden, cacheSize: cacheSize)
        hidden = depthwise(hidden)
        hidden = silu(normalization(hidden))
        return pointwise2(hidden)
    }
}

final class NemotronMLXConformerLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var feedForwardNorm1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: NemotronMLXFeedForward
    @ModuleInfo(key: "norm_self_att") var attentionNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: NemotronMLXRelativeAttention
    @ModuleInfo(key: "norm_conv") var convolutionNorm: LayerNorm
    @ModuleInfo var conv: NemotronMLXConvolution
    @ModuleInfo(key: "norm_feed_forward2") var feedForwardNorm2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: NemotronMLXFeedForward
    @ModuleInfo(key: "norm_out") var outputNorm: LayerNorm

    init(
        _ configuration: NemotronMLXConfiguration.Encoder,
        quantization: NemotronMLXQuantization
    ) {
        let hiddenSize = configuration.hiddenSize
        _feedForwardNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _feedForward1.wrappedValue = NemotronMLXFeedForward(
            hiddenSize: hiddenSize,
            expansion: configuration.feedForwardExpansion,
            quantization: quantization
        )
        _attentionNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _attention.wrappedValue = NemotronMLXRelativeAttention(
            hiddenSize: hiddenSize,
            heads: configuration.attentionHeads,
            quantization: quantization
        )
        _convolutionNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _conv.wrappedValue = NemotronMLXConvolution(
            hiddenSize: hiddenSize,
            kernelSize: configuration.convolutionKernelSize
        )
        _feedForwardNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize)
        _feedForward2.wrappedValue = NemotronMLXFeedForward(
            hiddenSize: hiddenSize,
            expansion: configuration.feedForwardExpansion,
            quantization: quantization
        )
        _outputNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        positionEmbedding: MLXArray,
        cache: NemotronMLXLayerCache,
        leftContext: Int,
        convolutionCacheSize: Int
    ) -> MLXArray {
        var hidden = input
            + 0.5 * feedForward1(feedForwardNorm1(input))
        hidden = hidden + attention(
            attentionNorm(hidden),
            positionEmbedding: positionEmbedding,
            cache: cache,
            leftContext: leftContext
        )
        hidden = hidden + conv(
            convolutionNorm(hidden),
            cache: cache,
            cacheSize: convolutionCacheSize
        )
        hidden = hidden
            + 0.5 * feedForward2(feedForwardNorm2(hidden))
        return outputNorm(hidden)
    }
}

final class NemotronMLXEncoder: Module {
    let configuration: NemotronMLXConfiguration

    @ModuleInfo(key: "pre_encode") var preEncode: NemotronMLXSubsampling
    @ModuleInfo var layers: [NemotronMLXConformerLayer]

    init(_ configuration: NemotronMLXConfiguration) {
        self.configuration = configuration
        _preEncode.wrappedValue = NemotronMLXSubsampling(
            configuration.encoder,
            quantization: configuration.quantization
        )
        _layers.wrappedValue = (0..<configuration.encoder.layers).map { _ in
            NemotronMLXConformerLayer(
                configuration.encoder,
                quantization: configuration.quantization
            )
        }
        super.init()
    }

    func stream(
        mel: MLXArray,
        caches: [NemotronMLXLayerCache]
    ) -> MLXArray {
        precondition(caches.count == layers.count)
        let outputFrames = configuration.streaming.outputFrames
        var hidden = preEncode(mel)
        hidden = hidden[
            0...,
            (hidden.dim(1) - outputFrames)...,
            0...
        ]
        hidden = hidden * sqrt(Float(configuration.encoder.hiddenSize))

        let seen = caches.first?.key?.dim(2) ?? 0
        let keyFrames =
            min(configuration.streaming.attentionLeftContext, seen)
            + hidden.dim(1)
        let positionEmbedding = Self.relativePositionEmbedding(
            keyFrames: keyFrames,
            hiddenSize: configuration.encoder.hiddenSize,
            dtype: hidden.dtype
        )
        for index in layers.indices {
            hidden = layers[index](
                hidden,
                positionEmbedding: positionEmbedding,
                cache: caches[index],
                leftContext:
                    configuration.streaming.attentionLeftContext,
                convolutionCacheSize:
                    configuration.streaming.convolutionCacheSize
            )
        }
        return hidden
    }

    static func relativePositionEmbedding(
        keyFrames: Int,
        hiddenSize: Int,
        dtype: DType
    ) -> MLXArray {
        let positions = MLXArray(
            stride(
                from: keyFrames - 1,
                through: -(keyFrames - 1),
                by: -1
            )
        )
        .asType(.float32)
        .expandedDimensions(axis: 1)
        let evenDimensions = MLXArray(
            stride(from: 0, to: hiddenSize, by: 2)
        ).asType(.float32)
        let divisor = MLX.exp(
            evenDimensions
                * Float(-Foundation.log(10_000) / Double(hiddenSize))
        )
        let angles = positions * divisor
        return MLX.stacked(
            [MLX.sin(angles), MLX.cos(angles)],
            axis: -1
        )
        .reshaped(1, positions.dim(0), hiddenSize)
        .asType(dtype)
    }
}

final class NemotronMLXPredictionNetworkBody: Module {
    @ModuleInfo var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var recurrent: NemotronMLXStackedLSTM

    init(_ configuration: NemotronMLXConfiguration.Decoder) {
        _embed.wrappedValue = Embedding(
            embeddingCount:
                configuration.vocabularySize
                + (configuration.blankAsPadding ? 1 : 0),
            dimensions: configuration.predictionNetwork.hiddenSize
        )
        _recurrent.wrappedValue = NemotronMLXStackedLSTM(
            hiddenSize: configuration.predictionNetwork.hiddenSize,
            layers: configuration.predictionNetwork.layers
        )
        super.init()
    }
}

final class NemotronMLXStackedLSTM: Module {
    @ModuleInfo var lstm: [LSTM]

    init(hiddenSize: Int, layers: Int) {
        _lstm.wrappedValue = (0..<layers).map { _ in
            LSTM(inputSize: hiddenSize, hiddenSize: hiddenSize)
        }
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        hidden: MLXArray?,
        cell: MLXArray?
    ) -> (output: MLXArray, hidden: MLXArray, cell: MLXArray) {
        var output = input
        var nextHidden: [MLXArray] = []
        var nextCell: [MLXArray] = []
        nextHidden.reserveCapacity(lstm.count)
        nextCell.reserveCapacity(lstm.count)
        for index in lstm.indices {
            let layerHidden = hidden.map { $0[index, 0..., 0...] }
            let layerCell = cell.map { $0[index, 0..., 0...] }
            let result = lstm[index](
                output,
                hidden: layerHidden,
                cell: layerCell
            )
            output = result.0
            nextHidden.append(result.0[0..., -1, 0...])
            nextCell.append(result.1[0..., -1, 0...])
        }
        return (
            output,
            MLX.stacked(nextHidden, axis: 0),
            MLX.stacked(nextCell, axis: 0)
        )
    }
}

final class NemotronMLXPredictionNetwork: Module {
    let hiddenSize: Int
    @ModuleInfo var prediction: NemotronMLXPredictionNetworkBody

    init(_ configuration: NemotronMLXConfiguration.Decoder) {
        hiddenSize = configuration.predictionNetwork.hiddenSize
        _prediction.wrappedValue = NemotronMLXPredictionNetworkBody(
            configuration
        )
        super.init()
    }

    func step(
        token: Int?,
        hidden: MLXArray?,
        cell: MLXArray?
    ) -> (output: MLXArray, hidden: MLXArray, cell: MLXArray) {
        let input: MLXArray
        if let token {
            input = prediction.embed(
                MLXArray([Int32(token)]).reshaped(1, 1)
            )
        } else {
            input = MLXArray.zeros([1, 1, hiddenSize])
        }
        return prediction.recurrent(
            input,
            hidden: hidden,
            cell: cell
        )
    }
}

final class NemotronMLXJointNetwork: Module {
    @ModuleInfo var pred: Linear
    @ModuleInfo var enc: Linear
    @ModuleInfo(key: "joint_net") var output: [Module]

    init(
        _ configuration: NemotronMLXConfiguration.Joint,
        quantization: NemotronMLXQuantization
    ) {
        _pred.wrappedValue = nemotronQuantizedLinear(
            configuration.network.predictionHiddenSize,
            configuration.network.hiddenSize,
            quantization: quantization
        )
        _enc.wrappedValue = nemotronQuantizedLinear(
            configuration.network.encoderHiddenSize,
            configuration.network.hiddenSize,
            quantization: quantization
        )
        _output.wrappedValue = [
            ReLU(),
            Identity(),
            nemotronQuantizedLinear(
                configuration.network.hiddenSize,
                configuration.classes + 1,
                quantization: quantization
            ),
        ]
        super.init()
    }

    func callAsFunction(
        encoder: MLXArray,
        prediction: MLXArray
    ) -> MLXArray {
        let joined =
            enc(encoder).expandedDimensions(axis: 2)
            + pred(prediction).expandedDimensions(axis: 1)
        var hidden = (output[0] as! ReLU)(joined)
        hidden = (output[1] as! Identity)(hidden)
        return (output[2] as! Linear)(hidden)
    }
}

final class NemotronMLXPromptKernel: Module {
    let promptCount: Int
    @ModuleInfo var layers: [Module]

    init(_ configuration: NemotronMLXConfiguration.PromptKernel) {
        promptCount = configuration.promptCount
        _layers.wrappedValue = [
            Linear(
                configuration.modelSize + configuration.promptCount,
                configuration.hiddenSize
            ),
            ReLU(),
            Linear(
                configuration.hiddenSize,
                configuration.modelSize
            ),
        ]
        super.init()
    }

    func callAsFunction(
        _ encoded: MLXArray,
        languageMask: MLXArray
    ) -> MLXArray {
        let mask = MLX.broadcast(
            languageMask.expandedDimensions(axis: 1),
            to: [encoded.dim(0), encoded.dim(1), promptCount]
        )
        var hidden = (layers[0] as! Linear)(
            MLX.concatenated([encoded, mask], axis: -1)
        )
        hidden = (layers[1] as! ReLU)(hidden)
        return (layers[2] as! Linear)(hidden)
    }
}

final class NemotronMLXNetwork: Module {
    @ModuleInfo var encoder: NemotronMLXEncoder
    @ModuleInfo var decoder: NemotronMLXPredictionNetwork
    @ModuleInfo var joint: NemotronMLXJointNetwork
    @ModuleInfo(key: "prompt_kernel") var promptKernel: NemotronMLXPromptKernel

    init(_ configuration: NemotronMLXConfiguration) {
        _encoder.wrappedValue = NemotronMLXEncoder(configuration)
        _decoder.wrappedValue = NemotronMLXPredictionNetwork(
            configuration.decoder
        )
        _joint.wrappedValue = NemotronMLXJointNetwork(
            configuration.joint,
            quantization: configuration.quantization
        )
        _promptKernel.wrappedValue = NemotronMLXPromptKernel(
            configuration.promptKernel
        )
        super.init()
    }
}
