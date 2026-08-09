import MLX
import MLXNN

private func speechBrainLeakyReLU(
    _ values: MLXArray,
    negativeSlope: Float = 0.01
) -> MLXArray {
    MLX.maximum(values, MLXArray(negativeSlope) * values)
}

func speechBrainReflectPad(_ values: MLXArray, padding: Int) -> MLXArray {
    guard padding > 0 else { return values }
    precondition(
        values.dim(1) > padding,
        "reflection padding requires more input frames than padding"
    )
    let frameCount = values.dim(1)
    let leftIndexes = MLXArray((1...padding).reversed().map(Int32.init))
    let rightIndexes = MLXArray(
        (0..<padding).map { Int32(frameCount - 2 - $0) }
    )
    return concatenated(
        [
            take(values, leftIndexes, axis: 1),
            values,
            take(values, rightIndexes, axis: 1),
        ],
        axis: 1
    )
}

final class SpeechBrainTDNNBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "norm") var norm: BatchNorm
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        dilation: Int = 1,
        groups: Int = 1
    ) {
        self.padding = (kernelSize - 1) * dilation / 2
        _conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: true
        )
        _norm.wrappedValue = BatchNorm(featureCount: outputChannels)
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        norm(relu(conv(speechBrainReflectPad(values, padding: padding))))
    }
}

final class SpeechBrainRes2NetBlock: Module {
    @ModuleInfo(key: "blocks") var blocks: [SpeechBrainTDNNBlock]
    let scale: Int

    init(channels: Int, kernelSize: Int, dilation: Int, scale: Int = 8) {
        precondition(channels.isMultiple(of: scale))
        self.scale = scale
        let width = channels / scale
        _blocks.wrappedValue = (0..<(scale - 1)).map { _ in
            SpeechBrainTDNNBlock(
                inputChannels: width,
                outputChannels: width,
                kernelSize: kernelSize,
                dilation: dilation
            )
        }
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        let chunks = MLX.split(values, parts: scale, axis: -1)
        var outputs = [chunks[0]]
        for index in blocks.indices {
            let input = index == 0
                ? chunks[index + 1]
                : chunks[index + 1] + outputs[index]
            outputs.append(blocks[index](input))
        }
        return concatenated(outputs, axis: -1)
    }
}

final class SpeechBrainSEBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(channels: Int, seChannels: Int = 128) {
        _conv1.wrappedValue = Conv1d(
            inputChannels: channels,
            outputChannels: seChannels,
            kernelSize: 1,
            bias: true
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: seChannels,
            outputChannels: channels,
            kernelSize: 1,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        let pooled = values.mean(axis: 1, keepDims: true)
        let scale = sigmoid(conv2(relu(conv1(pooled))))
        return values * scale
    }
}

final class SpeechBrainSERes2NetBlock: Module {
    @ModuleInfo(key: "tdnn1") var tdnn1: SpeechBrainTDNNBlock
    @ModuleInfo(key: "res2net_block") var res2netBlock: SpeechBrainRes2NetBlock
    @ModuleInfo(key: "tdnn2") var tdnn2: SpeechBrainTDNNBlock
    @ModuleInfo(key: "se_block") var seBlock: SpeechBrainSEBlock

    init(channels: Int, kernelSize: Int, dilation: Int) {
        _tdnn1.wrappedValue = SpeechBrainTDNNBlock(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1
        )
        _res2netBlock.wrappedValue = SpeechBrainRes2NetBlock(
            channels: channels,
            kernelSize: kernelSize,
            dilation: dilation
        )
        _tdnn2.wrappedValue = SpeechBrainTDNNBlock(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 1
        )
        _seBlock.wrappedValue = SpeechBrainSEBlock(channels: channels)
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        var hidden = tdnn1(values)
        hidden = res2netBlock(hidden)
        hidden = tdnn2(hidden)
        return seBlock(hidden) + values
    }
}

final class SpeechBrainECAPABlocks: Module {
    @ModuleInfo(key: "block0") var block0: SpeechBrainTDNNBlock
    @ModuleInfo(key: "block1") var block1: SpeechBrainSERes2NetBlock
    @ModuleInfo(key: "block2") var block2: SpeechBrainSERes2NetBlock
    @ModuleInfo(key: "block3") var block3: SpeechBrainSERes2NetBlock

    init(melBinCount: Int, channels: Int) {
        _block0.wrappedValue = SpeechBrainTDNNBlock(
            inputChannels: melBinCount,
            outputChannels: channels,
            kernelSize: 5
        )
        _block1.wrappedValue = SpeechBrainSERes2NetBlock(
            channels: channels,
            kernelSize: 3,
            dilation: 2
        )
        _block2.wrappedValue = SpeechBrainSERes2NetBlock(
            channels: channels,
            kernelSize: 3,
            dilation: 3
        )
        _block3.wrappedValue = SpeechBrainSERes2NetBlock(
            channels: channels,
            kernelSize: 3,
            dilation: 4
        )
        super.init()
    }
}

final class SpeechBrainAttentiveStatisticsPooling: Module {
    @ModuleInfo(key: "tdnn") var tdnn: SpeechBrainTDNNBlock
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, attentionChannels: Int = 128) {
        _tdnn.wrappedValue = SpeechBrainTDNNBlock(
            inputChannels: channels * 3,
            outputChannels: attentionChannels,
            kernelSize: 1
        )
        _conv.wrappedValue = Conv1d(
            inputChannels: attentionChannels,
            outputChannels: channels,
            kernelSize: 1,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        let mean = values.mean(axis: 1, keepDims: true)
        let centered = values - mean
        let standardDeviation = sqrt(
            MLX.maximum(
                (centered * centered).mean(axis: 1, keepDims: true),
                MLXArray(Float(1e-12))
            )
        )
        let context = concatenated(
            [
                values,
                MLX.broadcast(mean, to: values.shape),
                MLX.broadcast(standardDeviation, to: values.shape),
            ],
            axis: -1
        )
        let attention = softmax(conv(tanh(tdnn(context))), axis: 1)
        let weightedMean = (attention * values).sum(axis: 1)
        let weightedCentered = values - weightedMean.expandedDimensions(axis: 1)
        let weightedStandardDeviation = sqrt(
            MLX.maximum(
                (attention * weightedCentered * weightedCentered).sum(axis: 1),
                MLXArray(Float(1e-12))
            )
        )
        return concatenated(
            [weightedMean, weightedStandardDeviation],
            axis: -1
        )
    }
}

final class SpeechBrainECAPAEmbedding: Module {
    @ModuleInfo(key: "blocks") var blocks: SpeechBrainECAPABlocks
    @ModuleInfo(key: "mfa") var mfa: SpeechBrainTDNNBlock
    @ModuleInfo(key: "asp") var asp: SpeechBrainAttentiveStatisticsPooling
    @ModuleInfo(key: "asp_bn") var aspBN: BatchNorm
    @ModuleInfo(key: "fc") var fc: Conv1d

    init(melBinCount: Int = 60, embeddingDimension: Int = 256) {
        let channels = 1_024
        _blocks.wrappedValue = SpeechBrainECAPABlocks(
            melBinCount: melBinCount,
            channels: channels
        )
        _mfa.wrappedValue = SpeechBrainTDNNBlock(
            inputChannels: channels * 3,
            outputChannels: channels * 3,
            kernelSize: 1
        )
        _asp.wrappedValue = SpeechBrainAttentiveStatisticsPooling(
            channels: channels * 3
        )
        _aspBN.wrappedValue = BatchNorm(featureCount: channels * 6)
        _fc.wrappedValue = Conv1d(
            inputChannels: channels * 6,
            outputChannels: embeddingDimension,
            kernelSize: 1,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        let initial = blocks.block0(values)
        let output1 = blocks.block1(initial)
        let output2 = blocks.block2(output1)
        let output3 = blocks.block3(output2)
        var hidden = mfa(concatenated([output1, output2, output3], axis: -1))
        hidden = aspBN(asp(hidden))
        return fc(hidden.expandedDimensions(axis: 1))
    }
}

final class SpeechBrainWrappedLinear: Module {
    @ModuleInfo(key: "w") var linear: Linear

    init(inputDimension: Int, outputDimension: Int) {
        _linear.wrappedValue = Linear(inputDimension, outputDimension, bias: true)
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        linear(values)
    }
}

final class SpeechBrainClassifierBlock: Module {
    @ModuleInfo(key: "linear") var linear: SpeechBrainWrappedLinear
    @ModuleInfo(key: "norm") var norm: BatchNorm

    init(inputDimension: Int, outputDimension: Int) {
        _linear.wrappedValue = SpeechBrainWrappedLinear(
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        _norm.wrappedValue = BatchNorm(featureCount: outputDimension)
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        norm(speechBrainLeakyReLU(linear(values)))
    }
}

final class SpeechBrainClassifierDNN: Module {
    @ModuleInfo(key: "block_0") var block0: SpeechBrainClassifierBlock

    init(inputDimension: Int, outputDimension: Int) {
        _block0.wrappedValue = SpeechBrainClassifierBlock(
            inputDimension: inputDimension,
            outputDimension: outputDimension
        )
        super.init()
    }

    func callAsFunction(_ values: MLXArray) -> MLXArray {
        block0(values)
    }
}

final class SpeechBrainLanguageClassifier: Module {
    @ModuleInfo(key: "norm") var norm: BatchNorm
    @ModuleInfo(key: "DNN") var dnn: SpeechBrainClassifierDNN
    @ModuleInfo(key: "out") var out: SpeechBrainWrappedLinear

    init(
        embeddingDimension: Int = 256,
        hiddenDimension: Int = 512,
        classCount: Int = 107
    ) {
        _norm.wrappedValue = BatchNorm(featureCount: embeddingDimension)
        _dnn.wrappedValue = SpeechBrainClassifierDNN(
            inputDimension: embeddingDimension,
            outputDimension: hiddenDimension
        )
        _out.wrappedValue = SpeechBrainWrappedLinear(
            inputDimension: hiddenDimension,
            outputDimension: classCount
        )
        super.init()
    }

    func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        var values = embeddings.squeezed(axis: 1)
        values = norm(speechBrainLeakyReLU(values))
        values = dnn(values)
        let logits = out(values)
        return logits - MLX.logSumExp(logits, axis: -1, keepDims: true)
    }
}

final class SpeechBrainLanguageIDNetwork: Module {
    @ModuleInfo(key: "embedding_model") var embeddingModel: SpeechBrainECAPAEmbedding
    @ModuleInfo(key: "classifier") var classifier: SpeechBrainLanguageClassifier

    override init() {
        _embeddingModel.wrappedValue = SpeechBrainECAPAEmbedding()
        _classifier.wrappedValue = SpeechBrainLanguageClassifier()
        super.init()
    }

    func callAsFunction(_ melFeatures: MLXArray) -> MLXArray {
        let centered = melFeatures
            - melFeatures.mean(axis: 1, keepDims: true)
        return classifier(embeddingModel(centered))
    }
}
