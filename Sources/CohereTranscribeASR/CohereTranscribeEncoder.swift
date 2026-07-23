// Portions adapted from mlx-audio-swift (MIT License).
// Copyright (c) 2025 Blaizzy and contributors.

import Foundation
import MLX
import MLXNN
import MLXFast

final class ConvSubsampling: Module {
    let featIn: Int
    let featOut: Int
    let subsamplingFactor: Int

    @ModuleInfo(key: "conv0") var conv0: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "conv5") var conv5: Conv2d
    @ModuleInfo(key: "conv6") var conv6: Conv2d
    @ModuleInfo(key: "out") var out: Linear

    init(_ config: CohereTranscribeAudioEncoderConfig) {
        self.featIn = config.featIn
        self.featOut = config.dModel
        self.subsamplingFactor = config.subsamplingFactor

        let convChannels = config.subsamplingConvChannels
        self._conv0.wrappedValue = Conv2d(
            inputChannels: 1,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        self._conv2.wrappedValue = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            groups: convChannels
        )
        self._conv3.wrappedValue = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._conv5.wrappedValue = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            groups: convChannels
        )
        self._conv6.wrappedValue = Conv2d(
            inputChannels: convChannels,
            outputChannels: convChannels,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._out.wrappedValue = Linear(convChannels * (config.featIn / config.subsamplingFactor), config.dModel)
    }

    private func applyConvLength(_ lengths: MLXArray, kernel: Int = 3, stride: Int = 2, padding: Int = 1) -> MLXArray {
        ((lengths + 2 * padding - kernel) / stride) + 1
    }

    private func createMaskLike(_ x: MLXArray, lengths: MLXArray) -> MLXArray {
        let time = x.shape[1]
        let idx = MLXArray(0..<time).asType(.int32).expandedDimensions(axis: 0)
        var timeMask = idx .< lengths.expandedDimensions(axis: 1)
        while timeMask.ndim < x.ndim {
            timeMask = timeMask.expandedDimensions(axis: timeMask.ndim)
        }
        return timeMask
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        var h = x.transposed(0, 2, 1).expandedDimensions(axis: -1)
        var outLengths = lengths

        var mask = createMaskLike(h, lengths: outLengths)
        h = h * mask.asType(h.dtype)
        h = relu(conv0(h))

        outLengths = applyConvLength(outLengths)
        mask = createMaskLike(h, lengths: outLengths)
        h = h * mask.asType(h.dtype)
        h = conv2(h)
        h = conv3(h)
        h = relu(h)

        outLengths = applyConvLength(outLengths)
        mask = createMaskLike(h, lengths: outLengths)
        h = h * mask.asType(h.dtype)
        h = conv5(h)
        h = conv6(h)
        h = relu(h)

        outLengths = applyConvLength(outLengths).asType(.int32)
        mask = createMaskLike(h, lengths: outLengths)
        h = h * mask.asType(h.dtype)

        let batch = h.shape[0]
        let time = h.shape[1]
        let freq = h.shape[2]
        let channels = h.shape[3]
        h = h.transposed(0, 1, 3, 2).reshaped(batch, time, channels * freq)

        return (out(h), outLengths)
    }
}
final class RelPositionalEncoding: Module {
    let dModel: Int
    let maxLen: Int
    var pe: MLXArray?

    init(dModel: Int, maxLen: Int = 5000) {
        self.dModel = dModel
        self.maxLen = maxLen
    }

    private func createPE(positions: MLXArray, dtype: DType) -> MLXArray {
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32)
                * MLXArray(Float(-log(10000.0) / Float(dModel)))
        )

        let posLength = positions.shape[0]
        var pe = MLXArray.zeros([posLength, dModel])
        let sinVals = MLX.sin(positions * divTerm)
        let cosVals = MLX.cos(positions * divTerm)
        pe = pe.at[0..., .stride(by: 2)].add(sinVals)
        pe = pe.at[0..., .stride(from: 1, by: 2)].add(cosVals)
        return pe.expandedDimensions(axis: 0).asType(dtype)
    }

    private func materialize(length: Int, dtype: DType) {
        let neededSize = 2 * length - 1
        if let pe, pe.shape[1] >= neededSize, pe.dtype == dtype {
            return
        }

        let effectiveLength = max(length, maxLen)
        let positions = MLXArray(
            stride(from: Float(effectiveLength - 1), through: Float(-(effectiveLength - 1)), by: -1)
        ).expandedDimensions(axis: 1)
        pe = createPE(positions: positions, dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let inputLen = x.shape[1]
        materialize(length: inputLen, dtype: x.dtype)

        guard let pe else {
            fatalError("RelPositionalEncoding internal PE buffer was not initialized")
        }

        let centerPos = pe.shape[1] / 2 + 1
        let startPos = centerPos - inputLen
        let endPos = centerPos + inputLen - 1
        let posEmb = pe[0..., startPos..<endPos, 0...]
        return (x, posEmb)
    }
}

final class ConformerFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "dropout") var dropout: Dropout
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int, dropout: Float) {
        self._linear1.wrappedValue = Linear(dModel, dFF)
        self._dropout.wrappedValue = Dropout(p: dropout)
        self._linear2.wrappedValue = Linear(dFF, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(dropout(silu(linear1(x))))
    }
}

final class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(dModel: Int, kernelSize: Int) {
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel * 2,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: kernelSize,
            stride: 1,
            padding: (kernelSize - 1) / 2,
            groups: dModel
        )
        self._batchNorm.wrappedValue = BatchNorm(featureCount: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel,
            outputChannels: dModel,
            kernelSize: 1,
            stride: 1,
            padding: 0
        )
    }

    func callAsFunction(_ x: MLXArray, padMask: MLXArray? = nil) -> MLXArray {
        var h = pointwiseConv1(x)
        let split = MLX.split(h, parts: 2, axis: -1)
        h = split[0] * sigmoid(split[1])

        if let padMask {
            let valid = MLXArray(1.0).asType(h.dtype) - padMask.asType(h.dtype).expandedDimensions(axis: -1)
            h = h * valid
        }

        h = depthwiseConv(h)
        h = batchNorm(h)
        h = silu(h)
        return pointwiseConv2(h)
    }
}

final class RelPositionMultiHeadAttention: Module {
    let nHead: Int
    let dK: Int
    let nFeat: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "pos_proj") var posProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "dropout") var dropout: Dropout

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(nHead: Int, nFeat: Int, dropoutRate: Float) {
        self.nHead = nHead
        self.nFeat = nFeat
        self.dK = nFeat / nHead
        self.scale = pow(Float(dK), -0.5)

        self._qkvProj.wrappedValue = Linear(nFeat, 3 * nFeat)
        self._posProj.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._outProj.wrappedValue = Linear(nFeat, nFeat)
        self._dropout.wrappedValue = Dropout(p: dropoutRate)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, dK], type: Float.self)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let h = x.shape[1]
        let t = x.shape[2]
        let posLen = x.shape[3]

        var shifted = MLX.padded(
            x,
            widths: [
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair(0),
                IntOrPair((1, 0)),
            ]
        )
        shifted = shifted.reshaped(b, h, posLen + 1, t)
        shifted = shifted[0..., 0..., 1..., 0...]
        return shifted.reshaped(b, h, t, posLen)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let batchSize = x.shape[0]
        let qkv = qkvProj(x)
        let parts = qkv.split(parts: 3, axis: -1)

        let q = parts[0].reshaped(batchSize, -1, nHead, dK).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped(batchSize, -1, nHead, dK).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(batchSize, -1, nHead, dK).transposed(0, 2, 1, 3)

        let posInput = (posEmb.shape[0] == 1 && batchSize > 1)
            ? MLX.repeated(posEmb, count: batchSize, axis: 0)
            : posEmb
        let p = posProj(posInput).reshaped(batchSize, -1, nHead, dK).transposed(0, 2, 1, 3)

        let qWithU = q + posBiasU.expandedDimensions(axes: [0, 2])
        let qWithV = q + posBiasV.expandedDimensions(axes: [0, 2])

        let matrixAC = MLX.matmul(qWithU, k.transposed(0, 1, 3, 2))
        var matrixBD = MLX.matmul(qWithV, p.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)
        matrixBD = matrixBD[0..., 0..., 0..., ..<matrixAC.shape[3]]

        var scores = (matrixAC + matrixBD) * MLXArray(scale)
        if let mask {
            let expandedMask = mask.expandedDimensions(axis: 1)
            let additive = MLX.where(expandedMask, MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(scores.dtype)
            scores = scores + additive
        }

        var attn = softmax(scores, axis: -1)
        if let mask {
            let expandedMask = mask.expandedDimensions(axis: 1)
            attn = MLX.where(expandedMask, MLXArray(Float(0)).asType(attn.dtype), attn)
        }

        let output = MLX.matmul(dropout(attn), v)
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, -1, nHead * dK)
        return outProj(output)
    }
}

final class ConformerLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm
    @ModuleInfo(key: "dropout") var dropout: Dropout

    init(dModel: Int, dFF: Int, nHeads: Int, convKernelSize: Int, dropoutRate: Float) {
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF, dropout: dropoutRate)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(nHead: nHeads, nFeat: dModel, dropoutRate: dropoutRate)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(dModel: dModel, kernelSize: convKernelSize)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF, dropout: dropoutRate)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
        self._dropout.wrappedValue = Dropout(p: dropoutRate)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray? = nil, padMask: MLXArray? = nil) -> MLXArray {
        var residual = x
        var h = normFeedForward1(x)
        h = residual + MLXArray(0.5).asType(x.dtype) * dropout(feedForward1(h))

        residual = h
        h = normSelfAtt(h)
        h = residual + dropout(selfAttn(h, posEmb: posEmb, mask: mask))

        residual = h
        h = normConv(h)
        h = residual + dropout(conv(h, padMask: padMask))

        residual = h
        h = normFeedForward2(h)
        h = residual + MLXArray(0.5).asType(x.dtype) * dropout(feedForward2(h))

        return normOut(h)
    }
}

final class ConformerEncoder: Module {
    let dModel: Int
    let nLayers: Int

    @ModuleInfo(key: "subsampling") var subsampling: ConvSubsampling
    let positionalEncoding: RelPositionalEncoding
    @ModuleInfo(key: "layers") var layers: [ConformerLayer]

    init(_ config: CohereTranscribeAudioEncoderConfig) {
        self.dModel = config.dModel
        self.nLayers = config.nLayers

        let dFF = config.dModel * config.ffExpansionFactor
        self._subsampling.wrappedValue = ConvSubsampling(config)
        self.positionalEncoding = RelPositionalEncoding(dModel: config.dModel, maxLen: config.posEmbMaxLen)
        self._layers.wrappedValue = (0..<config.nLayers).map { _ in
            ConformerLayer(
                dModel: config.dModel,
                dFF: dFF,
                nHeads: config.nHeads,
                convKernelSize: config.convKernelSize,
                dropoutRate: 0
            )
        }
    }

    private func createMasks(lengths: MLXArray, maxAudioLength: Int) -> (padMask: MLXArray, attMask: MLXArray) {
        let idx = MLXArray(0..<maxAudioLength).asType(.int32).expandedDimensions(axis: 0)
        let valid = idx .< lengths.expandedDimensions(axis: 1)

        let validQ = valid.expandedDimensions(axis: 1)
        let validK = valid.expandedDimensions(axis: 2)
        let validAtt = logicalAnd(validQ, validK)

        let attMask = logicalNot(validAtt)
        let padMask = logicalNot(valid)
        return (padMask, attMask)
    }

    func callAsFunction(_ inputFeatures: MLXArray, lengths: MLXArray? = nil) -> (MLXArray, MLXArray) {
        let inLengths = lengths
            ?? MLXArray(Array(repeating: Int32(inputFeatures.shape[2]), count: inputFeatures.shape[0]))

        let (subsampled, outLengths) = subsampling(inputFeatures, lengths: inLengths.asType(.int32))
        let maxAudioLength = subsampled.shape[1]

        var encoded = subsampled
        let (x, posEmb) = positionalEncoding(encoded)
        encoded = x

        let masks = createMasks(lengths: outLengths.asType(.int32), maxAudioLength: maxAudioLength)
        for layer in layers {
            encoded = layer(encoded, posEmb: posEmb, mask: masks.attMask, padMask: masks.padMask)
        }

        return (encoded, outLengths.asType(.int32))
    }
}
