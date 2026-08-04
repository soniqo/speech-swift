// MLX port of the NemotronLabs VoiceChat streaming FastConformer encoder.
//
// Structure follows CohereTranscribeASR's conformer, which shares NeMo's
// parameter naming. Four things differ and each one fails silently if missed:
//
//   1. causal depthwise convolution — pad (K-1, 0), not the symmetric (K-1)/2
//   2. LayerNorm where the checkpoint says `batch_norm`  (conv_norm_type)
//   3. causal subsampling keeping 17 frequency bins, not 16
//   4. chunked-limited attention, 70 frames left and none right
//
// Plus every linear here is bias-free, matching NeMo.

import Foundation
import MLX
import MLXNN

// MARK: - Causal subsampling

/// NeMo `CausalConv2D`: pads (K-1, S-1) on each spatial axis, then a
/// zero-padding Conv2d. The asymmetric right pad of S-1 is what makes the
/// frequency axis land on 17 rather than 16.
final class CausalConv2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let leftPad: Int
    let rightPad: Int

    init(inChannels: Int, outChannels: Int, kernel: Int = 3, stride: Int = 2, groups: Int = 1) {
        self.leftPad = kernel - 1
        self.rightPad = stride - 1
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernel),
            stride: IntOrPair(stride),
            padding: IntOrPair(0),
            groups: groups
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(
            x,
            widths: [
                IntOrPair(0),
                IntOrPair((leftPad, rightPad)),
                IntOrPair((leftPad, rightPad)),
                IntOrPair(0),
            ]
        )
        return conv(padded)
    }
}

/// Three stride-2 depthwise-separable causal stages: (B, T, 128) -> (B, T/8, dModel).
///
/// The flat `conv` list mirrors NeMo's layout so weight keys line up:
/// 0 CausalConv2D, 1 ReLU, 2 CausalConv2D(dw), 3 Conv2d(pw), 4 ReLU,
/// 5 CausalConv2D(dw), 6 Conv2d(pw), 7 ReLU.
final class CausalSubsampling: Module {
    // The checkpoint stores these as a flat list (`conv.0`, `conv.2`, ...), but
    // a dotted @ModuleInfo key is treated as one literal name rather than a
    // path, so the indices are folded into the key here and the bundle's names
    // are rewritten to match in `VoiceChatPerception.load`.
    @ModuleInfo(key: "conv0") var conv0: CausalConv2D
    @ModuleInfo(key: "conv2") var conv2: CausalConv2D
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "conv5") var conv5: CausalConv2D
    @ModuleInfo(key: "conv6") var conv6: Conv2d
    @ModuleInfo(key: "out") var out: Linear

    private let channels: Int
    private let freqOut: Int

    init(_ config: VoiceChatEncoderConfig) {
        let c = config.subsamplingConvChannels
        self.channels = c
        self.freqOut = config.preEncodeFreqOut

        self._conv0.wrappedValue = CausalConv2D(inChannels: 1, outChannels: c)
        self._conv2.wrappedValue = CausalConv2D(inChannels: c, outChannels: c, groups: c)
        self._conv3.wrappedValue = Conv2d(
            inputChannels: c, outputChannels: c,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0))
        self._conv5.wrappedValue = CausalConv2D(inChannels: c, outChannels: c, groups: c)
        self._conv6.wrappedValue = Conv2d(
            inputChannels: c, outputChannels: c,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0))
        self._out.wrappedValue = Linear(c * config.preEncodeFreqOut, config.dModel)
    }

    /// - Parameter x: log-mel of shape (B, T, nMel)
    /// - Returns: (B, T/8, dModel)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Time stays on axis 1 and frequency on axis 2, matching NeMo's
        // (B, 1, T, F). Transposing here maps the 3x3 kernel to the wrong axis
        // and silently produces a transposed result.
        var h = x.expandedDimensions(axis: -1)      // (B, T, F, 1)
        h = relu(conv0(h))
        h = relu(conv3(conv2(h)))
        h = relu(conv6(conv5(h)))

        // (B, T, F, C) -> (B, T, C, F) -> flatten, matching NeMo's (C, F) order.
        h = h.transposed(0, 1, 3, 2)
        let b = h.shape[0], t = h.shape[1]
        return out(h.reshaped(b, t, channels * freqOut))
    }

    /// Output frame count for a given number of mel frames. Each causal stage
    /// is `floor(n / 2) + 1`, so the result runs one frame longer than a plain
    /// stride-2 stack would give.
    static func outputFrames(melFrames: Int) -> Int {
        var n = melFrames
        for _ in 0 ..< 3 { n = n / 2 + 1 }
        return n
    }
}

// MARK: - Relative positional encoding

/// Holds no trained weights, so it is deliberately not a `Module` — a
/// parameterless node with a nil optional array in the tree crashes
/// parameter traversal during `update`.
final class RelPositionalEncoding {
    let dModel: Int
    let maxLen: Int
    private var pe: MLXArray?

    init(dModel: Int, maxLen: Int) {
        self.dModel = dModel
        self.maxLen = maxLen
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { encode(x) }

    /// Symmetric relative positions from +(T-1) down to -(T-1).
    private func buildTable(length: Int, dtype: DType) -> MLXArray {
        let span = 2 * length - 1
        let positions = MLXArray(stride(from: Float(length - 1), through: Float(-length + 1), by: -1)
            .map { $0 }).reshaped(span, 1)
        let dim = MLXArray(stride(from: Float(0), to: Float(dModel), by: 2).map { $0 })
        let inv = MLX.exp(dim * MLXArray(-(Foundation.log(10000.0) / Double(dModel))).asType(.float32))
        let angles = positions * inv
        var table = MLXArray.zeros([span, dModel])
        table[0..., .stride(by: 2)] = MLX.sin(angles)
        table[0..., .stride(from: 1, by: 2)] = MLX.cos(angles)
        return table.expandedDimensions(axis: 0).asType(dtype)
    }

    private func encode(_ x: MLXArray) -> MLXArray {
        let length = x.shape[1]
        if pe == nil || pe!.shape[1] < 2 * length - 1 {
            pe = buildTable(length: max(length, maxLen / 2), dtype: x.dtype)
        }
        let centre = pe!.shape[1] / 2
        return pe![0..., (centre - length + 1) ..< (centre + length), 0...]
    }
}

// MARK: - Conformer submodules

final class ConformerFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int, useBias: Bool) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: useBias)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

/// Conformer convolution module with causal padding and a LayerNorm standing in
/// for the checkpoint's `batch_norm`.
final class CausalConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    private let leftPad: Int

    init(dModel: Int, kernelSize: Int, useBias: Bool) {
        self.leftPad = kernelSize - 1
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2,
            kernelSize: 1, stride: 1, padding: 0, bias: useBias)
        // padding 0 here: the causal pad is applied explicitly below.
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: kernelSize, stride: 1, padding: 0,
            groups: dModel, bias: useBias)
        self._batchNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel,
            kernelSize: 1, stride: 1, padding: 0, bias: useBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = pointwiseConv1(x)
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * sigmoid(parts[1])
        // Causal: all padding on the left, nothing on the right.
        h = MLX.padded(h, widths: [IntOrPair(0), IntOrPair((leftPad, 0)), IntOrPair(0)])
        h = depthwiseConv(h)
        h = batchNorm(h)
        h = silu(h)
        return pointwiseConv2(h)
    }
}

/// Relative-position multi-head attention with NeMo's separate q/k/v/out/pos
/// projections (the fused `qkv_proj` layout used elsewhere does not match).
final class RelPositionMultiHeadAttention: Module {
    let nHead: Int
    let dK: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(nHead: Int, nFeat: Int, useBias: Bool) {
        self.nHead = nHead
        self.dK = nFeat / nHead
        self.scale = pow(Float(nFeat / nHead), -0.5)
        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: useBias)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: useBias)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: useBias)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: useBias)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead], type: Float.self)
        self._posBiasV.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead], type: Float.self)
    }

    private func relShift(_ x: MLXArray) -> MLXArray {
        let b = x.shape[0], h = x.shape[1], t = x.shape[2], posLen = x.shape[3]
        var shifted = MLX.padded(
            x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair(0), IntOrPair((1, 0))])
        shifted = shifted.reshaped(b, h, posLen + 1, t)
        shifted = shifted[0..., 0..., 1..., 0...]
        return shifted.reshaped(b, h, t, posLen)
    }

    /// - Parameter mask: true marks a position that must not be attended to.
    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.shape[0]
        let q = linearQ(x).reshaped(b, -1, nHead, dK).transposed(0, 2, 1, 3)
        let k = linearK(x).reshaped(b, -1, nHead, dK).transposed(0, 2, 1, 3)
        let v = linearV(x).reshaped(b, -1, nHead, dK).transposed(0, 2, 1, 3)

        let posInput = (posEmb.shape[0] == 1 && b > 1)
            ? MLX.repeated(posEmb, count: b, axis: 0) : posEmb
        let p = linearPos(posInput).reshaped(b, -1, nHead, dK).transposed(0, 2, 1, 3)

        let qU = q + posBiasU.expandedDimensions(axes: [0, 2])
        let qV = q + posBiasV.expandedDimensions(axes: [0, 2])

        let matrixAC = MLX.matmul(qU, k.transposed(0, 1, 3, 2))
        var matrixBD = relShift(MLX.matmul(qV, p.transposed(0, 1, 3, 2)))
        matrixBD = matrixBD[0..., 0..., 0..., ..<matrixAC.shape[3]]

        var scores = (matrixAC + matrixBD) * MLXArray(scale)
        if let mask {
            let m = mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            scores = scores + MLX.where(m, MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(scores.dtype)
        }
        let attn = softmax(scores, axis: -1)
        let output = MLX.matmul(attn, v).transposed(0, 2, 1, 3).reshaped(b, -1, nHead * dK)
        return linearOut(output)
    }
}

final class ConformerLayer: Module {
    @ModuleInfo(key: "norm_feed_forward1") var normFeedForward1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RelPositionMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: CausalConformerConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFeedForward2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(_ config: VoiceChatEncoderConfig) {
        self._normFeedForward1.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._feedForward1.wrappedValue = ConformerFeedForward(
            dModel: config.dModel, dFF: config.dFF, useBias: config.useBias)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._selfAttn.wrappedValue = RelPositionMultiHeadAttention(
            nHead: config.nHeads, nFeat: config.dModel, useBias: config.useBias)
        self._normConv.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._conv.wrappedValue = CausalConformerConvolution(
            dModel: config.dModel, kernelSize: config.convKernelSize, useBias: config.useBias)
        self._normFeedForward2.wrappedValue = LayerNorm(dimensions: config.dModel)
        self._feedForward2.wrappedValue = ConformerFeedForward(
            dModel: config.dModel, dFF: config.dFF, useBias: config.useBias)
        self._normOut.wrappedValue = LayerNorm(dimensions: config.dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x + 0.5 * feedForward1(normFeedForward1(x))
        h = h + selfAttn(normSelfAtt(h), posEmb: posEmb, mask: mask)
        h = h + conv(normConv(h))
        h = h + 0.5 * feedForward2(normFeedForward2(h))
        return normOut(h)
    }
}

// MARK: - Encoder

public final class VoiceChatEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: CausalSubsampling
    // Not @ModuleInfo: carries no weights.
    private let posEnc: RelPositionalEncoding
    @ModuleInfo(key: "layers") var layers: [ConformerLayer]

    private let config: VoiceChatEncoderConfig

    public init(_ config: VoiceChatEncoderConfig) {
        self.config = config
        self._preEncode.wrappedValue = CausalSubsampling(config)
        self.posEnc = RelPositionalEncoding(
            dModel: config.dModel, maxLen: config.posEmbMaxLen)
        self._layers.wrappedValue = (0 ..< config.nLayers).map { _ in ConformerLayer(config) }
    }

    /// Chunked-limited streaming mask. `true` marks a masked position.
    ///
    /// Frames are grouped into chunks of `rightContext + 1`; a query may attend
    /// to its own chunk and the preceding `leftContext / chunkSize` chunks, and
    /// never to a later one. With [70, 0] the chunk size is 1, so this reduces
    /// to a causal band 71 frames wide.
    static func chunkedLimitedMask(length: Int, leftContext: Int, rightContext: Int) -> MLXArray {
        let chunk = rightContext + 1
        let leftChunks = chunk > 0 ? leftContext / chunk : leftContext
        let idx = MLXArray(0 ..< length)
        let chunkIdx = chunk > 0 ? idx / MLXArray(Int32(chunk)) : idx
        let diff = chunkIdx.expandedDimensions(axis: 1) - chunkIdx.expandedDimensions(axis: 0)
        let visible = MLX.logicalAnd(diff .>= MLXArray(Int32(0)),
                                     diff .<= MLXArray(Int32(leftChunks)))
        return MLX.logicalNot(visible)
    }

    /// - Parameter logMel: (B, T, featIn)
    /// - Returns: (B, T/8, dModel)
    public func callAsFunction(_ logMel: MLXArray) -> MLXArray {
        var h = preEncode(logMel)
        let posEmb = posEnc(h)
        let mask = Self.chunkedLimitedMask(
            length: h.shape[1],
            leftContext: config.leftContext,
            rightContext: config.rightContext)
        for layer in layers {
            h = layer(h, posEmb: posEmb, mask: mask)
        }
        return h
    }
}
