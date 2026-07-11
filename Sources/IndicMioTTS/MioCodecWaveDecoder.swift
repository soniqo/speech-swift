import Foundation
import MLX
import MLXCommon
import MLXNN

final class MioConv1dNCL: Module {
    @ModuleInfo var conv: Conv1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups,
            bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

final class MioConvTranspose1dNCL: Module {
    @ModuleInfo var conv: ConvTransposed1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int = 0,
        outputPadding: Int = 0,
        bias: Bool = true
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            outputPadding: outputPadding,
            bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

final class MioAdaLNZero: Module {
    let norm: LayerNorm
    @ModuleInfo(key: "condition_proj_1") var conditionProj: Linear
    let returnGate: Bool
    let dim: Int

    init(dim: Int, conditionDim: Int, returnGate: Bool) {
        self.dim = dim
        self.returnGate = returnGate
        self.norm = LayerNorm(dimensions: dim, eps: 1e-5, affine: false)
        self._conditionProj.wrappedValue = Linear(conditionDim, returnGate ? 3 * dim : 2 * dim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> (MLXArray, MLXArray?) {
        let params = conditionProj(silu(condition))
        let xNorm = norm(x)
        if returnGate {
            let shift = params[0..., 0..., 0..<dim]
            let scale = params[0..., 0..., dim..<(2 * dim)]
            let gate = params[0..., 0..., (2 * dim)..<(3 * dim)]
            return (xNorm * (1 + scale) + shift, gate)
        }
        let shift = params[0..., 0..., 0..<dim]
        let scale = params[0..., 0..., dim..<(2 * dim)]
        return (xNorm * (1 + scale) + shift, nil)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(
            to: conditionProj,
            prefix: "\(prefix).condition_proj.1",
            from: weights)
    }
}

final class MioTransformerAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let windowSize: Int
    let rope: RoPE

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(dim: Int, numHeads: Int, windowSize: Int, ropeTheta: Float) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / sqrt(Float(self.headDim))
        self.windowSize = windowSize
        self.rope = RoPE(dimensions: self.headDim, traditional: true, base: ropeTheta)
        self._wq.wrappedValue = Linear(dim, dim, bias: false)
        self._wk.wrappedValue = Linear(dim, dim, bias: false)
        self._wv.wrappedValue = Linear(dim, dim, bias: false)
        self._wo.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        var q = wq(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q, offset: 0)
        k = rope(k, offset: 0)

        let mask = localAttentionMask(sequenceLength: t, windowSize: windowSize, dtype: q.dtype)
        let attended = SDPA.attendAndMerge(
            qHeads: q,
            kHeads: k,
            vHeads: v,
            scale: scale,
            mask: mask)
        return wo(attended)
    }

    private func localAttentionMask(sequenceLength: Int, windowSize: Int, dtype: DType) -> MLXArray? {
        guard windowSize > 0, windowSize < sequenceLength else { return nil }
        let half = windowSize / 2
        var values = [Float](repeating: 0, count: sequenceLength * sequenceLength)
        let blocked = -Float.greatestFiniteMagnitude
        for i in 0..<sequenceLength {
            for j in 0..<sequenceLength where Swift.abs(i - j) > half {
                values[i * sequenceLength + j] = blocked
            }
        }
        return MLXArray(values)
            .reshaped([1, 1, sequenceLength, sequenceLength])
            .asType(dtype)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(to: wq, prefix: "\(prefix).wq", from: weights)
        CommonWeightLoader.applyLinearWeights(to: wk, prefix: "\(prefix).wk", from: weights)
        CommonWeightLoader.applyLinearWeights(to: wv, prefix: "\(prefix).wv", from: weights)
        CommonWeightLoader.applyLinearWeights(to: wo, prefix: "\(prefix).wo", from: weights)
    }
}

final class MioTransformerFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        self._w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        self._w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(to: w1, prefix: "\(prefix).w1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: w2, prefix: "\(prefix).w2", from: weights)
        CommonWeightLoader.applyLinearWeights(to: w3, prefix: "\(prefix).w3", from: weights)
    }
}

final class MioTransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: MioTransformerAttention
    @ModuleInfo(key: "feed_forward") var feedForward: MioTransformerFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: LayerNorm?
    @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm?
    @ModuleInfo(key: "attention_norm_adaln") var attentionAdaLN: MioAdaLNZero?
    @ModuleInfo(key: "ffn_norm_adaln") var ffnAdaLN: MioAdaLNZero?
    let useAdaLN: Bool

    init(dim: Int, numHeads: Int, windowSize: Int, useAdaLN: Bool) {
        self.useAdaLN = useAdaLN
        self._attention.wrappedValue = MioTransformerAttention(
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            ropeTheta: 10_000)
        self._feedForward.wrappedValue = MioTransformerFeedForward(
            dim: dim,
            hiddenDim: Self.feedForwardHiddenDim(for: dim))
        if useAdaLN {
            self._attentionNorm.wrappedValue = nil
            self._ffnNorm.wrappedValue = nil
            self._attentionAdaLN.wrappedValue = MioAdaLNZero(dim: dim, conditionDim: 128, returnGate: true)
            self._ffnAdaLN.wrappedValue = MioAdaLNZero(dim: dim, conditionDim: 128, returnGate: true)
        } else {
            self._attentionNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
            self._ffnNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
            self._attentionAdaLN.wrappedValue = nil
            self._ffnAdaLN.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray?) -> MLXArray {
        let h: MLXArray
        if useAdaLN {
            guard let condition, let attentionAdaLN, let ffnAdaLN else {
                fatalError("MioCodec AdaLN transformer requires speaker condition")
            }
            let (normed, gate) = attentionAdaLN(x, condition: condition)
            guard let gate else {
                fatalError("MioCodec AdaLN attention norm did not return a gate")
            }
            h = x + gate * attention(normed)
            let (ffnNormed, ffnGate) = ffnAdaLN(h, condition: condition)
            guard let ffnGate else {
                fatalError("MioCodec AdaLN feed-forward norm did not return a gate")
            }
            return h + ffnGate * feedForward(ffnNormed)
        } else {
            guard let attentionNorm, let ffnNorm else {
                fatalError("MioCodec transformer missing regular LayerNorm")
            }
            h = x + attention(attentionNorm(x))
            return h + feedForward(ffnNorm(h))
        }
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        attention.loadWeights(prefix: "\(prefix).attention", from: weights)
        feedForward.loadWeights(prefix: "\(prefix).feed_forward", from: weights)
        if useAdaLN {
            attentionAdaLN?.loadWeights(prefix: "\(prefix).attention_norm", from: weights)
            ffnAdaLN?.loadWeights(prefix: "\(prefix).ffn_norm", from: weights)
        } else {
            if let attentionNorm {
                CommonWeightLoader.applyLayerNormWeights(
                    to: attentionNorm,
                    prefix: "\(prefix).attention_norm",
                    from: weights)
            }
            if let ffnNorm {
                CommonWeightLoader.applyLayerNormWeights(
                    to: ffnNorm,
                    prefix: "\(prefix).ffn_norm",
                    from: weights)
            }
        }
    }

    private static func feedForwardHiddenDim(for dim: Int) -> Int {
        let raw = Int(2 * (4 * dim) / 3)
        let multiple = 256
        return multiple * ((raw + multiple - 1) / multiple)
    }
}

final class MioTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [MioTransformerBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm?
    @ModuleInfo(key: "norm_adaln") var normAdaLN: MioAdaLNZero?
    @ModuleInfo(key: "output_proj") var outputProj: Linear?
    let useAdaLN: Bool

    init(
        dim: Int,
        outputDim: Int?,
        numLayers: Int,
        numHeads: Int,
        windowSize: Int,
        useAdaLN: Bool
    ) {
        self.useAdaLN = useAdaLN
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            MioTransformerBlock(
                dim: dim,
                numHeads: numHeads,
                windowSize: windowSize,
                useAdaLN: useAdaLN)
        }
        if useAdaLN {
            self._norm.wrappedValue = nil
            self._normAdaLN.wrappedValue = MioAdaLNZero(dim: dim, conditionDim: 128, returnGate: false)
        } else {
            self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
            self._normAdaLN.wrappedValue = nil
        }
        self._outputProj.wrappedValue = outputDim.map { Linear(dim, $0, bias: true) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, condition: MLXArray? = nil) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, condition: condition)
        }
        if useAdaLN {
            guard let condition, let normAdaLN else {
                fatalError("MioCodec AdaLN transformer requires speaker condition")
            }
            h = normAdaLN(h, condition: condition).0
        } else if let norm {
            h = norm(h)
        }
        if let outputProj {
            h = outputProj(h)
        }
        return h
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        for i in layers.indices {
            layers[i].loadWeights(prefix: "\(prefix).layers.\(i)", from: weights)
        }
        if useAdaLN {
            normAdaLN?.loadWeights(prefix: "\(prefix).norm", from: weights)
        } else if let norm {
            CommonWeightLoader.applyLayerNormWeights(to: norm, prefix: "\(prefix).norm", from: weights)
        }
        if let outputProj {
            CommonWeightLoader.applyLinearWeights(
                to: outputProj,
                prefix: "\(prefix).output_proj",
                from: weights)
        }
    }
}

final class MioResNetBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: MioConv1dNCL
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: MioConv1dNCL

    init(channels: Int) {
        self._norm1.wrappedValue = GroupNorm(
            groupCount: 32,
            dimensions: channels,
            eps: 1e-6,
            affine: true,
            pytorchCompatible: true)
        self._conv1.wrappedValue = MioConv1dNCL(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1)
        self._norm2.wrappedValue = GroupNorm(
            groupCount: 32,
            dimensions: channels,
            eps: 1e-6,
            affine: true,
            pytorchCompatible: true)
        self._conv2.wrappedValue = MioConv1dNCL(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = silu(h)
        h = conv1(h)
        h = norm2(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = silu(h)
        h = conv2(h)
        return x + h
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        applyGroupNormWeights(to: norm1, prefix: "\(prefix).norm1", from: weights)
        CommonWeightLoader.applyConv1dWeights(
            to: conv1.conv,
            prefix: "\(prefix).conv1",
            from: weights,
            transpose: true)
        applyGroupNormWeights(to: norm2, prefix: "\(prefix).norm2", from: weights)
        CommonWeightLoader.applyConv1dWeights(
            to: conv2.conv,
            prefix: "\(prefix).conv2",
            from: weights,
            transpose: true)
    }
}

final class MioResNetStack: Module {
    @ModuleInfo(key: "blocks") var blocks: [MioResNetBlock]

    init(channels: Int, blockCount: Int = 2) {
        self._blocks.wrappedValue = (0..<blockCount).map { _ in MioResNetBlock(channels: channels) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in blocks {
            h = block(h)
        }
        return h
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        for i in blocks.indices {
            blocks[i].loadWeights(prefix: "\(prefix).blocks.\(i)", from: weights)
        }
    }
}

final class MioISTFTHead: Module {
    @ModuleInfo(key: "out") var out: Linear
    let nFFT: Int
    let hopLength: Int

    init(dim: Int, nFFT: Int, hopLength: Int) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self._out.wrappedValue = Linear(dim, nFFT + 2, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = out(x)
        let bins = nFFT / 2 + 1
        let magnitude = clip(exp(y[0..., 0..., 0..<bins]), max: MLXArray(Float(1e2)))
        let phase = y[0..., 0..., bins..<(bins * 2)]
        return istftSame(
            magnitude: magnitude.transposed(0, 2, 1),
            phase: phase.transposed(0, 2, 1),
            nFFT: nFFT,
            hopLength: hopLength)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(to: out, prefix: "\(prefix).out", from: weights)
    }
}

public final class MioCodecWaveDecoder: Module {
    public let config: MioCodecConfig

    @ModuleInfo(key: "wave_prenet") var wavePrenet: MioTransformer
    @ModuleInfo(key: "wave_conv_upsample") var waveConvUpsample: MioConvTranspose1dNCL
    @ModuleInfo(key: "wave_prior_net") var wavePriorNet: MioResNetStack
    @ModuleInfo(key: "wave_decoder") var waveDecoder: MioTransformer
    @ModuleInfo(key: "wave_post_net") var wavePostNet: MioResNetStack
    @ModuleInfo(key: "istft_head") var istftHead: MioISTFTHead

    public init(config: MioCodecConfig = .default) {
        self.config = config
        self._wavePrenet.wrappedValue = MioTransformer(
            dim: config.contentEmbeddingDim,
            outputDim: config.waveDecoderDim,
            numLayers: 6,
            numHeads: 12,
            windowSize: 65,
            useAdaLN: false)
        self._waveConvUpsample.wrappedValue = MioConvTranspose1dNCL(
            inputChannels: config.waveDecoderDim,
            outputChannels: config.waveDecoderDim,
            kernelSize: config.downsampleFactor,
            stride: config.downsampleFactor)
        self._wavePriorNet.wrappedValue = MioResNetStack(channels: config.waveDecoderDim)
        self._waveDecoder.wrappedValue = MioTransformer(
            dim: config.waveDecoderDim,
            outputDim: nil,
            numLayers: 8,
            numHeads: 8,
            windowSize: 65,
            useAdaLN: true)
        self._wavePostNet.wrappedValue = MioResNetStack(channels: config.waveDecoderDim)
        self._istftHead.wrappedValue = MioISTFTHead(
            dim: config.waveDecoderDim,
            nFFT: config.nFFT,
            hopLength: config.hopLength)
        super.init()
    }

    public func decode(
        contentEmbeddings: MLXArray,
        globalEmbedding: MLXArray,
        targetAudioLength: Int? = nil
    ) -> MLXArray {
        let tokenCount = contentEmbeddings.dim(1)
        let plan = MioCodecDecodePlan(
            tokenCount: tokenCount,
            targetAudioLength: targetAudioLength,
            config: config)
        var h = wavePrenet(contentEmbeddings)
        h = waveConvUpsample(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        if h.dim(1) != plan.stftFrames {
            h = linearInterpolateNLC(h, targetLength: plan.stftFrames)
        }
        h = wavePriorNet(h.transposed(0, 2, 1)).transposed(0, 2, 1)

        let condition = normalizeGlobalEmbedding(globalEmbedding)
        h = waveDecoder(h, condition: condition)
        h = wavePostNet(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        return istftHead(h)
    }

    public func loadWeights(from weights: [String: MLXArray]) {
        wavePrenet.loadWeights(prefix: "wave_prenet", from: weights)
        CommonWeightLoader.applyConvTransposed1dWeights(
            to: waveConvUpsample.conv,
            prefix: "wave_conv_upsample",
            from: weights,
            transpose: true)
        wavePriorNet.loadWeights(prefix: "wave_prior_net", from: weights)
        waveDecoder.loadWeights(prefix: "wave_decoder", from: weights)
        wavePostNet.loadWeights(prefix: "wave_post_net", from: weights)
        istftHead.loadWeights(prefix: "istft_head", from: weights)
        eval(self)
    }

    private func normalizeGlobalEmbedding(_ embedding: MLXArray) -> MLXArray {
        if embedding.ndim == 1 {
            return embedding.expandedDimensions(axis: 0).expandedDimensions(axis: 1)
        }
        if embedding.ndim == 2 {
            return embedding.expandedDimensions(axis: 1)
        }
        return embedding
    }
}

private func applyGroupNormWeights(
    to groupNorm: GroupNorm,
    prefix: String,
    from weights: [String: MLXArray]
) {
    var params: [String: NestedItem<String, MLXArray>] = [:]
    if let weight = weights["\(prefix).weight"] {
        params["weight"] = .value(weight)
    }
    if let bias = weights["\(prefix).bias"] {
        params["bias"] = .value(bias)
    }
    if !params.isEmpty {
        groupNorm.update(parameters: ModuleParameters(values: params))
    }
}

private func linearInterpolateNLC(_ x: MLXArray, targetLength: Int) -> MLXArray {
    let currentLength = x.dim(1)
    guard targetLength != currentLength else { return x }
    guard targetLength > 1, currentLength > 1 else {
        let index = MLXArray([Int32(0)])
        return x.take(index, axis: 1)
    }

    var frames: [MLXArray] = []
    frames.reserveCapacity(targetLength)
    let scale = Double(currentLength - 1) / Double(targetLength - 1)
    for i in 0..<targetLength {
        let pos = Double(i) * scale
        let left = Int(floor(pos))
        let right = Swift.min(left + 1, currentLength - 1)
        let frac = Float(pos - Double(left))
        let l = x[0..., left..<(left + 1), 0...]
        let r = x[0..., right..<(right + 1), 0...]
        frames.append(l * (1 - frac) + r * frac)
    }
    return concatenated(frames, axis: 1)
}

private func hannPeriodic(_ size: Int) -> [Float] {
    (0..<size).map { Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double($0) / Double(size)))) }
}

private func istftSame(
    magnitude: MLXArray,
    phase: MLXArray,
    nFFT: Int,
    hopLength: Int
) -> MLXArray {
    let batch = magnitude.dim(0)
    let bins = nFFT / 2 + 1
    let frames = magnitude.dim(2)

    let mag = magnitude.transposed(0, 2, 1)
    let ph = phase.transposed(0, 2, 1)
    let real = mag * cos(ph)
    let imag = mag * sin(ph)

    let fullReal: MLXArray
    let fullImag: MLXArray
    if bins >= 2 {
        let mirror = MLXArray((1...(bins - 2)).reversed().map { Int32($0) })
        fullReal = concatenated([real, real.take(mirror, axis: 2)], axis: 2)
        fullImag = concatenated([imag, -(imag.take(mirror, axis: 2))], axis: 2)
    } else {
        fullReal = real
        fullImag = imag
    }

    let invN = 1.0 / Double(nFFT)
    var idftCos = [Float](repeating: 0, count: nFFT * nFFT)
    var idftSin = [Float](repeating: 0, count: nFFT * nFFT)
    for n in 0..<nFFT {
        for k in 0..<nFFT {
            let angle = 2.0 * Double.pi * Double(n) * Double(k) / Double(nFFT)
            idftCos[n * nFFT + k] = Float(cos(angle) * invN)
            idftSin[n * nFFT + k] = Float(sin(angle) * invN)
        }
    }
    let cosMat = MLXArray(idftCos).reshaped([nFFT, nFFT])
    let sinMat = MLXArray(idftSin).reshaped([nFFT, nFFT])
    let timeDomain = matmul(fullReal, cosMat.transposed()) - matmul(fullImag, sinMat.transposed())

    let windowValues = hannPeriodic(nFFT)
    let window = MLXArray(windowValues)
    let windowed = timeDomain * window

    let segmentsPerFrame = nFFT / hopLength
    let outHops = frames + segmentsPerFrame - 1
    let outLength = outHops * hopLength
    let segments = windowed.reshaped([batch, frames, segmentsPerFrame, hopLength])

    var accumulated = MLXArray.zeros([batch, outLength])
    for segmentIndex in 0..<segmentsPerFrame {
        let segment = segments[0..., 0..., segmentIndex, 0...]
        let flat = segment.reshaped([batch, frames * hopLength])
        let leftPad = segmentIndex * hopLength
        let rightPad = outLength - leftPad - frames * hopLength
        var pieces: [MLXArray] = []
        if leftPad > 0 {
            pieces.append(MLXArray.zeros([batch, leftPad]))
        }
        pieces.append(flat)
        if rightPad > 0 {
            pieces.append(MLXArray.zeros([batch, rightPad]))
        }
        accumulated = accumulated + concatenated(pieces, axis: 1)
    }

    var envelope = [Float](repeating: 0, count: outLength)
    for frame in 0..<frames {
        for n in 0..<nFFT {
            let index = frame * hopLength + n
            if index < outLength {
                envelope[index] += windowValues[n] * windowValues[n]
            }
        }
    }
    for i in envelope.indices where envelope[i] < 1e-8 {
        envelope[i] = 1e-8
    }

    var audio = accumulated / MLXArray(envelope).reshaped([1, outLength])
    let trim = (nFFT - hopLength) / 2
    if trim > 0, audio.dim(1) > 2 * trim {
        audio = audio[0..., trim..<(audio.dim(1) - trim)]
    }
    return audio
}
