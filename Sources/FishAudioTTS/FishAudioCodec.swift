import Foundation
import AudioCommon
import MLX
import MLXCommon
import MLXFast
import MLXNN

public enum FishAudioCodecDefaults {
    public static let sampleRate = 44_100
    public static let inputDim = 1_024
    public static let decoderDim = 1_536
    public static let semanticCodebookSize = 4_096
    public static let residualCodebookSize = 1_024
    public static let codebookDim = 8
    public static let residualCodebooks = 9
    public static let totalCodebooks = 10
    public static let encoderRates = [2, 4, 8, 8]
    public static let upsampleFactors = [2, 2]
    public static let decoderRates = [8, 8, 4, 2]
    public static let hopLength = 512
    public static let samplesPerFrame = 2_048
}

struct FishCodecQuantizeResult {
    let quantized: MLXArray
    let codes: MLXArray
}

private func fishPadNCL(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    guard left > 0 || right > 0 else { return x }
    return padded(
        x,
        widths: [
            .init((low: 0, high: 0)),
            .init((low: 0, high: 0)),
            .init((low: left, high: right)),
        ],
        value: MLXArray(Float(0)))
}

private func fishExtraPadding(length: Int, kernelSize: Int, stride: Int, paddingTotal: Int) -> Int {
    let nFrames = (Double(length - kernelSize + paddingTotal) / Double(stride)) + 1.0
    let idealLength = (Int(ceil(nFrames)) - 1) * stride + (kernelSize - paddingTotal)
    return max(0, idealLength - length)
}

private func fishUpdate(_ module: Module, values: [String: MLXArray]) {
    guard !values.isEmpty else { return }
    module.update(parameters: ModuleParameters(values: values.mapValues { .value($0) }))
}

private func fishFuseWeightNormDim0(g: MLXArray, v: MLXArray) -> MLXArray {
    let vf = v.asType(.float32)
    let gf = g.asType(.float32)
    let axes = Array(1..<vf.ndim)
    let norm = sqrt((vf * vf).sum(axes: axes, keepDims: true))
    return gf * vf / maximum(norm, MLXArray(Float(1e-12)))
}

private func fishApplyConv1d(
    _ conv: Conv1d,
    prefix: String,
    from weights: [String: MLXArray],
    transpose: Bool = true
) {
    var values: [String: MLXArray] = [:]
    if let weight = weights["\(prefix).weight"] {
        values["weight"] = transpose ? weight.asType(.float32).transposed(0, 2, 1) : weight.asType(.float32)
    }
    if let bias = weights["\(prefix).bias"] {
        values["bias"] = bias.asType(.float32)
    }
    fishUpdate(conv, values: values)
}

private func fishApplyConvTransposed1d(
    _ conv: ConvTransposed1d,
    prefix: String,
    from weights: [String: MLXArray],
    transpose: Bool = true
) {
    var values: [String: MLXArray] = [:]
    if let weight = weights["\(prefix).weight"] {
        values["weight"] = transpose ? weight.asType(.float32).transposed(1, 2, 0) : weight.asType(.float32)
    }
    if let bias = weights["\(prefix).bias"] {
        values["bias"] = bias.asType(.float32)
    }
    fishUpdate(conv, values: values)
}

private func fishApplyWeightNormConv1d(
    _ conv: Conv1d,
    prefix: String,
    from weights: [String: MLXArray]
) {
    var values: [String: MLXArray] = [:]
    let g = weights["\(prefix).parametrizations.weight.original0"] ?? weights["\(prefix).weight_g"]
    let v = weights["\(prefix).parametrizations.weight.original1"] ?? weights["\(prefix).weight_v"]
    if let g, let v {
        let fused = fishFuseWeightNormDim0(g: g, v: v).transposed(0, 2, 1)
        values["weight"] = fused
    }
    if let bias = weights["\(prefix).bias"] {
        values["bias"] = bias.asType(.float32)
    }
    fishUpdate(conv, values: values)
}

private func fishApplyWeightNormConvTransposed1d(
    _ conv: ConvTransposed1d,
    prefix: String,
    from weights: [String: MLXArray]
) {
    var values: [String: MLXArray] = [:]
    let g = weights["\(prefix).parametrizations.weight.original0"] ?? weights["\(prefix).weight_g"]
    let v = weights["\(prefix).parametrizations.weight.original1"] ?? weights["\(prefix).weight_v"]
    if let g, let v {
        let fused = fishFuseWeightNormDim0(g: g, v: v).transposed(1, 2, 0)
        values["weight"] = fused
    }
    if let bias = weights["\(prefix).bias"] {
        values["bias"] = bias.asType(.float32)
    }
    fishUpdate(conv, values: values)
}

private func fishApplySnake(_ snake: FishCodecSnake1d, prefix: String, from weights: [String: MLXArray]) {
    guard let alpha = weights["\(prefix).alpha"] else { return }
    fishUpdate(snake, values: ["alpha": alpha.asType(.float32)])
}

private func fishApplyLayerScale(
    _ layerScale: FishCodecLayerScale,
    prefix: String,
    from weights: [String: MLXArray]
) {
    guard let gamma = weights["\(prefix).gamma"] else { return }
    fishUpdate(layerScale, values: ["gamma": gamma.asType(.float32)])
}

final class FishCodecSnake1d: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = sin(alpha * x)
        return x + (1.0 / (alpha + 1e-9)) * s * s
    }
}

final class FishCodecConv1dNCL: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    private let causal: Bool
    private let effectiveKernelSize: Int
    private let stride: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = false
    ) {
        self.causal = causal
        self.effectiveKernelSize = (kernelSize - 1) * dilation + 1
        self.stride = stride
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            dilation: dilation,
            groups: groups,
            bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if causal {
            let left = effectiveKernelSize - stride
            let right = fishExtraPadding(
                length: x.dim(2),
                kernelSize: effectiveKernelSize,
                stride: stride,
                paddingTotal: left)
            h = fishPadNCL(h, left: left, right: right)
        }
        return conv(h.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

final class FishCodecConvTransposed1dNCL: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d
    private let trimRight: Int

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int,
        bias: Bool = true,
        causal: Bool = false
    ) {
        self.trimRight = causal ? max(0, kernelSize - stride) : 0
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: 0,
            bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        if trimRight > 0 {
            y = y[0..., 0..., 0..<(y.dim(2) - trimRight)]
        }
        return y
    }
}

final class FishCodecLayerScale: Module {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(channels: Int, initValue: Float = 0.01) {
        self._gamma.wrappedValue = MLXArray(Array(repeating: initValue, count: channels))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

final class FishCodecConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var dwConv: FishCodecConv1dNCL
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwConv1: Linear
    @ModuleInfo(key: "pwconv2") var pwConv2: Linear
    @ModuleInfo var gamma: FishCodecLayerScale

    init(dim: Int, kernelSize: Int = 7, mlpRatio: Int = 4) {
        self._dwConv.wrappedValue = FishCodecConv1dNCL(
            inChannels: dim,
            outChannels: dim,
            kernelSize: kernelSize,
            groups: dim,
            causal: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwConv1.wrappedValue = Linear(dim, dim * mlpRatio)
        self._pwConv2.wrappedValue = Linear(dim * mlpRatio, dim)
        self._gamma.wrappedValue = FishCodecLayerScale(channels: dim, initValue: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, applyResidual: Bool = true) -> MLXArray {
        let residual = x
        var h = dwConv(x).transposed(0, 2, 1)
        h = norm(h)
        h = pwConv1(h)
        h = gelu(h)
        h = pwConv2(h)
        h = gamma(h).transposed(0, 2, 1)
        return applyResidual ? residual + h : h
    }
}

final class FishCodecUpsampleBlock: Module {
    @ModuleInfo(key: "0") var conv: FishCodecConvTransposed1dNCL
    @ModuleInfo(key: "1") var convNext: FishCodecConvNeXtBlock

    init(channels: Int, factor: Int) {
        self._conv.wrappedValue = FishCodecConvTransposed1dNCL(
            inChannels: channels,
            outChannels: channels,
            kernelSize: factor,
            stride: factor,
            causal: true)
        self._convNext.wrappedValue = FishCodecConvNeXtBlock(dim: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convNext(conv(x))
    }
}

final class FishCodecDownsampleBlock: Module {
    @ModuleInfo(key: "0") var conv: FishCodecConv1dNCL
    @ModuleInfo(key: "1") var convNext: FishCodecConvNeXtBlock

    init(channels: Int, factor: Int) {
        self._conv.wrappedValue = FishCodecConv1dNCL(
            inChannels: channels,
            outChannels: channels,
            kernelSize: factor,
            stride: factor,
            causal: true)
        self._convNext.wrappedValue = FishCodecConvNeXtBlock(dim: channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convNext(conv(x))
    }
}

final class FishCodecVectorQuantizer: Module {
    @ModuleInfo var codebook: Embedding
    @ModuleInfo(key: "in_proj") var inProj: FishCodecConv1dNCL
    @ModuleInfo(key: "out_proj") var outProj: FishCodecConv1dNCL

    init(codebookSize: Int, codebookDim: Int, outputDim: Int) {
        self._codebook.wrappedValue = Embedding(
            embeddingCount: codebookSize,
            dimensions: codebookDim)
        self._inProj.wrappedValue = FishCodecConv1dNCL(
            inChannels: outputDim,
            outChannels: codebookDim,
            kernelSize: 1)
        self._outProj.wrappedValue = FishCodecConv1dNCL(
            inChannels: codebookDim,
            outChannels: outputDim,
            kernelSize: 1)
        super.init()
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        outProj(codebook(codes).transposed(0, 2, 1))
    }

    func encode(_ input: MLXArray) -> FishCodecQuantizeResult {
        let projected = inProj(input).transposed(0, 2, 1)
        let table = codebook.weight.asType(.float32)
        let xSq = (projected * projected).sum(axis: -1, keepDims: true)
        let cSq = (table * table).sum(axis: -1, keepDims: true).T
        let logits = matmul(projected, table.T)
        let distances = xSq - 2 * logits + cSq
        let codes = argMin(distances, axis: -1).asType(.int32)
        return FishCodecQuantizeResult(quantized: decode(codes), codes: codes)
    }
}

final class FishCodecResidualVectorQuantizer: Module {
    @ModuleInfo var quantizers: [FishCodecVectorQuantizer]
    let numQuantizers: Int
    let codebookSize: Int

    init(numQuantizers: Int, codebookSize: Int, codebookDim: Int, outputDim: Int) {
        self.numQuantizers = numQuantizers
        self.codebookSize = codebookSize
        self._quantizers.wrappedValue = (0..<numQuantizers).map { _ in
            FishCodecVectorQuantizer(
                codebookSize: codebookSize,
                codebookDim: codebookDim,
                outputDim: outputDim)
        }
        super.init()
    }

    func decode(_ codes: MLXArray) -> MLXArray {
        let clamped = clip(codes, min: Int32(0), max: Int32(codebookSize - 1)).asType(.int32)
        var acc: MLXArray?
        for index in 0..<numQuantizers {
            let decoded = quantizers[index].decode(clamped[0..., index, 0...])
            acc = acc.map { $0 + decoded } ?? decoded
        }
        return acc ?? MLXArray.zeros([codes.dim(0), FishAudioCodecDefaults.inputDim, codes.dim(2)])
    }

    func encode(_ input: MLXArray) -> FishCodecQuantizeResult {
        var residual = input
        var acc: MLXArray?
        var codeRows: [MLXArray] = []
        codeRows.reserveCapacity(numQuantizers)

        for quantizer in quantizers {
            let result = quantizer.encode(residual)
            residual = residual - result.quantized
            acc = acc.map { $0 + result.quantized } ?? result.quantized
            codeRows.append(result.codes.expandedDimensions(axis: 1))
        }

        return FishCodecQuantizeResult(
            quantized: acc ?? MLXArray.zeros(input.shape),
            codes: concatenated(codeRows, axis: 1))
    }
}

final class FishCodecTransformerAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var wqkv: Linear
    @ModuleInfo var wo: Linear

    let rope: RoPE

    init(hiddenSize: Int = 1_024, numHeads: Int = 16, headDim: Int = 64, ropeTheta: Float = 10_000) {
        self.numHeads = numHeads
        self.headDim = headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self._wqkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: false)
        self._wo.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let hidden = numHeads * headDim
        let parts = split(wqkv(x), indices: [hidden, hidden * 2], axis: -1)
        var q = parts[0].reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q, offset: 0)
        k = rope(k, offset: 0)

        return wo(SDPA.attendAndMerge(
            qHeads: q,
            kHeads: k,
            vHeads: v,
            scale: scale,
            mask: mask))
    }
}

final class FishCodecTransformerFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(hiddenSize: Int = 1_024, intermediateSize: Int = 3_072) {
        self._w1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self._w3.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class FishCodecTransformerLayer: Module {
    @ModuleInfo var attention: FishCodecTransformerAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FishCodecTransformerFeedForward
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "attention_layer_scale") var attentionLayerScale: FishCodecLayerScale
    @ModuleInfo(key: "ffn_layer_scale") var ffnLayerScale: FishCodecLayerScale

    override init() {
        self._attention.wrappedValue = FishCodecTransformerAttention()
        self._feedForward.wrappedValue = FishCodecTransformerFeedForward()
        self._attentionNorm.wrappedValue = RMSNorm(dimensions: 1_024, eps: 1e-5)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: 1_024, eps: 1e-5)
        self._attentionLayerScale.wrappedValue = FishCodecLayerScale(channels: 1_024)
        self._ffnLayerScale.wrappedValue = FishCodecLayerScale(channels: 1_024)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + attentionLayerScale(attention(attentionNorm(x), mask: mask))
        return h + ffnLayerScale(feedForward(ffnNorm(h)))
    }
}

final class FishCodecPostTransformer: Module {
    @ModuleInfo var layers: [FishCodecTransformerLayer]
    @ModuleInfo var norm: RMSNorm
    let windowSize: Int

    init(layerCount: Int = 8, windowSize: Int = 128) {
        self.windowSize = windowSize
        self._layers.wrappedValue = (0..<layerCount).map { _ in FishCodecTransformerLayer() }
        self._norm.wrappedValue = RMSNorm(dimensions: 1_024, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        let mask = makeWindowMask(length: h.dim(1), dtype: h.dtype)
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return norm(h).transposed(0, 2, 1)
    }

    private func makeWindowMask(length: Int, dtype: DType) -> MLXArray? {
        guard length > 1 else { return nil }
        let rows = MLXArray(0..<Int32(length)).expandedDimensions(axis: 1)
        let cols = MLXArray(0..<Int32(length)).expandedDimensions(axis: 0)
        let causal = cols .<= rows
        let inWindow = cols .>= (rows - Int32(windowSize - 1))
        let allowed = causal .&& inWindow
        return MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .reshaped(1, 1, length, length)
            .asType(dtype)
    }
}

final class FishCodecQuantizer: Module {
    @ModuleInfo var downsample: [FishCodecDownsampleBlock]
    @ModuleInfo(key: "pre_module") var preModule: FishCodecPostTransformer
    @ModuleInfo(key: "semantic_quantizer") var semanticQuantizer: FishCodecResidualVectorQuantizer
    @ModuleInfo var quantizer: FishCodecResidualVectorQuantizer
    @ModuleInfo(key: "post_module") var postModule: FishCodecPostTransformer
    @ModuleInfo var upsample: [FishCodecUpsampleBlock]

    override init() {
        self._downsample.wrappedValue = FishAudioCodecDefaults.upsampleFactors.map {
            FishCodecDownsampleBlock(channels: FishAudioCodecDefaults.inputDim, factor: $0)
        }
        self._preModule.wrappedValue = FishCodecPostTransformer()
        self._semanticQuantizer.wrappedValue = FishCodecResidualVectorQuantizer(
            numQuantizers: 1,
            codebookSize: FishAudioCodecDefaults.semanticCodebookSize,
            codebookDim: FishAudioCodecDefaults.codebookDim,
            outputDim: FishAudioCodecDefaults.inputDim)
        self._quantizer.wrappedValue = FishCodecResidualVectorQuantizer(
            numQuantizers: FishAudioCodecDefaults.residualCodebooks,
            codebookSize: FishAudioCodecDefaults.residualCodebookSize,
            codebookDim: FishAudioCodecDefaults.codebookDim,
            outputDim: FishAudioCodecDefaults.inputDim)
        self._postModule.wrappedValue = FishCodecPostTransformer()
        self._upsample.wrappedValue = FishAudioCodecDefaults.upsampleFactors.map {
            FishCodecUpsampleBlock(channels: FishAudioCodecDefaults.inputDim, factor: $0)
        }
        super.init()
    }

    func decode(_ indices: MLXArray) -> MLXArray {
        let semantic = semanticQuantizer.decode(indices[0..., 0..<1, 0...])
        let residual = quantizer.decode(indices[0..., 1..., 0...])
        var h = semantic + residual
        h = postModule(h)
        for block in upsample {
            h = block(h)
        }
        return h
    }

    func encode(_ z: MLXArray) -> MLXArray {
        var h = z
        for block in downsample {
            h = block(h)
        }
        h = preModule(h)
        let semantic = semanticQuantizer.encode(h)
        let residual = quantizer.encode(h - semantic.quantized)
        return concatenated([semantic.codes, residual.codes], axis: 1).asType(.int32)
    }
}

final class FishCodecEncoderBlock: Module {
    @ModuleInfo(key: "block") var block: [Module]
    let residual1: FishCodecResidualUnit
    let residual2: FishCodecResidualUnit
    let residual3: FishCodecResidualUnit
    let snake: FishCodecSnake1d
    let conv: FishCodecConv1dNCL
    let transformer: FishCodecPostTransformer?

    init(inputDim: Int, outputDim: Int, stride: Int, transformerLayers: Int = 0) {
        let residual1 = FishCodecResidualUnit(channels: inputDim, dilation: 1)
        let residual2 = FishCodecResidualUnit(channels: inputDim, dilation: 3)
        let residual3 = FishCodecResidualUnit(channels: inputDim, dilation: 9)
        let snake = FishCodecSnake1d(channels: inputDim)
        let conv = FishCodecConv1dNCL(
            inChannels: inputDim,
            outChannels: outputDim,
            kernelSize: stride * 2,
            stride: stride,
            causal: true)
        let transformer = transformerLayers > 0
            ? FishCodecPostTransformer(layerCount: transformerLayers, windowSize: 512)
            : nil

        self.residual1 = residual1
        self.residual2 = residual2
        self.residual3 = residual3
        self.snake = snake
        self.conv = conv
        self.transformer = transformer
        var modules: [Module] = [residual1, residual2, residual3, snake, conv]
        if let transformer { modules.append(transformer) }
        self._block.wrappedValue = modules
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = residual1(x)
        h = residual2(h)
        h = residual3(h)
        h = conv(snake(h))
        if let transformer {
            h = transformer(h)
        }
        return h
    }
}

final class FishCodecDACEncoder: Module {
    @ModuleInfo(key: "block") var block: [Module]
    let inputConv: FishCodecConv1dNCL
    let blocks: [FishCodecEncoderBlock]
    let finalSnake: FishCodecSnake1d
    let finalConv: FishCodecConv1dNCL

    override init() {
        let inputConv = FishCodecConv1dNCL(
            inChannels: 1,
            outChannels: 64,
            kernelSize: 7,
            causal: true)
        var modules: [Module] = [inputConv]
        var blocks: [FishCodecEncoderBlock] = []
        var currentDim = 64
        for (index, stride) in FishAudioCodecDefaults.encoderRates.enumerated() {
            let outputDim = currentDim * 2
            let block = FishCodecEncoderBlock(
                inputDim: currentDim,
                outputDim: outputDim,
                stride: stride,
                transformerLayers: index == FishAudioCodecDefaults.encoderRates.count - 1 ? 4 : 0)
            blocks.append(block)
            modules.append(block)
            currentDim = outputDim
        }
        let finalSnake = FishCodecSnake1d(channels: currentDim)
        let finalConv = FishCodecConv1dNCL(
            inChannels: currentDim,
            outChannels: FishAudioCodecDefaults.inputDim,
            kernelSize: 3,
            causal: true)
        modules.append(finalSnake)
        modules.append(finalConv)

        self.inputConv = inputConv
        self.blocks = blocks
        self.finalSnake = finalSnake
        self.finalConv = finalConv
        self._block.wrappedValue = modules
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputConv(x)
        for block in blocks {
            h = block(h)
        }
        return finalConv(finalSnake(h))
    }
}

final class FishCodecResidualUnit: Module {
    @ModuleInfo(key: "block") var block: [Module]
    let snake1: FishCodecSnake1d
    let conv1: FishCodecConv1dNCL
    let snake2: FishCodecSnake1d
    let conv2: FishCodecConv1dNCL

    init(channels: Int, dilation: Int) {
        let snake1 = FishCodecSnake1d(channels: channels)
        let conv1 = FishCodecConv1dNCL(
            inChannels: channels,
            outChannels: channels,
            kernelSize: 7,
            dilation: dilation,
            causal: true)
        let snake2 = FishCodecSnake1d(channels: channels)
        let conv2 = FishCodecConv1dNCL(
            inChannels: channels,
            outChannels: channels,
            kernelSize: 1,
            causal: true)
        self.snake1 = snake1
        self.conv1 = conv1
        self.snake2 = snake2
        self.conv2 = conv2
        self._block.wrappedValue = [snake1, conv1, snake2, conv2]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(snake1(x))
        h = conv2(snake2(h))
        var residual = x
        let pad = x.dim(2) - h.dim(2)
        if pad > 0 {
            residual = residual[0..., 0..., 0..<(residual.dim(2) - pad)]
        }
        return residual + h
    }
}

final class FishCodecDecoderBlock: Module {
    @ModuleInfo(key: "block") var block: [Module]
    let snake: FishCodecSnake1d
    let convTranspose: FishCodecConvTransposed1dNCL
    let residual1: FishCodecResidualUnit
    let residual2: FishCodecResidualUnit
    let residual3: FishCodecResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int) {
        let snake = FishCodecSnake1d(channels: inputDim)
        let convTranspose = FishCodecConvTransposed1dNCL(
            inChannels: inputDim,
            outChannels: outputDim,
            kernelSize: stride * 2,
            stride: stride,
            causal: true)
        let residual1 = FishCodecResidualUnit(channels: outputDim, dilation: 1)
        let residual2 = FishCodecResidualUnit(channels: outputDim, dilation: 3)
        let residual3 = FishCodecResidualUnit(channels: outputDim, dilation: 9)
        self.snake = snake
        self.convTranspose = convTranspose
        self.residual1 = residual1
        self.residual2 = residual2
        self.residual3 = residual3
        self._block.wrappedValue = [snake, convTranspose, residual1, residual2, residual3]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convTranspose(snake(x))
        h = residual1(h)
        h = residual2(h)
        h = residual3(h)
        return h
    }
}

final class FishCodecDACDecoder: Module {
    @ModuleInfo(key: "model") var model: [Module]
    let inputConv: FishCodecConv1dNCL
    let blocks: [FishCodecDecoderBlock]
    let finalSnake: FishCodecSnake1d
    let finalConv: FishCodecConv1dNCL

    override init() {
        let inputConv = FishCodecConv1dNCL(
            inChannels: FishAudioCodecDefaults.inputDim,
            outChannels: FishAudioCodecDefaults.decoderDim,
            kernelSize: 7,
            causal: true)
        var blocks: [FishCodecDecoderBlock] = []
        var modules: [Module] = [inputConv]
        for (index, rate) in FishAudioCodecDefaults.decoderRates.enumerated() {
            let inputDim = FishAudioCodecDefaults.decoderDim / (1 << index)
            let outputDim = FishAudioCodecDefaults.decoderDim / (1 << (index + 1))
            let block = FishCodecDecoderBlock(inputDim: inputDim, outputDim: outputDim, stride: rate)
            blocks.append(block)
            modules.append(block)
        }
        let finalChannels = FishAudioCodecDefaults.decoderDim / (1 << FishAudioCodecDefaults.decoderRates.count)
        let finalSnake = FishCodecSnake1d(channels: finalChannels)
        let finalConv = FishCodecConv1dNCL(
            inChannels: finalChannels,
            outChannels: 1,
            kernelSize: 7,
            causal: true)
        modules.append(finalSnake)
        modules.append(finalConv)

        self.inputConv = inputConv
        self.blocks = blocks
        self.finalSnake = finalSnake
        self.finalConv = finalConv
        self._model.wrappedValue = modules
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputConv(x)
        for block in blocks {
            h = block(h)
        }
        h = finalConv(finalSnake(h))
        return tanh(h)
    }
}

public final class FishAudioCodec: Module {
    @ModuleInfo var encoder: FishCodecDACEncoder
    @ModuleInfo var quantizer: FishCodecQuantizer
    @ModuleInfo var decoder: FishCodecDACDecoder

    public let sampleRate = FishAudioCodecDefaults.sampleRate

    public override init() {
        self._encoder.wrappedValue = FishCodecDACEncoder()
        self._quantizer.wrappedValue = FishCodecQuantizer()
        self._decoder.wrappedValue = FishCodecDACDecoder()
        super.init()
    }

    public static func load(from directory: URL) throws -> FishAudioCodec {
        let codec = FishAudioCodec()
        try codec.loadWeights(from: directory.appendingPathComponent("codec.safetensors"))
        return codec
    }

    public func loadWeights(from codecSafetensors: URL) throws {
        guard FileManager.default.fileExists(atPath: codecSafetensors.path) else {
            throw FishAudioError.missingFile(codecSafetensors)
        }
        let weights = try MLX.loadArrays(url: codecSafetensors).mapValues { $0.asType(.float32) }
        loadEncoderWeights(from: weights)
        loadQuantizerWeights(from: weights)
        loadDecoderWeights(from: weights)
        eval(parameters())
    }

    public func encode(
        audio samples: [Float],
        sampleRate inputSampleRate: Int = FishAudioCodecDefaults.sampleRate
    ) throws -> FishAudioGeneratedCodebooks {
        guard !samples.isEmpty else {
            throw FishAudioError.invalidCodebookShape("reference audio must not be empty")
        }
        let prepared = inputSampleRate == FishAudioCodecDefaults.sampleRate
            ? samples
            : AudioFileLoader.resample(
                samples,
                from: inputSampleRate,
                to: FishAudioCodecDefaults.sampleRate)
        guard !prepared.isEmpty else {
            throw FishAudioError.invalidCodebookShape("reference audio resampled to an empty buffer")
        }

        let frame = FishAudioCodecDefaults.samplesPerFrame
        let paddedCount = ((prepared.count + frame - 1) / frame) * frame
        var paddedSamples = prepared
        if paddedSamples.count < paddedCount {
            paddedSamples.append(contentsOf: repeatElement(Float(0), count: paddedCount - paddedSamples.count))
        }

        let wav = MLXArray(paddedSamples).reshaped([1, 1, paddedSamples.count])
        let latent = encoder(wav)
        let codes = quantizer.encode(latent)
        eval(codes)
        return try Self.generatedCodebooks(from: codes)
    }

    public func decode(_ audioCodes: MLXArray) throws -> MLXArray {
        guard audioCodes.ndim == 3 else {
            throw FishAudioError.invalidCodebookShape("codec input must be [batch, codebooks, frames]")
        }
        guard audioCodes.dim(1) == FishAudioCodecDefaults.totalCodebooks else {
            throw FishAudioError.invalidCodebookShape(
                "codec expected \(FishAudioCodecDefaults.totalCodebooks) codebooks, got \(audioCodes.dim(1))")
        }
        guard audioCodes.dim(2) > 0 else {
            return MLXArray.zeros([0], dtype: .float32)
        }

        let z = quantizer.decode(audioCodes.asType(.int32))
        let wav = decoder(z)
        return wav.reshaped([-1]).asType(.float32)
    }

    public func decode(_ generated: FishAudioGeneratedCodebooks) throws -> [Float] {
        guard generated.codebookCount == FishAudioCodecDefaults.totalCodebooks else {
            throw FishAudioError.invalidCodebookShape(
                "codec expected \(FishAudioCodecDefaults.totalCodebooks) codebooks, got \(generated.codebookCount)")
        }
        guard generated.frameCount > 0 else { return [] }
        let flat = generated.codes.flatMap { row in row.map(Int32.init) }
        let codes = MLXArray(flat).reshaped([
            1,
            FishAudioCodecDefaults.totalCodebooks,
            generated.frameCount,
        ])
        let wav = try decode(codes)
        eval(wav)
        return wav.asArray(Float.self)
    }

    private static func generatedCodebooks(from codes: MLXArray) throws -> FishAudioGeneratedCodebooks {
        guard codes.ndim == 3 else {
            throw FishAudioError.invalidCodebookShape("encoded codec output must be [batch, codebooks, frames]")
        }
        guard codes.dim(0) == 1 else {
            throw FishAudioError.invalidCodebookShape("encoded codec output must have batch size 1")
        }
        guard codes.dim(1) == FishAudioCodecDefaults.totalCodebooks else {
            throw FishAudioError.invalidCodebookShape(
                "encoded codec output expected \(FishAudioCodecDefaults.totalCodebooks) codebooks, got \(codes.dim(1))")
        }

        let frameCount = codes.dim(2)
        let flat = codes.asType(.int32).asArray(Int32.self).map(Int.init)
        var rows: [[Int]] = []
        rows.reserveCapacity(FishAudioCodecDefaults.totalCodebooks)
        for codebook in 0..<FishAudioCodecDefaults.totalCodebooks {
            let start = codebook * frameCount
            rows.append(Array(flat[start..<(start + frameCount)]))
        }
        return FishAudioGeneratedCodebooks(codes: rows)
    }

    private func loadEncoderWeights(from weights: [String: MLXArray]) {
        fishApplyWeightNormConv1d(encoder.inputConv.conv, prefix: "encoder.block.0.conv", from: weights)
        for (index, block) in encoder.blocks.enumerated() {
            let prefix = "encoder.block.\(index + 1).block"
            loadResidualUnit(block.residual1, prefix: "\(prefix).0.block", from: weights)
            loadResidualUnit(block.residual2, prefix: "\(prefix).1.block", from: weights)
            loadResidualUnit(block.residual3, prefix: "\(prefix).2.block", from: weights)
            fishApplySnake(block.snake, prefix: "\(prefix).3", from: weights)
            fishApplyWeightNormConv1d(block.conv.conv, prefix: "\(prefix).4.conv", from: weights)
            if let transformer = block.transformer {
                loadPostTransformer(transformer, prefix: "\(prefix).5", from: weights)
            }
        }
        fishApplySnake(encoder.finalSnake, prefix: "encoder.block.5", from: weights)
        fishApplyWeightNormConv1d(encoder.finalConv.conv, prefix: "encoder.block.6.conv", from: weights)
    }

    private func loadQuantizerWeights(from weights: [String: MLXArray]) {
        for (index, block) in quantizer.downsample.enumerated() {
            let prefix = "quantizer.downsample.\(index)"
            fishApplyConv1d(block.conv.conv, prefix: "\(prefix).0.conv", from: weights)
            loadConvNeXt(block.convNext, prefix: "\(prefix).1", from: weights)
        }
        loadPostTransformer(quantizer.preModule, prefix: "quantizer.pre_module", from: weights)
        loadResidualVectorQuantizer(
            quantizer.semanticQuantizer,
            prefix: "quantizer.semantic_quantizer",
            from: weights)
        loadResidualVectorQuantizer(
            quantizer.quantizer,
            prefix: "quantizer.quantizer",
            from: weights)
        loadPostTransformer(quantizer.postModule, prefix: "quantizer.post_module", from: weights)
        for (index, block) in quantizer.upsample.enumerated() {
            let prefix = "quantizer.upsample.\(index)"
            fishApplyConvTransposed1d(block.conv.conv, prefix: "\(prefix).0.conv", from: weights)
            loadConvNeXt(block.convNext, prefix: "\(prefix).1", from: weights)
        }
    }

    private func loadResidualVectorQuantizer(
        _ rvq: FishCodecResidualVectorQuantizer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        for (index, quantizer) in rvq.quantizers.enumerated() {
            let qPrefix = "\(prefix).quantizers.\(index)"
            CommonWeightLoader.applyEmbeddingWeights(
                to: quantizer.codebook,
                prefix: "\(qPrefix).codebook",
                from: weights)
            fishApplyWeightNormConv1d(quantizer.inProj.conv, prefix: "\(qPrefix).in_proj", from: weights)
            fishApplyWeightNormConv1d(quantizer.outProj.conv, prefix: "\(qPrefix).out_proj", from: weights)
        }
    }

    private func loadPostTransformer(
        _ transformer: FishCodecPostTransformer,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        CommonWeightLoader.applyRMSNormWeights(to: transformer.norm, prefix: "\(prefix).norm", from: weights)
        for (index, layer) in transformer.layers.enumerated() {
            let lp = "\(prefix).layers.\(index)"
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.attentionNorm,
                prefix: "\(lp).attention_norm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.ffnNorm,
                prefix: "\(lp).ffn_norm",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.wqkv,
                prefix: "\(lp).attention.wqkv",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.wo,
                prefix: "\(lp).attention.wo",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.feedForward.w1,
                prefix: "\(lp).feed_forward.w1",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.feedForward.w2,
                prefix: "\(lp).feed_forward.w2",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.feedForward.w3,
                prefix: "\(lp).feed_forward.w3",
                from: weights)
            fishApplyLayerScale(
                layer.attentionLayerScale,
                prefix: "\(lp).attention_layer_scale",
                from: weights)
            fishApplyLayerScale(
                layer.ffnLayerScale,
                prefix: "\(lp).ffn_layer_scale",
                from: weights)
        }
    }

    private func loadConvNeXt(
        _ block: FishCodecConvNeXtBlock,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        fishApplyConv1d(block.dwConv.conv, prefix: "\(prefix).dwconv.conv", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: block.norm, prefix: "\(prefix).norm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: block.pwConv1, prefix: "\(prefix).pwconv1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: block.pwConv2, prefix: "\(prefix).pwconv2", from: weights)
        fishApplyLayerScale(block.gamma, prefix: prefix, from: weights)
    }

    private func loadDecoderWeights(from weights: [String: MLXArray]) {
        fishApplyWeightNormConv1d(decoder.inputConv.conv, prefix: "decoder.model.0.conv", from: weights)
        for (index, block) in decoder.blocks.enumerated() {
            let slot = index + 1
            let prefix = "decoder.model.\(slot).block"
            fishApplySnake(block.snake, prefix: "\(prefix).0", from: weights)
            fishApplyWeightNormConvTransposed1d(
                block.convTranspose.conv,
                prefix: "\(prefix).1.conv",
                from: weights)
            loadResidualUnit(block.residual1, prefix: "\(prefix).2.block", from: weights)
            loadResidualUnit(block.residual2, prefix: "\(prefix).3.block", from: weights)
            loadResidualUnit(block.residual3, prefix: "\(prefix).4.block", from: weights)
        }
        fishApplySnake(decoder.finalSnake, prefix: "decoder.model.5", from: weights)
        fishApplyWeightNormConv1d(decoder.finalConv.conv, prefix: "decoder.model.6.conv", from: weights)
    }

    private func loadResidualUnit(
        _ residual: FishCodecResidualUnit,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        fishApplySnake(residual.snake1, prefix: "\(prefix).0", from: weights)
        fishApplyWeightNormConv1d(residual.conv1.conv, prefix: "\(prefix).1.conv", from: weights)
        fishApplySnake(residual.snake2, prefix: "\(prefix).2", from: weights)
        fishApplyWeightNormConv1d(residual.conv2.conv, prefix: "\(prefix).3.conv", from: weights)
    }
}
