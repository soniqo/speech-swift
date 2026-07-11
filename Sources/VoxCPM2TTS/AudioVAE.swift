import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MLXCommon

// MARK: - Snake

public final class Snake1d: Module {
    public var alpha: MLXArray

    public init(channels: Int) {
        self.alpha = MLXArray.ones([1, 1, channels])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x + (1.0 / (alpha + 1e-9)) * (sin(alpha * x) * sin(alpha * x))
    }

    public func loadAlpha(_ alpha: MLXArray) {
        self.alpha = alpha
    }
}

public final class TableEmbedding: Module {
    public var weight: MLXArray
    private let dimensions: Int
    private let storageShape: [Int]

    public init(embeddingCount: Int, dimensions: Int) {
        let scale = sqrt(1.0 / Float(max(1, dimensions)))
        self.dimensions = dimensions
        if dimensions == 64 {
            self.storageShape = [embeddingCount, 8, 8]
        } else {
            self.storageShape = [embeddingCount, dimensions]
        }
        self.weight = MLXRandom.normal(self.storageShape, scale: scale)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gathered = weight[x]
        if dimensions == 64 {
            return gathered.reshaped([gathered.dim(0), dimensions])
        }
        return gathered
    }

    public func loadWeight(_ weight: MLXArray) {
        if dimensions == 64 {
            self.weight = weight.reshaped(storageShape)
        } else {
            self.weight = weight
        }
    }
}

// MARK: - Causal Conv Wrappers

public final class CausalConv1d: Module {
    public var weight: MLXArray
    public var bias: MLXArray?
    public let stride: Int
    public let dilation: Int
    public let groups: Int
    private let padVal: Int
    private let outputPadding: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padVal = padding
        self.outputPadding = outputPadding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        let scale = Float(1.0 / Double(max(1, inputChannels * kernelSize)))
        self.weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outputChannels, kernelSize, max(1, inputChannels / groups)]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if padVal > 0 {
            let pad = max(0, padVal * 2 - outputPadding)
            let zeros = MLXArray.zeros([x.dim(0), pad, x.dim(2)], dtype: x.dtype)
            h = concatenated([zeros, h], axis: 1)
        }
        var y = conv1d(h, weight, stride: stride, padding: 0, dilation: dilation, groups: groups)
        if let bias {
            y = y + bias
        }
        return y
    }
}

public final class CausalTransposeConv1d: Module {
    public var weight: MLXArray
    public var bias: MLXArray?
    public let stride: Int
    public let groups: Int
    private let padVal: Int
    private let outputPadding: Int

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        outputPadding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.padVal = padding
        self.outputPadding = outputPadding
        self.stride = stride
        self.groups = groups

        let scale = Float(1.0 / Double(max(1, inputChannels * kernelSize)))
        self.weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [outputChannels, kernelSize, max(1, inputChannels / groups)]
        )
        self.bias = bias ? MLXArray.zeros([outputChannels]) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed1d(x, weight, stride: stride, padding: 0, groups: groups)
        if let bias {
            y = y + bias
        }
        let trim = padVal * 2 - outputPadding
        if trim > 0 {
            y = y[0..., 0..<(y.dim(1) - trim), 0...]
        }
        return y
    }
}

public final class ConvStack1d: Module {
    @ModuleInfo public var layers: [CausalConv1d]

    public init(layers: [CausalConv1d]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

public final class CausalEncoderBlockStack: Module {
    @ModuleInfo public var layers: [CausalEncoderBlock]

    public init(layers: [CausalEncoderBlock]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

public final class CausalDecoderBlockStack: Module {
    @ModuleInfo public var layers: [CausalDecoderBlock]

    public init(layers: [CausalDecoderBlock]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

public final class SRConditionWeights {
    public var scale: [MLXArray]
    public var bias: [MLXArray]

    public init(scale: [MLXArray], bias: [MLXArray]) {
        self.scale = scale
        self.bias = bias
    }
}

public final class SampleRateConditionLayerStack: Module {
    @ModuleInfo public var layers: [SampleRateConditionLayer]

    public init(layers: [SampleRateConditionLayer]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, srCond: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, srCond: srCond)
        }
        return h
    }
}

// MARK: - Encoder / Decoder Residual Units

public final class CausalResidualUnit: Module {
    @ModuleInfo var snake1: Snake1d
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var snake2: Snake1d
    @ModuleInfo var conv2: CausalConv1d

    public init(dim: Int = 16, dilation: Int = 1, kernel: Int = 7, groups: Int = 1) {
        let pad = ((kernel - 1) * dilation) / 2
        self._snake1 = ModuleInfo(wrappedValue: Snake1d(channels: dim))
        self._conv1 = ModuleInfo(wrappedValue: CausalConv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: kernel,
            dilation: dilation,
            padding: pad,
            groups: groups
        ))
        self._snake2 = ModuleInfo(wrappedValue: Snake1d(channels: dim))
        self._conv2 = ModuleInfo(wrappedValue: CausalConv1d(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 1
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return residual + h
    }
}

public final class CausalEncoderBlock: Module {
    @ModuleInfo var res1: CausalResidualUnit
    @ModuleInfo var res2: CausalResidualUnit
    @ModuleInfo var res3: CausalResidualUnit
    @ModuleInfo var snake: Snake1d
    @ModuleInfo var conv: CausalConv1d

    public init(outputDim: Int = 16, inputDim: Int? = nil, stride: Int = 1, groups: Int = 1) {
        let input = inputDim ?? outputDim / 2
        self._res1 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: input, dilation: 1, groups: groups))
        self._res2 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: input, dilation: 3, groups: groups))
        self._res3 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: input, dilation: 9, groups: groups))
        self._snake = ModuleInfo(wrappedValue: Snake1d(channels: input))
        self._conv = ModuleInfo(wrappedValue: CausalConv1d(
            inputChannels: input,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2.0)),
            outputPadding: stride % 2
        ))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = res1(x)
        h = res2(h)
        h = res3(h)
        h = snake(h)
        return conv(h)
    }
}

public final class CausalDecoderBlock: Module {
    public let inputChannels: Int
    @ModuleInfo var snake: Snake1d
    @ModuleInfo var conv_t: CausalTransposeConv1d
    @ModuleInfo var noise: NoiseBlock?
    @ModuleInfo var res1: CausalResidualUnit
    @ModuleInfo var res2: CausalResidualUnit
    @ModuleInfo var res3: CausalResidualUnit

    public init(
        inputDim: Int = 16,
        outputDim: Int = 8,
        stride: Int = 1,
        groups: Int = 1,
        useNoiseBlock: Bool = false
    ) {
        self.inputChannels = inputDim
        self._snake = ModuleInfo(wrappedValue: Snake1d(channels: inputDim))
        self._conv_t = ModuleInfo(wrappedValue: CausalTransposeConv1d(
            inputChannels: inputDim,
            outputChannels: outputDim,
            kernelSize: 2 * stride,
            stride: stride,
            padding: Int(ceil(Double(stride) / 2.0)),
            outputPadding: stride % 2
        ))
        self._noise = ModuleInfo(wrappedValue: useNoiseBlock ? NoiseBlock(dim: outputDim) : nil)
        self._res1 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: outputDim, dilation: 1, groups: groups))
        self._res2 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: outputDim, dilation: 3, groups: groups))
        self._res3 = ModuleInfo(wrappedValue: CausalResidualUnit(dim: outputDim, dilation: 9, groups: groups))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake(x)
        h = conv_t(h)
        if let noise { h = noise(h) }
        h = res1(h)
        h = res2(h)
        h = res3(h)
        return h
    }
}

// MARK: - Conditioning Helpers

public final class SampleRateConditionLayer: Module {
    public let condType: String
    @ModuleInfo var scale_embed: TableEmbedding
    @ModuleInfo var bias_embed: TableEmbedding
    private let srBoundaries: [Int]?

    public init(
        inputDim: Int,
        srBinBuckets: Int,
        condType: String = "scale_bias",
        condDim: Int = 128,
        outLayer: Bool = false,
        srBoundaries: [Int]? = nil
    ) {
        self.condType = condType
        self.srBoundaries = srBoundaries

        self._scale_embed = ModuleInfo(
            wrappedValue: TableEmbedding(embeddingCount: srBinBuckets, dimensions: inputDim)
        )
        self._bias_embed = ModuleInfo(
            wrappedValue: TableEmbedding(embeddingCount: srBinBuckets, dimensions: inputDim)
        )

        if condType != "scale_bias" && condType != "scale_bias_init" {
            fatalError("Invalid cond_type: \(condType)")
        }
        super.init()
    }

    public func getSrIdx(_ sr: MLXArray) -> Int {
        guard let srBoundaries else {
            return 0
        }
        let value = Int(sr.item(Int32.self))
        return srBoundaries.reduce(0) { partial, boundary in
            partial + (value >= boundary ? 1 : 0)
        }
    }

    public func callAsFunction(_ x: MLXArray, srCond: MLXArray) -> MLXArray {
        var h = x
        let srIdx = MLXArray([Int32(getSrIdx(srCond))])
        let scale = scale_embed(srIdx).expandedDimensions(axis: 1)
        let bias = bias_embed(srIdx).expandedDimensions(axis: 1)
        h = h * scale + bias
        return h
    }

    public func loadScaleWeight(_ weight: MLXArray) {
        self._scale_embed.wrappedValue.loadWeight(weight)
    }

    public func loadBiasWeight(_ weight: MLXArray) {
        self._bias_embed.wrappedValue.loadWeight(weight)
    }
}

public final class NoiseBlock: Module {
    @ModuleInfo var linear: CausalConv1d
    private let noiseStd: Float

    public init(dim: Int, noiseStd: Float = 0.003) {
        self.noiseStd = noiseStd
        self._linear = ModuleInfo(wrappedValue: CausalConv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: false))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let noise = MLXRandom.normal([x.dim(0), x.dim(1), 1], dtype: x.dtype) * MLXArray(noiseStd)
        return x + noise * linear(x)
    }
}

// MARK: - Encoder / Decoder

public final class CausalEncoder: Module {
    @ModuleInfo var conv_in: CausalConv1d
    @ModuleInfo var blocks: CausalEncoderBlockStack
    @ModuleInfo var fc_mu: CausalConv1d

    public init(
        dModel: Int = 64,
        latentDim: Int = 32,
        strides: [Int] = [2, 5, 8, 8],
        depthwise: Bool = false
    ) {
        self._conv_in = ModuleInfo(wrappedValue: CausalConv1d(inputChannels: 1, outputChannels: dModel, kernelSize: 7, padding: 3))
        var outDim = dModel
        var blocks: [CausalEncoderBlock] = []
        for stride in strides {
            let nextDim = outDim * 2
            let groups = depthwise ? (nextDim / 2) : 1
            blocks.append(CausalEncoderBlock(outputDim: nextDim, inputDim: outDim, stride: stride, groups: groups))
            outDim = nextDim
        }
        self._blocks = ModuleInfo(wrappedValue: CausalEncoderBlockStack(layers: blocks))
        self._fc_mu = ModuleInfo(wrappedValue: CausalConv1d(inputChannels: outDim, outputChannels: latentDim, kernelSize: 3, padding: 1))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv_in(x)
        h = blocks(h)
        return fc_mu(h)
    }
}

public final class CausalDecoder: Module {
    @ModuleInfo var conv_in: ConvStack1d
    @ModuleInfo var blocks: CausalDecoderBlockStack
    @ModuleInfo var srCondLayers: SampleRateConditionLayerStack
    @ModuleInfo var snake_out: Snake1d
    @ModuleInfo var conv_out: CausalConv1d
    public let srBoundaries: [Int]?

    public init(
        inputChannel: Int,
        channels: Int,
        rates: [Int],
        depthwise: Bool = false,
        dOut: Int = 1,
        useNoiseBlock: Bool = false,
        srBinBoundaries: [Int]? = nil,
        condType: String = "scale_bias",
        condDim: Int = 128,
        condOutLayer: Bool = false,
        srCondLayers: [SampleRateConditionLayer] = []
    ) {
        self.srBoundaries = srBinBoundaries
        let convInLayers: [CausalConv1d] = [
            CausalConv1d(
                inputChannels: inputChannel,
                outputChannels: inputChannel,
                kernelSize: 7,
                padding: 3,
                groups: depthwise ? inputChannel : 1
            ),
            CausalConv1d(
                inputChannels: inputChannel,
                outputChannels: channels,
                kernelSize: 1
            )
        ]
        self._conv_in = ModuleInfo(wrappedValue: ConvStack1d(layers: convInLayers))

        var blocks: [CausalDecoderBlock] = []
        for (idx, stride) in rates.enumerated() {
            let inputDim = channels / (1 << idx)
            let outputDim = channels / (1 << (idx + 1))
            let groups = depthwise ? outputDim : 1
            blocks.append(CausalDecoderBlock(
                inputDim: inputDim,
                outputDim: outputDim,
                stride: stride,
                groups: groups,
                useNoiseBlock: useNoiseBlock
            ))
        }
        self._blocks = ModuleInfo(wrappedValue: CausalDecoderBlockStack(layers: blocks))
        self._srCondLayers = ModuleInfo(wrappedValue: SampleRateConditionLayerStack(layers: srCondLayers))
        self._snake_out = ModuleInfo(wrappedValue: Snake1d(channels: channels / (1 << rates.count)))
        self._conv_out = ModuleInfo(wrappedValue: CausalConv1d(
            inputChannels: channels / (1 << rates.count),
            outputChannels: dOut,
            kernelSize: 7,
            padding: 3
        ))

        super.init()
    }

    public func getSrIdx(_ sr: MLXArray) -> MLXArray {
        guard let srBoundaries else {
            return MLXArray([Int32(0)])
        }
        let boundaries = MLXArray(srBoundaries.map(Int32.init))
        let idx = sum(sr .>= boundaries).asType(.int32)
        return idx.reshaped([1])
    }

    public func callAsFunction(_ x: MLXArray, srCond: MLXArray? = nil) -> MLXArray {
        var h = x
        h = conv_in(h)

        let effectiveSrCond = srCond ?? MLXArray([Int32(48_000)])
        if !srCondLayers.layers.isEmpty {
            let count = min(blocks.layers.count, srCondLayers.layers.count)
            for index in 0..<count {
                h = srCondLayers.layers[index](h, srCond: effectiveSrCond)
                h = blocks.layers[index](h)
            }
            if count < blocks.layers.count {
                for index in count..<blocks.layers.count {
                    h = blocks.layers[index](h)
                }
            }
        } else {
            for block in blocks.layers {
                h = block(h)
            }
        }

        h = snake_out(h)
        h = conv_out(h)
        return tanh(h)
    }
}

// MARK: - Audio VAE

public final class AudioVAE: Module {
    public let config: AudioVAEConfig
    public let hopLength: Int
    public let chunkSize: Int
    public let decodeChunkSize: Int
    public let latentDim: Int
    public let sampleRate: Int
    public let outSampleRate: Int

    @ModuleInfo public var encoder: CausalEncoder
    @ModuleInfo public var decoder: CausalDecoder

    public init(_ config: AudioVAEConfig) {
        self.config = config
        self.hopLength = config.encoderRates.reduce(1, *)
        self.chunkSize = config.encoderRates.reduce(1, *)
        self.decodeChunkSize = config.decoderRates.reduce(1, *)
        self.latentDim = config.latentDim
        self.sampleRate = config.sampleRate
        self.outSampleRate = config.outSampleRate
        self._encoder = ModuleInfo(wrappedValue: CausalEncoder(
            dModel: config.encoderDim,
            latentDim: config.latentDim,
            strides: config.encoderRates,
            depthwise: config.depthwise
        ))
        self._decoder = ModuleInfo(wrappedValue: CausalDecoder(
            inputChannel: config.latentDim,
            channels: config.decoderDim,
            rates: config.decoderRates,
            depthwise: config.depthwise,
            dOut: 1,
            useNoiseBlock: config.useNoiseBlock,
            srBinBoundaries: config.srBinBoundaries,
            condType: config.condType,
            condDim: config.condDim,
            condOutLayer: config.condOutLayer,
            srCondLayers: config.decoderRates.enumerated().map { index, _ in
                let inputDim = config.decoderDim / (1 << index)
                return SampleRateConditionLayer(
                    inputDim: inputDim,
                    srBinBuckets: max(1, config.srBinBoundaries.count + 1),
                    condType: config.condType,
                    condDim: config.condDim,
                    outLayer: config.condOutLayer && index == config.decoderRates.count - 1,
                    srBoundaries: config.srBinBoundaries
                )
            }
        ))
        super.init()
    }

    public func encode(_ x: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        var h = x
        if h.ndim == 2 {
            h = h.expandedDimensions(axis: 2)
        }
        if h.shape[1] < h.shape[2] {
            h = h.transposed(0, 2, 1)
        }
        h = preprocess(h, sampleRate: sampleRate)
        return encoder(h)
    }

    public func decode(_ z: MLXArray, srCond: MLXArray? = nil) -> MLXArray {
        let cond = srCond ?? MLXArray([Int32(outSampleRate)])
        let out = decoder(z, srCond: cond)
        return out.squeezed(axis: 2)
    }

    public func preprocess(_ audioData: MLXArray, sampleRate: Int? = nil) -> MLXArray {
        let effectiveSampleRate = sampleRate ?? self.sampleRate
        precondition(effectiveSampleRate == self.sampleRate, "AudioVAE expects \(self.sampleRate) Hz input")
        let length = audioData.dim(1)
        let rightPad = Int(ceil(Double(length) / Double(hopLength))) * hopLength - length
        guard rightPad > 0 else { return audioData }
        let pad = MLXArray.zeros([audioData.dim(0), rightPad, audioData.dim(2)], dtype: audioData.dtype)
        return concatenated([audioData, pad], axis: 1)
    }

    public func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        // Minimal fusion for weight-norm converted checkpoints.
        var fused: [String: MLXArray] = [:]
        let keys = Array(weights.keys).sorted()
        var processed = Set<String>()

        for key in keys {
            if processed.contains(key) || key.contains("fc_logvar") {
                continue
            }
            if key.hasSuffix(".weight_g") {
                let base = String(key.dropLast(".weight_g".count))
                let vKey = base + ".weight_v"
                if let g = weights[key], let v = weights[vKey] {
                    let vFlat = v.reshaped([v.shape[0], -1])
                    let norm = sqrt((vFlat * vFlat).sum(axis: 1)).reshaped(g.shape)
                    let w = g * (v / (norm + 1e-9))
                    fused[base + ".weight"] = w
                    processed.insert(key)
                    processed.insert(vKey)
                    continue
                }
            }
            if key.hasSuffix(".weight_v") {
                continue
            }
            fused[key] = weights[key]
        }

        var remapped: [String: MLXArray] = [:]
        for (key, value) in fused {
            if key.hasPrefix("audio_vae.") {
                remapped[key] = value
            } else if key.hasPrefix("encoder.") || key.hasPrefix("decoder.") {
                remapped["audio_vae.\(key)"] = value
            } else {
                remapped[key] = value
            }
        }
        return remapped
    }

    /// Promote all AudioVAE parameters to float32 after loading.
    /// This mirrors the repo's existing traversal pattern used by `clearParameters()`
    /// but preserves the loaded values.
    public func castParametersToFloat32() {
        apply(filter: Self.filterAll) { array in
            array.asType(.float32)
        }
    }
}
