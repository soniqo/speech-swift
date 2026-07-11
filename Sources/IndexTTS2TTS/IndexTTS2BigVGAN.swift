import Foundation
import MLX
import MLXNN

final class IndexTTS2SeqLayers: Module {
    @ModuleInfo var layers: [Module]

    init(_ layers: [Module]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }
}

final class IndexTTS2SnakeBeta: Module {
    @ParameterInfo var alpha: MLXArray
    @ParameterInfo var beta: MLXArray

    init(channels: Int) {
        self._alpha = ParameterInfo(wrappedValue: MLXArray.zeros([channels]))
        self._beta = ParameterInfo(wrappedValue: MLXArray.zeros([channels]))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha)
        let b = exp(beta)
        let s = sin(x * a)
        return x + (Float(1.0) / (b + MLXArray(Float(1e-9)))) * (s * s)
    }
}

final class IndexTTS2BigVGANUpsampleFilter: Module {
    @ParameterInfo var filter: MLXArray

    override init() {
        self._filter = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, 12]))
        super.init()
    }
}

final class IndexTTS2BigVGANLowpassFilter: Module {
    @ParameterInfo var filter: MLXArray

    override init() {
        self._filter = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, 12]))
        super.init()
    }
}

final class IndexTTS2BigVGANDownsampleFilter: Module {
    @ModuleInfo var lowpass: IndexTTS2BigVGANLowpassFilter

    override init() {
        self._lowpass = ModuleInfo(wrappedValue: IndexTTS2BigVGANLowpassFilter())
        super.init()
    }
}

final class IndexTTS2BigVGANActivation1d: Module {
    @ModuleInfo var act: IndexTTS2SnakeBeta
    @ModuleInfo var upsample: IndexTTS2BigVGANUpsampleFilter
    @ModuleInfo var downsample: IndexTTS2BigVGANDownsampleFilter

    init(channels: Int) {
        self._act = ModuleInfo(wrappedValue: IndexTTS2SnakeBeta(channels: channels))
        self._upsample = ModuleInfo(wrappedValue: IndexTTS2BigVGANUpsampleFilter())
        self._downsample = ModuleInfo(wrappedValue: IndexTTS2BigVGANDownsampleFilter())
        super.init()
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = IndexTTS2BigVGAN.firUpsample(xIn, filt: upsample.filter, factor: 2)
        x = act(x)
        x = IndexTTS2BigVGAN.firDownsample(x, filt: downsample.lowpass.filter, factor: 2)
        return x
    }
}

final class IndexTTS2BigVGANAMPBlock1: Module {
    @ModuleInfo var convs1: [Conv1d]
    @ModuleInfo var convs2: [Conv1d]
    @ModuleInfo var activations: [IndexTTS2BigVGANActivation1d]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        func padding(_ kernel: Int, _ dilation: Int) -> Int {
            (kernel * dilation - dilation) / 2
        }

        var convs1: [Conv1d] = []
        var convs2: [Conv1d] = []
        var activations: [IndexTTS2BigVGANActivation1d] = []
        for dilation in dilations {
            convs1.append(Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding(kernelSize, dilation),
                dilation: dilation,
                bias: true))
            convs2.append(Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: kernelSize,
                stride: 1,
                padding: padding(kernelSize, 1),
                dilation: 1,
                bias: true))
            activations.append(IndexTTS2BigVGANActivation1d(channels: channels))
            activations.append(IndexTTS2BigVGANActivation1d(channels: channels))
        }

        self._convs1 = ModuleInfo(wrappedValue: convs1)
        self._convs2 = ModuleInfo(wrappedValue: convs2)
        self._activations = ModuleInfo(wrappedValue: activations)
        super.init()
    }

    func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        for index in 0..<convs1.count {
            var h = activations[index * 2](x)
            h = convs1[index](h)
            h = activations[index * 2 + 1](h)
            h = convs2[index](h)
            x = x + h
        }
        return x
    }
}

final class IndexTTS2BigVGAN: Module {
    struct Config {
        let numMels = 80
        let upsampleInitialChannel = 1536
        let upsampleRates = [4, 4, 2, 2, 2, 2]
        let upsampleKernelSizes = [8, 8, 4, 4, 4, 4]
        let resblockKernelSizes = [3, 7, 11]
        let resblockDilationSizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    }

    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo var ups: [IndexTTS2SeqLayers]
    @ModuleInfo var resblocks: [IndexTTS2BigVGANAMPBlock1]
    @ModuleInfo(key: "activation_post") var activationPost: IndexTTS2BigVGANActivation1d
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let config: Config
    let numKernels: Int
    let numUpsamples: Int

    init(config: Config = Config()) {
        self.config = config
        self.numKernels = config.resblockKernelSizes.count
        self.numUpsamples = config.upsampleRates.count

        self._convPre = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: config.numMels,
            outputChannels: config.upsampleInitialChannel,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            bias: true), key: "conv_pre")

        var upsamplers: [IndexTTS2SeqLayers] = []
        for index in 0..<config.upsampleRates.count {
            let inputChannels = config.upsampleInitialChannel / (1 << index)
            let outputChannels = config.upsampleInitialChannel / (1 << (index + 1))
            let rate = config.upsampleRates[index]
            let kernel = config.upsampleKernelSizes[index]
            upsamplers.append(IndexTTS2SeqLayers([
                ConvTransposed1d(
                    inputChannels: inputChannels,
                    outputChannels: outputChannels,
                    kernelSize: kernel,
                    stride: rate,
                    padding: (kernel - rate) / 2,
                    bias: true),
            ]))
        }
        self._ups = ModuleInfo(wrappedValue: upsamplers)

        var blocks: [IndexTTS2BigVGANAMPBlock1] = []
        for upsampleIndex in 0..<numUpsamples {
            let channels = config.upsampleInitialChannel / (1 << (upsampleIndex + 1))
            for (kernelIndex, kernelSize) in config.resblockKernelSizes.enumerated() {
                blocks.append(IndexTTS2BigVGANAMPBlock1(
                    channels: channels,
                    kernelSize: kernelSize,
                    dilations: config.resblockDilationSizes[kernelIndex]))
            }
        }
        self._resblocks = ModuleInfo(wrappedValue: blocks)

        let finalChannels = config.upsampleInitialChannel / (1 << numUpsamples)
        self._activationPost = ModuleInfo(
            wrappedValue: IndexTTS2BigVGANActivation1d(channels: finalChannels),
            key: "activation_post")
        self._convPost = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: finalChannels,
            outputChannels: 1,
            kernelSize: 7,
            stride: 1,
            padding: 3,
            bias: false), key: "conv_post")
        super.init()
    }

    func loadWeights(_ weights: [String: MLXArray]) throws {
        let prepared = Self.prepareWeights(weights)
        try update(parameters: ModuleParameters.unflattened(prepared), verify: .all)
    }

    /// Expects mel as `(B, T, 80)` and returns waveform `(B, T_out)`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for upsampleIndex in 0..<numUpsamples {
            let up = ups[upsampleIndex].layers[0] as! ConvTransposed1d
            x = up(x)

            var accumulator: MLXArray?
            for kernelIndex in 0..<numKernels {
                let block = resblocks[upsampleIndex * numKernels + kernelIndex]
                let h = block(x)
                accumulator = accumulator == nil ? h : accumulator! + h
            }
            x = accumulator! / Float(numKernels)
        }

        x = activationPost(x)
        x = convPost(x)
        x = maximum(minimum(x, MLXArray(Float(1))), MLXArray(Float(-1)))
        return x.squeezed(axis: -1)
    }

    private static func prepareWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var consumed = Set<String>()

        for key in weights.keys.sorted() {
            if consumed.contains(key) { continue }
            let stripped = key.hasPrefix("generator.")
                ? String(key.dropFirst("generator.".count))
                : key

            if stripped.hasSuffix(".weight_g") {
                let base = String(stripped.dropLast(".weight_g".count))
                let sourceBase = key.hasPrefix("generator.")
                    ? "generator.\(base)"
                    : base
                let gKey = "\(sourceBase).weight_g"
                let vKey = "\(sourceBase).weight_v"
                if let g = weights[gKey], let v = weights[vKey] {
                    let fused = fuseWeightNormDim0(g: g, v: v)
                    out[remapSequentialKey("\(base).weight")] = convertConvWeight(fused, base: base)
                    consumed.insert(gKey)
                    consumed.insert(vKey)
                    continue
                }
            }

            if stripped.hasSuffix(".weight_v") {
                consumed.insert(key)
                continue
            }

            guard let value = weights[key] else { continue }
            out[remapSequentialKey(stripped)] = value.asType(.float32)
            consumed.insert(key)
        }

        return out
    }

    private static func convertConvWeight(_ weight: MLXArray, base: String) -> MLXArray {
        let fp32 = weight.asType(.float32)
        if base.hasPrefix("ups.") {
            return fp32.transposed(1, 2, 0)
        }
        return fp32.transposed(0, 2, 1)
    }

    private static func fuseWeightNormDim0(g: MLXArray, v: MLXArray) -> MLXArray {
        let v32 = v.asType(.float32)
        let g32 = g.asType(.float32)
        let flat = v32.reshaped([v32.shape[0], -1])
        let norm = sqrt((flat * flat).sum(axis: 1)).reshaped(g32.shape)
        return g32 * (v32 / (norm + MLXArray(Float(1e-9))))
    }

    private static func remapSequentialKey(_ key: String) -> String {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard parts.count >= 3,
              parts[0] == "ups",
              Int(parts[1]) != nil,
              Int(parts[2]) != nil
        else {
            return key
        }
        return ([parts[0], parts[1], "layers"] + parts[2...]).joined(separator: ".")
    }

    static func firUpsample(_ x: MLXArray, filt: MLXArray, factor: Int) -> MLXArray {
        let kernel = filt.dim(-1)
        let pad = kernel / factor - 1
        let padLeft = pad * factor + (kernel - factor) / 2
        let padRight = pad * factor + (kernel - factor + 1) / 2

        let paddedInput = replicatePad1D(x, left: pad, right: pad)
        let batch = paddedInput.dim(0)
        let time = paddedInput.dim(1)
        let channels = paddedInput.dim(2)

        var stacks: [MLXArray] = [paddedInput]
        for _ in 1..<factor {
            stacks.append(MLXArray.zeros([batch, time, channels], dtype: paddedInput.dtype))
        }
        let stretched = stacked(stacks, axis: 2).reshaped([batch, time * factor, channels])
        let trimmed = stretched[0..., 0..<((time - 1) * factor + 1), 0...]
        let convolutionInput = padded(
            trimmed,
            widths: [.init((0, 0)), .init((kernel - 1, kernel - 1)), .init((0, 0))],
            value: MLXArray(Float(0)))

        let out = firFilter(convolutionInput, filter: filt * Float(factor))
        return out[0..., padLeft..<(out.dim(1) - padRight), 0...]
    }

    /// Applies one FIR filter identically to every channel as a dense
    /// `conv1d` with channels folded into the batch dimension. Equivalent to
    /// materializing sliding windows and reducing, without the `[B, T, K, C]`
    /// intermediate that dominated vocoder time.
    private static func firFilter(_ x: MLXArray, filter: MLXArray, stride: Int = 1) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let channels = x.dim(2)
        let flat = x.transposed(0, 2, 1).reshaped([batch * channels, time, 1])
        let weight = filter.asType(x.dtype).reshaped([1, filter.size, 1])
        let filtered = conv1d(flat, weight, stride: stride)
        return filtered
            .reshaped([batch, channels, filtered.dim(1)])
            .transposed(0, 2, 1)
    }

    static func firDownsample(_ x: MLXArray, filt: MLXArray, factor: Int) -> MLXArray {
        let kernel = filt.dim(-1)
        let even = kernel % 2 == 0
        let padLeft = kernel / 2 - (even ? 1 : 0)
        let padRight = kernel / 2
        let paddedInput = replicatePad1D(x, left: padLeft, right: padRight)
        return firFilter(paddedInput, filter: filt, stride: factor)
    }

    private static func replicatePad1D(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        if left == 0 && right == 0 { return x }
        let batch = x.dim(0)
        let channels = x.dim(2)
        var parts: [MLXArray] = []
        if left > 0 {
            let edge = x[0..., 0..<1, 0...]
            parts.append(broadcast(edge, to: [batch, left, channels]))
        }
        parts.append(x)
        if right > 0 {
            let edge = x[0..., (x.dim(1) - 1)..<x.dim(1), 0...]
            parts.append(broadcast(edge, to: [batch, right, channels]))
        }
        return concatenated(parts, axis: 1)
    }
}
