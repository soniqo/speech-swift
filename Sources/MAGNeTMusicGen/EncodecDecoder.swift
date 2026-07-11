import Foundation
import MLX
import MLXNN
import MLXCommon

// MARK: - Reflect padding helper (NLC layout)

/// Reflect-pad an `[N, L, C]` tensor along the L axis.
///
/// Mirrors `np.pad(..., mode='reflect')` which excludes the boundary sample
/// itself (Python: `prefix = x[:, 1:padL+1][:, ::-1]`).
private func reflectPad1d(_ x: MLXArray, leading: Int, trailing: Int) -> MLXArray {
    if leading == 0 && trailing == 0 { return x }
    let L = x.dim(1)
    var parts: [MLXArray] = []
    if leading > 0 {
        // x[:, 1 : leading+1][:, ::-1]
        let pre = x[0..., 1..<(leading + 1), 0...]
        parts.append(pre[0..., .stride(by: -1), 0...])
    }
    parts.append(x)
    if trailing > 0 {
        // x[:, max(L - (trailing+1), 0) : L - 1][:, ::-1]
        let start = max(L - (trailing + 1), 0)
        let suf = x[0..., start..<(L - 1), 0...]
        parts.append(suf[0..., .stride(by: -1), 0...])
    }
    return concatenated(parts, axis: 1)
}

// EnCodec LSTM modules live in MLXCommon (shared between Encodec-style codecs).

// MARK: - EnCodec Conv1d (with reflect padding)

/// `EncodecConv1d` mirrors the Python module: applies symmetric reflect
/// padding (non-causal, our config has `useCausalConv=false`) so the output
/// length is exactly `length / stride`.
public final class EncodecConv1d: Module {
    @ModuleInfo public var conv: Conv1d
    public let stride: Int
    public let effKernel: Int
    public let padTotal: Int
    public let padMode: String

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int = 1, dilation: Int = 1, padMode: String = "reflect") {
        self._conv = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, dilation: dilation, bias: true))
        self.stride = stride
        self.effKernel = (kernelSize - 1) * dilation + 1
        self.padTotal = kernelSize - stride
        self.padMode = padMode
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Extra padding so the output length lines up exactly. Mirror of the
        // Python `_get_extra_padding_for_conv1d`:
        //   n_frames    = (length - effK + padTotal) / stride + 1
        //   ideal_len   = (ceil(n_frames) - 1) * stride + effK - padTotal
        //   extra       = max(0, ideal_len - length)
        let length = x.dim(1)
        let nFrames = Double(length - effKernel + padTotal) / Double(stride) + 1.0
        let idealLength = (Int(ceil(nFrames)) - 1) * stride + effKernel - padTotal
        let extra = max(0, idealLength - length)
        let padRight = padTotal / 2
        let padLeft = padTotal - padRight
        let padded: MLXArray
        if padMode == "reflect" {
            padded = reflectPad1d(x, leading: padLeft, trailing: padRight + extra)
        } else {
            padded = MLX.padded(x, widths: [.init((0,0)), .init((padLeft, padRight + extra)), .init((0,0))], value: MLXArray(Float(0)))
        }
        return conv(padded)
    }
}

// MARK: - EnCodec ConvTranspose1d (with right-trim)

public final class EncodecConvTranspose1d: Module {
    @ModuleInfo public var conv: ConvTransposed1d
    public let padTotal: Int
    public let trimRightRatio: Double
    public let causal: Bool

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int, trimRightRatio: Double = 1.0, causal: Bool = false) {
        self._conv = ModuleInfo(wrappedValue: ConvTransposed1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, bias: true))
        self.padTotal = kernelSize - stride
        self.trimRightRatio = trimRightRatio
        self.causal = causal
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x)
        let padRight: Int
        if causal {
            padRight = Int(ceil(Double(padTotal) * trimRightRatio))
        } else {
            padRight = padTotal / 2
        }
        let padLeft = padTotal - padRight
        let end = y.dim(1) - padRight
        return y[0..., padLeft..<end, 0...]
    }
}

// MARK: - EnCodec ResnetBlock

/// `block` list = [ELU, Conv1d (dilation `d`), ELU, Conv1d (dilation 1)].
/// Indices 0/2 are ELU (no params), 1/3 are Conv1d. Shortcut is Identity
/// because our config has `useConvShortcut=false`.
public final class EncodecResnetBlock: Module {
    @ModuleInfo public var block: [Module]

    public init(dim: Int, dilations: [Int], compress: Int, kernelSize: Int) {
        let hidden = dim / compress
        let kernelSizes = [kernelSize, 1]
        precondition(dilations.count == kernelSizes.count)
        var modules: [Module] = []
        for i in 0..<kernelSizes.count {
            let inCh = (i == 0) ? dim : hidden
            let outCh = (i == kernelSizes.count - 1) ? dim : hidden
            modules.append(ELUNoParam())
            modules.append(EncodecConv1d(
                inChannels: inCh, outChannels: outCh,
                kernelSize: kernelSizes[i], dilation: dilations[i]))
        }
        self._block = ModuleInfo(wrappedValue: modules)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let elu = layer as? ELUNoParam {
                h = elu(h)
            } else if let c = layer as? EncodecConv1d {
                h = c(h)
            }
        }
        return x + h
    }
}

// MARK: - ELU module (no params, lets us keep it in a `block` ModuleList)

public final class ELUNoParam: Module {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLX.where(x .> MLXArray(Float(0)), x, exp(x) - MLXArray(Float(1)))
    }
}

// MARK: - EnCodec decoder

/// Builds the SEANet decoder layer list to match the mlx-community EnCodec
/// safetensors `decoder.layers.*` indices:
///
/// 0: Conv (hidden_size → scaling*num_filters, kernel=7)
/// 1: LSTM block
/// 2: ELU                         3: ConvTranspose (×8 upsample)
/// 4: ResnetBlock                 5: ELU
/// 6: ConvTranspose (×5)          7: ResnetBlock
/// 8: ELU                         9: ConvTranspose (×4)
/// 10: ResnetBlock                11: ELU
/// 12: ConvTranspose (×4)         13: ResnetBlock
/// 14: ELU                        15: Conv (num_filters → audio_channels, kernel=7)
public final class EncodecDecoder: Module {
    @ModuleInfo public var layers: [Module]

    public init(config: EncodecModelConfig) {
        var modules: [Module] = []
        var scaling = Int(pow(2.0, Double(config.upsamplingRatios.count)))
        let baseChannels = scaling * config.numFilters
        modules.append(EncodecConv1d(
            inChannels: config.hiddenSize, outChannels: baseChannels,
            kernelSize: config.kernelSize, padMode: config.padMode))
        modules.append(EncodecLSTM(
            dimension: baseChannels, numLayers: config.numLstmLayers))
        for ratio in config.upsamplingRatios {
            let curScale = scaling * config.numFilters
            modules.append(ELUNoParam())
            modules.append(EncodecConvTranspose1d(
                inChannels: curScale, outChannels: curScale / 2,
                kernelSize: ratio * 2, stride: ratio,
                trimRightRatio: config.trimRightRatio, causal: config.useCausalConv))
            for j in 0..<config.numResidualLayers {
                let dil = Int(pow(Double(config.dilationGrowthRate), Double(j)))
                modules.append(EncodecResnetBlock(
                    dim: curScale / 2, dilations: [dil, 1],
                    compress: config.compress, kernelSize: config.residualKernelSize))
            }
            scaling /= 2
        }
        modules.append(ELUNoParam())
        modules.append(EncodecConv1d(
            inChannels: config.numFilters, outChannels: config.audioChannels,
            kernelSize: config.lastKernelSize, padMode: config.padMode))
        self._layers = ModuleInfo(wrappedValue: modules)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let elu = layer as? ELUNoParam { h = elu(h) }
            else if let c = layer as? EncodecConv1d { h = c(h) }
            else if let l = layer as? EncodecLSTM { h = l(h) }
            else if let t = layer as? EncodecConvTranspose1d { h = t(h) }
            else if let r = layer as? EncodecResnetBlock { h = r(h) }
        }
        return h
    }
}

// MARK: - Residual Vector Quantizer (decode-only)

public final class EncodecCodebook: Module {
    @ParameterInfo public var embed: MLXArray

    public init(codebookSize: Int, codebookDim: Int) {
        self._embed = ParameterInfo(wrappedValue: MLXArray.zeros([codebookSize, codebookDim]))
        super.init()
    }

    public func decode(_ ind: MLXArray) -> MLXArray {
        return embed[ind]
    }
}

public final class EncodecVQ: Module {
    @ModuleInfo public var codebook: EncodecCodebook
    public init(codebookSize: Int, codebookDim: Int) {
        self._codebook = ModuleInfo(wrappedValue: EncodecCodebook(
            codebookSize: codebookSize, codebookDim: codebookDim))
        super.init()
    }
    public func decode(_ ind: MLXArray) -> MLXArray { codebook.decode(ind) }
}

public final class EncodecRVQ: Module {
    @ModuleInfo public var layers: [EncodecVQ]

    public init(config: EncodecModelConfig, numQuantizers: Int) {
        self._layers = ModuleInfo(wrappedValue: (0..<numQuantizers).map { _ in
            EncodecVQ(codebookSize: config.codebookSize, codebookDim: config.codebookDim)
        })
        super.init()
    }

    /// `codes: [B, K, T]` int → `[B, T, codebookDim]` summed quantized embedding.
    public func decode(_ codes: MLXArray) -> MLXArray {
        let K = codes.dim(1)
        precondition(K == layers.count, "codes K=\(K) doesn't match RVQ layers \(layers.count)")
        var sum: MLXArray? = nil
        for k in 0..<K {
            let ids = codes[0..., k, 0...]               // [B, T]
            let q = layers[k].decode(ids)                 // [B, T, D]
            sum = (sum == nil) ? q : (sum! + q)
        }
        return sum!
    }
}

// MARK: - EnCodec model (decode-only convenience)

public final class EncodecModelMLX: Module {
    @ModuleInfo public var decoder: EncodecDecoder
    @ModuleInfo public var quantizer: EncodecRVQ
    public let config: EncodecModelConfig

    public init(config: EncodecModelConfig, numQuantizers: Int = 4) {
        self.config = config
        self._decoder = ModuleInfo(wrappedValue: EncodecDecoder(config: config))
        self._quantizer = ModuleInfo(wrappedValue: EncodecRVQ(
            config: config, numQuantizers: numQuantizers))
        super.init()
    }

    /// `codes: [B, K, T]` int → `[B, audio_channels, samples]` waveform.
    public func decode(_ codes: MLXArray) -> MLXArray {
        let embeddings = quantizer.decode(codes)          // [B, T, D]
        return decoder(embeddings)                         // [B, samples, audioChannels]
    }
}
