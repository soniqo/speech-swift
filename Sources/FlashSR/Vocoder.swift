import Foundation
import MLX
import MLXNN

// MARK: - SnakeBeta

/// Per-channel learnable α, β with log-scale parameterisation:
/// `x + (1/β) * sin²(α x)` where α = exp(α_param), β = exp(β_param).
public final class FlashSRSnakeBeta: Module {
    @ParameterInfo public var alpha: MLXArray
    @ParameterInfo public var beta: MLXArray

    public init(channels: Int) {
        self._alpha = ParameterInfo(wrappedValue: MLXArray.zeros([channels]))
        self._beta = ParameterInfo(wrappedValue: MLXArray.zeros([channels]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = exp(alpha)
        let b = exp(beta)
        let s = sin(x * a)
        let eps: Float = 1e-9
        return x + (Float(1.0) / (b + MLXArray(eps))) * (s * s)
    }
}

// MARK: - Anti-aliased FIR up/down

/// Replicate-pads the time axis. `x: (B, T, C)` → `(B, T+left+right, C)`.
/// Mirrors `torch.nn.functional.pad(..., mode='replicate')` used by BigVGAN's
/// alias-free up/down samplers. Using zero-pad here would add boundary
/// discontinuities that show up as HF noise in the synthesised audio.
private func replicatePad1D(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
    if left == 0 && right == 0 { return x }
    let B = x.dim(0); let C = x.dim(2)
    var parts: [MLXArray] = []
    if left > 0 {
        let edge = x[0..., 0..<1, 0...]                                       // (B, 1, C)
        parts.append(MLX.broadcast(edge, to: [B, left, C]))
    }
    parts.append(x)
    if right > 0 {
        let edge = x[0..., (x.dim(1) - 1)..<x.dim(1), 0...]                   // (B, 1, C)
        parts.append(MLX.broadcast(edge, to: [B, right, C]))
    }
    return MLX.concatenated(parts, axis: 1)
}

/// Replicate-pad + depth-wise FIR + decimate. Mirrors PyTorch
/// `BigVGAN/alias_free_torch/filter.LowPassFilter1d.forward`:
///   K = kernel_size; even = (K%2==0)
///   padLeft  = K//2 - int(even)
///   padRight = K//2
/// `x: (B, T, C)`, `filt: (1, 1, K)`. Output: `(B, ceil(T/factor), C)`.
private func firDownsample(_ x: MLXArray, filt: MLXArray, factor: Int = 2) -> MLXArray {
    let K = filt.dim(-1)
    let even = (K % 2 == 0)
    let padLeft = K / 2 - (even ? 1 : 0)
    let padRight = K / 2
    let xp = replicatePad1D(x, left: padLeft, right: padRight)
    let TOut = xp.dim(1) - K + 1
    let arangeK = MLXArray(0..<Int32(K)).expandedDimensions(axis: 0)
    let arangeT = MLXArray(0..<Int32(TOut)).expandedDimensions(axis: 1)
    let idx = arangeK + arangeT                                              // (TOut, K)
    let windowed = xp[0..., idx, 0...]                                       // (B, TOut, K, C)
    let w = filt.reshaped([1, 1, K, 1])
    let out = (windowed * w).sum(axis: 2)
    return out[0..., .stride(by: factor), 0...]
}

/// Replicate-pad → zero-insert → conv → trim. Equivalent to PyTorch
/// `BigVGAN/alias_free_torch/resample.UpSample1d.forward`:
///   pad       = K/factor - 1
///   padLeft   = pad*factor + (K-factor)/2
///   padRight  = pad*factor + (K-factor+1)/2
///   x = F.pad(x, (pad,pad), mode='replicate')
///   x = factor * F.conv_transpose1d(x, filt, stride=factor)
///   x = x[..., padLeft : -padRight]
/// `x: (B, T, C)`, `filt: (1, 1, K)`. Output: `(B, T*factor, C)`.
private func firUpsample(_ x: MLXArray, filt: MLXArray, factor: Int = 2) -> MLXArray {
    let K = filt.dim(-1)
    let pad = K / factor - 1
    let padLeft = pad * factor + (K - factor) / 2
    let padRight = pad * factor + (K - factor + 1) / 2

    // 1. Replicate-pad input by `pad` on each side
    let xp = replicatePad1D(x, left: pad, right: pad)                       // (B, T+2*pad, C)
    let B = xp.dim(0); let Tp = xp.dim(1); let C = xp.dim(2)

    // 2. Insert (factor-1) zeros between samples (ConvTranspose1d-style)
    //    Stretched length = (Tp-1)*factor + 1
    var stacks: [MLXArray] = [xp]
    for _ in 1..<factor {
        stacks.append(MLXArray.zeros([B, Tp, C], dtype: xp.dtype))
    }
    let upFull = stacked(stacks, axis: 2).reshaped([B, Tp * factor, C])
    // Trim the trailing (factor-1) zeros to get length (Tp-1)*factor + 1
    let upTrim = upFull[0..., 0..<((Tp - 1) * factor + 1), 0...]

    // 3. Convolve with K-tap filter (* factor gain). Pad K-1 on each side then
    //    valid-conv gives length (Tp-1)*factor + K (= ConvTranspose1d output).
    let upPad = MLX.padded(upTrim,
                            widths: [.init((0, 0)), .init((K - 1, K - 1)), .init((0, 0))],
                            value: MLXArray(Float(0)))
    let TFull = upPad.dim(1) - K + 1
    let arangeK = MLXArray(0..<Int32(K)).expandedDimensions(axis: 0)
    let arangeT = MLXArray(0..<Int32(TFull)).expandedDimensions(axis: 1)
    let idx = arangeK + arangeT
    let windowed = upPad[0..., idx, 0...]                                   // (B, TFull, K, C)
    let w = (filt * Float(factor)).reshaped([1, 1, K, 1])
    let out = (windowed * w).sum(axis: 2)                                    // (B, TFull, C)

    // 4. Trim padLeft/padRight (output resolution) → final length T*factor.
    let endIdx = out.dim(1) - padRight
    return out[0..., padLeft..<endIdx, 0...]
}

// MARK: - Activation1d (anti-aliased SnakeBeta)

public final class FlashSRUpsampleFilt: Module {
    @ParameterInfo public var filter: MLXArray
    public override init() {
        self._filter = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, 12]))
        super.init()
    }
}

public final class FlashSRLowpass: Module {
    @ParameterInfo public var filter: MLXArray
    public override init() {
        self._filter = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, 12]))
        super.init()
    }
}

public final class FlashSRDownsampleFilt: Module {
    @ModuleInfo public var lowpass: FlashSRLowpass
    public override init() {
        self._lowpass = ModuleInfo(wrappedValue: FlashSRLowpass())
        super.init()
    }
}

public final class FlashSRActivation1d: Module {
    @ModuleInfo public var act: FlashSRSnakeBeta
    @ModuleInfo public var upsample: FlashSRUpsampleFilt
    @ModuleInfo public var downsample: FlashSRDownsampleFilt

    public init(channels: Int) {
        self._act = ModuleInfo(wrappedValue: FlashSRSnakeBeta(channels: channels))
        self._upsample = ModuleInfo(wrappedValue: FlashSRUpsampleFilt())
        self._downsample = ModuleInfo(wrappedValue: FlashSRDownsampleFilt())
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = firUpsample(xIn, filt: upsample.filter, factor: 2)
        x = act(x)
        x = firDownsample(x, filt: downsample.lowpass.filter, factor: 2)
        return x
    }
}

// MARK: - AMPBlock (BigVGAN block)

/// `convs1`/`convs2` are arrays of Conv1d; `activations` interleaves them.
/// Forward:
///   for i in 0..<n_dil:
///     h = act[2i](x); h = convs1[i](h); h = act[2i+1](h); h = convs2[i](h); x = x + h
public final class FlashSRAMPBlock1: Module {
    @ModuleInfo public var convs1: [Conv1d]
    @ModuleInfo public var convs2: [Conv1d]
    @ModuleInfo public var activations: [FlashSRActivation1d]

    public init(channels: Int, kernelSize: Int = 3, dilations: [Int] = [1, 3, 5]) {
        func pad(_ k: Int, _ d: Int) -> Int { (k * d - d) / 2 }
        var convs1List: [Conv1d] = []
        var convs2List: [Conv1d] = []
        var acts: [FlashSRActivation1d] = []
        for d in dilations {
            convs1List.append(Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: pad(kernelSize, d),
                dilation: d, bias: true))
            convs2List.append(Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, stride: 1, padding: pad(kernelSize, 1),
                dilation: 1, bias: true))
            acts.append(FlashSRActivation1d(channels: channels))
            acts.append(FlashSRActivation1d(channels: channels))
        }
        self._convs1 = ModuleInfo(wrappedValue: convs1List)
        self._convs2 = ModuleInfo(wrappedValue: convs2List)
        self._activations = ModuleInfo(wrappedValue: acts)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray) -> MLXArray {
        var x = xIn
        for i in 0..<convs1.count {
            let a1 = activations[2 * i]
            let a2 = activations[2 * i + 1]
            var h = a1(x)
            h = convs1[i](h)
            h = a2(h)
            h = convs2[i](h)
            x = x + h
        }
        return x
    }
}

// MARK: - Audio block (LR audio → conditioning pyramid)

/// Wraps a single Conv1d (or ConvTransposed1d) under slot "0" to match
/// PyTorch nn.Sequential([conv, LeakyReLU]) keying — uses the shared
/// FlashSRSeqLayers list pattern so mlx-swift's unflatten matches.
public typealias FlashSRSlot0Conv1d = FlashSRSeqLayers
public typealias FlashSRSlot0ConvT1d = FlashSRSeqLayers

public final class FlashSRAudioBlock: Module {
    @ModuleInfo public var emb: Conv1d
    @ModuleInfo public var downsamples: [FlashSRSeqLayers]

    public init(cfg: FlashSRVocoderConfig) {
        let ich = cfg.upsampleInitialChannel / (1 << cfg.upsampleRates.count)
        self._emb = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: 1, outputChannels: ich,
            kernelSize: 7, stride: 1, padding: 3, bias: true))
        var dsLayers: [FlashSRSeqLayers] = []
        for i in stride(from: cfg.upsampleKernelSizes.count - 1, through: 0, by: -1) {
            let inC = cfg.upsampleInitialChannel / (1 << (i + 1))
            let outC = cfg.upsampleInitialChannel / (1 << i)
            let k = cfg.upsampleKernelSizes[i]
            let s = cfg.upsampleRates[i]
            let pad = s - (k % 2 == 0 ? 1 : 0)
            dsLayers.append(FlashSRSeqLayers([
                Conv1d(inputChannels: inC, outputChannels: outC,
                       kernelSize: k, stride: s, padding: pad, bias: true),
            ]))
        }
        self._downsamples = ModuleInfo(wrappedValue: dsLayers)
        super.init()
    }
}

// MARK: - SR Vocoder config + module

public struct FlashSRVocoderConfig {
    public var numMels: Int
    public var upsampleInitialChannel: Int
    public var resblockKernelSizes: [Int]
    public var resblockDilationSizes: [[Int]]
    public var upsampleRates: [Int]
    public var upsampleKernelSizes: [Int]

    public init(numMels: Int = 256,
                upsampleInitialChannel: Int = 1536,
                resblockKernelSizes: [Int] = [3, 7, 11],
                resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                upsampleRates: [Int] = [10, 6, 2, 2, 2]) {
        self.numMels = numMels
        self.upsampleInitialChannel = upsampleInitialChannel
        self.resblockKernelSizes = resblockKernelSizes
        self.resblockDilationSizes = resblockDilationSizes
        self.upsampleRates = upsampleRates
        self.upsampleKernelSizes = upsampleRates.map { $0 * 2 }
    }
}

public final class FlashSRSRVocoder: Module {
    @ModuleInfo(key: "audio_block") public var audioBlock: FlashSRAudioBlock
    @ModuleInfo(key: "conv_pre") public var convPre: Conv1d
    /// PyTorch saved as ModuleList of single-element Sequential ([ConvTranspose1d]).
    /// We use FlashSRWrappedConv1d-style wrappers around ConvTransposed1d.
    /// Each `ups[i]` is a `Sequential([ConvTransposed1d])` — slot 0 holds the
    /// transposed conv. We use the shared `FlashSRSeqLayers` list pattern.
    @ModuleInfo public var ups: [FlashSRSeqLayers]
    @ModuleInfo public var resblocks: [FlashSRAMPBlock1]
    @ModuleInfo(key: "activation_post") public var activationPost: FlashSRActivation1d
    @ModuleInfo(key: "conv_post") public var convPost: Conv1d

    public let cfg: FlashSRVocoderConfig
    public let numKernels: Int
    public let numUpsamples: Int

    public init(cfg: FlashSRVocoderConfig = .init()) {
        self.cfg = cfg
        self.numKernels = cfg.resblockKernelSizes.count
        self.numUpsamples = cfg.upsampleRates.count

        self._audioBlock = ModuleInfo(wrappedValue: FlashSRAudioBlock(cfg: cfg), key: "audio_block")
        self._convPre = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: cfg.numMels, outputChannels: cfg.upsampleInitialChannel,
            kernelSize: 7, stride: 1, padding: 3, bias: true), key: "conv_pre")

        var upsList: [FlashSRSeqLayers] = []
        for i in 0..<cfg.upsampleRates.count {
            let inC = cfg.upsampleInitialChannel / (1 << i)
            let outC = cfg.upsampleInitialChannel / (1 << (i + 1))
            let u = cfg.upsampleRates[i]
            let k = cfg.upsampleKernelSizes[i]
            upsList.append(FlashSRSeqLayers([
                ConvTransposed1d(inputChannels: inC, outputChannels: outC,
                                  kernelSize: k, stride: u, padding: (k - u) / 2, bias: true),
            ]))
        }
        self._ups = ModuleInfo(wrappedValue: upsList)

        var resblocksList: [FlashSRAMPBlock1] = []
        for i in 0..<numUpsamples {
            let chN = cfg.upsampleInitialChannel / (1 << (i + 1))
            for (j, kSize) in cfg.resblockKernelSizes.enumerated() {
                let d = cfg.resblockDilationSizes[j]
                resblocksList.append(FlashSRAMPBlock1(channels: chN, kernelSize: kSize, dilations: d))
            }
        }
        self._resblocks = ModuleInfo(wrappedValue: resblocksList)

        let finalCh = cfg.upsampleInitialChannel / (1 << numUpsamples)
        self._activationPost = ModuleInfo(wrappedValue: FlashSRActivation1d(channels: finalCh),
                                          key: "activation_post")
        self._convPost = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: finalCh, outputChannels: 1,
            kernelSize: 7, stride: 1, padding: 3, bias: true), key: "conv_post")
        super.init()
    }

    public func callAsFunction(_ melSpec: MLXArray, lrAudio: MLXArray) -> MLXArray {
        // mel_spec: (B, T_mel, num_mels), lr_audio: (B, T_wav)
        var au = lrAudio.expandedDimensions(axis: -1)                                    // (B, T_wav, 1)
        au = audioBlock.emb(au)
        var audioEmbList: [MLXArray] = [au]
        for i in 0..<(numUpsamples - 1) {
            let dsConv = audioBlock.downsamples[i].layers[0] as! Conv1d
            au = dsConv(au)
            // LeakyReLU(0.1) — manual.
            au = MLX.maximum(au, MLXArray(Float(0.1)) * au)
            audioEmbList.append(au)
        }

        var x = convPre(melSpec)
        for i in 0..<numUpsamples {
            let upConv = ups[i].layers[0] as! ConvTransposed1d
            x = upConv(x)
            x = x + audioEmbList[audioEmbList.count - 1 - i]
            var xs: MLXArray? = nil
            for j in 0..<numKernels {
                let h = resblocks[i * numKernels + j](x)
                xs = (xs == nil) ? h : (xs! + h)
            }
            x = xs! / Float(numKernels)
        }

        x = activationPost(x)
        x = convPost(x)
        x = tanh(x)
        // Output is (B, T_wav_hr, 1) — squeeze channel dim.
        return x.squeezed(axis: -1)
    }
}

