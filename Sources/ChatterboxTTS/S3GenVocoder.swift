import Foundation
import MLX
import MLXNN

// MARK: - S3Gen HiFTGenerator vocoder (HiFi-GAN + NSF + ISTFT)
//
// Self-contained port of the S3Gen `mel2wav` HiFTGenerator used by Chatterbox:
// HiFi-GAN with a neural-source-filter (NSF) excitation branch and an ISTFT
// output head, mapping an 80-band mel spectrogram to a 24 kHz waveform.
//
// Weights live under `s3gen.mel2wav.*`; after stripping that prefix the module
// tree here matches the converted parameter keys exactly so the whole vocoder
// can be loaded with `update(parameters:verify:.all)`:
//
//   conv_pre.{weight,bias}                       Conv1d  80 -> 512
//   ups.{0,1,2}.{weight,bias}                    ConvTranspose1d  512->256->128->64
//   source_downs.{0,1,2}.{weight,bias}           Conv1d  18 -> {256,128,64}
//   source_resblocks.{0,1,2}.…                   ResBlock
//   resblocks.{0..8}.…                           ResBlock (3 stages x 3 kernels)
//   conv_post.{weight,bias}                       Conv1d  64 -> 18 (= n_fft+2)
//   m_source.l_linear.{weight,bias}              Linear  9 -> 1
//   f0_predictor.condnet.{0..4}.{weight,bias}    Conv1d
//   f0_predictor.classifier.{weight,bias}        Linear  512 -> 1
//   resblock/source_resblock convs use convs1/convs2 + activations1/activations2.alpha
//
// MLX-Swift Conv1d / ConvTranspose1d weights are [out, kernel, in] (NLC), the
// same layout as the converted tensors, so no transpose is needed at load time.
// All sequence tensors are carried in NCL ([B, C, T]); each conv transposes to
// NLC internally and back, mirroring the channels-first reference graph.

// MARK: Config

public struct S3GenVocoderConfig: Sendable {
    public var inChannels: Int = 80
    public var baseChannels: Int = 512
    public var nbHarmonics: Int = 8
    public var sampleRate: Int = 24000
    public var nsfAlpha: Float = 0.1     // sine amplitude
    public var nsfSigma: Float = 0.003   // additive noise std
    public var nsfVoicedThreshold: Float = 10.0
    public var upsampleRates: [Int] = [8, 5, 3]
    public var upsampleKernelSizes: [Int] = [16, 11, 7]
    public var istftNFFT: Int = 16
    public var istftHopLen: Int = 4
    public var resblockKernelSizes: [Int] = [3, 7, 11]
    public var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var sourceResblockKernelSizes: [Int] = [7, 7, 11]
    public var sourceResblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var lreluSlope: Float = 0.1
    public var audioLimit: Float = 0.99

    public init() {}

    /// prod(upsampleRates) * istftHopLen — the F0 -> waveform upsample factor.
    public var f0UpsampleScale: Int { upsampleRates.reduce(1, *) * istftHopLen }
}

// MARK: Snake activation

/// Snake: `x + (1/alpha) * sin^2(alpha * x)` with `alpha_logscale=False`
/// (alpha stored raw, initialised to 1.0). Operates in NCL: alpha -> [1, C, 1].
final class S3Snake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray  // [C]

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = alpha.reshaped([1, -1, 1])
        let s = sin(a * x)
        return x + (1.0 / (a + 1e-9)) * (s * s)
    }
}

// MARK: NCL Conv1d wrapper (symmetric padding)

/// Conv1d in NCL layout with standard (symmetric) zero padding. The HiFTGenerator
/// is non-causal, so padding is `(kernel*dilation - dilation)/2` on both sides.
///
/// Weight/bias are held as direct parameters (keys `weight`/`bias`) rather than a
/// nested `conv`, so the flattened parameter tree matches the converted tensors
/// (e.g. `conv_pre.weight`) and loads via `update(parameters:verify:.all)`.
/// Weight layout is [out, kernel, in] (NLC), matching the converted tensors.
final class S3Conv1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int
    let dilation: Int
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Int = 0,
        bias: Bool = true
    ) {
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        let scale = sqrt(1.0 / Float(inputChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        super.init()
    }

    /// [B, C, T] -> [B, C_out, T_out]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)  // NCL -> NLC
        h = conv1d(h, weight, stride: stride, padding: padding, dilation: dilation)
        h = h + bias
        return h.transposed(0, 2, 1)   // NLC -> NCL
    }
}

// MARK: NCL ConvTranspose1d wrapper

/// ConvTranspose1d in NCL layout, used for the `ups` stages (true transposed
/// convolution upsampling, padding `(kernel - stride)/2`). Weight/bias are direct
/// parameters (keys `weight`/`bias`) so the flat tree matches the converted tensors
/// (e.g. `ups.0.weight`). Transposed-conv weight layout is [out, kernel, in].
final class S3ConvTranspose1d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let stride: Int
    let padding: Int

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int,
        padding: Int
    ) {
        self.stride = stride
        self.padding = padding
        let scale = sqrt(1.0 / Float(inputChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale, high: scale, [outputChannels, kernelSize, inputChannels])
        self._bias.wrappedValue = MLXArray.zeros([outputChannels])
        super.init()
    }

    /// [B, C, T] -> [B, C_out, T*stride]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)  // NCL -> NLC
        h = convTransposed1d(h, weight, stride: stride, padding: padding)
        h = h + bias
        return h.transposed(0, 2, 1)   // NLC -> NCL
    }
}

// MARK: ResBlock

/// Residual block: per dilation, `snake1 -> conv1(dilated) -> snake2 -> conv2 -> +x`.
/// Param keys: convs1/convs2 (S3Conv1d) and activations1/activations2 (S3Snake.alpha).
final class S3ResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [S3Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [S3Conv1d]
    @ModuleInfo(key: "activations1") var activations1: [S3Snake]
    @ModuleInfo(key: "activations2") var activations2: [S3Snake]

    init(channels: Int, kernelSize: Int, dilations: [Int]) {
        var c1: [S3Conv1d] = []
        var c2: [S3Conv1d] = []
        var a1: [S3Snake] = []
        var a2: [S3Snake] = []
        for d in dilations {
            c1.append(S3Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, dilation: d,
                padding: S3ResBlock.padding(kernelSize, d)))
            c2.append(S3Conv1d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: kernelSize, dilation: 1,
                padding: S3ResBlock.padding(kernelSize, 1)))
            a1.append(S3Snake(channels: channels))
            a2.append(S3Snake(channels: channels))
        }
        self._convs1 = ModuleInfo(wrappedValue: c1)
        self._convs2 = ModuleInfo(wrappedValue: c2)
        self._activations1 = ModuleInfo(wrappedValue: a1)
        self._activations2 = ModuleInfo(wrappedValue: a2)
        super.init()
    }

    static func padding(_ kernelSize: Int, _ dilation: Int) -> Int {
        (kernelSize * dilation - dilation) / 2
    }

    /// [B, C, T] -> [B, C, T]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0 ..< convs1.count {
            var xt = activations1[i](h)
            xt = convs1[i](xt)
            xt = activations2[i](xt)
            xt = convs2[i](xt)
            h = h + xt
        }
        return h
    }
}

// MARK: Sine generator (NSF)

/// Generates harmonic sine waves from F0 with voiced/unvoiced masking.
/// Non-interpolation path (matches the reference default `use_interpolation=False`).
final class S3SineGen {
    let sampleRate: Int
    let harmonicNum: Int
    let sineAmp: Float
    let noiseStd: Float
    let voicedThreshold: Float

    init(sampleRate: Int, harmonicNum: Int, sineAmp: Float, noiseStd: Float, voicedThreshold: Float) {
        self.sampleRate = sampleRate
        self.harmonicNum = harmonicNum
        self.sineAmp = sineAmp
        self.noiseStd = noiseStd
        self.voicedThreshold = voicedThreshold
    }

    /// - Parameter f0: [B, 1, T] fundamental frequency (Hz), channels-first.
    /// - Returns: (sineWaves [B, H, T], uv [B, 1, T]) where H = harmonicNum + 1.
    func callAsFunction(_ f0: MLXArray) -> (MLXArray, MLXArray) {
        let totalHarmonics = harmonicNum + 1

        // harmonic multipliers [1, H, 1]
        let mult = MLXArray((1 ... totalHarmonics).map { Float($0) }).reshaped([1, -1, 1])
        // F_mat = f0 * [1..H] / sr  -> [B, H, T]
        let fMat = (f0 * mult) / Float(sampleRate)

        // theta = 2*pi * (cumsum_T(F_mat) % 1) along time (last axis)
        let csum = cumsum(fMat, axis: -1)
        let theta = (csum - floor(csum)) * Float(2.0 * Float.pi)

        // Random per-harmonic phase offset, zeroed for the fundamental (index 0).
        var phaseVec = MLXRandom.uniform(low: -Float.pi, high: Float.pi, [f0.dim(0), totalHarmonics, 1])
        let idx = MLXArray((0 ..< totalHarmonics).map { Float($0) }).reshaped([1, -1, 1])
        phaseVec = MLX.which(idx .> 0, phaseVec, MLXArray(Float(0.0)))

        var sineWaves = MLXArray(sineAmp) * sin(theta + phaseVec)  // [B, H, T]

        // Voiced mask from F0 (broadcast over harmonics).
        let uv = MLX.which(f0 .> MLXArray(voicedThreshold), MLXArray(Float(1.0)), MLXArray(Float(0.0)))  // [B,1,T]
        let noiseAmp = uv * noiseStd + (1.0 - uv) * (sineAmp / 3.0)
        let noise = noiseAmp * MLXRandom.normal(sineWaves.shape)
        sineWaves = sineWaves * uv + noise

        return (sineWaves, uv)
    }
}

// MARK: Source module (NSF)

/// NSF excitation: sine harmonics -> linear merge -> tanh, plus a noise channel.
/// Param key: `l_linear` (Linear H -> 1).
final class S3SourceModuleHnNSF: Module {
    let sineGen: S3SineGen
    @ModuleInfo(key: "l_linear") var lLinear: Linear
    let sineAmp: Float

    init(sampleRate: Int, harmonicNum: Int, sineAmp: Float, noiseStd: Float, voicedThreshold: Float) {
        self.sineGen = S3SineGen(
            sampleRate: sampleRate, harmonicNum: harmonicNum,
            sineAmp: sineAmp, noiseStd: noiseStd, voicedThreshold: voicedThreshold)
        self.sineAmp = sineAmp
        self._lLinear.wrappedValue = Linear(harmonicNum + 1, 1)
        super.init()
    }

    /// - Parameter x: [B, T, 1] upsampled F0 (NLC).
    /// - Returns: source excitation [B, 1, T] (NCL).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Reference: l_sin_gen takes [B, 1, T]; x arrives as [B, T, 1].
        let f0NCL = x.transposed(0, 2, 1)          // [B, 1, T]
        let (sineWavs, _) = sineGen(f0NCL)         // [B, H, T]
        let sineNLC = sineWavs.transposed(0, 2, 1)  // [B, T, H]
        let merged = tanh(lLinear(sineNLC))         // [B, T, 1]
        return merged.transposed(0, 2, 1)           // [B, 1, T]
    }
}

// MARK: F0 predictor (ConvRNNF0Predictor)

/// Predicts F0 from mel: 5x Conv1d (kernel 3, symmetric pad 1) with ELU, then a
/// Linear classifier; output `abs(.)`. Param keys: `condnet.{0..4}`, `classifier`.
final class S3F0Predictor: Module {
    @ModuleInfo(key: "condnet") var condnet: [S3Conv1d]
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(inChannels: Int = 80, condChannels: Int = 512, numClass: Int = 1) {
        var layers: [S3Conv1d] = []
        for i in 0 ..< 5 {
            let inC = (i == 0) ? inChannels : condChannels
            layers.append(S3Conv1d(
                inputChannels: inC, outputChannels: condChannels,
                kernelSize: 3, padding: 1))
        }
        self._condnet = ModuleInfo(wrappedValue: layers)
        self._classifier.wrappedValue = Linear(condChannels, numClass)
        super.init()
    }

    /// - Parameter mel: [B, C, T] (NCL).
    /// - Returns: [B, T] positive F0 values.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var h = mel  // [B, C, T]
        for conv in condnet {
            h = conv(h)                              // [B, cond, T]
            h = MLX.which(h .> 0, h, exp(h) - 1)     // ELU
        }
        h = h.transposed(0, 2, 1)  // [B, T, cond]
        h = classifier(h)          // [B, T, 1]
        h = h.squeezed(axis: 2)    // [B, T]
        return abs(h)
    }
}

// MARK: STFT / ISTFT (small n_fft via DFT matmul)

/// Periodic Hann window (fftbins=True): denominator is `size`, not `size-1`.
private func hannPeriodic(_ size: Int) -> [Float] {
    (0 ..< size).map { Float(0.5 * (1.0 - cos(2.0 * Double.pi * Double($0) / Double(size)))) }
}

/// STFT for the small vocoder n_fft (16) via a precomputed DFT matrix.
/// Matches `torch.stft(center=True, pad_mode='reflect')`: reflect-pad n_fft/2
/// on each side, frame, window, then real DFT.
/// - Returns: (real, imag) each [B, nBins, nFrames], nBins = n_fft/2 + 1.
private func s3Stft(signal: MLXArray, nFFT: Int, hopLen: Int) -> (MLXArray, MLXArray) {
    let batch = signal.dim(0)
    let sigLen = signal.dim(1)

    let window = MLXArray(hannPeriodic(nFFT))  // [n_fft]

    // DFT matrices [nBins, n_fft]: real = cos, imag = -sin.
    let nBins = nFFT / 2 + 1
    var dftReal = [Float](repeating: 0, count: nBins * nFFT)
    var dftImag = [Float](repeating: 0, count: nBins * nFFT)
    for k in 0 ..< nBins {
        for n in 0 ..< nFFT {
            let angle = 2.0 * Double.pi * Double(k) * Double(n) / Double(nFFT)
            dftReal[k * nFFT + n] = Float(cos(angle))
            dftImag[k * nFFT + n] = Float(-sin(angle))
        }
    }
    let dftRealMat = MLXArray(dftReal).reshaped([nBins, nFFT])
    let dftImagMat = MLXArray(dftImag).reshaped([nBins, nFFT])

    // Reflect padding of n_fft/2 on each side.
    var sig = signal
    let pad = nFFT / 2
    if pad > 0 {
        let leftIdx = MLXArray((1 ... pad).reversed().map { Int32($0) })
        let leftReflect = sig.take(leftIdx, axis: 1)
        let rightIdx = MLXArray(((sigLen - 1 - pad) ..< (sigLen - 1)).reversed().map { Int32($0) })
        let rightReflect = sig.take(rightIdx, axis: 1)
        sig = concatenated([leftReflect, sig, rightReflect], axis: 1)
    }
    if sig.dim(1) < nFFT {
        sig = concatenated([sig, MLXArray.zeros([batch, nFFT - sig.dim(1)])], axis: 1)
    }
    let paddedLen = sig.dim(1)
    let nFrames = Swift.max((paddedLen - nFFT) / hopLen + 1, 1)

    var frames: [MLXArray] = []
    for f in 0 ..< nFrames {
        let start = f * hopLen
        let end = Swift.min(start + nFFT, paddedLen)
        if end - start == nFFT {
            frames.append(sig[0..., start ..< end])
        } else {
            let partial = sig[0..., start ..< end]
            let padArr = MLXArray.zeros([batch, nFFT - (end - start)])
            frames.append(concatenated([partial, padArr], axis: 1))
        }
    }
    let framed = stacked(frames, axis: 1)  // [B, nFrames, n_fft]
    let windowed = framed * window

    let real = matmul(windowed, dftRealMat.transposed())  // [B, nFrames, nBins]
    let imag = matmul(windowed, dftImagMat.transposed())
    return (real.transposed(0, 2, 1), imag.transposed(0, 2, 1))  // [B, nBins, nFrames]
}

/// Inverse STFT via IDFT matmul + Hann-windowed overlap-add, then remove the
/// n_fft/2 padding added by the forward transform.
/// - Parameters magnitude/phase: [B, nBins, nFrames].
/// - Returns: [B, samples].
private func s3Istft(magnitude: MLXArray, phase: MLXArray, nFFT: Int, hopLen: Int) -> MLXArray {
    let batch = magnitude.dim(0)
    let nBins = nFFT / 2 + 1
    let nFrames = magnitude.dim(2)

    // Clip magnitude (matches reference clamp at 1e2).
    let mag = clip(magnitude.transposed(0, 2, 1), max: MLXArray(Float(1e2)))  // [B, nFrames, nBins]
    let ph = phase.transposed(0, 2, 1)

    let real = mag * cos(ph)
    let imag = mag * sin(ph)

    // Conjugate-symmetric mirror for bins [nBins-2 ... 1].
    let fullReal: MLXArray
    let fullImag: MLXArray
    if nBins < nFFT, nBins >= 2 {
        let mirrorIdx = MLXArray((1 ... (nBins - 2)).reversed().map { Int32($0) })
        let mr = real.take(mirrorIdx, axis: 2)
        let mi = -(imag.take(mirrorIdx, axis: 2))
        fullReal = concatenated([real, mr], axis: 2)  // [B, nFrames, n_fft]
        fullImag = concatenated([imag, mi], axis: 2)
    } else {
        fullReal = real
        fullImag = imag
    }

    // IDFT matrices [n_fft, n_fft], scaled by 1/N.
    let invN = 1.0 / Double(nFFT)
    var idftCos = [Float](repeating: 0, count: nFFT * nFFT)
    var idftSin = [Float](repeating: 0, count: nFFT * nFFT)
    for n in 0 ..< nFFT {
        for k in 0 ..< nFFT {
            let angle = 2.0 * Double.pi * Double(n) * Double(k) / Double(nFFT)
            idftCos[n * nFFT + k] = Float(cos(angle) * invN)
            idftSin[n * nFFT + k] = Float(sin(angle) * invN)
        }
    }
    let idftCosMat = MLXArray(idftCos).reshaped([nFFT, nFFT])
    let idftSinMat = MLXArray(idftSin).reshaped([nFFT, nFFT])

    // time[n] = sum_k real[k]*cos - imag[k]*sin
    let timeDomain = matmul(fullReal, idftCosMat.transposed()) - matmul(fullImag, idftSinMat.transposed())

    // Synthesis window (same periodic Hann).
    let hannCoeffs = hannPeriodic(nFFT)
    let window = MLXArray(hannCoeffs)
    let windowed = timeDomain * window  // [B, nFrames, n_fft]

    // Overlap-add: split each frame into n_fft/hopLen segments, accumulate.
    let segmentsPerFrame = nFFT / hopLen
    let outHops = nFrames + segmentsPerFrame - 1
    let outLen = outHops * hopLen
    let segments = windowed.reshaped([batch, nFrames, segmentsPerFrame, hopLen])

    var accumulated = MLXArray.zeros([batch, outLen])
    for s in 0 ..< segmentsPerFrame {
        let seg = segments[0..., 0..., s, 0...]          // [B, nFrames, hopLen]
        let segFlat = seg.reshaped([batch, nFrames * hopLen])
        let leftPad = s * hopLen
        let rightPad = outLen - leftPad - nFrames * hopLen
        if leftPad > 0 || rightPad > 0 {
            var parts: [MLXArray] = []
            if leftPad > 0 { parts.append(MLXArray.zeros([batch, leftPad])) }
            parts.append(segFlat)
            if rightPad > 0 { parts.append(MLXArray.zeros([batch, rightPad])) }
            accumulated = accumulated + concatenated(parts, axis: 1)
        } else {
            accumulated = accumulated + segFlat
        }
    }

    // Window-power normalisation.
    var windowSum = [Float](repeating: 0, count: outLen)
    for f in 0 ..< nFrames {
        for n in 0 ..< nFFT {
            let outIdx = f * hopLen + n
            if outIdx < outLen { windowSum[outIdx] += hannCoeffs[n] * hannCoeffs[n] }
        }
    }
    for i in 0 ..< outLen where windowSum[i] < 1e-8 { windowSum[i] = 1e-8 }
    let windowNorm = MLXArray(windowSum).reshaped([1, outLen])
    var out = accumulated / windowNorm

    // Remove the n_fft/2 padding added in the forward STFT (center=True).
    let trim = nFFT / 2
    if trim > 0, out.dim(1) > 2 * trim {
        out = out[0..., trim ..< (out.dim(1) - trim)]
    }
    return out
}

// MARK: HiFTGenerator

/// S3Gen HiFTGenerator: 80-band mel -> 24 kHz waveform.
///
/// Pipeline:
///  1. F0 predictor: mel -> F0.
///  2. NSF: nearest-neighbour upsample F0 by `f0UpsampleScale`, sine excitation.
///  3. STFT of the excitation -> 18-channel source feature.
///  4. conv_pre -> per stage [leaky_relu -> ups (ConvTranspose1d) -> source inject
///     -> averaged multi-receptive-field resblocks].
///  5. leaky_relu -> conv_post -> split into magnitude (exp) / phase (sin) -> ISTFT.
///  6. Clamp to ±audioLimit.
public final class S3GenVocoder: Module {
    public let config: S3GenVocoderConfig

    @ModuleInfo(key: "m_source") var mSource: S3SourceModuleHnNSF
    @ModuleInfo(key: "f0_predictor") var f0Predictor: S3F0Predictor
    @ModuleInfo(key: "conv_pre") var convPre: S3Conv1d
    @ModuleInfo(key: "ups") var ups: [S3ConvTranspose1d]
    @ModuleInfo(key: "source_downs") var sourceDowns: [S3Conv1d]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [S3ResBlock]
    @ModuleInfo(key: "resblocks") var resblocks: [S3ResBlock]  // flat: stage*numKernels + j
    @ModuleInfo(key: "conv_post") var convPost: S3Conv1d

    private let numUpsamples: Int
    private let numKernels: Int

    public init(config: S3GenVocoderConfig = S3GenVocoderConfig()) {
        self.config = config
        self.numUpsamples = config.upsampleRates.count
        self.numKernels = config.resblockKernelSizes.count

        // NSF source.
        self._mSource.wrappedValue = S3SourceModuleHnNSF(
            sampleRate: config.sampleRate,
            harmonicNum: config.nbHarmonics,
            sineAmp: config.nsfAlpha,
            noiseStd: config.nsfSigma,
            voicedThreshold: config.nsfVoicedThreshold)

        // F0 predictor.
        self._f0Predictor.wrappedValue = S3F0Predictor(inChannels: config.inChannels)

        // conv_pre: in -> base, kernel 7, pad 3.
        self._convPre.wrappedValue = S3Conv1d(
            inputChannels: config.inChannels,
            outputChannels: config.baseChannels,
            kernelSize: 7, padding: 3)

        // Channel progression: base, base/2, base/4, ...
        func chan(_ i: Int) -> Int { config.baseChannels / (1 << i) }

        // ups: ConvTranspose1d base/2^i -> base/2^(i+1), pad (k-u)/2.
        var upLayers: [S3ConvTranspose1d] = []
        for i in 0 ..< numUpsamples {
            let u = config.upsampleRates[i]
            let k = config.upsampleKernelSizes[i]
            upLayers.append(S3ConvTranspose1d(
                inputChannels: chan(i),
                outputChannels: chan(i + 1),
                kernelSize: k, stride: u, padding: (k - u) / 2))
        }
        self._ups = ModuleInfo(wrappedValue: upLayers)

        // source_downs / source_resblocks. The source-down stride is the reverse
        // cumulative product of [1] + upsampleRates[::-1][:-1]; for [8,5,3] that is
        // [1,3,15] -> reversed -> [15,3,1]. stride 1 uses kernel 1 (no pad), else
        // kernel u*2 with pad u/2. Out channels = base/2^(i+1).
        let downRates = [1] + Array(config.upsampleRates.reversed().dropLast())
        var cum = 1
        var downCum: [Int] = []
        for r in downRates { cum *= r; downCum.append(cum) }
        downCum.reverse()
        let stftChannels = config.istftNFFT + 2  // 18

        var downs: [S3Conv1d] = []
        var srcResblks: [S3ResBlock] = []
        for i in 0 ..< numUpsamples {
            let u = downCum[i]
            if u == 1 {
                downs.append(S3Conv1d(
                    inputChannels: stftChannels, outputChannels: chan(i + 1),
                    kernelSize: 1, stride: 1, padding: 0))
            } else {
                downs.append(S3Conv1d(
                    inputChannels: stftChannels, outputChannels: chan(i + 1),
                    kernelSize: u * 2, stride: u, padding: u / 2))
            }
            srcResblks.append(S3ResBlock(
                channels: chan(i + 1),
                kernelSize: config.sourceResblockKernelSizes[i],
                dilations: config.sourceResblockDilationSizes[i]))
        }
        self._sourceDowns = ModuleInfo(wrappedValue: downs)
        self._sourceResblocks = ModuleInfo(wrappedValue: srcResblks)

        // resblocks: numUpsamples x numKernels, flattened (matches `resblocks.N`).
        var rbs: [S3ResBlock] = []
        for i in 0 ..< numUpsamples {
            let ch = chan(i + 1)
            for j in 0 ..< numKernels {
                rbs.append(S3ResBlock(
                    channels: ch,
                    kernelSize: config.resblockKernelSizes[j],
                    dilations: config.resblockDilationSizes[j]))
            }
        }
        self._resblocks = ModuleInfo(wrappedValue: rbs)

        // conv_post: base/2^numUpsamples -> n_fft+2, kernel 7, pad 3.
        self._convPost.wrappedValue = S3Conv1d(
            inputChannels: chan(numUpsamples),
            outputChannels: stftChannels,
            kernelSize: 7, padding: 3)

        super.init()
    }

    /// Nearest-neighbour upsample of F0 along time. f0: [B, 1, T] -> [B, 1, T*scale].
    private func f0Upsample(_ f0: MLXArray) -> MLXArray {
        let scale = config.f0UpsampleScale
        if scale == 1 { return f0 }
        let b = f0.dim(0), c = f0.dim(1), t = f0.dim(2)
        let expanded = f0.expandedDimensions(axis: 3)            // [B, 1, T, 1]
        let rep = MLX.repeated(expanded, count: scale, axis: 3)  // [B, 1, T, scale]
        return rep.reshaped([b, c, t * scale])
    }

    /// decode: mel + source excitation -> waveform.
    /// - Parameters x: mel [B, C, T] (NCL); s: source [B, 1, T_s] (NCL).
    private func decode(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        // STFT of the source signal -> [B, 18, T_stft].
        let (sReal, sImag) = s3Stft(
            signal: s.squeezed(axis: 1), nFFT: config.istftNFFT, hopLen: config.istftHopLen)
        let sStft = concatenated([sReal, sImag], axis: 1)

        var h = convPre(x)  // [B, base, T]

        for i in 0 ..< numUpsamples {
            h = leakyRelu(h, config.lreluSlope)
            h = ups[i](h)

            if i == numUpsamples - 1 {
                // ReflectionPad1d((1, 0)): prepend the 2nd time sample.
                let reflected = h[0..., 0..., 1 ..< 2]
                h = concatenated([reflected, h], axis: 2)
            }

            // Source injection.
            var si = sourceDowns[i](sStft)
            si = sourceResblocks[i](si)

            // Align time (the reference adds directly; lengths match by design, but
            // trim defensively to the shorter of the two).
            let hLen = h.dim(2), sLen = si.dim(2)
            let minLen = Swift.min(hLen, sLen)
            if hLen > minLen { h = h[0..., 0..., 0 ..< minLen] }
            if sLen > minLen { si = si[0..., 0..., 0 ..< minLen] }
            h = h + si

            // Averaged multi-receptive-field fusion.
            let start = i * numKernels
            var fused = resblocks[start](h)
            for j in 1 ..< numKernels { fused = fused + resblocks[start + j](h) }
            h = fused / Float(numKernels)
        }

        h = leakyRelu(h, config.lreluSlope)
        h = convPost(h)  // [B, 18, T_final]

        // Split into magnitude (exp) and phase (sin).
        let nBins = config.istftNFFT / 2 + 1  // 9
        let magPart = h[0..., 0 ..< nBins, 0...]
        let phasePart = h[0..., nBins ..< (config.istftNFFT + 2), 0...]
        let magnitude = exp(magPart)
        let phase = sin(phasePart)

        var audio = s3Istft(
            magnitude: magnitude, phase: phase,
            nFFT: config.istftNFFT, hopLen: config.istftHopLen)  // [B, samples]
        audio = clip(audio, min: -config.audioLimit, max: config.audioLimit)
        return audio
    }

    /// Generate a 24 kHz waveform from a mel spectrogram.
    /// - Parameter mel: [B, T, 80] or [B, 80, T] mel (NLC or NCL accepted).
    /// - Returns: waveform [B, samples].
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // Normalise to NCL [B, 80, T].
        var melNCL: MLXArray
        if mel.dim(mel.ndim - 1) == config.inChannels && mel.dim(1) != config.inChannels {
            melNCL = mel.transposed(0, 2, 1)
        } else {
            melNCL = mel
        }

        // 1. F0 -> [B, T].
        let f0 = f0Predictor(melNCL)

        // 2. Upsample F0 and build the source excitation.
        let f0NCL = f0.expandedDimensions(axis: 1)  // [B, 1, T]
        let sUp = f0Upsample(f0NCL)                 // [B, 1, T*scale]
        let sUpNLC = sUp.transposed(0, 2, 1)        // [B, T*scale, 1]
        let source = mSource(sUpNLC)                // [B, 1, T*scale]

        // 3-6. Decode mel + source -> waveform.
        return decode(melNCL, source)
    }

    /// Mirrors the reference `mel2wav.inference(speech_feat=...)`. The Python form
    /// also returns the source signal; here we expose just the waveform, which is
    /// what callers consume.
    /// - Parameter mel: mel spectrogram (NLC or NCL).
    /// - Returns: 24 kHz waveform [B, samples].
    public func inference(mel: MLXArray) -> MLXArray {
        callAsFunction(mel)
    }
}

// MARK: Helpers

/// LeakyReLU with explicit negative slope (NCL/NLC agnostic, elementwise).
private func leakyRelu(_ x: MLXArray, _ slope: Float) -> MLXArray {
    maximum(x, MLXArray(slope) * x)
}
