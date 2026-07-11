import Foundation
import Accelerate
import MLX

/// 80-bin log-mel spectrogram for CosyVoice 3's flow model `prompt_feat` input.
///
/// Mirrors `matcha.utils.audio.mel_spectrogram` (which CosyVoice 3 inherits via
/// `cosyvoice3.yaml`'s `feat_extractor`):
///
///   - 24 kHz mono input
///   - n_fft = 1920, hop = 480, win = 1920 (50 Hz frame rate)
///   - 80 mel bins, fmin = 0, fmax = Nyquist (12 kHz)
///   - Window: hann (periodic)
///   - Mel scale: **librosa default = Slaney**, with Slaney triangular
///     normalisation (2 / (right - left))
///   - STFT center = False, with manual reflect pre-pad of
///     `(n_fft - hop) / 2 = 720` samples each side
///   - log compression: `torch.log(torch.clamp(mel, min=1e-5))` — natural log
///     with a fixed 1e-5 floor. **No dynamic-range clip and no offset/scale**
///     (this is what distinguishes it from the Whisper-style extractor used
///     by the speech tokenizer).
///
/// Returns `[1, n_mels, T]` (channels-first), aligned with the layout the
/// flow's `conds` slot expects after `prompt_feat.transpose(1, 2)` upstream.
final class FlowMelExtractor {
    public let sampleRate: Int = 24_000
    public let nFFT: Int = 1_920
    public let hopLength: Int = 480
    public let winLength: Int = 1_920
    public let nMels: Int = 80
    public let fMin: Float = 0
    public let fMax: Float = 12_000  // sample_rate / 2

    private let paddedFFT: Int = 2_048               // smallest 2^k >= n_fft
    private let log2PaddedFFT: vDSP_Length = 11
    private var fftSetup: FFTSetup
    private var hannWindow: [Float]
    private var melFilterbank: [Float] = []          // row-major [nMels, nBins]

    public init() {
        hannWindow = [Float](repeating: 0, count: 1_920)
        for i in 0..<1_920 {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(1_920)))
        }
        guard let setup = vDSP_create_fftsetup(11, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup for paddedFFT=2048")
        }
        fftSetup = setup
        setupMelFilterbank()
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    // MARK: - Slaney mel ↔ Hz (librosa default)

    private func hzToMel(_ hz: Float) -> Float {
        // Slaney: linear ramp up to 1000 Hz, log above.
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000
        let minLogMel: Float = (minLogHz - fMin) / fSp                  // 15
        let logStep: Float = log(6.4) / 27.0
        if hz < minLogHz {
            return (hz - fMin) / fSp
        }
        return minLogMel + log(hz / minLogHz) / logStep
    }

    private func melToHz(_ mel: Float) -> Float {
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000
        let minLogMel: Float = (minLogHz - fMin) / fSp
        let logStep: Float = log(6.4) / 27.0
        if mel < minLogMel {
            return fMin + fSp * mel
        }
        return minLogHz * exp(logStep * (mel - minLogMel))
    }

    private func setupMelFilterbank() {
        let nBins = paddedFFT / 2 + 1

        // Mel grid: nMels + 2 anchor points.
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let nPts = nMels + 2
        var melPts = [Float](repeating: 0, count: nPts)
        for i in 0..<nPts {
            let t = Float(i) / Float(nPts - 1)
            melPts[i] = melToHz(melMin + (melMax - melMin) * t)
        }

        // FFT bin frequencies — use paddedFFT (n_fft padded to 2048).
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        // Triangular filters.
        var fb = [Float](repeating: 0, count: nBins * nMels)
        for m in 0..<nMels {
            let left = melPts[m]
            let center = melPts[m + 1]
            let right = melPts[m + 2]
            for bin in 0..<nBins {
                let hz = fftFreqs[bin]
                var v: Float = 0
                if hz >= left && hz <= center {
                    v = (hz - left) / max(center - left, 1e-12)
                } else if hz > center && hz <= right {
                    v = (right - hz) / max(right - center, 1e-12)
                }
                fb[bin * nMels + m] = v
            }
        }

        // Slaney norm: 2 / (right - left)
        for m in 0..<nMels {
            let enorm: Float = 2.0 / max(melPts[m + 2] - melPts[m], 1e-12)
            for bin in 0..<nBins {
                fb[bin * nMels + m] *= enorm
            }
        }

        // Transpose to [nMels, nBins] for the gemm path.
        var t = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            for bin in 0..<nBins {
                t[m * nBins + bin] = fb[bin * nMels + m]
            }
        }
        self.melFilterbank = t
    }

    // MARK: - Extraction

    /// - Parameter audio: mono float samples at 24 kHz
    /// - Returns: `[1, 80, T]` natural-log-mel MLXArray
    public func extract(_ audio: [Float]) -> MLXArray {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // matcha pre-pads by (n_fft - hop) / 2 reflect samples each side,
        // then calls torch.stft with center=False.
        let pad = (nFFT - hopLength) / 2          // 720
        var padded = [Float](repeating: 0, count: pad + audio.count + pad)
        for i in 0..<pad {
            let src = min(pad - i, audio.count - 1)
            padded[i] = audio[max(0, src)]
        }
        for i in 0..<audio.count {
            padded[pad + i] = audio[i]
        }
        for i in 0..<pad {
            let src = audio.count - 2 - i
            padded[pad + audio.count + i] = audio[max(0, src)]
        }

        // center=False: frames are placed every hop, starting at 0 (no implicit
        // pad inside stft). n_frames = floor((N - n_fft) / hop) + 1 if N >= n_fft.
        let N = padded.count
        let nFrames = N >= nFFT ? (N - nFFT) / hopLength + 1 : 0

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
        // Matcha uses MAGNITUDE — `torch.sqrt(spec.pow(2).sum(-1) + 1e-9)` — not
        // power. Using power here would square the dynamic range before the log
        // and break the flow's cond conditioning (cond values land in a totally
        // different scale than what the model was trained on).
        var magnitude = [Float](repeating: 0, count: nFrames * nBins)

        for frame in 0..<nFrames {
            let start = frame * hopLength
            padded.withUnsafeBufferPointer { buf in
                vDSP_vmul(buf.baseAddress! + start, 1, hannWindow, 1,
                          &paddedFrame, 1, vDSP_Length(nFFT))
            }
            for i in nFFT..<paddedFFT { paddedFrame[i] = 0 }
            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }
            splitReal.withUnsafeMutableBufferPointer { rb in
                splitImag.withUnsafeMutableBufferPointer { ib in
                    var sc = DSPSplitComplex(realp: rb.baseAddress!, imagp: ib.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &sc, 1, log2PaddedFFT,
                                  FFTDirection(kFFTDirection_Forward))
                }
            }
            let base = frame * nBins
            // Matcha computes `sqrt(spec.pow(2).sum(-1) + 1e-9)` = magnitude
            // with an epsilon floor. vDSP_fft_zrip scales non-DC/non-Nyquist
            // bins by 2x relative to a standard DFT, so we divide those by 2
            // before taking the magnitude.
            magnitude[base] = sqrt(splitReal[0] * splitReal[0] + 1e-9)               // DC
            magnitude[base + halfPadded] = sqrt(splitImag[0] * splitImag[0] + 1e-9)  // Nyquist
            for k in 1..<halfPadded {
                let re = splitReal[k] * 0.5
                let im = splitImag[k] * 0.5
                magnitude[base + k] = sqrt(re * re + im * im + 1e-9)
            }
        }

        // mel[nFrames, nMels] = mag[nFrames, nBins] @ filterbank^T[nBins, nMels]
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(melFilterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // Clamp to 1e-5 (matcha's spectral_normalize_torch floor), then natural log.
        let count = melSpec.count
        var countN = Int32(count)
        var lo: Float = 1e-5
        var hi: Float = .greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &lo, &hi, &melSpec, 1, vDSP_Length(count))
        vvlogf(&melSpec, melSpec, &countN)

        // Transpose [nFrames, nMels] → [nMels, nFrames] for channels-first output.
        var melCT = [Float](repeating: 0, count: nMels * nFrames)
        vDSP_mtrans(melSpec, 1, &melCT, 1, vDSP_Length(nMels), vDSP_Length(nFrames))

        return MLXArray(melCT, [1, nMels, nFrames])
    }
}
