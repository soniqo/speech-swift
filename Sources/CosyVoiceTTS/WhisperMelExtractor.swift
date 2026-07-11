import Foundation
import Accelerate
import MLX

/// 128-bin log-mel spectrogram extractor for the s3tokenizer-v3 encoder.
///
/// Exactly mirrors `s3tokenizer.utils.log_mel_spectrogram` (which itself is
/// Whisper-compatible):
///   - 16 kHz mono input
///   - n_fft = 400, hop = 160, window = hann (periodic)
///   - 128 mel bins (HTK scale, Slaney norm)
///   - power = |stft|^2, then `log10(max(mel, 1e-10))`
///   - dynamic range: `max(log, log.max() - 8.0)`, then `(x + 4.0) / 4.0`
///
/// Note: this duplicates `Qwen3ASR.WhisperFeatureExtractor` byte-for-byte
/// because CosyVoiceTTS doesn't depend on Qwen3ASR (and shouldn't — they
/// serve different model families). If/when we have a shared mel-extractor
/// library under `AudioCommon`, this file should move there.
final class WhisperMelExtractor {
    public let sampleRate: Int = 16_000
    public let nFFT: Int = 400
    public let hopLength: Int = 160
    public let nMels: Int = 128

    private let paddedFFT: Int = 512                 // power-of-2 for vDSP
    private let log2PaddedFFT: vDSP_Length = 9
    private var fftSetup: FFTSetup
    private var hannWindow: [Float]
    private var melFilterbank: [Float] = []           // row-major [nMels, nBins]

    public init() {
        hannWindow = [Float](repeating: 0, count: 400)
        for i in 0..<400 {
            hannWindow[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(400)))
        }
        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup for paddedFFT=512")
        }
        fftSetup = setup
        setupMelFilterbank()
    }

    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0
        let nBins = paddedFFT / 2 + 1

        // s3tokenizer uses `librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)`,
        // i.e. librosa's default = **Slaney** mel scale (NOT HTK). The HTK
        // formula in Qwen3ASR.WhisperFeatureExtractor was correct for that
        // model, but produces a different filterbank for s3tokenizer and was
        // costing us ~50% relative error vs upstream.
        func hzToMel(_ hz: Float) -> Float {
            let fSp: Float = 200.0 / 3.0
            let minLogHz: Float = 1_000
            let minLogMel: Float = minLogHz / fSp           // 15
            let logStep: Float = log(6.4) / 27.0
            if hz < minLogHz {
                return hz / fSp
            }
            return minLogMel + log(hz / minLogHz) / logStep
        }
        func melToHz(_ mel: Float) -> Float {
            let fSp: Float = 200.0 / 3.0
            let minLogHz: Float = 1_000
            let minLogMel: Float = minLogHz / fSp           // 15
            let logStep: Float = log(6.4) / 27.0
            if mel < minLogMel {
                return fSp * mel
            }
            return minLogHz * exp(logStep * (mel - minLogMel))
        }

        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        let nMelPoints = nMels + 2
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            let t = Float(i) / Float(nMelPoints - 1)
            melPoints[i] = melToHz(melMin + (melMax - melMin) * t)
        }

        // Triangular filters in [nBins, nMels]
        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for mel in 0..<nMels {
            let leftHz = melPoints[mel]
            let centerHz = melPoints[mel + 1]
            let rightHz = melPoints[mel + 2]
            for bin in 0..<nBins {
                let hz = fftFreqs[bin]
                var v: Float = 0
                if hz >= leftHz && hz <= centerHz {
                    v = (hz - leftHz) / max(centerHz - leftHz, 1e-12)
                } else if hz > centerHz && hz <= rightHz {
                    v = (rightHz - hz) / max(rightHz - centerHz, 1e-12)
                }
                filterbank[bin * nMels + mel] = v
            }
        }
        // Slaney normalization: 2 / (right - left)
        for mel in 0..<nMels {
            let leftHz = melPoints[mel]
            let rightHz = melPoints[mel + 2]
            let enorm: Float = 2.0 / (rightHz - leftHz)
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Store as [nMels, nBins]
        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }
        self.melFilterbank = transposed
    }

    /// Compute the 128-mel log-mel spectrogram of an audio waveform.
    /// - Parameter audio: mono float samples at 16 kHz
    /// - Returns: `[1, nMels, T]` MLXArray ready to feed into `SpeechTokenizerModel.encode`
    public func extract(_ audio: [Float]) -> MLXArray {
        let nBins = paddedFFT / 2 + 1
        let halfPadded = paddedFFT / 2

        // torch.stft with center=True (the default) uses reflect-padded input —
        // match that so frame boundaries line up with the upstream output.
        let pad = nFFT / 2
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

        // s3tokenizer drops the last frame: `stft[..., :-1]`. We replicate by
        // computing one less frame than torch.stft would.
        let nFramesAll = (padded.count - nFFT) / hopLength + 1
        let nFrames = max(nFramesAll - 1, 0)

        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)
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
            // vDSP_fft_zrip scales non-DC/non-Nyquist bins by 2x relative to a
            // standard DFT. We need to compensate so the mel values match a
            // torch.stft-based reference. For power (|x|²), the correction is /4.
            magnitude[base] = splitReal[0] * splitReal[0]                 // DC, no scaling
            magnitude[base + halfPadded] = splitImag[0] * splitImag[0]    // Nyquist, no scaling
            for k in 1..<halfPadded {
                let p = splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]
                magnitude[base + k] = p * 0.25
            }
        }

        // mel = filterbank[nMels, nBins] @ magnitude^T[nBins, nFrames]
        // Equivalent to magnitude[nFrames, nBins] @ filterbank^T[nBins, nMels] → mel[nFrames, nMels]
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(melFilterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)
        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // log10(clamp(mel, 1e-10, +inf)) — then dynamic range + offset/scale.
        let count = melSpec.count
        var countN = Int32(count)
        var epsilon: Float = 1e-10
        var hi: Float = .greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &epsilon, &hi, &melSpec, 1, vDSP_Length(count))
        vvlog10f(&melSpec, melSpec, &countN)

        var maxVal: Float = -.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))
        var minClamp = maxVal - 8.0
        vDSP_vclip(melSpec, 1, &minClamp, &hi, &melSpec, 1, vDSP_Length(count))

        // (x + 4) / 4
        var add: Float = 4.0
        var div: Float = 4.0
        vDSP_vsadd(melSpec, 1, &add, &melSpec, 1, vDSP_Length(count))
        vDSP_vsdiv(melSpec, 1, &div, &melSpec, 1, vDSP_Length(count))

        // melSpec is [nFrames, nMels] row-major. Upstream expects [batch, n_mels, T] — transpose.
        var melCT = [Float](repeating: 0, count: nMels * nFrames)
        vDSP_mtrans(melSpec, 1, &melCT, 1, vDSP_Length(nMels), vDSP_Length(nFrames))

        return MLXArray(melCT, [1, nMels, nFrames])
    }
}
