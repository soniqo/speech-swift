import Accelerate
import CoreML
import Foundation

/// Mel spectrogram preprocessor for Canary.
///
/// Implements NeMo's `AudioToMelSpectrogramPreprocessor` as the checkpoint was
/// trained with it: pre-emphasis → centred, **constant**-padded STFT with a
/// **symmetric** Hann window → power spectrum → Slaney-normalised mel bank →
/// `log(x + 2^-24)` → per-feature normalisation over the **sample** (N−1)
/// variance with a 1e-5 epsilon.
///
/// The two details that look like nits are not: reflect padding instead of
/// constant, or a periodic window instead of symmetric, is a measurably
/// different front end. On FLEURS the off-contract combination cost +0.3 WER
/// for English, +0.6 for German and +2.0 for French against this one.
struct MelPreprocessor {
    let config: CanaryConfig

    private let paddedFFT: Int = 512
    private let log2PaddedFFT: vDSP_Length = 9
    private let nBins: Int = 257           // paddedFFT / 2 + 1
    private let centrePad: Int = 256       // n_fft / 2
    private let logGuard: Float = 5.960464477539063e-08  // 2^-24
    private let normEpsilon: Float = 1e-5

    private let fftSetup: FFTSetup
    private let hannWindow: [Float]        // win_length = 400, symmetric
    private let melFilterbank: [Float]     // [nBins, nMels], bin-major

    init(config: CanaryConfig) {
        self.config = config

        // Symmetric Hann — torch.hann_window(periodic: false), denominator N-1.
        var window = [Float](repeating: 0, count: config.winLength)
        for i in 0..<config.winLength {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(config.winLength - 1)))
        }
        self.hannWindow = window

        guard let setup = vDSP_create_fftsetup(log2PaddedFFT, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create vDSP FFT setup")
        }
        self.fftSetup = setup

        self.melFilterbank = MelPreprocessor.buildMelFilterbank(
            nMels: config.numMelBins,
            nBins: 257,
            sampleRate: config.sampleRate,
            paddedFFT: 512
        )
    }

    /// Mel spectrogram for `audio`, padded to the encoder's fixed window.
    ///
    /// - Returns: `[1, numMelBins, window]` and the number of frames that hold
    ///   real audio. The caller passes that count as the encoder's `length`,
    ///   which drives masking — so the zero padding does not change the result.
    func extract(_ audio: [Float], window: Int) throws -> (mel: MLMultiArray, frames: Int) {
        guard !audio.isEmpty else {
            throw CanaryError.emptyAudio
        }

        // Pre-emphasis: x[n] - 0.97 * x[n-1]
        var emphasized = [Float](repeating: 0, count: audio.count)
        emphasized[0] = audio[0]
        if audio.count > 1 {
            audio.withUnsafeBufferPointer { src in
                emphasized.withUnsafeMutableBufferPointer { dst in
                    var negative = -config.preEmphasis
                    vDSP_vsma(src.baseAddress!, 1, &negative,
                              src.baseAddress! + 1, 1,
                              dst.baseAddress! + 1, 1,
                              vDSP_Length(audio.count - 1))
                }
            }
        }

        // Centre padding with zeros — torch.stft(center: true, pad_mode: "constant").
        var padded = [Float](repeating: 0, count: centrePad + emphasized.count + centrePad)
        for i in 0..<emphasized.count {
            padded[centrePad + i] = emphasized[i]
        }

        let stftFrames = max(0, (padded.count - paddedFFT) / config.hopLength + 1)
        // NeMo reports floor(samples / hop) as the valid length, one less than
        // the centred STFT produces. Normalising over the extra frame shifts
        // every bin's statistics.
        let frames = min(stftFrames, audio.count / config.hopLength)
        guard frames > 0 else { throw CanaryError.emptyAudio }
        let usable = min(frames, window)

        let mel = try MLMultiArray(
            shape: [1, NSNumber(value: config.numMelBins), NSNumber(value: window)],
            dataType: .float32)
        let melPointer = UnsafeMutablePointer<Float>(OpaquePointer(mel.dataPointer))
        for i in 0..<(config.numMelBins * window) { melPointer[i] = 0 }

        // A shorter window sits centred inside the FFT frame, as torch.stft
        // places it — 56 samples of leading zeros at 400 in 512.
        let windowOffset = (paddedFFT - config.winLength) / 2

        var frame = [Float](repeating: 0, count: paddedFFT)
        var real = [Float](repeating: 0, count: paddedFFT / 2)
        var imaginary = [Float](repeating: 0, count: paddedFFT / 2)
        var power = [Float](repeating: 0, count: nBins)
        var melFrame = [Float](repeating: 0, count: config.numMelBins)

        for t in 0..<usable {
            for i in 0..<paddedFFT { frame[i] = 0 }
            let start = t * config.hopLength
            for i in 0..<config.winLength {
                frame[windowOffset + i] = padded[start + windowOffset + i] * hannWindow[i]
            }

            real.withUnsafeMutableBufferPointer { realBuffer in
                imaginary.withUnsafeMutableBufferPointer { imagBuffer in
                    var split = DSPSplitComplex(realp: realBuffer.baseAddress!,
                                                imagp: imagBuffer.baseAddress!)
                    frame.withUnsafeBufferPointer { source in
                        source.baseAddress!.withMemoryRebound(
                            to: DSPComplex.self, capacity: paddedFFT / 2
                        ) { complex in
                            vDSP_ctoz(complex, 2, &split, 1, vDSP_Length(paddedFFT / 2))
                        }
                    }
                    vDSP_fft_zrip(fftSetup, &split, 1, log2PaddedFFT, FFTDirection(FFT_FORWARD))

                    // vDSP packs Nyquist into imagp[0] and scales by 2.
                    let nyquist = split.imagp[0] * 0.5
                    let dc = split.realp[0] * 0.5
                    split.imagp[0] = 0
                    power[0] = dc * dc
                    for bin in 1..<(paddedFFT / 2) {
                        let re = split.realp[bin] * 0.5
                        let im = split.imagp[bin] * 0.5
                        power[bin] = re * re + im * im
                    }
                    power[paddedFFT / 2] = nyquist * nyquist
                }
            }

            // mel = filterbank^T · power, then log(x + 2^-24)
            melFilterbank.withUnsafeBufferPointer { bank in
                power.withUnsafeBufferPointer { spectrum in
                    melFrame.withUnsafeMutableBufferPointer { out in
                        vDSP_mmul(spectrum.baseAddress!, 1, bank.baseAddress!, 1,
                                  out.baseAddress!, 1, 1, vDSP_Length(config.numMelBins),
                                  vDSP_Length(nBins))
                    }
                }
            }
            for m in 0..<config.numMelBins {
                melPointer[m * window + t] = log(melFrame[m] + logGuard)
            }
        }

        normalizePerFeature(melPointer, window: window, frames: usable)
        return (mel, usable)
    }

    /// Per-feature normalisation over the frames that hold audio.
    ///
    /// Sample variance (N−1) with a 1e-5 epsilon on the deviation, matching the
    /// reference extractor — not population variance, and not a bare divide.
    private func normalizePerFeature(_ mel: UnsafeMutablePointer<Float>, window: Int, frames: Int) {
        guard frames > 1 else { return }
        for m in 0..<config.numMelBins {
            let row = mel + m * window
            var sum: Float = 0
            vDSP_sve(row, 1, &sum, vDSP_Length(frames))
            let mean = sum / Float(frames)

            var squaredDeviation: Float = 0
            for t in 0..<frames {
                let d = row[t] - mean
                squaredDeviation += d * d
            }
            let variance = squaredDeviation / Float(frames - 1)
            let deviation = (variance > 0 ? sqrt(variance) : 0) + normEpsilon

            var negativeMean = -mean
            var scale = 1 / deviation
            vDSP_vsadd(row, 1, &negativeMean, row, 1, vDSP_Length(frames))
            vDSP_vsmul(row, 1, &scale, row, 1, vDSP_Length(frames))
        }
    }

    /// Slaney-scale mel filterbank, area-normalised, built in double precision
    /// in Hz space exactly as librosa does. Constructing it in float bin space
    /// shifts triangle edges by ~1e-3 bins, which the log amplifies in the
    /// near-floor top bins.
    private static func buildMelFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, paddedFFT: Int
    ) -> [Float] {
        func hzToMel(_ hz: Double) -> Double {
            let fMin = 0.0, fSp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = (minLogHz - fMin) / fSp
            let logStep = log(6.4) / 27.0
            return hz < minLogHz ? (hz - fMin) / fSp
                                 : minLogMel + log(hz / minLogHz) / logStep
        }
        func melToHz(_ mel: Double) -> Double {
            let fMin = 0.0, fSp = 200.0 / 3.0
            let minLogHz = 1000.0
            let minLogMel = (minLogHz - fMin) / fSp
            let logStep = log(6.4) / 27.0
            return mel < minLogMel ? fMin + fSp * mel
                                   : minLogHz * exp(logStep * (mel - minLogMel))
        }

        let melMin = hzToMel(0)
        let melMax = hzToMel(Double(sampleRate) / 2)
        var edges = [Double](repeating: 0, count: nMels + 2)
        for i in 0...(nMels + 1) {
            edges[i] = melToHz(melMin + (melMax - melMin) * Double(i) / Double(nMels + 1))
        }

        let binHz = Double(sampleRate) / Double(paddedFFT)
        var bank = [Float](repeating: 0, count: nBins * nMels)  // bin-major for vDSP_mmul
        for m in 0..<nMels {
            let left = edges[m], centre = edges[m + 1], right = edges[m + 2]
            let enorm = right > left ? 2.0 / (right - left) : 0.0
            for bin in 0..<nBins {
                let hz = Double(bin) * binHz
                var weight = 0.0
                if hz >= left && hz <= centre && centre > left {
                    weight = (hz - left) / (centre - left)
                } else if hz > centre && hz <= right && right > centre {
                    weight = (right - hz) / (right - centre)
                }
                bank[bin * nMels + m] = Float(weight * enorm)
            }
        }
        return bank
    }
}
