import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Streaming mel spectrogram preprocessor for Parakeet EOU.
///
/// Accumulates audio samples and extracts mel spectrograms for fixed-size chunks.
/// Uses the same DSP pipeline as the batch Parakeet mel preprocessor (pre-emphasis,
/// Hann window, FFT, Slaney mel filterbank, log, normalization) but operates on
/// streaming windows with running normalization.
struct StreamingMelPreprocessor {
    let config: ParakeetEOUConfig

    private let paddedFFT: Int = 512
    private let nBins: Int = 257  // paddedFFT / 2 + 1
    private let logGuard: Float = 5.960464477539063e-08  // 2^{-24}

    /// Mel filterbank matrix [numMelBins, nBins]
    private let melFilterbank: [Float]
    /// Hann window (periodic)
    private let hannWindow: [Float]

    init(config: ParakeetEOUConfig) {
        self.config = config

        // Build Slaney-normalized mel filterbank
        let sampleRate = Float(config.sampleRate)
        let nBins = 257
        let numMelBins = config.numMelBins

        let fftFreqs = (0..<nBins).map { Float($0) * sampleRate / Float(512) }
        let melLow = Self.hzToMel(0)
        let melHigh = Self.hzToMel(sampleRate / 2)
        let melPoints = (0...numMelBins + 1).map { i in
            Self.melToHz(melLow + Float(i) * (melHigh - melLow) / Float(numMelBins + 1))
        }

        var filterbank = [Float](repeating: 0, count: numMelBins * nBins)
        for m in 0..<numMelBins {
            let fLow = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fHigh = melPoints[m + 2]
            let norm: Float = 2.0 / (fHigh - fLow)

            for k in 0..<nBins {
                let f = fftFreqs[k]
                var val: Float = 0
                if f >= fLow && f <= fCenter && fCenter > fLow {
                    val = (f - fLow) / (fCenter - fLow)
                } else if f > fCenter && f <= fHigh && fHigh > fCenter {
                    val = (fHigh - f) / (fHigh - fCenter)
                }
                filterbank[m * nBins + k] = val * norm
            }
        }
        self.melFilterbank = filterbank

        // Periodic Hann window
        let winLen = config.winLength
        var window = [Float](repeating: 0, count: winLen)
        for i in 0..<winLen {
            window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(winLen)))
        }
        self.hannWindow = window
    }

    /// Extract mel spectrogram from audio samples for one streaming chunk.
    ///
    /// - Parameter audio: PCM Float32 samples at config.sampleRate
    /// - Returns: Mel MLMultiArray [1, numMelBins, melFrames] in float16 + valid frame count
    func extract(_ audio: [Float]) throws -> (mel: MLMultiArray, melLength: Int) {
        guard !audio.isEmpty else {
            let mel = try MLMultiArray(shape: [1, config.numMelBins as NSNumber, 1], dataType: .float16)
            return (mel, 0)
        }

        // Pre-emphasis: y[n] = x[n] - alpha * x[n-1]
        var emphasized = [Float](repeating: 0, count: audio.count)
        emphasized[0] = audio[0]
        for i in 1..<audio.count {
            emphasized[i] = audio[i] - config.preEmphasis * audio[i - 1]
        }

        // Reflect padding for STFT center=True
        let reflectPad = paddedFFT / 2
        var padded = [Float](repeating: 0, count: reflectPad + audio.count + reflectPad)
        for i in 0..<reflectPad {
            padded[reflectPad - 1 - i] = emphasized[min(i + 1, emphasized.count - 1)]
        }
        padded[reflectPad..<reflectPad + emphasized.count] = emphasized[...]
        for i in 0..<reflectPad {
            let srcIdx = max(0, emphasized.count - 2 - i)
            padded[reflectPad + emphasized.count + i] = emphasized[srcIdx]
        }

        // STFT
        let hopLength = config.hopLength
        let winLength = config.winLength
        let nFrames = (padded.count - winLength) / hopLength + 1

        // FFT setup
        let log2n = vDSP_Length(log2(Float(paddedFFT)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw AudioModelError.inferenceFailed(operation: "mel FFT", reason: "Failed to create FFT setup")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var powerSpec = [Float](repeating: 0, count: nBins * nFrames)
        var windowedFrame = [Float](repeating: 0, count: paddedFFT)
        var splitReal = [Float](repeating: 0, count: paddedFFT / 2)
        var splitImag = [Float](repeating: 0, count: paddedFFT / 2)

        for frame in 0..<nFrames {
            let start = frame * hopLength
            // Zero out frame buffer
            memset(&windowedFrame, 0, paddedFFT * MemoryLayout<Float>.stride)
            // Apply window
            vDSP_vmul(Array(padded[start..<start + winLength]), 1, hannWindow, 1,
                       &windowedFrame, 1, vDSP_Length(winLength))

            // Pack for real FFT
            windowedFrame.withUnsafeMutableBufferPointer { buf in
                splitReal.withUnsafeMutableBufferPointer { real in
                    splitImag.withUnsafeMutableBufferPointer { imag in
                        var split = DSPSplitComplex(realp: real.baseAddress!, imagp: imag.baseAddress!)
                        buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: paddedFFT / 2) { complex in
                            vDSP_ctoz(complex, 2, &split, 1, vDSP_Length(paddedFFT / 2))
                        }
                        vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))

                        // Power spectrum |X[k]|^2
                        // DC component
                        powerSpec[0 * nFrames + frame] = split.realp[0] * split.realp[0]
                        // Nyquist
                        powerSpec[(nBins - 1) * nFrames + frame] = split.imagp[0] * split.imagp[0]
                        // Other bins
                        for k in 1..<(nBins - 1) {
                            powerSpec[k * nFrames + frame] =
                                split.realp[k] * split.realp[k] + split.imagp[k] * split.imagp[k]
                        }
                    }
                }
            }
        }

        // Mel filterbank: melSpec[m, t] = sum_k(filterbank[m, k] * powerSpec[k, t])
        let numMelBins = config.numMelBins
        var melSpec = [Float](repeating: 0, count: numMelBins * nFrames)

        // Matrix multiply: filterbank [numMelBins, nBins] × powerSpec [nBins, nFrames]
        vDSP_mmul(melFilterbank, 1, powerSpec, 1, &melSpec, 1,
                  vDSP_Length(numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        // Log mel
        var guard_val = logGuard
        vDSP_vsadd(melSpec, 1, &guard_val, &melSpec, 1, vDSP_Length(melSpec.count))
        var count = Int32(melSpec.count)
        vvlogf(&melSpec, melSpec, &count)

        // Per-bin normalization (mean-subtraction + variance normalization)
        for bin in 0..<numMelBins {
            let offset = bin * nFrames

            // Mean subtraction
            var mean: Float = 0
            melSpec.withUnsafeBufferPointer { buf in
                vDSP_meanv(buf.baseAddress! + offset, 1, &mean, vDSP_Length(nFrames))
            }
            var negMean = -mean
            melSpec.withUnsafeMutableBufferPointer { buf in
                vDSP_vsadd(buf.baseAddress! + offset, 1, &negMean,
                            buf.baseAddress! + offset, 1, vDSP_Length(nFrames))
            }

            // Variance normalization
            var meanSq: Float = 0
            melSpec.withUnsafeBufferPointer { buf in
                vDSP_measqv(buf.baseAddress! + offset, 1, &meanSq, vDSP_Length(nFrames))
            }
            let std = sqrt(max(meanSq, 1e-10))
            var invStd: Float = 1.0 / (std + 1e-5)
            melSpec.withUnsafeMutableBufferPointer { buf in
                vDSP_vsmul(buf.baseAddress! + offset, 1, &invStd,
                            buf.baseAddress! + offset, 1, vDSP_Length(nFrames))
            }
        }

        // Convert to MLMultiArray [1, numMelBins, nFrames] float32
        // (encoder CoreML model expects float32 input)
        let mel = try MLMultiArray(
            shape: [1, numMelBins as NSNumber, nFrames as NSNumber], dataType: .float32)
        let dstPtr = mel.dataPointer.assumingMemoryBound(to: Float.self)
        memcpy(dstPtr, melSpec, numMelBins * nFrames * MemoryLayout<Float>.stride)

        return (mel, nFrames)
    }

    // MARK: - Mel Scale (Slaney)

    private static func hzToMel(_ hz: Float) -> Float {
        if hz < 1000 { return 3.0 * hz / 200.0 }
        return 15.0 + 27.0 / log(6.4) * log(hz / 1000.0)
    }

    private static func melToHz(_ mel: Float) -> Float {
        if mel < 15 { return 200.0 * mel / 3.0 }
        return 1000.0 * exp((mel - 15.0) * log(6.4) / 27.0)
    }
}
