import Accelerate
import AudioCommon
import CoreML
import Foundation

/// Mel spectrogram preprocessor for Parakeet TDT models.
///
/// Implements the NeMo `AudioToMelSpectrogramPreprocessor` operations in Swift:
/// pre-emphasis → STFT → power spectrum → mel filterbank → log → per-feature normalization.
/// The default log floor retains this package's existing feature calibration.
///
/// Uses vDSP/Accelerate for performance. No CoreML model needed — pure signal processing.
struct MelPreprocessor {
    enum Calibration {
        /// Preserve the power-to-floor ratio used before the FFT correction.
        case legacy
        /// Use NeMo's absolute log floor with mathematically scaled FFT power.
        case nemo
    }

    let config: ParakeetConfig

    private static let powerCorrection: Float = 0.25
    private static let nemoLogGuard: Float = 0x1p-24

    private let paddedFFT: Int = 512      // n_fft (already power of 2)
    private let log2PaddedFFT: vDSP_Length = 9
    private let nBins: Int = 257          // paddedFFT / 2 + 1
    private let centrePad: Int = 256      // n_fft / 2
    /// torch.stft centres a window shorter than n_fft inside the frame.
    private let windowOffset: Int = 56    // (512 - 400) / 2
    private let logGuard: Float

    private let fftSetup: FFTSetup
    private let hannWindow: [Float]       // win_length = 400
    private let melFilterbank: [Float]    // [nMels, nBins] = [128, 257]

    init(config: ParakeetConfig, calibration: Calibration = .legacy) {
        self.config = config
        switch calibration {
        case .legacy:
            // log(4E + ε) = log(4) + log(E + ε/4). Centering removes the
            // constant, so scaling the floor with power retains the previous
            // normalized features, subject to Float32 rounding.
            self.logGuard = Self.nemoLogGuard * Self.powerCorrection
        case .nemo:
            self.logGuard = Self.nemoLogGuard
        }

        // Hann window, symmetric: 0.5 * (1 - cos(2π*i/(N-1))).
        // The NeMo featurizer builds it with torch.hann_window(periodic: false);
        // the periodic form (denominator N) is a different window.
        var window = [Float](repeating: 0, count: config.winLength)
        for i in 0..<config.winLength {
            window[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(config.winLength - 1)))
        }
        self.hannWindow = window

        guard let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2)) else {
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

    /// Extract mel spectrogram from audio samples.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 samples at 16kHz
    ///   - normalizationWindowLength: Normalize each encoder window independently.
    ///     Nil uses all valid frames, as in NeMo's utterance preprocessor.
    /// - Returns: Tuple of (mel MLMultiArray [1, 128, T], melLength Int)
    func extract(_ audio: [Float], normalizationWindowLength: Int? = nil) throws
        -> (mel: MLMultiArray, melLength: Int)
    {
        guard !audio.isEmpty else {
            throw AudioModelError.inferenceFailed(operation: "mel extraction", reason: "Audio is empty")
        }
        if let normalizationWindowLength, normalizationWindowLength <= 0 {
            throw AudioModelError.invalidConfiguration(
                model: "Parakeet", reason: "Normalization window must contain at least one frame")
        }
        // Pre-emphasis: x[n] = audio[n] - 0.97 * audio[n-1]
        var preemphasized = [Float](repeating: 0, count: audio.count)
        preemphasized[0] = audio[0]
        audio.withUnsafeBufferPointer { src in
            preemphasized.withUnsafeMutableBufferPointer { dst in
                var negCoeff = -config.preEmphasis  // -0.97
                // dst[n] = audio[n] + (-0.97) * audio[n-1]
                vDSP_vsma(src.baseAddress!, 1, &negCoeff,
                          src.baseAddress! + 1, 1,
                          dst.baseAddress! + 1, 1,
                          vDSP_Length(audio.count - 1))
            }
        }

        // Centre padding with ZEROS (n_fft // 2 = 256 each side). The NeMo
        // featurizer calls torch.stft(center: true, pad_mode: "constant") —
        // reflect padding is a different front end, not a detail.
        var padded = [Float](repeating: 0, count: centrePad + preemphasized.count + centrePad)
        for i in 0..<preemphasized.count {
            padded[centrePad + i] = preemphasized[i]
        }

        let nFrames = (padded.count - paddedFFT) / config.hopLength + 1
        let melLength = audio.count / config.hopLength  // NeMo: floor(samples / hop)

        let halfPadded = paddedFFT / 2

        // STFT + power spectrum
        var powerSpec = [Float](repeating: 0, count: nFrames * nBins)
        var splitReal = [Float](repeating: 0, count: halfPadded)
        var splitImag = [Float](repeating: 0, count: halfPadded)
        var paddedFrame = [Float](repeating: 0, count: paddedFFT)

        for frame in 0..<nFrames {
            let start = frame * config.hopLength

            // A 400-sample window sits CENTRED in the 512-point frame, which
            // is where torch.stft puts it: 56 zeros, the windowed samples, 56
            // zeros. Windowing from index 0 instead shifts every frame.
            for i in 0..<paddedFFT { paddedFrame[i] = 0 }
            padded.withUnsafeBufferPointer { buf in
                paddedFrame.withUnsafeMutableBufferPointer { dst in
                    vDSP_vmul(buf.baseAddress! + start + windowOffset, 1,
                              hannWindow, 1,
                              dst.baseAddress! + windowOffset, 1,
                              vDSP_Length(config.winLength))
                }
            }

            // Pack into split-complex format for vDSP
            for i in 0..<halfPadded {
                splitReal[i] = paddedFrame[2 * i]
                splitImag[i] = paddedFrame[2 * i + 1]
            }

            // FFT
            splitReal.withUnsafeMutableBufferPointer { realBuf in
                splitImag.withUnsafeMutableBufferPointer { imagBuf in
                    var splitComplex = DSPSplitComplex(
                        realp: realBuf.baseAddress!,
                        imagp: imagBuf.baseAddress!)
                    vDSP_fft_zrip(fftSetup, &splitComplex, 1,
                                  log2PaddedFFT, FFTDirection(kFFTDirection_Forward))
                }
            }

            // Power spectrum: |X[k]|^2 = real^2 + imag^2, scaled by 1/4 because
            // vDSP_fft_zrip returns 2x the DFT that torch.stft returns.
            let base = frame * nBins
            // DC component (packed in realp[0])
            powerSpec[base] = splitReal[0] * splitReal[0] * Self.powerCorrection
            // Nyquist component (packed in imagp[0])
            powerSpec[base + halfPadded] = splitImag[0] * splitImag[0] * Self.powerCorrection
            // Other bins
            for k in 1..<halfPadded {
                powerSpec[base + k] = (splitReal[k] * splitReal[k] + splitImag[k] * splitImag[k]) * Self.powerCorrection
            }
        }

        // Mel filterbank: [nMels, nBins] × [nBins, nFrames] → [nMels, nFrames]
        // powerSpec is [nFrames, nBins], need to transpose for matmul
        var powerSpecT = [Float](repeating: 0, count: nBins * nFrames)
        vDSP_mtrans(powerSpec, 1, &powerSpecT, 1,
                    vDSP_Length(nBins), vDSP_Length(nFrames))

        var melSpec = [Float](repeating: 0, count: config.numMelBins * nFrames)
        vDSP_mmul(melFilterbank, 1, powerSpecT, 1, &melSpec, 1,
                  vDSP_Length(config.numMelBins), vDSP_Length(nFrames), vDSP_Length(nBins))

        // Log: log(mel + guard)
        let count = melSpec.count
        var guardVal = logGuard
        vDSP_vsadd(melSpec, 1, &guardVal, &melSpec, 1, vDSP_Length(count))
        var countN = Int32(count)
        vvlogf(&melSpec, melSpec, &countN)

        // Match the encoder's context: a later window must not inherit statistics
        // dominated by louder speech or silence elsewhere in the recording.
        Self.normalizeLogMel(
            &melSpec, bins: config.numMelBins, bufferFrames: nFrames,
            validFrames: melLength, windowLength: normalizationWindowLength ?? max(melLength, 1))

        // Create MLMultiArray [1, numMelBins, nFrames] in float16
        let mel = try MLMultiArray(
            shape: [1, config.numMelBins as NSNumber, nFrames as NSNumber],
            dataType: .float16)
        let melPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)

        // melSpec is [numMelBins, nFrames] row-major, same as MLMultiArray [1, 128, T]
        for i in 0..<(config.numMelBins * nFrames) {
            melPtr[i] = Float16(melSpec[i])
        }

        return (mel, melLength)
    }

    /// NeMo per-feature normalization over the valid frames of each window.
    /// The centered STFT can emit an extra frame, which remains zero padding.
    static func normalizeLogMel(
        _ features: inout [Float], bins: Int, bufferFrames: Int,
        validFrames: Int, windowLength: Int
    ) {
        precondition(windowLength > 0 && validFrames >= 0 && validFrames <= bufferFrames)
        precondition(features.count == bins * bufferFrames)
        features.withUnsafeMutableBufferPointer { buffer in
            for bin in 0..<bins {
                let row = buffer.baseAddress! + bin * bufferFrames
                for start in stride(from: 0, to: validFrames, by: windowLength) {
                    let count = min(windowLength, validFrames - start)
                    let values = row + start
                    // A one-frame tail has no sample variance. NeMo replaces its
                    // undefined standard deviation with zero; centering yields zero.
                    let first = values.pointee
                    let constant = UnsafeBufferPointer(start: values, count: count).allSatisfy { $0 == first }
                    if count == 1 || constant {
                        // Avoid magnifying mean-rounding error in silent windows.
                        values.update(repeating: 0, count: count)
                        continue
                    }
                    var mean: Float = 0
                    vDSP_meanv(values, 1, &mean, vDSP_Length(count))
                    var negativeMean = -mean
                    vDSP_vsadd(values, 1, &negativeMean, values, 1, vDSP_Length(count))
                    var meanSquare: Float = 0
                    vDSP_measqv(values, 1, &meanSquare, vDSP_Length(count))
                    let standardDeviation = sqrt(Float(count) * meanSquare / Float(count - 1))
                    var inverseDeviation = 1 / (standardDeviation + 1e-5)
                    vDSP_vsmul(values, 1, &inverseDeviation, values, 1, vDSP_Length(count))
                }
                row.advanced(by: validFrames).update(repeating: 0, count: bufferFrames - validFrames)
            }
        }
    }

    /// Remove STFT padding by retaining the valid prefix, before encoder shape selection.
    static func validPrefix(_ mel: MLMultiArray, length: Int) throws -> MLMultiArray {
        try slice(mel, start: 0, length: length)
    }

    /// Copy a valid frame range using the source's bin-major buffer stride.
    static func slice(_ mel: MLMultiArray, start: Int, length: Int) throws -> MLMultiArray {
        let frames = mel.shape[2].intValue
        precondition(start >= 0 && length > 0 && start + length <= frames && mel.dataType == .float16)
        if start == 0 && length == frames { return mel }
        let bins = mel.shape[1].intValue
        let result = try MLMultiArray(shape: [1, bins as NSNumber, length as NSNumber], dataType: .float16)
        let source = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let destination = result.dataPointer.assumingMemoryBound(to: Float16.self)
        for bin in 0..<bins {
            destination.advanced(by: bin * length).update(from: source.advanced(by: bin * frames + start), count: length)
        }
        return result
    }

    // MARK: - Mel Filterbank Construction

    /// Build Slaney-normalized mel filterbank matrix.
    private static func buildMelFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, paddedFFT: Int
    ) -> [Float] {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0

        // Slaney mel scale (piecewise: linear <1000 Hz, log ≥1000 Hz)
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)
        let logstepMelToHz: Float = log(6.4) / 27.0

        func hzToMel(_ hz: Float) -> Float {
            hz < minLogHertz ? 3.0 * hz / 200.0 : minLogMel + log(hz / minLogHertz) * logstepHzToMel
        }

        func melToHz(_ mel: Float) -> Float {
            mel < minLogMel ? 200.0 * mel / 3.0 : minLogHertz * exp((mel - minLogMel) * logstepMelToHz)
        }

        // FFT bin frequencies
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(paddedFFT)
        }

        // Mel-spaced center frequencies
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }
        let filterFreqs = melPoints.map { melToHz($0) }

        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        // Build filterbank [nBins, nMels] then transpose
        var filterbank = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = fftFreqs[bin]
            for mel in 0..<nMels {
                let downSlope = (freq - filterFreqs[mel]) / filterDiff[mel]
                let upSlope = (filterFreqs[mel + 2] - freq) / filterDiff[mel + 1]
                filterbank[bin * nMels + mel] = max(0.0, min(downSlope, upSlope))
            }
        }

        // Slaney normalization
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Transpose to [nMels, nBins]
        var transposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                transposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        return transposed
    }
}
