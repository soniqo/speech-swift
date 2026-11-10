import Foundation
import Accelerate
import MLX
import AudioCommon

/// MLX-free container for mel spectrogram data.
/// Layout: `data` is a flat [Float] in row-major [melBins, timeFrames] order.
public struct MelFeatures {
    public let data: [Float]
    public let melBins: Int
    public let timeFrames: Int

    public init(data: [Float], melBins: Int, timeFrames: Int) {
        self.data = data
        self.melBins = melBins
        self.timeFrames = timeFrames
    }
}

/// Whisper-style feature extractor for Qwen3-ASR
/// Converts raw audio to mel spectrograms
/// Parameters from HuggingFace preprocessor_config.json
public class WhisperFeatureExtractor {
    public let sampleRate: Int = 16000      // HF WhisperFeatureExtractor uses 16kHz
    public let nFFT: Int = 400              // FFT size (from config)
    public let hopLength: Int = 160         // 10ms hop at 16kHz (from config)
    public let nMels: Int = 128             // Mel filterbank bins (feature_size)
    public let chunkLength: Int = 30        // Max audio chunk in seconds

    private var melFilterbank: [Float]?

    // Windowed 400-point DFT as two [nFFT, nBins] matrices (Hann window folded in), so each
    // frame's spectrum is exactly HuggingFace's `np.fft.rfft(frame * window, n=400)`.
    // vDSP has no 400-point FFT, and zero-padding to 512 changes both the bin grid and the
    // power (vDSP's real FFT also returns 2x the DFT), which shifted every log-mel value.
    private var dftCos: [Float]
    private var dftSin: [Float]

    public init() {
        // Periodic Hann window, as HuggingFace's window_function(400, "hann")
        let nBins = nFFT / 2 + 1
        var cosTable = [Float](repeating: 0, count: nFFT * nBins)
        var sinTable = [Float](repeating: 0, count: nFFT * nBins)
        for n in 0..<nFFT {
            let window = 0.5 * (1.0 - cos(2.0 * Double.pi * Double(n) / Double(nFFT)))
            for k in 0..<nBins {
                let angle = 2.0 * Double.pi * Double(k * n % nFFT) / Double(nFFT)
                cosTable[n * nBins + k] = Float(window * cos(angle))
                sinTable[n * nBins + k] = Float(window * sin(angle))
            }
        }
        dftCos = cosTable
        dftSin = sinTable
        setupMelFilterbank()
    }

    /// Setup mel filterbank matrix with slaney normalization
    /// Matches HuggingFace transformers.audio_utils.mel_filter_bank exactly
    private func setupMelFilterbank() {
        let fMin: Float = 0.0
        let fMax: Float = Float(sampleRate) / 2.0  // Nyquist frequency (8000 Hz for 16kHz)

        // Slaney mel scale conversion functions (HuggingFace style)
        // This is a piecewise function: linear below 1000 Hz, logarithmic above
        let minLogHertz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logstepHzToMel: Float = 27.0 / log(6.4)  // For Hz->Mel
        let logstepMelToHz: Float = log(6.4) / 27.0  // For Mel->Hz

        func hzToMel(_ hz: Float) -> Float {
            if hz < minLogHertz {
                return 3.0 * hz / 200.0  // Linear region
            } else {
                return minLogMel + log(hz / minLogHertz) * logstepHzToMel  // Log region
            }
        }

        func melToHz(_ mel: Float) -> Float {
            if mel < minLogMel {
                return 200.0 * mel / 3.0  // Linear region
            } else {
                return minLogHertz * exp((mel - minLogMel) * logstepMelToHz)  // Exp region
            }
        }

        let nBins = nFFT / 2 + 1  // 201

        // FFT bin frequencies: k * fs / nFFT
        var fftFreqs = [Float](repeating: 0, count: nBins)
        for i in 0..<nBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        // Create mel filter center frequencies
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // nMels + 2 points for triangular filters (includes low and high edges)
        let nMelPoints = nMels + 2
        var melPoints = [Float](repeating: 0, count: nMelPoints)
        for i in 0..<nMelPoints {
            melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMelPoints - 1)
        }

        // Convert mel points to Hz - these are the filter edge frequencies
        let filterFreqs = melPoints.map { melToHz($0) }

        // Calculate filter frequency differences for normalization
        var filterDiff = [Float](repeating: 0, count: nMelPoints - 1)
        for i in 0..<(nMelPoints - 1) {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i]
        }

        // Create filterbank using HuggingFace's _create_triangular_filter_bank approach
        // This creates smooth triangular filters in frequency space
        // Output shape: [nBins, nMels] - we'll transpose later for our use
        var filterbank = [Float](repeating: 0, count: nBins * nMels)

        for bin in 0..<nBins {
            let fftFreq = fftFreqs[bin]

            for mel in 0..<nMels {
                // Filter edges: filterFreqs[mel], filterFreqs[mel+1], filterFreqs[mel+2]
                let lowFreq = filterFreqs[mel]
                let centerFreq = filterFreqs[mel + 1]
                let highFreq = filterFreqs[mel + 2]

                // Calculate slopes (HuggingFace formula)
                // slopes = filter_freqs - fft_freqs (broadcast)
                // down_slopes = -slopes[:, :-2] / filter_diff[:-1]
                // up_slopes = slopes[:, 2:] / filter_diff[1:]

                let downSlope = (fftFreq - lowFreq) / filterDiff[mel]     // Rising edge
                let upSlope = (highFreq - fftFreq) / filterDiff[mel + 1]  // Falling edge

                // Triangular filter: max(0, min(down_slope, up_slope))
                let filterValue = max(0.0, min(downSlope, upSlope))

                // Store in [nBins, nMels] layout
                filterbank[bin * nMels + mel] = filterValue
            }
        }

        // Apply slaney normalization: 2.0 / (high_freq - low_freq) for each mel filter
        for mel in 0..<nMels {
            let enorm = 2.0 / (filterFreqs[mel + 2] - filterFreqs[mel])
            for bin in 0..<nBins {
                filterbank[bin * nMels + mel] *= enorm
            }
        }

        // Transpose to [nMels, nBins] for our matrix multiplication
        var filterbankTransposed = [Float](repeating: 0, count: nMels * nBins)
        for mel in 0..<nMels {
            for bin in 0..<nBins {
                filterbankTransposed[mel * nBins + bin] = filterbank[bin * nMels + mel]
            }
        }

        self.melFilterbank = filterbankTransposed
    }

    /// Power spectrum of every frame, [nFrames, nFFT / 2 + 1] row-major: |rfft(frame * window)|^2.
    private func powerSpectrum(_ paddedAudio: [Float], nFrames: Int) -> [Float] {
        let nBins = nFFT / 2 + 1
        // Bound scratch storage for long recordings; only the returned spectrum
        // grows with duration. Reuse the same frame/DFT buffers for each batch.
        let batchCapacity = min(nFrames, 256)
        var frames = [Float](repeating: 0, count: batchCapacity * nFFT)
        var real = [Float](repeating: 0, count: batchCapacity * nBins)
        var imag = [Float](repeating: 0, count: batchCapacity * nBins)
        var power = [Float](repeating: 0, count: nFrames * nBins)

        for firstFrame in stride(from: 0, to: nFrames, by: batchCapacity) {
            let batchFrames = min(batchCapacity, nFrames - firstFrame)
            paddedAudio.withUnsafeBufferPointer { audio in
                frames.withUnsafeMutableBufferPointer { rows in
                    for frame in 0..<batchFrames {
                        (rows.baseAddress! + frame * nFFT).update(
                            from: audio.baseAddress! + (firstFrame + frame) * hopLength,
                            count: nFFT)
                    }
                }
            }
            vDSP_mmul(frames, 1, dftCos, 1, &real, 1,
                      vDSP_Length(batchFrames), vDSP_Length(nBins), vDSP_Length(nFFT))
            vDSP_mmul(frames, 1, dftSin, 1, &imag, 1,
                      vDSP_Length(batchFrames), vDSP_Length(nBins), vDSP_Length(nFFT))
            power.withUnsafeMutableBufferPointer { output in
                vDSP_vmma(real, 1, real, 1, imag, 1, imag, 1,
                          output.baseAddress! + firstFrame * nBins, 1,
                          vDSP_Length(batchFrames * nBins))
            }
        }
        return power
    }

    /// Extract mel spectrogram features from audio samples
    /// - Parameter audio: Raw audio samples (Float array, mono, at sampleRate)
    /// - Returns: Mel spectrogram [mel_bins, time_frames]
    public func extractFeatures(_ audio: [Float]) -> MLXArray {
        let nBins = nFFT / 2 + 1  // 201

        // Pad audio with reflect padding (like Whisper/librosa)
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        // Reflect pad left side
        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }

        // Copy original audio
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }

        // Reflect pad right side
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        // Calculate number of frames
        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        let magnitude = powerSpectrum(paddedAudio, nFrames: nFrames)

        // --- BLAS sgemm for mel filterbank ---
        guard let filterbank = melFilterbank else {
            fatalError("Mel filterbank not initialized")
        }

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)

        // melSpec[nFrames, nMels] = magnitude[nFrames, nBins] * filterbankT[nBins, nMels]
        // filterbank is [nMels, nBins]. We need A * B^T.
        // vDSP_mmul computes C = A * B, so we need to pre-transpose filterbank.
        // Transpose filterbank [nMels, nBins] -> filterbankT [nBins, nMels]
        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(filterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        // C[nFrames, nMels] = A[nFrames, nBins] * B[nBins, nMels]
        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // --- Vectorized log10, clamp, normalize ---
        let count = melSpec.count
        var countN = Int32(count)

        // Clamp minimum to epsilon before log
        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))

        // log10 using vForce
        vvlog10f(&melSpec, melSpec, &countN)

        // Find max value for dynamic range compression
        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))

        // Clamp minimum to max - 8.0
        var minClamp = maxVal - 8.0
        var maxClamp = Float.greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &minClamp, &maxClamp, &melSpec, 1, vDSP_Length(count))

        // Normalize: (x + 4.0) / 4.0 = x * 0.25 + 1.0
        var scale: Float = 0.25
        var offset: Float = 1.0
        vDSP_vsmsa(melSpec, 1, &scale, &offset, &melSpec, 1, vDSP_Length(count))

        // CRITICAL: HuggingFace WhisperFeatureExtractor removes the last frame: log_spec[:, :-1]
        let trimmedFrames = nFrames - 1
        let trimmedMelSpec = Array(melSpec.prefix(trimmedFrames * nMels))

        // Qwen3-ASR encoder handles arbitrary-length audio via windowed attention.
        // The chunkLength=30 from preprocessor_config.json is inherited from the HuggingFace
        // WhisperFeatureExtractor class but is not enforced by the official Qwen3-ASR pipeline.
        // Cap at 1200 seconds (120000 frames) to match the official pipeline's upper bound
        // and prevent OOM on extremely long inputs.
        let maxFrames = 1200 * sampleRate / hopLength  // 120000 frames at 16kHz/160hop
        let finalFrames: Int
        let finalMelSpec: [Float]
        if trimmedFrames > maxFrames {
            finalFrames = maxFrames
            finalMelSpec = Array(trimmedMelSpec.prefix(maxFrames * nMels))
        } else {
            finalFrames = trimmedFrames
            finalMelSpec = trimmedMelSpec
        }

        let array = MLXArray(finalMelSpec, [finalFrames, nMels])
        return array.transposed(1, 0)  // [mel_bins, time_frames]
    }

    /// Process audio for Qwen3-ASR model
    /// - Parameter audio: Raw audio samples (any sample rate)
    /// - Parameter inputSampleRate: Sample rate of input audio
    /// - Returns: Preprocessed mel features ready for the model
    public func process(_ audio: [Float], sampleRate inputSampleRate: Int) -> MLXArray {
        var processedAudio = audio

        // Resample if needed
        if inputSampleRate != sampleRate {
            processedAudio = AudioFileLoader.resample(audio, from: inputSampleRate, to: sampleRate)
        }

        // NOTE: HuggingFace WhisperFeatureExtractor does NOT normalize audio amplitude
        // The model expects raw audio values (typically in [-1, 1] range from int16 conversion)
        // Do NOT divide by max absolute value!

        // Extract features
        return extractFeatures(processedAudio)
    }

    // MARK: - MLX-free variants (for CoreML-only path)

    /// Extract mel spectrogram features without MLXArray dependency.
    /// Produces identical output to `extractFeatures(_:)` but returns a plain `MelFeatures`
    /// struct with layout `[melBins, timeFrames]` (transposed, same as the MLXArray version).
    ///
    /// All mel computation uses the same Accelerate/vDSP pipeline; only the final
    /// transpose is done with a pure-Swift loop instead of `MLXArray.transposed`.
    public func extractFeaturesRaw(_ audio: [Float]) -> MelFeatures {
        let nBins = nFFT / 2 + 1  // 201

        // Pad audio with reflect padding (like Whisper/librosa)
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: padLength + audio.count + padLength)

        // Reflect pad left side
        for i in 0..<padLength {
            let srcIdx = min(padLength - i, audio.count - 1)
            paddedAudio[i] = audio[max(0, srcIdx)]
        }

        // Copy original audio
        for i in 0..<audio.count {
            paddedAudio[padLength + i] = audio[i]
        }

        // Reflect pad right side
        for i in 0..<padLength {
            let srcIdx = audio.count - 2 - i
            paddedAudio[padLength + audio.count + i] = audio[max(0, srcIdx)]
        }

        // Calculate number of frames
        let nFrames = (paddedAudio.count - nFFT) / hopLength + 1

        let magnitude = powerSpectrum(paddedAudio, nFrames: nFrames)

        // --- BLAS sgemm for mel filterbank ---
        guard let filterbank = melFilterbank else {
            fatalError("Mel filterbank not initialized")
        }

        var melSpec = [Float](repeating: 0, count: nFrames * nMels)

        var filterbankT = [Float](repeating: 0, count: nBins * nMels)
        vDSP_mtrans(filterbank, 1, &filterbankT, 1, vDSP_Length(nBins), vDSP_Length(nMels))

        vDSP_mmul(magnitude, 1, filterbankT, 1, &melSpec, 1,
                  vDSP_Length(nFrames), vDSP_Length(nMels), vDSP_Length(nBins))

        // --- Vectorized log10, clamp, normalize ---
        let count = melSpec.count
        var countN = Int32(count)

        var epsilon: Float = 1e-10
        vDSP_vclip(melSpec, 1, &epsilon, [Float.greatestFiniteMagnitude], &melSpec, 1, vDSP_Length(count))

        vvlog10f(&melSpec, melSpec, &countN)

        var maxVal: Float = -Float.infinity
        vDSP_maxv(melSpec, 1, &maxVal, vDSP_Length(count))

        var minClamp = maxVal - 8.0
        var maxClamp = Float.greatestFiniteMagnitude
        vDSP_vclip(melSpec, 1, &minClamp, &maxClamp, &melSpec, 1, vDSP_Length(count))

        var scale: Float = 0.25
        var offset: Float = 1.0
        vDSP_vsmsa(melSpec, 1, &scale, &offset, &melSpec, 1, vDSP_Length(count))

        // CRITICAL: HuggingFace WhisperFeatureExtractor removes the last frame
        let trimmedFrames = nFrames - 1
        let trimmedMelSpec = Array(melSpec.prefix(trimmedFrames * nMels))

        let maxFrames = 1200 * sampleRate / hopLength  // 120000 frames
        let finalFrames: Int
        let finalMelSpec: [Float]
        if trimmedFrames > maxFrames {
            finalFrames = maxFrames
            finalMelSpec = Array(trimmedMelSpec.prefix(maxFrames * nMels))
        } else {
            finalFrames = trimmedFrames
            finalMelSpec = trimmedMelSpec
        }

        // Transpose [timeFrames, melBins] -> [melBins, timeFrames] in pure Swift
        var transposed = [Float](repeating: 0, count: finalFrames * nMels)
        for t in 0..<finalFrames {
            for m in 0..<nMels {
                transposed[m * finalFrames + t] = finalMelSpec[t * nMels + m]
            }
        }
        return MelFeatures(data: transposed, melBins: nMels, timeFrames: finalFrames)
    }

    /// Process audio for Qwen3-ASR without MLXArray dependency.
    /// MLX-free equivalent of `process(_:sampleRate:)`.
    /// - Parameter audio: Raw audio samples (any sample rate)
    /// - Parameter inputSampleRate: Sample rate of input audio
    /// - Returns: `MelFeatures` with layout `[melBins, timeFrames]`
    public func processRaw(_ audio: [Float], sampleRate inputSampleRate: Int) -> MelFeatures {
        var processedAudio = audio

        // Resample if needed
        if inputSampleRate != sampleRate {
            processedAudio = AudioFileLoader.resample(audio, from: inputSampleRate, to: sampleRate)
        }

        // NOTE: HuggingFace WhisperFeatureExtractor does NOT normalize audio amplitude
        // The model expects raw audio values (typically in [-1, 1] range from int16 conversion)
        // Do NOT divide by max absolute value!

        return extractFeaturesRaw(processedAudio)
    }
}
