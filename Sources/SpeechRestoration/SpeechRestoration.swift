import Foundation
import CoreML
import AudioCommon

/// Speech restoration (joint denoise + dereverb) with **Sidon** on Core ML.
///
/// Sidon (sarulab-speech, ICASSP 2026; arXiv 2509.17052) restores degraded
/// speech while **preserving speaker identity**, which makes it well suited to
/// cleaning a noisy/reverberant **voice-cloning reference** before TTS. The
/// pipeline is two CoreML graphs plus a DSP front-end:
///
/// ```
/// audio(16k) ──[SeamlessM4T log-mel front-end, DSP]──► input_features[1,T,160]
///   predictor (w2v-BERT 2.0, 8 layers + merged LoRA) ─► features[1,T,1024]
///   vocoder   (DAC decoder, rates [8,5,4,3,2], ×960)  ─► audio(48k)[1,M]
/// ```
///
/// The CoreML graphs are exported at a **fixed** sequence length
/// (`SidonConfig.frames` ≈ 10 s), so longer inputs are processed window by
/// window and concatenated. Output is 48 kHz mono.
///
/// ```swift
/// let restorer = try await SpeechRestorer.fromPretrained()
/// let clean = try restorer.restore(audio: noisy, sampleRate: 16_000)  // → 48 kHz
/// ```
///
/// This is an **opt-in** capability — it changes the audio materially and (per
/// the upstream caveat) shifts speaker similarity slightly, so it never runs
/// implicitly.
public final class SpeechRestorer {

    /// Default HuggingFace repo id (provisional — parameterizable).
    public static let defaultModelId = SidonConfig.defaultModelId
    /// Native output sample rate (DAC vocoder).
    public static let outputSampleRate = 48_000
    /// Native input sample rate (w2v-BERT front-end).
    public static let inputSampleRate = 16_000

    let config: SidonConfig
    var predictor: SidonPredictorModel?
    var vocoder: SidonVocoderModel?
    let variant: SidonVariant
    var _isLoaded = true

    init(
        predictor: SidonPredictorModel,
        vocoder: SidonVocoderModel,
        config: SidonConfig,
        variant: SidonVariant
    ) {
        self.predictor = predictor
        self.vocoder = vocoder
        self.config = config
        self.variant = variant
    }

    // MARK: - Loading

    /// Load a pretrained Sidon model. Downloads on first use, then caches.
    ///
    /// - Parameters:
    ///   - variant: precision variant (`.fp16` default, `.int8` for smaller).
    ///   - modelId: HuggingFace repo id (override if the published id differs).
    ///   - computeUnits: CoreML compute units (default `.all`; honors the
    ///     `SPEECH_COREML_COMPUTE_UNITS` env override via `CoreMLLoader`).
    public static func fromPretrained(
        variant: SidonVariant = .fp16,
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SpeechRestorer {
        progressHandler?(0.0, "Downloading model...")

        let paths = try await SidonDownloader.ensureDownloaded(
            modelId: modelId,
            variant: variant,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.8, "Downloading model...") }
        )

        progressHandler?(0.85, "Loading models...")

        let predictorURL = try SidonModelResolver.resolve(
            directory: paths.bundleDir,
            compiledName: SidonConfig.predictorCompiledName,
            packageName: SidonConfig.predictorPackageName)
        let vocoderURL = try SidonModelResolver.resolve(
            directory: paths.bundleDir,
            compiledName: SidonConfig.vocoderCompiledName,
            packageName: SidonConfig.vocoderPackageName)

        let predictor = try SidonPredictorModel(modelURL: predictorURL, computeUnits: computeUnits)
        let vocoder = try SidonVocoderModel(modelURL: vocoderURL, computeUnits: computeUnits)

        progressHandler?(1.0, "Ready")
        return SpeechRestorer(
            predictor: predictor, vocoder: vocoder,
            config: .default, variant: variant)
    }

    /// Load the two models directly from a local directory holding
    /// `Sidon-Predictor.{mlmodelc,mlpackage}` + `Sidon-Vocoder.{mlmodelc,mlpackage}`,
    /// skipping HuggingFace entirely. Useful for testing locally-converted
    /// artifacts (mirrors the other engines' local-bundle affordance).
    public static func fromLocalBundle(
        directory: URL,
        variant: SidonVariant = .fp16,
        computeUnits: MLComputeUnits = .all
    ) throws -> SpeechRestorer {
        let predictorURL = try SidonModelResolver.resolve(
            directory: directory,
            compiledName: SidonConfig.predictorCompiledName,
            packageName: SidonConfig.predictorPackageName)
        let vocoderURL = try SidonModelResolver.resolve(
            directory: directory,
            compiledName: SidonConfig.vocoderCompiledName,
            packageName: SidonConfig.vocoderPackageName)
        let predictor = try SidonPredictorModel(modelURL: predictorURL, computeUnits: computeUnits)
        let vocoder = try SidonVocoderModel(modelURL: vocoderURL, computeUnits: computeUnits)
        return SpeechRestorer(
            predictor: predictor, vocoder: vocoder,
            config: .default, variant: variant)
    }

    // MARK: - Inference

    /// Restore arbitrary-length audio. Input is resampled to 16 kHz; output is
    /// 48 kHz mono. Longer-than-window inputs are processed in fixed windows and
    /// concatenated; the result is trimmed to the input's true duration
    /// (rescaled to the 48 kHz timeline).
    public func restore(audio: [Float], sampleRate: Int) throws -> [Float] {
        guard predictor != nil, vocoder != nil else {
            throw SidonModelError.predictionFailed("model unloaded")
        }
        var samples = audio
        if sampleRate != config.inputSampleRate {
            samples = AudioFileLoader.resample(
                audio, from: sampleRate, to: config.inputSampleRate)
        }
        guard !samples.isEmpty else { return [] }

        let win = config.windowSamples
        let nWindows = (samples.count + win - 1) / win
        var out = [Float]()
        out.reserveCapacity(nWindows * config.outputSamplesPerWindow)

        for w in 0..<nWindows {
            let start = w * win
            let end = Swift.min(start + win, samples.count)
            let chunk = Array(samples[start..<end])
            let restored = try restoreWindow(samples: chunk)
            out.append(contentsOf: restored)
        }

        // Trim to the input's true duration mapped onto the 48 kHz timeline.
        // (Each fixed window emits a fixed number of output samples regardless
        // of how much of it was real audio, so a partial final window is padded
        // with vocoded silence we don't want to keep.)
        let ratio = Double(config.outputSampleRate) / Double(config.inputSampleRate)
        let targetLen = Int((Double(samples.count) * ratio).rounded())
        if out.count > targetLen { out.removeLast(out.count - targetLen) }
        return out
    }

    /// Restore a single window (≤ `windowSamples` 16 kHz samples). Shorter
    /// inputs are zero-padded to the fixed window so the front-end yields exactly
    /// `frames` stacked frames.
    public func restoreWindow(samples: [Float]) throws -> [Float] {
        guard let predictor, let vocoder else {
            throw SidonModelError.predictionFailed("model unloaded")
        }

        // Pad/clip to exactly windowSamples so the fixed-shape graphs line up.
        var window = samples
        if window.count < config.windowSamples {
            window.append(contentsOf:
                [Float](repeating: 0, count: config.windowSamples - window.count))
        } else if window.count > config.windowSamples {
            window = Array(window.prefix(config.windowSamples))
        }

        // 1. Front-end → input_features[frames, 160].
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: window)
        // The fixed window is sized to yield exactly config.frames; guard anyway.
        let usableFrames = Swift.min(frames, config.frames)
        guard usableFrames > 0 else { return [] }

        let inputArray = try MLMultiArray(
            shape: [1, config.frames as NSNumber, config.featureDim as NSNumber],
            dataType: .float32)
        let inPtr = inputArray.dataPointer.assumingMemoryBound(to: Float.self)
        let copyCount = usableFrames * config.featureDim
        feats.withUnsafeBufferPointer { src in
            inPtr.update(from: src.baseAddress!, count: copyCount)
        }
        // Zero any frames beyond what the front-end produced (defensive; the
        // window is sized so this is normally a no-op).
        if copyCount < config.frames * config.featureDim {
            for i in copyCount..<(config.frames * config.featureDim) { inPtr[i] = 0 }
        }

        // 2. Predictor → features[1, frames, 1024].
        let features = try predictor.predict(inputFeatures: inputArray)

        // 3. Vocoder → audio[1, M].
        let audioArray = try vocoder.predict(features: features)

        // Extract to [Float] (CoreML may emit FP16).
        let count = audioArray.count
        var result = [Float](repeating: 0, count: count)
        extractFlat(audioArray, into: &result, count: count)
        return result
    }

    /// Extract float32 data from an MLMultiArray, handling FP16 output.
    private func extractFlat(_ array: MLMultiArray, into output: inout [Float], count: Int) {
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { output[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            output.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
    }
}
