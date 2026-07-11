import CoreML
import Foundation
import AudioCommon

/// Parakeet TDT 0.6B v3 — CoreML-based automatic speech recognition.
///
/// Uses a FastConformer encoder with a Token-and-Duration Transducer (TDT) decoder.
/// Mel preprocessing is done in Swift using Accelerate/vDSP. The encoder, decoder, and
/// joint network run on CoreML with INT8-quantized encoder for Neural Engine acceleration.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class ParakeetASRModel {
    /// Model configuration.
    public let config: ParakeetConfig

    /// Default HuggingFace model ID: INT8 encoder, single fixed 3000-frame
    /// shape (30s max). Single fixed shape (no EnumeratedShapes) so it loads
    /// on any CoreML compute unit, including `.cpuOnly`. Audio longer than 30s
    /// is window-chunked in `transcribeAudio`.
    public static let defaultModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s"

    /// iOS-optimized model: single 500-frame shape (5s max), lower runtime
    /// memory than the 30s default. Long audio is window-chunked.
    public static let iosModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT8-iOS-5s"

    /// Whether the model is loaded and ready for inference.
    var _isLoaded = true

    private let melPreprocessor: MelPreprocessor
    var encoder: MLModel?
    var decoder: MLModel?
    var joint: MLModel?
    private let vocabulary: ParakeetVocabulary
    /// Mel frame counts the loaded encoder accepts, sorted ascending.
    /// Derived from the encoder's CoreML input constraint at load time. Our
    /// shipped models are single fixed shapes (default = `[3000]`, iOS-5s =
    /// `[500]`); legacy enumerated-shape exports also work without any
    /// per-variant Swift logic.
    /// Internal access so tests can assert discovery results.
    let supportedMelLengths: [Int]
    /// Compute units the encoder ended up using (after the prefer-ANE,
    /// fall-back-to-GPU loader). Surfaced so the demo can show which
    /// hardware backend is active.
    public let encoderComputeUnits: MLComputeUnits
    /// Confidence from the last transcription (0.0–1.0).
    public private(set) var lastConfidence: Float = 0
    /// Per-word confidence scores from the last transcription.
    public private(set) var lastWordConfidences: [WordConfidence]?

    private init(
        config: ParakeetConfig,
        encoder: MLModel?,
        decoder: MLModel?,
        joint: MLModel?,
        vocabulary: ParakeetVocabulary,
        supportedMelLengths: [Int],
        encoderComputeUnits: MLComputeUnits
    ) {
        self.config = config
        self.melPreprocessor = MelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
        self.supportedMelLengths = supportedMelLengths
        self.encoderComputeUnits = encoderComputeUnits
    }

    // MARK: - Warmup

    /// Warm up CoreML models by running a dummy inference.
    ///
    /// CoreML compiles computation graphs on the first `prediction()` call, which causes
    /// a ~4x latency penalty on cold inference. Calling `warmUp()` triggers this compilation
    /// on a minimal 1-second silence input so that subsequent real transcriptions run at full speed.
    public func warmUp() throws {
        let dummyAudio = [Float](repeating: 0, count: config.sampleRate)  // 1s silence
        _ = try transcribeAudio(dummyAudio, sampleRate: config.sampleRate)
    }

    // MARK: - Transcription

    /// Transcribe audio to text.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: Sample rate of the input audio in Hz
    ///   - language: Language steering — a comma-separated shortlist of ISO codes ("en", or
    ///     "en,ru" for bilingual). The decoder is masked so it can only emit these languages'
    ///     tags. Empty/nil (default) lets the model auto-detect natively.
    /// - Returns: Transcribed text
    /// - Throws: `AudioModelError` on CoreML inference failure
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        // Resample to 16kHz if needed
        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }

        // Step 1: Mel spectrogram extraction (Swift/Accelerate)
        AudioLog.inference.debug("Parakeet: preprocessing \(samples.count) samples")
        let tMel0 = CFAbsoluteTimeGetCurrent()
        let (mel, melLength) = try melPreprocessor.extract(samples)
        let tMel1 = CFAbsoluteTimeGetCurrent()

        // Steps 2-3: encode + TDT greedy decode. Audio longer than the
        // encoder's largest supported window is split into consecutive
        // windows and the per-window token streams are concatenated. This
        // lets a fixed-shape export (e.g. the 500-frame iOS-5s model, which
        // has no EnumeratedShapes) transcribe arbitrarily long audio instead
        // of truncating to the tail.
        let maxWindow = supportedMelLengths.last ?? melLength
        // Language steering: a per-call language wins; otherwise fall back to the persistent
        // override. Empty/nil both mean auto-detect.
        let effectiveLanguage = Self.effectiveLanguageHint(
            perCall: language,
            override: languageOverride)
        let masked = vocabulary.maskedLanguageTokenIds(allowing: effectiveLanguage)
        let tInfer0 = CFAbsoluteTimeGetCurrent()
        var tokenIds: [Int] = []
        var tokenLogProbs: [Float] = []

        if melLength <= maxWindow {
            let r = try encodeAndDecodeWindow(mel: mel, actualLength: melLength, maskedTokenIds: masked)
            tokenIds = r.tokens; tokenLogProbs = r.tokenLogProbs
        } else {
            // The mel buffer is [1, 128, bufferFrames] where bufferFrames
            // (mel.shape[2]) can exceed the valid melLength — extract zero-pads
            // the tail. Slice with the buffer's real per-bin stride, but only
            // iterate over valid frames.
            let bufferFrames = mel.shape[2].intValue
            var start = 0
            var windows = 0
            while start < melLength {
                let win = min(maxWindow, melLength - start)
                let windowMel = try sliceMel(mel: mel, start: start, length: win, totalFrames: bufferFrames)
                let r = try encodeAndDecodeWindow(mel: windowMel, actualLength: win, maskedTokenIds: masked)
                tokenIds += r.tokens; tokenLogProbs += r.tokenLogProbs
                start += maxWindow; windows += 1
            }
            AudioLog.inference.debug(
                "Parakeet: split \(melLength) mel frames into \(windows) windows of \(maxWindow)")
        }
        let tInfer1 = CFAbsoluteTimeGetCurrent()

        // Step 4: Vocabulary decode + overall confidence. Compute confidence
        // from the emitted token log-probs (the decoder's own formula) rather
        // than averaging per-window — otherwise silent windows in chunked long
        // audio would dilute it toward zero.
        let text = vocabulary.decode(tokenIds)
        lastConfidence = tokenLogProbs.isEmpty
            ? 0
            : min(1.0, exp(tokenLogProbs.reduce(0, +) / Float(tokenLogProbs.count)))
        lastWordConfidences = vocabulary.decodeWords(tokenIds, logProbs: tokenLogProbs)

        let melMs = (tMel1 - tMel0) * 1000
        let inferMs = (tInfer1 - tInfer0) * 1000
        AudioLog.inference.info("Parakeet: mel=\(String(format: "%.1f", melMs))ms infer=\(String(format: "%.1f", inferMs))ms (\(tokenIds.count) tokens, \(melLength) mel frames)")

        return text
    }

    /// Encode one mel window and run TDT greedy decode on it.
    private func encodeAndDecodeWindow(mel: MLMultiArray, actualLength: Int, maskedTokenIds: Set<Int> = [])
        throws -> (tokens: [Int], tokenLogProbs: [Float], confidence: Float)
    {
        let (paddedMel, effectiveLength) = try padMelToSupportedShape(mel: mel, actualLength: actualLength)
        let encoderOutput = try runEncoder(mel: paddedMel, length: effectiveLength)
        let encoded = encoderOutput.featureValue(for: "encoded")!.multiArrayValue!
        let encodedLength = encoderOutput.featureValue(for: "encoded_length")!.multiArrayValue![0].intValue
        let tdtDecoder = TDTGreedyDecoder(config: config, decoder: decoder!, joint: joint!, maskedTokenIds: maskedTokenIds)
        return try tdtDecoder.decode(encoded: encoded, encodedLength: encodedLength)
    }

    /// Persistent language steering applied when a per-call `language` isn't given. Accepts an
    /// ISO code / shortlist ("en", "en,ru"), or nil for auto-detect.
    public var languageOverride: String?

    static func effectiveLanguageHint(perCall: String?, override: String?) -> String? {
        let trimmed = perCall?.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed?.isEmpty == false ? trimmed : override
    }

    /// Copy frames `[start, start+length)` out of a `[1, 128, totalFrames]`
    /// mel into a fresh `[1, 128, length]` array (bin-major Float16 layout).
    private func sliceMel(mel: MLMultiArray, start: Int, length: Int, totalFrames: Int) throws -> MLMultiArray {
        let sub = try MLMultiArray(shape: [1, 128, length as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = sub.dataPointer.assumingMemoryBound(to: Float16.self)
        for bin in 0..<config.numMelBins {
            dstPtr.advanced(by: bin * length)
                .update(from: srcPtr.advanced(by: bin * totalFrames + start), count: length)
        }
        return sub
    }

    // MARK: - CoreML Inference Helpers

    /// Fallback used only if the encoder's input constraint can't be introspected.
    /// Real values come from `discoverSupportedMelLengths(from:)` at load time.
    private static let defaultMelLengths = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000]

    /// Pad / truncate mel spectrogram to the smallest supported shape that
    /// fits. Supported shapes come from the encoder's CoreML input constraint
    /// at load time (see `discoverSupportedMelLengths(from:)`), so fixed-shape
    /// exports work without forcing the caller to know about them.
    ///
    /// If the mel exceeds the encoder's max supported length, we keep the
    /// most recent frames (i.e., drop from the start). For voice pipelines
    /// this preserves the actual speech (which sits at the end of the
    /// utterance after pre-roll silence) and turns force-cut overflows into
    /// partial-but-useful transcriptions instead of hard errors.
    ///
    /// Returns `(padded, effectiveLength)` because truncation also reduces
    /// the masked length the encoder uses.
    private func padMelToSupportedShape(
        mel: MLMultiArray, actualLength: Int
    ) throws -> (padded: MLMultiArray, length: Int) {
        let melFrames = mel.shape[2].intValue
        let maxSupported = supportedMelLengths.last ?? 0

        // Truncation path: input exceeds the encoder's max supported size.
        if melFrames > maxSupported {
            AudioLog.inference.notice(
                "Parakeet: mel \(melFrames) > max \(maxSupported); truncating to last \(maxSupported) frames")
            return try truncateMel(mel: mel, melFrames: melFrames, targetLength: maxSupported)
        }

        // Find the smallest supported length >= melFrames
        guard let targetLength = supportedMelLengths.first(where: { $0 >= melFrames }) else {
            // Should be unreachable given the truncation guard above, but
            // keep the typed error for any future shape config that breaks
            // the invariant.
            throw AudioModelError.inferenceFailed(
                operation: "mel padding",
                reason: "No supported shape >= \(melFrames) frames in \(supportedMelLengths)")
        }

        if targetLength == melFrames {
            return (mel, actualLength)
        }

        // Create zero-padded mel array [1, 128, targetLength]
        let padded = try MLMultiArray(
            shape: [1, 128, targetLength as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = padded.dataPointer.assumingMemoryBound(to: Float16.self)

        let numMelBins = config.numMelBins
        for bin in 0..<numMelBins {
            let srcOffset = bin * melFrames
            let dstOffset = bin * targetLength
            dstPtr.advanced(by: dstOffset)
                .update(from: srcPtr.advanced(by: srcOffset), count: melFrames)
            dstPtr.advanced(by: dstOffset + melFrames)
                .update(repeating: Float16(0), count: targetLength - melFrames)
        }

        return (padded, actualLength)
    }

    /// Drop the leading `melFrames - targetLength` frames and copy the tail
    /// into a `[1, 128, targetLength]` array. Effective length passed to the
    /// encoder is clamped to `targetLength` so the model's mask covers the
    /// whole tensor (no zero padding region to ignore).
    private func truncateMel(
        mel: MLMultiArray, melFrames: Int, targetLength: Int
    ) throws -> (padded: MLMultiArray, length: Int) {
        let dropPrefix = melFrames - targetLength
        let truncated = try MLMultiArray(
            shape: [1, 128, targetLength as NSNumber], dataType: mel.dataType)
        let srcPtr = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let dstPtr = truncated.dataPointer.assumingMemoryBound(to: Float16.self)
        let numMelBins = config.numMelBins
        for bin in 0..<numMelBins {
            let srcOffset = bin * melFrames + dropPrefix
            let dstOffset = bin * targetLength
            dstPtr.advanced(by: dstOffset)
                .update(from: srcPtr.advanced(by: srcOffset), count: targetLength)
        }
        return (truncated, targetLength)
    }

    private func runEncoder(mel: MLMultiArray, length: Int) throws -> MLFeatureProvider {
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: Int32(length))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "length": MLFeatureValue(multiArray: lengthArray),
        ])
        return try encoder!.prediction(from: input)
    }

    // MARK: - Model Loading

    /// Load a pretrained Parakeet model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model identifier
    ///   - encoderVariant: For multi-encoder repos (e.g.
    ///     `aufklarer/Parakeet-TDT-v3-CoreML-INT8` after the M5 fix), the
    ///     shape suffix to pick: `"5s"`, `"15s"`, etc. Selects
    ///     `encoder_{variant}.mlmodelc`. `nil` (default) picks the repo's
    ///     `encoder.mlmodelc`, which is what single-shape repos
    ///     (`-30s`, `-iOS-5s`) ship.
    ///   - progressHandler: Optional callback for download/load progress `(fraction, status)`
    /// - Returns: Initialized model ready for transcription
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        encoderVariant: String? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ParakeetASRModel {
        let effectiveModelId: String
        if let modelId {
            effectiveModelId = modelId
        } else {
            #if os(iOS)
            effectiveModelId = iosModelId
            #else
            effectiveModelId = defaultModelId
            #endif
        }

        let encoderFileName: String = encoderVariant.map { "encoder_\($0).mlmodelc" }
            ?? "encoder.mlmodelc"

        AudioLog.modelLoading.info("Loading Parakeet model: \(effectiveModelId) (encoder: \(encoderFileName))")

        // Step 1: Get/create cache directory
        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        // Step 2: Download model files (no preprocessor needed — mel is computed in Swift)
        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: resolvedCacheDir,
                additionalFiles: [
                    "\(encoderFileName)/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        // Step 3: Load config
        progressHandler?(0.70, "Loading configuration...")
        let config: ParakeetConfig
        let configURL = resolvedCacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ParakeetConfig.self, from: data)
            AudioLog.modelLoading.debug("Loaded config from \(configURL.path)")
        } else {
            config = .default
            AudioLog.modelLoading.debug("Using default config")
        }

        // Step 4: Load vocabulary
        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = resolvedCacheDir.appendingPathComponent("vocab.json")
        let vocabulary: ParakeetVocabulary
        do {
            vocabulary = try ParakeetVocabulary.load(from: vocabURL)
            AudioLog.modelLoading.debug("Loaded vocabulary: \(vocabulary.count) tokens")
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to load vocabulary", underlying: error)
        }

        // Step 5: Load CoreML models (encoder, decoder, joint — no preprocessor)
        progressHandler?(0.80, "Loading CoreML models...")
        // Compute-unit preference per platform:
        //   iOS Simulator: pin `.cpuOnly`. The simulator has no MPSGraph
        //     backend ("Espresso compiled without MPSGraph engine") — any
        //     other choice silently falls back to a partial CPU/GPU mix
        //     that returns 0 tokens for the INT8-quantized encoder.
        //   iOS device + macOS: prefer `.cpuAndNeuralEngine` for the
        //     3-10× ANE speedup, fall back to `.cpuAndGPU` if loading
        //     throws. macOS was historically pinned to `.cpuAndGPU`
        //     because the iOS17-target INT8 encoder SIGSEGV'd in
        //     bnns::GraphCompile on Mac ANE (issue #313 surfaced this on
        //     M5). The iOS18-target single-shape encoders shipped in
        //     `Parakeet-TDT-v3-CoreML-INT8-30s`, `-iOS-5s`, and the
        //     multi-encoder `Parakeet-TDT-v3-CoreML-INT8` repos fix
        //     that, so macOS now also tries ANE first.
        #if targetEnvironment(simulator)
        let computeUnitsToTry: [MLComputeUnits] = [.cpuOnly]
        #else
        let computeUnitsToTry: [MLComputeUnits] = [.cpuAndNeuralEngine, .cpuAndGPU]
        #endif
        let (encoder, encoderUnits) = try loadCoreMLModelWithFallback(
            name: encoderFileName.replacingOccurrences(of: ".mlmodelc", with: ""),
            from: resolvedCacheDir, computeUnitsToTry: computeUnitsToTry)
        progressHandler?(0.90, "Loading decoder...")
        let (decoder, _) = try loadCoreMLModelWithFallback(
            name: "decoder", from: resolvedCacheDir, computeUnitsToTry: computeUnitsToTry)
        progressHandler?(0.95, "Loading joint network...")
        let (joint, _) = try loadCoreMLModelWithFallback(
            name: "joint", from: resolvedCacheDir, computeUnitsToTry: computeUnitsToTry)

        let supportedMelLengths = discoverSupportedMelLengths(from: encoder)
        AudioLog.modelLoading.info("Parakeet encoder accepts mel frame counts: \(supportedMelLengths)")

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Parakeet model loaded successfully")

        return ParakeetASRModel(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            supportedMelLengths: supportedMelLengths,
            encoderComputeUnits: encoderUnits
        )
    }

    /// Read the encoder's mel input shape constraint and return the supported
    /// frame counts (dim 2 of `[batch, mel_bins, frames]`), sorted ascending.
    ///
    /// Handles every export style produced by `models/parakeet-asr/export/convert.py`:
    /// - `--single-shape` → fixed shape, returns one element (e.g. `[3000]` for
    ///   the 30s default, `[500]` for iOS-5s)
    /// - legacy default → `EnumeratedShapes`, returns the enumerated frame counts
    ///
    /// Falls back to `defaultMelLengths` if introspection fails — that matches
    /// the historical hardcoded list, so behaviour on previously-working exports
    /// is unchanged.
    private static func discoverSupportedMelLengths(from encoder: MLModel) -> [Int] {
        guard let melDescription = encoder.modelDescription.inputDescriptionsByName["mel"],
              let arrayConstraint = melDescription.multiArrayConstraint else {
            return defaultMelLengths
        }

        let shapeConstraint = arrayConstraint.shapeConstraint
        switch shapeConstraint.type {
        case .enumerated:
            let frames = shapeConstraint.enumeratedShapes
                .compactMap { $0.count >= 3 ? $0[2].intValue : nil }
                .sorted()
            return frames.isEmpty ? defaultMelLengths : frames
        case .unspecified:
            // Single fixed shape — the canonical shape on the constraint is the
            // only one the model accepts.
            let canonical = arrayConstraint.shape
            return canonical.count >= 3 ? [canonical[2].intValue] : defaultMelLengths
        case .range:
            // Range-on-time-axis exports: pick supported lengths that fall
            // inside the model's allowed range. Snap to the historical list
            // (intersected with the range) so we don't explode into thousands
            // of candidate paddings for permissive ranges.
            let perDim = shapeConstraint.sizeRangeForDimension
            guard perDim.count >= 3 else { return defaultMelLengths }
            let timeRange = perDim[2].rangeValue
            let lower = max(1, timeRange.location)
            let upper = lower + timeRange.length
            let candidates = defaultMelLengths.filter { $0 >= lower && $0 <= upper }
            return candidates.isEmpty ? [upper] : candidates
        @unknown default:
            return defaultMelLengths
        }
    }

    /// Load a compiled CoreML model, trying compute-unit preferences in
    /// order until one loads successfully. Returns `(model, units)` so
    /// the caller knows which backend actually picked up the model —
    /// e.g. ANE on devices where it's supported, GPU on devices where
    /// ANE compilation fails (`Unknown aneSubType`).
    private static func loadCoreMLModelWithFallback(
        name: String,
        from directory: URL,
        computeUnitsToTry: [MLComputeUnits]
    ) throws -> (model: MLModel, units: MLComputeUnits) {
        var lastError: Error?
        for computeUnits in computeUnitsToTry {
            do {
                let model = try loadCoreMLModel(
                    name: name, from: directory, computeUnits: computeUnits)
                return (model, computeUnits)
            } catch {
                lastError = error
                AudioLog.modelLoading.notice(
                    "Loading \(name) on \(String(describing: computeUnits)) failed: \(error.localizedDescription) — trying next compute unit")
            }
        }
        throw lastError ?? AudioModelError.modelLoadFailed(
            modelId: name, reason: "All compute-unit options failed: \(computeUnitsToTry)")
    }

    /// Load a compiled CoreML model from a `.mlmodelc` directory.
    private static func loadCoreMLModel(
        name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)

        do {
            let model = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            AudioLog.modelLoading.info("Loaded CoreML model: \(name) (compute: \(String(describing: computeUnits)))")
            return model
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "Failed to compile/load CoreML model", underlying: error)
        }
    }
}
