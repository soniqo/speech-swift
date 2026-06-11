import CoreML
import Foundation
import AudioCommon

/// Nemotron-3.5 ASR Streaming 0.6B — multilingual streaming ASR on CoreML.
///
/// Cache-aware FastConformer encoder + prompt-conditioned RNN-T decoder.
/// 600 M parameters, INT8 palettized encoder, 76 languages. Native
/// punctuation and capitalization emitted as regular BPE tokens — no
/// EOU/EOB heads; caller signals end of stream via `finalize()`.
///
/// Language is set per-session via the `language` parameter on
/// `transcribeAudio` / `transcribeStream`, or directly via
/// `createSession(language:)`. The encoder receives a one-hot
/// `language_mask` of shape `[1, numPrompts]`; the slot index is resolved
/// from `languages.json` by `NemotronLanguages.slot(for:)`.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class NemotronStreamingASRModel {
    public let config: NemotronStreamingConfig
    public let languages: NemotronLanguages

    public static let defaultModelId = "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8"

    var _isLoaded = true
    private let melPreprocessor: StreamingMelPreprocessor
    var encoder: MLModel?
    var decoder: MLModel?
    var joint: MLModel?
    private let vocabulary: NemotronVocabulary
    private let wordBoostingTokenizer: NemotronSentencePieceUnigramTokenizer?
    public let wordBoostingTokenizerStatus: WordBoostingTokenizerStatus

    private init(
        config: NemotronStreamingConfig,
        languages: NemotronLanguages,
        encoder: MLModel?,
        decoder: MLModel?,
        joint: MLModel?,
        vocabulary: NemotronVocabulary,
        wordBoostingTokenizer: NemotronSentencePieceUnigramTokenizer?,
        wordBoostingTokenizerStatus: WordBoostingTokenizerStatus
    ) {
        self.config = config
        self.languages = languages
        self.melPreprocessor = StreamingMelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
        self.wordBoostingTokenizer = wordBoostingTokenizer
        self.wordBoostingTokenizerStatus = wordBoostingTokenizerStatus
    }

    /// A partial transcript from streaming recognition.
    public struct PartialTranscript: Sendable {
        public let text: String
        public let isFinal: Bool
        public let confidence: Float
        public let segmentIndex: Int
        public let wordBoostingChangedDecisions: Int
    }

    /// Create a streaming session. `language` is a BCP-47 tag (e.g. `"en-US"`,
    /// `"ja-JP"`); `nil` or unknown falls back to the model's `"auto"` slot.
    /// `wordBoosting` biases RNN-T decoding toward the provided phrases.
    public func createSession(
        language: String? = nil,
        wordBoosting: WordBoostingConfig? = nil
    ) throws -> StreamingSession {
        guard _isLoaded, let encoder, let decoder, let joint else {
            throw AudioModelError.inferenceFailed(operation: "createSession", reason: "Model not loaded")
        }
        let slot = languages.slot(for: language)
        return try StreamingSession(
            config: config,
            languageSlot: slot,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            melPreprocessor: melPreprocessor,
            wordBoosting: wordBoosting,
            wordBoostingTokenizer: wordBoostingTokenizer
        )
    }

    /// Analyze phrases with the loaded Nemotron vocabulary and suggest a
    /// conservative boost strength for each one.
    public func wordBoostingSuggestions(for phrases: [String]) -> [WordBoostingSuggestion] {
        WordBoostingContext.suggestions(
            for: phrases,
            vocabulary: vocabulary,
            tokenizer: wordBoostingTokenizer
        )
    }

    /// Convenience: stream transcription from a buffer, yielding partial results.
    public func transcribeStream(
        audio: [Float],
        sampleRate: Int,
        language: String? = nil,
        chunkDuration: Float? = nil,
        wordBoosting: WordBoostingConfig? = nil
    ) -> AsyncStream<PartialTranscript> {
        let chunkMs = chunkDuration.map { Int($0 * 1000) } ?? config.streaming.chunkMs

        return AsyncStream { continuation in
            Task {
                do {
                    let samples: [Float]
                    if sampleRate != self.config.sampleRate {
                        samples = AudioFileLoader.resample(audio, from: sampleRate, to: self.config.sampleRate)
                    } else {
                        samples = audio
                    }
                    let actualSamplesPerChunk = chunkMs * self.config.sampleRate / 1000
                    let session = try self.createSession(language: language, wordBoosting: wordBoosting)
                    var offset = 0
                    while offset < samples.count {
                        let end = min(offset + actualSamplesPerChunk, samples.count)
                        let chunk = Array(samples[offset..<end])
                        let partials = try session.pushAudio(chunk)
                        for partial in partials { continuation.yield(partial) }
                        offset = end
                    }
                    let finals = try session.finalize()
                    for partial in finals { continuation.yield(partial) }
                    continuation.finish()
                } catch {
                    AudioLog.inference.error("Nemotron streaming transcription failed: \(error)")
                    continuation.finish()
                }
            }
        }
    }

    /// Transcribe a full audio buffer (non-streaming fallback).
    ///
    /// The streaming encoder starts each session with an all-zero attention /
    /// convolution cache and an all-zero mel pre-cache. Short TTS-generated
    /// audio with sharp onsets can lose the first word at chunk boundaries,
    /// so by default we pad 0.1 s of silence at both ends to prime the cache
    /// with one "silent" encoder frame. For natural speech (FLEURS, mic
    /// capture, etc.) the padding can subtly shift chunk alignment — pass
    /// `padSilence: false` to skip it. Measured impact: ~−5 pp WER for
    /// Hindi FLEURS, 0–1 pp for other languages.
    public func transcribeAudio(
        _ audio: [Float],
        sampleRate: Int,
        language: String? = nil,
        padSilence: Bool = true,
        wordBoosting: WordBoostingConfig? = nil
    ) throws -> String {
        var samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }
        if padSilence {
            let padSamples = config.sampleRate / 10  // 100 ms
            samples = [Float](repeating: 0, count: padSamples) + samples + [Float](repeating: 0, count: padSamples)
        }

        let session = try createSession(language: language, wordBoosting: wordBoosting)
        var allPartials = try session.pushAudio(samples)
        allPartials.append(contentsOf: try session.finalize())
        if let lastFinal = allPartials.last(where: { $0.isFinal }) {
            return lastFinal.text
        }
        return allPartials.last?.text ?? ""
    }

    /// Warm up CoreML models on a dummy input.
    public func warmUp() throws {
        let dummy = [Float](repeating: 0, count: config.sampleRate)
        _ = try transcribeAudio(dummy, sampleRate: config.sampleRate)
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading Nemotron Streaming model: \(effectiveModelId)")

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: cacheDir,
                additionalFiles: [
                    "encoder.mlmodelc/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "tokenizer.model",
                    "*_tokenizer.model",
                    "vocab.txt",
                    "*_vocab.txt",
                    "languages.json",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        return try await load(from: cacheDir, source: effectiveModelId, progressHandler: progressHandler)
    }

    /// Load a model from a local directory (no download). The directory must
    /// contain `encoder.mlmodelc/`, `decoder.mlmodelc/`, `joint.mlmodelc/`,
    /// `vocab.json`, `languages.json`, and optionally `config.json`.
    public static func fromLocal(
        bundleDir: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRModel {
        AudioLog.modelLoading.info("Loading Nemotron Streaming from local: \(bundleDir.path)")
        return try await load(from: bundleDir, source: bundleDir.path, progressHandler: progressHandler)
    }

    private static func load(
        from cacheDir: URL,
        source: String,
        progressHandler: ((Double, String) -> Void)?
    ) async throws -> NemotronStreamingASRModel {
        progressHandler?(0.70, "Loading configuration...")
        let config: NemotronStreamingConfig
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(NemotronStreamingConfig.self, from: data)
        } else {
            config = .default
        }

        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        let vocabulary = try NemotronVocabulary.load(from: vocabURL)

        progressHandler?(0.78, "Loading language map...")
        let languagesURL = cacheDir.appendingPathComponent("languages.json")
        let languages: NemotronLanguages
        if FileManager.default.fileExists(atPath: languagesURL.path) {
            languages = try NemotronLanguages.load(from: languagesURL)
        } else {
            // English-only fallback (single auto slot).
            languages = NemotronLanguages(promptDictionary: ["en-US": 0, "en": 0, "auto": 0])
        }

        progressHandler?(0.79, "Loading word boosting tokenizer...")
        let loadedWordBoostingTokenizer = try loadWordBoostingTokenizer(from: cacheDir)
        let wordBoostingTokenizer = loadedWordBoostingTokenizer?.tokenizer
        let wordBoostingTokenizerStatus = WordBoostingTokenizerStatus(
            mode: loadedWordBoostingTokenizer == nil ? .vocabFallback : .sentencePieceModel,
            path: loadedWordBoostingTokenizer?.url.path
        )

        // `.all` lets CoreML schedule the encoder onto the ANE (which is what
        // Python coremltools' `ComputeUnit.ALL` does). Encoder gains ~40% RTF
        // over `.cpuAndGPU`. Decoder + joint are tiny enough that ANE vs CPU
        // is a wash, but using `.all` keeps the unit selection consistent.
        progressHandler?(0.80, "Loading CoreML models...")
        let encoder = try loadCoreMLModel(name: "encoder", from: cacheDir, computeUnits: .all)
        progressHandler?(0.90, "Loading decoder...")
        let decoder = try loadCoreMLModel(name: "decoder", from: cacheDir, computeUnits: .all)
        progressHandler?(0.95, "Loading joint network...")
        let joint = try loadCoreMLModel(name: "joint", from: cacheDir, computeUnits: .all)

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info(
            "Nemotron Streaming loaded from \(source) (\(vocabulary.count) tokens, \(languages.count) lang aliases)")

        return NemotronStreamingASRModel(
            config: config,
            languages: languages,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            wordBoostingTokenizer: wordBoostingTokenizer,
            wordBoostingTokenizerStatus: wordBoostingTokenizerStatus
        )
    }

    private static func loadWordBoostingTokenizer(
        from directory: URL
    ) throws -> (tokenizer: NemotronSentencePieceUnigramTokenizer, url: URL)? {
        let fm = FileManager.default
        let direct = directory.appendingPathComponent("tokenizer.model")
        if fm.fileExists(atPath: direct.path) {
            return (try NemotronSentencePieceUnigramTokenizer(modelURL: direct), direct)
        }

        let contents = (try? fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        guard let hashed = contents.first(where: { $0.lastPathComponent.hasSuffix("_tokenizer.model") }) else {
            return nil
        }
        return (try NemotronSentencePieceUnigramTokenizer(modelURL: hashed), hashed)
    }

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
        return try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }

}
