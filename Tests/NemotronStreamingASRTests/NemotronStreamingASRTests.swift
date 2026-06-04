import XCTest
@testable import NemotronStreamingASR
@testable import AudioCommon
@testable import KokoroTTS
import CoreML

final class NemotronStreamingConfigTests: XCTestCase {

    func testDefaultConfigIsMultilingual() {
        let config = NemotronStreamingConfig.default
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.encoderHidden, 1024)
        XCTAssertEqual(config.encoderLayers, 24)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.vocabSize, 13087)
        XCTAssertEqual(config.blankTokenId, 13087)
        XCTAssertEqual(config.attentionContext, 56)
        XCTAssertEqual(config.numPrompts, 128)
    }

    func testStreamingDefaultsForChunk320ms() {
        let s = NemotronStreamingConfig.default.streaming
        XCTAssertEqual(s.chunkMs, 320)
        XCTAssertEqual(s.chunkSize, 4)
        XCTAssertEqual(s.rightContext, 3)
        XCTAssertEqual(s.melFrames, 32)
        XCTAssertEqual(s.preCacheSize, 9)
        XCTAssertEqual(s.outputFrames, 4)
    }

    func testConfigRoundtripsThroughJSON() throws {
        let config = NemotronStreamingConfig.default
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(NemotronStreamingConfig.self, from: encoded)
        XCTAssertEqual(decoded.encoderHidden, config.encoderHidden)
        XCTAssertEqual(decoded.decoderLayers, config.decoderLayers)
        XCTAssertEqual(decoded.numPrompts, config.numPrompts)
        XCTAssertEqual(decoded.streaming.chunkMs, config.streaming.chunkMs)
    }

    /// English-only bundles ship a config.json without `numPrompts`;
    /// the decoder should default to 128 and not crash.
    func testConfigDecodesWithoutNumPrompts() throws {
        let json = """
        {
          "numMelBins": 128, "sampleRate": 16000, "nFFT": 512, "hopLength": 160,
          "winLength": 400, "preEmphasis": 0.97, "encoderHidden": 1024,
          "encoderLayers": 24, "subsamplingFactor": 8, "attentionContext": 70,
          "convCacheSize": 8, "decoderHidden": 640, "decoderLayers": 2,
          "vocabSize": 1024, "blankTokenId": 1024,
          "streaming": {
            "chunkMs": 160, "chunkSize": 2, "rightContext": 1,
            "melFrames": 17, "preCacheSize": 16, "outputFrames": 2
          }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NemotronStreamingConfig.self, from: json)
        XCTAssertEqual(cfg.numPrompts, 128)  // default fallback
        XCTAssertEqual(cfg.vocabSize, 1024)
    }
}

final class NemotronLanguagesTests: XCTestCase {

    func testWrappedDictionaryParses() throws {
        let json = """
        {"promptDictionary": {"en-US": 0, "de-DE": 9, "auto": 101}}
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("lang-\(UUID()).json")
        try json.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let langs = try NemotronLanguages.load(from: url)
        XCTAssertEqual(langs.slot(for: "en-US"), 0)
        XCTAssertEqual(langs.slot(for: "de-DE"), 9)
        XCTAssertEqual(langs.slot(for: nil), 101)
    }

    func testFlatDictionaryParses() throws {
        let json = """
        {"en-US": 0, "ja-JP": 10, "auto": 101}
        """.data(using: .utf8)!
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("lang-\(UUID()).json")
        try json.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let langs = try NemotronLanguages.load(from: url)
        XCTAssertEqual(langs.slot(for: "ja-JP"), 10)
    }

    func testPrefixFallback() {
        let langs = NemotronLanguages(promptDictionary: ["en": 0, "en-US": 0, "auto": 101])
        XCTAssertEqual(langs.slot(for: "en-GB"), 0,
            "Unknown subtag should fall back to base language slot")
        XCTAssertEqual(langs.slot(for: "fr-FR"), 101,
            "Completely unknown language should fall back to 'auto'")
    }

    func testUnderscoreNormalization() {
        let langs = NemotronLanguages(promptDictionary: ["en-US": 0, "auto": 101])
        XCTAssertEqual(langs.slot(for: "en_US"), 0)
    }
}

final class NemotronVocabularyTests: XCTestCase {

    func testDecodeJoinsSentencePieceTokens() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁hello",
            1: ",",
            2: "▁world",
            3: ".",
        ])
        let text = vocab.decode([0, 1, 2, 3])
        XCTAssertEqual(text, "hello, world.")
    }

    func testDecodeStripsUnknownIds() {
        let vocab = NemotronVocabulary(idToToken: [0: "▁the", 1: "▁cat"])
        XCTAssertEqual(vocab.decode([0, 999, 1]), "the cat")
    }

    func testDecodeWordsEmitsConfidences() {
        let vocab = NemotronVocabulary(idToToken: [
            0: "▁hello",
            1: "▁world",
        ])
        let logProbs: [Float] = [log(0.9), log(0.8)]
        let words = vocab.decodeWords([0, 1], logProbs: logProbs)
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "hello")
        XCTAssertEqual(words[1].word, "world")
        XCTAssertEqual(words[0].confidence, 0.9, accuracy: 1e-4)
        XCTAssertEqual(words[1].confidence, 0.8, accuracy: 1e-4)
    }
}

// MARK: - E2E Tests (require local CoreML bundle or HF download)

/// Path to a locally-converted CoreML bundle, settable via env var.
/// Falls back to the canonical /tmp path used by the export pipeline.
private func localBundlePath() -> URL? {
    if let env = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"], !env.isEmpty {
        let url = URL(fileURLWithPath: env)
        if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    let fallback = URL(fileURLWithPath: "/tmp/Nemotron-3.5-CoreML-320ms")
    if FileManager.default.fileExists(atPath: fallback.path) { return fallback }
    return nil
}

/// Path to the English-only bundle (`aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8`).
/// English-only Nemotron Speech Streaming is its own model variant — smaller bundle,
/// 160 ms chunks, attCtx=70, vocab=1024.
/// First looks at the env override, then the HF cache.
private func englishOnlyBundlePath() -> URL? {
    if let env = ProcessInfo.processInfo.environment["NEMOTRON_EN_LOCAL_BUNDLE"], !env.isEmpty {
        let url = URL(fileURLWithPath: env)
        if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    let home = FileManager.default.homeDirectoryForCurrentUser
    let cache = home
        .appendingPathComponent(".cache/huggingface/hub/models--aufklarer--Nemotron-Speech-Streaming-0.6B-CoreML-INT8/snapshots")
    guard let contents = try? FileManager.default.contentsOfDirectory(at: cache, includingPropertiesForKeys: nil) else {
        return nil
    }
    for entry in contents where entry.hasDirectoryPath {
        if FileManager.default.fileExists(atPath: entry.appendingPathComponent("encoder.mlmodelc").path) {
            return entry
        }
    }
    return nil
}

final class E2ENemotronStreamingASRTests: XCTestCase {

    private static var _model: NemotronStreamingASRModel?

    private var model: NemotronStreamingASRModel {
        get throws {
            guard let m = Self._model else { throw XCTSkip("Model not loaded") }
            return m
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            // Prefer local bundle (no download); fall back to HF.
            if let local = localBundlePath() {
                Self._model = try await NemotronStreamingASRModel.fromLocal(bundleDir: local)
            } else {
                Self._model = try await NemotronStreamingASRModel.fromPretrained()
            }
        }
    }

    func testModelLoading() throws {
        let m = try model
        XCTAssertTrue(m.isLoaded)
        XCTAssertEqual(m.config.encoderHidden, 1024)
        XCTAssertEqual(m.config.encoderLayers, 24)
        XCTAssertEqual(m.config.decoderLayers, 2)
        XCTAssertEqual(m.config.vocabSize, 13087)
        XCTAssertEqual(m.config.numPrompts, 128)
        XCTAssertGreaterThanOrEqual(m.languages.count, 60,
            "Multilingual bundle should expose 60+ language aliases")
        XCTAssertNotNil(m.languages.promptDictionary["en-US"])
        XCTAssertNotNil(m.languages.promptDictionary["ja-JP"])
    }

    func testWarmup() throws {
        try model.warmUp()
    }

    func testBatchTranscriptionEnglish() throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let text = try m.transcribeAudio(audio, sampleRate: 16000, language: "en-US")
        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        print("Batch en-US: \(text)")
    }

    func testStreamingTranscriptionEnglish() async throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        var partials: [NemotronStreamingASRModel.PartialTranscript] = []
        for await partial in m.transcribeStream(audio: audio, sampleRate: 16000, language: "en-US") {
            partials.append(partial)
        }
        XCTAssertFalse(partials.isEmpty)
        let last = partials.last!
        XCTAssertTrue(last.isFinal)
        XCTAssertFalse(last.text.isEmpty)
        print("Streamed en-US final: \(last.text)")
    }

    func testStreamingSessionSilence() throws {
        let m = try model
        let session = try m.createSession(language: "en-US")
        let samplesPerChunk = m.config.streaming.chunkMs * m.config.sampleRate / 1000
        let silence = [Float](repeating: 0, count: samplesPerChunk)
        for _ in 0..<3 {
            _ = try session.pushAudio(silence)
        }
        let finals = try session.finalize()
        XCTAssertNotNil(finals, "finalize() should return a (possibly empty) array")
    }

    func testMemoryManagement() async throws {
        // Resolve a model via the same path setUp uses, but reload fresh so
        // we can exercise unload independently of the shared instance.
        let m: NemotronStreamingASRModel
        if let local = localBundlePath() {
            m = try await NemotronStreamingASRModel.fromLocal(bundleDir: local)
        } else {
            m = try await NemotronStreamingASRModel.fromPretrained()
        }
        XCTAssertTrue(m.isLoaded)
        XCTAssertGreaterThan(m.memoryFootprint, 0)
        m.unload()
        XCTAssertFalse(m.isLoaded)
        XCTAssertEqual(m.memoryFootprint, 0)
    }

    /// English-only bundle (`aufklarer/Nemotron-Speech-Streaming-0.6B-CoreML-INT8`)
    /// loads and transcribes through the same `NemotronStreamingASR` target,
    /// using the same `transcribeAudio` API. The encoder doesn't accept
    /// `language_mask`, so `StreamingSession` skips it (geometry differs from
    /// multilingual: chunk=160 ms, attCtx=70, vocab=1024, no prompt kernel).
    func testEnglishOnlyBundleTranscription() async throws {
        guard let bundle = englishOnlyBundlePath() else {
            throw XCTSkip("English-only bundle not in HF cache")
        }
        let m = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundle)
        XCTAssertTrue(m.isLoaded)
        // English bundle has the older geometry.
        XCTAssertEqual(m.config.vocabSize, 1024)
        XCTAssertEqual(m.config.attentionContext, 70)
        XCTAssertEqual(m.config.streaming.chunkMs, 160)
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let text = try m.transcribeAudio(audio, sampleRate: 16000, language: "en-US")
        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        print("English-only bundle: \(text)")
    }

    /// Round-trip: same Kokoro-synthesized phrase, transcribe with multiple
    /// language prompt slots. en-US must recover all content words; other
    /// slots should still produce non-empty output (language conditioning
    /// shifts the BPE distribution but the encoder still hears speech).
    func testMultilingualLanguageSwitching() async throws {
        let nemotron = try model
        let tts = try await KokoroTTSModel.fromPretrained()

        let phrase = "The quick brown fox jumps over the lazy dog"
        let audio24k = try tts.synthesize(text: phrase, voice: "af_heart")

        let enText = try nemotron.transcribeAudio(audio24k, sampleRate: 24000, language: "en-US")
        print("en-US: \"\(enText)\"")
        let expected = ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        let matched = expected.filter { enText.lowercased().contains($0) }
        XCTAssertEqual(matched.count, expected.count,
            "en-US prompt should recover every content word; got \(matched)/\(expected)")

        // Same audio with a different language slot should still produce output
        // (the encoder runs regardless of slot choice; we're verifying the slot
        // wiring doesn't crash the pipeline for other languages).
        for lang in ["de-DE", "fr-FR", "ja-JP"] {
            let other = try nemotron.transcribeAudio(audio24k, sampleRate: 24000, language: lang)
            print("\(lang): \"\(other)\"")
            // Don't assert accuracy — wrong-language prompt usually still
            // produces *some* transcription, just often nonsensical.
        }
    }

    /// Round-trip with the English-only bundle: same Kokoro phrase + the same
    /// Swift `transcribeAudio` API + the same content-word assertion as the
    /// multilingual round-trip. Proves both bundles produce equivalent
    /// end-to-end output through one Swift target.
    func testEnglishOnlyBundleTTSRoundTrip() async throws {
        guard let bundle = englishOnlyBundlePath() else {
            throw XCTSkip("English-only bundle not in HF cache")
        }
        let nemotron = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundle)
        let tts = try await KokoroTTSModel.fromPretrained()

        let phrase = "The quick brown fox jumps over the lazy dog"
        let audio24k = try tts.synthesize(text: phrase, voice: "af_heart")

        let text = try nemotron.transcribeAudio(audio24k, sampleRate: 24000, language: "en-US")
        print("English-only round-trip: \"\(text)\"")
        let expected = ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        let matched = expected.filter { text.lowercased().contains($0) }
        XCTAssertEqual(matched.count, expected.count,
            "English-only bundle should recover every content word; got \(matched)/\(expected)")
    }
}
