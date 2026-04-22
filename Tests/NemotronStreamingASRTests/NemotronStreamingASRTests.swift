import XCTest
@testable import NemotronStreamingASR
@testable import AudioCommon
@testable import KokoroTTS
import CoreML

final class NemotronStreamingConfigTests: XCTestCase {

    func testDefaultConfigIsSensible() {
        let config = NemotronStreamingConfig.default
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.encoderHidden, 1024)
        XCTAssertEqual(config.encoderLayers, 24)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 2)
        XCTAssertEqual(config.vocabSize, 1024)
        XCTAssertEqual(config.blankTokenId, 1024)
    }

    func testStreamingDefaultsForChunk160ms() {
        let s = NemotronStreamingConfig.default.streaming
        XCTAssertEqual(s.chunkMs, 160)
        XCTAssertEqual(s.chunkSize, 2)
        XCTAssertEqual(s.rightContext, 1)
        XCTAssertEqual(s.melFrames, 17)
        XCTAssertEqual(s.preCacheSize, 16)
        XCTAssertEqual(s.outputFrames, 2)
    }

    func testConfigRoundtripsThroughJSON() throws {
        let config = NemotronStreamingConfig.default
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(NemotronStreamingConfig.self, from: encoded)
        XCTAssertEqual(decoded.encoderHidden, config.encoderHidden)
        XCTAssertEqual(decoded.decoderLayers, config.decoderLayers)
        XCTAssertEqual(decoded.streaming.chunkMs, config.streaming.chunkMs)
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

// MARK: - E2E Tests (require model download + CoreML)

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
            Self._model = try await NemotronStreamingASRModel.fromPretrained()
        }
    }

    func testModelLoading() throws {
        let m = try model
        XCTAssertTrue(m.isLoaded)
        XCTAssertEqual(m.config.encoderHidden, 1024)
        XCTAssertEqual(m.config.encoderLayers, 24)
        XCTAssertEqual(m.config.decoderLayers, 2)
        XCTAssertEqual(m.config.vocabSize, 1024)
    }

    func testWarmup() throws {
        try model.warmUp()
    }

    func testBatchTranscription() throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let text = try m.transcribeAudio(audio, sampleRate: 16000)
        XCTAssertFalse(text.isEmpty, "Transcription should not be empty")
        print("Batch result: \(text)")
    }

    func testStreamingTranscription() async throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        var partials: [NemotronStreamingASRModel.PartialTranscript] = []
        for await partial in m.transcribeStream(audio: audio, sampleRate: 16000) {
            partials.append(partial)
        }
        XCTAssertFalse(partials.isEmpty, "Should produce at least one partial")
        let last = partials.last!
        XCTAssertTrue(last.isFinal, "Last partial should be final (from finalize())")
        XCTAssertFalse(last.text.isEmpty, "Final text should not be empty")
        print("Streamed final: \(last.text)")
    }

    func testStreamingSessionSilence() throws {
        let m = try model
        let session = try m.createSession()
        let samplesPerChunk = m.config.streaming.chunkMs * m.config.sampleRate / 1000
        let silence = [Float](repeating: 0, count: samplesPerChunk)
        for _ in 0..<3 {
            _ = try session.pushAudio(silence)
        }
        let finals = try session.finalize()
        XCTAssertNotNil(finals, "finalize() should return a (possibly empty) array")
    }

    func testMemoryManagement() async throws {
        let m = try await NemotronStreamingASRModel.fromPretrained()
        XCTAssertTrue(m.isLoaded)
        XCTAssertGreaterThan(m.memoryFootprint, 0)
        m.unload()
        XCTAssertFalse(m.isLoaded)
        XCTAssertEqual(m.memoryFootprint, 0)
    }

    /// Synthesize a known phrase with Kokoro and transcribe it through Nemotron.
    /// Kokoro produces 24 kHz audio; Nemotron resamples internally to 16 kHz.
    ///
    /// The threshold is 3/5 content words rather than full-string match: TTS→ASR
    /// chains routinely introduce leading-breath artifacts (e.g. "The" heard as
    /// "But a") and cache-aware streaming can clip a few hundred ms at the tail.
    /// Three-of-five content-word recovery is a strong end-to-end signal without
    /// being brittle to acoustic noise from the synthesizer.
    func testTTSRoundTrip() async throws {
        let nemotron = try model
        let tts = try await KokoroTTSModel.fromPretrained()

        let phrase = "The quick brown fox jumps over the lazy dog"
        let audio24k = try tts.synthesize(text: phrase, voice: "af_heart")
        XCTAssertGreaterThan(audio24k.count, 24000, "TTS should produce at least 1s of audio")

        let transcript = try nemotron.transcribeAudio(audio24k, sampleRate: 24000)
        print("Round-trip input:  \"\(phrase)\"")
        print("Round-trip output: \"\(transcript)\"")

        let normalized = transcript.lowercased()
        let expected = ["quick", "brown", "fox", "jumps", "over"]
        let matched = expected.filter { normalized.contains($0) }
        XCTAssertGreaterThanOrEqual(matched.count, 3,
            "Round-trip transcript should recover at least 3/\(expected.count) content words. " +
            "Matched: \(matched). Transcript: \"\(transcript)\"")
    }
}
