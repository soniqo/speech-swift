import CoreML
import XCTest
@testable import ParakeetStreamingASR
import AudioCommon

// MARK: - Unit Tests (no model/GPU required)

final class ParakeetStreamingASRTests: XCTestCase {

    // MARK: - Configuration Tests

    func testDefaultConfig() {
        let config = ParakeetEOUConfig.default
        XCTAssertEqual(config.numMelBins, 128)
        XCTAssertEqual(config.sampleRate, 16000)
        XCTAssertEqual(config.nFFT, 512)
        XCTAssertEqual(config.hopLength, 160)
        XCTAssertEqual(config.winLength, 400)
        XCTAssertEqual(config.preEmphasis, 0.97)
        XCTAssertEqual(config.encoderHidden, 512)
        XCTAssertEqual(config.encoderLayers, 17)
        XCTAssertEqual(config.subsamplingFactor, 8)
        XCTAssertEqual(config.attentionContext, 70)
        XCTAssertEqual(config.convCacheSize, 8)
        XCTAssertEqual(config.decoderHidden, 640)
        XCTAssertEqual(config.decoderLayers, 1)
        XCTAssertEqual(config.vocabSize, 1026)
        XCTAssertEqual(config.blankTokenId, 1026)
        XCTAssertEqual(config.eouTokenId, 1024)
        XCTAssertEqual(config.eobTokenId, 1025)
    }

    func testStreamingConfig() {
        let config = ParakeetEOUConfig.default
        XCTAssertEqual(config.streaming.chunkMs, 320)
        XCTAssertEqual(config.streaming.melFrames, 33)
        XCTAssertEqual(config.streaming.preCacheSize, 9)
        XCTAssertEqual(config.streaming.outputFrames, 4)
    }

    func testConfigCodable() throws {
        let original = ParakeetEOUConfig.default
        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ParakeetEOUConfig.self, from: data)

        XCTAssertEqual(decoded.encoderHidden, original.encoderHidden)
        XCTAssertEqual(decoded.encoderLayers, original.encoderLayers)
        XCTAssertEqual(decoded.decoderHidden, original.decoderHidden)
        XCTAssertEqual(decoded.vocabSize, original.vocabSize)
        XCTAssertEqual(decoded.eouTokenId, original.eouTokenId)
        XCTAssertEqual(decoded.streaming.chunkMs, original.streaming.chunkMs)
    }

    func testConfigSendable() async {
        let config = ParakeetEOUConfig.default
        let result = await Task { config }.value
        XCTAssertEqual(result.encoderHidden, config.encoderHidden)
        XCTAssertEqual(result.eouTokenId, config.eouTokenId)
    }

    func testModelIdConstant() {
        XCTAssertEqual(
            ParakeetStreamingASRModel.defaultModelId,
            "aufklarer/Parakeet-EOU-120M-CoreML-INT8"
        )
    }

    // MARK: - Vocabulary Tests

    func testVocabularyDecode() {
        let vocab = ParakeetEOUVocabulary(idToToken: [
            0: "\u{2581}the",
            1: "\u{2581}cat",
            2: "\u{2581}sat",
        ])
        let text = vocab.decode([0, 1, 2])
        XCTAssertEqual(text, "the cat sat")
    }

    func testVocabularyDecodeSkipsUnknown() {
        let vocab = ParakeetEOUVocabulary(idToToken: [
            0: "\u{2581}hello",
            1: "\u{2581}world",
        ])
        let text = vocab.decode([0, 999, 1])
        XCTAssertEqual(text, "hello world")
    }

    func testVocabularyDecodeEmpty() {
        let vocab = ParakeetEOUVocabulary(idToToken: [:])
        let text = vocab.decode([])
        XCTAssertEqual(text, "")
    }

    func testVocabularyDecodeSubwordMerging() {
        let vocab = ParakeetEOUVocabulary(idToToken: [
            0: "\u{2581}un",
            1: "believ",
            2: "able",
        ])
        let text = vocab.decode([0, 1, 2])
        XCTAssertEqual(text, "unbelievable")
    }

    func testVocabularyDecodeWords() {
        let vocab = ParakeetEOUVocabulary(idToToken: [
            0: "\u{2581}hello",
            1: "\u{2581}world",
        ])
        let words = vocab.decodeWords([0, 1], logProbs: [-0.1, -0.2])
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(words[0].word, "hello")
        XCTAssertEqual(words[1].word, "world")
        XCTAssertGreaterThan(words[0].confidence, 0)
        XCTAssertLessThanOrEqual(words[0].confidence, 1.0)
    }

    func testVocabularyDecodeWordsMismatchedCounts() {
        let vocab = ParakeetEOUVocabulary(idToToken: [0: "\u{2581}a"])
        let words = vocab.decodeWords([0], logProbs: [])
        XCTAssertEqual(words.count, 0)  // Mismatched counts returns empty
    }

    // MARK: - Mel Preprocessor Tests

    func testMelPreprocessorSilence() throws {
        let config = ParakeetEOUConfig.default
        let preprocessor = StreamingMelPreprocessor(config: config)

        // 320ms of silence at 16kHz
        let silence = [Float](repeating: 0, count: 5120)
        let (mel, melLength) = try preprocessor.extract(silence)

        XCTAssertEqual(mel.shape[0].intValue, 1)
        XCTAssertEqual(mel.shape[1].intValue, 128)
        XCTAssertGreaterThan(melLength, 0)
    }

    func testMelPreprocessorSineWave() throws {
        let config = ParakeetEOUConfig.default
        let preprocessor = StreamingMelPreprocessor(config: config)

        // 320ms of 440Hz sine at 16kHz
        let numSamples = 5120
        var audio = [Float](repeating: 0, count: numSamples)
        for i in 0..<numSamples {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0) * 0.5
        }

        let (mel, melLength) = try preprocessor.extract(audio)

        XCTAssertEqual(mel.shape[0].intValue, 1)
        XCTAssertEqual(mel.shape[1].intValue, 128)
        XCTAssertGreaterThan(melLength, 0)

        // Verify mel values are finite (output is float32)
        let ptr = mel.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<min(128, mel.count) {
            XCTAssertFalse(ptr[i].isNaN, "Mel value at index \(i) is NaN")
            XCTAssertFalse(ptr[i].isInfinite, "Mel value at index \(i) is infinite")
        }
    }

    func testMelPreprocessorEmptyInput() throws {
        let config = ParakeetEOUConfig.default
        let preprocessor = StreamingMelPreprocessor(config: config)

        let (mel, melLength) = try preprocessor.extract([])
        XCTAssertEqual(melLength, 0)
        XCTAssertEqual(mel.shape[0].intValue, 1)
    }

    // MARK: - Cache Shape Tests

    func testEncoderCacheShapes() throws {
        let config = ParakeetEOUConfig.default

        // Verify expected cache dimensions
        let channelSize = config.encoderLayers * 1 * config.attentionContext * config.encoderHidden
        let timeSize = config.encoderLayers * 1 * config.encoderHidden * config.convCacheSize

        // cache_last_channel: [17, 1, 70, 512] = 608,640 floats = ~2.4 MB
        XCTAssertEqual(channelSize, 17 * 70 * 512)
        let channelMB = Float(channelSize * 4) / (1024 * 1024)
        XCTAssertLessThan(channelMB, 3.0, "Channel cache should be < 3 MB")

        // cache_last_time: [17, 1, 512, 8] = 69,632 floats = ~0.27 MB
        XCTAssertEqual(timeSize, 17 * 512 * 8)
        let timeMB = Float(timeSize * 4) / (1024 * 1024)
        XCTAssertLessThan(timeMB, 0.5, "Time cache should be < 0.5 MB")
    }

    // MARK: - PartialTranscript Tests

    func testPartialTranscriptSendable() async {
        let partial = ParakeetStreamingASRModel.PartialTranscript(
            text: "hello",
            isFinal: false,
            confidence: 0.95,
            eouDetected: false,
            segmentIndex: 0
        )
        let result = await Task { partial }.value
        XCTAssertEqual(result.text, "hello")
        XCTAssertFalse(result.isFinal)
        XCTAssertEqual(result.confidence, 0.95)
    }

    func testPartialTranscriptFinal() {
        let partial = ParakeetStreamingASRModel.PartialTranscript(
            text: "hello world",
            isFinal: true,
            confidence: 0.9,
            eouDetected: true,
            segmentIndex: 2
        )
        XCTAssertTrue(partial.isFinal)
        XCTAssertTrue(partial.eouDetected)
        XCTAssertEqual(partial.segmentIndex, 2)
    }
}

// MARK: - E2E Tests (require model download + CoreML)

final class E2EParakeetStreamingASRTests: XCTestCase {

    private static var _model: ParakeetStreamingASRModel?

    private var model: ParakeetStreamingASRModel {
        get throws {
            guard let m = Self._model else {
                throw XCTSkip("Model not loaded")
            }
            return m
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            Self._model = try await ParakeetStreamingASRModel.fromPretrained()
        }
    }

    func testModelLoading() throws {
        let m = try model
        XCTAssertTrue(m.isLoaded)
        XCTAssertEqual(m.config.encoderHidden, 512)
        XCTAssertEqual(m.config.encoderLayers, 17)
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
    }

    func testStreamingTranscription() async throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        var partials: [ParakeetStreamingASRModel.PartialTranscript] = []
        for await partial in m.transcribeStream(audio: audio, sampleRate: 16000) {
            partials.append(partial)
        }

        XCTAssertFalse(partials.isEmpty, "Should produce at least one partial")
        let lastPartial = partials.last!
        XCTAssertTrue(lastPartial.isFinal, "Last partial should be final")
        XCTAssertFalse(lastPartial.text.isEmpty, "Final text should not be empty")
    }

    func testStreamingSession() throws {
        let m = try model
        let session = try m.createSession()

        // Push 1 second of silence in 320ms chunks
        let samplesPerChunk = m.config.streaming.chunkMs * m.config.sampleRate / 1000
        let silence = [Float](repeating: 0, count: samplesPerChunk)

        for _ in 0..<3 {
            _ = try session.pushAudio(silence)
        }

        let finals = try session.finalize()
        // Silence may or may not produce text, but should not crash
        XCTAssertNotNil(finals)
    }

    func testMemoryManagement() async throws {
        // Use a separate model instance for this test to avoid affecting others
        let m = try await ParakeetStreamingASRModel.fromPretrained()
        XCTAssertTrue(m.isLoaded)
        XCTAssertGreaterThan(m.memoryFootprint, 0)

        m.unload()
        XCTAssertFalse(m.isLoaded)
        XCTAssertEqual(m.memoryFootprint, 0)
    }

    func testStreamingLatency() throws {
        let m = try model
        let session = try m.createSession()
        let samplesPerChunk = m.config.streaming.chunkMs * m.config.sampleRate / 1000
        let chunkMs = Float(m.config.streaming.chunkMs)

        // Generate 440Hz tone chunk
        var audio = [Float](repeating: 0, count: samplesPerChunk)
        for i in 0..<samplesPerChunk {
            audio[i] = sin(2.0 * .pi * 440.0 * Float(i) / 16000.0) * 0.3
        }

        // Warmup
        _ = try session.pushAudio(audio)

        // Benchmark 10 chunks
        var times: [Double] = []
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try session.pushAudio(audio)
            times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }

        let avgMs = times.reduce(0, +) / Double(times.count)
        let rtf = avgMs / Double(chunkMs)
        print("Streaming latency: avg=\(String(format: "%.1f", avgMs))ms RTF=\(String(format: "%.3f", rtf))")
        XCTAssertLessThan(rtf, 1.0, "RTF should be < 1.0 for real-time streaming")
    }
}
