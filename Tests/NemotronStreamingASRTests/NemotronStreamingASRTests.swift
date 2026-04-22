import XCTest
@testable import NemotronStreamingASR
@testable import AudioCommon

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
