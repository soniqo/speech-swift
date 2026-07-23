import XCTest
@testable import MossTranscribe

final class MossTranscriptParserTests: XCTestCase {
    func testParsesTimestampedSpeakerSegmentsAndPlainText() {
        let raw =
            "[0.93][S01] Hello there.[2.10]"
            + "[1.75][S02] General Kenobi.[3.25]"

        let parsed = MossTranscriptParser.plainText(from: raw)

        XCTAssertEqual(parsed.text, "Hello there. General Kenobi.")
        XCTAssertEqual(
            parsed.segments,
            [
                MossTranscriptSegment(
                    startTime: 0.93,
                    endTime: 2.10,
                    speaker: "S01",
                    text: "Hello there."
                ),
                MossTranscriptSegment(
                    startTime: 1.75,
                    endTime: 3.25,
                    speaker: "S02",
                    text: "General Kenobi."
                ),
            ]
        )
    }

    func testMalformedStructureRemainsVisible() {
        let malformed = "[2.0][S01] backwards[1.0]"
        let parsed = MossTranscriptParser.plainText(from: malformed)

        XCTAssertEqual(parsed.text, malformed)
        XCTAssertTrue(parsed.segments.isEmpty)
    }

    func testEmptyAndBackwardsSegmentsAreIgnored() {
        let raw =
            "[0][S01]   [1]"
            + "[4][S02] backwards[3]"
            + "[5][S03] valid[6]"

        let segments = MossTranscriptParser.parse(raw)

        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].speaker, "S03")
        XCTAssertEqual(segments[0].text, "valid")
    }
}

final class MossPromptProcessorTests: XCTestCase {
    private let configuration = MossProcessorConfiguration(
        audioTokensPerSecond: 12.5,
        audioMergeSize: 4,
        timeMarkerEverySeconds: 5,
        enableTimeMarker: true
    )

    func testDefaultPromptMatchesPublishedChatTemplate() {
        let expected =
            "<|im_start|>system\n"
            + "You are a helpful assistant.<|im_end|>\n"
            + "<|im_start|>user\n"
            + "<|audio_start|><|audio_pad|><|audio_end|>\n"
            + MossPromptProcessor.defaultInstruction
            + "<|im_end|>\n"
            + "<|im_start|>assistant\n"

        XCTAssertEqual(MossPromptProcessor.renderPrompt(), expected)
    }

    func testThirtySecondAudioSpanMatchesReferenceMarkerPositions() {
        let digitIDs = Dictionary(
            uniqueKeysWithValues: zip(
                Array("0123456789"),
                Array(15...24)
            )
        )
        let span = MossPromptProcessor.makeAudioSpan(
            audioTokenCount: 375,
            audioTokenID: 151_671,
            digitTokenIDs: digitIDs,
            configuration: configuration
        )

        XCTAssertEqual(span.filter { $0 == 151_671 }.count, 375)
        XCTAssertEqual(span.count, 386)
        XCTAssertEqual(span[62], 20)       // "5"
        XCTAssertEqual(Array(span[125...126]), [16, 15])  // "10"
        XCTAssertEqual(Array(span[189...190]), [16, 20])  // "15"
        XCTAssertEqual(Array(span[253...254]), [17, 15])  // "20"
        XCTAssertEqual(Array(span[317...318]), [17, 20])  // "25"
        XCTAssertEqual(Array(span[381...382]), [18, 15])  // "30"
    }

    func testAudioTokenLengthUsesCeilingStride() {
        XCTAssertEqual(
            MossWhisperFeatureExtractor.audioTokenCount(sampleCount: 1),
            1
        )
        XCTAssertEqual(
            MossWhisperFeatureExtractor.audioTokenCount(sampleCount: 16_000),
            13
        )
        XCTAssertEqual(
            MossWhisperFeatureExtractor.audioTokenCount(sampleCount: 480_000),
            375
        )
    }
}

final class MossDecoderConfigurationTests: XCTestCase {
    func testRangePrefillUsesFewestCalls() throws {
        let configuration = try JSONDecoder().decode(
            MossDecoderConfiguration.self,
            from: Data(
                """
                {
                  "hidden_size": 1024,
                  "max_seq_length": 1024,
                  "vocab_size": 151936,
                  "enumerated_t": [1, 128],
                  "shape_mode": "range",
                  "io_precision": "float16",
                  "multifunction": {
                    "decoder_function": "decoder",
                    "embedding_function": "embedding",
                    "file": "decoder.mlmodelc"
                  }
                }
                """.utf8
            )
        )

        try configuration.validate()
        XCTAssertEqual(
            try configuration.prefillChunks(tokenCount: 472),
            [128, 128, 128, 88]
        )
    }

    func testEnumeratedPrefillRequiresSingleTokenShape() throws {
        let configuration = try JSONDecoder().decode(
            MossDecoderConfiguration.self,
            from: Data(
                """
                {
                  "hidden_size": 1024,
                  "max_seq_length": 1024,
                  "vocab_size": 151936,
                  "enumerated_t": [64, 128],
                  "shape_mode": "enumerated",
                  "io_precision": "float16",
                  "multifunction": {
                    "decoder_function": "decoder",
                    "embedding_function": "embedding",
                    "file": "decoder.mlmodelc"
                  }
                }
                """.utf8
            )
        )

        XCTAssertThrowsError(
            try configuration.prefillChunks(tokenCount: 65)
        )
    }
}

final class MossBundleConfigurationTests: XCTestCase {
    func testPublishedHostAndPreprocessorContractsValidate() throws {
        let decoder = try JSONDecoder().decode(
            MossDecoderConfiguration.self,
            from: Data(
                """
                {
                  "hidden_size": 1024,
                  "max_seq_length": 1024,
                  "vocab_size": 151936,
                  "enumerated_t": [1, 128],
                  "shape_mode": "range",
                  "io_precision": "float16",
                  "multifunction": {
                    "decoder_function": "decoder",
                    "embedding_function": "embedding",
                    "file": "decoder.mlmodelc"
                  }
                }
                """.utf8
            )
        )
        let bundle = try JSONDecoder().decode(
            MossBundleConfiguration.self,
            from: Data(
                """
                {
                  "backend": "coreml",
                  "model_type": "moss-transcribe-diarize-coreml",
                  "host_contract": {
                    "audio_chunk_samples": 480000,
                    "audio_tokens_per_second": 12.5,
                    "decoder_cache_length": 1024,
                    "sample_rate": 16000
                  }
                }
                """.utf8
            )
        )
        let preprocessor = try JSONDecoder().decode(
            MossPreprocessorConfiguration.self,
            from: Data(
                """
                {
                  "feature_size": 80,
                  "hop_length": 160,
                  "n_fft": 400,
                  "n_samples": 480000,
                  "nb_max_frames": 3000,
                  "sampling_rate": 16000
                }
                """.utf8
            )
        )

        XCTAssertNoThrow(try bundle.validate(decoder: decoder))
        XCTAssertNoThrow(try preprocessor.validate())
    }

    func testMismatchedFrontendIsRejected() throws {
        let preprocessor = try JSONDecoder().decode(
            MossPreprocessorConfiguration.self,
            from: Data(
                """
                {
                  "feature_size": 128,
                  "hop_length": 160,
                  "n_fft": 400,
                  "n_samples": 480000,
                  "nb_max_frames": 3000,
                  "sampling_rate": 16000
                }
                """.utf8
            )
        )

        XCTAssertThrowsError(try preprocessor.validate())
    }
}

final class MossWhisperFeatureExtractorTests: XCTestCase {
    func testMatchesHuggingFaceWhisperFrontend() throws {
        let audio = (0..<16_000).map {
            Float(($0 % 257) - 128) / 1_024
        }
        let features = try MossWhisperFeatureExtractor()
            .extractPaddedChunk(audio)

        XCTAssertEqual(features.melBins, 80)
        XCTAssertEqual(features.timeFrames, 3_000)
        XCTAssertEqual(features.data.count, 240_000)

        let fixtures: [(mel: Int, frame: Int, value: Float)] = [
            (0, 0, 1.1125666),
            (0, 1, 1.0164478),
            (0, 10, 0.99898136),
            (5, 0, 0.31692964),
            (5, 50, 0.8176489),
            (10, 99, 0.5653128),
            (20, 10, 0.5086993),
            (20, 50, 0.54108286),
            (40, 20, 0.18729377),
            (40, 99, 0.25527418),
            (79, 0, -0.8874334),
            (79, 50, 0.1336894),
            (0, 100, 0.91144305),
            (20, 100, 0.49058753),
            (40, 100, 0.29211235),
            (79, 100, 0.09761804),
            (0, 101, 0.06924677),
            (20, 250, -0.8874334),
            (79, 2_999, -0.8874334),
        ]
        for fixture in fixtures {
            let actual =
                features.data[
                    fixture.mel * features.timeFrames + fixture.frame
                ]
            XCTAssertEqual(
                actual,
                fixture.value,
                accuracy: 5e-4,
                "mel=\(fixture.mel), frame=\(fixture.frame)"
            )
        }
    }
}
