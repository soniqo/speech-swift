import AudioCommon
import Tokenizers
import XCTest
@testable import MossTranscribe

@available(macOS 15.0, iOS 18.0, *)
final class E2EMossTranscribeTests: XCTestCase {
    func testSwiftTokenizerPromptMatchesPinnedReference() async throws {
        guard let directory = ProcessInfo.processInfo.environment[
            "MOSS_TOKENIZER_DIR"
        ] else {
            throw XCTSkip("set MOSS_TOKENIZER_DIR to a MOSS model bundle")
        }
        let folder = URL(fileURLWithPath: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let configuration = try JSONDecoder().decode(
            MossProcessorConfiguration.self,
            from: Data(
                contentsOf: folder.appendingPathComponent(
                    "processor_config.json"
                )
            )
        )
        let processor = try MossPromptProcessor(
            tokenizer: tokenizer,
            configuration: configuration
        )
        let prepared = try processor.prepare(audioTokenCount: 13)

        XCTAssertEqual(
            prepared.inputIDs,
            [
                151_644, 8_948, 198, 2_610, 525, 264, 10_950,
                17_847, 13, 151_645, 198, 151_644, 872, 198,
                151_669, 151_671, 151_671, 151_671, 151_671,
                151_671, 151_671, 151_671, 151_671, 151_671,
                151_671, 151_671, 151_671, 151_671, 151_670,
                198, 14_880, 44_063, 111_268, 46_670, 61_443,
                17_714, 108_704, 3_837, 73_157, 104_383, 58_362,
                23_031, 71_618, 26_606, 20_450, 111_420, 33_108,
                104_283, 17_340, 72_640, 9_909, 58, 50, 15, 16,
                60, 5_373, 58, 50, 15, 17, 60, 5_373, 58, 50,
                15, 18, 60, 1_940, 7_552, 111_749, 3_837, 110_644,
                17_714, 110_019, 105_761, 43_815, 90_395, 18_493,
                37_474, 100_072, 111_066, 80_565, 20_450, 111_420,
                3_837, 23_031, 104_542, 117_932, 75_882, 37_474,
                105_761, 101_121, 1_773, 151_645, 198, 151_644,
                77_091, 198,
            ]
        )
    }

    func testINT8RoundTripProducesStructuredTranscript() async throws {
        try await assertRoundTrip(
            variant: .int8,
            localEnvironment: "MOSS_E2E_MODEL_DIR"
        )
    }

    func testFP16RoundTripProducesStructuredTranscript() async throws {
        try await assertRoundTrip(
            variant: .fp16,
            localEnvironment: "MOSS_E2E_FP16_MODEL_DIR"
        )
    }

    func testMLXINT5RoundTripProducesStructuredTranscript() async throws {
        try await assertMLXRoundTrip(
            variant: .int5,
            localEnvironment: "MOSS_E2E_MLX_INT5_MODEL_DIR"
        )
    }

    func testMLXINT8RoundTripProducesStructuredTranscript() async throws {
        try await assertMLXRoundTrip(
            variant: .int8,
            localEnvironment: "MOSS_E2E_MLX_INT8_MODEL_DIR"
        )
    }

    func testMLXINT5WeightsWithINT8KVCacheProduceStructuredTranscript()
        async throws
    {
        try await assertMLXINT8KVCache(
            variant: .int5,
            localEnvironment: "MOSS_E2E_MLX_INT5_MODEL_DIR"
        )
    }

    func testMLXINT8WeightsWithINT8KVCacheProduceStructuredTranscript()
        async throws
    {
        try await assertMLXINT8KVCache(
            variant: .int8,
            localEnvironment: "MOSS_E2E_MLX_INT8_MODEL_DIR"
        )
    }

    private func assertMLXINT8KVCache(
        variant: MossMLXVariant,
        localEnvironment: String
    ) async throws {
        let (audio, model) = try await loadMLXFixture(
            variant: variant,
            localEnvironment: localEnvironment
        )
        let reference =
            "Can you guarantee that the replacement part will be shipped tomorrow?"

        let result = try model.transcribeDetailed(
            audio: audio,
            sampleRate: MossMLXModel.inputSampleRate,
            options: MossMLXDecodingOptions(
                kvCachePrecision: .int8
            )
        )

        XCTAssertEqual(result.text, reference)
        XCTAssertEqual(
            result.segments,
            [
                MossTranscriptSegment(
                    startTime: 5,
                    endTime: 8.4,
                    speaker: "S01",
                    text: reference
                )
            ]
        )
        XCTAssertEqual(result.metrics.stopReason, .endOfSequence)
    }

    private func assertRoundTrip(
        variant: MossModelVariant,
        localEnvironment: String
    ) async throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let audioURL = repositoryRoot
            .appendingPathComponent(
                "Tests/Qwen3ASRTests/Resources/test_audio.wav"
            )
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("shared ASR test_audio.wav is unavailable")
        }
        let audio = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: MossTranscribeModel.inputSampleRate
        )

        let model: MossTranscribeModel
        if let local = ProcessInfo.processInfo.environment[
            localEnvironment
        ] {
            model = try await MossTranscribeModel.fromDirectory(
                URL(fileURLWithPath: local),
                modelId:
                    "local/MOSS-CoreML-\(variant.rawValue.uppercased())"
            )
        } else {
            model = try await MossTranscribeModel.fromPretrained(
                variant: variant
            )
        }
        try model.warmUp()

        let result = try model.transcribeDetailed(
            audio: audio,
            sampleRate: MossTranscribeModel.inputSampleRate
        )
        let reference =
            "Can you guarantee that the replacement part will be shipped tomorrow?"
        XCTAssertEqual(result.text, reference)
        XCTAssertEqual(
            result.segments,
            [
                MossTranscriptSegment(
                    startTime: 5,
                    endTime: 8.4,
                    speaker: "S01",
                    text: reference
                )
            ]
        )
        let normalized = result.text.lowercased()
        for keyword in ["guarantee", "replacement", "shipped", "tomorrow"] {
            XCTAssertTrue(
                normalized.contains(keyword),
                "Expected \(keyword) in: \(result.text)"
            )
        }
        XCTAssertFalse(result.rawText.isEmpty)
        XCTAssertFalse(result.segments.isEmpty)
        XCTAssertTrue(
            result.segments.allSatisfy {
                $0.speaker.hasPrefix("S")
                    && $0.endTime >= $0.startTime
                    && !$0.text.isEmpty
            }
        )
        XCTAssertEqual(result.metrics.stopReason, .endOfSequence)
        XCTAssertGreaterThan(result.metrics.generatedTokens, 0)

        print(
            String(
                format:
                    "[MOSS-COREML-\(variant.rawValue.uppercased())] RTF=%.4f throughput=%.2fx preprocessing=%.3fs encoder=%.3fs prefill=%.3fs decode=%.3fs prompt=%d generated=%d",
                result.metrics.realTimeFactor,
                result.metrics.realtimeThroughput,
                result.metrics.preprocessingSeconds,
                result.metrics.audioEncoderSeconds,
                result.metrics.decoderPrefillSeconds,
                result.metrics.tokenDecodeSeconds,
                result.metrics.promptTokens,
                result.metrics.generatedTokens
            )
        )
    }

    private func assertMLXRoundTrip(
        variant: MossMLXVariant,
        localEnvironment: String
    ) async throws {
        let (audio, model) = try await loadMLXFixture(
            variant: variant,
            localEnvironment: localEnvironment
        )
        let result = try model.transcribeDetailed(
            audio: audio,
            sampleRate: MossMLXModel.inputSampleRate
        )
        let reference =
            "Can you guarantee that the replacement part will be shipped tomorrow?"
        XCTAssertEqual(result.text, reference)
        XCTAssertEqual(
            result.segments,
            [
                MossTranscriptSegment(
                    startTime: 5,
                    endTime: 8.4,
                    speaker: "S01",
                    text: reference
                )
            ]
        )
        XCTAssertEqual(result.metrics.stopReason, .endOfSequence)
        XCTAssertGreaterThan(result.metrics.promptTokens, 0)
        XCTAssertGreaterThan(result.metrics.generatedTokens, 0)

        print(
            String(
                format:
                    "[MOSS-MLX-\(variant.rawValue.uppercased())] RTF=%.4f throughput=%.2fx preprocessing=%.3fs encoder=%.3fs prefill=%.3fs decode=%.3fs prompt=%d generated=%d",
                result.metrics.realTimeFactor,
                result.metrics.realtimeThroughput,
                result.metrics.preprocessingSeconds,
                result.metrics.audioEncoderSeconds,
                result.metrics.decoderPrefillSeconds,
                result.metrics.tokenDecodeSeconds,
                result.metrics.promptTokens,
                result.metrics.generatedTokens
            )
        )
    }

    private func loadMLXFixture(
        variant: MossMLXVariant,
        localEnvironment: String
    ) async throws -> (audio: [Float], model: MossMLXModel) {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let audioURL = repositoryRoot.appendingPathComponent(
            "Tests/Qwen3ASRTests/Resources/test_audio.wav"
        )
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            throw XCTSkip("shared ASR test_audio.wav is unavailable")
        }
        let audio = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: MossMLXModel.inputSampleRate
        )

        let model: MossMLXModel
        if let local = ProcessInfo.processInfo.environment[
            localEnvironment
        ] {
            model = try await MossMLXModel.fromDirectory(
                URL(fileURLWithPath: local),
                modelId: "local/MOSS-MLX-\(variant.rawValue.uppercased())"
            )
        } else {
            model = try await MossMLXModel.fromPretrained(
                variant: variant
            )
        }
        return (audio, model)
    }
}
