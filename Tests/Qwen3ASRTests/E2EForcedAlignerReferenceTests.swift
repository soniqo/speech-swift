import XCTest
import AudioCommon
@testable import Qwen3ASR

/// Word-start timing regression for corrected mel features and word-first timestamp slots.
extension E2EForcedAlignerTests {
    func testMLXWordStartsMatchReference() async throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        let (audio, sampleRate) = try AudioFileLoader.loadWAV(url: url)
        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit")
        let aligned = aligner.align(
            audio: audio,
            text: "Can you guarantee that the replacement part will be shipped tomorrow?",
            sampleRate: sampleRate)

        // PyTorch Qwen3-ForcedAligner reference for this fixture (issue #546).
        // Allow three 80 ms classes for quantization and resampling differences.
        let words = ["Can", "you", "guarantee", "that", "the", "replacement",
                     "part", "will", "be", "shipped", "tomorrow?"]
        let starts: [Float] = [5.28, 5.52, 5.52, 6.16, 6.80, 6.88, 7.04, 7.36, 7.44, 8.00, 8.40]
        XCTAssertEqual(aligned.map(\.text), words)
        for (word, expected) in zip(aligned, starts) {
            print("[ALIGNMENT-REFERENCE] \(word.text) \(word.startTime)-\(word.endTime) expected=\(expected)")
            XCTAssertEqual(word.startTime, expected, accuracy: 0.24,
                           "Incorrect start time for \(word.text)")
            XCTAssertTrue(word.endTime.isFinite)
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime)
        }
    }
}
