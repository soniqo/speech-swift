import XCTest
import AudioCommon
@testable import Qwen3ASR

/// Word-boundary regression for mel features, timestamp slots and monotonicity correction.
extension E2EForcedAlignerTests {
    func testMLXWordBoundsMatchReference() async throws {
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
        let words = [
            "Can", "you", "guarantee", "that", "the", "replacement", "part", "will", "be", "shipped", "tomorrow?"
        ]
        let starts: [Float] = [
            5.20, 5.36, 5.44, 6.08, 6.24, 6.32, 6.96, 7.28, 7.36, 7.44, 7.84
        ]
        let ends: [Float] = [
            5.36, 5.44, 6.08, 6.24, 6.32, 6.96, 7.28, 7.36, 7.44, 7.84, 8.32
        ]
        XCTAssertEqual(aligned.map(\.text), words)
        for (word, expected) in zip(aligned, zip(starts, ends)) {
            print("[ALIGNMENT-REFERENCE] \(word.text) \(word.startTime)-\(word.endTime) expected=\(expected)")
            XCTAssertEqual(word.startTime, expected.0, accuracy: 0.24,
                           "Incorrect start time for \(word.text)")
            XCTAssertEqual(word.endTime, expected.1, accuracy: 0.24,
                           "Incorrect end time for \(word.text)")
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime)
        }
    }
}
