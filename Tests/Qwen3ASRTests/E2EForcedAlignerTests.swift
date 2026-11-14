import XCTest
import AudioCommon
@testable import Qwen3ASR

/// Prompt-layout regression using a fixture with about five seconds of leading silence.
final class E2EForcedAlignerTests: XCTestCase {
    func testLeadingWordStartsAfterInitialSilence() async throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        let (audio, sampleRate) = try AudioFileLoader.loadWAV(url: url)
        let aligner = try await Qwen3ForcedAligner.fromPretrained(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-4bit")
        let aligned = aligner.align(
            audio: audio,
            text: "Can you guarantee that the replacement part will be shipped tomorrow?",
            sampleRate: sampleRate)

        // PyTorch reference for this fixture (issue #546). Reading a timestamp
        // slot before the first word incorrectly puts its start at 0 s.
        XCTAssertEqual(aligned.count, 11)
        let first = try XCTUnwrap(aligned.first)
        XCTAssertEqual(first.text, "Can")
        XCTAssertEqual(first.startTime, 5.20, accuracy: 0.24)
    }
}
