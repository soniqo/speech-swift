#if canImport(CoreML)
import XCTest
import Foundation
import AudioCommon
@testable import Qwen3ASR

/// End-to-end tests for the CoreML forced aligner pipeline (audio_encoder
/// + embedding + text_decoder + classify head). Loads the published
/// ``aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16`` bundle and verifies
/// that the produced alignment is finite, monotonic, and lands inside the
/// audio clip's duration window.
///
/// Regression coverage:
/// - Catches the prior NaN bug in the text decoder (the upstream fp16
///   softmax of an ``-inf`` causal mask), now fixed by exporting the mask
///   as a constant with a finite ``-1e4`` fill value. A reintroduction of
///   ``-inf`` in the mask would surface here as all-zero indices and
///   degenerate ``[0.0s, 0.0s]`` word stamps.
/// - Catches the prior audio-encoder divergence (global attention over
///   zero-padded mel) by checking that the alignment durations roughly
///   match the real audio length, not the 30 s padded grid.
final class E2ECoreMLForcedAlignerTests: XCTestCase {

    private static let referenceText =
        "Can you guarantee that the replacement part will be shipped tomorrow?"

    func testFP16AlignerProducesMonotonicTimestamps() async throws {
        try await runAligner(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16",
            variantLabel: "FP16"
        )
    }

    func testINT8AlignerProducesMonotonicTimestamps() async throws {
        try await runAligner(
            modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
            variantLabel: "INT8"
        )
    }

    // MARK: - Shared body

    private func runAligner(modelId: String, variantLabel: String) async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in Qwen3ASRTests resources")
        }

        let aligner: CoreMLForcedAligner
        do {
            aligner = try await CoreMLForcedAligner.fromPretrained(
                modelId: modelId
            ) { progress, status in
                if Int(progress * 100) % 25 == 0 {
                    print(String(format: "  [%@] loading: %.0f%% — %@",
                                 variantLabel, progress * 100, status))
                }
            }
        } catch {
            throw XCTSkip("CoreML aligner \(variantLabel) bundle unavailable: \(error)")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 16000
        let audio: [Float] = sampleRate == targetSampleRate
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        let audioSeconds = Float(audio.count) / Float(targetSampleRate)
        XCTAssertGreaterThan(audioSeconds, 0)

        let start = CFAbsoluteTimeGetCurrent()
        let aligned = try aligner.align(
            audio: audio,
            text: Self.referenceText,
            sampleRate: targetSampleRate,
            language: "English"
        )
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let audioMs = Double(audioSeconds) * 1000

        print(String(format: "[COREML-ALIGN-%@-PERF] align=%.0fms audio=%.0fms rtf=%.3f words=%d",
                     variantLabel, elapsedMs, audioMs, elapsedMs / audioMs, aligned.count))
        for word in aligned {
            print(String(format: "  [%.2f-%.2f] %@", word.startTime, word.endTime, word.text))
        }

        XCTAssertFalse(aligned.isEmpty, "[\(variantLabel)] Aligner returned no words")
        XCTAssertEqual(
            aligned.count,
            Self.referenceText.split(separator: " ").map { $0.trimmingCharacters(in: .punctuationCharacters) }
                .filter { !$0.isEmpty }.count,
            "[\(variantLabel)] Aligner should emit one entry per word in the reference text"
        )

        // No NaN, no negatives, no end-before-start.
        for word in aligned {
            XCTAssertTrue(word.startTime.isFinite, "[\(variantLabel)] non-finite start for '\(word.text)'")
            XCTAssertTrue(word.endTime.isFinite, "[\(variantLabel)] non-finite end for '\(word.text)'")
            XCTAssertGreaterThanOrEqual(word.startTime, 0,
                                        "[\(variantLabel)] negative start for '\(word.text)'")
            XCTAssertGreaterThanOrEqual(word.endTime, word.startTime,
                                        "[\(variantLabel)] end < start for '\(word.text)'")
        }

        // Monotonically non-decreasing across the sequence.
        for i in 1..<aligned.count {
            XCTAssertGreaterThanOrEqual(
                aligned[i].startTime, aligned[i - 1].startTime,
                "[\(variantLabel)] Non-monotonic start at index \(i): "
                + "\(aligned[i - 1].text)@\(aligned[i - 1].startTime) → "
                + "\(aligned[i].text)@\(aligned[i].startTime)")
        }

        // Timestamps should land inside the clip's duration window (with a
        // small tolerance for the trailing word's end overshooting slightly).
        guard let lastWord = aligned.last else {
            XCTFail("[\(variantLabel)] No words to inspect")
            return
        }
        XCTAssertLessThanOrEqual(
            lastWord.startTime, audioSeconds + 0.5,
            "[\(variantLabel)] Last word starts past audio end "
            + "(\(lastWord.startTime)s vs audio \(audioSeconds)s)")
        XCTAssertLessThanOrEqual(
            lastWord.endTime, audioSeconds + 1.0,
            "[\(variantLabel)] Last word ends past audio end "
            + "(\(lastWord.endTime)s vs audio \(audioSeconds)s)")

        // The total alignment should cover a meaningful fraction of the clip.
        // The fixture is ~3-4 s of speech; if alignment collapses to 0 s, the
        // CoreML graph likely regressed (audio encoder divergence or text
        // decoder NaN producing argmax=0 everywhere).
        let span = lastWord.endTime - aligned.first!.startTime
        XCTAssertGreaterThan(
            span, 0.5,
            "[\(variantLabel)] Alignment span \(span)s is too small — "
            + "suggests NaN/argmax-0 collapse")
    }
}
#endif
