#if canImport(CoreML)
import XCTest
import Foundation
import AudioCommon
@testable import Qwen3ASR

/// Local-bundle smoke test for ``CoreMLForcedAligner``. Skips by default
/// unless ``ALIGNER_COREML_LOCAL_DIR`` points at a freshly produced
/// converter output. Runs the full encode → embed → decode → LIS pipeline
/// against the local mlpackages so we can validate the runtime *before*
/// uploading to HuggingFace.
///
/// Invoke with:
///   ALIGNER_COREML_LOCAL_DIR=/tmp/aligner-coreml-fp16 \
///       swift test --filter E2ECoreMLForcedAlignerLocalSmokeTests
final class E2ECoreMLForcedAlignerLocalSmokeTests: XCTestCase {

    func testLocalAlignerProducesFiniteMonotonicWords() throws {
        guard let path = ProcessInfo.processInfo.environment["ALIGNER_COREML_LOCAL_DIR"],
              !path.isEmpty else {
            throw XCTSkip("Set ALIGNER_COREML_LOCAL_DIR to a converter output directory")
        }
        let dir = URL(fileURLWithPath: path, isDirectory: true)
        guard FileManager.default.fileExists(atPath: dir.path) else {
            throw XCTSkip("ALIGNER_COREML_LOCAL_DIR=\(path) does not exist")
        }

        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in Qwen3ASRTests resources")
        }

        let aligner = try CoreMLForcedAligner.fromDirectory(dir)
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 16000
        let audio: [Float] = sampleRate == targetSampleRate
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        let audioSeconds = Float(audio.count) / Float(targetSampleRate)

        let referenceText =
            "Can you guarantee that the replacement part will be shipped tomorrow?"
        let start = CFAbsoluteTimeGetCurrent()
        let aligned = try aligner.align(
            audio: audio,
            text: referenceText,
            sampleRate: targetSampleRate,
            language: "English"
        )
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let audioMs = Double(audioSeconds) * 1000

        print(String(format: "[COREML-ALIGN-LOCAL] dir=%@", dir.path))
        print(String(format: "[COREML-ALIGN-LOCAL-PERF] align=%.0fms audio=%.0fms rtf=%.3f words=%d",
                     elapsedMs, audioMs, elapsedMs / audioMs, aligned.count))
        for w in aligned {
            print(String(format: "  [%.2f-%.2f] %@", w.startTime, w.endTime, w.text))
        }

        XCTAssertFalse(aligned.isEmpty, "Aligner returned no words")
        for word in aligned {
            XCTAssertTrue(word.startTime.isFinite, "non-finite start for '\(word.text)'")
            XCTAssertTrue(word.endTime.isFinite, "non-finite end for '\(word.text)'")
            XCTAssertGreaterThanOrEqual(word.startTime, 0,
                                        "negative start for '\(word.text)'")
        }
        for i in 1..<aligned.count {
            XCTAssertGreaterThanOrEqual(
                aligned[i].startTime, aligned[i - 1].startTime,
                "Non-monotonic at index \(i)")
        }

        guard let last = aligned.last else { return }
        XCTAssertLessThanOrEqual(last.endTime, audioSeconds + 1.0,
                                 "Last word ends past audio end")
        let span = last.endTime - aligned.first!.startTime
        XCTAssertGreaterThan(span, 0.5,
                             "Alignment span \(span)s too small — likely NaN/argmax-0 collapse")
    }
}
#endif
