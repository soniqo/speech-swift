import XCTest
@testable import SpeechVAD
import AudioCommon

// MARK: - Unit Tests

final class FireRedVADTests: XCTestCase {

    func testKaldiFbankExtractorOutputShape() {
        let extractor = KaldiFbankExtractor()
        // 1 second of audio at 16kHz = 16000 samples
        // With 25ms window and 10ms shift, snip_edges: (16000 - 400) / 160 + 1 = 98 frames
        let audio = [Float](repeating: 0.1, count: 16000)
        let features = extractor.extract(audio)
        XCTAssertEqual(features.count, 98 * 80)
    }

    func testKaldiFbankExtractorEmptyInput() {
        let extractor = KaldiFbankExtractor()
        let features = extractor.extract([])
        XCTAssertTrue(features.isEmpty)
    }

    func testKaldiFbankExtractorShortInput() {
        let extractor = KaldiFbankExtractor()
        // Less than one frame (400 samples needed)
        let features = extractor.extract([Float](repeating: 0, count: 100))
        XCTAssertTrue(features.isEmpty)
    }

    func testKaldiFbankExtractorSilenceFloor() {
        let extractor = KaldiFbankExtractor()
        // All zeros should produce log-floored values (~-15.94)
        let audio = [Float](repeating: 0, count: 800)
        let features = extractor.extract(audio)
        XCTAssertFalse(features.isEmpty)
        // All values should be the floor value (log(FLT_EPSILON) ≈ -15.94)
        for val in features {
            XCTAssertLessThan(val, -10.0, "Silence should produce low energy")
        }
    }

    func testKaldiFbankExtractorNonSilence() {
        let extractor = KaldiFbankExtractor()
        // 1kHz sine wave should produce non-floor values
        var audio = [Float](repeating: 0, count: 16000)
        for i in 0..<16000 {
            audio[i] = 0.5 * sin(2.0 * Float.pi * 1000.0 * Float(i) / 16000.0)
        }
        let features = extractor.extract(audio)
        let maxVal = features.max() ?? -100
        XCTAssertGreaterThan(maxVal, 0.0, "Non-silence should have positive log energy in some bins")
    }

    func testDefaultConfig() {
        // Verify default thresholds
        let model = FireRedVADModel.defaultModelId
        XCTAssertEqual(model, "aufklarer/FireRedVAD-CoreML")
    }
}

// MARK: - E2E Tests

final class E2EFireRedVADTests: XCTestCase {

    func testE2EDetection() async throws {
        let vad = try await FireRedVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")

        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        XCTAssertGreaterThan(audio.count, 0)

        let segments = vad.detectSpeech(audio: audio, sampleRate: 16000)
        XCTAssertFalse(segments.isEmpty, "Should detect speech in test audio")

        // Test audio has speech from ~5.17s to ~8.37s
        let firstSeg = segments[0]
        XCTAssertEqual(firstSeg.startTime, 5.17, accuracy: 0.3,
                       "Speech start should be ~5.17s")
        XCTAssertEqual(firstSeg.endTime, 8.37, accuracy: 0.3,
                       "Speech end should be ~8.37s")
    }

    func testE2ELatency() async throws {
        let vad = try await FireRedVADModel.fromPretrained()

        let audioURL = URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav")
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        let start = CFAbsoluteTimeGetCurrent()
        _ = vad.detectSpeech(audio: audio, sampleRate: 16000)
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        let duration = Float(audio.count) / 16000.0
        let rtf = elapsed / Double(duration)
        print("FireRedVAD: \(String(format: "%.3f", elapsed))s for \(String(format: "%.1f", duration))s audio, RTF=\(String(format: "%.4f", rtf))")

        XCTAssertLessThan(rtf, 1.0, "RTF should be under 1.0 (real-time)")
    }
}
