import XCTest
@testable import SpeechRestoration

/// Unit tests for Sidon config + variant wiring (no model download).
final class SpeechRestorationConfigTests: XCTestCase {

    func testVariantSubfoldersAndRepo() {
        XCTAssertEqual(SidonVariant.fp16.subfolder, "fp16")
        XCTAssertEqual(SidonVariant.int8.subfolder, "int8")
        XCTAssertEqual(SidonVariant.fp16.defaultModelId, "aufklarer/Sidon-CoreML")
        XCTAssertEqual(SidonVariant.allCases.count, 2)
        XCTAssertEqual(SidonVariant(rawValue: "fp16"), .fp16)
        XCTAssertEqual(SidonVariant(rawValue: "int8"), .int8)
        XCTAssertNil(SidonVariant(rawValue: "int4"))
    }

    func testDefaultConfigMatchesExport() {
        let c = SidonConfig.default
        XCTAssertEqual(c.inputSampleRate, 16_000)
        XCTAssertEqual(c.outputSampleRate, 48_000)
        XCTAssertEqual(c.frames, 499)
        XCTAssertEqual(c.hiddenSize, 1024)
        XCTAssertEqual(c.featureDim, 160)
        // The fixed window is exactly 10 s of 16 kHz audio → 499 stacked frames.
        XCTAssertEqual(c.windowSamples, 160_000)
        XCTAssertEqual(c.outputSamplesPerWindow, 479_014)
    }

    func testWindowSpansTenSeconds() {
        let c = SidonConfig.default
        // 499 stacked frames ⇒ 998 mel frames; 1 + (N-400)/160 = 998 ⇒ N≈159_920;
        // the extractor yields exactly 499 stacked frames at 160_000 samples.
        let (_, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0, count: c.windowSamples))
        XCTAssertEqual(frames, c.frames,
            "A full \(c.windowSamples)-sample window must produce exactly \(c.frames) frames")
    }

    func testEngineConstants() {
        XCTAssertEqual(SpeechRestorer.inputSampleRate, 16_000)
        XCTAssertEqual(SpeechRestorer.outputSampleRate, 48_000)
        XCTAssertEqual(SpeechRestorer.defaultModelId, "aufklarer/Sidon-CoreML")
    }
}
