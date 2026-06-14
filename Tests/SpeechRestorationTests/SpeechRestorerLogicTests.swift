import XCTest
import AudioCommon
@testable import SpeechRestoration

/// Unit tests for `SpeechRestorer` logic that doesn't need the CoreML graphs:
/// fixed-window chunking math (the >10 s path and 48 kHz output length), the
/// downloader's path layout, and the memory-footprint / variant wiring.
final class SpeechRestorerLogicTests: XCTestCase {

    // MARK: - Window planning / chunking (>10 s, 48 kHz output length)

    func testEmptyInputPlan() {
        let p = SpeechRestorer.windowPlan(inputSamples: 0)
        XCTAssertEqual(p, .init(windows: 0, rawOutputLength: 0, outputLength: 0))
    }

    func testSingleWindowPlanShortClip() {
        // A clip shorter than one window still occupies one window and emits the
        // full per-window vocoder length, trimmed down to the clip's true 48 kHz
        // duration (3× the 16 kHz sample count, rounded).
        let c = SidonConfig.default
        let inSamples = 100_000  // ~6.25 s, < one 10 s window
        let p = SpeechRestorer.windowPlan(inputSamples: inSamples)
        XCTAssertEqual(p.windows, 1)
        XCTAssertEqual(p.rawOutputLength, c.outputSamplesPerWindow)
        let target = Int((Double(inSamples) * 3.0).rounded())
        XCTAssertEqual(p.outputLength, min(c.outputSamplesPerWindow, target))
        XCTAssertEqual(p.outputLength, target,
            "a <1-window clip is trimmed to its true 48 kHz duration")
    }

    func testExactlyOneWindowPlan() {
        let c = SidonConfig.default
        let p = SpeechRestorer.windowPlan(inputSamples: c.windowSamples)
        XCTAssertEqual(p.windows, 1)
        // 160_000 in → ideal 480_000 at 48 kHz, but the vocoder emits only
        // 479_014 (conv-stack trim), so the output is the raw vocoder length.
        XCTAssertEqual(p.rawOutputLength, c.outputSamplesPerWindow)
        XCTAssertEqual(p.outputLength, c.outputSamplesPerWindow)
    }

    func testTwoWindowPlanJustOverOneWindow() {
        // Just one sample over a full window → 2 windows. The raw vocoder output
        // is 2× per-window, but the trim clamps to the input's true 48 kHz
        // duration, discarding the (almost entirely silent) second window.
        let c = SidonConfig.default
        let inSamples = c.windowSamples + 1
        let p = SpeechRestorer.windowPlan(inputSamples: inSamples)
        XCTAssertEqual(p.windows, 2)
        XCTAssertEqual(p.rawOutputLength, 2 * c.outputSamplesPerWindow)
        let ideal = Int((Double(inSamples) * 3.0).rounded())
        XCTAssertEqual(p.outputLength, ideal,
            "a barely-over-one-window input is trimmed back to its true duration")
        XCTAssertLessThan(p.outputLength, p.rawOutputLength)
    }

    func testTwoFullWindowsKeepRawLength() {
        // Two full windows: the ideal target (2×160000×3 = 960000) exceeds raw
        // (2×479014 = 958028), so the output is the raw vocoder length.
        let c = SidonConfig.default
        let inSamples = 2 * c.windowSamples
        let p = SpeechRestorer.windowPlan(inputSamples: inSamples)
        XCTAssertEqual(p.windows, 2)
        XCTAssertEqual(p.rawOutputLength, 2 * c.outputSamplesPerWindow)
        XCTAssertEqual(p.outputLength, 2 * c.outputSamplesPerWindow)
    }

    func testManyWindowPlanForFiveMinutes() {
        // 5 min @ 16 kHz = 4_800_000 samples → 30 windows of 160_000.
        let c = SidonConfig.default
        let inSamples = 16_000 * 300
        let p = SpeechRestorer.windowPlan(inputSamples: inSamples)
        XCTAssertEqual(p.windows, 30)
        XCTAssertEqual(p.rawOutputLength, 30 * c.outputSamplesPerWindow)
        XCTAssertEqual(p.outputLength, 30 * c.outputSamplesPerWindow)
    }

    func testOutputIsAlwaysAtThreeTimesOrTrimmed() {
        // Output length is never larger than the input duration rescaled to 48 kHz.
        for s in [1, 1000, 80_000, 160_000, 240_000, 500_000] {
            let p = SpeechRestorer.windowPlan(inputSamples: s)
            let ideal = Int((Double(s) * 3.0).rounded())
            XCTAssertLessThanOrEqual(p.outputLength, ideal,
                "output (\(p.outputLength)) must not exceed ideal 48 kHz length (\(ideal)) for \(s) in")
            XCTAssertLessThanOrEqual(p.outputLength, p.rawOutputLength)
        }
    }

    func testWindowCountMatchesCeilDivide() {
        let win = SidonConfig.default.windowSamples
        for windows in 1...5 {
            // Exactly `windows` full windows.
            XCTAssertEqual(SpeechRestorer.windowPlan(inputSamples: windows * win).windows, windows)
            // One sample into the next window bumps the count.
            XCTAssertEqual(
                SpeechRestorer.windowPlan(inputSamples: windows * win + 1).windows, windows + 1)
        }
    }

    // MARK: - Downloader path layout

    func testDownloaderPathsUseVariantSubfolder() {
        let cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("sidon-dl-test", isDirectory: true)
        // ensureDownloaded is async + hits the network; just assert the
        // documented layout: bundleDir = <repoDir>/<variant.subfolder>.
        let fp16Dir = cache.appendingPathComponent(SidonVariant.fp16.subfolder, isDirectory: true)
        let int8Dir = cache.appendingPathComponent(SidonVariant.int8.subfolder, isDirectory: true)
        XCTAssertEqual(fp16Dir.lastPathComponent, "fp16")
        XCTAssertEqual(int8Dir.lastPathComponent, "int8")
    }

    // MARK: - Variant selection / config defaults

    func testVariantSelection() {
        XCTAssertEqual(SidonVariant.fp16.subfolder, "fp16")
        XCTAssertEqual(SidonVariant.int8.subfolder, "int8")
        XCTAssertEqual(Set(SidonVariant.allCases), [.fp16, .int8])
        XCTAssertEqual(SidonVariant(rawValue: "fp16"), .fp16)
        XCTAssertEqual(SidonVariant(rawValue: "int8"), .int8)
        XCTAssertNil(SidonVariant(rawValue: "bf16"))
    }

    func testConfigDefaultsConsistency() {
        let c = SidonConfig.default
        // featureDim == 80 mels × stride 2.
        XCTAssertEqual(c.featureDim, SeamlessM4TFrontEnd.numMelBins * SeamlessM4TFrontEnd.stride)
        // 48 kHz is exactly 3× 16 kHz.
        XCTAssertEqual(c.outputSampleRate, c.inputSampleRate * 3)
        // The per-window output is just under the ideal 3× window length.
        XCTAssertLessThanOrEqual(c.outputSamplesPerWindow, c.windowSamples * 3)
        XCTAssertGreaterThan(c.outputSamplesPerWindow, c.windowSamples * 3 - 2000)
    }

    // MARK: - Memory footprint / variant wiring

    func testMemoryFootprintDiffersByVariant() {
        // fp16 weights are larger on disk than the int8-palettized predictor.
        let fp16 = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .fp16, loaded: true)
        let int8 = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .int8, loaded: true)
        XCTAssertGreaterThan(fp16.memoryFootprint, int8.memoryFootprint)
        XCTAssertGreaterThan(int8.memoryFootprint, 0)
    }

    func testMemoryFootprintZeroAfterUnload() {
        let r = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .fp16, loaded: true)
        XCTAssertTrue(r.isLoaded)
        XCTAssertGreaterThan(r.memoryFootprint, 0)
        r.unload()
        XCTAssertFalse(r.isLoaded)
        XCTAssertEqual(r.memoryFootprint, 0)
    }

    func testRestoreThrowsWhenUnloaded() {
        let r = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .fp16, loaded: true)
        r.unload()
        XCTAssertThrowsError(try r.restore(audio: [Float](repeating: 0, count: 16000), sampleRate: 16000))
        XCTAssertThrowsError(try r.restoreWindow(samples: [Float](repeating: 0, count: 16000)))
    }

    // MARK: - Protocol conformance constants

    func testSpeechEnhancementModelConformance() {
        let r = SpeechRestorer(
            predictor: nil, vocoder: nil, config: .default, variant: .fp16, loaded: true)
        // The protocol advertises the front-end's 16 kHz input rate.
        XCTAssertEqual(r.inputSampleRate, 16_000)
        XCTAssertEqual(SpeechRestorer.inputSampleRate, 16_000)
        XCTAssertEqual(SpeechRestorer.outputSampleRate, 48_000)
    }
}
