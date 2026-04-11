import XCTest
@testable import SpeechUI

final class WaveformDownsampleTests: XCTestCase {

    func testEmptyInputReturnsEmpty() {
        XCTAssertEqual(WaveformView.downsamplePeaks([], into: 16), [])
    }

    func testZeroBucketsReturnsEmpty() {
        XCTAssertEqual(WaveformView.downsamplePeaks([0.5, -0.5], into: 0), [])
    }

    func testInputSmallerThanBucketsReturnsAbs() {
        let result = WaveformView.downsamplePeaks([-0.4, 0.6, -0.1], into: 16)
        XCTAssertEqual(result, [0.4, 0.6, 0.1])
    }

    func testDownsamplePicksPeakPerBucket() {
        // 16 samples → 4 bars. Each bucket of 4 samples picks the absolute peak.
        let samples: [Float] = [
            0.0, 0.1, 0.2, 0.3,        // peak 0.3
            -0.4, 0.5, -0.6, 0.7,      // peak 0.7
            0.0, 0.0, 0.0, 0.0,        // peak 0.0
            -0.9, -0.8, -0.95, -0.7,   // peak 0.95
        ]
        let bars = WaveformView.downsamplePeaks(samples, into: 4)
        XCTAssertEqual(bars.count, 4)
        XCTAssertEqual(bars[0], 0.3, accuracy: 1e-6)
        XCTAssertEqual(bars[1], 0.7, accuracy: 1e-6)
        XCTAssertEqual(bars[2], 0.0, accuracy: 1e-6)
        XCTAssertEqual(bars[3], 0.95, accuracy: 1e-6)
    }

    func testDownsampleHandlesNonDivisibleBuckets() {
        // 10 samples → 3 bars. Buckets are unequal but every sample contributes.
        let samples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        let bars = WaveformView.downsamplePeaks(samples, into: 3)
        XCTAssertEqual(bars.count, 3)
        XCTAssertGreaterThan(bars[2], bars[0], "Later bars should reflect higher-magnitude samples")
    }
}

final class MicLevelTests: XCTestCase {

    func testRMSOfSilenceIsZero() {
        XCTAssertEqual(MicLevelView.rmsLevel(samples: Array(repeating: 0, count: 256)), 0)
    }

    func testRMSOfEmptyIsZero() {
        XCTAssertEqual(MicLevelView.rmsLevel(samples: []), 0)
    }

    func testRMSOfFullScaleIsOne() {
        let level = MicLevelView.rmsLevel(samples: Array(repeating: 1, count: 64))
        XCTAssertEqual(level, 1.0, accuracy: 1e-6)
    }

    func testRMSIsNonNegative() {
        let level = MicLevelView.rmsLevel(samples: [-0.5, 0.5, -0.5, 0.5])
        XCTAssertEqual(level, 0.5, accuracy: 1e-6)
    }

    func testRMSClampedToOne() {
        // Out-of-range samples (shouldn't happen for proper PCM but defensively clamp)
        let level = MicLevelView.rmsLevel(samples: [3.0, -3.0])
        XCTAssertLessThanOrEqual(level, 1.0)
    }
}

@MainActor
final class TranscriptionStoreTests: XCTestCase {

    func testInitiallyEmpty() {
        let store = TranscriptionStore()
        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testPartialUpdatesPartialOnly() {
        let store = TranscriptionStore()
        store.apply(text: "hello", isFinal: false)
        XCTAssertEqual(store.currentPartial, "hello")
        XCTAssertTrue(store.finalLines.isEmpty)

        store.apply(text: "hello world", isFinal: false)
        XCTAssertEqual(store.currentPartial, "hello world")
        XCTAssertTrue(store.finalLines.isEmpty)
    }

    func testFinalCommitsAndClearsPartial() {
        let store = TranscriptionStore()
        store.apply(text: "hello world", isFinal: false)
        store.apply(text: "hello world", isFinal: true)

        XCTAssertEqual(store.finalLines, ["hello world"])
        XCTAssertNil(store.currentPartial)
    }

    func testMultipleFinalsAccumulate() {
        let store = TranscriptionStore()
        store.apply(text: "first sentence", isFinal: true)
        store.apply(text: "second sentence", isFinal: true)
        store.apply(text: "third sentence", isFinal: true)

        XCTAssertEqual(store.finalLines, ["first sentence", "second sentence", "third sentence"])
        XCTAssertNil(store.currentPartial)
    }

    func testEmptyFinalIsIgnored() {
        let store = TranscriptionStore()
        store.apply(text: "   ", isFinal: true)
        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testEmptyPartialClearsPartialState() {
        let store = TranscriptionStore()
        store.apply(text: "hello", isFinal: false)
        store.apply(text: "", isFinal: false)
        XCTAssertNil(store.currentPartial)
    }

    func testWhitespaceTrimming() {
        let store = TranscriptionStore()
        store.apply(text: "  hello  ", isFinal: true)
        XCTAssertEqual(store.finalLines, ["hello"])
    }

    func testReset() {
        let store = TranscriptionStore()
        store.apply(text: "first", isFinal: true)
        store.apply(text: "in progress", isFinal: false)

        store.reset()

        XCTAssertTrue(store.finalLines.isEmpty)
        XCTAssertNil(store.currentPartial)
    }

    func testInterleavedPartialsAndFinals() {
        let store = TranscriptionStore()
        store.apply(text: "first", isFinal: false)
        store.apply(text: "first sentence", isFinal: true)
        store.apply(text: "sec", isFinal: false)
        store.apply(text: "second", isFinal: false)
        store.apply(text: "second sentence", isFinal: true)

        XCTAssertEqual(store.finalLines, ["first sentence", "second sentence"])
        XCTAssertNil(store.currentPartial)
    }
}

final class SpeechUIVersionTests: XCTestCase {
    func testVersionExposed() {
        XCTAssertFalse(SpeechUI.version.isEmpty)
    }
}
