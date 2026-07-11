import XCTest
@testable import SpeechRestoration

/// DSP unit tests for the SeamlessM4T log-mel front-end.
///
/// Golden values come from the reference HuggingFace
/// `SeamlessM4TFeatureExtractor` (the extractor `facebook/w2v-bert-2.0` ships),
/// for a fully reproducible synthetic input. Tolerances absorb float32-vs-float64
/// and FFT-implementation rounding (the Python reference computes in float64).
final class SeamlessM4TFrontEndTests: XCTestCase {

    /// Reproducible, well-conditioned test signal: a linear chirp 100→2000 Hz
    /// plus a small fixed ripple, 1.0 s @ 16 kHz. Non-stationary so the per-bin
    /// variance the front-end normalizes by is healthy (a pure stationary tone
    /// is ill-conditioned for that step and amplifies float32 noise).
    ///
    ///   `phase = 2π·(f0·t + (f1-f0)/(2T)·t²)`,  `x = 0.4·sin(phase) + 0.05·sin(2π·123·t)`
    private func testSignal(_ n: Int = 16000) -> [Float] {
        var x = [Float](repeating: 0, count: n)
        let f0: Float = 100, f1: Float = 2000, T: Float = 1.0
        for i in 0..<n {
            let t = Float(i) / 16000.0
            let phase = 2 * .pi * (f0 * t + (f1 - f0) / (2 * T) * t * t)
            x[i] = 0.4 * Foundation.sin(phase) + 0.05 * Foundation.sin(2 * .pi * 123 * t)
        }
        return x
    }

    func testWindowTable() {
        let w = SeamlessM4TFrontEnd.poveyWindow(length: 400)
        XCTAssertEqual(w.count, 400)
        // Reference: sum=212.14698, w[1]=0.000265, w[200]=0.999987, w[399]=0.000000
        let sum = w.reduce(0, +)
        XCTAssertEqual(sum, 212.14698, accuracy: 0.01)
        XCTAssertEqual(w[1], 0.000265, accuracy: 1e-5)
        XCTAssertEqual(w[200], 0.999987, accuracy: 1e-4)
        XCTAssertEqual(w[399], 0.0, accuracy: 1e-5)
        // Symmetric window.
        XCTAssertEqual(w[1], w[398], accuracy: 1e-6)
    }

    func testMelScaleKaldiRoundTrip() {
        for hz: Float in [20, 200, 1000, 4000, 8000] {
            let mel = SeamlessM4TFrontEnd.hzToMelKaldi(hz)
            let back = SeamlessM4TFrontEnd.melToHzKaldi(mel)
            XCTAssertEqual(back, hz, accuracy: max(1e-2, hz * 1e-4))
        }
    }

    func testMelFilterbankShapeAndStructure() {
        let fb = SeamlessM4TFrontEnd.makeKaldiMelFilterbank()
        // Row-major [257, 80].
        XCTAssertEqual(fb.count, 257 * 80)
        // The padded 257th row (bin 256) is all zero.
        for m in 0..<80 {
            XCTAssertEqual(fb[256 * 80 + m], 0.0, accuracy: 0)
        }
        // Reference column/row sums (triangularize_in_mel_space, kaldi, norm=None):
        //   col0 ≈ 0.6397, col40 ≈ 2.5760; row(bin 10) ≈ 1.0.
        var col0: Float = 0, col40: Float = 0, rowBin10: Float = 0
        for f in 0..<257 {
            col0 += fb[f * 80 + 0]
            col40 += fb[f * 80 + 40]
        }
        for m in 0..<80 { rowBin10 += fb[10 * 80 + m] }
        XCTAssertEqual(col0, 0.639707, accuracy: 5e-3)
        XCTAssertEqual(col40, 2.576026, accuracy: 5e-3)
        XCTAssertEqual(rowBin10, 1.0, accuracy: 5e-3)
        // No negative weights.
        XCTAssertFalse(fb.contains { $0 < 0 })
    }

    func testInputFeaturesShape() {
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: testSignal())
        // Reference: 1 s @ 16 kHz → (1, 49, 160).
        XCTAssertEqual(frames, 49)
        XCTAssertEqual(feats.count, 49 * 160)
    }

    func testInputFeaturesGoldenValues() {
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: testSignal())
        XCTAssertEqual(frames, 49)

        func row(_ r: Int) -> ArraySlice<Float> {
            feats[(r * 160)..<((r + 1) * 160)]
        }

        // Golden values from the HF SeamlessM4TFeatureExtractor on the chirp.
        let row0First8: [Float] = [4.86029, 6.51729, 7.31894, 5.99928,
                                   4.59500, 3.25905, 1.90246, 1.32079]
        let r0 = Array(row(0))
        for (k, expected) in row0First8.enumerated() {
            XCTAssertEqual(r0[k], expected, accuracy: 5e-2, "row0[\(k)]")
        }

        let row10First8: [Float] = [1.53234, -0.67938, -0.09944, -0.20113,
                                    -0.25268, -0.30566, -0.27090, -0.15525]
        let r10 = Array(row(10))
        for (k, expected) in row10First8.enumerated() {
            XCTAssertEqual(r10[k], expected, accuracy: 5e-2, "row10[\(k)]")
        }
        // Second stacked half of row 10 (mel bins 80..88).
        let row10At80: [Float] = [0.05244, 0.12960, -0.18585, -0.21077,
                                  -0.23860, -0.26268, -0.59610, -1.02409]
        for (k, expected) in row10At80.enumerated() {
            XCTAssertEqual(r10[80 + k], expected, accuracy: 5e-2, "row10[80+\(k)]")
        }

        // Global stats over the whole tensor.
        var mn: Float = .greatestFiniteMagnitude, mx: Float = -.greatestFiniteMagnitude
        var sum: Float = 0
        for v in feats { sum += v; mn = min(mn, v); mx = max(mx, v) }
        let mean = sum / Float(feats.count)
        XCTAssertEqual(mean, 0.0, accuracy: 5e-3)
        // Per-bin normalization → overall std ≈ 1.
        let variance = feats.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(feats.count)
        XCTAssertEqual(variance.squareRoot(), 0.994885, accuracy: 2e-2)
        XCTAssertEqual(mn, -3.670550, accuracy: 0.1)
        XCTAssertEqual(mx, 7.318937, accuracy: 0.1)
    }

    func testPerBinNormalizationZeroMean() {
        // After per-bin normalization the global mean of the stacked features
        // must be ~0 and the std ~1.
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: testSignal())
        XCTAssertGreaterThan(frames, 0)
        let mean = feats.reduce(0, +) / Float(feats.count)
        XCTAssertEqual(mean, 0.0, accuracy: 5e-3)
    }

    func testShortInputReturnsEmpty() {
        // Fewer than one analysis frame (400 samples) → no features.
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.1, count: 100))
        XCTAssertEqual(frames, 0)
        XCTAssertTrue(feats.isEmpty)
    }

    func testOddFrameCountDropsTrailingFrame() {
        // 1 + (N-400)/160 mel frames; pick N so mel frames is odd, then stacked
        // = floor(odd/2) and one mel frame is dropped.
        // N = 400 + 160*(3-1) = 720 ⇒ 3 mel frames ⇒ 1 stacked frame.
        let (_, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.05, count: 720))
        XCTAssertEqual(frames, 1)
    }
}
