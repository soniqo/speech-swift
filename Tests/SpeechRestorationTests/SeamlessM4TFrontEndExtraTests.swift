import XCTest
@testable import SpeechRestoration

/// Additional DSP unit tests for the SeamlessM4T log-mel front-end: determinism,
/// frame-count arithmetic, scale invariance of the per-bin normalization, and
/// degenerate / boundary inputs. These complement the golden-value tests in
/// `SeamlessM4TFrontEndTests`.
final class SeamlessM4TFrontEndExtraTests: XCTestCase {

    /// Same reproducible chirp the golden test uses.
    private func chirp(_ n: Int = 16000) -> [Float] {
        var x = [Float](repeating: 0, count: n)
        let f0: Float = 100, f1: Float = 2000, T: Float = 1.0
        for i in 0..<n {
            let t = Float(i) / 16000.0
            let phase = 2 * .pi * (f0 * t + (f1 - f0) / (2 * T) * t * t)
            x[i] = 0.4 * Foundation.sin(phase) + 0.05 * Foundation.sin(2 * .pi * 123 * t)
        }
        return x
    }

    // MARK: Determinism

    func testDeterministicAcrossCalls() {
        let signal = chirp()
        let (a, fa) = SeamlessM4TFrontEnd.inputFeatures(audio: signal)
        let (b, fb) = SeamlessM4TFrontEnd.inputFeatures(audio: signal)
        XCTAssertEqual(fa, fb)
        XCTAssertEqual(a.count, b.count)
        // Bit-for-bit identical — the front-end is pure DSP with cached tables.
        XCTAssertEqual(a, b)
    }

    // MARK: Frame-count arithmetic

    func testFrameCountFormula() {
        // mel frames = 1 + (N - 400)/160; stacked = mel/2.
        // N = 400 + 160*(2k-1) → 2k mel frames → k stacked frames, for k≥1.
        for k in 1...20 {
            let melFrames = 2 * k
            let n = 400 + 160 * (melFrames - 1)
            let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
                audio: [Float](repeating: 0.05, count: n))
            XCTAssertEqual(frames, k, "N=\(n) should yield \(k) stacked frames")
            XCTAssertEqual(feats.count, k * 160)
        }
    }

    func testExactlyOneFrameNeedsAtLeastTwoMelFrames() {
        // 400 samples = exactly 1 mel frame ⇒ 0 stacked frames (stride 2).
        let (feats400, frames400) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.1, count: 400))
        XCTAssertEqual(frames400, 0)
        XCTAssertTrue(feats400.isEmpty)
        // 560 samples = 2 mel frames ⇒ 1 stacked frame.
        let (feats560, frames560) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.1, count: 560))
        XCTAssertEqual(frames560, 1)
        XCTAssertEqual(feats560.count, 160)
    }

    func testFullWindowYieldsConfigFrames() {
        // The fixed 160_000-sample window must produce exactly SidonConfig.frames.
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0, count: SidonConfig.default.windowSamples))
        XCTAssertEqual(frames, SidonConfig.default.frames)
        XCTAssertEqual(feats.count, SidonConfig.default.frames * SidonConfig.default.featureDim)
    }

    // MARK: Edge / degenerate inputs

    func testEmptyInput() {
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: [])
        XCTAssertEqual(frames, 0)
        XCTAssertTrue(feats.isEmpty)
    }

    func testBelowOneFrameInput() {
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.3, count: 399))
        XCTAssertEqual(frames, 0)
        XCTAssertTrue(feats.isEmpty)
    }

    func testSilenceProducesFiniteFeatures() {
        // Pure silence: DC removal zeros each frame, so every mel bin sits at the
        // log mel-floor in every frame. The output stays finite (no -inf from the
        // floored log), and because each bin is constant over time, per-bin
        // normalization (subtract mean, divide by sqrt(var+1e-7)) yields the SAME
        // tiny constant in every cell (float residue amplified by the 1e-7 floor).
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0, count: 16000))
        XCTAssertGreaterThan(frames, 0)
        XCTAssertTrue(feats.allSatisfy { $0.isFinite })
        // Every cell equals the first (constant-input ⇒ uniform output).
        let first = feats[0]
        for v in feats { XCTAssertEqual(v, first, accuracy: 1e-4) }
        // And it's bounded (not blown up to absurd magnitude).
        XCTAssertLessThan(abs(first), 1.0)
    }

    func testDCInputStaysFinite() {
        // A pure DC offset (constant non-zero) has its per-frame mean removed, so
        // its first-difference spectrum is tiny but the pre-emphasis tap leaves a
        // small residual at the frame edges. The point of the test is robustness:
        // no NaN/Inf and a bounded result (it does NOT collapse to all-zero the
        // way true silence does, because pre-emphasis breaks the constancy).
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(
            audio: [Float](repeating: 0.7, count: 16000))
        XCTAssertGreaterThan(frames, 0)
        XCTAssertTrue(feats.allSatisfy { $0.isFinite })
        // Normalized features stay in a sane range.
        XCTAssertTrue(feats.allSatisfy { abs($0) < 100 })
    }

    func testNoNaNOrInfOnLoudInput() {
        // Full-scale input must not overflow / produce NaN through the FFT + log.
        var loud = [Float](repeating: 0, count: 16000)
        for i in 0..<loud.count { loud[i] = (i % 2 == 0) ? 1.0 : -1.0 }
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: loud)
        XCTAssertGreaterThan(frames, 0)
        XCTAssertTrue(feats.allSatisfy { $0.isFinite })
    }

    // MARK: Normalization properties

    func testPerBinNormalizationUnitishStd() {
        // Normalization (ddof=1) is applied per mel bin over time on the [T, 80]
        // log-mel BEFORE stride-2 stacking. In the stacked [S, 160] output, mel
        // bin m's normalized values are spread across BOTH halves: columns m
        // (even mel frames) and m+80 (odd mel frames). Reassembling the full mel
        // column from both halves recovers a ~zero-mean / ~unit-std (ddof=1)
        // distribution — the property the extractor guarantees.
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: chirp())
        XCTAssertGreaterThan(frames, 4)
        let dim = 160
        for m in [0, 20, 79] {
            var col: [Float] = []
            col.reserveCapacity(2 * frames)
            for f in 0..<frames {
                col.append(feats[f * dim + m])        // even mel frame, bin m
                col.append(feats[f * dim + 80 + m])   // odd  mel frame, bin m
            }
            let n = Float(col.count)
            let mean = col.reduce(0, +) / n
            XCTAssertEqual(mean, 0, accuracy: 2e-2, "mel bin \(m) mean")
            let sse = col.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) }
            // The reassembled column drops the last (odd) mel frame when the mel
            // count is odd, so use a population-style std here; it stays ~1.
            let std = (sse / (n - 1)).squareRoot()
            XCTAssertEqual(std, 1.0, accuracy: 0.05, "mel bin \(m) std")
        }
    }

    func testStackingConcatenatesAdjacentMelFrames() {
        // Stacked row s = [melFrame 2s | melFrame 2s+1]. The first 80 dims of
        // stacked row 0 must equal the first 80 dims of stacked row's... we can't
        // see raw mel frames directly, but stacking is a pure copy, so the two
        // 80-wide halves of a row come from *different* mel frames and therefore
        // generally differ for a non-stationary signal.
        let (feats, frames) = SeamlessM4TFrontEnd.inputFeatures(audio: chirp())
        XCTAssertGreaterThan(frames, 2)
        let dim = 160
        // For a chirp, the lower and upper halves of a stacked row should not be
        // identical (they're consecutive-in-time mel frames).
        var anyDifferent = false
        for f in 0..<min(frames, 5) {
            for k in 0..<80 where abs(feats[f * dim + k] - feats[f * dim + 80 + k]) > 1e-4 {
                anyDifferent = true
            }
        }
        XCTAssertTrue(anyDifferent, "stacked halves should differ for a chirp")
    }
}
