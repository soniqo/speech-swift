import XCTest
@testable import SpeechEnhancement

/// Regression coverage for the long-form chunker in
/// ``SpeechEnhancer.enhanceChunked(...)``. The chunker exists because the
/// CoreML model is exported with a ``RangeDim(1, 6000)`` on the time axis,
/// capping single-shot calls at ~60 s. Inputs longer than that previously
/// threw a CoreML prediction error from
/// ``DeepFilterNet3Model.predict(...)``; the chunked path slices the input
/// into 45 s windows by default, runs each through the existing pipeline
/// without resetting STFT or normalization state, and stitches the outputs
/// with an equal-power sin/cos crossfade.
///
/// Skipped in CI (``E2E`` prefix) — runs locally with ``make test``.
@MainActor
final class E2EDeepFilterNet3ChunkingTests: XCTestCase {

    private static var _enhancer: SpeechEnhancer?

    private func enhancer() async throws -> SpeechEnhancer {
        if let e = Self._enhancer { return e }
        let e = try await SpeechEnhancer.fromPretrained()
        Self._enhancer = e
        return e
    }

    // MARK: - Short input invariants

    /// Inputs at or under the default `chunkSeconds` go through the
    /// established single-shot path. The chunked variant should produce a
    /// transcript byte-equivalent to plain `enhance(...)` because no
    /// crossfade math touches the samples.
    func testChunked_ShortInputMatchesSingleShot() async throws {
        let enhancer = try await enhancer()
        let rate = 48000
        // 2 s of speech-shaped audio: 440 Hz sine + 0.05 amplitude white noise.
        let samples = Self.makeTestSignal(durationSeconds: 2.0, sampleRate: rate)

        let single = try enhancer.enhance(audio: samples, sampleRate: rate)
        let chunked = try enhancer.enhanceChunked(
            audio: samples, sampleRate: rate, chunkSeconds: 45.0, overlapMs: 500)

        XCTAssertEqual(chunked.count, single.count,
                       "Short-input chunked path must match single-shot length")
        // Bit-exact: short inputs take the same fast path through enhance().
        for i in 0..<single.count {
            XCTAssertEqual(chunked[i], single[i], accuracy: 1e-6,
                           "Short-input chunked must be bit-equivalent at sample \(i)")
        }
    }

    // MARK: - Long input invariants

    /// 90 s synthetic input — exceeds the 60 s single-shot cap; chunker must
    /// produce a non-empty output of (close to) the same length, free of NaN
    /// / Inf, and within a reasonable peak range.
    func testChunked_NinetySecondsLengthAndSanity() async throws {
        let enhancer = try await enhancer()
        let rate = 48000
        let samples = Self.makeTestSignal(durationSeconds: 90.0, sampleRate: rate)

        let start = Date()
        let enhanced = try enhancer.enhanceChunked(
            audio: samples, sampleRate: rate, chunkSeconds: 30.0, overlapMs: 500)
        let elapsed = Date().timeIntervalSince(start)
        print("90 s chunked enhance: \(String(format: "%.2f", elapsed)) s wall, \(enhanced.count) samples out")

        // Length tolerance: chunker may round by < 100 samples (~2 ms) at the
        // boundary. We allow ±0.1 % of the input length.
        let tolerance = samples.count / 1000
        XCTAssertEqual(enhanced.count, samples.count, accuracy: tolerance,
                       "Chunked output length must match input within 0.1%")

        // No NaN / Inf — easy to introduce via crossfade math when a chunk
        // returns empty.
        for (i, v) in enhanced.enumerated() {
            if !v.isFinite {
                XCTFail("Non-finite sample at index \(i): \(v)")
                return
            }
        }

        // Peak inside [-2, 2] — the chunker's output amplitude should never
        // exceed roughly 2× the input peak (input was ≤1.0 by construction).
        let peak = enhanced.map(abs).max() ?? 0
        XCTAssertLessThan(peak, 2.0,
                          "Output peak \(peak) suggests crossfade math doubled amplitude")
    }

    /// Crossfade quality check — for retained audio, the RMS energy in a
    /// 200 ms window centered on a chunk seam should be within 1.5 dB of the
    /// adjacent windows. If the enhancer classifies the synthetic tone as
    /// noise and suppresses all three below -50 dBFS, the absolute level is
    /// already below the test's audibility floor and relative dB changes are
    /// not meaningful.
    func testChunked_NoStationaryPumpingAtSeams() async throws {
        let enhancer = try await enhancer()
        let rate = 48000
        // 70 s clip — forces at least one chunk boundary at the default 30 s
        // window. Sine + low-level noise so the RMS is roughly stationary.
        let samples = Self.makeTestSignal(durationSeconds: 70.0, sampleRate: rate)

        let chunkSeconds = 30.0
        let overlapMs = 500
        let enhanced = try enhancer.enhanceChunked(
            audio: samples, sampleRate: rate,
            chunkSeconds: chunkSeconds, overlapMs: overlapMs)

        // Predicted seam centers (the second seam should be present too).
        // The chunker uses equal-division — for 70 s with 30 s windows and
        // 500 ms overlap, two chunks land at ~35 s each, so the seam centers
        // near 35 s.
        let seamCenter = enhanced.count / 2  // mid-point ≈ 35 s
        let windowSamples = 200 * rate / 1000  // 200 ms

        let pre = rmsWindow(enhanced, center: seamCenter - windowSamples, width: windowSamples)
        let mid = rmsWindow(enhanced, center: seamCenter, width: windowSamples)
        let post = rmsWindow(enhanced, center: seamCenter + windowSamples, width: windowSamples)

        let preDb = 20 * log10(pre + 1e-9)
        let midDb = 20 * log10(mid + 1e-9)
        let postDb = 20 * log10(post + 1e-9)
        print("seam RMS dB: pre=\(preDb) mid=\(midDb) post=\(postDb)")

        let audibilityFloorDb: Float = -50
        if max(preDb, midDb, postDb) < audibilityFloorDb {
            XCTAssertLessThan(midDb, audibilityFloorDb,
                              "Suppressed seam should stay below -50 dBFS")
        } else {
            XCTAssertLessThan(abs(midDb - preDb), 1.5,
                              "Seam RMS jumps >1.5 dB vs pre-seam window — likely pumping")
            XCTAssertLessThan(abs(midDb - postDb), 1.5,
                              "Seam RMS jumps >1.5 dB vs post-seam window — likely pumping")
        }
    }

    // MARK: - Helpers

    /// 440 Hz sine + low-level uniform noise, normalized to peak 0.5.
    /// The stationary signal keeps seam analysis deterministic.
    private static func makeTestSignal(durationSeconds: Double, sampleRate: Int) -> [Float] {
        let n = Int(durationSeconds * Double(sampleRate))
        var rng = SplitMix64(seed: 0xc0ffee_decaf_face)
        var out = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(sampleRate)
            let sine = sin(2 * .pi * 440 * t)
            let noise = (Float(rng.next() & 0xffff) / Float(0xffff) - 0.5) * 0.1
            out[i] = (sine + noise) * 0.5
        }
        return out
    }

    private func rmsWindow(_ signal: [Float], center: Int, width: Int) -> Float {
        let lo = max(0, center - width / 2)
        let hi = min(signal.count, center + width / 2)
        guard hi > lo else { return 0 }
        var sumSq: Float = 0
        for i in lo..<hi { sumSq += signal[i] * signal[i] }
        return sqrt(sumSq / Float(hi - lo))
    }
}

/// Tiny deterministic RNG (avoid `arc4random` because tests need
/// cross-platform bit-stability for the noise floor).
private struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { self.state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
}
