import XCTest
@testable import AudioCommon

/// Unit tests for `AudioFileLoader.resample` / `.resampleStereo`.
///
/// These pin the mastering-grade SRC behaviour: correct output length, no
/// truncated tail, anti-aliasing on downsample, HF preservation on
/// near-rate conversion, and phase-aligned stereo. No GPU / model downloads.
final class ResampleTests: XCTestCase {

    // MARK: - Helpers

    private func sine(freq: Double, seconds: Double, sampleRate: Int, amp: Float = 1.0) -> [Float] {
        let n = Int(seconds * Double(sampleRate))
        return (0..<n).map { amp * Float(sin(2.0 * Double.pi * freq * Double($0) / Double(sampleRate))) }
    }

    private func rms(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return 0 }
        let sum = x.reduce(0.0) { $0 + Double($1) * Double($1) }
        return (sum / Double(x.count)).squareRoot()
    }

    /// Amplitude of a single tone at `freq` via direct DFT correlation.
    /// Returns ≈ the peak amplitude of a pure sine at that frequency.
    private func toneAmplitude(_ x: [Float], freq: Double, sampleRate: Int) -> Double {
        guard !x.isEmpty else { return 0 }
        var re = 0.0, im = 0.0
        let w = 2.0 * Double.pi * freq / Double(sampleRate)
        for (n, s) in x.enumerated() {
            re += Double(s) * cos(w * Double(n))
            im -= Double(s) * sin(w * Double(n))
        }
        return 2.0 * (re * re + im * im).squareRoot() / Double(x.count)
    }

    // MARK: - Length

    func testLengthMatchesRatio() {
        let input = sine(freq: 1000, seconds: 1.0, sampleRate: 48000)
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100)
        let expected = Int((Double(input.count) * 44100.0 / 48000.0).rounded())
        XCTAssertEqual(out.count, expected,
            "48k→44.1k length must be exactly round(N·ratio) (got \(out.count), expected \(expected))")
    }

    func testUpsampleLength() {
        let input = sine(freq: 1000, seconds: 0.5, sampleRate: 16000)
        let out = AudioFileLoader.resample(input, from: 16000, to: 48000)
        let expected = Int((Double(input.count) * 48000.0 / 16000.0).rounded())
        XCTAssertEqual(out.count, expected)
    }

    func testExactLengthOddRatio() {
        // Non-integer ratio downsample (44.1k→16k) must still land on exactly
        // round(N·ratio) — the normalization step enforces this.
        let input = sine(freq: 1000, seconds: 0.7, sampleRate: 44100)
        let out = AudioFileLoader.resample(input, from: 44100, to: 16000)
        let expected = Int((Double(input.count) * 16000.0 / 44100.0).rounded())
        XCTAssertEqual(out.count, expected)
    }

    func testIdentityReturnsInput() {
        let input = sine(freq: 1000, seconds: 0.1, sampleRate: 44100)
        let out = AudioFileLoader.resample(input, from: 44100, to: 44100)
        XCTAssertEqual(out, input, "Same in/out rate must return the input unchanged")
    }

    func testEmptyInput() {
        XCTAssertEqual(AudioFileLoader.resample([], from: 48000, to: 44100), [])
    }

    // MARK: - Tail preservation (drain)

    func testTailNotTruncated() {
        // 1 kHz tone — after resampling the final samples must still carry
        // signal energy (the old single-convert path dropped the sinc tail).
        let input = sine(freq: 1000, seconds: 1.0, sampleRate: 48000)
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100)
        let lastChunk = Array(out.suffix(64))
        XCTAssertGreaterThan(rms(lastChunk), 0.05,
            "Tail samples should retain signal, not be zeroed/truncated")
    }

    func testRMSPreserved() {
        let input = sine(freq: 1000, seconds: 1.0, sampleRate: 48000)  // amp 1.0 → RMS ≈ 0.707
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100)
        XCTAssertEqual(rms(out), rms(input), accuracy: 0.02,
            "Overall energy should be preserved across resampling")
    }

    func testEndImpulseLandsInTail() {
        // A unit impulse as the *last* input sample is the strongest drain
        // test: its sinc response sits in the final output samples. A broken
        // drain (the old single-convert path) truncates the converter's
        // latency tail and would zero these out. A continuous sine can't catch
        // this — it has energy at the end regardless of correct draining.
        let n = 48000
        var input = [Float](repeating: 0, count: n)
        input[n - 1] = 1.0
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100)

        let expected = Int((Double(n) * 44100.0 / 48000.0).rounded())
        XCTAssertEqual(out.count, expected, "exact length even with an end impulse")

        // The response must actually be present at the tail, not truncated.
        XCTAssertGreaterThan(rms(Array(out.suffix(32))), 0.1,
            "End-impulse response must survive the drain (tail not zeroed)")
        // Peak should sit at/near the very last sample.
        let peakIdx = out.indices.max(by: { abs(out[$0]) < abs(out[$1]) }) ?? 0
        XCTAssertGreaterThan(peakIdx, out.count - 16,
            "Impulse peak should land in the final samples, not be truncated/shifted")
    }

    func testLateImpulseRetainsFullEnergy() {
        // An impulse a few ms before the end (response fully inside the output)
        // must retain the same energy as the identical impulse mid-buffer. If
        // the drain dropped the tail latency, the late impulse would lose
        // energy relative to the centered one.
        let n = 48000
        var mid = [Float](repeating: 0, count: n)
        mid[n / 2] = 1.0
        var late = [Float](repeating: 0, count: n)
        late[n - 100] = 1.0

        let outMid = AudioFileLoader.resample(mid, from: 48000, to: 44100)
        let outLate = AudioFileLoader.resample(late, from: 48000, to: 44100)

        func energy(_ x: [Float]) -> Double { x.reduce(0.0) { $0 + Double($1) * Double($1) } }
        let eMid = energy(outMid)
        let eLate = energy(outLate)
        XCTAssertGreaterThan(eMid, 0.5, "impulse should resample to non-trivial energy")
        XCTAssertEqual(eLate / eMid, 1.0, accuracy: 0.05,
            "A late impulse must retain the same energy as a centered one (tail not lost)")
        // And that energy must live in the tail region, not be shifted earlier.
        XCTAssertGreaterThan(energy(Array(outLate.suffix(128))) / eLate, 0.5,
            "Late-impulse response should sit in the final output samples")
    }

    // MARK: - Anti-aliasing

    func testNoAliasOnDownsample() {
        // 22 kHz tone is valid at 48k (Nyq 24k) but must be filtered out when
        // downsampling to 16k (Nyq 8k). A naive resampler would fold it to
        // |22k - 16k| = 6 kHz. Assert no 6 kHz alias and near-silent output.
        let input = sine(freq: 22000, seconds: 0.5, sampleRate: 48000)
        let out = AudioFileLoader.resample(input, from: 48000, to: 16000)
        XCTAssertLessThan(toneAmplitude(out, freq: 6000, sampleRate: 16000), 0.05,
            "No aliased 6 kHz image should appear")
        XCTAssertLessThan(rms(out), 0.1,
            "A 22 kHz tone should be filtered out when downsampling to 16k")
    }

    // MARK: - HF preservation

    func testHighFreqPreserved48to44() {
        // 18 kHz is below both Nyquist limits (24k and 22.05k) so it must
        // survive even on the default (speech) quality, which only rolls off
        // closer to Nyquist.
        let input = sine(freq: 18000, seconds: 0.5, sampleRate: 48000)
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100)
        let amp = toneAmplitude(out, freq: 18000, sampleRate: 44100)
        XCTAssertGreaterThan(amp, 0.8,
            "18 kHz content should be preserved on 48k→44.1k (got amp \(amp))")
    }

    // MARK: - Quality (mastering for music/upsampling)

    func testMasteringRetainsMoreHighFreqThanStandard() {
        // The whole point of `.mastering` for music: fuller high-frequency
        // retention near Nyquist. At 18 kHz on 48k→44.1k, mastering keeps
        // essentially all of it while the default rolls off a little.
        let input = sine(freq: 18000, seconds: 0.5, sampleRate: 48000)
        let std = AudioFileLoader.resample(input, from: 48000, to: 44100, quality: .standard)
        let mas = AudioFileLoader.resample(input, from: 48000, to: 44100, quality: .mastering)
        let stdAmp = toneAmplitude(std, freq: 18000, sampleRate: 44100)
        let masAmp = toneAmplitude(mas, freq: 18000, sampleRate: 44100)
        XCTAssertGreaterThan(masAmp, 0.95, "mastering should retain ~all of 18 kHz (got \(masAmp))")
        XCTAssertGreaterThan(masAmp, stdAmp,
            "mastering should retain more HF than standard (mastering \(masAmp) vs standard \(stdAmp))")
    }

    func testMasteringDrainsAndExactLength() {
        // The mastering branch must drain and hit exact length just like the
        // default — verify with an end impulse on a non-integer ratio.
        let n = 48000
        var input = [Float](repeating: 0, count: n)
        input[n - 1] = 1.0
        let out = AudioFileLoader.resample(input, from: 48000, to: 44100, quality: .mastering)
        let expected = Int((Double(n) * 44100.0 / 48000.0).rounded())
        XCTAssertEqual(out.count, expected, "mastering must produce exact length")
        XCTAssertGreaterThan(rms(Array(out.suffix(32))), 0.1,
            "mastering must retain the end-impulse tail")
    }

    func testMasteringStereoStaysAlignedAndExact() {
        let mono = sine(freq: 5000, seconds: 0.4, sampleRate: 48000)
        let out = AudioFileLoader.resampleStereo([mono, mono], from: 48000, to: 44100, quality: .mastering)
        let expected = Int((Double(mono.count) * 44100.0 / 48000.0).rounded())
        XCTAssertEqual(out[0].count, expected)
        XCTAssertEqual(out[0], out[1], "identical channels must resample identically at mastering quality")
    }

    // MARK: - Stereo

    func testStereoChannelsStayAligned() {
        // Identical L and R must remain bit-identical after a single-pass
        // stereo resample (independent converters could drift).
        let mono = sine(freq: 3000, seconds: 0.5, sampleRate: 48000)
        let out = AudioFileLoader.resampleStereo([mono, mono], from: 48000, to: 44100)
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0], out[1], "Identical channels must resample identically")
    }

    func testStereoPreservesAmplitudeRatio() {
        let left = sine(freq: 2000, seconds: 0.5, sampleRate: 48000, amp: 1.0)
        let right = sine(freq: 2000, seconds: 0.5, sampleRate: 48000, amp: 0.5)
        let out = AudioFileLoader.resampleStereo([left, right], from: 48000, to: 44100)
        let lAmp = toneAmplitude(out[0], freq: 2000, sampleRate: 44100)
        let rAmp = toneAmplitude(out[1], freq: 2000, sampleRate: 44100)
        XCTAssertEqual(rAmp / lAmp, 0.5, accuracy: 0.02,
            "Per-channel amplitude ratio must be preserved (no channel bleed)")
    }

    func testStereoMatchesMonoPath() {
        // Resampling a channel through the stereo path should match the mono
        // path within tight tolerance.
        let sig = sine(freq: 4000, seconds: 0.3, sampleRate: 48000)
        let mono = AudioFileLoader.resample(sig, from: 48000, to: 44100)
        let stereo = AudioFileLoader.resampleStereo([sig, sig], from: 48000, to: 44100)
        XCTAssertEqual(stereo[0].count, mono.count)
        let n = min(stereo[0].count, mono.count)
        var maxDiff = 0.0
        for i in 0..<n { maxDiff = max(maxDiff, abs(Double(stereo[0][i] - mono[i]))) }
        XCTAssertLessThan(maxDiff, 1e-3, "Stereo path should match mono path")
    }

    func testStereoIdentityReturnsInput() {
        let left = sine(freq: 1000, seconds: 0.1, sampleRate: 44100)
        let right = sine(freq: 1500, seconds: 0.1, sampleRate: 44100)
        let out = AudioFileLoader.resampleStereo([left, right], from: 44100, to: 44100)
        XCTAssertEqual(out[0], left)
        XCTAssertEqual(out[1], right)
    }
}
