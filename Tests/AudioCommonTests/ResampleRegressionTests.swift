import XCTest
@testable import AudioCommon

/// Regression coverage for `AudioFileLoader.resample` across every (input →
/// output) sample-rate pair the codebase actually uses. The resampler feeds
/// every pipeline (ASR 16k, TTS 24k, denoise/upsampling 48k, separation
/// 44.1k), so a defect here breaks many consumers at once. These pin the
/// invariants every consumer relies on, for both quality settings:
///   - exact output length == round(N · ratio)
///   - finite output, no clipping/blow-up
///   - in-band energy preserved
///   - out-of-band content rejected on downsample (anti-aliasing)
final class ResampleRegressionTests: XCTestCase {

    /// Sample rates that appear as resample sources/targets in the codebase
    /// (file inputs vary; targets are the model rates 16k/24k/44.1k/48k).
    private let inputRates  = [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
    private let targetRates = [16000, 24000, 44100, 48000]
    private let qualities: [ResampleQuality] = [.standard, .mastering]

    private func tone(freq: Double, sr: Int, seconds: Double = 0.5, amp: Float = 0.5) -> [Float] {
        let n = Int(seconds * Double(sr))
        return (0..<n).map { amp * Float(sin(2.0 * Double.pi * freq * Double($0) / Double(sr))) }
    }
    private func rms(_ x: [Float]) -> Double {
        guard !x.isEmpty else { return 0 }
        return (x.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(x.count)).squareRoot()
    }

    /// Exact length, finite samples, no clipping, and preserved in-band energy
    /// for a 1 kHz tone (well below every Nyquist here) across all rate pairs.
    func testInBandFidelityAcrossAllConsumerRates() {
        let expectedRMS = 0.5 / 2.0.squareRoot()  // amp-0.5 sine ≈ 0.3536
        for inR in inputRates {
            for outR in targetRates where inR != outR {
                let input = tone(freq: 1000, sr: inR)
                let expectedLen = Int((Double(input.count) * Double(outR) / Double(inR)).rounded())
                for q in qualities {
                    let out = AudioFileLoader.resample(input, from: inR, to: outR, quality: q)
                    let ctx = "\(inR)->\(outR) \(q)"

                    XCTAssertEqual(out.count, expectedLen, "exact length (\(ctx))")
                    XCTAssertTrue(out.allSatisfy { $0.isFinite }, "finite output (\(ctx))")
                    let maxAbs = out.map { abs($0) }.max() ?? 0
                    XCTAssertLessThan(maxAbs, 0.55, "no clipping/blow-up for amp-0.5 input (\(ctx))")
                    XCTAssertEqual(rms(out), expectedRMS, accuracy: 0.02, "in-band energy preserved (\(ctx))")
                }
            }
        }
    }

    /// Every downsample must reject content above the output Nyquist instead of
    /// folding it back as an alias. Place a tone midway into the kill band and
    /// assert it is strongly attenuated.
    func testAntiAliasOnAllDownsamples() {
        for inR in inputRates {
            for outR in targetRates where outR < inR {
                let outNyq = Double(outR) / 2.0
                let inNyq = Double(inR) / 2.0
                let killFreq = outNyq + (inNyq - outNyq) * 0.5
                for q in qualities {
                    let killed = AudioFileLoader.resample(tone(freq: killFreq, sr: inR), from: inR, to: outR, quality: q)
                    XCTAssertLessThan(rms(killed), 0.05,
                        "tone at \(Int(killFreq))Hz must be filtered out when \(inR)->\(outR) \(q) (got rms \(rms(killed)))")
                }
            }
        }
    }
}
