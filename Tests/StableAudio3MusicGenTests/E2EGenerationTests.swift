import XCTest
import MLX
@testable import StableAudio3MusicGen

/// End-to-end generation test for `aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit`.
/// Downloads the bundle on first run (~5 GB) so it is gated behind the `E2E`
/// prefix that CI's `make test` `--skip E2E` filter excludes.
final class E2EGenerationTests: XCTestCase {

    func testGenerateMediumInt8ShortClip() async throws {
        let model = try await StableAudio3MusicGen.fromPretrained(
            variant: .mediumInt8,
            tLatHint: StableAudio3MusicGen.computeTLat(seconds: 2.0))

        let (left, right) = model.generate(
            prompt: "lofi house loop",
            params: StableAudio3GenerationParams(
                seconds: 2.0, steps: 4, cfgScale: 1.0,
                sigmaMax: 1.0, seed: 42))

        XCTAssertEqual(left.count, 2 * SA3Audio.sampleRate,
                       "Expected exactly 2 s × 44.1 kHz frames per channel")
        XCTAssertEqual(left.count, right.count, "Stereo channel lengths must match")

        // Audio must be finite (no NaN/Inf) and non-silent.
        XCTAssertTrue(left.allSatisfy { $0.isFinite }, "left channel has non-finite samples")
        XCTAssertTrue(right.allSatisfy { $0.isFinite }, "right channel has non-finite samples")
        let leftPeak = left.map(abs).max() ?? 0
        let rightPeak = right.map(abs).max() ?? 0
        XCTAssertGreaterThan(leftPeak, 0.001, "left channel is silent (peak=\(leftPeak))")
        XCTAssertGreaterThan(rightPeak, 0.001, "right channel is silent (peak=\(rightPeak))")
    }
}
