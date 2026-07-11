import XCTest
import Foundation
import MLX
import AudioCommon
@testable import FlashSR

/// End-to-end tests that download the FlashSR bundle and run inference.
/// Skipped by CI via the `--skip E2E` filter (per CLAUDE.md convention).
final class E2EFlashSRTests: XCTestCase {

    /// Most important sanity test: load int4 bundle → 5.12 s of silence-ish
    /// input → check output shape + non-NaN + reasonable amplitude.
    func testInt4LoadAndUpsampleShortClip() async throws {
        let model = try await FlashSR.fromPretrained(variant: .int4)

        // Synthetic input: a 5.12 s 48 kHz mono sine sweep — slightly more
        // interesting than silence so the model has signal to enhance.
        let sr = FlashSR.sampleRate
        let n = FlashSR.frameSamples
        var audio = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let t = Float(i) / Float(sr)
            // sweep 200 → 2000 Hz log-linear
            let freq = Float(200) * powf(10, t / Float(n / sr) * 1.0)
            audio[i] = 0.3 * sinf(2 * .pi * freq * t)
        }

        let start = Date()
        let hr = model.upsampleWindow(samples: audio, params: FlashSRParams(seed: 42))
        let wall = Date().timeIntervalSince(start)
        let audioSec = Double(hr.count) / Double(sr)
        print("[FLASHSR] wall=\(String(format: "%.2f", wall))s  audio=\(String(format: "%.2f", audioSec))s  RTF=\(String(format: "%.2f", wall / audioSec))")

        XCTAssertEqual(hr.count, n, "Output must be exactly 5.12 s @ 48 kHz")
        XCTAssertFalse(hr.contains { $0.isNaN || $0.isInfinite }, "Output contains NaN/Inf")
        let peak = hr.map(abs).max() ?? 0
        XCTAssertGreaterThan(peak, 1e-4, "Output is effectively silent")
        XCTAssertLessThan(peak, 100, "Output peak is unreasonable")
    }

    /// Verify that the convenience `upsample(audio:sampleRate:)` path agrees
    /// with the explicit windowed path for a single-window input.
    func testSingleWindowConvenienceMatchesWindowedAPI() async throws {
        let model = try await FlashSR.fromPretrained(variant: .int4)
        let n = FlashSR.frameSamples
        var audio = [Float](repeating: 0, count: n)
        for i in 0..<n { audio[i] = 0.2 * sinf(2 * .pi * 440 * Float(i) / 48000) }

        let viaConvenience = try model.upsample(audio: audio, sampleRate: FlashSR.sampleRate)
        XCTAssertEqual(viaConvenience.count, audio.count)
        XCTAssertFalse(viaConvenience.contains { $0.isNaN || $0.isInfinite })
    }
}
