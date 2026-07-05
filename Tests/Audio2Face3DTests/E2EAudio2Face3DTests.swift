import XCTest
import Foundation
@testable import Audio2Face3D

/// End-to-end tests that download the published MLX bundles from Hugging Face
/// and run real inference. Skipped by CI via `--skip E2E` filter.
final class E2EAudio2Face3DTests: XCTestCase {
    func testFromPretrainedMarkGeneratesFrames() async throws {
        try await assertPublishedBundleGeneratesFrames(
            modelId: Audio2Face3DConfiguration.markModelId,
            expectedCoefficientCount: 301)
    }

    func testFromPretrainedClaireGeneratesFrames() async throws {
        try await assertPublishedBundleGeneratesFrames(
            modelId: Audio2Face3DConfiguration.claireModelId,
            expectedCoefficientCount: 169)
    }

    func testFromPretrainedJamesGeneratesFrames() async throws {
        try await assertPublishedBundleGeneratesFrames(
            modelId: Audio2Face3DConfiguration.jamesModelId,
            expectedCoefficientCount: 169)
    }

    private func assertPublishedBundleGeneratesFrames(
        modelId: String,
        expectedCoefficientCount: Int
    ) async throws {
        let model = try await Audio2Face3DModel.fromPretrained(modelId: modelId)

        let layout = model.configuration.coefficientLayout
        XCTAssertEqual(
            layout.skinCount + layout.tongueCount + layout.jawCount + layout.eyeCount,
            expectedCoefficientCount,
            "\(modelId) layout mismatch")

        // 1 s of speech-band chirp with an envelope, 16 kHz mono.
        let sampleRate = 16_000
        let samples = (0..<sampleRate).map { index -> Float in
            let t = Float(index) / Float(sampleRate)
            return 0.5 * sin(2 * .pi * (120 + 80 * t) * t) * sin(.pi * t)
        }

        let frames = try model.frames(for: samples, sampleRate: sampleRate)

        XCTAssertGreaterThan(frames.count, 1)
        for frame in frames {
            XCTAssertEqual(frame.coefficients.count, expectedCoefficientCount)
            XCTAssertTrue(frame.coefficients.allSatisfy(\.isFinite))
        }

        // The chirp must actually drive the face: coefficients vary across frames.
        let first = frames[0].coefficients
        let anyMotion = frames.dropFirst().contains { frame in
            zip(frame.coefficients, first).contains { abs($0 - $1) > 1e-3 }
        }
        XCTAssertTrue(anyMotion, "\(modelId) produced static frames")
    }
}
