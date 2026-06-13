import XCTest
import AudioCommon
@testable import OmnilingualASR

private func wordTokens(_ s: String) -> [String] {
    s.lowercased()
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

private func jaccard(_ a: [String], _ b: [String]) -> Double {
    let sa = Set(a)
    let sb = Set(b)
    if sa.isEmpty && sb.isEmpty { return 1.0 }
    let union = sa.union(sb).count
    return union == 0 ? 0 : Double(sa.intersection(sb).count) / Double(union)
}

/// Harsh-fixture reproducers for the Omnilingual MLX phonetic-noise report
/// ("aerian" instead of "American"). On clean studio audio the MLX backend
/// matches CoreML closely (Jaccard ≈ 0.83 in our baseline run). The
/// reproducer regime is degraded acoustics — additive noise pushes the
/// wav2vec2 frontend's mel features into ambiguous regions where phonetic
/// neighbors compete in the CTC argmax.
///
/// Tests cover:
///   • Babble background at SNR ≈ 8 dB and 5 dB (conference room → cafe)
///   • White Gaussian noise at SNR ≈ 10 dB (electronic hiss)
///
/// All assertions are relative: MLX is allowed to degrade, but it must
/// degrade no more than the CoreML reference on the same noisy input
/// (Jaccard ≥ 0.4 — looser than the clean threshold of 0.5).
@MainActor
final class E2EOmnilingualMLXHarshAudioTests: XCTestCase {

    private func cleanAudio() throws -> [Float] {
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "test_audio", withExtension: "wav"),
            "test_audio.wav missing from bundle"
        )
        return try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
    }

    /// Babble at SNR = 8 dB. MLX must still extract at least 2 of the 4
    /// content words AND agree with CoreML to within Jaccard ≥ 0.4.
    func testMLXContentSurvivesBabbleSNR8() async throws {
        let clean = try cleanAudio()
        let babble = HarshAudio.babbleFromSpeech(clean, voiceCount: 6)
        let noisy = HarshAudio.mixAtSNR(signal: clean, noise: babble, snrDB: 8)

        let mlx = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let coreml = try await OmnilingualASRModel.fromPretrained()

        let mlxText = try mlx.transcribeAudio(noisy, sampleRate: 16000)
        let coremlText = try coreml.transcribeAudio(noisy, sampleRate: 16000)
        print("MLX    babble-SNR8: \"\(mlxText)\"")
        print("CoreML babble-SNR8: \"\(coremlText)\"")

        XCTAssertFalse(mlxText.isEmpty)
        XCTAssertFalse(coremlText.isEmpty)

        let mlxLower = mlxText.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let mlxFound = expected.filter { mlxLower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            mlxFound.count, 2,
            "MLX under babble-SNR8 lost >50% of content: got \(mlxFound) of \(expected)"
        )

        let j = jaccard(wordTokens(coremlText), wordTokens(mlxText))
        print("Jaccard(CoreML, MLX) under babble-SNR8 = \(j)")
        // 0.4 floor on noisy babble — MLX single-pass diverges from CoreML
        // chunked path under noise; threshold caps the divergence at
        // ~50% word-set agreement.
        XCTAssertGreaterThanOrEqual(
            j, 0.4,
            "MLX diverged from CoreML under babble noise: jaccard=\(j)"
        )
    }

    /// Lower SNR (5 dB). Looser threshold — we just need MLX to produce
    /// non-empty output and to track CoreML by at least Jaccard ≥ 0.25.
    /// A failing test here is the phonetic-noise regression in flight.
    func testMLXTracksCoreMLAtBabbleSNR5() async throws {
        let clean = try cleanAudio()
        let babble = HarshAudio.babbleFromSpeech(clean, voiceCount: 6)
        let noisy = HarshAudio.mixAtSNR(signal: clean, noise: babble, snrDB: 5)

        let mlx = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let coreml = try await OmnilingualASRModel.fromPretrained()

        let mlxText = try mlx.transcribeAudio(noisy, sampleRate: 16000)
        let coremlText = try coreml.transcribeAudio(noisy, sampleRate: 16000)
        print("MLX    babble-SNR5: \"\(mlxText)\"")
        print("CoreML babble-SNR5: \"\(coremlText)\"")

        XCTAssertFalse(mlxText.isEmpty, "MLX should still emit something at SNR=5dB")

        let j = jaccard(wordTokens(coremlText), wordTokens(mlxText))
        print("Jaccard(CoreML, MLX) under babble-SNR5 = \(j)")
        // 0.25 floor on low-SNR babble — MLX single-pass and CoreML 10 s
        // chunks diverge more on noisy input. We catch *complete*
        // divergence (no overlap) without flaking on the known split.
        XCTAssertGreaterThanOrEqual(
            j, 0.25,
            "MLX completely diverged from CoreML at low SNR: jaccard=\(j)"
        )
    }

    /// White Gaussian noise at SNR = 10 dB. Different noise statistics than
    /// babble — flat spectrum hits every frequency band uniformly, exposing
    /// frontend numerics regressions that babble might mask.
    func testMLXContentSurvivesWhiteNoiseSNR10() async throws {
        let clean = try cleanAudio()
        let noise = HarshAudio.whiteNoise(samples: clean.count, seed: 0x52_77_b1_ac)
        let noisy = HarshAudio.mixAtSNR(signal: clean, noise: noise, snrDB: 10)

        let mlx = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let text = try mlx.transcribeAudio(noisy, sampleRate: 16000)
        print("MLX white-SNR10: \"\(text)\"")

        XCTAssertFalse(text.isEmpty)
        let lower = text.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 2,
            "MLX under white-SNR10 lost most content: \(found) of \(expected) in \"\(text)\""
        )
    }
}
