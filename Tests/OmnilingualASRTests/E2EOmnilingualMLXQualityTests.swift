import XCTest
import AudioCommon
@testable import OmnilingualASR

// MARK: - Word-set helpers (file-private mirror of helpers in other modules)

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

/// Quality regression tests for the Omnilingual MLX backend.
///
/// Two known defects this catches:
///   1. The MLX backend emits phonetically-near-correct but wrong words
///      (e.g. "aerian" instead of "American", "henivix pefore" for "Zenivex
///      E4"). The existing `testTranscribeRealAudio` only requires ONE of
///      four expected keywords — phonetic noise that lands on one keyword
///      still passes. This test demands ≥3 of 4.
///   2. The MLX backend has a 10 s receptive field but no internal chunking,
///      so audio longer than ~10 s degrades silently. The 20 s `test_audio.wav`
///      is the natural reproducer: a hard 10 s prefix slice should produce
///      similar content recall, but the full 20 s clip currently produces
///      worse output than the CoreML reference.
///
/// Also adds a cross-backend comparison: MLX vs CoreML on the same audio,
/// asserting word-set overlap. CoreML 300M is the known-good reference.
@MainActor
final class E2EOmnilingualMLXQualityTests: XCTestCase {

    private func loadAudio() throws -> [Float] {
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "test_audio", withExtension: "wav"),
            "test_audio.wav missing from bundle"
        )
        return try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
    }

    /// The full 20 s fixture is "Can you guarantee that the replacement part
    /// will be shipped tomorrow?" — at least 3 of 4 content words must
    /// survive. The single-pass MLX path produces "shiped" (one BPE-typo
    /// short of "shipped") because the CTC argmax is a single per-frame
    /// decision over the whole utterance's log-mel; a chunker that fixed
    /// this one word regressed LibriSpeech test-clean WER by 1.39 pp on
    /// asr-bench, so the production behaviour is single-pass. A drop below
    /// 3-of-4 indicates a real regression.
    func testMLXMatchesGroundTruthOnTwentySeconds() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let audio = try loadAudio()
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        print("Omnilingual MLX 300M-4bit (20 s clip): \"\(text)\"")

        XCTAssertFalse(text.isEmpty, "transcript should not be empty")
        let lower = text.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "MLX should recover ≥3 of 4 content words on the clean 20 s clip; got \(found): \"\(text)\""
        )
    }

    /// First-10s slice baseline. Confirms that within the MLX backend's
    /// receptive field the model can still recover content words — if this
    /// passes AND the 20 s test above fails, the bug is localized to the
    /// long-audio path (no internal chunking past 10 s). If both fail, the
    /// bug is in the feature extractor / quantization.
    func testMLXMatchesGroundTruthOnFirstTenSeconds() async throws {
        let model = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let audio = try loadAudio()
        let prefix = Array(audio.prefix(10 * 16000))
        let text = try model.transcribeAudio(prefix, sampleRate: 16000)
        print("Omnilingual MLX 300M-4bit (first 10 s): \"\(text)\"")

        XCTAssertFalse(text.isEmpty)
        let lower = text.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 2,
            "MLX failed even on a 10 s slice (within the receptive field): \(found) of \(expected) in \"\(text)\""
        )
    }

    /// MLX vs CoreML cross-backend agreement on the same fixture. CoreML 300M
    /// is the known-good reference. A pass at strict-content-words above can
    /// still mask MLX degradation if both backends regress the same way;
    /// asserting word-set overlap with CoreML pins MLX quality to the
    /// CoreML baseline.
    ///
    /// Threshold: Jaccard ≥ 0.5 on lowercase word tokens. Tight enough to
    /// catch phonetic-noise outputs, loose enough to survive natural
    /// punctuation / casing differences.
    func testMLXAgreesWithCoreMLOnTwentySeconds() async throws {
        let coreml = try await OmnilingualASRModel.fromPretrained()
        let mlx = try await OmnilingualASRMLXModel.fromPretrained(variant: .m300, bits: 4)
        let audio = try loadAudio()

        let coremlText = try coreml.transcribeAudio(audio, sampleRate: 16000)
        let mlxText = try mlx.transcribeAudio(audio, sampleRate: 16000)
        print("CoreML 300M: \"\(coremlText)\"")
        print("MLX    300M: \"\(mlxText)\"")

        XCTAssertFalse(coremlText.isEmpty, "CoreML reference is empty — fixture broken?")
        XCTAssertFalse(mlxText.isEmpty)

        let coremlWords = wordTokens(coremlText)
        let mlxWords = wordTokens(mlxText)
        let j = jaccard(coremlWords, mlxWords)
        print("Jaccard(CoreML, MLX) = \(j)")
        // 0.5 floor: CoreML chunks at 10 s and may differ from single-pass
        // MLX on multi-window inputs. Measured baseline on this 20 s clip
        // is ~0.83 (CoreML gets "shipped"; MLX single-pass gets "shiped" —
        // one BPE-quantization-drift word). 0.5 catches genuine cross-
        // backend divergence; tighter would flake on this known mismatch.
        XCTAssertGreaterThanOrEqual(
            j, 0.5,
            "MLX backend diverges from CoreML reference: jaccard=\(j) CoreML=\(coremlWords) MLX=\(mlxWords)"
        )
    }
}
