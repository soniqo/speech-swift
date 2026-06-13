import XCTest
@testable import Qwen3ASR
@testable import AudioCommon
@testable import KokoroTTS

private func maxNgramRepeat(_ tokens: [String], n: Int) -> (gram: [String], count: Int) {
    guard tokens.count >= n else { return ([], 0) }
    var counts: [[String]: Int] = [:]
    for i in 0...(tokens.count - n) {
        counts[Array(tokens[i..<(i + n)]), default: 0] += 1
    }
    return counts.max(by: { $0.value < $1.value })
        .map { ($0.key, $0.value) } ?? ([], 0)
}

private func wordTokens(_ s: String) -> [String] {
    s.lowercased()
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

/// Harsh-fixture reproducers for the 0.6B greedy-decode loop reported on
/// >15 s spans. Two scenarios that the clean 20-s `test_audio.wav` does not
/// stress:
///
///   • A long stitched buffer of 5 Kokoro-synthesized sentences with NO
///     padding between them — the greedy decoder sees continuous mel content
///     with no acoustic break, the worst case for length-extrapolation in
///     the audio encoder + decoder.
///   • The same long buffer with additive white noise at SNR ≈ 10 dB —
///     degraded acoustics push the decoder toward higher-entropy regions
///     where degenerate trajectories are more likely.
///
/// Both are positive assertions: no trigram repeats >3×, output length stays
/// well below the pathological cap, and ≥3 of 5 expected sentence-fragments
/// survive in the transcript. If a future change introduces broader looping
/// (e.g. a regression in the audio encoder's positional encoding) these
/// catch it.
final class E2EQwen3ASRHarshAudioTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    private static var _model: Qwen3ASRModel?

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            Self._model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        }
    }

    /// Five Kokoro phrases stitched with zero inter-phrase silence to form
    /// ~15-20 s of dense continuous speech. Returns the audio plus the list
    /// of expected sentence-fragment keywords so the caller can verify
    /// content coverage downstream.
    private func continuousStitchedSpeech(noise: Bool) async throws -> (audio: [Float], expected: [String]) {
        let tts = try await KokoroTTSModel.fromPretrained()
        let phrases = [
            "Margaret stood by the window as the rain tapped against the glass",
            "She poured another cup of coffee and watched the gray sky",
            "The morning newspaper sat unread on the kitchen table",
            "A black cat slipped quietly between the garden hedges",
            "She wondered when her sister would finally call her back",
        ]
        var clips: [[Float]] = []
        for p in phrases {
            clips.append(try tts.synthesize(text: p, voice: "af_heart"))
        }
        var stitched = HarshAudio.stitch(clips, paddingSamples: 0)
        if noise {
            let n = HarshAudio.whiteNoise(samples: stitched.count, seed: 0x71_15_27_b9)
            stitched = HarshAudio.mixAtSNR(signal: stitched, noise: n, snrDB: 10)
        }
        let expected = [
            "margaret", "window", "rain", "glass",
            "coffee", "sky",
            "newspaper", "kitchen",
            "cat", "garden", "hedges",
            "sister", "call",
        ]
        return (stitched, expected)
    }

    /// 5-phrase continuous stitch at 24 kHz (Kokoro's native rate) — roughly
    /// 16-18 s of speech with no sentence-boundary silence. Asserts the
    /// greedy decoder neither loops nor produces a pathologically long
    /// output, and that the content survives.
    func testNoLoopOnContinuousStitchedSpeech() async throws {
        guard let model = Self._model else { throw XCTSkip("model not loaded") }
        let (audio, expected) = try await continuousStitchedSpeech(noise: false)
        print("stitched continuous: \(audio.count) samples @ 24kHz = \(Float(audio.count) / 24000) s")

        let start = Date()
        let result = model.transcribe(audio: audio, sampleRate: 24000)
        let elapsed = Date().timeIntervalSince(start)
        print("0.6B greedy continuous (\(elapsed)s): \"\(result)\"")

        XCTAssertFalse(result.isEmpty)
        XCTAssertLessThan(
            result.count, 1500,
            "result is pathologically long (\(result.count) chars) — likely degenerate loop"
        )

        let tokens = wordTokens(result)
        let (gram, count) = maxNgramRepeat(tokens, n: 3)
        XCTAssertLessThanOrEqual(
            count, 3,
            "greedy decoder degenerated on continuous stitched speech: trigram \(gram) ×\(count)"
        )

        let lower = result.lowercased()
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 5,
            "long-form lost most content: only \(found) of expected fragments \(expected)"
        )
    }

    /// Same continuous stitch + additive white Gaussian noise at SNR ≈ 10 dB.
    /// Noisy long-form is the regime the original report names; this asserts
    /// the decoder does not degenerate even when its mel features are
    /// degraded.
    func testNoLoopOnNoisyContinuousSpeech() async throws {
        guard let model = Self._model else { throw XCTSkip("model not loaded") }
        let (audio, expected) = try await continuousStitchedSpeech(noise: true)
        print("noisy stitched continuous: \(audio.count) samples @ 24kHz = \(Float(audio.count) / 24000) s")

        let result = model.transcribe(audio: audio, sampleRate: 24000)
        print("0.6B greedy noisy continuous: \"\(result)\"")

        XCTAssertFalse(result.isEmpty)
        XCTAssertLessThan(
            result.count, 1500,
            "noisy result is pathologically long — likely degenerate loop"
        )

        let tokens = wordTokens(result)
        let (gram, count) = maxNgramRepeat(tokens, n: 3)
        XCTAssertLessThanOrEqual(
            count, 4,
            "greedy decoder degenerated on noisy continuous speech: trigram \(gram) ×\(count)"
        )

        let lower = result.lowercased()
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "noisy long-form lost most content: only \(found) of \(expected)"
        )
    }

    /// `noRepeatNgramSize: 3` on the same noisy continuous speech. Confirms
    /// the options path still threads end-to-end and produces sane output
    /// when the input is harsh.
    func testNoRepeatNgramHandlesNoisyContinuousSpeech() async throws {
        guard let model = Self._model else { throw XCTSkip("model not loaded") }
        let (audio, expected) = try await continuousStitchedSpeech(noise: true)

        let result = model.transcribe(
            audio: audio,
            sampleRate: 24000,
            options: Qwen3DecodingOptions(maxTokens: 448, noRepeatNgramSize: 3)
        )
        print("0.6B noRepeatNgram=3 noisy: \"\(result)\"")

        XCTAssertFalse(result.isEmpty)
        XCTAssertLessThan(result.count, 1500)
        let lower = result.lowercased()
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "options path destroyed content under noise: \(found) of \(expected)"
        )
    }
}
