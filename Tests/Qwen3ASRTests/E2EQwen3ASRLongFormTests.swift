import XCTest
@testable import Qwen3ASR
@testable import AudioCommon

// MARK: - Trigram-loop detector

/// Count occurrences of every consecutive `n`-token window in `tokens`. Used to
/// detect degenerate looping where the greedy decoder collapses onto a short
/// pattern and repeats it for the rest of the generation horizon.
private func maxNgramRepeat(_ tokens: [String], n: Int) -> (gram: [String], count: Int) {
    guard tokens.count >= n else { return ([], 0) }
    var counts: [[String]: Int] = [:]
    for i in 0...(tokens.count - n) {
        let gram = Array(tokens[i..<(i + n)])
        counts[gram, default: 0] += 1
    }
    return counts.max(by: { $0.value < $1.value })
        .map { ($0.key, $0.value) } ?? ([], 0)
}

private func wordTokens(_ s: String) -> [String] {
    s.lowercased()
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

/// Regression tests for the 0.6B greedy-decode loop on long-form audio
/// (reported on speech spans >15 s). The greedy fast path at
/// `Qwen3ASR.generateGreedyAsyncEval` runs for `maxTokens` (default 448) with
/// no built-in degeneration guard — `repetitionPenalty=1.0` and
/// `noRepeatNgramSize=0` are off by default. On long inputs the decoder can
/// latch onto a short n-gram and emit it until the cap.
///
/// NOTE on reproducibility: the original report ("0.6B loops/degenerates on
/// >15 s spans") manifests on the reporter's specific audio (overlapped
/// speakers / noise / continuous music+speech). Tiling our clean 20 s
/// `test_audio.wav` twice (40 s) does NOT reliably trigger looping — clean
/// speech is well-behaved on the greedy fast path. These tests are positive
/// regression guards: they fail if the decoder degenerates *even on clean
/// long-form audio*, which would catch a broader regression than the
/// originally reported defect.
///
/// These tests:
///   1. Feed ~40 s of speech-shaped audio (tile of the 20 s fixture) and
///      assert no trigram appears more than 3 times. Also asserts result
///      length is well below the pathological cap.
///   2. Re-run the same input with `noRepeatNgramSize: 3` and assert sane
///      output + known content words. Note: `noRepeatNgramSize` operates on
///      BPE token IDs, not on whitespace-split word tokens, so a strict
///      word-level "no trigram repeats" assertion is not appropriate here.
final class E2EQwen3ASRLongFormTests: XCTestCase {

    static let modelId = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"

    private static var _model: Qwen3ASRModel?

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            Self._model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        }
    }

    /// Build a ~40 s buffer of real speech by tiling the 20 s test fixture
    /// twice. Speech-shaped mel content reliably triggers the looping
    /// regression on the greedy fast path, while pure silence or a sine wave
    /// would not.
    private func longFormAudio() throws -> (samples: [Float], sampleRate: Int) {
        let wavURL = try XCTUnwrap(
            Bundle.module.url(forResource: "test_audio", withExtension: "wav"),
            "test_audio.wav missing from bundle"
        )
        let (samples, sr) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 24000
        let oneCopy: [Float] = (sr == targetSampleRate)
            ? samples
            : AudioFileLoader.resample(samples, from: sr, to: targetSampleRate)
        let tiled = oneCopy + oneCopy  // ~40 s of speech-shaped audio
        return (tiled, targetSampleRate)
    }

    /// Greedy decode on ~40 s of speech must not degenerate into a repeating
    /// n-gram. Threshold of 3 catches loops without flagging legitimate
    /// repetition ("the the" is common; "the part the part the part the part"
    /// is the bug).
    func testNoTrigramLoopOnLongAudio() throws {
        guard let model = Self._model else { throw XCTSkip("model not loaded") }

        let (audio, sr) = try longFormAudio()
        let start = Date()
        let result = model.transcribe(audio: audio, sampleRate: sr)
        let elapsed = Date().timeIntervalSince(start)
        print("0.6B greedy long-form (\(audio.count) samples / \(sr) Hz) in \(elapsed)s: \"\(result)\"")

        XCTAssertFalse(result.isEmpty, "result should not be empty")

        // Pathological-length guard: degenerate output saturates near the
        // maxTokens horizon. UTF-8 byte-level decoder emits ~1-4 chars/token;
        // a clean transcription of 40 s of English speech is well under 800
        // chars. Cap at 1500 to allow some slack while still catching runs
        // that exceed the legitimate transcript size by 2-3x.
        XCTAssertLessThan(
            result.count, 1500,
            "result is pathologically long (\(result.count) chars) — likely degenerate loop"
        )

        // N-gram loop detector.
        let tokens = wordTokens(result)
        let (gram, count) = maxNgramRepeat(tokens, n: 3)
        print("most-repeated trigram: \(gram) ×\(count)")
        XCTAssertLessThanOrEqual(
            count, 3,
            "greedy decoder degenerated on long-form audio: trigram \(gram) repeated \(count)× in \"\(result)\""
        )

        // Belt-and-suspenders: the expected content words must still appear.
        // The fixture says "Can you guarantee that the replacement part will
        // be shipped tomorrow?" — tiled twice the same words repeat in
        // legitimate fashion.
        let lower = result.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "expected at least 3 of \(expected) in long-form transcript, got \(found): \"\(result)\""
        )
    }

    /// Same long-form input + `noRepeatNgramSize: 3` must produce a sane
    /// transcript with the expected keywords. The options-based pickNextToken
    /// path is the documented fix surface — this confirms it flows end-to-end
    /// without crashing or destroying content on long inputs.
    ///
    /// Subtle: `noRepeatNgramSize` operates on **BPE token IDs**, not on
    /// whitespace-split word tokens. A side-effect on tiled audio is that the
    /// model produces near-duplicate but BPE-distinct second renderings (e.g.
    /// "tomorrow" → "to morrow", "will" → "wiill"). The same word can still
    /// surface twice through different sub-word paths, so a strict word-level
    /// trigram check is inappropriate here. We assert content-word coverage
    /// and a soft trigram bound matching what natural tiled speech allows.
    func testNoRepeatNgramOptionPreservesContent() throws {
        guard let model = Self._model else { throw XCTSkip("model not loaded") }

        let (audio, sr) = try longFormAudio()
        let withGuard = model.transcribe(
            audio: audio,
            sampleRate: sr,
            options: Qwen3DecodingOptions(maxTokens: 448, noRepeatNgramSize: 3)
        )
        print("0.6B noRepeatNgram=3 long-form: \"\(withGuard)\"")

        XCTAssertFalse(withGuard.isEmpty)

        // Same trigram threshold as the greedy guard — clean output even with
        // the n-gram blocker should not loop more than ×3 on tiled audio.
        let tokens = wordTokens(withGuard)
        let (gram, count) = maxNgramRepeat(tokens, n: 3)
        XCTAssertLessThanOrEqual(
            count, 3,
            "noRepeatNgramSize=3 path still degenerated: trigram \(gram) ×\(count) in \"\(withGuard)\""
        )

        // The decoder must still recover the content even under the n-gram
        // blocker. Loss of all 4 keywords would mean the option destroyed
        // signal rather than bounding degeneration.
        let lower = withGuard.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertGreaterThanOrEqual(
            found.count, 3,
            "options path lost content: only \(found) of \(expected) in \"\(withGuard)\""
        )
    }
}
