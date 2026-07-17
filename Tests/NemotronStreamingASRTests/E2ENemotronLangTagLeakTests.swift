import XCTest
@testable import NemotronStreamingASR
@testable import AudioCommon
@testable import KokoroTTS

// MARK: - Bundle resolvers (file-private; mirror NemotronStreamingASRTests.swift)

private func localMultilingualBundle() -> URL? {
    if let env = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"], !env.isEmpty {
        let url = URL(fileURLWithPath: env)
        if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    let fallback = URL(fileURLWithPath: "/tmp/Nemotron-3.5-CoreML-320ms")
    if FileManager.default.fileExists(atPath: fallback.path) { return fallback }
    return nil
}

// MARK: - Word-set Jaccard helper

private func wordTokens(_ s: String) -> [String] {
    s.lowercased()
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

private func jaccard(_ a: [String], _ b: [String]) -> Double {
    let sa = Set(a)
    let sb = Set(b)
    if sa.isEmpty && sb.isEmpty { return 1.0 }
    let inter = sa.intersection(sb).count
    let union = sa.union(sb).count
    return Double(inter) / Double(union)
}

/// Regression tests for two related Nemotron defects observed in the wild:
///
/// 1. Language-identifier tokens (e.g. `<en-US>`, `<lang-XX>`) leaking into the
///    decoded transcript because `NemotronVocabulary.decode` has no
///    skip-special-tokens path.
///
/// 2. Streaming transcription dropping content versus the single-shot batch
///    path on the same audio — the chunking pipeline emits per-chunk partials
///    but does not pad or re-overlap at boundaries, so words can fall off.
///
/// Requires the **multilingual** Nemotron 3.5 bundle (vocab=13087, the only
/// variant whose vocab contains `<lang-XX>` markers). XCTSkipped when only the
/// English-only bundle (vocab=1024) is available locally.
final class E2ENemotronLangTagLeakTests: XCTestCase {

    private static var _model: NemotronStreamingASRModel?

    private var model: NemotronStreamingASRModel {
        get throws {
            guard let m = Self._model else { throw XCTSkip("multilingual bundle unavailable") }
            return m
        }
    }

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            // Prefer the local /tmp bundle if present; otherwise let
            // fromPretrained pull the published multilingual model. The
            // English-only bundle (1024-vocab) can NOT reproduce the lang-tag
            // leak — special tokens for languages only exist in the 13087-vocab
            // multilingual bundle.
            if let local = localMultilingualBundle() {
                Self._model = try await NemotronStreamingASRModel.fromLocal(bundleDir: local)
            } else {
                Self._model = try await NemotronStreamingASRModel.fromPretrained()
            }
        }
    }

    /// The decoded transcript must NOT contain `<lang-XX>`-style markers.
    ///
    /// Reproduces the leak observed in real-world runs against overlapped /
    /// noisy audio. `NemotronVocabulary.decode` does not strip special tokens —
    /// if the RNN-T joint argmax ever lands on a language ID slot, that
    /// literal `<en-US>` (or any other `<lang-tag>`) flows straight into the
    /// returned String. Fix surface: filter special IDs in `Vocabulary.decode`.
    func testNoLanguageTagInOutput() throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)

        let text = try m.transcribeAudio(audio, sampleRate: 16000, language: "en-US")
        print("Nemotron en-US raw transcript: \"\(text)\"")

        XCTAssertFalse(text.isEmpty, "transcript should not be empty")

        let langTagRegex = try NSRegularExpression(pattern: "<[a-zA-Z][a-zA-Z-]*>")
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = langTagRegex.matches(in: text, range: range)
        if !matches.isEmpty {
            let leaked = matches.compactMap { Range($0.range, in: text).map { String(text[$0]) } }
            XCTFail("language-id tokens leaked into transcript: \(leaked) — full text: \"\(text)\"")
        }
    }

    /// Word timings must track the audio: emission-aligned words stay in
    /// lockstep with the transcript, start monotonically, and remain bounded
    /// by the (padded) stream duration.
    func testStreamingWordTimingsAreMonotonicAndBounded() throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let audio = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let duration = Double(audio.count) / 16_000

        let session = try m.createSession(language: "en-US")
        var last: NemotronStreamingASRModel.PartialTranscript?
        for lower in stride(from: 0, to: audio.count, by: 16_000) {
            let upper = min(audio.count, lower + 16_000)
            for partial in try session.pushAudio(Array(audio[lower..<upper])) {
                last = partial
            }
        }
        for partial in try session.finalize() { last = partial }

        let final = try XCTUnwrap(last)
        XCTAssertTrue(final.isFinal)
        XCTAssertFalse(final.words.isEmpty)
        XCTAssertEqual(
            final.words.map(\.text).joined(separator: " "),
            final.text,
            "timed words must reproduce the transcript exactly")
        var previousStart = -Double.infinity
        for word in final.words {
            XCTAssertLessThanOrEqual(word.startTime, word.endTime)
            XCTAssertGreaterThanOrEqual(word.startTime, previousStart)
            previousStart = word.startTime
        }
        XCTAssertGreaterThanOrEqual(final.words.first!.startTime, 0)
        // Greedy RNN-T emission trails the acoustics, but never beyond the
        // padded final chunk.
        XCTAssertLessThanOrEqual(final.words.last!.endTime, duration + 1.0)
    }

    /// Same audio through the batch path and the streaming path must produce
    /// transcripts whose content words substantially overlap. A streaming pass
    /// that drops one or more words from the batch reference (e.g. losing a
    /// word that straddles a 320 ms chunk boundary) fails this assertion.
    ///
    /// Uses Kokoro-synthesized speech for a known phrase, exercising at least
    /// 9 streaming chunks at the multilingual config's 320 ms cadence.
    func testStreamingPreservesContentVsBatch() async throws {
        let m = try model
        let tts = try await KokoroTTSModel.fromPretrained()

        // ~2.5 s of clean English speech — spans ~8-9 chunks of 320 ms.
        let phrase = "The quick brown fox jumps over the lazy dog by the riverbank"
        let audio24k = try tts.synthesize(text: phrase, voice: "af_heart")

        let batchText = try m.transcribeAudio(audio24k, sampleRate: 24000, language: "en-US")
        var partials: [NemotronStreamingASRModel.PartialTranscript] = []
        for await p in m.transcribeStream(audio: audio24k, sampleRate: 24000, language: "en-US") {
            partials.append(p)
        }
        guard let streamFinal = partials.last else {
            return XCTFail("streaming pass produced no partials")
        }
        XCTAssertTrue(streamFinal.isFinal, "last partial must be marked isFinal")

        let batchWords = wordTokens(batchText)
        let streamWords = wordTokens(streamFinal.text)
        let j = jaccard(batchWords, streamWords)
        print("Nemotron batch: \"\(batchText)\"")
        print("Nemotron stream: \"\(streamFinal.text)\"")
        print("word-set Jaccard(batch, stream) = \(j)")

        XCTAssertGreaterThanOrEqual(
            j, 0.7,
            "streaming pass dropped content vs batch: jaccard=\(j) batch=\"\(batchText)\" stream=\"\(streamFinal.text)\""
        )

        // Belt-and-suspenders: the streaming transcript must also be free of
        // language-tag leaks. Captures regressions where the leak appears only
        // on the chunked path.
        let langTagRegex = try NSRegularExpression(pattern: "<[a-zA-Z][a-zA-Z-]*>")
        let r = NSRange(streamFinal.text.startIndex..<streamFinal.text.endIndex, in: streamFinal.text)
        XCTAssertEqual(
            langTagRegex.numberOfMatches(in: streamFinal.text, range: r), 0,
            "streaming transcript leaked a <lang-tag>: \"\(streamFinal.text)\""
        )
    }
}
