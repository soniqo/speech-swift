import XCTest
@testable import NemotronStreamingASR
@testable import AudioCommon
@testable import KokoroTTS

// MARK: - Bundle resolver mirrored from sibling test files
//
// Lang-tag leak only surfaces on the multilingual bundle (13087-vocab) whose
// vocab.json contains literal `<en-US>`, `<de-DE>`, etc. tokens. The
// English-only bundle (1024-vocab) cannot reproduce the bug at all.

private func localMultilingualBundle() -> URL? {
    if let env = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"], !env.isEmpty {
        let url = URL(fileURLWithPath: env)
        if FileManager.default.fileExists(atPath: url.path) { return url }
    }
    let fallback = URL(fileURLWithPath: "/tmp/Nemotron-3.5-CoreML-320ms")
    if FileManager.default.fileExists(atPath: fallback.path) { return fallback }
    return nil
}

private let langTagRegex: NSRegularExpression = {
    try! NSRegularExpression(pattern: "<[a-zA-Z][a-zA-Z-]*>")
}()

private func leakedLangTags(in text: String) -> [String] {
    let range = NSRange(text.startIndex..<text.endIndex, in: text)
    return langTagRegex.matches(in: text, range: range).compactMap {
        Range($0.range, in: text).map { String(text[$0]) }
    }
}

private func wordTokens(_ s: String) -> [String] {
    s.lowercased()
        .components(separatedBy: CharacterSet.alphanumerics.inverted)
        .filter { !$0.isEmpty }
}

/// Tries to reproduce the reported `<en-US>` leak under audio conditions the
/// clean studio fixture doesn't trigger:
///
///   • Overlapped voices (two Kokoro speakers, one mixed -3 dB) — forces the
///     RNN-T joint to disambiguate competing speech, the regime where the
///     greedy argmax can wander into language-ID-token territory.
///   • Babble background at SNR ≈ 8 dB — degraded acoustic conditions.
///
/// All assertions require the **multilingual** bundle (13087 vocab) — only
/// that vocab contains `<lang-XX>` tokens. The English-only bundle (1024
/// vocab) is structurally incapable of leaking these tokens.
final class E2ENemotronHarshAudioTests: XCTestCase {

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
            if let local = localMultilingualBundle() {
                Self._model = try await NemotronStreamingASRModel.fromLocal(bundleDir: local)
            } else {
                Self._model = try await NemotronStreamingASRModel.fromPretrained()
            }
        }
    }

    /// Two Kokoro voices speaking different sentences, overlaid with the
    /// second at -3 dB. The transcript must not contain any `<lang-XX>`
    /// markers. (Quality of the transcript itself is intentionally not
    /// asserted strictly — overlapping speech is genuinely hard.)
    func testNoLangTagOnOverlappedSpeech() async throws {
        let m = try model
        let tts = try await KokoroTTSModel.fromPretrained()

        let primary = "The conference room had a long oak table beneath the windows"
        let secondary = "Please verify your credentials and confirm the booking time"

        let voiceA = try tts.synthesize(text: primary, voice: "af_heart")
        let voiceB = try tts.synthesize(text: secondary, voice: "am_adam")
        let mixed = HarshAudio.overlay(voiceA, voiceB, gainBdB: -3)
        print("overlap mix: \(mixed.count) samples @ 24kHz = \(Float(mixed.count) / 24000) s")

        let text = try m.transcribeAudio(mixed, sampleRate: 24000, language: "en-US")
        print("Nemotron overlapped transcript: \"\(text)\"")

        XCTAssertFalse(text.isEmpty, "overlapped audio should not produce empty output")
        let tags = leakedLangTags(in: text)
        XCTAssertTrue(
            tags.isEmpty,
            "Nemotron leaked language tokens \(tags) under overlapped speech: \"\(text)\""
        )
    }

    /// `test_audio.wav` + multi-talker babble derived from the same audio at
    /// SNR ≈ 8 dB. Asserts (a) no `<lang-tag>` leak, (b) at least one of the
    /// four content words still recoverable — purely-noise output would
    /// indicate the model collapsed.
    func testNoLangTagUnderBabbleNoise() throws {
        let m = try model
        let audioURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav")!
        let clean = try AudioFileLoader.load(url: audioURL, targetSampleRate: 16000)
        let babble = HarshAudio.babbleFromSpeech(clean, voiceCount: 6)
        let noisy = HarshAudio.mixAtSNR(signal: clean, noise: babble, snrDB: 8)
        print("babble mix: \(noisy.count) samples @ 16kHz = \(Float(noisy.count) / 16000) s")

        let text = try m.transcribeAudio(noisy, sampleRate: 16000, language: "en-US")
        print("Nemotron babble-SNR8 transcript: \"\(text)\"")

        XCTAssertFalse(text.isEmpty)
        let tags = leakedLangTags(in: text)
        XCTAssertTrue(
            tags.isEmpty,
            "Nemotron leaked language tokens \(tags) under babble noise: \"\(text)\""
        )

        // Sanity: at SNR=8 dB the model should still extract *something*. A
        // pure-noise output (zero keyword hits) would indicate a much deeper
        // regression than a lang-tag leak.
        let lower = text.lowercased()
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        let found = expected.filter { lower.contains($0) }
        XCTAssertFalse(
            found.isEmpty,
            "noisy transcription has zero content overlap with truth — model collapsed: \"\(text)\""
        )
    }

    /// Streaming-vs-batch parity on a long, clean, single-voice utterance.
    /// Tests what the Swift chunker actually controls — that chunk boundaries
    /// don't drop content on audio the model is well-equipped to transcribe.
    ///
    /// Why this shape, not overlapped voices: the topology investigation
    /// (workflow `nemotron-streaming-chunker-fix`) confirmed the encoder's
    /// CoreML graph already incorporates `rightContext` via the streaming
    /// caches (`keep_all_outputs=False` trim at export time, see
    /// `speech-models/.../convert.py:172`). The Swift chunker feeding
    /// `chunk_size` frames with no audio overlap is exactly what the model
    /// was trained for. Overlapped two-voice fixtures conflate chunker
    /// correctness (this test) with model difficulty on multi-speaker audio
    /// (a checkpoint property, not a Swift bug).
    ///
    /// Expected recall ≥ 0.95 on continuous single-voice speech. A drop
    /// below this floor indicates a real chunker regression — the boundary
    /// math being silently changed, the cache I/O wiring breaking, or a
    /// well-meaning "fix" introducing audio overlap (which would corrupt
    /// the RNN-T decoder's LSTM state, see RNNTGreedyDecoder.swift:38-89).
    func testStreamingMatchesBatchOnCleanLongUtterance() async throws {
        let m = try model
        let tts = try await KokoroTTSModel.fromPretrained()

        // ~18 s of continuous English speech, single speaker, no silence
        // gaps, ~56 chunk crossings at the 320 ms multilingual cadence.
        let longText = """
        The quarterly report indicated steady growth across all three divisions. \
        Manufacturing output rose by twelve percent while logistics costs declined. \
        Our research team confirmed the prototype passed every regression benchmark \
        and is scheduled for limited deployment next month across the western region.
        """
        let audio = try tts.synthesize(text: longText, voice: "af_heart")

        let batchText = try m.transcribeAudio(audio, sampleRate: 24000, language: "en-US")
        var partials: [NemotronStreamingASRModel.PartialTranscript] = []
        for await p in m.transcribeStream(audio: audio, sampleRate: 24000, language: "en-US") {
            partials.append(p)
        }
        guard let streamFinal = partials.last else {
            return XCTFail("streaming pass produced no partials")
        }

        let batchWords = Set(wordTokens(batchText))
        let streamWords = Set(wordTokens(streamFinal.text))
        let recall = batchWords.isEmpty
            ? 1.0
            : Double(batchWords.intersection(streamWords).count) / Double(batchWords.count)
        print("clean-long recall (stream covers batch): \(recall)")
        print("  batch:  \"\(batchText)\"")
        print("  stream: \"\(streamFinal.text)\"")

        XCTAssertGreaterThanOrEqual(
            recall, 0.95,
            "streaming chunker dropped content vs batch on clean continuous speech: recall=\(recall). This indicates a real chunker regression — boundary math, cache I/O, or accidental audio overlap. See `nemotron-streaming-chunker-fix` workflow notes."
        )
    }
}
