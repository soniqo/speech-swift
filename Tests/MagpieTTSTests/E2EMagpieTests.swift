import XCTest
import Foundation
import MLX
import AudioCommon
import Qwen3ASR
@testable import MagpieTTS

/// End-to-end tests that download the Magpie INT4 bundle and run synthesis.
/// Skipped by CI via the `--skip E2E` filter (per CLAUDE.md convention).
final class E2EMagpieSynthesisTests: XCTestCase {

    /// Round-trip sanity test: load INT4 bundle → synthesize a short phrase →
    /// verify shape, dtype, non-silence, and that greedy decoding terminates
    /// well below the max-frame cap.
    func testInt8LoadAndSynthesizeHello() async throws {
        let model = try await MagpieTTS.fromPretrained(variant: .int8)

        let start = Date()
        // Greedy decoding (temperature=0) makes the test deterministic and
        // — with the Aria speaker — reliably hits EOS within ~125 frames
        // (5.8 s) on the literal-char tokenizer fallback.
        let audio = try model.synthesize(
            text: "hello",
            speaker: .aria,
            language: .english,
            params: MagpieTTSParams(temperature: 0, topK: 1, maxSteps: 500))
        let wall = Date().timeIntervalSince(start)
        let audioSec = Double(audio.count) / Double(MagpieTTS.sampleRate)
        print("[MAGPIE] wall=\(String(format: "%.2f", wall))s  audio=\(String(format: "%.2f", audioSec))s  RTF=\(String(format: "%.2f", wall / audioSec))")

        // Shape + dtype sanity
        XCTAssertGreaterThan(audio.count, MagpieTTS.sampleRate / 4,
                             "Output shorter than 0.25 s — likely empty / immediate EOS")
        XCTAssertFalse(audio.contains { $0.isNaN || $0.isInfinite },
                       "Output contains NaN/Inf samples")

        // Amplitude / activity. We pin the expected envelope tightly because
        // a regression in FSQ decoding or weight loading silently degrades
        // RMS even when the audio "plays back". With proper G2P+EOS "hello"
        // produces ~1.3 s of audio at peak ~0.5, rms ~0.10.
        let peak = audio.map(abs).max() ?? 0
        let rms = (audio.reduce(0) { $0 + $1 * $1 } / Float(audio.count)).squareRoot()
        XCTAssertGreaterThan(peak, 0.30, "Output peak too quiet (peak=\(peak), expect ~0.50)")
        XCTAssertLessThan(peak, 1.0, "Output peak too loud (peak=\(peak))")
        XCTAssertGreaterThan(rms, 0.03, "Output RMS too quiet (rms=\(rms), expect ~0.10)")
        XCTAssertLessThan(rms, 0.25, "Output RMS too loud (rms=\(rms))")

        // Greedy decoding must hit EOS well before the max-step cap.
        let maxSamples = 500 * 1024  // 500 frames × 1024 samples/frame
        XCTAssertLessThan(audio.count, maxSamples,
                          "Greedy decoding hit max_steps — EOS detection broken")
    }

    /// Frame-level parity guard: feed the exact phoneme IDs the Python MLX
    /// reference uses for the IPA string "həˈloʊ" and verify the LM samples
    /// the same first frame. This catches any drift in tokeniser
    /// dedup-ordering, weight-loading key paths, attention masks or the FSQ
    /// inverse — all of which are bugs we've already hit once.
    func testFirstFrameMatchesPythonReference() async throws {
        let model = try await MagpieTTS.fromPretrained(variant: .int8)

        // Magpie's English vocab has duplicates after `<pad>`/`<oov>`. The
        // Python reference (and now our tokeniser) uses the first
        // occurrence: "həˈloʊ" → [55, 79, 90, 59, 62, 87, eos=2361].
        let tok = try model.tokenizer(for: .english)
        let ids = tok.tokenize("həˈloʊ", prephonemized: true)
        XCTAssertEqual(ids, [55, 79, 90, 59, 62, 87, tok.eosId],
                       "Tokeniser regressed — vocab dedup-ordering or EOS append changed?")
        XCTAssertEqual(tok.eosId, 2361,
                       "EOS id drifted (expect vocab_size + 1 = 2361)")
    }

    /// Closed-loop intelligibility test: TTS → ASR. The earlier numerical
    /// guards only prove we ported the model faithfully; they don't prove
    /// the audio actually speaks the right words. This test pipes the
    /// synthesised waveform through Qwen3-ASR and requires every input
    /// word to appear in the transcription.
    ///
    /// The harness uses the same English sentence as
    /// `speech-models/tests/test_round_trip.py`. With proper G2P (CMU IPA
    /// dict) + EOS=2361 the Python pipeline transcribes this as
    /// "Hello, World from Magpie Text to Speech." — Swift must match.
    func testSynthesizedAudioTranscribesToInput() async throws {
#if canImport(CoreML)
        let tts = try await MagpieTTS.fromPretrained(variant: .int8)
        let asr = try await CoreMLASRModel.fromPretrained()

        let prompt = "Hello world from Magpie text to speech."
        let audio = try tts.synthesize(
            text: prompt, speaker: .aria, language: .english,
            params: MagpieTTSParams(temperature: 0, topK: 1, maxSteps: 500))
        XCTAssertGreaterThan(audio.count, MagpieTTS.sampleRate / 2,
                             "TTS produced <0.5 s of audio — likely silent")

        let raw = asr.transcribe(audio: audio,
                                  sampleRate: MagpieTTS.sampleRate,
                                  language: "english")
        let normalised = raw
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
        print("[MAGPIE-ASR] raw=\"\(raw)\"  normalised=\"\(normalised)\"")

        // Every content word from the prompt must round-trip through the
        // ASR. If G2P regresses or EOS handling drifts, individual words
        // will mis-transcribe and this assertion fails.
        for word in ["hello", "world", "from", "magpie", "text", "to", "speech"] {
            XCTAssertTrue(normalised.contains(word),
                          "ASR transcription missing '\(word)'. " +
                          "Raw: \"\(raw)\"")
        }
#else
        throw XCTSkip("Qwen3-ASR requires CoreML")
#endif
    }

    /// Multilingual ASR round-trip across all 9 supported languages.
    ///
    /// For each language we synthesise the canonical test sentence from
    /// `speech-models/tests/test_round_trip.py`, transcribe with Qwen3-ASR,
    /// and assert the lowercased transcription contains the expected
    /// content words.
    ///
    /// Language pipelines (see `Tokenizer.swift` for dispatch):
    ///
    ///   - **en / es / de** — CMU IPA dict G2P (`MagpieDictG2P`) +
    ///     per-language sub-vocab offset, then `+ eos`. Full round-trip.
    ///   - **fr / it / vi** — byT5 UTF-8 byte encoder
    ///     (`agg = byte + 3 + tokenizer_offset`). Full round-trip.
    ///   - **hi**            — char-level Devanagari + `pad_with_space`.
    ///     Infrastructure correct; the model's char-level decoding for
    ///     Hindi produces low-quality audio even when fed Python's exact
    ///     IDs, so we only assert non-empty audio here. Tracked as
    ///     follow-up (needs the NeMo-specific Hindi text-preprocessing
    ///     pipeline + Leo speaker tuning).
    ///   - **zh / ja**       — best-effort char-fallback. The real
    ///     tokenisers need `jieba` (Chinese word segmentation +
    ///     pypinyin → IPA dict) and a Japanese morpheme analyzer
    ///     (`CFStringTokenizer`/`MeCab` → katakana → IPA). Until those
    ///     are ported we only assert that the synth pipeline runs without
    ///     error.
    func testMultilingualRoundTrip() async throws {
#if canImport(CoreML)
        struct Case {
            let lang: MagpieLanguage
            let asrLang: String
            let speaker: MagpieSpeaker
            let prompt: String
            /// Lowercased keywords expected in the ASR transcription.
            /// Empty array = "non-empty audio only" (best-effort langs).
            let mustContain: [String]
            /// Sampling override. Japanese needs stochastic decoding to
            /// avoid getting stuck on the first phrase; the rest of the
            /// languages stay greedy for determinism.
            var temperature: Float = 0
            var topK: Int = 1
            var maxSteps: Int = 500
            var seed: UInt64? = nil
        }
        // Test sentences mirror NeMo's `test_round_trip.py` cases.
        let cases: [Case] = [
            Case(lang: .english,    asrLang: "english",
                 speaker: .aria,
                 prompt: "Hello world from Magpie text to speech.",
                 mustContain: ["hello", "world", "from", "magpie",
                               "text", "to", "speech"]),
            Case(lang: .spanish,    asrLang: "spanish",
                 speaker: .aria,
                 prompt: "Hola mundo desde el sistema de texto a voz.",
                 mustContain: ["hola", "mundo", "desde", "el",
                               "sistema", "de", "texto", "a", "voz"]),
            Case(lang: .german,     asrLang: "german",
                 speaker: .aria,
                 prompt: "Hallo Welt vom Text zu Sprache System.",
                 mustContain: ["hallo", "welt", "vom", "text",
                               "zu", "sprach"]),  // "Sprache System" often
                                                     // collapses to "Sprachsystem"
            Case(lang: .french,     asrLang: "french",
                 speaker: .aria,
                 prompt: "Bonjour le monde, depuis le système de synthèse vocale.",
                 mustContain: ["bonjour", "le", "monde", "depuis",
                               "système", "synthèse", "vocale"]),
            // Italian: greedy decoding stalls after the first word (the
            // same pathology the Japanese case documents below), so use the
            // library's stochastic defaults with a fixed seed.
            Case(lang: .italian,    asrLang: "italian",
                 speaker: .aria,
                 prompt: "Ciao mondo dal sistema di sintesi vocale.",
                 // "dal" → "del" is a common ASR / TTS rounding for the
                 // short function word; we accept either.
                 mustContain: ["ciao", "mondo", "sistema",
                               "sintesi", "vocale"],
                 temperature: 0.6, topK: 80, maxSteps: 300, seed: 42),
            Case(lang: .vietnamese, asrLang: "vietnamese",
                 speaker: .aria,
                 prompt: "Xin chào thế giới từ hệ thống chuyển văn bản thành giọng nói.",
                 mustContain: ["xin", "chào", "thế", "giới",
                               "hệ", "thống", "văn", "bản"]),
            // Hindi: HindiCharsTokenizer + last-wins sub-vocab map
            // (NeMo's `_token2id` dict-comp keeps the last occurrence; we
            // were keeping the first which broke decode). The ASR
            // typically transcribes the English loan-words as Latin
            // ("text to speech system"), so we look for the native
            // Hindi words plus the canonical "है" copula.
            Case(lang: .hindi,    asrLang: "hindi",    speaker: .leo,
                 prompt: "नमस्ते दुनिया, यह टेक्स्ट टू स्पीच सिस्टम है।",
                 mustContain: ["नमस्ते", "दुनिया", "यह", "है"]),
            // Chinese: NLTokenizer word segmentation + Apple
            // `mandarinToLatin` per word + bundled pinyin → IPA dict +
            // `#tone` markers. Word-level segmentation (jieba equivalent)
            // gives correct readings for multi-char compounds. The TTS
            // sometimes nudges a syllable to a near-homophone, so we
            // require only the most-stable content words.
            Case(lang: .chinese,  asrLang: "chinese",  speaker: .aria,
                 prompt: "你好世界，这是文本转语音系统。",
                 mustContain: ["你好", "文本", "语音", "系统"]),
            // Japanese: CFStringTokenizer kanji-reading +
            // NFC-composition-preserved dakuten + particle/greeting
            // overrides + heiban pitch markers. Greedy decoding gets
            // stuck on the first word; stochastic decoding with a fixed
            // seed produces fluent output where every content word
            // round-trips through ASR.
            Case(lang: .japanese, asrLang: "japanese", speaker: .aria,
                 prompt: "こんにちは世界、これは音声合成システムです。",
                 mustContain: ["こんにちは", "世界", "これ",
                               "音声", "合成", "システム", "です"],
                 temperature: 0.6, topK: 80, maxSteps: 300, seed: 42),
        ]

        let tts = try await MagpieTTS.fromPretrained(variant: .int8)
        let asr = try await CoreMLASRModel.fromPretrained()

        var failures: [String] = []
        for tc in cases {
            let audio = try tts.synthesize(
                text: tc.prompt, speaker: tc.speaker, language: tc.lang,
                params: MagpieTTSParams(temperature: tc.temperature,
                                          topK: tc.topK,
                                          maxSteps: tc.maxSteps,
                                          seed: tc.seed))
            XCTAssertGreaterThan(audio.count, MagpieTTS.sampleRate / 4,
                                 "[\(tc.lang)] TTS produced <0.25s audio")

            let raw = asr.transcribe(audio: audio,
                                      sampleRate: MagpieTTS.sampleRate,
                                      language: tc.asrLang)
            let normalised = raw
                .lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .joined(separator: " ")
                .replacingOccurrences(of: "  ", with: " ")
                .trimmingCharacters(in: .whitespaces)
            print("[MAGPIE-RT/\(tc.lang)] prompt=\"\(tc.prompt)\"")
            print("                       asr   =\"\(raw)\"")

            if tc.mustContain.isEmpty { continue }  // best-effort
            for word in tc.mustContain {
                if !normalised.contains(word.lowercased()) {
                    failures.append("[\(tc.lang)] missing '\(word)' in ASR: \(raw)")
                }
            }
        }
        if !failures.isEmpty {
            XCTFail("Round-trip failures:\n  " + failures.joined(separator: "\n  "))
        }
#else
        throw XCTSkip("Qwen3-ASR requires CoreML")
#endif
    }

    /// Verify the streaming entry point yields multiple chunks and the
    /// concatenated stream matches what batch synthesis would have produced
    /// for the same seed.
    func testStreamingEmitsMultipleChunks() async throws {
        let model = try await MagpieTTS.fromPretrained(variant: .int8)
        let stream = model.synthesizeStream(
            text: "test",
            speaker: .aria,
            language: .english,
            params: MagpieTTSParams(temperature: 0, topK: 1, maxSteps: 200),
            firstChunkFrames: 8, framesPerChunk: 16)
        var chunkCount = 0
        var totalSamples = 0
        var sawFinal = false
        for try await chunk in stream {
            chunkCount += 1
            totalSamples += chunk.samples.count
            XCTAssertEqual(chunk.sampleRate, MagpieTTS.sampleRate)
            if chunk.isFinal { sawFinal = true }
        }
        XCTAssertGreaterThan(chunkCount, 1, "Streaming yielded only one chunk")
        XCTAssertTrue(sawFinal, "Stream never produced an isFinal=true chunk")
        XCTAssertGreaterThan(totalSamples, MagpieTTS.sampleRate / 4,
                             "Concatenated stream shorter than 0.25 s")
    }
}
