import XCTest
import MLX
import MLXNN
import MLXRandom
import AudioCommon
import ParakeetASR
import PersonaPlex
import Qwen3TTS
import MADLADTranslation
@testable import HibikiTranslate

/// E2E tests that download the Hibiki Zero-3B model from HuggingFace and
/// verify weights load + forward pass produces sensible outputs.
///
/// Skipped by default. Enable with `HIBIKI_E2E=1`. Override the model id with
/// `HIBIKI_MODEL_ID=<repo>` (default `aufklarer/Hibiki-Zero-3B-MLX-4bit`).
final class E2EHibikiTranslateTests: XCTestCase {

    /// Verifies the model downloads, weights load with `verify: .noUnusedKeys`,
    /// and a forward pass on synthetic source-audio tokens produces non-NaN
    /// finite text logits with the expected shape.
    func testZero3BLoadAndForward() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E tests (~2.7 GB download)")

        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId

        let model = try await HibikiTranslateModel.fromPretrained(
            modelId: modelId,
            progressHandler: { p, msg in
                if Int(p * 100) % 10 == 0 {
                    print("[hibiki-load] \(Int(p * 100))% \(msg)")
                }
            }
        )

        // Synthetic 1-frame input: text=padding, all 32 audio streams = -1 (masked).
        let cfg = model.cfg
        let textTokens = MLXArray([Int32(cfg.temporal.textPaddingId)]).reshaped([1, 1])
        let audioTokens = MLXArray.full(
            [1, cfg.temporal.numAudioEmbeddings, 1],
            values: MLXArray(Int32(-1))
        )

        let (hidden, textLogits) = model.temporal.forward(
            textTokens: textTokens, audioTokens: audioTokens, offset: 0)
        eval(hidden)
        eval(textLogits)

        XCTAssertEqual(hidden.shape, [1, 1, cfg.temporal.dim],
                       "hidden state should be [1, 1, dim]")
        XCTAssertEqual(textLogits.shape, [1, 1, cfg.temporal.textCard],
                       "text logits should be [1, 1, textCard]")

        // Verify logits are finite (not NaN, not Inf) — proves weights loaded
        // and the forward pass through 28 GQA layers works on real weights.
        let logitsHost = textLogits.asArray(Float.self)
        let nanCount = logitsHost.filter { $0.isNaN }.count
        let infCount = logitsHost.filter { $0.isInfinite }.count
        XCTAssertEqual(nanCount, 0, "text logits should have zero NaN values")
        XCTAssertEqual(infCount, 0, "text logits should have zero Inf values")

        // Top-5 tokens — eyeball check: should not be wildly degenerate.
        let topIdx = argSort(textLogits.squeezed()).asArray(Int32.self)
        let top5 = Array(topIdx.suffix(5).reversed())
        print("[hibiki-forward] top-5 text token ids: \(top5)")
    }

    /// Smoke-tests `HibikiDepformer.generate` on real weights with a trivial
    /// argmax sampler. Verifies the 9-slice schedule wired correctly through
    /// all 16 generation steps.
    func testZero3BDepformerGenerateOnRealWeights() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E tests")

        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId

        let model = try await HibikiTranslateModel.fromPretrained(modelId: modelId)
        let cfg = model.cfg

        // Random temporal hidden state (will produce gibberish tokens but the
        // shape and schedule should be exercised correctly).
        let temporalHidden = MLXRandom.normal([1, 1, cfg.temporal.dim])
        eval(temporalHidden)
        let textToken = MLXArray([Int32(cfg.temporal.textPaddingId)])

        // Simple argmax sampler.
        let sampler: (MLXArray, Int) -> MLXArray = { logits, _ in
            argMax(logits, axis: -1).asType(.int32)
        }

        let tokens = model.depformer.generate(
            temporalHidden: temporalHidden,
            textToken: textToken,
            sampleFn: sampler
        )
        eval(tokens)

        XCTAssertEqual(tokens.shape, [1, cfg.depformer.numSteps],
                       "depformer should emit [1, 16] target tokens")
        let tokensHost = tokens.asArray(Int32.self)
        for t in tokensHost {
            XCTAssertGreaterThanOrEqual(t, 0, "token \(t) below valid range")
            XCTAssertLessThan(t, Int32(cfg.depformer.card),
                              "token \(t) above audio cardinality \(cfg.depformer.card)")
        }
    }

    /// **End-to-end round-trip translation test**: known FR audio →
    /// Hibiki Zero-3B → EN audio → Parakeet ASR → assert plausible English.
    ///
    /// Reference content for `fleurs_fr.wav` (FLEURS dataset):
    ///   FR: "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire."
    ///   EN: "Think of the ski route as a similar hiking route."
    ///   Expected English keywords (any subset is meaningful): think, ski,
    ///   route, trail, hiking, hike, similar.
    ///
    /// Exercises every milestone end-to-end: config, rope_concat, GQA temporal,
    /// scheduled depformer + per-step LayerNorm, weight loader, driver, Mimi
    /// encode/decode, ASR pipeline. Skip with `HIBIKI_E2E` unset.
    ///
    /// - FR: strict by default — fails if 0 expected keywords match (this clip
    ///   is the regression canary; set `HIBIKI_LENIENT=1` to demote to warn).
    /// - ES/PT/DE: warn-only initially; set `HIBIKI_STRICT_ALL=1` to require
    ///   ≥1 keyword on those too.
    func testFrenchToEnglishTranslation() async throws {
        try await runHibikiTranslationCase(
            resource: "fleurs_fr",
            referenceSource: "Pensez à l'itinéraire de ski comme à un itinéraire de randonnée similaire.",
            referenceEN: "Think of the ski route as a similar hiking route.",
            expectedKeywords: ["think", "ski", "route", "trail", "hike",
                               "hiking", "similar", "path"],
            sourceLang: .fr,
            // FR is the canary: strict by default.
            strictByDefault: true
        )
    }

    /// Spanish test uses a 5s trimmed excerpt from Hibiki Zero's official
    /// samples space (europarl_st/es/source/5dc1d533...mp3, 24 kHz native,
    /// TTS-generated by 11labs). FLEURS Spanish clips (16 kHz human news
    /// recordings) are out-of-distribution for Hibiki Zero and trigger
    /// degenerate generation in BOTH the Python upstream and the Swift
    /// port — Python emits 1643 steps / 131 s of broken audio; Swift's
    /// output is ASR-undecipherable. This in-distribution sample produces
    /// clean, EOS-terminated English on the first try.
    func testSpanishToEnglishTranslation() async throws {
        try await runHibikiTranslationCase(
            resource: "hibiki_official_es_5s",
            referenceSource: "(europarl_st 5dc1d533, Spanish political-speech excerpt)",
            referenceEN: "Gentlemen, the data is worrying. (Hibiki greedy output.)",
            expectedKeywords: ["gentlemen", "data", "worrying", "concerning",
                               "alarming", "figures"],
            sourceLang: .es,
            // Strict by default — this is an in-distribution clip and a
            // clean regression signal (Hibiki greedy reliably hits ≥3 of
            // the keywords above).
            strictByDefault: true
        )
    }

    func testPortugueseToEnglishTranslation() async throws {
        try await runHibikiTranslationCase(
            resource: "fleurs_pt",
            referenceSource: "É o quinto CEP do Martelly em quatro anos.",
            referenceEN: "It is the fifth CEP for Martelly in four years.",
            expectedKeywords: ["fifth", "fourth", "four", "year", "years",
                               "cep", "martelly"],
            sourceLang: .pt,
            strictByDefault: false
        )
    }

    func testGermanToEnglishTranslation() async throws {
        try await runHibikiTranslationCase(
            resource: "fleurs_de",
            referenceSource: "Das erschien mir nicht sinnvoll; es war ganz gewiss nicht fair.",
            referenceEN: "It didn't seem sensible to me; it certainly wasn't fair.",
            expectedKeywords: ["seem", "sensible", "meaningful", "fair",
                               "certainly", "sure", "right", "wasn't", "was"],
            sourceLang: .de,
            strictByDefault: false
        )
    }

    /// Shared runner for the per-language translation tests.
    /// FR is strict by default (the regression canary). ES/PT/DE are warn-only
    /// initially because quality on those languages hasn't been characterized
    /// — flip with `HIBIKI_STRICT_ALL=1`. `HIBIKI_LENIENT=1` demotes FR back
    /// to warn-only when debugging.
    private func runHibikiTranslationCase(
        resource: String,
        referenceSource: String,
        referenceEN: String,
        expectedKeywords: [String],
        sourceLang: HibikiSourceLanguage,
        strictByDefault: Bool
    ) async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run Hibiki E2E translation")

        let env = ProcessInfo.processInfo.environment
        let lenient = env["HIBIKI_LENIENT"] != nil
        let strictAll = env["HIBIKI_STRICT_ALL"] != nil
        let strict = (strictByDefault && !lenient) || strictAll

        let tag = "[hibiki-e2e:\(sourceLang.rawValue)]"

        // 1. Load source audio.
        guard let srcURL = Bundle.module.url(forResource: resource, withExtension: "wav") else {
            XCTFail("\(resource).wav missing from test resources"); return
        }
        let pcm = try AudioFileLoader.load(url: srcURL, targetSampleRate: 24000)
        let inputDuration = Double(pcm.count) / 24000.0
        print("\(tag) loaded \(resource).wav: \(pcm.count) samples, " +
              "\(String(format: "%.2f", inputDuration))s")
        print("\(tag) reference source: \(referenceSource)")
        print("\(tag) reference EN: \(referenceEN)")

        // 2. Load Hibiki Zero-3B.
        let modelId = env["HIBIKI_MODEL_ID"] ?? HibikiTranslateModel.defaultModelId
        let model = try await HibikiTranslateModel.fromPretrained(
            modelId: modelId,
            progressHandler: { p, msg in
                if Int(p * 100) % 20 == 0 { print("\(tag) load \(Int(p * 100))% \(msg)") }
            }
        )

        // 3. Translate.
        let (audio, textTokens) = model.translate(
            sourceAudio: pcm, sourceLanguage: sourceLang, verbose: true
        )
        let outputDuration = Double(audio.count) / 24000.0

        XCTAssertGreaterThan(audio.count, Int(0.5 * 24000),
                             "should produce > 0.5 s of English audio")
        // Hibiki Zero generates PAD during the audio-streaming window, then
        // content, then EOS — total output runs 1.0×–2.5× the input duration.
        // Python upstream emits ~1.5× for FLEURS clips. Generation stops on
        // sampled EOS (post-source), capped at 2.5× as a safety bound.
        XCTAssertGreaterThan(outputDuration, inputDuration * 0.7,
                             "output duration should be at least 70% of input")
        XCTAssertLessThan(outputDuration, inputDuration * 2.6,
                          "output duration should be at most 260% of input")

        print("\(tag) translated → \(audio.count) samples, " +
              "\(String(format: "%.2f", outputDuration))s; " +
              "\(textTokens.count) text tokens")

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        let absMax = audio.map { abs($0) }.max() ?? 0
        print("\(tag) output audio RMS: \(String(format: "%.4f", rms)), peak: \(String(format: "%.4f", absMax))")
        XCTAssertGreaterThan(rms, 0.001,
            "output audio should have non-trivial RMS energy (not silence)")

        // 4. Save output for inspection.
        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hibiki-e2e", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let outURL = outDir.appendingPathComponent("translated_\(sourceLang.rawValue)_en.wav")
        try WAVWriter.write(samples: audio, sampleRate: 24000, to: outURL)
        print("\(tag) wrote translated audio: \(outURL.path)")

        // 5. ASR.
        let asr = try await ParakeetASRModel.fromPretrained()
        let transcript = asr.transcribe(audio: audio, sampleRate: 24000, language: nil)
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        print("\(tag) Parakeet ASR of EN output: '\(transcript)'")

        let hits = expectedKeywords.filter { transcript.contains($0) }
        print("\(tag) expected keyword hits: \(hits.isEmpty ? "[none]" : hits.joined(separator: ", "))")

        if strict {
            XCTAssertFalse(transcript.isEmpty,
                "STRICT: Parakeet transcript empty for output of '\(referenceSource)'.")
            XCTAssertFalse(hits.isEmpty,
                "STRICT: expected ≥1 of \(expectedKeywords) in transcript '\(transcript)'. " +
                "Reference EN: '\(referenceEN)'.")
        } else {
            if transcript.isEmpty {
                print("\(tag) WARNING: Parakeet transcript empty (low-quality on this clip)")
            } else if hits.isEmpty {
                print("\(tag) WARNING: 0 expected keywords matched (set HIBIKI_STRICT_ALL=1 to fail)")
            }
        }
    }

    /// **Closed-loop diagnostic round trip**: known FR text → Qwen3TTS → FR
    /// audio → Hibiki Zero-3B → EN audio → Parakeet ASR → EN text.
    ///
    /// This isolates "is the model output meaningful?" from "is the input
    /// sample in-distribution?" — TTS-generated speech is consistent and
    /// in-distribution for Hibiki's training. Run with `HIBIKI_E2E=1`.
    ///
    /// Logs all four stages so we can inspect where quality degrades.
    func testClosedLoopTTSToHibikiToASR() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run closed-loop TTS↔Hibiki↔ASR round trip")

        // Test cases: known FR text + expected EN keywords (any reasonable subset).
        // Three short, simple sentences from common everyday speech — should be
        // well within Hibiki Zero-3B's training distribution.
        struct Case {
            let frText: String
            let expectedEN: String
            let keywords: [String]
        }
        let cases: [Case] = [
            Case(frText: "Bonjour, comment allez-vous aujourd'hui?",
                 expectedEN: "Hello, how are you today?",
                 keywords: ["hello", "hi", "how", "are", "you", "today"]),
            Case(frText: "J'aime beaucoup les pommes rouges.",
                 expectedEN: "I really like red apples.",
                 keywords: ["like", "love", "red", "apple", "apples"]),
            Case(frText: "Le chat dort sur le canapé.",
                 expectedEN: "The cat sleeps on the couch.",
                 keywords: ["cat", "sleep", "sleeps", "couch", "sofa"]),
        ]

        // 1. Load TTS, Hibiki, ASR — once.
        print("[hibiki-loop] loading Qwen3TTS...")
        let tts = try await Qwen3TTSModel.fromPretrained()
        print("[hibiki-loop] loading Hibiki Zero-3B...")
        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId
        let hibiki = try await HibikiTranslateModel.fromPretrained(modelId: modelId)
        print("[hibiki-loop] loading Parakeet ASR...")
        let asr = try await ParakeetASRModel.fromPretrained()

        // SPM-48k decoder for Hibiki's inner-monologue text. Cached alongside
        // the model. Used to print the model's "thoughts" — what English the
        // temporal+text path is producing — independently of the audio path.
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let spmPath = modelDir.appendingPathComponent("tokenizer_spm_48k_multi6_2.model").path
        let spmDecoder: SentencePieceDecoder?
        if FileManager.default.fileExists(atPath: spmPath) {
            spmDecoder = try? SentencePieceDecoder(modelPath: spmPath)
            if spmDecoder != nil {
                print("[hibiki-loop] loaded SPM-48k decoder for inner-monologue")
            }
        } else {
            spmDecoder = nil
            print("[hibiki-loop] WARN: SPM-48k tokenizer not found at \(spmPath)")
        }

        var anyKeywordHit = false
        var allResults: [(Case, String, String)] = []   // (case, ASR transcript, inner monologue)

        for (idx, c) in cases.enumerated() {
            print("\n[hibiki-loop] === Case \(idx + 1)/\(cases.count) ===")
            print("[hibiki-loop] FR text:   \(c.frText)")
            print("[hibiki-loop] Reference EN: \(c.expectedEN)")

            // 2. Synthesize French audio.
            let frAudio = tts.synthesize(text: c.frText, language: "french", languageExplicit: true)
            let frDur = Double(frAudio.count) / 24000.0
            print("[hibiki-loop] Qwen3TTS produced FR audio: " +
                  "\(frAudio.count) samples, \(String(format: "%.2f", frDur))s")
            XCTAssertGreaterThan(frAudio.count, Int(0.5 * 24000),
                                 "Qwen3TTS should produce > 0.5s of audio")

            // 3. Translate FR → EN audio with Hibiki.
            let (enAudio, textTokens) = hibiki.translate(
                sourceAudio: frAudio, sourceLanguage: .fr, verbose: false
            )
            let enDur = Double(enAudio.count) / 24000.0
            let rms = sqrt(enAudio.map { $0 * $0 }.reduce(0, +) / Float(enAudio.count))
            print("[hibiki-loop] Hibiki produced EN audio: " +
                  "\(enAudio.count) samples, \(String(format: "%.2f", enDur))s, " +
                  "RMS=\(String(format: "%.4f", rms)), " +
                  "\(textTokens.count) text tokens")

            // 3a. Decode inner-monologue text tokens — what is the model
            // "thinking" in English, regardless of audio quality?
            let innerMonologue = spmDecoder?.decode(textTokens) ?? "<no SPM decoder>"
            print("[hibiki-loop] Hibiki inner-monologue: '\(innerMonologue)'")
            // Raw token IDs for diagnostics (first 20).
            let rawIds = textTokens.prefix(20).map { String($0) }.joined(separator: ",")
            print("[hibiki-loop] first 20 text token IDs: \(rawIds)")

            // Save for inspection.
            let outDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("hibiki-loop", isDirectory: true)
            try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
            let frURL = outDir.appendingPathComponent("case\(idx + 1)_fr.wav")
            let enURL = outDir.appendingPathComponent("case\(idx + 1)_en.wav")
            try WAVWriter.write(samples: frAudio, sampleRate: 24000, to: frURL)
            try WAVWriter.write(samples: enAudio, sampleRate: 24000, to: enURL)
            print("[hibiki-loop] saved: \(frURL.lastPathComponent), \(enURL.lastPathComponent)")

            // 4. ASR the EN audio.
            let transcript = asr.transcribe(audio: enAudio, sampleRate: 24000, language: nil)
                .lowercased()
                .trimmingCharacters(in: .whitespacesAndNewlines)
            print("[hibiki-loop] Parakeet EN transcript: '\(transcript)'")

            // 5. Keyword hit count.
            let hits = c.keywords.filter { transcript.contains($0) }
            print("[hibiki-loop] keyword hits: \(hits.isEmpty ? "[none]" : hits.joined(separator: ", "))")
            if !hits.isEmpty { anyKeywordHit = true }

            allResults.append((c, transcript, innerMonologue))

            // Per-case structural assertion: synchronous 1:1 from Hibiki.
            XCTAssertGreaterThan(enDur, frDur * 0.7, "1:1 sync: EN ≥ 0.7×FR")
            XCTAssertLessThan(enDur, frDur * 1.3, "1:1 sync: EN ≤ 1.3×FR")
        }

        // Summary table.
        print("\n[hibiki-loop] ================ SUMMARY ================")
        for (i, (c, transcript, monologue)) in allResults.enumerated() {
            print("[hibiki-loop] \(i + 1). FR input:        \(c.frText)")
            print("[hibiki-loop]    Reference EN:    \(c.expectedEN)")
            print("[hibiki-loop]    Inner-monologue: '\(monologue)'")
            print("[hibiki-loop]    ASR transcript:  '\(transcript)'")
            let h = c.keywords.filter { transcript.contains($0) }
            let m = c.keywords.filter { monologue.lowercased().contains($0) }
            print("[hibiki-loop]    Keyword hits — text: \(m.isEmpty ? "NONE" : m.joined(separator: ", "))")
            print("[hibiki-loop]    Keyword hits — audio: \(h.isEmpty ? "NONE" : h.joined(separator: ", "))")
        }
        print("[hibiki-loop] ============================================")
        print("[hibiki-loop]")
        print("[hibiki-loop] Diagnostic interpretation:")
        print("[hibiki-loop]   - text hits = inner-monologue (temporal+text path)")
        print("[hibiki-loop]   - audio hits = ASR'd output (full pipeline incl. depformer + Mimi decode)")
        print("[hibiki-loop]   - text OK + audio FAIL → audio path bug (depformer or Mimi decode)")
        print("[hibiki-loop]   - both FAIL → temporal forward broken")
        print("[hibiki-loop]   - both OK   → translation works")

        // Soft success criterion: at least one case across all 3 should produce
        // a recognizable English keyword. This is a low bar but a much stricter
        // signal than the FLEURS-only test.
        if !anyKeywordHit {
            print("[hibiki-loop] WARNING: 0 keyword hits across all 3 test cases. " +
                  "This suggests a real translation-quality issue in our port.")
        }
        // Don't hard-fail unless explicitly requested; this test is diagnostic
        // and we want the full output table even when quality is poor.
        if ProcessInfo.processInfo.environment["HIBIKI_STRICT"] != nil {
            XCTAssertTrue(anyKeywordHit,
                "STRICT: at least one of 3 cases should produce ≥1 expected keyword")
        }
    }

    /// **Full English ↔ English loop** for human verification.
    ///
    /// You provide an English source sentence; we run:
    ///
    ///   EN text  → MADLAD     → FR text
    ///            → Qwen3TTS   → FR audio (saved to /tmp/hibiki-loop/case_N_fr.wav)
    ///            → Hibiki     → EN audio (saved to /tmp/hibiki-loop/case_N_en.wav)
    ///            → Parakeet   → EN text  ← compared back to source
    ///
    /// Every text stage is printed so you can read the chain end-to-end.
    /// Skip with `HIBIKI_E2E` unset.
    func testEnglishLoopThroughFrench() async throws {
        let hasEnv = ProcessInfo.processInfo.environment["HIBIKI_E2E"] != nil
        try XCTSkipUnless(hasEnv, "Set HIBIKI_E2E=1 to run English↔English loop test")

        // Five everyday English sentences across different topics.
        let englishSources = [
            "Hello, how are you today?",
            "I really like red apples.",
            "The cat sleeps on the couch.",
            "Tomorrow we will go to the park.",
            "Could you please pass the salt?",
        ]

        // 1. Load all four models — once.
        print("[en-loop] loading MADLAD translator...")
        let madlad = try await MADLADTranslator.fromPretrained()
        print("[en-loop] loading Qwen3TTS...")
        let tts = try await Qwen3TTSModel.fromPretrained()
        print("[en-loop] loading Hibiki Zero-3B...")
        let modelId = ProcessInfo.processInfo.environment["HIBIKI_MODEL_ID"]
            ?? HibikiTranslateModel.defaultModelId
        let hibiki = try await HibikiTranslateModel.fromPretrained(modelId: modelId)
        print("[en-loop] loading Parakeet ASR...")
        let asr = try await ParakeetASRModel.fromPretrained()

        // SPM-48k decoder for inner-monologue.
        let modelDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let spmPath = modelDir.appendingPathComponent("tokenizer_spm_48k_multi6_2.model").path
        let spmDecoder = try? SentencePieceDecoder(modelPath: spmPath)

        let outDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("hibiki-loop", isDirectory: true)
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        struct Result {
            let originalEN: String
            let translatedFR: String
            let innerMonologue: String
            let parakeetEN: String
            let frPath: URL
            let enPath: URL
        }
        var results: [Result] = []

        for (i, sourceEN) in englishSources.enumerated() {
            print("\n[en-loop] === Case \(i + 1)/\(englishSources.count) ===")
            print("[en-loop] EN source:           '\(sourceEN)'")

            // 2. EN → FR via MADLAD.
            let frText = try madlad.translate(sourceEN, to: "fr")
            print("[en-loop] MADLAD EN→FR:        '\(frText)'")

            // 3. FR text → FR audio via Qwen3TTS.
            let frAudio = tts.synthesize(text: frText, language: "french", languageExplicit: true)
            let frDur = Double(frAudio.count) / 24000.0
            print("[en-loop] Qwen3TTS FR audio:   \(String(format: "%.2f", frDur))s")

            // 4. FR audio → EN audio via Hibiki.
            let (enAudio, textTokens) = hibiki.translate(
                sourceAudio: frAudio, sourceLanguage: .fr, verbose: false
            )
            let enDur = Double(enAudio.count) / 24000.0
            let rms = sqrt(enAudio.map { $0 * $0 }.reduce(0, +) / Float(enAudio.count))
            print("[en-loop] Hibiki EN audio:     " +
                  "\(String(format: "%.2f", enDur))s, RMS=\(String(format: "%.3f", rms)), " +
                  "\(textTokens.count) text tokens")

            let inner = spmDecoder?.decode(textTokens) ?? ""
            print("[en-loop] Hibiki inner-thought: '\(inner)'")

            // 5. EN audio → EN text via Parakeet.
            let parakeetEN = asr.transcribe(audio: enAudio, sampleRate: 24000, language: nil)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            print("[en-loop] Parakeet EN:         '\(parakeetEN)'")

            let frURL = outDir.appendingPathComponent("case\(i + 1)_fr.wav")
            let enURL = outDir.appendingPathComponent("case\(i + 1)_en.wav")
            try WAVWriter.write(samples: frAudio, sampleRate: 24000, to: frURL)
            try WAVWriter.write(samples: enAudio, sampleRate: 24000, to: enURL)

            results.append(Result(
                originalEN: sourceEN, translatedFR: frText,
                innerMonologue: inner, parakeetEN: parakeetEN,
                frPath: frURL, enPath: enURL
            ))
        }

        // 6. Final summary table — readable end-to-end.
        print("\n[en-loop] ============== FINAL TABLE ==============")
        for (i, r) in results.enumerated() {
            print("[en-loop] \(i + 1).")
            print("[en-loop]   EN  in:    '\(r.originalEN)'")
            print("[en-loop]   FR  trans: '\(r.translatedFR)'")
            print("[en-loop]   inner:     '\(r.innerMonologue)'")
            print("[en-loop]   EN  out:   '\(r.parakeetEN)'")
            print("[en-loop]   audio:     \(r.frPath.path)")
            print("[en-loop]              \(r.enPath.path)")
        }
        print("[en-loop] =========================================")
        print("[en-loop] You can listen to the audio files at /tmp/hibiki-loop/")
    }
}
