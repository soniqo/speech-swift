import XCTest
@testable import SupertonicTTS
@testable import Qwen3ASR

/// E2E tests using fromPretrained() — downloads `aufklarer/Supertonic-3-CoreML` from HuggingFace.
/// Run with: `swift test --filter SupertonicE2ETests`
/// Offline: set `SUPERTONIC_LOCAL_PATH=/path/to/Supertonic-3-CoreML` to skip the download.
final class SupertonicE2ETests: XCTestCase {
    private static var _tts: SupertonicTTSModel?
    private static var _asr: Qwen3ASRModel?

    private func model() async throws -> SupertonicTTSModel {
        if let m = Self._tts { return m }
        let localPath = ProcessInfo.processInfo.environment["SUPERTONIC_LOCAL_PATH"]
        let m = try await SupertonicTTSModel.fromPretrained(localPath: localPath) { p, s in
            print("[Supertonic \(Int(p * 100))%] \(s)")
        }
        Self._tts = m
        return m
    }

    private func asr() async throws -> Qwen3ASRModel {
        if let m = Self._asr { return m }
        let m = try await Qwen3ASRModel.fromPretrained()
        Self._asr = m
        return m
    }

    func testModelLoadsAndHasVoices() async throws {
        let m = try await model()
        XCTAssertGreaterThan(m.availableVoices.count, 1, "fromPretrained must download the full voice catalog")
        XCTAssertTrue(m.availableVoices.contains(m.defaultVoice))
        XCTAssertEqual(m.sampleRate, 44100)
    }

    func testSynthesizeEnglish() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "Hello from soniqo dot audio.", voice: "F1", language: "en")

        XCTAssertGreaterThan(audio.count, 1000, "Should produce meaningful audio")
        let duration = Double(audio.count) / 44100.0
        print("English: \(audio.count) samples (\(String(format: "%.2f", duration))s)")
        XCTAssertGreaterThan(duration, 0.3)
        XCTAssertLessThan(duration, 12.0)

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")
        XCTAssertLessThan(audio.map { abs($0) }.max() ?? 0, 1.001, "No clipping past full scale")
    }

    func testSynthesizeShortText() async throws {
        let m = try await model()
        let audio = try m.synthesize(text: "Hi.", voice: "M1", language: "en")
        XCTAssertGreaterThan(audio.count, 0, "Even short text should produce audio")
    }

    func testReproducibleWithSeed() async throws {
        let m = try await model()
        let opts = SupertonicOptions(totalStep: 8, speed: 1.05, seed: 1234)
        let a = try m.synthesize(text: "Reproducible.", voice: "F1", language: "en", options: opts)
        let b = try m.synthesize(text: "Reproducible.", voice: "F1", language: "en", options: opts)
        XCTAssertEqual(a.count, b.count)
        XCTAssertEqual(a.first, b.first, "A fixed seed must be deterministic")
    }

    // MARK: - Roundtrip (TTS → ASR → words come back)

    /// Synthesize English with Supertonic, transcribe with Qwen3-ASR, verify the words survive.
    func testEnglishRoundTrip() async throws {
        let tts = try await model()
        let asr = try await asr()

        let inputText = "The quick brown fox jumps over the lazy dog."
        let audio = try tts.synthesize(text: inputText, voice: "F2", language: "en")
        XCTAssertGreaterThan(audio.count, 1000)

        let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
        XCTAssertGreaterThan(rms, 0.001, "Audio should not be silence")

        let transcription = asr.transcribe(audio: audio, sampleRate: 44100).lowercased()
        print("Input:  \"\(inputText)\"")
        print("Output: \"\(transcription)\"")

        let keywords = ["quick", "brown", "fox", "lazy", "dog"]
        let matched = keywords.filter { transcription.contains($0) }
        print("Matched \(matched.count)/\(keywords.count): \(matched)")
        XCTAssertGreaterThanOrEqual(matched.count, 2,
            "At least 2 of \(keywords) should be transcribed from Supertonic speech: \"\(transcription)\"")
    }

    /// Multilingual synthesis across languages + voices. The G2P-free front-end has no per-language
    /// phonemizer (NFKD + index only), so every language goes through the identical path — each must
    /// produce non-silent, sensibly-sized 44.1 kHz audio. (Intelligibility is covered by the English
    /// TTS→ASR roundtrip above; this asserts the multilingual path itself, CoreML-only, no MLX/ASR.)
    func testMultilingualSynthesis() async throws {
        let tts = try await model()
        struct Case { let lang: String; let voice: String; let text: String }
        let cases = [
            Case(lang: "de", voice: "M2", text: "Guten Morgen, wie geht es Ihnen heute?"),
            Case(lang: "es", voice: "F3", text: "Hola, esto es una prueba de la voz."),
            Case(lang: "fr", voice: "M3", text: "Bonjour, ceci est un test de la voix."),
            Case(lang: "ru", voice: "F4", text: "Привет, это проверка голоса."),
            Case(lang: "it", voice: "M4", text: "Ciao, questa è una prova vocale."),
            Case(lang: "ko", voice: "F5", text: "안녕하세요, 음성 테스트입니다."),
        ]
        for c in cases {
            let audio = try tts.synthesize(text: c.text, voice: c.voice, language: c.lang)
            let dur = Double(audio.count) / 44100.0
            let rms = sqrt(audio.map { $0 * $0 }.reduce(0, +) / Float(audio.count))
            print("[\(c.lang)/\(c.voice)] \(audio.count) samples (\(String(format: "%.2f", dur))s) rms=\(String(format: "%.4f", rms))")
            XCTAssertGreaterThan(audio.count, 1000, "\(c.lang): should produce audio")
            XCTAssertGreaterThan(dur, 0.3, "\(c.lang): plausible duration")
            XCTAssertLessThan(dur, 12.0, "\(c.lang): plausible duration")
            XCTAssertGreaterThan(rms, 0.001, "\(c.lang): audio should not be silence")
            XCTAssertLessThan(audio.map { abs($0) }.max() ?? 0, 1.001, "\(c.lang): no clipping")
        }
    }
}
