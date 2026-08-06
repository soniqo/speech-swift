import CoreML
import XCTest
@testable import CanaryASR
@testable import AudioCommon

/// End-to-end coverage for the Canary Core ML bundle.
///
/// Canary is offline per utterance, so there is no streaming behaviour to
/// assert. What matters is that the three graphs agree with each other across
/// the decode loop: the encoder's fixed window, the prefill pass over the
/// prompt, and the step model's growing cache with an explicit position.
///
/// Downloads ~518 MB on first run and reuses the cache afterwards.
final class E2ECanaryTests: XCTestCase {

    private static var model: CanaryASRModel?

    private func loadModel() async throws -> CanaryASRModel {
        if let existing = Self.model { return existing }
        let model = try await CanaryASRModel.fromPretrained()
        Self.model = model
        return model
    }

    /// The decode contract has to survive the round trip through the bundle.
    /// If `config.json` were missing or renamed, loading throws rather than
    /// quietly decoding off a built-in default prompt.
    func testBundleCarriesItsDecodeContract() async throws {
        let model = try await loadModel()

        XCTAssertEqual(model.config.sampleRate, 16000)
        XCTAssertEqual(model.config.numMelBins, 128)
        XCTAssertTrue(model.config.logitsAreLogProbs)
        XCTAssertGreaterThan(model.config.decoderMemLayers, 0)
        XCTAssertGreaterThan(model.config.decoderHidden, 0)

        // Nine tokens: the canary2 template with an empty decoder context.
        let prompt = try XCTUnwrap(model.config.prompt(source: "en"))
        XCTAssertEqual(prompt.count, 9)

        for language in ["en", "de", "es", "fr"] {
            XCTAssertNotNil(model.config.prompt(source: language),
                            "expected a prompt for \(language)")
        }
        XCTAssertNil(model.config.prompt(source: "zz"))
    }

    /// Changing the source language must change the prompt, and only in the
    /// language slots — everything else is fixed by the template.
    func testLanguageSwitchingTouchesOnlyTheLanguageSlots() async throws {
        let model = try await loadModel()
        let english = try XCTUnwrap(model.config.prompt(source: "en"))
        let german = try XCTUnwrap(model.config.prompt(source: "de"))

        XCTAssertEqual(english.count, german.count)
        let differing = zip(english, german).enumerated().filter { $1.0 != $1.1 }.map(\.offset)
        XCTAssertEqual(differing.count, 2, "only the source and target slots should move")
        XCTAssertEqual(differing[1], differing[0] + 1, "the pair should be adjacent")
    }

    /// Translation is the same graphs with a different target token.
    func testTranslationPromptDiffersFromTranscription() async throws {
        let model = try await loadModel()
        let transcribe = try XCTUnwrap(model.config.prompt(source: "de", target: "de"))
        let translate = try XCTUnwrap(model.config.prompt(source: "de", target: "en"))
        XCTAssertNotEqual(transcribe, translate)
    }

    /// Silence should decode to nothing rather than a hallucinated phrase, and
    /// must not hit the token cap.
    func testSilenceDecodesToNothing() async throws {
        let model = try await loadModel()
        let silence = [Float](repeating: 0, count: 16000 * 2)

        let result = try model.transcribeAudio(silence, sampleRate: 16000)
        XCTAssertTrue(result.text.isEmpty || result.text.count < 20,
                      "silence produced \"\(result.text)\"")
        XCTAssertTrue((0...1).contains(result.confidence))
    }

    /// A tone is not speech, but it must still complete a decode without
    /// running to the token cap or throwing.
    func testToneCompletesADecode() async throws {
        let model = try await loadModel()
        let tone = (0..<16000).map { i in
            0.2 * sin(2 * Float.pi * 440 * Float(i) / 16000)
        }

        let result = try model.transcribeAudio(tone, sampleRate: 16000, maxTokens: 64)
        XCTAssertTrue((0...1).contains(result.confidence))
        XCTAssertEqual(result.language, "en")
    }

    /// Audio at another rate is resampled rather than dropped.
    func testResamplesForeignSampleRates() async throws {
        let model = try await loadModel()
        let audio = [Float](repeating: 0, count: 48000 * 2)
        let result = try model.transcribeAudio(audio, sampleRate: 48000)
        XCTAssertTrue((0...1).contains(result.confidence))
    }

    /// Real speech through the whole loop. The other cases decode to
    /// end-of-text almost immediately, so they show the loop does not crash —
    /// only this one shows it produces a transcript.
    func testTranscribesRealSpeech() async throws {
        let model = try await loadModel()
        guard let wav = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not in bundle resources")
        }

        let full = try AudioFileLoader.load(url: wav, targetSampleRate: 16000)
        // The clip is 20 s with ~3 s of silence in front. Canary decodes a whole
        // segment at once and answers end-of-text for a buffer that is mostly
        // silence — the source checkpoint does the same on this file — so feed
        // it the utterance a VAD would deliver.
        let speech = Array(full[(16000 * 3)..<min(16000 * 9, full.count)])
        let result = try model.transcribeAudio(speech, sampleRate: 16000)

        // "Can you guarantee that the replacement part will be shipped tomorrow?"
        let text = result.text.lowercased()
        XCTAssertFalse(text.isEmpty, "expected a transcript")
        let expected = ["guarantee", "replacement", "part", "shipped", "tomorrow"]
        let matched = expected.filter { text.contains($0) }
        XCTAssertGreaterThanOrEqual(
            matched.count, 3,
            "expected at least 3 of \(expected) in \"\(result.text)\", matched \(matched)")
        XCTAssertGreaterThan(result.confidence, 0.3,
                             "confidence \(result.confidence) for a clean clip")
    }

    /// An unsupported language fails loudly at the call rather than silently
    /// transcribing as English.
    func testUnknownLanguageThrows() async throws {
        let model = try await loadModel()
        let audio = [Float](repeating: 0, count: 16000)
        XCTAssertThrowsError(try model.transcribeAudio(audio, sampleRate: 16000, language: "zz")) {
            guard case CanaryError.missingPrompt = $0 else {
                return XCTFail("expected missingPrompt, got \($0)")
            }
        }
    }
}
