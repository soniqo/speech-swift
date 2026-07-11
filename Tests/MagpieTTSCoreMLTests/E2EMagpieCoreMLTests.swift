import XCTest
import AudioCommon
import Qwen3ASR
@testable import MagpieTTSCoreML

/// End-to-end tests against the soniqo CoreML Magpie bundle. The smoke
/// test asserts the synthesis pipeline produces non-trivial audio; the
/// ASR transcription test pipes a captured Magpie waveform through
/// Qwen3-ASR and validates the prompt's content words come back out.
/// Skipped from PR CI via the `--skip E2E` filter; runs nightly.
final class E2EMagpieCoreMLTests: XCTestCase {

    func testLoadAndSynthesizeEnglish() async throws {
        let model: MagpieTTSCoreML
        do {
            model = try await MagpieTTSCoreML.fromPretrained(
                progressHandler: { progress in
                    if Int(progress * 100) % 25 == 0 {
                        print(String(format: "  download: %.0f%%", progress * 100))
                    }
                })
        } catch {
            throw XCTSkip("model bundle download failed: \(error)")
        }

        let audio = try model.synthesize(
            text: "Hello world.",
            speaker: .aria,
            language: .english,
            params: MagpieCoreMLParams(
                temperature: 0,  // greedy for determinism in smoke test
                maxSteps: 100,
                seed: 42))

        XCTAssertFalse(audio.isEmpty, "expected non-empty audio output")
        // 100-step cap → up to ~4.6 s @ 22.05 kHz. We're greedy so the
        // model usually terminates much earlier; any nontrivial buffer
        // means the pipeline wired up end-to-end.
        XCTAssertGreaterThan(audio.count, 1000, "audio too short — likely empty frames")
        let peak = audio.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 1e-3, "audio peak \(peak) too low — model likely produced zeros")
        print("E2E OK: \(audio.count) samples, peak=\(peak), duration=\(Double(audio.count)/22050.0)s")
    }

    /// ASR check against a captured Magpie waveform (the prompt "Hello
    /// world from Magpie text to speech."). Uses a committed WAV fixture
    /// instead of synthesising at test time because greedy Magpie isn't
    /// numerically deterministic across hardware — the same prompt with
    /// `temperature=0 seed=0` produces ~280 ms-different output on the
    /// macos-15 hosted runner vs local macOS 16. The fixture was captured
    /// from a known-good local run and is what every CI agent should be
    /// able to transcribe correctly via Qwen3-ASR.
    ///
    /// If any word goes missing, suspect the Qwen3-ASR CoreML decoder
    /// path, FSQ inverse, codec windowing, or audio_emb averaging.
    func testAsrTranscribeCapturedMagpieAudio() async throws {
#if canImport(CoreML)
        guard let audioURL = Bundle.module.url(
            forResource: "magpie-hello-world", withExtension: "wav"
        ) else {
            throw XCTSkip("magpie-hello-world.wav not found in test resources")
        }

        let audio = try AudioFileLoader.load(
            url: audioURL,
            targetSampleRate: Int(MagpieTTSCoreML.sampleRate))
        XCTAssertGreaterThan(audio.count, Int(MagpieTTSCoreML.sampleRate) / 2,
                             "fixture audio <0.5 s — file may be corrupt")

        let asr = try await CoreMLASRModel.fromPretrained()
        let raw = asr.transcribe(audio: audio,
                                  sampleRate: Int(MagpieTTSCoreML.sampleRate),
                                  language: "english")
        let normalised = raw
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
        print("[MAGPIE-COREML-ASR] raw=\"\(raw)\"  normalised=\"\(normalised)\"")

        for word in ["hello", "world", "magpie", "text", "speech"] {
            XCTAssertTrue(normalised.contains(word),
                          "ASR transcription missing '\(word)'. Raw=\"\(raw)\"")
        }
#else
        throw XCTSkip("Qwen3-ASR requires CoreML")
#endif
    }
}
