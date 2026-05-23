import XCTest
import Qwen3ASR
@testable import MagpieTTSCoreML

/// End-to-end tests against the soniqo CoreML Magpie bundle. The smoke
/// test asserts the pipeline produces non-trivial audio; the ASR
/// round-trip routes the synthesised waveform through Qwen3-ASR and
/// validates the prompt's content words actually come back out. Skipped
/// from CI via the `--skip E2E` filter.
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

    /// ASR round-trip: synthesize → Qwen3-ASR → must contain every
    /// content word from the prompt. This is the real quality validator
    /// for the hybrid CoreML+MLX pipeline (CoreML drives the big models;
    /// MLX runs the 1-layer LocalTransformer for per-frame sampling).
    func testAsrRoundTripEnglish() async throws {
#if canImport(CoreML)
        let model: MagpieTTSCoreML
        do {
            model = try await MagpieTTSCoreML.fromPretrained()
        } catch {
            throw XCTSkip("model bundle download failed: \(error)")
        }
        let asr = try await CoreMLASRModel.fromPretrained()

        let prompt = "Hello world from Magpie text to speech."
        let audio = try model.synthesize(
            text: prompt, speaker: .aria, language: .english,
            params: MagpieCoreMLParams(
                temperature: 0, topK: 1, maxSteps: 300, seed: 0))
        XCTAssertGreaterThan(audio.count, MagpieTTSCoreML.sampleRate / 2,
                             "TTS produced <0.5 s of audio — likely silent")

        let raw = asr.transcribe(audio: audio,
                                  sampleRate: MagpieTTSCoreML.sampleRate,
                                  language: "english")
        let normalised = raw
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
        print("[MAGPIE-COREML-ASR] raw=\"\(raw)\"  normalised=\"\(normalised)\"")

        // Every content word from the prompt must appear in the ASR.
        // If any single word goes missing, suspect FSQ inverse, codec
        // windowing, or audio_emb averaging — the MLX backend has its
        // own equivalent test that catches G2P regressions.
        for word in ["hello", "world", "magpie", "text", "speech"] {
            XCTAssertTrue(normalised.contains(word),
                          "ASR transcription missing '\(word)'. Raw=\"\(raw)\"")
        }
#else
        throw XCTSkip("Qwen3-ASR requires CoreML")
#endif
    }
}
