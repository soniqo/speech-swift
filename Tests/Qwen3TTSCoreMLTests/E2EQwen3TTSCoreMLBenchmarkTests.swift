import XCTest
@testable import Qwen3TTSCoreML
@testable import Qwen3ASR
import AudioCommon
import Foundation

#if canImport(CoreML)
/// Per-stage throughput benchmark for Qwen3-TTS CoreML.
///
/// Reports wall time, RTFx, and per-stage cost (prompt build, CD prefill,
/// CD decode, MCD predict, embedder sum loop, SpeechDecoder) so we can
/// target optimization at the actual bottleneck on the current routing.
///
/// Bundle source:
///   - QWEN3TTS_BENCH_BUNDLE=<path>   load .mlmodelc bundle from path
///   - default                        download aufklarer/Qwen3-TTS-CoreML
///
/// Run:
///   QWEN3TTS_BENCH_BUNDLE=/tmp/qwen3tts-ane-export-full \
///     swift test --filter Qwen3TTSCoreMLTests.E2EQwen3TTSCoreMLBenchmarkTests
final class E2EQwen3TTSCoreMLBenchmarkTests: XCTestCase {

    private static var _model: Qwen3TTSCoreMLModel?

    override class func setUp() {
        super.setUp()
        // Force per-stage printout in synthesize()
        setenv("QWEN3TTS_BENCH", "1", 1)
    }

    private func loadModel() async throws -> Qwen3TTSCoreMLModel {
        if let m = Self._model { return m }
        let bundlePath = ProcessInfo.processInfo.environment["QWEN3TTS_BENCH_BUNDLE"]
        let m: Qwen3TTSCoreMLModel
        do {
            m = try await Qwen3TTSCoreMLModel.fromPretrained(localPath: bundlePath)
        } catch {
            throw XCTSkip("Model not available (bundle=\(bundlePath ?? "default")): \(error)")
        }
        Self._model = m
        return m
    }

    /// Short prompt — exercises prefill-dominated path
    func testBenchmarkShort() async throws {
        let model = try await loadModel()
        // Warm caches: ANE/CoreML first-call latency is unrepresentative
        _ = try model.synthesize(text: "Warm up.", language: "english", maxTokens: 25)

        let audio = try model.synthesize(
            text: "Hello world.",
            language: "english",
            temperature: 0.0, topK: 1,        // deterministic for run-to-run comparability
            maxTokens: 50)
        XCTAssertGreaterThan(audio.count, 0)
    }

    /// Medium prompt — exercises both prefill and decode
    func testBenchmarkMedium() async throws {
        let model = try await loadModel()
        _ = try model.synthesize(text: "Warm up.", language: "english", maxTokens: 25)

        let audio = try model.synthesize(
            text: "The quick brown fox jumps over the lazy dog.",
            language: "english",
            temperature: 0.0, topK: 1,
            maxTokens: 125)
        XCTAssertGreaterThan(audio.count, 0)
    }

    /// Long prompt — saturates decode (capped at maxTokens=125 = 10s audio)
    func testBenchmarkLong() async throws {
        let model = try await loadModel()
        _ = try model.synthesize(text: "Warm up.", language: "english", maxTokens: 25)

        let audio = try model.synthesize(
            text: "Coreml on the Apple Neural Engine is fast for vocoders but the autoregressive decoder loop pays per call dispatch overhead.",
            language: "english",
            temperature: 0.0, topK: 1,
            maxTokens: 125)
        XCTAssertGreaterThan(audio.count, 0)
    }

    /// Roundtrip ASR check on the current bundle. Synthesizes a known prompt,
    /// transcribes it with Qwen3-ASR, and asserts ≥3 keyword matches. This is
    /// the gate that catches regressions where the bundle compiles + runs but
    /// produces noise (precision drift, broken routing, etc.).
    func testRoundtripIntelligibility() async throws {
        let model = try await loadModel()

        let asr: Qwen3ASRModel
        do {
            asr = try await Qwen3ASRModel.fromPretrained()
        } catch {
            throw XCTSkip("ASR model not available: \(error)")
        }

        let prompt = "The quick brown fox jumps over the lazy dog."
        let audio = try model.synthesize(
            text: prompt, language: "english",
            temperature: 0.0, topK: 1, maxTokens: 125)
        XCTAssertGreaterThan(audio.count, 0)

        let transcript = asr.transcribe(audio: audio, sampleRate: 24000)
        print("\n[ROUNDTRIP] input:  \"\(prompt)\"")
        print("[ROUNDTRIP] output: \"\(transcript)\"")

        let keywords = ["quick", "brown", "fox", "jumps", "lazy", "dog"]
        let matched = keywords.filter { transcript.lowercased().contains($0) }
        print("[ROUNDTRIP] matched \(matched.count)/\(keywords.count): \(matched)")

        XCTAssertGreaterThanOrEqual(matched.count, 3,
            "ANE bundle should produce intelligible speech. Got: \"\(transcript)\"")
    }

    /// Save the medium benchmark output to a WAV for manual playback when the
    /// MLX-backed ASR isn't available (no Metal toolchain). Set
    /// QWEN3TTS_BENCH_WAV=/path/to/out.wav to enable.
    func testSaveBenchmarkAudio() async throws {
        guard let wavPath = ProcessInfo.processInfo.environment["QWEN3TTS_BENCH_WAV"] else {
            throw XCTSkip("set QWEN3TTS_BENCH_WAV=/path/to/out.wav to save")
        }
        let model = try await loadModel()
        let text = ProcessInfo.processInfo.environment["QWEN3TTS_BENCH_TEXT"]
            ?? "The quick brown fox jumps over the lazy dog."
        let audio = try model.synthesize(
            text: text,
            language: "english",
            temperature: 0.0, topK: 1, maxTokens: 125)
        XCTAssertGreaterThan(audio.count, 0)
        try WAVWriter.write(samples: audio, sampleRate: 24000,
                            to: URL(fileURLWithPath: wavPath))
        print("[WAV] saved to \(wavPath) (\(audio.count) samples, \(String(format: "%.2f", Double(audio.count)/24000))s)")
    }
}
#endif
