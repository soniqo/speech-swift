import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Exercises the high-level `OmniVoiceTTSModel` facade the Studio sidecar drives:
/// load a local bundle, then synthesize cloned speech from a reference WAV + text.
/// Writes the waveform for an external ASR check. The optional bundle test below
/// still uses int8 so the quantization-from-config load path remains covered.
final class OmniVoiceModelFacadeTests: XCTestCase {
    func testDefaultModelIdUsesFp16Bundle() {
        XCTAssertEqual(OmniVoiceTTSModel.defaultModelId, OmniVoiceConfig.defaultModelId)
        XCTAssertEqual(OmniVoiceTTSModel.defaultModelId, "aufklarer/OmniVoice-MLX-fp16")
    }

    func testFromBundleAndGenerate() async throws {
        let bundleDir = "/tmp/omnivoice-int8"
        let refWav = "/tmp/cbx_swift_en.wav"
        let dir = "/tmp/omnivoice_golden"
        for p in ["\(bundleDir)/model.safetensors", "\(bundleDir)/audio_tokenizer/model.safetensors",
            "\(bundleDir)/tokenizer.json", "\(bundleDir)/config.json", refWav]
        where !FileManager.default.fileExists(atPath: p) { throw XCTSkip("missing \(p)") }

        let tts = try await OmniVoiceTTSModel.fromBundle(URL(fileURLWithPath: bundleDir))

        // Estimate-driven duration (the real default path) — print what it chose.
        let estimated = tts.builder.estimateTargetLen(
            text: "The quick brown fox jumps over the lazy dog.", lang: "en", frameRate: tts.frameRate)
        print("[OmniVoice] facade: estimated targetLen = \(estimated) frames (\(Double(estimated) / tts.frameRate)s)")

        let samples = try tts.generate(
            text: "The quick brown fox jumps over the lazy dog.",
            referenceAudio: URL(fileURLWithPath: refWav),
            referenceText: "Hello there, this is a cloned voice.",
            language: "en",
            duration: 2.52,  // pin to a known length for a deterministic ASR gate
            numSteps: 16)

        XCTAssertGreaterThan(samples.count, 24000)  // > 1s
        var sumSq = 0.0
        for s in samples { sumSq += Double(s) * Double(s) }
        let rms = (sumSq / Double(samples.count)).squareRoot()
        print("[OmniVoice] facade: \(samples.count) samples (\(Double(samples.count) / Double(tts.sampleRate))s) rms=\(rms)")
        XCTAssertGreaterThan(rms, 0.01, "facade produced (near) silence")

        let out = "\(dir)/swift_facade_e2e.f32"
        samples.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: out)) }
        print("[OmniVoice] facade: wrote \(out)")
    }
}
