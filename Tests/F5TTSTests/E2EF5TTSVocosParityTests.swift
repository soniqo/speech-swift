import AudioCommon
@testable import F5TTS
import Foundation
import MLX
import MLXCommon
@testable import Qwen3ASR
import XCTest

final class E2EF5TTSVocosParityTests: XCTestCase {
    func testDecodesUpstreamMelAndOptionalASRRoundTrip() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["F5TTS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set F5TTS_E2E_BUNDLE to run F5-TTS Vocos parity")
        }
        guard let melPath = env["F5TTS_VOCOS_MEL_F32"], !melPath.isEmpty else {
            throw XCTSkip("Set F5TTS_VOCOS_MEL_F32 to run F5-TTS Vocos parity")
        }
        guard let frameText = env["F5TTS_VOCOS_MEL_FRAMES"],
              let frames = Int(frameText), frames > 0 else {
            throw XCTSkip("Set F5TTS_VOCOS_MEL_FRAMES to run F5-TTS Vocos parity")
        }

        let bundleURL = URL(fileURLWithPath: bundlePath)
        let info = try F5TTSBundleLoader.load(from: bundleURL)
        let weights = try CommonWeightLoader.loadSafetensors(
            url: bundleURL.appendingPathComponent(info.config.files.vocoder))
        try F5TTSVocos.validate(weights)
        let vocoder = F5TTSVocos(weights: weights)

        let mel = try loadFloat32Array(path: melPath, expectedCount: 100 * frames)
        let started = Date()
        let waveform = vocoder.decode(mel: MLXArray(mel, [100, frames]))
            .asType(.float32)
        eval(waveform)
        let samples = waveform.asArray(Float.self)
        let elapsed = Date().timeIntervalSince(started)
        assertUsableWaveform(samples, sampleRate: info.sampleRate)
        print(String(format: "[F5-TTS Vocos] duration %.2fs elapsed %.2fs RTF %.3f",
                     Double(samples.count) / Double(info.sampleRate),
                     elapsed,
                     elapsed / max(Double(samples.count) / Double(info.sampleRate), 0.001)))

        if let wavPath = env["F5TTS_VOCOS_WAV"], !wavPath.isEmpty {
            try WAVWriter.write(samples: samples, sampleRate: info.sampleRate, to: URL(fileURLWithPath: wavPath))
            print("[F5-TTS Vocos] saved \(wavPath)")
        }

        guard env["F5TTS_ASR_E2E"] == "1" else { return }
        let target = env["F5TTS_TARGET_TEXT"] ?? "This is a short F five TTS voice cloning test running locally on Apple Silicon."
        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(audio: samples, sampleRate: info.sampleRate, language: "english")
        print("[F5-TTS Vocos] ASR: \(transcription)")
        XCTAssertGreaterThan(
            lexicalOverlap(expected: target, actual: transcription),
            0.55,
            "ASR should recover the generated English content from upstream mel decoded by Swift Vocos")
    }

    private func loadFloat32Array(path: String, expectedCount: Int) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        XCTAssertEqual(data.count, expectedCount * MemoryLayout<Float>.stride)
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }

    private func assertUsableWaveform(_ samples: [Float], sampleRate: Int) {
        XCTAssertGreaterThan(samples.count, sampleRate / 2)
        XCTAssertLessThan(samples.count, sampleRate * 30)
        XCTAssertFalse(samples.contains { !$0.isFinite })
        let peak = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 0.001)
        XCTAssertLessThanOrEqual(peak, 1.2)
    }

    private func lexicalOverlap(expected: String, actual: String) -> Double {
        let expectedWords = Set(words(expected))
        let actualWords = Set(words(actual))
        guard !expectedWords.isEmpty else { return 0 }
        return Double(expectedWords.intersection(actualWords).count) / Double(expectedWords.count)
    }

    private func words(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }
}
