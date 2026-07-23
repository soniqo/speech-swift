import AudioCommon
import XCTest
@testable import VoxtralASR

final class E2EVoxtralASRTests: XCTestCase {
    func testLocalExportTranscribesKnownSpeech() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["VOXTRAL_MLX_MODEL_PATH"],
              !modelPath.isEmpty else {
            throw XCTSkip("Set VOXTRAL_MLX_MODEL_PATH to an exported FP16, INT5, or INT8 bundle")
        }
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("Shared ASR speech fixture is unavailable")
        }

        let model = try await VoxtralModel.load(modelPath, offlineMode: true)
        let (audio, sampleRate) = try AudioFileLoader.loadWAV(url: fixture)
        let text = model.transcribe(audio: audio, sampleRate: sampleRate, language: "en")
            .lowercased()

        XCTAssertFalse(text.isEmpty)
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        XCTAssertGreaterThanOrEqual(expected.filter(text.contains).count, 3, "Got: \(text)")
    }
}
