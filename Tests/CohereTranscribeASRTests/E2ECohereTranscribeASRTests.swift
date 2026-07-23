import AudioCommon
import XCTest
@testable import CohereTranscribeASR

final class E2ECohereTranscribeASRTests: XCTestCase {
    func testLocalExportTranscribesKnownSpeech() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["COHERE_MLX_MODEL_PATH"],
              !modelPath.isEmpty else {
            throw XCTSkip("Set COHERE_MLX_MODEL_PATH to an exported FP16, INT5, or INT8 bundle")
        }
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Qwen3ASRTests/Resources/test_audio.wav")
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("Shared ASR speech fixture is unavailable")
        }

        let model = try await CohereTranscribeModel.load(modelPath, offlineMode: true)
        let (audio, sampleRate) = try AudioFileLoader.loadWAV(url: fixture)
        let text = model.transcribe(audio: audio, sampleRate: sampleRate, language: "en")
            .lowercased()

        XCTAssertFalse(text.isEmpty)
        let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
        XCTAssertGreaterThanOrEqual(expected.filter(text.contains).count, 3, "Got: \(text)")
    }
}
