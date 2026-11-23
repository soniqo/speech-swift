import AudioCommon
import XCTest
@testable import ParakeetASR

final class E2EParakeetUltraTests: XCTestCase {
    func testUltraTranscribesAcrossThirtySecondWindow() async throws {
        let directory = ProcessInfo.processInfo.environment["PARAKEET_ULTRA_MODEL_DIR"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
        let model = try await ParakeetASRModel.fromPretrained(
            modelId: ParakeetASRModel.ultraModelId, cacheDir: directory, offlineMode: directory != nil)
        XCTAssertEqual(model.supportedMelLengths, [3000])

        let englishURL = try XCTUnwrap(Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        let germanURL = try XCTUnwrap(Bundle.module.url(forResource: "test_audio_german", withExtension: "wav"))
        let sampleRate = 16000
        let english = try AudioFileLoader.load(url: englishURL, targetSampleRate: sampleRate)
        let german = try AudioFileLoader.load(url: germanURL, targetSampleRate: sampleRate)
        let silence = try model.transcribeAudio([Float](repeating: 0, count: sampleRate), sampleRate: sampleRate)
        XCTAssertTrue(silence.isEmpty, "Ultra decoded silence as '\(silence)'")

        let boundary = 30 * sampleRate
        let padding = [Float](repeating: 0, count: max(0, boundary - english.count))
        let audio = english + padding + german
        let result = model.transcribeWithLanguage(audio: audio, sampleRate: sampleRate, language: "en,de")
        XCTAssertTrue(result.text.lowercased().contains("guarantee"), result.text)
        XCTAssertTrue(result.text.lowercased().contains("guten tag"), result.text)
        XCTAssertGreaterThan(result.confidence ?? 0, 0)
        let words = try XCTUnwrap(result.words)
        XCTAssertTrue(words.contains { ($0.startTime ?? 0) >= 30 }, "Second window timestamps must be absolute")
    }
}
