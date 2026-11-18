import AudioCommon
import XCTest
@testable import ParakeetASR

final class E2EParakeetCalibrationTests: XCTestCase {
    func testQuietSpeechRetainsTheMiddleWindow() async throws {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "quiet_window_calibration", withExtension: "wav"))
        let samples = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let peak = try XCTUnwrap(samples.map { abs($0) }.max())
        XCTAssertGreaterThan(peak, 0)
        let audio = samples.map { $0 * 0.001 / peak }
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(audio, sampleRate: 16000).lowercased()
        for phrase in ["trip to the state", "website gets remodeled", "online"] {
            XCTAssertTrue(text.contains(phrase), "Quiet speech lost '\(phrase)': \(text)")
        }
        XCTAssertTrue(text.contains("old versions") || text.contains("older versions"),
            "Quiet speech lost the versions passage: \(text)")
    }

    func testQuietNoiseAndSilenceStayEmpty() async throws {
        var state: UInt32 = 559
        let noise: [Float] = (0..<(2 * 16000)).map { _ in
            state = state &* 1664525 &+ 1013904223
            return (Float(state) / 2147483648 - 1) * 0.001
        }
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        for audio in [[Float](repeating: 0, count: 5 * 16000), noise] {
            let text = try model.transcribeAudio(audio, sampleRate: 16000)
            XCTAssertTrue(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Silence or quiet noise produced speech: \(text)")
        }
    }
}
