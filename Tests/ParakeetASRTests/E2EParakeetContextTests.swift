import AudioCommon
import XCTest
@testable import ParakeetASR

final class E2EParakeetContextTests: XCTestCase {
    func testEmptyPrimaryWindowRetriesWithBroaderNormalization() async throws {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "context_empty_window", withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(audio, sampleRate: 16000).lowercased()
        XCTAssertTrue(text.contains("invaders instead of fighting them"), text)
    }

    func testQuietFrenchKeepsTheMiddleAndFinalPhrases() async throws {
        let audio = try quietFixture("context_french")
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(audio, sampleRate: 16000).lowercased()
        for word in ["semblables", "comportement", "mortel", "proie"] {
            XCTAssertTrue(text.contains(word), "Quiet context lost '\(word)': \(text)")
        }
    }

    func testQuietEnglishOmitsTheInventedContinuationAndResetsBetweenCalls() async throws {
        let audio = try quietFixture("context_english")
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let first = try model.transcribeAudio(audio, sampleRate: 16000)
        XCTAssertTrue(first.lowercased().contains("elsewhere can cause cell death or cancer"), first)
        XCTAssertFalse(first.lowercased().contains("not a very important thing to use"), first)
        let words = try XCTUnwrap(model.lastWordConfidences)
        var previousStart = 0.0
        for word in words {
            let start = try XCTUnwrap(word.startTime)
            let end = try XCTUnwrap(word.endTime)
            XCTAssertGreaterThanOrEqual(start, previousStart)
            XCTAssertGreaterThanOrEqual(end, start)
            XCTAssertLessThanOrEqual(end, Double(audio.count) / 16000)
            previousStart = start
        }
        let confidence = model.lastConfidence
        let second = try model.transcribeAudio(audio, sampleRate: 16000)
        XCTAssertEqual(second, first)
        XCTAssertEqual(model.lastConfidence, confidence, accuracy: 1e-6)
        XCTAssertEqual(model.lastWordConfidences?.map(\.startTime), words.map(\.startTime))
    }

    func testOriginalWindowPositionsStillRecoverTheQuietPassage() async throws {
        let audio = try quietFixture("context_phase")
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(audio, sampleRate: 16000).lowercased()
        XCTAssertTrue(text.contains("glittering bar chart"), text)
        XCTAssertTrue(text.contains("made apparent"), text)
    }

    func testLongNoiseAndSilenceStayEmptyAcrossContextWindows() async throws {
        var state: UInt32 = 559
        let count = 12 * 16000
        let noise: [Float] = (0..<count).map { _ in
            state = state &* 1664525 &+ 1013904223
            return (Float(state) / 2147483648 - 1) * 0.001
        }
        let hum: [Float] = (0..<count).map { 0.001 * sin(2 * .pi * 60 * Float($0) / 16000) }
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        for audio in [[Float](repeating: 0, count: count), noise, hum] {
            let text = try model.transcribeAudio(audio, sampleRate: 16000)
            XCTAssertTrue(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "Nonspeech produced words across context windows: \(text)")
        }
    }

    private func quietFixture(_ name: String) throws -> [Float] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "wav"))
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: 16000)
        let peak = try XCTUnwrap(audio.map { abs($0) }.max())
        XCTAssertGreaterThan(peak, 0)
        return audio.map { $0 * 0.001 / peak }
    }
}
