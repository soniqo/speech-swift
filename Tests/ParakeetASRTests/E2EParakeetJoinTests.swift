import AudioCommon
import XCTest
@testable import ParakeetASR

final class E2EParakeetJoinTests: XCTestCase {
    func testQuietFrenchKeepsThePhraseBeforeALateSharedWord() async throws {
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(quietFixture("join_late_anchor"), sampleRate: 16000).lowercased()
        XCTAssertTrue(text.contains("des milliers de personnes sont mortes au combat"), text)
        XCTAssertFalse(text.contains("the million person"), text)
    }

    func testQuietFrenchKeepsTheOriginalCrusadesPhrase() async throws {
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(quietFixture("join_earlier_anchor"), sampleRate: 16000).lowercased()
        XCTAssertTrue(text.contains("europe occidentale"), text)
        XCTAssertTrue(text.contains("suite aux croisades"), text)
        XCTAssertFalse(text.contains("sudocroisade"), text)
    }

    func testQuietFrenchContinuationDoesNotDuplicateTheOverlap() async throws {
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let audio = try quietFixture("join_french_continuation")
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        XCTAssertTrue(text.lowercased().contains("vivent et travaillent"), text)
        XCTAssertFalse(text.lowercased().contains("person who visit"), text)
        XCTAssertEqual(text.lowercased().components(separatedBy: "personnes").count - 1, 1, text)
        let words = try XCTUnwrap(model.lastWordConfidences)
        XCTAssertEqual(words.map(\.word).joined(separator: " "), text)
        var previousStart = 0.0
        for word in words {
            let start = try XCTUnwrap(word.startTime), end = try XCTUnwrap(word.endTime)
            XCTAssertGreaterThanOrEqual(start, previousStart)
            XCTAssertGreaterThanOrEqual(end, start)
            XCTAssertLessThanOrEqual(end, Double(audio.count) / 16000)
            XCTAssertTrue(word.confidence.isFinite)
            XCTAssertGreaterThanOrEqual(word.confidence, 0)
            XCTAssertLessThanOrEqual(word.confidence, 1)
            previousStart = start
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
