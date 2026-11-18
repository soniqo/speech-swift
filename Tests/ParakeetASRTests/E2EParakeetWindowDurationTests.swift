import AudioCommon
import XCTest
@testable import ParakeetASR

final class E2EParakeetWindowDurationTests: XCTestCase {
    func testQuietSecondWindowPreservesTheSentence() async throws {
        let samples = try fixture("test_audio")
        let sentence = Array(samples[(5 * 16000)..<(9 * 16000)])
        let peak = try XCTUnwrap(sentence.map { abs($0) }.max())
        XCTAssertGreaterThan(peak, 0)
        let loud = sentence.map { $0 * 0.3 / peak }
        let quiet = sentence.map { $0 * 0.001 / peak }
        let audio = loud + [Float](repeating: 0, count: 16000) + quiet
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        _ = try model.transcribeAudio(audio, sampleRate: 16000)
        let words = try XCTUnwrap(model.lastWordConfidences)
        let tail = words.filter { ($0.startTime ?? -1) >= 5 }.map(\.word).joined(separator: " ").lowercased()
        for keyword in ["guarantee", "replacement", "shipped"] {
            XCTAssertTrue(tail.contains(keyword), "Quiet second window lost '\(keyword)': \(tail)")
        }
        for word in words {
            let start = try XCTUnwrap(word.startTime)
            let end = try XCTUnwrap(word.endTime)
            XCTAssertTrue(start.isFinite && end.isFinite)
            XCTAssertGreaterThanOrEqual(end, start)
        }
    }

    func testDurationZeroPreservesWordsOnTheCurrentFrame() async throws {
        let audio = try fixture("tdt_duration_zero")
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        let text = try model.transcribeAudio(audio, sampleRate: 16000)
        XCTAssertTrue(text.lowercased().contains("you call on"),
            "Duration-zero decoding skipped words in the reference sentence: \(text)")
    }

    func testFullWindowKeepsTheFirstWordTime() async throws {
        let audio = try fixture("test_audio")
        let start = 4 * 16000
        let model = try await ParakeetASRModel.fromPretrained(modelId: ParakeetASRModel.iosModelId)
        func firstWord(sampleCount: Int) throws -> WordConfidence {
            _ = try model.transcribeAudio(Array(audio[start..<(start + sampleCount)]), sampleRate: 16000)
            return try XCTUnwrap(model.lastWordConfidences?.first)
        }
        let full = try firstWord(sampleCount: 80000)
        let shorter = try firstWord(sampleCount: 79999)
        XCTAssertEqual(full.word, shorter.word)
        XCTAssertEqual(try XCTUnwrap(full.startTime), try XCTUnwrap(shorter.startTime), accuracy: 0.02,
            "The extra STFT padding frame must not shift a full window's first word")
    }

    private func fixture(_ name: String) throws -> [Float] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "wav"))
        return try AudioFileLoader.load(url: url, targetSampleRate: 16000)
    }
}
