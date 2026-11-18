import AudioCommon
import XCTest
@testable import ParakeetASR

final class ParakeetWordDurationTests: XCTestCase {
    private let vocabulary = ParakeetVocabulary(idToToken: [
        0: "\u{2581}hel", 1: "lo", 2: "\u{2581}world", 3: "?",
    ])

    func testWordEndsUsePredictedTokenDurationsIncludingZero() {
        let words = vocabulary.decodeWordsWithTokenTimes(
            [0, 1, 2], logProbs: [-0.1, -0.1, -0.1],
            tokenStartTimes: [0.24, 0.32, 0.80], tokenEndTimes: [0.32, 0.32, 0.96])
        XCTAssertEqual(words.map(\.word), ["hello", "world"])
        XCTAssertEqual(words[0].startTime, 0.24)
        XCTAssertEqual(words[0].endTime, 0.32)
        XCTAssertEqual(words[1].startTime, 0.80)
        XCTAssertEqual(words[1].endTime, 0.96)
    }

    func testWordDetailsKeepEveryRetainedSubwordProbability() {
        let words = vocabulary.decodeWordDetailsWithTokenTimes(
            [0, 1, 2, 3], logProbs: [-0.1, -0.2, -0.3, -0.4],
            tokenStartTimes: [0.24, 0.32, 0.80, 0.96], tokenEndTimes: [0.32, 0.32, 0.96, 1.04])
        XCTAssertEqual(words.map { $0.value.word }, ["hello", "world?"])
        XCTAssertEqual(words[0].tokenLogProbs, [-0.1, -0.2])
        XCTAssertEqual(words[1].tokenLogProbs, [-0.3, -0.4])
        XCTAssertEqual(words[0].value.confidence, exp(-0.15), accuracy: 1e-6)
        XCTAssertEqual(words[1].value.endTime, 0.96)
    }

    func testAttachedPunctuationKeepsTheLastTextTokenEnd() {
        let words = vocabulary.decodeWordsWithTokenTimes(
            [2, 3], logProbs: [-0.1, -0.1],
            tokenStartTimes: [7.24, 7.88], tokenEndTimes: [7.48, 7.96])
        XCTAssertEqual(words.map(\.word), ["world?"])
        XCTAssertEqual(words[0].endTime, 7.48)
    }

    func testMissingOrMismatchedEndsOmitTimes() {
        for ends: [Double]? in [nil, [], [0.24]] {
            let words = vocabulary.decodeWordsWithTokenTimes(
                [0, 1], logProbs: [-0.1, -0.1],
                tokenStartTimes: [0.16, 0.24], tokenEndTimes: ends)
            XCTAssertEqual(words.map(\.word), ["hello"])
            XCTAssertNil(words[0].startTime)
            XCTAssertNil(words[0].endTime)
        }
        let words = vocabulary.decodeWordsWithTokenTimes(
            [2], logProbs: [-0.1], tokenStartTimes: nil, tokenEndTimes: [0.24])
        XCTAssertNil(words[0].startTime)
        XCTAssertNil(words[0].endTime)
    }

    func testLegacyMethodReferenceAndFrameDurationRemainCompatible() {
        let decode = vocabulary.decodeWords
        let words = decode([2, 3], [-0.1, -0.1], [7.24, 7.88], 0.08)
        XCTAssertEqual(words.map(\.word), ["world?"])
        XCTAssertEqual(words[0].endTime ?? -1, 7.96, accuracy: 1e-9)
    }

    func testDurationEndsKeepTheAbsoluteWindowOffset() {
        let starts = [0, 3]
        let durations = [0, 2]
        let ends = ParakeetASRModel.absoluteTokenTimes(
            windowStartMelFrame: 500,
            tokenFrames: zip(starts, durations).map { $0.0 + $0.1 }, config: .default)
        XCTAssertEqual(ends[0], 5, accuracy: 1e-9)
        XCTAssertEqual(ends[1], 5.4, accuracy: 1e-9)
    }
}
