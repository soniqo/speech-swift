import AudioCommon
@testable import BenchmarkSupport
import SpeechVAD
import XCTest

final class DiarizationSpeakerCountAggregationTests: XCTestCase {
    func testAggregatesDurationWeightedDERByReferenceSpeakerCount() throws {
        var accumulator = DiarizationSpeakerCountAccumulator()

        let longTwoSpeakerReference = [
            segment(0, 3, 0),
            segment(3, 6, 1),
        ]
        accumulator.add(
            referenceSpeakerCount: 2,
            predictedSpeakerCount: 2,
            score: computeDERWithOptimalMapping(
                reference: longTwoSpeakerReference,
                hypothesis: longTwoSpeakerReference,
                collar: 0,
                resolution: 0.1))

        let shortTwoSpeakerReference = [
            segment(0, 1, 0),
            segment(1, 2, 1),
        ]
        accumulator.add(
            referenceSpeakerCount: 2,
            predictedSpeakerCount: 0,
            score: computeDERWithOptimalMapping(
                reference: shortTwoSpeakerReference,
                hypothesis: [],
                collar: 0,
                resolution: 0.1))

        let threeSpeakerReference = [
            segment(0, 1, 0),
            segment(1, 2, 1),
            segment(2, 3, 2),
        ]
        accumulator.add(
            referenceSpeakerCount: 3,
            predictedSpeakerCount: 3,
            score: computeDERWithOptimalMapping(
                reference: threeSpeakerReference,
                hypothesis: threeSpeakerReference,
                collar: 0,
                resolution: 0.1))

        let fourSpeakerReference = [
            segment(0, 1, 0),
            segment(1, 2, 1),
            segment(2, 3, 2),
            segment(3, 4, 3),
        ]
        accumulator.add(
            referenceSpeakerCount: 4,
            predictedSpeakerCount: 0,
            score: computeDERWithOptimalMapping(
                reference: fourSpeakerReference,
                hypothesis: [],
                collar: 0,
                resolution: 0.1))

        let results = accumulator.results
        XCTAssertEqual(results.map(\.referenceSpeakerCount), [2, 3, 4])

        let twoSpeakers = try XCTUnwrap(results.first { $0.referenceSpeakerCount == 2 })
        XCTAssertEqual(twoSpeakers.files, 2)
        XCTAssertEqual(twoSpeakers.totalSpeechSeconds, 8, accuracy: 0.001)
        XCTAssertEqual(twoSpeakers.derPercent, 25, accuracy: 0.001)
        XCTAssertEqual(twoSpeakers.missPercent, 25, accuracy: 0.001)
        XCTAssertEqual(twoSpeakers.falseAlarmPercent, 0, accuracy: 0.001)
        XCTAssertEqual(twoSpeakers.speakerErrorPercent, 0, accuracy: 0.001)
        XCTAssertEqual(twoSpeakers.speakerCountAccuracyPercent, 50, accuracy: 0.001)

        let threeSpeakers = try XCTUnwrap(results.first { $0.referenceSpeakerCount == 3 })
        XCTAssertEqual(threeSpeakers.derPercent, 0, accuracy: 0.001)
        XCTAssertEqual(threeSpeakers.speakerCountAccuracyPercent, 100, accuracy: 0.001)

        let fourSpeakers = try XCTUnwrap(results.first { $0.referenceSpeakerCount == 4 })
        XCTAssertEqual(fourSpeakers.derPercent, 100, accuracy: 0.001)
        XCTAssertEqual(fourSpeakers.speakerCountAccuracyPercent, 0, accuracy: 0.001)
    }

    func testIgnoresEntriesWithoutAReferenceSpeaker() {
        var accumulator = DiarizationSpeakerCountAccumulator()
        let emptyScore = computeDERWithOptimalMapping(
            reference: [], hypothesis: [], collar: 0, resolution: 0.1)

        accumulator.add(
            referenceSpeakerCount: 0,
            predictedSpeakerCount: 0,
            score: emptyScore)

        XCTAssertTrue(accumulator.results.isEmpty)
    }

    private func segment(
        _ start: Float,
        _ end: Float,
        _ speaker: Int
    ) -> DiarizedSegment {
        DiarizedSegment(startTime: start, endTime: end, speakerId: speaker)
    }
}
