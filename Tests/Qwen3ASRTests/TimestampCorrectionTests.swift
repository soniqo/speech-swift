import XCTest
@testable import Qwen3ASR

/// Correction logic has no model dependency and must run in the hosted unit suite.
final class TimestampCorrectionTests: XCTestCase {
    func testEmptyAndSingleTimestampAreUnchanged() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([]), [])
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([7]), [7])
    }

    func testStrictlyIncreasingTimestampsAreUnchanged() {
        let input = [1, 3, 5, 7, 9, 11]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input), input)
    }

    func testNondecreasingTimestampSlotsAreUnchanged() {
        let input = [1, 2, 2, 3]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input), input)
    }

    func testRepeatedBoundariesPreserveWordEndTimes() {
        let input = [66, 68, 69, 69, 69, 77, 77, 79]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input), input)
    }

    func testAllEqualTimestampsAreUnchanged() {
        let input = [5, 5, 5, 5]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input), input)
    }

    func testSingleOutOfOrderTimestampUsesTheEarlierAnchor() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([1, 3, 2, 7, 9, 11]),
                       [1, 3, 3, 7, 9, 11])
    }

    func testDescendingTimestampsUseTheFirstEndpoint() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([10, 8, 6, 4, 2]),
                       [10, 10, 10, 10, 10])
    }

    func testEqualLengthAnchorChainsPreferEarlierIndices() {
        XCTAssertEqual(TimestampCorrection.longestNondecreasingSubsequencePositions([1, 3, 2, 4]),
                       [0, 1, 3])
        XCTAssertEqual(TimestampCorrection.longestNondecreasingSubsequencePositions([3, 1, 4, 1, 5, 9, 2, 6]),
                       [0, 2, 4, 5])
    }

    func testSmallGapsUseNearestAnchors() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([10, 99, 98, 20, 30]),
                       [10, 10, 20, 20, 30])
    }

    func testLargeGapsAreInterpolated() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([0, 99, 98, 97, 20, 30]),
                       [0, 5, 10, 15, 20, 30])
    }

    func testLeadingAndTrailingAnomaliesAreClamped() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([9, 1, 2, 3]), [1, 1, 2, 3])
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([1, 2, 3, 0]), [1, 2, 3, 3])
    }

    func testIssue546RawSlotsMatchQwenCorrection() {
        let raw = [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 80, 94, 88, 92, 92, 93, 93, 100, 93, 105, 98, 112]
        let expected = [66, 68, 69, 69, 69, 77, 77, 79, 85, 86, 86, 88, 88, 92, 92, 93, 93, 100, 100, 105, 105, 112]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(raw), expected)
    }

    func testOutOfRangePredictionsCannotDisplaceGenuineAnchors() {
        let raw = [10, 12, 3000, 3001, 3002, 3003, 3004, 3005, 14, 16]
        let corrected = TimestampCorrection.enforceMonotonicity(raw, maxIndex: 50)
        XCTAssertEqual(Array(corrected.suffix(2)), [14, 16])
        XCTAssertLessThanOrEqual(corrected.max() ?? 0, 50)

        // Both unbounded predictions and a plateau produced by clamping first
        // can win the anchor competition. Neither represents a genuine boundary.
        XCTAssertTrue(TimestampCorrection.enforceMonotonicity(raw).suffix(2).allSatisfy { $0 >= 3000 })
        let clamped = raw.map { min($0, 50) }
        XCTAssertEqual(Array(TimestampCorrection.enforceMonotonicity(clamped).suffix(2)), [50, 50])
    }

    func testBoundedCorrectionPreservesValidPlateaus() {
        let input = [0, 1, 1, 2]
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input, maxIndex: 2), input)
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([900, 10, 12, 900], maxIndex: 20),
                       [10, 10, 12, 12])
    }

    func testBoundedCorrectionHandlesNoValidAnchors() {
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([], maxIndex: 3), [])
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([9], maxIndex: 3), [3])
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([9, 10, 11], maxIndex: 3), [3, 3, 3])
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity([0, 9, 0], maxIndex: 0), [0, 0, 0])
    }

    func testLargeNondecreasingPlateauKeepsEveryAnchor() {
        let input = (0..<5000).flatMap { [$0, $0] }
        XCTAssertEqual(TimestampCorrection.longestNondecreasingSubsequencePositions(input), Array(input.indices))
        XCTAssertEqual(TimestampCorrection.enforceMonotonicity(input), input)
    }

    func testAnchorSearchMatchesReferenceForAllShortSequences() {
        var caseCount = 1
        for length in 0...8 {
            for code in 0..<caseCount {
                var remaining = code
                let input = (0..<length).map { _ -> Int in
                    defer { remaining /= 3 }
                    return remaining % 3
                }
                let anchors = TimestampCorrection.longestNondecreasingSubsequencePositions(input)
                XCTAssertEqual(anchors, referenceAnchorPositions(input), "Input: \(input)")
                let corrected = TimestampCorrection.enforceMonotonicity(input)
                XCTAssertEqual(corrected.count, input.count)
                XCTAssertEqual(corrected, corrected.sorted(), "Input: \(input)")
                XCTAssertEqual(TimestampCorrection.enforceMonotonicity(corrected), corrected)
                for anchor in anchors { XCTAssertEqual(corrected[anchor], input[anchor]) }

                let inRange = input.indices.filter { input[$0] <= 1 }
                let boundedAnchors = referenceAnchorPositions(inRange.map { input[$0] }).map { inRange[$0] }
                let bounded = TimestampCorrection.enforceMonotonicity(input, maxIndex: 1)
                XCTAssertEqual(bounded.count, input.count)
                XCTAssertEqual(bounded, bounded.sorted(), "Bounded input: \(input)")
                XCTAssertTrue(bounded.allSatisfy { $0 <= 1 })
                for anchor in boundedAnchors { XCTAssertEqual(bounded[anchor], input[anchor]) }
            }
            caseCount *= 3
        }
    }

    /// Independent quadratic oracle: Qwen's fix_timestamp at c17a131.
    /// https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L147
    private func referenceAnchorPositions(_ input: [Int]) -> [Int] {
        guard !input.isEmpty else { return [] }
        var lengths = [Int](repeating: 1, count: input.count)
        var parents = [Int](repeating: -1, count: input.count)
        var endpoint = 0
        for i in input.indices {
            for j in 0..<i where input[j] <= input[i] {
                if lengths[j] + 1 > lengths[i] {
                    lengths[i] = lengths[j] + 1
                    parents[i] = j
                }
            }
            if lengths[i] > lengths[endpoint] { endpoint = i }
        }
        var positions: [Int] = []
        while endpoint >= 0 {
            positions.append(endpoint)
            endpoint = parents[endpoint]
        }
        return positions.reversed()
    }
}
