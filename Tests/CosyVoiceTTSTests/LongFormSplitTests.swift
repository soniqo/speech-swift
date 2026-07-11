import XCTest
@testable import CosyVoiceTTS

/// Unit tests for `CosyVoiceTTSModel.splitForLongForm`, the helper that breaks
/// long input text into LLM-friendly segments. No model load needed — this is
/// a pure string-processing function.
final class LongFormSplitTests: XCTestCase {

    private func split(_ text: String) -> [String] {
        CosyVoiceTTSModel.splitForLongForm(text)
    }

    // MARK: - Trivial cases

    func testSingleSentenceReturnsItself() {
        let segments = split("This is a single short sentence.")
        XCTAssertEqual(segments, ["This is a single short sentence."])
    }

    func testEmptyInputReturnsEmpty() {
        XCTAssertEqual(split("").count, 0)
        XCTAssertEqual(split("   \n\t ").count, 0)
    }

    func testNoPunctuationFallsBackToSingleSegment() {
        let text = "this has no terminating punctuation"
        XCTAssertEqual(split(text), [text])
    }

    // MARK: - Sentence-terminator splits
    //
    // Most fixture sentences are ≥ 6 words so the short-fragment merge logic
    // doesn't collapse them into a single segment unless the test says so.

    func testSplitsOnPeriod() {
        let s = split("This is the first sentence here. " +
                      "This is the second sentence here. " +
                      "This is the third sentence here.")
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s.first, "This is the first sentence here.")
        XCTAssertEqual(s.last,  "This is the third sentence here.")
    }

    func testSplitsOnQuestionMark() {
        let s = split("Is this really truly a question? " +
                      "Yes it really is a statement. " +
                      "And another small follow-up question?")
        XCTAssertEqual(s.count, 3)
        XCTAssertTrue(s.first!.hasSuffix("?"))
        XCTAssertTrue(s.last!.hasSuffix("?"))
    }

    func testShortQuestionMergesForward() {
        let s = split("Hello there how nice to see! " +
                      "How are you doing today? " +
                      "I am doing fine thanks for asking.")
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s.first, "Hello there how nice to see!")
        XCTAssertTrue(s.last!.contains("How are you doing today?"))
        XCTAssertTrue(s.last!.contains("I am doing fine thanks for asking."))
    }

    func testTrailingTextWithoutTerminatorIsKept() {
        // The buffer after the last terminator must not be dropped.
        // Both fragments long enough to avoid the merge logic.
        let s = split("This is the first complete sentence here. " +
                      "Trailing fragment without any period at the end")
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s.first, "This is the first complete sentence here.")
        XCTAssertEqual(s.last, "Trailing fragment without any period at the end")
    }

    // MARK: - Short-segment merging

    func testShortLeadFragmentMergesIntoNext() {
        // "Hi." is 1 word and should merge forward.
        let s = split("Hi. This is the second sentence which is longer.")
        XCTAssertEqual(s.count, 1)
        XCTAssertEqual(s[0], "Hi. This is the second sentence which is longer.")
    }

    func testShortQuestionLeadMergesIntoFollowingSentence() {
        let text = "Why does this matter? " +
                   "Because the next decade of audio software is going to look very different."
        let s = split(text)
        XCTAssertEqual(s, [text])
    }

    func testTwoVeryShortSegmentsMerge() {
        let s = split("Ok. Sure. Now this one is the third sentence with enough words.")
        // "Ok." (1 word) merges forward into "Sure." (still short, merges into the next).
        XCTAssertLessThan(s.count, 3)
        XCTAssertTrue(s.last!.contains("third sentence"))
    }

    // MARK: - Long-segment clause splits

    func testLongSentenceSplitsOnComma() {
        // 30+ words exceeds maxWordsPerSegment=25, so the splitter should look for
        // clause boundaries (commas).
        let long = "This is a very long sentence with many many many many many words separated by clauses, " +
                   "and we want it broken up at sensible boundaries, before the model loses coherence."
        let s = split(long)
        XCTAssertGreaterThan(s.count, 1, "Long sentence should split into multiple clauses")
        for seg in s {
            XCTAssertLessThanOrEqual(
                seg.split(whereSeparator: { $0.isWhitespace }).count, 25,
                "Each clause should stay below the 25-word cap: \(seg)")
        }
    }

    // MARK: - Multi-paragraph realistic input

    func testRealisticLongFormInput() {
        let text = """
        Hi, this is an extended demonstration of zero-shot voice cloning running entirely on Apple Silicon. \
        Everything you are hearing was synthesized in real time. There was no fine-tuning, no cloud calls. \
        We think the next year of audio software is going to look very different.
        """
        let s = split(text)
        XCTAssertGreaterThanOrEqual(s.count, 3, "Should produce at least 3 sentence-level segments")
        // The concatenation should preserve all the text (minus whitespace normalisation).
        let joined = s.joined(separator: " ")
        XCTAssertTrue(joined.contains("Apple Silicon"))
        XCTAssertTrue(joined.contains("very different"))
    }

    // MARK: - Idempotency

    func testRepeatedSplitsAreStable() {
        let text = "One sentence. Two sentences. Three sentences."
        let first = split(text)
        // Joining and re-splitting should produce the same partitions.
        let joined = first.joined(separator: " ")
        let second = split(joined)
        XCTAssertEqual(first, second)
    }

    // MARK: - Long-form audio stitching

    func testStitchLongFormSegmentsFadesInsertedGapEdges() {
        let left = [Float](repeating: 1.0, count: 100)
        let right = [Float](repeating: -1.0, count: 100)

        let stitched = CosyVoiceTTSModel.stitchLongFormSegments(
            [left, right],
            sampleRate: 1_000,
            gapSeconds: 0.2,
            fadeSeconds: 0.03
        )

        XCTAssertEqual(stitched.count, 400)
        XCTAssertEqual(stitched[100..<300].allSatisfy { $0 == 0 }, true)

        // The pre-fix stitch jumped from full-scale segment audio straight to
        // zero-gap silence. The fade should leave both sides near zero instead.
        XCTAssertLessThan(abs(stitched[99]), 0.1)
        XCTAssertLessThan(abs(stitched[300]), 0.1)
        XCTAssertLessThan(abs(stitched[100] - stitched[99]), 0.1)
        XCTAssertLessThan(abs(stitched[300] - stitched[299]), 0.1)

        // Every edge is faded — including the first segment's head (the
        // vocoder starts from zero left-context, so an unfaded head is an
        // audible click at utterance start) and the final tail.
        XCTAssertLessThan(abs(stitched[0]), 0.1)
        XCTAssertLessThan(abs(stitched[399]), 0.1)
    }

    func testStitchLongFormSegmentsFadesTailOfSingleSegment() {
        // Regression: a multi-segment split can render down to ONE surviving
        // segment. That path must still fade both edges — the head (start
        // click) and the tail (end click) — matching `cleanCloneOutput`.
        let single = [Float](repeating: 1.0, count: 100)
        let stitched = CosyVoiceTTSModel.stitchLongFormSegments(
            [single],
            sampleRate: 1_000,
            gapSeconds: 0.2,
            fadeSeconds: 0.03
        )
        XCTAssertEqual(stitched.count, 100)
        XCTAssertLessThan(abs(stitched[0]), 0.1)
        XCTAssertEqual(stitched[69], 1.0, "interior samples must be untouched")
        XCTAssertLessThan(abs(stitched[99]), 0.1)
    }

    func testStitchLongFormSegmentsCollapsedMultiSegmentStillFades() {
        // Two rendered segments where one came back empty — the collapsed
        // result must behave like the single-segment case above, not skip
        // the fades via an early exit.
        let survivor = [Float](repeating: -1.0, count: 100)
        let stitched = CosyVoiceTTSModel.stitchLongFormSegments(
            [[], survivor],
            sampleRate: 1_000,
            gapSeconds: 0.2,
            fadeSeconds: 0.03
        )
        XCTAssertEqual(stitched.count, 100)
        XCTAssertLessThan(abs(stitched[0]), 0.1)
        XCTAssertLessThan(abs(stitched[99]), 0.1)
    }

    func testCleanCloneOutputFadesEdgesWithoutTrimming() {
        // The prompt region is sliced off in the mel domain before vocoding,
        // so cleanup must NOT drop samples — only fade the edges.
        let samples = [Float](repeating: 1.0, count: 700)
        let cleaned = CosyVoiceTTSModel.cleanCloneOutput(
            samples,
            sampleRate: 1_000,
            edgeFadeSeconds: 0.03
        )

        XCTAssertEqual(cleaned.count, 700)
        XCTAssertLessThan(abs(cleaned.first ?? 1.0), 0.1)
        XCTAssertLessThan(abs(cleaned.last ?? 1.0), 0.1)
        XCTAssertEqual(cleaned[350], 1.0, "interior samples must be untouched")
    }
}
