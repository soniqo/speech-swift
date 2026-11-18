import AudioCommon
import CoreML
import XCTest
@testable import ParakeetASR

final class ParakeetWindowStitcherTests: XCTestCase {
    private func word(
        _ text: String, _ time: Double, start: Double? = nil, end: Double? = nil,
        probabilities: [Float] = [-0.1]
    ) -> ParakeetDecodedWord {
        ParakeetDecodedWord(
            value: WordConfidence(word: text, confidence: exp(probabilities.reduce(0, +) / Float(probabilities.count)),
                startTime: start ?? time - 0.05, endTime: end ?? time + 0.05), tokenLogProbs: probabilities)
    }

    private func text(_ stitcher: ParakeetWindowStitcher) -> [String] {
        stitcher.words.map { $0.value.word }
    }

    func testContextPlanPreservesTheOriginalCuts() {
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 1100, maxWindow: 500, overlap: true),
            [0, 250, 500, 750, 1000])
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 1100, maxWindow: 500, overlap: false),
            [0, 500, 1000])
    }

    func testShortInputsKeepOnePass() {
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 0, maxWindow: 500, overlap: true), [])
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 1, maxWindow: 500, overlap: true), [0])
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 500, maxWindow: 500, overlap: true), [0])
    }

    func testOddWindowSizesStillPreservePrimaryCuts() {
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 14, maxWindow: 5, overlap: true),
            [0, 2, 5, 7, 10, 12])
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 3, maxWindow: 1, overlap: true), [0, 1, 2])
    }

    func testLongEncoderPlanKeepsConsecutiveWindows() {
        XCTAssertEqual(ParakeetWindowPlan.starts(validFrames: 6500, maxWindow: 3000, overlap: false),
            [0, 3000, 6000])
    }

    func testNormalizationContextIsBoundedAndContainsTheWindow() {
        XCTAssertEqual(ParakeetWindowPlan.normalizationContext(start: 500, length: 500, total: 1600, limit: 3000),
            0..<1600)
        XCTAssertEqual(ParakeetWindowPlan.normalizationContext(start: 6000, length: 500, total: 12000, limit: 3000),
            4750..<7750)
        XCTAssertEqual(ParakeetWindowPlan.normalizationContext(start: 11800, length: 200, total: 12000, limit: 3000),
            9000..<12000)
    }

    func testContextSliceKeepsTheFrameOrderAcrossBins() throws {
        let mel = try MLMultiArray(shape: [1, 2, 5], dataType: .float16)
        let source = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<10 { source[i] = Float16(i) }
        let result = try MelPreprocessor.slice(mel, start: 1, length: 3)
        XCTAssertEqual(result.shape, [1, 2, 3])
        let values = result.dataPointer.assumingMemoryBound(to: Float16.self)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: values, count: 6)), [1, 2, 3, 6, 7, 8])
        XCTAssertEqual(Array(UnsafeBufferPointer(start: source, count: 10)), (0..<10).map(Float16.init))
    }

    func testMatchingOverlapIsIncludedOnce() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("we", 3), word("go", 4.1), word("home", 4.6)], start: 0, end: 5)
        stitcher.append([word("go", 4.1), word("home", 4.6), word("now", 5.3)], start: 4, end: 9)
        XCTAssertEqual(text(stitcher), ["we", "go", "home", "now"])
    }

    func testPrimaryWindowKeepsRecognizedWordsAtAJoin() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("the", 4), word("illness", 4.6)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("the", 4), word("illness", 4.6), word("wrong", 4.95),
            word("middle", 5.1), word("recovered", 6.4),
        ], start: 2.5, end: 7.5, primary: false)
        stitcher.append([
            word("illness", 5), word("true", 5.3), word("middle", 5.5),
            word("recovered", 6.4), word("tail", 8),
        ], start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["the", "illness", "true", "middle", "recovered", "tail"])
    }

    func testContextStillFillsAnEmptyPrimaryWindow() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("begin", 1), word("crossing", 4.6)], start: 0, end: 5, primary: true)
        stitcher.append([word("crossing", 4.6), word("recovered", 6.4)],
            start: 2.5, end: 7.5, primary: false)
        stitcher.append([], start: 5, end: 10, primary: true)
        stitcher.append([word("next", 8.5), word("end", 10.5)], start: 7.5, end: 12.5, primary: false)
        XCTAssertEqual(text(stitcher), ["begin", "crossing", "recovered", "next", "end"])
    }

    func testEarlierAnchorCannotReplaceAProtectedPrimaryPhrase() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([
            word("important", 4.1), word("original", 4.4),
            word("phrase", 4.64, start: 4.48, end: 4.8),
        ], start: 0, end: 5, primary: true)
        stitcher.append([
            word("important", 4.1), word("wrong", 4.5), word("word", 4.7), word("next", 5.2),
        ], start: 2.5, end: 7.5, primary: false)
        XCTAssertEqual(text(stitcher), ["important", "original", "phrase", "next"])
    }

    func testLateAnchorCannotDiscardTheIncomingPrimaryPhrase() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("chinoise.", 4.74)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("chinoise.", 4.74), word("The", 5.18), word("million", 5.5),
            word("person", 5.82), word("combat", 6.66), word("pour", 6.9),
        ], start: 2.5, end: 7.5, primary: false)
        let primary = [
            word("Des", 5.24, start: 5.16, end: 5.32), word("milliers", 5.52),
            word("de", 5.76), word("personnes", 5.96), word("sont", 6.2),
            word("mortes", 6.4), word("au", 6.68), word("combat", 6.88),
            word("pour", 7.16), word("la", 8), word("suite.", 9.1),
        ]
        stitcher.append(primary, start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["chinoise."] + primary.map { $0.value.word })
    }

    func testSingleLateAnchorCannotKeepAConflictingContextPrefix() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("personnes.", 4.64, start: 4.48, end: 4.8)],
            start: 0, end: 5, primary: true)
        stitcher.append([
            word("person", 4.7), word("who", 5.26), word("visit", 5.54),
            word("and", 5.78), word("travail", 6.06), word("in", 6.3),
            word("coopération", 6.54), word("society.", 7.14),
        ], start: 2.5, end: 7.5, primary: false)
        let primary = [
            word("Ils", 5.28, start: 5.16, end: 5.4), word("vivent", 5.56),
            word("et", 5.8), word("travaillent", 6.08), word("en", 6.3),
            word("coopération,", 6.68), word("en", 7.24), word("société.", 7.72),
        ]
        stitcher.append(primary, start: 5, end: 8.28, primary: true)
        XCTAssertEqual(text(stitcher), ["personnes."] + primary.map { $0.value.word })
    }

    func testPrimaryAgreementCannotRetainAnExtraContextInsertion() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("begin", 1)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("extra", 5.3), word("correct", 5.45), word("phrase", 6.1), word("later", 6.6),
        ], start: 2.5, end: 7.5, primary: false)
        stitcher.append([
            word("correct", 5.24), word("phrase", 6.1), word("later", 6.6), word("tail", 8),
        ], start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["begin", "correct", "phrase", "later", "tail"])
    }

    func testLateAgreementStillPreservesContextBeforeTheFirstPrimaryWord() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("begin", 1)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("missing", 6), word("words", 6.4), word("ending", 7), word("phrase", 7.3),
        ], start: 2.5, end: 7.5, primary: false)
        stitcher.append([
            word("ending", 7), word("phrase", 7.3), word("tail", 9),
        ], start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["begin", "missing", "words", "ending", "phrase", "tail"])
    }

    func testFallbackPreservesContextUntilTheFirstPrimaryWord() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("begin", 1)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("kept", 5.8), word("passage", 6.4), word("wrong", 7), word("ending", 7.3),
        ], start: 2.5, end: 7.5, primary: false)
        stitcher.append([
            word("true", 7), word("ending", 7.3), word("tail", 9),
        ], start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["begin", "kept", "passage", "true", "ending", "tail"])
    }

    func testStrongerContextCanStillRepairAnUncertainPrimaryPhrase() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([
            word("important", 4.1, probabilities: [-0.05]),
            word("uncertain", 4.4, probabilities: [-2]),
            word("phrase", 4.64, start: 4.48, end: 4.8, probabilities: [-2]),
        ], start: 0, end: 5, primary: true)
        stitcher.append([
            word("important", 4.1, probabilities: [-0.05]), word("good", 4.5, probabilities: [-0.01]),
            word("words", 4.7, probabilities: [-0.01]), word("next", 5.2, probabilities: [-0.01]),
        ], start: 2.5, end: 7.5, primary: false)
        XCTAssertEqual(text(stitcher), ["important", "good", "words", "next"])
    }

    func testUncertainPrimaryDoesNotEraseStrongerContextBeforeALateAnchor() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("begin", 1)], start: 0, end: 5, primary: true)
        stitcher.append([
            word("clear", 5.3, probabilities: [-0.01]), word("speech", 5.8, probabilities: [-0.01]),
            word("combat", 6.7, probabilities: [-0.02]), word("pour", 7, probabilities: [-0.02]),
        ], start: 2.5, end: 7.5, primary: false)
        stitcher.append([
            word("uncertain", 5.3, probabilities: [-2]), word("combat", 6.7, probabilities: [-0.02]),
            word("pour", 7, probabilities: [-0.02]), word("tail", 8, probabilities: [-0.02]),
        ], start: 5, end: 10, primary: true)
        XCTAssertEqual(text(stitcher), ["begin", "clear", "speech", "combat", "pour", "tail"])
    }

    func testLegitimateRepeatedWordsSurvive() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("no", 3.8), word("no", 4.1), word("no", 4.6)], start: 0, end: 5)
        stitcher.append([word("no", 4.1), word("no", 4.6), word("no", 5.1)], start: 4, end: 9)
        XCTAssertEqual(text(stitcher), ["no", "no", "no", "no"])
    }

    func testEmptyMiddleWindowDoesNotEraseTheFollowingWords() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("keep", 4.7), word("this", 4.9)], start: 0, end: 5)
        stitcher.append([], start: 2.5, end: 7.5)
        XCTAssertEqual(stitcher.coveredEnd, 5)
        stitcher.append([word("next", 5.3)], start: 5, end: 10)
        XCTAssertEqual(text(stitcher), ["keep", "this", "next"])
    }

    func testEmptyFirstWindowAllowsTheNextHypothesis() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([], start: 0, end: 5)
        XCTAssertNil(stitcher.coveredEnd)
        stitcher.append([word("keep", 4.1), word("this", 4.3)], start: 4, end: 9)
        XCTAssertEqual(text(stitcher), ["keep", "this"])
    }

    func testContainedSuffixCannotReplaceACompleteHypothesis() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("hello", 1), word("world", 4.7)], start: 0, end: 5)
        stitcher.append([word("wrong", 4.8)], start: 2.5, end: 5)
        XCTAssertEqual(text(stitcher), ["hello", "world"])
        XCTAssertEqual(stitcher.coveredEnd, 5)
    }

    func testDisjointSpeechIsRetained() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("hello", 1)], start: 0, end: 5)
        stitcher.append([word("world", 8)], start: 4, end: 9)
        XCTAssertEqual(text(stitcher), ["hello", "world"])
    }

    func testUnicodeAndPunctuationMatchWithoutChangingTheText() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("un", 2), word("l’été", 4.2), word("français", 4.6)], start: 0, end: 5)
        stitcher.append([word("l'e\u{301}té", 4.2), word("français,", 4.6), word("oui", 5.4)], start: 4, end: 9)
        XCTAssertEqual(text(stitcher), ["un", "l’été", "français", "oui"])
    }

    func testApostrophesArePartOfTheWordIdentity() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("she’ll", 4.1), word("before", 4.3)], start: 0, end: 6)
        stitcher.append([word("shell", 4.1), word("after", 5.2)], start: 3, end: 8)
        XCTAssertEqual(text(stitcher), ["she’ll", "before", "after"])
    }

    func testSingleShortWordDoesNotOverrideAnUnmatchedPhrase() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("no", 4.1), word("before", 4.3)], start: 0, end: 6)
        stitcher.append([word("no", 4.1), word("after", 5.2)], start: 3, end: 8)
        XCTAssertEqual(text(stitcher), ["no", "before", "after"])
    }

    func testConfidenceUsesOnlyTheRetainedTokenProbabilities() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([
            word("we", 3, probabilities: [-0.2]), word("go", 4.1, probabilities: [-0.4, -0.3]),
            word("home", 4.6, probabilities: [-0.1]),
        ], start: 0, end: 5)
        stitcher.append([
            word("go", 4.1, probabilities: [-5]), word("home", 4.6, probabilities: [-5]),
            word("now", 5.3, probabilities: [-0.05]),
        ], start: 4, end: 9)
        XCTAssertEqual(stitcher.words.flatMap(\.tokenLogProbs), [-0.2, -0.4, -0.3, -0.1, -0.05])
    }

    func testFinalTimesAreMonotonicAndWithinTheAudio() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([word("we", 3.9), word("house", 4.5)], start: 0, end: 5)
        stitcher.append([word("house", 4.4), word("next", 4.45), word("end", 9.1)], start: 4, end: 9.2)
        let result = stitcher.finalized(audioDuration: 9)
        XCTAssertEqual(result.map { $0.value.word }, ["we", "house", "next", "end"])
        let starts = result.map { $0.value.startTime! }
        XCTAssertEqual(starts, starts.sorted())
        for word in result {
            XCTAssertGreaterThanOrEqual(word.value.startTime!, 0)
            XCTAssertLessThanOrEqual(word.value.endTime!, 9)
            XCTAssertGreaterThanOrEqual(word.value.endTime!, word.value.startTime!)
        }
        XCTAssertEqual(result.flatMap(\.tokenLogProbs), stitcher.words.flatMap(\.tokenLogProbs))
    }

    func testEmptyWindowsStayEmpty() {
        var stitcher = ParakeetWindowStitcher()
        stitcher.append([], start: 0, end: 5)
        stitcher.append([], start: 4, end: 9)
        XCTAssertTrue(stitcher.words.isEmpty)
        XCTAssertTrue(stitcher.finalized(audioDuration: 9).isEmpty)
    }
}
