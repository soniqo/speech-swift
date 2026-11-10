import XCTest
@testable import AudioCommon
@testable import Qwen3ASR

final class AlignmentTimestampSlotTests: XCTestCase {
    private let timestamp = Qwen3ASRTokens.timestampTokenId

    func testBothTimestampSlotsFollowEveryWord() {
        let tokenizer = Qwen3Tokenizer(idToToken: [
            1: "C", 2: "a", 3: "n", 4: "y", 5: "o", 6: "u",
        ])
        let slotted = TextPreprocessor.prepareForAlignment(
            text: "Can you!", tokenizer: tokenizer)

        XCTAssertEqual(slotted.words, ["Can", "you!"])
        XCTAssertEqual(slotted.tokenIds, [1, 2, 3, timestamp, timestamp, 4, 5, 6, timestamp, timestamp])
        XCTAssertEqual(slotted.timestampPositions, [3, 4, 8, 9])
    }

    func testSingleWordHasTrailingStartAndEndSlots() {
        let tokenizer = Qwen3Tokenizer(idToToken: [1: "H", 2: "i"])
        let slotted = TextPreprocessor.prepareForAlignment(
            text: "Hi.", tokenizer: tokenizer)

        XCTAssertEqual(slotted.words, ["Hi."])
        XCTAssertEqual(slotted.tokenIds, [1, 2, timestamp, timestamp])
        XCTAssertEqual(slotted.timestampPositions, [2, 3])
    }

    func testSlotsFollowEachHanCharacterPreservingPunctuation() {
        let tokenizer = Qwen3Tokenizer(idToToken: [1: "你", 2: "好"])
        let slotted = TextPreprocessor.prepareForAlignment(
            text: "你，好！", tokenizer: tokenizer, language: "Chinese")

        XCTAssertEqual(slotted.words, ["你，", "好！"])
        XCTAssertEqual(slotted.tokenIds, [1, timestamp, timestamp, 2, timestamp, timestamp])
        XCTAssertEqual(slotted.timestampPositions, [1, 2, 4, 5])
    }

    func testUnencodableWordsDoNotCreateOrphanTimestampSlots() {
        let tokenizer = Qwen3Tokenizer(idToToken: [1: "A"])
        let slotted = TextPreprocessor.prepareForAlignment(
            text: "A missing, A!", tokenizer: tokenizer)

        XCTAssertEqual(slotted.words, ["Amissing,", "A!"])
        XCTAssertEqual(slotted.tokenIds, [1, timestamp, timestamp, 1, timestamp, timestamp])
        XCTAssertEqual(slotted.timestampPositions, [1, 2, 4, 5])
    }

    func testEmptyAndPunctuationOnlyTextHasNoSlots() {
        let tokenizer = Qwen3Tokenizer(idToToken: [1: "A"])
        for text in ["", " \n\t", "... ！？"] {
            let slotted = TextPreprocessor.prepareForAlignment(text: text, tokenizer: tokenizer)
            XCTAssertTrue(slotted.words.isEmpty)
            XCTAssertTrue(slotted.tokenIds.isEmpty)
            XCTAssertTrue(slotted.timestampPositions.isEmpty)
        }
    }
}
