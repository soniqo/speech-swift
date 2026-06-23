import XCTest
@testable import Qwen3Chat

/// Regression tests for byte-level BPE streaming decode.
///
/// Decoding each token's bytes in isolation corrupts any multi-byte UTF-8 character (emoji,
/// Cyrillic, CJK) that BPE split across token boundaries — the symptom was output like
/// `todayĠðŁĺĬ` instead of `today 😊`. `decodeUTF8Prefix` fixes this by emitting only complete
/// characters and carrying incomplete trailing bytes forward to the next token.
final class ChatTokenizerDecodeTests: XCTestCase {

    func testEmptyBuffer() {
        let (text, remainder) = ChatTokenizer.decodeUTF8Prefix([])
        XCTAssertEqual(text, "")
        XCTAssertTrue(remainder.isEmpty)
    }

    func testCompleteASCII() {
        let (text, remainder) = ChatTokenizer.decodeUTF8Prefix(Array("hello".utf8))
        XCTAssertEqual(text, "hello")
        XCTAssertTrue(remainder.isEmpty)
    }

    /// A 4-byte emoji split mid-character must hold back the incomplete tail, not corrupt it.
    func testSplitEmojiCarriesRemainder() {
        let emoji = Array("😊".utf8)        // F0 9F 98 8A
        XCTAssertEqual(emoji.count, 4)

        // First two bytes alone are an incomplete sequence → no text yet, all carried over.
        let (t1, r1) = ChatTokenizer.decodeUTF8Prefix(Array(emoji.prefix(2)))
        XCTAssertEqual(t1, "")
        XCTAssertEqual(r1, Array(emoji.prefix(2)))

        // Completing the sequence yields the emoji and drains the buffer.
        let (t2, r2) = ChatTokenizer.decodeUTF8Prefix(r1 + Array(emoji.suffix(2)))
        XCTAssertEqual(t2, "😊")
        XCTAssertTrue(r2.isEmpty)
    }

    /// Mixed text + emoji where the emoji is split across the boundary decodes to its prefix
    /// plus a carried remainder — never the raw byte-level-BPE characters.
    func testValidPrefixWithIncompleteTail() {
        let bytes = Array("today ".utf8) + Array("😊".utf8).prefix(1)
        let (text, remainder) = ChatTokenizer.decodeUTF8Prefix(Array(bytes))
        XCTAssertEqual(text, "today ")
        XCTAssertEqual(remainder, Array("😊".utf8).prefix(1).map { $0 })
    }

    /// Simulate a token-by-token stream of arbitrary byte chunks and assert the reassembled
    /// text exactly equals the original — covers emoji, Cyrillic, and CJK split anywhere.
    func testStreamingReassembly() {
        let samples = [
            "Hey! How's your day going? 😊✨",
            "Как дела? Чем занят сегодня?",
            "今日は何をしていますか？🤖",
        ]
        for original in samples {
            let all = Array(original.utf8)
            var out = ""
            var pending: [UInt8] = []
            // Chunk the bytes irregularly (1..3 bytes) to mimic BPE token boundaries.
            var i = 0
            while i < all.count {
                let n = min((i % 3) + 1, all.count - i)
                pending.append(contentsOf: all[i..<i + n])
                let (text, remainder) = ChatTokenizer.decodeUTF8Prefix(pending)
                out += text
                pending = remainder
                i += n
            }
            if !pending.isEmpty { out += String(decoding: pending, as: UTF8.self) }
            XCTAssertEqual(out, original, "streaming reassembly corrupted: \(original)")
        }
    }
}
