import XCTest
@testable import SupertonicTTS

/// Offline unit tests for the G2P-free front-end (no model download).
final class SupertonicTokenizerTests: XCTestCase {
    /// Identity table over the BMP: codepoint == id.
    private func identityTokenizer() -> SupertonicTokenizer {
        SupertonicTokenizer(indexer: (0..<65536).map { Int32($0) })
    }

    func testAvailableLanguages() {
        let t = identityTokenizer()
        XCTAssertTrue(t.supports("en"))
        XCTAssertTrue(t.supports("de"))
        XCTAssertTrue(t.supports("ru"))
        XCTAssertTrue(t.supports("ko"))
        XCTAssertFalse(t.supports("zz"))
        XCTAssertFalse(t.supports("zh"))  // Supertonic excludes zh
    }

    func testUnsupportedLanguageThrows() {
        let t = identityTokenizer()
        XCTAssertThrowsError(try t.process("hi", lang: "zz", textLength: 128))
    }

    func testLangWrapAndTerminalPunctuation() throws {
        let t = identityTokenizer()
        let tok = try t.process("hi", lang: "en", textLength: 128)
        // wrapped = "<en>hi.</en>" → ids start with '<','e','n','>'
        XCTAssertEqual(Array(tok.ids.prefix(4)),
                       [Int32(UInt32("<")), Int32(UInt32("e")), Int32(UInt32("n")), Int32(UInt32(">"))])
        XCTAssertEqual(tok.mask[0], 1.0)
    }

    /// NFKD keystone: "Käse" must decompose ä → 'a' + combining diaeresis (U+0308).
    func testNFKDDecomposesUmlaut() throws {
        let t = identityTokenizer()
        let tok = try t.process("Käse", lang: "de", textLength: 128)
        // wrapped = "<de>" + "Ka◌̈se." + "</de>" → index 4 = 'K'? "<de>" is 4 codepoints (0..3),
        // so position 4 is 'K', 5 'a', 6 = U+0308 (combining diaeresis).
        XCTAssertEqual(tok.ids[4], Int32(UInt32("K")))
        XCTAssertEqual(tok.ids[5], Int32(UInt32("a")))
        XCTAssertEqual(tok.ids[6], 0x0308, "NFKD must split ä into a + combining diaeresis")
    }

    func testUnknownCodepointIsMinusOne() throws {
        // indexer covers only 0..127; everything above is unknown (-1).
        let t = SupertonicTokenizer(indexer: (0..<128).map { Int32($0) })
        let tok = try t.process("é", lang: "fr", textLength: 64)
        // After NFKD "é" → 'e' + U+0301; 'e' is in-table, U+0301 (769) is out → -1.
        XCTAssertTrue(tok.ids.contains(Int32(UInt32("e"))))
        XCTAssertTrue(tok.ids.contains(-1), "Out-of-table codepoints must resolve to -1, never crash")
    }

    func testChunkingStaysWithinTextLength() {
        let t = identityTokenizer()
        let long = String(repeating: "word ", count: 200)
        let chunks = t.chunk(long, lang: "en", textLength: 128)
        XCTAssertGreaterThan(chunks.count, 1, "Long text must split into multiple chunks")
        for c in chunks {
            // wrapped length must fit the fixed text graph (T + a little slack for the tag).
            XCTAssertLessThanOrEqual(c.unicodeScalars.count + 2 * 2 + 5, 128 + 1)
        }
    }
}
