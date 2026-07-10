import AudioCommon
@testable import F5TTS
import XCTest

final class F5TTSPinyinTests: XCTestCase {
    private struct Fixture: Decodable {
        let text: String
        let tokens: [String]
    }

    private func subsetConverter() throws -> F5TTSPinyinConverter {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "pinyin_lexicon_subset", withExtension: "tsv"))
        return try F5TTSPinyinConverter(lexiconURL: url)
    }

    private func fixtures() throws -> [Fixture] {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "pinyin_fixtures", withExtension: "json"))
        return try JSONDecoder().decode([Fixture].self, from: Data(contentsOf: url))
    }

    func testLexiconParsesCommentsEntriesAndRejectsEmpty() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("f5-pinyin-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let lexicon = dir.appendingPathComponent("lex.tsv")
        try "# comment\n好\thao3\n你好\tni2 hao3\n".write(to: lexicon, atomically: true, encoding: .utf8)
        let converter = try F5TTSPinyinConverter(lexiconURL: lexicon)
        XCTAssertEqual(converter.entries.count, 2)
        XCTAssertEqual(converter.maxKeyLength, 2)
        XCTAssertEqual(converter.entries["你好"], ["ni2", "hao3"])

        let empty = dir.appendingPathComponent("empty.tsv")
        try "# only comments\n".write(to: empty, atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try F5TTSPinyinConverter(lexiconURL: empty))
    }

    func testCanonicalMandarinSentences() throws {
        let converter = try subsetConverter()

        XCTAssertEqual(
            converter.convert("你好世界。"),
            [" ", "ni2", " ", "hao3", " ", "shi4", " ", "jie4", "。"])
        XCTAssertEqual(converter.convert("银行"), [" ", "yin2", " ", "hang2"])
        XCTAssertEqual(converter.convert("音乐"), [" ", "yin1", " ", "yue4"])
        XCTAssertEqual(converter.convert("一个"), [" ", "yi2", " ", "ge4"])
    }

    func testEnglishTextPassesThroughUnchanged() throws {
        let converter = try subsetConverter()
        let text = "Hello world, it's 3:45 PM!"
        XCTAssertEqual(converter.convert(text), text.map(String.init))
    }

    func testSystemFallbackForOutOfLexiconHanzi() {
        let converter = F5TTSPinyinConverter(entries: ["好": ["hao3"]])
        let tokens = converter.convert("好嗎")
        XCTAssertEqual(tokens.count, 4)
        XCTAssertEqual(tokens[0], " ")
        XCTAssertEqual(tokens[1], "hao3")
        XCTAssertEqual(tokens[2], " ")
        XCTAssertTrue(tokens[3].hasPrefix("ma"), "expected transform fallback, got \(tokens[3])")
    }

    func testUpstreamFixtureParity() throws {
        let converter = try subsetConverter()
        var exactSequences = 0
        var pinyinTotal = 0
        var pinyinExact = 0
        var nonToneMisses: [String] = []

        func isPinyinToken(_ token: String) -> Bool {
            token.count > 1 && token.unicodeScalars.allSatisfy(\.isASCII)
                && (token.last!.isNumber || token.allSatisfy(\.isLetter))
        }

        for fixture in try fixtures() {
            let got = converter.convert(F5TTSTokenizer.normalize(fixture.text))
            if got == fixture.tokens { exactSequences += 1 }
            XCTAssertEqual(
                got.count, fixture.tokens.count,
                "token count diverged for: \(fixture.text)")
            guard got.count == fixture.tokens.count else { continue }
            for (g, e) in zip(got, fixture.tokens) where isPinyinToken(e) {
                pinyinTotal += 1
                if g == e {
                    pinyinExact += 1
                } else if g.dropLast() != e.dropLast() {
                    nonToneMisses.append("\(e) -> \(g) in \(fixture.text)")
                }
            }
        }

        let accuracy = Double(pinyinExact) / Double(max(pinyinTotal, 1))
        XCTAssertGreaterThanOrEqual(exactSequences, 60, "exact sequences regressed")
        XCTAssertGreaterThanOrEqual(accuracy, 0.98, "pinyin token accuracy regressed: \(accuracy)")
        XCTAssertEqual(nonToneMisses, [], "only tone-level differences from upstream are acceptable")
    }

    func testTokenizerRoutesCJKThroughPinyin() throws {
        let converter = F5TTSPinyinConverter(entries: ["你好": ["ni2", "hao3"]])
        let tokenizer = F5TTSTokenizer(
            vocab: [" ": 0, "!": 1, "ni2": 100, "hao3": 101],
            pinyin: converter)

        let tokenized = try tokenizer.tokenize("你好!")
        XCTAssertEqual(tokenized.symbols, [" ", "ni2", " ", "hao3", "!"])
        XCTAssertEqual(tokenized.ids, [0, 100, 0, 101, 1])
    }

    func testTokenizerWithoutLexiconKeepsRejectingCJK() {
        let tokenizer = F5TTSTokenizer(vocab: [" ": 0])
        XCTAssertThrowsError(try tokenizer.encode("你好")) { error in
            XCTAssertTrue((error as? F5TTSError)?.localizedDescription.contains("pinyin") == true)
        }
    }

    func testToneMarkToTone3Conversion() {
        XCTAssertEqual(F5TTSPinyinConverter.toneMarkToTone3("nǐ"), "ni3")
        XCTAssertEqual(F5TTSPinyinConverter.toneMarkToTone3("lǜ"), "lv4")
        XCTAssertEqual(F5TTSPinyinConverter.toneMarkToTone3("ma"), "ma")
        XCTAssertEqual(F5TTSPinyinConverter.toneMarkToTone3("zhōng"), "zhong1")
    }
}
