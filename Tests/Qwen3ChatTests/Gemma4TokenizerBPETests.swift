import XCTest
import Foundation
@testable import Qwen3Chat

/// Gemma 4 BPE against a miniature hand-built vocabulary, no model download needed.
///
/// Every case asserts the encoder's ids equal `ReferenceGemma4BPE`'s — the implementation as it
/// stood before the merge loop was rewritten. A tokenizer that is fast and subtly different changes
/// what the model reads and raises no error anywhere, so identity, not plausibility, is the gate.
final class Gemma4TokenizerBPETests: XCTestCase {

    // MARK: - Fixture

    /// A vocabulary small enough to reason about that still reaches every branch of the encoder:
    /// space runs, a merge that spans a space, byte fallback, one merge whose result the vocab does
    /// not carry, and a rule that occurs twice in one string.
    struct Fixture {
        let directory: URL
        let vocab: [String: Int]

        func id(_ token: String) -> Int {
            guard let id = vocab[token] else {
                XCTFail("fixture vocabulary has no token \(token.debugDescription)")
                return -1
            }
            return id
        }
    }

    private static let specialTokens = ["<pad>", "<eos>", "<bos>", "<unk>",
                                        "<|turn>", "<turn|>", "<|channel>"]

    /// Dependencies first: BPE always applies the lowest rank available, so a rule can only fire
    /// once the rules that build its operands rank below it.
    private static let mergeRules: [(String, String)] = [
        ("▁", "▁"), ("▁▁", "▁"),
        ("<", "/"), ("▁", "</"), (">", "▁</"),
        ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"), ("▁", "hello"),
        ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d"), ("▁", "world"),
        ("п", "р"), ("пр", "и"), ("при", "в"), ("прив", "е"), ("приве", "т"), ("▁", "привет"),
        ("a", "b"), ("ab", "ab"),
        ("q", "q"),
    ]
    /// `qq` is deliberately left out of the vocabulary: `tokenizer.json` does not require a merge
    /// result to be a token, and the encoder must byte-fall-back on it exactly as it used to.
    private static let mergeResultsOutsideVocab: Set<String> = ["qq"]

    private static func buildVocab() -> [String: Int] {
        var vocab: [String: Int] = [:]
        var next = 0
        func add(_ token: String) {
            if vocab[token] == nil { vocab[token] = next; next += 1 }
        }
        for token in specialTokens { add(token) }
        for b in 0..<256 { add(String(format: "<0x%02X>", b)) }
        add("▁"); add("\n"); add("\t")
        for scalar in UnicodeScalar("a").value...UnicodeScalar("z").value {
            add(String(UnicodeScalar(scalar)!))
        }
        for scalar in UnicodeScalar("0").value...UnicodeScalar("9").value {
            add(String(UnicodeScalar(scalar)!))
        }
        for ch in ".,!?<>/-:" { add(String(ch)) }
        for ch in "абвгдежзийклмнопрстуфхцчшщъыьэюя" { add(String(ch)) }
        for (left, right) in mergeRules where !mergeResultsOutsideVocab.contains(left + right) {
            add(left + right)
        }
        return vocab
    }

    /// `mergesAsStrings` writes the `["left right"]` encoding of the merge list instead of
    /// `[[left, right]]`; `load` accepts both and must read them the same way.
    private static func makeFixture(mergesAsStrings: Bool) -> Fixture {
        let vocab = buildVocab()
        let merges: Any = mergesAsStrings
            ? mergeRules.map { "\($0.0) \($0.1)" }
            : mergeRules.map { [$0.0, $0.1] }
        let json: [String: Any] = [
            "model": ["vocab": vocab, "merges": merges],
            "added_tokens": specialTokens.map {
                ["content": $0, "id": vocab[$0]!, "special": true] as [String: Any]
            },
        ]
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gemma4-bpe-fixture-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: json)
            try data.write(to: dir.appendingPathComponent("tokenizer.json"))
        } catch {
            fatalError("could not write Gemma 4 tokenizer fixture: \(error)")
        }
        return Fixture(directory: dir, vocab: vocab)
    }

    private static let fixture = makeFixture(mergesAsStrings: false)
    private static let stringMergesFixture = makeFixture(mergesAsStrings: true)

    private var tokenizer: Gemma4Tokenizer!
    private var reference: ReferenceGemma4BPE!

    override func setUpWithError() throws {
        tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.fixture.directory)
        reference = ReferenceGemma4BPE()
        try reference.load(from: Self.fixture.directory)
    }

    /// Asserts the encoder agrees with the pre-rewrite implementation, and hands the ids back so a
    /// case can also pin what they should be.
    @discardableResult
    private func assertParity(_ text: String,
                              file: StaticString = #filePath, line: UInt = #line) -> [Int] {
        let produced = tokenizer.encode(text)
        XCTAssertEqual(produced, reference.encode(text),
                       "ids diverged for \(text.debugDescription)", file: file, line: line)
        return produced
    }

    // MARK: - Degenerate input

    func testEmptyString() {
        XCTAssertEqual(assertParity(""), [])
    }

    func testWhitespaceOnly() {
        let v = Self.fixture
        XCTAssertEqual(assertParity(" "), [v.id("▁")])
        XCTAssertEqual(assertParity("  "), [v.id("▁▁")])
        XCTAssertEqual(assertParity("   "), [v.id("▁▁▁")])
        // Four spaces take the two-space rule twice rather than three-plus-one: the lowest rank
        // available wins each round, wherever it sits.
        XCTAssertEqual(assertParity("    "), [v.id("▁▁"), v.id("▁▁")])
        XCTAssertEqual(assertParity("\n"), [v.id("\n")])
        assertParity("\t")
        assertParity(" \n\t ")
    }

    func testLeadingAndTrailingSpaces() {
        let v = Self.fixture
        XCTAssertEqual(assertParity(" hello"), [v.id("▁hello")])
        XCTAssertEqual(assertParity("hello "), [v.id("hello"), v.id("▁")])
        XCTAssertEqual(assertParity(" hello "), [v.id("▁hello"), v.id("▁")])
        assertParity("  hello  ")
        assertParity("\n hello \n")
    }

    func testRepeatedInteriorSpaces() {
        let v = Self.fixture
        XCTAssertEqual(assertParity(" hello world"), [v.id("▁hello"), v.id("▁world")])
        // Two spaces merge with each other before either can start a word, so `world` loses its
        // `▁world` form. This is exactly what a word-splitting pre-tokenizer would retokenize.
        XCTAssertEqual(assertParity(" hello  world"),
                       [v.id("▁hello"), v.id("▁▁"), v.id("world")])
        assertParity(" hello   world")
        assertParity("a   b")
    }

    // MARK: - Merges that span a space

    func testMergeAcrossASpaceStillFires() {
        let v = Self.fixture
        // `>` + `▁</` is a real Gemma rule (the vocabulary has 466 rules whose right operand starts
        // with `▁`). Splitting on whitespace before BPE would put the operands in different units.
        XCTAssertEqual(assertParity("> </"), [v.id(">▁</")])
        assertParity("a> </b")
    }

    func testTieBreakPrefersLeftmostOccurrence() {
        let v = Self.fixture
        // Two positions hold the `a`+`b` rule at the same rank; the leftmost has to merge first for
        // `ab`+`ab` to become reachable.
        XCTAssertEqual(assertParity("abab"), [v.id("abab")])
        assertParity("ababab")
        assertParity("abababab")
    }

    func testNoMergeAvailable() {
        let v = Self.fixture
        XCTAssertEqual(assertParity("xyz"), [v.id("x"), v.id("y"), v.id("z")])
        assertParity("x y z")
        assertParity("z")
    }

    // MARK: - Byte fallback

    func testCharactersOutsideTheVocabularyByteFallBack() {
        let v = Self.fixture
        // No `你` token: it decomposes into its three UTF-8 bytes.
        XCTAssertEqual(assertParity("你"),
                       Array("你".utf8).map { v.id(String(format: "<0x%02X>", $0)) })
        assertParity("你好")
        assertParity("é")
        assertParity(" hello 你好 world")
    }

    func testEmojiAndGraphemeClustersByteFallBack() {
        assertParity("😊")
        // A ZWJ sequence is one Swift Character but several scalars; the split is per character, so
        // the whole cluster falls back together.
        assertParity("👩‍💻")
        assertParity("hello 😊 world")
    }

    func testMergeResultOutsideTheVocabularyByteFallsBack() {
        let v = Self.fixture
        // `q`+`q` merges to a piece the vocabulary lacks, so the merged symbol byte-falls-back.
        XCTAssertEqual(assertParity("qq"),
                       [v.id("<0x71>"), v.id("<0x71>")])
        XCTAssertEqual(assertParity("qqq"),
                       [v.id("<0x71>"), v.id("<0x71>"), v.id("q")])
        assertParity("qq hello")
    }

    // MARK: - Special tokens

    func testSpecialTokensAloneAndAdjacentToText() {
        let v = Self.fixture
        XCTAssertEqual(assertParity("<bos>"), [v.id("<bos>")])
        XCTAssertEqual(assertParity("<|turn>"), [v.id("<|turn>")])
        XCTAssertEqual(assertParity("<bos>hello"), [v.id("<bos>"), v.id("hello")])
        XCTAssertEqual(assertParity("hello<|turn>"), [v.id("hello"), v.id("<|turn>")])
        XCTAssertEqual(assertParity("<bos><|turn><turn|>"),
                       [v.id("<bos>"), v.id("<|turn>"), v.id("<turn|>")])
        assertParity("<|turn>user\nhello<turn|>\n")
        assertParity(" hello <|turn> world ")
        assertParity("a<|turn>b<turn|>c")
    }

    func testTextThatLooksLikeASpecialTokenButIsNot() {
        assertParity("<turn>")
        assertParity("<|turn")
        assertParity("< bos >")
    }

    // MARK: - Mixed text

    func testMixedScriptSentences() {
        for text in [
            " привет world",
            "hello привет 你好 😊",
            "abab qq > </ xyz",
            " hello  world\n привет\t你好 ",
        ] {
            assertParity(text)
        }
    }

    func testLongInputMatchesReference() {
        let unit = " hello  world привет 你好 abab > </ xyz qq 😊\n"
        var text = ""
        while text.count < 4_000 { text += unit }
        assertParity(text)
        // And every prefix boundary, since the merge queue's state is carried across a whole run.
        for length in [1, 7, 64, 512, 2_048] {
            assertParity(String(text.prefix(length)))
        }
    }

    // MARK: - Merge list encodings

    func testStringAndPairMergeEncodingsAgree() throws {
        let alternate = Gemma4Tokenizer()
        try alternate.load(from: Self.stringMergesFixture.directory)
        for text in [" hello  world", "abab", "> </", "привет 你好", "<bos>hello<|turn>"] {
            XCTAssertEqual(alternate.encode(text), tokenizer.encode(text),
                           "merge-list encoding changed the ids for \(text.debugDescription)")
        }
    }
}
