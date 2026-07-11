import XCTest
@testable import MagpieTTS

final class MagpieTokenizerTests: XCTestCase {
    /// Synthesize a tiny vocab config and exercise the char-level tokenizer.
    private func makeTokenizer(language: MagpieLanguage = .english,
                               padWithSpace: Bool = false) -> MagpieTokenizer {
        let tokens = ["<pad>", "<oov>", " ", "a", "b", "c", "!", ".", "h", "e", "l", "o"]
        let sub = MagpieTokenizerConfig.MagpieTokenizerSubConfig(
            punct: true, apostrophe: true, padWithSpace: padWithSpace,
            pretrainedModel: nil)
        let cfg = MagpieTokenizerConfig(
            language: language.rawValue,
            tokenizerName: "test",
            type: "ipa_phoneme",
            vocabSize: tokens.count,
            tokens: tokens,
            bosId: nil, eosId: nil, padId: 0,
            config: sub)
        return MagpieTokenizer(language: language, config: cfg)
    }

    /// Every `tokenize` call now appends `eos_id = vocab_size + 1` to mirror
    /// NeMo's round-trip pipeline (the Magpie text-embedding has 2 extra
    /// rows for BOS/EOS past the bundle vocab; without EOS the AR loop
    /// fails to terminate naturally). All assertions below account for that.
    private func eosId(_ tok: MagpieTokenizer) -> Int { tok.eosId }

    func testCharLookup() {
        // `prephonemized: true` bypasses the English CMU G2P (which would
        // expand "abc" → "ABC" → "ˈeɪˌbiˌsi" since it's in the dict).
        let tok = makeTokenizer()
        XCTAssertEqual(tok.tokenize("abc", prephonemized: true),
                       [3, 4, 5, eosId(tok)])
    }

    func testOOVFallback() {
        let tok = makeTokenizer()
        let ids = tok.tokenize("axc", prephonemized: true)
        XCTAssertEqual(ids[0], 3)
        XCTAssertEqual(ids[1], 1)  // <oov>
        XCTAssertEqual(ids[2], 5)
        XCTAssertEqual(ids.last, eosId(tok))
    }

    func testPadWithSpaceWraps() {
        let tok = makeTokenizer(padWithSpace: true)
        let ids = tok.tokenize("abc", prephonemized: true)
        XCTAssertEqual(ids.first, 2)  // space pad before content
        XCTAssertEqual(ids.last, eosId(tok))  // EOS after trailing space
        XCTAssertEqual(ids[ids.count - 2], 2)  // trailing space pad
    }

    func testPunctuationLookedUp() {
        let tok = makeTokenizer()
        XCTAssertEqual(tok.tokenize("hello!", prephonemized: true),
                       [8, 9, 10, 10, 11, 6, eosId(tok)])
    }

    func testEmptyInput() {
        let tok = makeTokenizer()
        // Empty input still emits a single EOS so the AR decoder gets a
        // well-formed (non-empty) sequence.
        XCTAssertEqual(tok.tokenize(""), [eosId(tok)])
    }

    func testEosIdMatchesShape() {
        let tok = makeTokenizer()
        // Synthetic vocab is 12 tokens; expected eos = vocab_size + 1.
        XCTAssertEqual(tok.eosId, 12 + 1)
        XCTAssertEqual(tok.bosId, 12)
    }
}
