import XCTest
@testable import MagpieTTS

/// Regression tests for the NeMo-parity fixes to the Chinese frontend:
/// the phoneme stream is flat (no space token between syllables or words),
/// neutral-tone syllables carry an explicit "#5" marker, and ASCII letters
/// map to the uppercase-only zh sub-vocab. Inserting a space between every
/// syllable group put ~half the sequence out of distribution and made the
/// model ignore text conditioning entirely (free-running babble).
final class MagpieZhTokenizationTests: XCTestCase {

    /// Every Han syllable group must end with a tone marker — including
    /// neutral-tone syllables, which Apple's transform leaves unmarked and
    /// NeMo renders as "#5" (`neutral_tone_with_five`).
    func testHanGroupsEndWithToneMarkers() {
        let tones: Set<String> = ["#1", "#2", "#3", "#4", "#5"]
        for text in ["你好世界", "好吗"] {
            let groups = MagpieChineseG2P.shared.phonemize(text)
            XCTAssertFalse(groups.isEmpty, "no groups for \(text)")
            for group in groups {
                XCTAssertFalse(group.contains(" "),
                               "unexpected space group in \(text): \(groups)")
                XCTAssertTrue(tones.contains(group.last ?? ""),
                              "group missing tone marker for \(text): \(group)")
            }
        }
    }

    /// ASCII letters inside Chinese text map to uppercase (the zh sub-vocab
    /// carries A–Z only) and whitespace collapses to a single space group.
    func testAsciiLettersUppercasedAndWhitespaceCollapsed() {
        let groups = MagpieChineseG2P.shared.phonemize("你好  Ab")
        XCTAssertEqual(groups.filter { $0 == [" "] }.count, 1, "\(groups)")
        XCTAssertTrue(groups.contains(["A"]), "\(groups)")
        XCTAssertTrue(groups.contains(["B"]), "\(groups)")
    }

    /// The encoded ID stream must contain no space tokens between syllable
    /// groups — NeMo's ChinesePhonemesTokenizer emits the flat G2P stream
    /// verbatim. (pad_with_space is disabled in this fixture so any space
    /// id in the output is a between-group regression.)
    func testTokenizeEmitsNoInterSyllableSpaces() throws {
        var tokens = (0..<349).map { "filler\($0)" }
        tokens += [" ", "a", "ai", "au", "e", "ei", "i", "j", "n", "o", "u",
                   "w", "x", "y", "ʂ", "tɕ", "ʈʂ", "ɤ", "ə",
                   "#1", "#2", "#3", "#4", "#5",
                   "A", "B", "，", "。", "<pad>", "<oov>"]
        let json: [String: Any] = [
            "language": "zh", "tokenizer_name": "mandarin_phoneme",
            "type": "mandarin_pinyin", "vocab_size": tokens.count,
            "bos_id": NSNull(), "eos_id": NSNull(), "pad_id": NSNull(),
            "config": ["punct": true, "pad_with_space": false],
            "tokens": tokens,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        let cfg = try JSONDecoder().decode(MagpieTokenizerConfig.self, from: data)
        let tok = MagpieTokenizer(language: .chinese, config: cfg)
        let spaceId = tok.tokenToId[" "]!

        let ids = tok.tokenize("你好世界")
        XCTAssertFalse(ids.dropLast().contains(spaceId),
                       "space token between syllable groups: \(ids)")
        XCTAssertGreaterThan(ids.count, 4, "suspiciously short: \(ids)")
    }
}
