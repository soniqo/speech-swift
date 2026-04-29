import Foundation
import NaturalLanguage
import AudioCommon

/// Result of preprocessing text for forced alignment
public struct SlottedText: Sendable {
    /// Token IDs with timestamp tokens inserted around each word
    public let tokenIds: [Int]
    /// Indices within tokenIds that are timestamp tokens
    public let timestampPositions: [Int]
    /// Surface forms of the words (one per timestamp pair). Adjacent
    /// punctuation is preserved here so callers can reconstruct sentences;
    /// the model itself only sees the punctuation-stripped form.
    public let words: [String]
}

/// Surface + cleaned form of a single word emitted by tokenization.
/// `surface` keeps adjacent punctuation; `cleaned` is what the model tokenizer
/// sees (letters, numbers, and combining marks only).
struct WordPair: Sendable {
    var surface: String
    let cleaned: String
}

/// Language-specific text preprocessing for forced alignment.
///
/// Three paths:
/// - Japanese → morpheme-level segmentation
/// - Korean   → word-level segmentation
/// - Other    → whitespace split + per-Han-ideograph break
///
/// Tokens are filtered to keep only Unicode Letters / Numbers and the ASCII
/// apostrophe — punctuation, symbols, and marks are stripped before
/// timestamp slots are inserted. Han-ideograph splitting is restricted to
/// CJK Unified + Extensions A–E + Compatibility; hiragana, katakana, and
/// Hangul are NOT split per character (the model emits timestamp slots
/// between morphemes for those scripts, not between every kana or jamo).
///
/// Punctuation that surrounds a word (commas, periods, brackets, CJK
/// `，。！？` etc.) is preserved on the `surface` form so subtitle / SRT
/// pipelines can still split on sentence boundaries after alignment.
public enum TextPreprocessor {

    /// Split text into words and insert timestamp slots for alignment.
    ///
    /// For each word, inserts `<timestamp><timestamp>` pairs so the model
    /// can predict start/end timestamps at those positions.
    public static func prepareForAlignment(
        text: String,
        tokenizer: Qwen3Tokenizer,
        language: String = "English"
    ) -> SlottedText {
        let pairs = splitIntoWordPairs(text, language: language)
        let tsId = Qwen3ASRTokens.timestampTokenId

        var tokenIds: [Int] = []
        var timestampPositions: [Int] = []
        var validWords: [String] = []

        for pair in pairs {
            let wordTokens = tokenizer.encode(pair.cleaned)
            guard !wordTokens.isEmpty else {
                // Cleaned form unencodable: attach surface to previous word
                // so we don't drop punctuation that anchored to it.
                if !validWords.isEmpty {
                    validWords[validWords.count - 1] += pair.surface
                }
                continue
            }

            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            tokenIds.append(contentsOf: wordTokens)

            timestampPositions.append(tokenIds.count)
            tokenIds.append(tsId)

            validWords.append(pair.surface)
        }

        return SlottedText(
            tokenIds: tokenIds,
            timestampPositions: timestampPositions,
            words: validWords
        )
    }

    /// Split text into words using the language-appropriate strategy.
    /// Returned strings are the cleaned (punctuation-stripped) forms,
    /// intended for callers that don't care about surface preservation.
    static func splitIntoWords(_ text: String, language: String) -> [String] {
        return splitIntoWordPairs(text, language: language).map { $0.cleaned }
    }

    /// Split text into (surface, cleaned) pairs. Surface keeps adjacent
    /// punctuation; cleaned is what the model tokenizer sees.
    static func splitIntoWordPairs(_ text: String, language: String) -> [WordPair] {
        let lang = language.lowercased()

        if lang.contains("japanese") || lang == "ja" {
            return nlTokenizePairs(text, language: .japanese)
        }
        if lang.contains("korean") || lang == "ko" {
            return nlTokenizePairs(text, language: .korean)
        }
        // Scripts without word-level whitespace where Apple's NLTokenizer
        // provides native segmentation. Without these dispatches the
        // default whitespace path collapses each sentence to one token.
        if let nlLang = nlLanguageForUnspaced(lang) {
            return nlTokenizePairs(text, language: nlLang)
        }
        return tokenizeSpaceLangPairs(text)
    }

    private static func nlLanguageForUnspaced(_ lang: String) -> NLLanguage? {
        if lang.contains("thai")     || lang == "th" { return .thai }
        if lang.contains("lao")      || lang == "lo" { return .lao }
        if lang.contains("khmer")    || lang == "km" { return .khmer }
        if lang.contains("burmese")  || lang.contains("myanmar") || lang == "my" { return .burmese }
        if lang.contains("tibetan")  || lang == "bo" { return .tibetan }
        return nil
    }

    // MARK: - Japanese / Korean / unspaced scripts

    /// Apple's `NLTokenizer` reports word ranges (without surrounding
    /// punctuation). We attach trailing non-letter, non-whitespace
    /// characters between consecutive word ranges to the preceding word's
    /// surface so commas, full-width periods, etc. ride along.
    private static func nlTokenizePairs(_ text: String, language: NLLanguage) -> [WordPair] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.setLanguage(language)
        tokenizer.string = text

        var ranges: [Range<String.Index>] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            ranges.append(range)
            return true
        }

        var pairs: [WordPair] = []
        for (i, range) in ranges.enumerated() {
            let cleaned = cleanToken(String(text[range]))
            guard !cleaned.isEmpty else { continue }
            var surface = String(text[range])

            let nextStart = i + 1 < ranges.count ? ranges[i + 1].lowerBound : text.endIndex
            var idx = range.upperBound
            while idx < nextStart {
                let ch = text[idx]
                if ch.isWhitespace { break }
                let allPunct = String(ch).unicodeScalars.allSatisfy { !isKeptScalar($0) }
                if !allPunct { break }
                surface.append(ch)
                idx = text.index(after: idx)
            }
            pairs.append(WordPair(surface: surface, cleaned: cleaned))
        }
        return pairs
    }

    // MARK: - Default path (whitespace + per-Han break)

    /// Whitespace split, then per-Han-ideograph break inside each segment.
    /// Default path for languages with reliable word-level whitespace —
    /// English, European languages, Hindi, Arabic, Vietnamese, Mongolian,
    /// Indonesian, etc. Chinese works because most Chinese text has no
    /// whitespace, so each ideograph becomes its own token via the per-Han
    /// break inside the (single) whitespace-bounded segment.
    static func tokenizeSpaceLangPairs(_ text: String) -> [WordPair] {
        var pairs: [WordPair] = []
        for raw in text.split(whereSeparator: \.isWhitespace) {
            let segment = String(raw)
            let segPairs = pairsForSegment(segment)
            // Reattach a leading whitespace separator to the first new pair
            // so the surface reads naturally when concatenated. We don't
            // actually reinsert spaces — callers can join with spaces — but
            // we do rejoin punctuation that ended up segment-leading with
            // no anchor word (rare, e.g. a stray "—") to the previous word.
            if segPairs.isEmpty {
                if !pairs.isEmpty {
                    pairs[pairs.count - 1].surface += segment
                }
                continue
            }
            pairs.append(contentsOf: segPairs)
        }
        return pairs
    }

    /// Convert one whitespace-bounded segment to (surface, cleaned) pairs.
    /// For non-Han segments: a single pair with surface = segment, cleaned
    /// = letters/numbers only. For segments containing Han ideographs:
    /// each Han is its own pair; consecutive non-Han runs become their own
    /// pair if they contain any letter/number, or attach to the
    /// neighbouring pair's surface if they are pure punctuation/symbols.
    private static func pairsForSegment(_ seg: String) -> [WordPair] {
        let hasHan = seg.unicodeScalars.contains(where: isHanIdeograph)
        if !hasHan {
            let cleaned = cleanToken(seg)
            if cleaned.isEmpty { return [] }
            return [WordPair(surface: seg, cleaned: cleaned)]
        }
        var pairs: [WordPair] = []
        var nonHanBuf = ""

        func flushNonHan(beforeHan: Bool) {
            guard !nonHanBuf.isEmpty else { return }
            let cleaned = cleanToken(nonHanBuf)
            if cleaned.isEmpty {
                // Pure punctuation: attach to the previous pair's trailing
                // surface. If we're at the start with no previous pair,
                // leave the buffer for the upcoming Han to absorb.
                if !pairs.isEmpty {
                    pairs[pairs.count - 1].surface += nonHanBuf
                    nonHanBuf = ""
                } else if !beforeHan {
                    // Trailing pure-punct with no anchor at all — drop.
                    nonHanBuf = ""
                }
                return
            }
            pairs.append(WordPair(surface: nonHanBuf, cleaned: cleaned))
            nonHanBuf = ""
        }

        for scalar in seg.unicodeScalars {
            if isHanIdeograph(scalar) {
                flushNonHan(beforeHan: true)
                let han = String(scalar)
                if !nonHanBuf.isEmpty {
                    // Leading pure-punct waiting for a Han anchor.
                    pairs.append(WordPair(surface: nonHanBuf + han, cleaned: han))
                    nonHanBuf = ""
                } else {
                    pairs.append(WordPair(surface: han, cleaned: han))
                }
            } else {
                nonHanBuf.append(Character(scalar))
            }
        }
        flushNonHan(beforeHan: false)
        return pairs
    }

    // MARK: - Legacy entry points (kept for tests / external callers)

    static func tokenizeJapanese(_ text: String) -> [String] {
        return nlTokenizePairs(text, language: .japanese).map { $0.cleaned }
    }

    static func tokenizeKorean(_ text: String) -> [String] {
        return nlTokenizePairs(text, language: .korean).map { $0.cleaned }
    }

    static func tokenizeSpaceLang(_ text: String) -> [String] {
        return tokenizeSpaceLangPairs(text).map { $0.cleaned }
    }

    // MARK: - Cleaning + classification

    /// Keep only Unicode Letters (`L*`), Numbers (`N*`), and ASCII apostrophe.
    /// Strips punctuation (e.g. the full-width period `。`), symbols,
    /// separators, and marks.
    static func cleanToken(_ token: String) -> String {
        var out = ""
        for scalar in token.unicodeScalars {
            if isKeptScalar(scalar) {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    private static func isKeptScalar(_ scalar: Unicode.Scalar) -> Bool {
        if scalar == "'" { return true }
        let cat = scalar.properties.generalCategory
        switch cat {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
             .modifierLetter, .otherLetter,
             .decimalNumber, .letterNumber, .otherNumber,
             // Combining marks are essential for scripts like Thai, Lao,
             // Khmer, Burmese, Tibetan, Devanagari, Bengali, Arabic harakat
             // — stripping them mangles the word (e.g. "สวัสดี" → "สวสด").
             .nonspacingMark, .spacingMark, .enclosingMark:
            return true
        default:
            return false
        }
    }

    /// Han ideograph ranges only. Notably **excludes** hiragana
    /// (0x3040–0x309F), katakana (0x30A0–0x30FF), and Hangul syllables
    /// (0xAC00–0xD7AF), which are handled by the language-specific
    /// tokenizers (Japanese / Korean) instead.
    static func isHanIdeograph(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        if v >= 0x4E00 && v <= 0x9FFF   { return true }  // CJK Unified
        if v >= 0x3400 && v <= 0x4DBF   { return true }  // Extension A
        if v >= 0x20000 && v <= 0x2A6DF { return true }  // Extension B
        if v >= 0x2A700 && v <= 0x2B73F { return true }  // Extension C
        if v >= 0x2B740 && v <= 0x2B81F { return true }  // Extension D
        if v >= 0x2B820 && v <= 0x2CEAF { return true }  // Extension E
        if v >= 0xF900 && v <= 0xFAFF   { return true }  // Compatibility
        return false
    }
}
