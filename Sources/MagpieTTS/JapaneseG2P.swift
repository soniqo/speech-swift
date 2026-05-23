import Foundation
import CoreFoundation

/// Japanese G2P for Magpie's `japanese_phoneme` tokeniser.
///
/// **Note on duplication:** `KokoroTTS/JapanesePhonemizer.swift` shares the
/// CFStringTokenizer-based segmentation pattern, but produces a different
/// output format (IPA against Kokoro's vocab vs katakana+pitch against
/// Magpie's vocab). The two are kept separate so each TTS can evolve with
/// its model's specific requirements; if a third caller appears the
/// CFStringTokenizer wrapper is a candidate for extraction to
/// `AudioCommon`.
///
/// NeMo's reference pipeline (`JapaneseKatakanaAccentG2p`) needs MeCab +
/// pyopenjtalk to read kanji *and* extract per-word pitch-accent values.
/// The model is then fed `[pitch, mora, pitch, mora, …]` where pitch is
/// `'0'` (low) or `'1'` (high) preceding every mora.
///
/// We replicate as much as Apple's frameworks allow:
///
///   1. Word segmentation + kanji → romaji via `CFStringTokenizer` with
///      `kCFStringTokenizerAttributeLatinTranscription` (gives proper
///      Japanese readings, e.g. 世界 → "sekai" not Mandarin "shì jiè").
///   2. Romaji → katakana via `applyingTransform(.latinToKatakana)`.
///   3. Mora-split with NeMo's regex
///      `[ア-ンヴ][ャュョァィゥェォヮ]?|[ァィゥェォヵヶッャュョヮ]|ー`.
///   4. **Pitch fallback**: without pyopenjtalk we can't look up the real
///      accent, so we apply the *heiban* (acc=0) pattern (L-H-H…-H) to
///      every word chain. Heiban is the single most common Japanese
///      accent class and the model degrades the most gracefully on it.
///
/// Output: phoneme groups ready for `lookupGroups(_:)` — each group is
/// `[pitch_marker, mora_kana...]` so the pitch and its mora are emitted
/// as adjacent tokens in the final ID stream (matching NeMo).
public enum MagpieJapaneseG2P {

    // MARK: - Particle / greeting overrides
    //
    // Apple's `CFStringTokenizer` doesn't run a grammatical parser, so it
    // returns the literal reading for every kana — e.g. the topic particle
    // は always comes back as "ha" instead of the spoken "wa".
    // pyopenjtalk (the NeMo reference) knows particles. We replicate the
    // most-common cases with two small tables:
    //
    //   - `wholeWordOverrides`: greetings / fixed phrases whose final
    //     particle is fused into the word ("こんにちは" → コンニチワ).
    //   - `particleOverrides`: standalone single-char tokens that Apple
    //     reads literally (は ha → ワ wa, へ he → エ e, を wo → オ o).
    //
    // The lists are intentionally short — adding more entries is a safe
    // followup, and the model degrades gracefully if a particle is read
    // literally.

    private static let wholeWordOverrides: [String: String] = [
        "こんにちは": "コンニチワ",
        "こんばんは": "コンバンワ",
    ]

    private static let particleOverrides: [String: String] = [
        "は": "ワ",
        "へ": "エ",
        "を": "オ",
    ]

    /// Returns a flat list of phoneme tokens (pitch markers + mora kana +
    /// punctuation + literal " " entries) ready for vocab lookup.
    ///
    /// Matches NeMo `JapaneseKatakanaAccentG2p.__call__`: pitch markers
    /// precede each mora within a word chain; explicit `' '` tokens appear
    /// only where the source text has whitespace; punctuation passes
    /// through as its own single token. We do **not** add space between
    /// consecutive content words — NeMo only inserts space at explicit
    /// whitespace tokens.
    public static func phonemes(_ text: String) -> [String] {
        let cfText = text as CFString
        let len = CFStringGetLength(cfText)
        let tokenizer = CFStringTokenizerCreate(
            nil, cfText, CFRangeMake(0, len),
            kCFStringTokenizerUnitWord, Locale(identifier: "ja_JP") as CFLocale)

        var out: [String] = []
        // Track text position so we can grab whitespace/punctuation runs
        // between tokens (CFStringTokenizer skips whitespace in UnitWord).
        var cursor = text.startIndex
        var tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while tokenType != [] {
            let cfRange = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let nsRange = NSRange(location: cfRange.location, length: cfRange.length)
            guard let swiftRange = Range(nsRange, in: text) else {
                tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                continue
            }

            // Gap before this token → whitespace (insert space) or
            // CJK/ASCII punctuation (insert verbatim).
            if cursor < swiftRange.lowerBound {
                for ch in text[cursor..<swiftRange.lowerBound] {
                    if ch.isWhitespace {
                        out.append(" ")
                    } else {
                        out.append(String(ch))
                    }
                }
            }
            cursor = swiftRange.upperBound

            let raw = String(text[swiftRange])
            if raw.trimmingCharacters(in: .whitespaces).isEmpty {
                tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                continue
            }

            // Particle / greeting override paths.
            if let override = wholeWordOverrides[raw] {
                appendWord(katakana: override, into: &out)
                tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                continue
            }
            if let override = particleOverrides[raw] {
                appendWord(katakana: override, into: &out)
                tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                continue
            }

            // Word-like token: read kanji/kana via Apple's romaji attribute
            // then transform romaji → katakana for mora-splitting.
            if let romaji = CFStringTokenizerCopyCurrentTokenAttribute(
                tokenizer, kCFStringTokenizerAttributeLatinTranscription) as? String,
               !romaji.isEmpty {
                let katakana = romajiToKatakana(romaji)
                let moras = splitMoras(katakana)
                if !moras.isEmpty {
                    appendWord(moras: moras, into: &out)
                    tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                    continue
                }
            }
            // Fallback: emit codepoints directly.
            for scalar in raw.unicodeScalars { out.append(String(scalar)) }
            tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        // Trailing punctuation past the last word.
        if cursor < text.endIndex {
            for ch in text[cursor..<text.endIndex] {
                if ch.isWhitespace { out.append(" ") }
                else { out.append(String(ch)) }
            }
        }
        return out
    }

    /// Append `[0, ka, 1, ta, 1, ka, …]` to `out` from a katakana string
    /// by splitting into moras and applying the heiban pitch pattern.
    private static func appendWord(katakana: String, into out: inout [String]) {
        appendWord(moras: splitMoras(katakana), into: &out)
    }
    private static func appendWord(moras: [String], into out: inout [String]) {
        let pitches = heibanPattern(count: moras.count)
        for (pitch, mora) in zip(pitches, moras) {
            out.append(pitch)
            out.append(mora)
        }
    }

    /// Romaji → katakana via Apple's transform, then strip the Mandarin
    /// tone marks (̀ ́ ̄ ̌) that the transform leaves behind on syllables
    /// derived from kanji read via the Mandarin CLDR table. We must NOT
    /// strip Japanese dakuten / handakuten (゛ ゜) because those are
    /// phonemically meaningful (デ vs テ, ゴ vs コ). NFC composition fuses
    /// dakuten into the precomposed katakana glyphs first, leaving only
    /// the Latin tone marks behind as standalone combining scalars; we
    /// remove those by codepoint range.
    private static func romajiToKatakana(_ romaji: String) -> String {
        guard let katakana = romaji.applyingTransform(.latinToKatakana, reverse: false) else {
            return romaji
        }
        // NFC composes 'デ' = 'テ' + '゙' into the single codepoint 0x30C7.
        let composed = katakana.precomposedStringWithCanonicalMapping
        // Drop the remaining Latin tone combining marks (Mandarin pinyin
        // leftovers). Range 0x0300–0x036F is "Combining Diacritical Marks".
        var out = String.UnicodeScalarView()
        out.reserveCapacity(composed.unicodeScalars.count)
        for scalar in composed.unicodeScalars {
            if scalar.value >= 0x0300 && scalar.value <= 0x036F { continue }
            out.append(scalar)
        }
        return String(out)
    }

    /// Split a katakana string into morae using NeMo's reference regex:
    /// `[ア-ンヴ][ャュョァィゥェォヮ]?|[ァィゥェォヵヶッャュョヮ]|ー`.
    private static func splitMoras(_ katakana: String) -> [String] {
        let smallFollow: Set<Character> = [
            "ャ", "ュ", "ョ", "ァ", "ィ", "ゥ", "ェ", "ォ", "ヮ"
        ]
        let standaloneSmall: Set<Character> = [
            "ァ", "ィ", "ゥ", "ェ", "ォ", "ヵ", "ヶ", "ッ", "ャ", "ュ", "ョ", "ヮ"
        ]
        var moras: [String] = []
        let chars = Array(katakana)
        var i = 0
        while i < chars.count {
            let ch = chars[i]
            if ch == "ー" {
                moras.append("ー")
                i += 1
                continue
            }
            // Main katakana ア-ン or ヴ optionally followed by a small kana.
            let inMain = (ch >= "ア" && ch <= "ン") || ch == "ヴ"
            if inMain {
                var mora = String(ch)
                if i + 1 < chars.count, smallFollow.contains(chars[i + 1]) {
                    mora.append(chars[i + 1])
                    i += 2
                } else {
                    i += 1
                }
                moras.append(mora)
                continue
            }
            if standaloneSmall.contains(ch) {
                moras.append(String(ch))
                i += 1
                continue
            }
            // Anything else (punctuation that slipped through, non-katakana)
            // is emitted as its own single-char mora so the loop terminates.
            moras.append(String(ch))
            i += 1
        }
        return moras
    }

    /// Heiban (`acc=0`) pitch pattern: L-H-H-H… for N moras → `["0", "1", "1", …]`.
    private static func heibanPattern(count: Int) -> [String] {
        if count == 0 { return [] }
        if count == 1 { return ["0"] }
        return ["0"] + Array(repeating: "1", count: count - 1)
    }
}
