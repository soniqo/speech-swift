import Foundation
import NaturalLanguage

/// Mandarin Chinese G2P for Magpie's `mandarin_phoneme` tokeniser.
///
/// **Note on duplication:** `KokoroTTS/ChinesePhonemizer.swift` also wraps
/// Apple's `.mandarinToLatin` transform but lacks word-level segmentation
/// (it walks chars) and emits IPA against Kokoro's vocab — different
/// target format. We keep this implementation independent so the bundled
/// `cmudict_pinyin_zh.txt` (Magpie-specific) stays local.
///
///
/// NeMo's `ChineseG2p` pipeline is roughly:
///   1. jieba word-segment input text
///   2. Convert each Hanzi to pinyin (with tone numbers) via pypinyin
///   3. Look up each pinyin syllable in the bundled IPA dict
///      (`cmudict_pinyin_zh.txt`) — e.g. `BA → "p a"`
///   4. Append `#N` tone marker after each syllable
///
/// We approximate the pipeline with Apple frameworks:
///   1. `NLTokenizer(unit: .word)` with `.simplifiedChinese` for word
///      segmentation (jieba equivalent — handles e.g. "世界" as one word
///      instead of two characters with potentially wrong readings).
///   2. `applyingTransform(.mandarinToLatin)` for word → tone-marked
///      pinyin (CLDR table picks the *contextual* reading per word, so
///      multi-character compounds get the correct syllables).
///   3. Bundled `cmudict_pinyin_zh.txt` for pinyin syllable → IPA token
///      list.
///   4. Extract the tone from the syllable's diacritic and append `#N`.
public final class MagpieChineseG2P {

    public static let shared = MagpieChineseG2P()

    private var dict: [String: String] = [:]
    private var loaded = false
    private let loadLock = NSLock()

    private init() {}

    public func ensureLoaded() throws {
        loadLock.lock()
        defer { loadLock.unlock() }
        if loaded { return }
        guard let url = Bundle.module.url(forResource: "cmudict_pinyin_zh",
                                            withExtension: "txt") else {
            throw MagpieTTSError.missingFile("cmudict_pinyin_zh.txt (resource bundle)")
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        for raw in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix(";;;") { continue }
            // Format: `PINYIN_SYLLABLE<TAB>IPA tokens space separated`.
            let parts = line.split(separator: "\t", omittingEmptySubsequences: true)
            if parts.count < 2 { continue }
            dict[String(parts[0])] = String(parts[1])
        }
        loaded = true
    }

    /// Hanzi+pinyin+punctuation → grouped phoneme stream.
    ///
    /// Returns a list of phoneme groups. Each group's phonemes get
    /// **concatenated as adjacent tokens** in the model input (no space
    /// between, matching NeMo `ChinesePhonemesTokenizer.encode_from_g2p`).
    /// Groups themselves are separated by space in the final ID sequence.
    public func phonemize(_ text: String) -> [[String]] {
        do { try ensureLoaded() } catch { return [] }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.simplifiedChinese)

        var groups: [[String]] = []
        var cursor = text.startIndex
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            // Capture any punctuation / whitespace between the previous
            // word and this one as their own single-token groups so the
            // model sees commas / periods / spaces at the right positions.
            if cursor < range.lowerBound {
                for scalar in text[cursor..<range.lowerBound].unicodeScalars {
                    if !CharacterSet.whitespaces.contains(scalar) {
                        groups.append([String(scalar)])
                    }
                }
            }
            cursor = range.upperBound

            let word = String(text[range])
            if let pinyin = word.applyingTransform(.mandarinToLatin, reverse: false) {
                // A single word may emit multiple pinyin syllables separated
                // by spaces, e.g. "世界" → "shì jiè" (two syllables).
                for syllable in pinyin.split(separator: " ", omittingEmptySubsequences: true) {
                    let cleaned = syllable.filter(\.isLetter) + syllable.filter { !$0.isLetter && !$0.isWhitespace }
                    if cleaned.isEmpty { continue }
                    groups.append(phonemesForSyllable(String(syllable)))
                }
            } else {
                // Non-Han content — punctuation, Latin runs, digits.
                for scalar in word.unicodeScalars where !CharacterSet.whitespaces.contains(scalar) {
                    groups.append([String(scalar)])
                }
            }
            return true
        }
        // Trailing punctuation after the last word.
        if cursor < text.endIndex {
            for scalar in text[cursor..<text.endIndex].unicodeScalars {
                if !CharacterSet.whitespaces.contains(scalar) {
                    groups.append([String(scalar)])
                }
            }
        }
        return groups
    }

    /// Convert one accented-vowel pinyin syllable to the `[ipa_phoneme...,
    /// "#<tone>"]` token list NeMo's ChinesePhonemesTokenizer expects.
    private func phonemesForSyllable(_ syllable: String) -> [String] {
        let tone = extractTone(syllable)
        let stripped = stripDiacritics(syllable).uppercased()
        guard let ipa = dict[stripped] else {
            // Fallback: emit the raw stripped form as its own token so the
            // model sees *something* rather than `<oov>`.
            return [stripped.lowercased()]
        }
        var phonemes = ipa.split(separator: " ").map(String.init)
        if tone > 0 { phonemes.append("#\(tone)") }
        return phonemes
    }

    /// Detect Mandarin tone via the combining mark on the syllable's vowel.
    /// Returns 1–4 for tones, 5 for neutral, 0 if no tone mark found.
    private func extractTone(_ syllable: String) -> Int {
        // Apply NFD so combining marks appear separately.
        let decomposed = syllable.decomposedStringWithCanonicalMapping
        for scalar in decomposed.unicodeScalars {
            switch scalar.value {
            case 0x0304: return 1   // ̄ (macron)
            case 0x0301: return 2   // ́ (acute)
            case 0x030C: return 3   // ̌ (caron)
            case 0x0300: return 4   // ̀ (grave)
            default: continue
            }
        }
        return 0
    }

    private func stripDiacritics(_ s: String) -> String {
        s.applyingTransform(.stripDiacritics, reverse: false) ?? s
    }
}
