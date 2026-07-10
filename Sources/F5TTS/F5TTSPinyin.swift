import CoreFoundation
import Foundation

/// Mandarin text frontend for F5-TTS.
///
/// Upstream F5 converts Chinese text to TONE3 pinyin with rjieba word
/// segmentation plus pypinyin (`tone_sandhi=True`) applied per word. This
/// runtime replaces that with longest-match lookup over an exported lexicon
/// (`pinyin_lexicon.tsv` in the model bundle) whose multi-char entries have
/// tone sandhi baked in at export time, so no tone rules run here. Characters
/// missing from the lexicon fall back to the system Mandarin-Latin transform.
public struct F5TTSPinyinConverter: Sendable {
    public let entries: [String: [String]]
    public let maxKeyLength: Int

    public init(entries: [String: [String]]) {
        self.entries = entries
        self.maxKeyLength = entries.keys.map(\.count).max() ?? 1
    }

    public init(lexiconURL: URL) throws {
        let text: String
        do {
            text = try String(contentsOf: lexiconURL, encoding: .utf8)
        } catch {
            throw F5TTSError.invalidConfig(lexiconURL, error.localizedDescription)
        }
        var entries: [String: [String]] = [:]
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            if line.hasPrefix("#") { continue }
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else { continue }
            entries[String(parts[0])] = parts[1].split(separator: " ").map(String.init)
        }
        guard !entries.isEmpty else {
            throw F5TTSError.invalidConfig(lexiconURL, "empty pinyin lexicon")
        }
        self.init(entries: entries)
    }

    /// Matches upstream `is_chinese`: the range checked by F5's frontend.
    public static func isHanzi(_ ch: Character) -> Bool {
        guard let scalar = ch.unicodeScalars.first else { return false }
        return (0x3100...0x9FFF).contains(Int(scalar.value))
    }

    /// Converts normalized text into F5 frontend tokens: each hanzi becomes
    /// `" "` + a TONE3 pinyin syllable, ASCII words keep their characters with
    /// upstream's boundary-space rule, and other characters pass through.
    public func convert(_ text: String) -> [String] {
        var out: [String] = []
        out.reserveCapacity(text.count * 2)
        let chars = Array(text)
        var index = 0
        while index < chars.count {
            let kind = Self.kind(of: chars[index])
            var end = index
            while end < chars.count, Self.kind(of: chars[end]) == kind {
                end += 1
            }
            let segment = chars[index..<end]
            switch kind {
            case .hanzi:
                appendHanzi(segment, to: &out)
            case .ascii:
                appendASCII(segment, to: &out)
            case .other:
                out.append(contentsOf: segment.map(String.init))
            }
            index = end
        }
        return out
    }

    private enum CharKind {
        case hanzi
        case ascii
        case other
    }

    private static func kind(of ch: Character) -> CharKind {
        if isHanzi(ch) { return .hanzi }
        if let value = ch.unicodeScalars.first?.value, value < 256 { return .ascii }
        return .other
    }

    private func appendHanzi(_ segment: ArraySlice<Character>, to out: inout [String]) {
        let chars = Array(segment)
        var i = 0
        while i < chars.count {
            var matched: [String]?
            var matchedLength = 1
            let longest = min(maxKeyLength, chars.count - i)
            if longest >= 2 {
                for length in stride(from: longest, through: 2, by: -1) {
                    let candidate = String(chars[i..<(i + length)])
                    if let syllables = entries[candidate] {
                        matched = syllables
                        matchedLength = length
                        break
                    }
                }
            }
            if matched == nil {
                let single = String(chars[i])
                matched = entries[single] ?? [Self.systemPinyin(single) ?? single]
            }
            for syllable in matched ?? [] {
                out.append(" ")
                out.append(syllable)
            }
            i += matchedLength
        }
    }

    private func appendASCII(_ segment: ArraySlice<Character>, to out: inout [String]) {
        var word: [Character] = []
        func flushWord() {
            guard !word.isEmpty else { return }
            if !out.isEmpty, word.count > 1,
               let last = out.last, !(last.count == 1 && " :'\"".contains(last)) {
                out.append(" ")
            }
            out.append(contentsOf: word.map(String.init))
            word.removeAll(keepingCapacity: true)
        }
        for ch in segment {
            if ch == " " {
                flushWord()
                out.append(" ")
            } else {
                word.append(ch)
            }
        }
        flushWord()
    }

    /// System Mandarin-Latin transform as an out-of-lexicon fallback,
    /// converted from tone marks to TONE3.
    static func systemPinyin(_ hanzi: String) -> String? {
        let mutable = NSMutableString(string: hanzi)
        guard CFStringTransform(mutable, nil, kCFStringTransformMandarinLatin, false),
              (mutable as String) != hanzi else {
            return nil
        }
        return toneMarkToTone3(mutable as String)
    }

    static func toneMarkToTone3(_ syllable: String) -> String {
        var base = ""
        var tone = 0
        for ch in syllable.precomposedStringWithCanonicalMapping {
            if let (plain, mark) = Self.toneTable[ch] {
                base.append(plain)
                if mark > 0 { tone = mark }
            } else {
                base.append(ch)
            }
        }
        return tone > 0 ? base + String(tone) : base
    }

    private static let toneTable: [Character: (Character, Int)] = [
        "ā": ("a", 1), "á": ("a", 2), "ǎ": ("a", 3), "à": ("a", 4),
        "ē": ("e", 1), "é": ("e", 2), "ě": ("e", 3), "è": ("e", 4),
        "ī": ("i", 1), "í": ("i", 2), "ǐ": ("i", 3), "ì": ("i", 4),
        "ō": ("o", 1), "ó": ("o", 2), "ǒ": ("o", 3), "ò": ("o", 4),
        "ū": ("u", 1), "ú": ("u", 2), "ǔ": ("u", 3), "ù": ("u", 4),
        "ǖ": ("v", 1), "ǘ": ("v", 2), "ǚ": ("v", 3), "ǜ": ("v", 4),
        "ü": ("v", 0),
        "ń": ("n", 2), "ň": ("n", 3), "ǹ": ("n", 4),
        "ḿ": ("m", 2),
        "ê": ("e", 0),
    ]
}
