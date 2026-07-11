import Foundation

/// SupertonicTTS G2P-free text front-end: **NFKD + regex cleanup + `<lang>…</lang>` wrap +
/// codepoint→token-id table lookup**. No phonemizer, no IPA, no espeak.
///
/// Faithful Swift port of `Supertone/supertonic` `py/helper.py::UnicodeProcessor` (validated in
/// `speech-models/stmodels/infer.py`) and the C++ `SupertonicTokenizer`. NFKD — the keystone that
/// decomposes ä → a +◌̈ and Hangul into in-vocab jamo — is Foundation's
/// `decomposedStringWithCompatibilityMapping`.
public struct SupertonicTokenizer: Sendable {
    /// Token id for codepoints absent from the indexer table.
    public static let unknownId: Int32 = -1

    /// `Supertone/supertonic` AVAILABLE_LANGS (32 entries; includes "na", excludes "zh").
    public static let availableLangs: Set<String> = [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
        "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
        "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
        "vi", "na",
    ]

    /// `indexer[codepoint] = id`, -1 if unsupported. Flat array of 65536 ints.
    private let indexer: [Int32]

    public init(indexer: [Int32]) { self.indexer = indexer }

    /// Load `unicode_indexer.json` — a flat JSON array of 65536 ints.
    public static func load(from url: URL) throws -> SupertonicTokenizer {
        let data = try Data(contentsOf: url)
        guard let raw = try JSONSerialization.jsonObject(with: data) as? [Any] else {
            throw SupertonicError.badAsset("unicode_indexer.json must be a flat JSON array")
        }
        var indexer = [Int32](); indexer.reserveCapacity(raw.count)
        for v in raw { indexer.append(Int32(truncatingIfNeeded: (v as? NSNumber)?.intValue ?? -1)) }
        return SupertonicTokenizer(indexer: indexer)
    }

    public func supports(_ lang: String) -> Bool { Self.availableLangs.contains(lang) }

    @inline(__always)
    private func lookup(_ cp: UInt32) -> Int32 {
        cp < UInt32(indexer.count) ? indexer[Int(cp)] : Self.unknownId
    }

    // MARK: - cleanup

    private static let emojiRanges: [ClosedRange<UInt32>] = [
        0x1F600...0x1F64F, 0x1F300...0x1F5FF, 0x1F680...0x1F6FF, 0x1F700...0x1F77F,
        0x1F780...0x1F7FF, 0x1F800...0x1F8FF, 0x1F900...0x1F9FF, 0x1FA00...0x1FA6F,
        0x1FA70...0x1FAFF, 0x2600...0x26FF, 0x2700...0x27BF, 0x1F1E6...0x1F1FF,
    ]

    private static func isEmoji(_ v: UInt32) -> Bool { emojiRanges.contains { $0.contains(v) } }

    /// helper.py::_char_repl + the `[♥☆♡©\]` strip, on a single scalar. Returns the replacement
    /// scalar(s), or nil to drop it.
    private static func charRepl(_ s: Unicode.Scalar) -> Unicode.Scalar? {
        switch s.value {
        case 0x2013, 0x2011, 0x2014: return "-"            // – ‑ —
        case 0x5F:                   return " "            // _
        case 0x201C, 0x201D:         return "\""           // “ ”
        case 0x2018, 0x2019:         return "'"            // ‘ ’
        case 0x00B4, 0x60:           return "'"            // ´ `
        case 0x5B, 0x5D, 0x7C, 0x2F, 0x23: return " "      // [ ] | / #
        case 0x2192, 0x2190:         return " "            // → ←
        case 0x2665, 0x2606, 0x2661, 0x00A9, 0x5C: return nil  // ♥ ☆ ♡ © backslash
        default:                     return s
        }
    }

    // helper.py terminal set: [.!?;:,'")\]}…。」』】〉》›»]
    private static let terminalSet: Set<UInt32> = Set(
        ".!?;:,'\")]}".unicodeScalars.map { $0.value } +
        [0x2026, 0x3002, 0x300D, 0x300F, 0x3011, 0x3009, 0x300B, 0x203A, 0x00BB])

    /// NFKD + cleanup + `<lang>` wrap. Throws on unsupported language.
    func preprocess(_ text: String, lang: String) throws -> String {
        // 1) NFKD.
        let nfkd = text.decomposedStringWithCompatibilityMapping

        // 2) emoji removal + char-level map/strip.
        var scalars = String.UnicodeScalarView()
        for s in nfkd.unicodeScalars {
            if Self.isEmoji(s.value) { continue }
            if let r = Self.charRepl(s) { scalars.append(r) }
        }
        var out = String(scalars)

        // 3) expression replacements.
        out = out.replacingOccurrences(of: "@", with: " at ")
        out = out.replacingOccurrences(of: "e.g.,", with: "for example, ")
        out = out.replacingOccurrences(of: "i.e.,", with: "that is, ")

        // 4) drop a space before punctuation.
        for p in [",", ".", "!", "?", ";", ":", "'"] {
            out = out.replacingOccurrences(of: " " + p, with: p)
        }

        // 5) collapse repeated quotes.
        while out.contains("\"\"") { out = out.replacingOccurrences(of: "\"\"", with: "\"") }
        while out.contains("''")   { out = out.replacingOccurrences(of: "''", with: "'") }
        while out.contains("``")   { out = out.replacingOccurrences(of: "``", with: "`") }

        // 6) collapse whitespace + trim.
        var collapsed = String.UnicodeScalarView()
        var prevWs = false
        for s in out.unicodeScalars {
            let ws = s == " " || s == "\t" || s == "\n" || s == "\r"
                || s.value == 0x0B || s.value == 0x0C
            if ws { if !prevWs { collapsed.append(" ") }; prevWs = true }
            else { collapsed.append(s); prevWs = false }
        }
        var trimmed = Array(collapsed)
        while trimmed.first == " " { trimmed.removeFirst() }
        while trimmed.last == " " { trimmed.removeLast() }

        // 7) ensure terminal punctuation.
        if let last = trimmed.last, !Self.terminalSet.contains(last.value) {
            trimmed.append(".")
        } else if trimmed.isEmpty {
            trimmed.append(".")
        }
        let cleaned = String(String.UnicodeScalarView(trimmed))

        // 8) validate + wrap.
        guard supports(lang) else { throw SupertonicError.unsupportedLanguage(lang) }
        return "<\(lang)>\(cleaned)</\(lang)>"
    }

    // MARK: - tokenize

    public struct Tokens: Sendable {
        public let ids: [Int32]    // length == textLength (zero-padded)
        public let mask: [Float]   // length == textLength (1.0 real, 0.0 pad)
    }

    /// Full front-end for one chunk: preprocess → per-codepoint lookup → right-pad to `textLength`.
    func process(_ text: String, lang: String, textLength: Int) throws -> Tokens {
        let wrapped = try preprocess(text, lang: lang)
        var ids = [Int32](repeating: 0, count: textLength)
        var mask = [Float](repeating: 0, count: textLength)
        var i = 0
        for s in wrapped.unicodeScalars {
            if i >= textLength { break }
            ids[i] = lookup(s.value)
            mask[i] = 1.0
            i += 1
        }
        return Tokens(ids: ids, mask: mask)
    }

    // MARK: - chunking

    /// Split free-form text into per-synthesis chunks that fit the fixed text length after the
    /// `<lang>` wrap. CoreML latent length L is dynamic (RangeDim), so only the text axis bounds us.
    func chunk(_ text: String, lang: String, textLength: Int) -> [String] {
        var cap = textLength - (2 * lang.count + 5) - 1
        if cap < 8 { cap = 8 }

        let cps = Array(text.unicodeScalars)
        let term: Set<UInt32> = [0x2E, 0x21, 0x3F, 0x2026, 0x3002, 0xFF01, 0xFF1F]
        func isWs(_ s: Unicode.Scalar) -> Bool { s == " " || s == "\n" || s == "\t" || s == "\r" }

        // sentence-ish split at terminal punctuation + following whitespace.
        var sentences: [[Unicode.Scalar]] = []
        var cur: [Unicode.Scalar] = []
        for (i, s) in cps.enumerated() {
            cur.append(s)
            if term.contains(s.value), i + 1 < cps.count, isWs(cps[i + 1]) {
                sentences.append(cur); cur = []
            }
        }
        if !cur.isEmpty { sentences.append(cur) }

        var out: [String] = []
        var chunk: [Unicode.Scalar] = []
        func flush() {
            var a = 0, b = chunk.count
            while a < b, chunk[a] == " " { a += 1 }
            while b > a, chunk[b - 1] == " " { b -= 1 }
            if b > a { out.append(String(String.UnicodeScalarView(chunk[a..<b]))) }
            chunk.removeAll(keepingCapacity: true)
        }
        func fits(_ n: Int) -> Bool { chunk.count + (chunk.isEmpty ? 0 : 1) + n <= cap }
        func appendUnit(_ u: ArraySlice<Unicode.Scalar>) {
            if !chunk.isEmpty { chunk.append(" ") }
            chunk.append(contentsOf: u)
        }

        for sent in sentences {
            if sent.count <= cap {
                if !fits(sent.count) { flush() }
                appendUnit(sent[...])
                continue
            }
            flush()  // oversize sentence: hard-split on words
            var word: [Unicode.Scalar] = []
            func pushWord() {
                if word.isEmpty { return }
                if !fits(word.count) { flush() }
                appendUnit(word[...]); word.removeAll(keepingCapacity: true)
            }
            for s in sent {
                if isWs(s) { pushWord(); continue }
                word.append(s)
                if word.count >= cap { pushWord() }
            }
            pushWord()
        }
        flush()
        return out.isEmpty ? [""] : out
    }
}
