import Foundation
import CoreFoundation
import NaturalLanguage

/// Special tokens of the Chatterbox multilingual grapheme tokenizer.
public enum ChatterboxToken {
    public static let start = "[START]"
    public static let stop = "[STOP]"
    public static let unk = "[UNK]"
    public static let space = "[SPACE]"
}

public enum ChatterboxTokenizerError: Error, LocalizedError {
    case malformedTokenizerJSON
    case missingCangjieMap
    case hebrewRequiresDiacritics

    public var errorDescription: String? {
        switch self {
        case .malformedTokenizerJSON:
            return "Malformed Chatterbox tokenizer.json"
        case .missingCangjieMap:
            return "Chatterbox tokenizer is missing Cangjie5_TC.json"
        case .hebrewRequiresDiacritics:
            return "Chatterbox Hebrew input must include niqqud/diacritics; automatic diacritization is not bundled yet"
        }
    }
}

/// Multilingual grapheme tokenizer for Chatterbox — a self-contained Swift port
/// of the reference `MTLTokenizer`. The common encode pipeline is:
/// lowercase → NFKD → prepend the `[lang]` token → replace spaces with
/// `[SPACE]` → BPE.
///
/// The BPE faithfully reproduces the HF `tokenizer.json` (model `BPE`,
/// pre-tokenizer `Whitespace`, no normalizer, `[UNK]` fallback): added/control
/// tokens are extracted by longest match, gaps are split with the `\w+|[^\w\s]+`
/// Whitespace rule, and each word is merged by merge-rank. It's implemented here
/// (rather than via swift-transformers' generic tokenizer) so token ids match the
/// reference implementation exactly.
///
/// Some writing systems additionally need a language-specific text frontend
/// before this grapheme path. Runtime language support is intentionally limited
/// to languages whose frontend is implemented here; unsupported upstream tags
/// stay visible as metadata but are not accepted by synthesis.
public final class MTLTokenizer {
    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let mergeRank: [String: Int]
    private let cangjie: [String: String]
    private let cangjieDuplicateIndex: [String: Int]
    /// (content, id) of added/control tokens, sorted longest content first so the
    /// scan is greedy-longest-match like the HF tokenizer.
    private let addedTokens: [(token: [Character], id: Int)]
    private let unkId: Int

    /// Languages published by Resemble Chatterbox Multilingual V3.
    public static let upstreamLanguages: Set<String> = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh",
    ]

    /// Languages whose text frontend is implemented locally without falling
    /// back to raw text. Hebrew is accepted only when the caller supplies
    /// pre-diacritized text; automatic Dicta ONNX diacritization is not ported.
    public static let supportedLanguages: Set<String> = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw",
        "tr", "zh",
    ]

    /// Backwards-compatible alias retained for callers that used this as a
    /// runtime language gate.
    public static let frontendFreeLanguages: Set<String> = supportedLanguages

    private struct TokenizerJSON: Decodable {
        struct Model: Decodable {
            let vocab: [String: Int]
            let merges: [String]
            let unk_token: String?
        }
        struct Added: Decodable {
            let id: Int
            let content: String
        }
        let model: Model
        let added_tokens: [Added]?
    }

    /// Load from a model folder that contains `tokenizer.json` (the Chatterbox MLX
    /// bundle ships one).
    public init(modelFolder: URL) throws {
        let data = try Data(contentsOf: modelFolder.appendingPathComponent("tokenizer.json"))
        guard let tj = try? JSONDecoder().decode(TokenizerJSON.self, from: data) else {
            throw ChatterboxTokenizerError.malformedTokenizerJSON
        }
        self.vocab = tj.model.vocab
        var i2t = [Int: String](minimumCapacity: tj.model.vocab.count)
        for (t, id) in tj.model.vocab { i2t[id] = t }
        self.idToToken = i2t

        var ranks = [String: Int](minimumCapacity: tj.model.merges.count)
        for (rank, merge) in tj.model.merges.enumerated() { ranks[merge] = rank }
        self.mergeRank = ranks
        let cangjieMap = try Self.loadCangjieMap(from: modelFolder)
        self.cangjie = cangjieMap.codes
        self.cangjieDuplicateIndex = cangjieMap.duplicateIndex

        let unk = tj.model.unk_token ?? ChatterboxToken.unk
        self.unkId = tj.model.vocab[unk] ?? 1

        let added = (tj.added_tokens ?? [])
            .sorted { $0.content.count > $1.content.count }
            .map { (token: Array($0.content), id: $0.id) }
        self.addedTokens = added
    }

    /// Encode text to token ids. `languageId` (e.g. "en", "ar", "hi") is prepended
    /// as a `[lang]` control token, matching the multilingual model's training.
    public func encode(_ text: String, languageId: String? = nil) -> [Int] {
        let lang = languageId?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var s = text.lowercased()
        if lang == "ja" {
            s = Self.normalizeJapaneseToHiragana(s)
            // Japanese reference normalization applies NFKD after hiragana conversion.
            s = s.decomposedStringWithCompatibilityMapping
        } else {
            // NFKD — Swift's compatibility decomposition matches Python `normalize("NFKD", …)`.
            s = s.decomposedStringWithCompatibilityMapping
        }
        if lang == "zh", !cangjie.isEmpty {
            s = applyCangjieFrontend(Self.segmentChinese(s))
        } else if lang == "ko" {
            s = Self.decomposeHangulToJamo(s)
        }
        if let lang, !lang.isEmpty {
            s = "[\(lang)]" + s
        }
        s = s.replacingOccurrences(of: " ", with: ChatterboxToken.space)
        return tokenize(s)
    }

    /// Strict encode path used by synthesis. It preserves `encode`'s legacy
    /// behaviour for tests/debugging, but rejects Hebrew text that has not been
    /// pre-diacritized; upstream relies on Dicta before tokenization for this.
    public func encodeStrict(_ text: String, languageId: String? = nil) throws -> [Int] {
        let lang = languageId?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if lang == "he",
           Self.containsHebrewLetters(text),
           !Self.containsHebrewDiacritics(text) {
            throw ChatterboxTokenizerError.hebrewRequiresDiacritics
        }
        return encode(text, languageId: languageId)
    }

    /// Inverse of `encode` for debugging.
    public func decode(_ ids: [Int]) -> String {
        var s = ids.map { idToToken[$0] ?? "" }.joined()
        s = s.replacingOccurrences(of: ChatterboxToken.space, with: " ")
        for t in [ChatterboxToken.start, ChatterboxToken.stop, ChatterboxToken.unk] {
            s = s.replacingOccurrences(of: t, with: "")
        }
        return s
    }

    // MARK: - Internals

    private static func loadCangjieMap(from modelFolder: URL) throws -> (
        codes: [String: String],
        duplicateIndex: [String: Int]
    ) {
        let url = modelFolder.appendingPathComponent("Cangjie5_TC.json")
        guard let data = try? Data(contentsOf: url),
              let rows = try? JSONDecoder().decode([String].self, from: data) else {
            throw ChatterboxTokenizerError.missingCangjieMap
        }
        var codes: [String: String] = [:]
        var glyphsByCode: [String: [String]] = [:]
        codes.reserveCapacity(rows.count)
        for row in rows {
            let parts = row.split(separator: "\t", omittingEmptySubsequences: false)
            guard parts.count >= 2 else { continue }
            let glyph = String(parts[0])
            let code = String(parts[1])
            if codes[glyph] == nil {
                codes[glyph] = code
                glyphsByCode[code, default: []].append(glyph)
            }
        }
        var duplicateIndex: [String: Int] = [:]
        for glyphs in glyphsByCode.values {
            for (index, glyph) in glyphs.enumerated() where index > 0 {
                duplicateIndex[glyph] = index
            }
        }
        return (codes, duplicateIndex)
    }

    private func applyCangjieFrontend(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.count * 4)
        for character in text.map(String.init) {
            guard let code = cangjie[character] else {
                out += character
                continue
            }
            let indexedCode = if let index = cangjieDuplicateIndex[character] {
                "\(code)\(index)"
            } else {
                code
            }
            for scalar in indexedCode.unicodeScalars {
                out += "[cj_\(String(scalar))]"
            }
            out += "[cj_.]"
        }
        return out
    }

    private static func segmentChinese(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        tokenizer.setLanguage(.simplifiedChinese)

        var tokens: [String] = []
        var cursor = text.startIndex

        func appendGap(_ gap: Substring) {
            for scalar in gap.unicodeScalars where !CharacterSet.whitespacesAndNewlines.contains(scalar) {
                tokens.append(String(scalar))
            }
        }

        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            if cursor < range.lowerBound {
                appendGap(text[cursor..<range.lowerBound])
            }
            tokens.append(String(text[range]))
            cursor = range.upperBound
            return true
        }
        if cursor < text.endIndex {
            appendGap(text[cursor..<text.endIndex])
        }
        return tokens.isEmpty ? text : tokens.joined(separator: " ")
    }

    private static func normalizeJapaneseToHiragana(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        let cfText = text as CFString
        let tokenizer = CFStringTokenizerCreate(
            nil,
            cfText,
            CFRangeMake(0, CFStringGetLength(cfText)),
            kCFStringTokenizerUnitWord,
            Locale(identifier: "ja_JP") as CFLocale
        )

        var out = ""
        var cursor = text.startIndex
        var tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        while tokenType != [] {
            let cfRange = CFStringTokenizerGetCurrentTokenRange(tokenizer)
            let nsRange = NSRange(location: cfRange.location, length: cfRange.length)
            guard let swiftRange = Range(nsRange, in: text) else {
                tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
                continue
            }
            if cursor < swiftRange.lowerBound {
                out += text[cursor..<swiftRange.lowerBound]
            }
            let raw = String(text[swiftRange])
            if Self.containsKanji(raw),
               let latin = CFStringTokenizerCopyCurrentTokenAttribute(
                    tokenizer,
                    kCFStringTokenizerAttributeLatinTranscription
               ) as? String,
               let hiragana = latin.applyingTransform(.latinToHiragana, reverse: false),
               !hiragana.isEmpty {
                if let first = hiragana.first, first == "は" || first == "へ" {
                    out += " "
                }
                out += hiragana
            } else {
                out += raw
            }
            cursor = swiftRange.upperBound
            tokenType = CFStringTokenizerAdvanceToNextToken(tokenizer)
        }
        if cursor < text.endIndex {
            out += text[cursor..<text.endIndex]
        }
        return out
    }

    private static func containsKanji(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
        }
    }

    public static func containsHebrewLetters(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x05D0...0x05EA).contains(scalar.value)
        }
    }

    public static func containsHebrewDiacritics(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            switch scalar.value {
            case 0x0591...0x05BD, 0x05BF, 0x05C1...0x05C2, 0x05C4...0x05C5, 0x05C7:
                return true
            default:
                return false
            }
        }
    }

    public static func isHebrewTextDiacritized(_ text: String) -> Bool {
        !containsHebrewLetters(text) || containsHebrewDiacritics(text)
    }

    private static func decomposeHangulToJamo(_ text: String) -> String {
        var out = ""
        for scalar in text.unicodeScalars {
            let value = scalar.value
            guard (0xAC00...0xD7A3).contains(value) else {
                out.unicodeScalars.append(scalar)
                continue
            }
            let base = value - 0xAC00
            let initial = 0x1100 + base / (21 * 28)
            let medial = 0x1161 + (base % (21 * 28)) / 28
            let final = base % 28
            out.unicodeScalars.append(UnicodeScalar(initial)!)
            out.unicodeScalars.append(UnicodeScalar(medial)!)
            if final > 0 {
                out.unicodeScalars.append(UnicodeScalar(0x11A7 + final)!)
            }
        }
        return out
    }

    /// Split the preprocessed string by added/control tokens (longest match), then
    /// pre-tokenize each gap with the Whitespace rule and BPE each word.
    private func tokenize(_ s: String) -> [Int] {
        let chars = Array(s)
        var ids: [Int] = []
        var buffer: [Character] = []

        func flush() {
            guard !buffer.isEmpty else { return }
            for word in whitespaceSplit(String(buffer)) {
                ids.append(contentsOf: bpe(word))
            }
            buffer.removeAll(keepingCapacity: true)
        }

        var i = 0
        while i < chars.count {
            var matched = false
            for (tok, id) in addedTokens where !tok.isEmpty && i + tok.count <= chars.count {
                if Array(chars[i ..< i + tok.count]) == tok {
                    flush()
                    ids.append(id)
                    i += tok.count
                    matched = true
                    break
                }
            }
            if !matched {
                buffer.append(chars[i])
                i += 1
            }
        }
        flush()
        return ids
    }

    /// The HF `Whitespace` pre-tokenizer: `\w+|[^\w\s]+` (Unicode-aware), dropping
    /// whitespace. Separates words from runs of punctuation.
    private func whitespaceSplit(_ s: String) -> [String] {
        let ns = s as NSString
        let matches = Self.whitespaceRegex.matches(
            in: s, range: NSRange(location: 0, length: ns.length))
        return matches.map { ns.substring(with: $0.range) }
    }

    private static let whitespaceRegex = try! NSRegularExpression(pattern: "\\w+|[^\\w\\s]+")

    /// Byte-pair encoding over Unicode characters using merge ranks; OOV symbols
    /// fall back to `[UNK]`.
    private func bpe(_ word: String) -> [Int] {
        // Split on Unicode scalars (codepoints), not Swift Characters — the BPE
        // vocab is keyed per codepoint, and Devanagari/Arabic combining marks
        // would otherwise be glued into one grapheme cluster and miss the vocab.
        var symbols = word.unicodeScalars.map { String($0) }
        guard symbols.count > 1 else {
            return symbols.map { vocab[$0] ?? unkId }
        }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0 ..< symbols.count - 1 {
                if let r = mergeRank[symbols[i] + " " + symbols[i + 1]], r < bestRank {
                    bestRank = r
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            symbols[bestIdx] += symbols[bestIdx + 1]
            symbols.remove(at: bestIdx + 1)
        }
        return symbols.map { vocab[$0] ?? unkId }
    }
}
