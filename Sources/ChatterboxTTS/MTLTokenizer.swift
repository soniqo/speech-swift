import Foundation

/// Special tokens of the Chatterbox multilingual grapheme tokenizer.
public enum ChatterboxToken {
    public static let start = "[START]"
    public static let stop = "[STOP]"
    public static let unk = "[UNK]"
    public static let space = "[SPACE]"
}

public enum ChatterboxTokenizerError: Error {
    case malformedTokenizerJSON
}

/// Multilingual grapheme tokenizer for Chatterbox — a self-contained Swift port
/// of the reference `MTLTokenizer`. The encode pipeline for Latin-script languages
/// plus Arabic and Hindi is: lowercase → NFKD → prepend the `[lang]` token →
/// replace spaces with `[SPACE]` → BPE.
///
/// The BPE faithfully reproduces the HF `tokenizer.json` (model `BPE`,
/// pre-tokenizer `Whitespace`, no normalizer, `[UNK]` fallback): added/control
/// tokens are extracted by longest match, gaps are split with the `\w+|[^\w\s]+`
/// Whitespace rule, and each word is merged by merge-rank. It's implemented here
/// (rather than via swift-transformers' generic tokenizer) so token ids match the
/// reference implementation exactly.
///
/// Chinese, Japanese, Hebrew, Korean and Russian additionally need a
/// language-specific text frontend (Cangjie / kakasi / diacritics / Jamo /
/// stress) that is **not yet ported** — see `frontendFreeLanguages`.
public final class MTLTokenizer {
    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let mergeRank: [String: Int]
    /// (content, id) of added/control tokens, sorted longest content first so the
    /// scan is greedy-longest-match like the HF tokenizer.
    private let addedTokens: [(token: [Character], id: Int)]
    private let unkId: Int

    /// Languages whose encode path needs no special frontend, fully supported today.
    public static let frontendFreeLanguages: Set<String> =
        ["en", "ar", "hi", "de", "es", "fr", "it", "pt"]

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
        var s = text.lowercased()
        // NFKD — Swift's compatibility decomposition matches Python `normalize("NFKD", …)`.
        s = s.decomposedStringWithCompatibilityMapping
        if let lang = languageId, !lang.isEmpty {
            s = "[\(lang.lowercased())]" + s
        }
        s = s.replacingOccurrences(of: " ", with: ChatterboxToken.space)
        return tokenize(s)
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
