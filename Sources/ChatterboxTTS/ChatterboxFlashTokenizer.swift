#if canImport(CoreML)
import Foundation

public final class ChatterboxFlashTokenizer: @unchecked Sendable {
    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let mergeRank: [String: Int]
    private let addedTokens: [(token: [Character], id: Int)]
    private let unkId: Int

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

    public init(tokenizerURL: URL) throws {
        let data = try Data(contentsOf: tokenizerURL)
        let tokenizer = try JSONDecoder().decode(TokenizerJSON.self, from: data)
        self.vocab = tokenizer.model.vocab
        var idToToken: [Int: String] = [:]
        idToToken.reserveCapacity(tokenizer.model.vocab.count)
        for (token, id) in tokenizer.model.vocab {
            idToToken[id] = token
        }
        self.idToToken = idToToken

        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(tokenizer.model.merges.count)
        for (rank, merge) in tokenizer.model.merges.enumerated() {
            ranks[merge] = rank
        }
        self.mergeRank = ranks

        let unkToken = tokenizer.model.unk_token ?? "[UNK]"
        self.unkId = tokenizer.model.vocab[unkToken] ?? 1
        self.addedTokens = (tokenizer.added_tokens ?? [])
            .sorted { $0.content.count > $1.content.count }
            .map { (token: Array($0.content), id: $0.id) }
    }

    public convenience init(directory: URL) throws {
        try self.init(tokenizerURL: directory.appendingPathComponent("t3/tokenizer.json"))
    }

    public func encode(
        _ text: String,
        addSpecialTokens: Bool = true,
        config: ChatterboxFlashT3Config = .fallback
    ) -> [Int] {
        let ids = tokenize(text)
        guard addSpecialTokens else { return ids }
        return [config.startTextToken] + ids + [config.stopTextToken]
    }

    public func encodePadded(_ text: String, config: ChatterboxFlashT3Config) throws -> [Int32] {
        let ids = encode(text, addSpecialTokens: true, config: config)
        guard ids.count <= config.textLen else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration(
                "text encodes to \(ids.count) tokens, exceeding exported text_len \(config.textLen)"
            )
        }
        return (ids + Array(repeating: config.stopTextToken, count: config.textLen - ids.count)).map(Int32.init)
    }

    public func decode(_ ids: [Int]) -> String {
        ids.map { idToToken[$0] ?? "" }.joined()
    }

    private func tokenize(_ text: String) -> [Int] {
        let chars = Array(text)
        var ids: [Int] = []
        var buffer: [Character] = []

        func flush() {
            guard !buffer.isEmpty else { return }
            for word in whitespaceSplit(String(buffer)) {
                ids.append(contentsOf: bpe(word))
            }
            buffer.removeAll(keepingCapacity: true)
        }

        var index = 0
        while index < chars.count {
            var matched = false
            for (token, id) in addedTokens where !token.isEmpty && index + token.count <= chars.count {
                if Array(chars[index ..< index + token.count]) == token {
                    flush()
                    ids.append(id)
                    index += token.count
                    matched = true
                    break
                }
            }
            if !matched {
                buffer.append(chars[index])
                index += 1
            }
        }
        flush()
        return ids
    }

    private func whitespaceSplit(_ text: String) -> [String] {
        let ns = text as NSString
        let matches = Self.whitespaceRegex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        return matches.map { ns.substring(with: $0.range) }
    }

    private static let whitespaceRegex = try! NSRegularExpression(pattern: "\\w+|[^\\w\\s]+")

    private func bpe(_ word: String) -> [Int] {
        var symbols = word.unicodeScalars.map { String($0) }
        guard symbols.count > 1 else {
            return symbols.map { vocab[$0] ?? unkId }
        }
        while symbols.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for index in 0 ..< symbols.count - 1 {
                if let rank = mergeRank[symbols[index] + " " + symbols[index + 1]], rank < bestRank {
                    bestRank = rank
                    bestIndex = index
                }
            }
            if bestIndex < 0 { break }
            symbols[bestIndex] += symbols[bestIndex + 1]
            symbols.remove(at: bestIndex + 1)
        }
        return symbols.map { vocab[$0] ?? unkId }
    }
}
#endif
