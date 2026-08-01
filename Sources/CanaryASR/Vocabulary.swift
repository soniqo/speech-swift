import Foundation

/// SentencePiece vocabulary for Canary.
///
/// Loads the bundle's `vocab.json` — a flat map from token id to piece — and
/// joins pieces back into text, with `▁` (U+2581) marking a word boundary.
///
/// Only id → piece is exposed on purpose. The reverse direction is not safe
/// here: this is an aggregate tokenizer, so an ordinary piece appears once per
/// sub-tokenizer and a lookup by text can return any of them. Everything that
/// needs a specific token — the prompt, end-of-text, the language tags — takes
/// its id from `config.json` instead.
public struct CanaryVocabulary: Sendable {
    private let idToToken: [Int: String]

    /// Number of tokens in the vocabulary.
    public var count: Int { idToToken.count }

    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
    }

    /// Load from a `vocab.json` of `{"0": "<piece>", …}`.
    public static func load(from url: URL) throws -> CanaryVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        var mapping: [Int: String] = [:]
        mapping.reserveCapacity(raw.count)
        for (key, value) in raw {
            if let id = Int(key) { mapping[id] = value }
        }
        return CanaryVocabulary(idToToken: mapping)
    }

    /// Piece for an id, or nil when the id is out of range.
    public func token(_ id: Int) -> String? { idToToken[id] }

    /// Join tokens into text, dropping control tokens and turning the
    /// SentencePiece marker into a space.
    public func decode(_ ids: [Int]) -> String {
        var text = ""
        for id in ids {
            guard let piece = idToToken[id], !piece.isEmpty else { continue }
            if piece.hasPrefix("<|") { continue }
            text += piece
        }
        return text
            .replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }
}
