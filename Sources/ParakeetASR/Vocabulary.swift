import AudioCommon
import Foundation

/// SentencePiece-based vocabulary for Parakeet TDT.
///
/// Loads a `vocab.json` mapping from token ID strings to token strings,
/// and decodes sequences using SentencePiece conventions (`▁` → space).
public struct ParakeetVocabulary: Sendable {
    /// Mapping from token ID to token string.
    private let idToToken: [Int: String]

    /// Number of tokens in the vocabulary (excluding blank).
    public var count: Int { idToToken.count }

    /// Token IDs of language tags (`<|xx|>` for ISO codes, id >= 24 so control tokens like
    /// `<|pnc|>` / `<|timestamp|>` / `<|predict_lang|>` at 0..23 are excluded), keyed by
    /// lowercase language code. Used to steer greedy decoding to a chosen language by masking
    /// the others.
    public let languageTagIds: [String: Int]

    /// Initialize from a pre-loaded dictionary.
    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
        self.languageTagIds = Self.extractLanguageTags(from: idToToken)
    }

    /// Pull `<|xx|>` language tags out of the vocab. Control tokens (`<|pnc|>`, `<|emo:...|>`,
    /// `<|predict_lang|>`, …) all sit below id 24, so an id + shape filter isolates the
    /// per-language tags cleanly.
    private static func extractLanguageTags(from idToToken: [Int: String]) -> [String: Int] {
        var map = [String: Int]()
        for (id, token) in idToToken where id >= 24 {
            guard token.hasPrefix("<|"), token.hasSuffix("|>") else { continue }
            let code = token.dropFirst(2).dropLast(2)
            guard (2...3).contains(code.count), code.allSatisfy({ $0.isLetter && $0.isLowercase })
            else { continue }
            map[String(code)] = id
        }
        return map
    }

    /// Load vocabulary from a `vocab.json` file.
    ///
    /// Expected format: `{"0": "▁the", "1": "▁a", ...}`
    public static func load(from url: URL) throws -> ParakeetVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)

        var mapping = [Int: String]()
        mapping.reserveCapacity(raw.count)
        for (key, value) in raw {
            guard let id = Int(key) else { continue }
            mapping[id] = value
        }

        return ParakeetVocabulary(idToToken: mapping)
    }

    /// Decode a sequence of token IDs into text.
    ///
    /// Applies SentencePiece conventions:
    /// - `▁` (U+2581) at the start of a token becomes a space
    /// - Leading space on the final result is trimmed
    public func decode(_ tokenIds: [Int]) -> String {
        var pieces = [String]()
        pieces.reserveCapacity(tokenIds.count)

        for id in tokenIds {
            guard let token = idToToken[id] else { continue }
            // Replace SentencePiece word-boundary marker with space
            let text = token.replacingOccurrences(of: "\u{2581}", with: " ")
            pieces.append(text)
        }

        let joined = pieces.joined()
        // Trim leading space that comes from the first token's ▁ prefix
        return joined.trimmingCharacters(in: .whitespaces)
    }

    /// Decode token IDs into words with per-word confidence scores.
    ///
    /// Groups consecutive tokens into words using SentencePiece `▁` boundaries.
    /// Each word's confidence is exp(mean log-prob of its tokens), clamped to 0–1.
    public func decodeWords(_ tokenIds: [Int], logProbs: [Float]) -> [WordConfidence] {
        guard tokenIds.count == logProbs.count else {
            return [WordConfidence(word: decode(tokenIds), confidence: 0)]
        }

        var words = [WordConfidence]()
        var currentWord = ""
        var currentLogProbs = [Float]()

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }

            let startsNewWord = token.hasPrefix("\u{2581}")
            let text = token.replacingOccurrences(of: "\u{2581}", with: "")

            if startsNewWord && !currentWord.isEmpty {
                // Flush previous word
                let meanLP = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                words.append(WordConfidence(word: currentWord, confidence: min(1.0, exp(meanLP))))
                currentWord = ""
                currentLogProbs = []
            }

            currentWord += text
            currentLogProbs.append(logProbs[i])
        }

        // Flush last word
        if !currentWord.isEmpty {
            let meanLP = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
            words.append(WordConfidence(word: currentWord, confidence: min(1.0, exp(meanLP))))
        }

        return words
    }
}
