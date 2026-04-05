import Foundation
import AudioCommon

/// SentencePiece vocabulary for Parakeet EOU (1026 tokens + blank).
public struct ParakeetEOUVocabulary: Sendable {
    private let idToToken: [Int: String]

    public var count: Int { idToToken.count }

    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
    }

    /// Load vocabulary from `vocab.json` — format: `{"0": "▁the", "1": "▁a", ...}`
    public static func load(from url: URL) throws -> ParakeetEOUVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        var mapping: [Int: String] = [:]
        for (key, value) in raw {
            if let id = Int(key) {
                mapping[id] = value
            }
        }
        return ParakeetEOUVocabulary(idToToken: mapping)
    }

    /// Decode token IDs to text. Strips EOU/EOB/blank tokens.
    public func decode(_ tokenIds: [Int]) -> String {
        var text = ""
        for id in tokenIds {
            guard let token = idToToken[id] else { continue }
            text += token
        }
        // SentencePiece: ▁ → space, trim leading space
        return text.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// Decode with per-word confidence scores.
    public func decodeWords(_ tokenIds: [Int], logProbs: [Float]) -> [WordConfidence] {
        guard tokenIds.count == logProbs.count else { return [] }

        var words: [WordConfidence] = []
        var currentWord = ""
        var currentLogProbs: [Float] = []

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }
            let isWordStart = token.hasPrefix("▁") && !currentWord.isEmpty

            if isWordStart {
                // Emit previous word
                let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                    .trimmingCharacters(in: .whitespaces)
                if !word.isEmpty {
                    let meanLogProb = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                    words.append(WordConfidence(word: word, confidence: min(1.0, exp(meanLogProb))))
                }
                currentWord = token
                currentLogProbs = [logProbs[i]]
            } else {
                currentWord += token
                currentLogProbs.append(logProbs[i])
            }
        }

        // Emit last word
        if !currentWord.isEmpty {
            let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                .trimmingCharacters(in: .whitespaces)
            if !word.isEmpty {
                let meanLogProb = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                words.append(WordConfidence(word: word, confidence: min(1.0, exp(meanLogProb))))
            }
        }

        return words
    }
}
