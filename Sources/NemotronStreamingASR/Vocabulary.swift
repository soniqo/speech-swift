import Foundation
import AudioCommon

/// SentencePiece vocabulary for Nemotron-3.5 ASR Streaming Multilingual
/// (13087 BPE pieces + 1 blank). Punctuation and capitalization render as
/// regular BPE tokens; per-language tags (`<en-US>`, `<ar-AR>`, …) and
/// other angle-bracket special tokens (`<unk>`, `<s>`, `</s>`, `<auto>`)
/// are stripped during decode so they don't pollute the user-facing text.
public struct NemotronVocabulary: Sendable {
    private let idToToken: [Int: String]

    public var count: Int { idToToken.count }

    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
    }

    /// Load vocabulary from vocab.json (format: `{"0": "▁the", "1": "▁a", ...}`).
    public static func load(from url: URL) throws -> NemotronVocabulary {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        var mapping: [Int: String] = [:]
        for (key, value) in raw {
            if let id = Int(key) {
                mapping[id] = value
            }
        }
        return NemotronVocabulary(idToToken: mapping)
    }

    /// True when `token` is an angle-bracket-wrapped special marker
    /// (`<en-US>`, `<unk>`, `<s>`, `</s>`, `<auto>`, etc.) that should be
    /// stripped from the user-facing transcript. A token with whitespace
    /// inside the brackets is considered ordinary text (defensive — the
    /// SentencePiece vocab doesn't produce these, but keeps the filter
    /// from misclassifying contrived inputs).
    static func isSpecialToken(_ token: String) -> Bool {
        guard token.first == "<", token.last == ">", token.count >= 3 else { return false }
        return !token.contains(" ") && !token.contains("\t")
    }

    /// Decode token IDs to text. Blank is filtered upstream; angle-bracket
    /// special tokens (language tags, `<unk>`, `<s>`, …) are filtered here.
    public func decode(_ tokenIds: [Int]) -> String {
        var text = ""
        for id in tokenIds {
            guard let token = idToToken[id] else { continue }
            if Self.isSpecialToken(token) { continue }
            text += token
        }
        return text.replacingOccurrences(of: "▁", with: " ").trimmingCharacters(in: .whitespaces)
    }

    /// Decode with per-word confidence scores. Angle-bracket special tokens
    /// are skipped so their log-probs don't contaminate adjacent words'
    /// confidence averages.
    public func decodeWords(_ tokenIds: [Int], logProbs: [Float]) -> [WordConfidence] {
        guard tokenIds.count == logProbs.count else { return [] }

        var words: [WordConfidence] = []
        var currentWord = ""
        var currentLogProbs: [Float] = []

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }
            if Self.isSpecialToken(token) { continue }
            let isWordStart = token.hasPrefix("▁") && !currentWord.isEmpty

            if isWordStart {
                let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                    .trimmingCharacters(in: .whitespaces)
                if !word.isEmpty {
                    let mean = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                    words.append(WordConfidence(word: word, confidence: min(1.0, exp(mean))))
                }
                currentWord = token
                currentLogProbs = [logProbs[i]]
            } else {
                currentWord += token
                currentLogProbs.append(logProbs[i])
            }
        }

        if !currentWord.isEmpty {
            let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                .trimmingCharacters(in: .whitespaces)
            if !word.isEmpty {
                let mean = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
                words.append(WordConfidence(word: word, confidence: min(1.0, exp(mean))))
            }
        }

        return words
    }
}
