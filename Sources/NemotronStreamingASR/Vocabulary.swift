import Foundation
import AudioCommon

/// SentencePiece vocabulary for Nemotron-3.5 ASR Streaming Multilingual
/// (13087 BPE pieces + 1 blank). Punctuation and capitalization render as
/// regular BPE tokens; per-language tags (`<en-US>`, `<ar-AR>`, …) and
/// other angle-bracket special tokens (`<unk>`, `<s>`, `</s>`, `<auto>`)
/// are stripped during decode so they don't pollute the user-facing text.
/// One decoded word with its encoder-frame emission span, before the
/// session converts frames to seconds.
public struct TimedWordPiece: Sendable, Equatable {
    public let word: String
    public let startFrame: Int
    public let endFrame: Int

    public init(word: String, startFrame: Int, endFrame: Int) {
        self.word = word
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
}

public struct NemotronVocabulary: Sendable {
    private let idToToken: [Int: String]
    private let tokenToId: [String: Int]

    public var count: Int { idToToken.count }

    public init(idToToken: [Int: String]) {
        self.idToToken = idToToken
        var reverse: [String: Int] = [:]
        for (id, token) in idToToken {
            if let existing = reverse[token], existing < id { continue }
            reverse[token] = id
        }
        self.tokenToId = reverse
    }

    /// Load vocabulary from either the Core ML dictionary form
    /// (`{"0": "▁the"}`) or the MLX ordered-array form (`["▁the", ...]`).
    public static func load(from url: URL) throws -> NemotronVocabulary {
        let data = try Data(contentsOf: url)
        if let ordered = try? JSONDecoder().decode([String].self, from: data) {
            return NemotronVocabulary(
                idToToken: Dictionary(
                    uniqueKeysWithValues: ordered.enumerated().map {
                        ($0.offset, $0.element)
                    }
                )
            )
        }
        let raw = try JSONDecoder().decode(
            [String: String].self,
            from: data
        )
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

    /// Deterministically encode a phrase using the loaded SentencePiece-style
    /// vocabulary. This is intentionally simple and setup-time only: phrases
    /// are normalized to a leading word-boundary marker and greedily segmented
    /// by longest matching vocab token.
    func encodeForWordBoosting(_ phrase: String) -> [Int]? {
        let pieceText = pieceTextForWordBoosting(phrase)
        guard !pieceText.isEmpty else { return nil }

        var ids: [Int] = []
        var start = pieceText.startIndex

        while start < pieceText.endIndex {
            var end = pieceText.endIndex
            var matched: (id: Int, end: String.Index)?
            while end > start {
                let token = String(pieceText[start..<end])
                if let id = tokenToId[token] {
                    matched = (id, end)
                    break
                }
                end = pieceText.index(before: end)
            }

            guard let match = matched else { return nil }
            ids.append(match.id)
            start = match.end
        }

        return ids
    }

    func encodingsForWordBoosting(_ phrase: String, maxCount: Int) -> [[Int]] {
        let pieceText = pieceTextForWordBoosting(phrase)
        guard !pieceText.isEmpty, maxCount > 0 else { return [] }

        let greedy = encodeForWordBoosting(phrase)
        let characters = Array(pieceText)
        var memo: [Int: [[Int]]] = [:]

        func collect(from offset: Int) -> [[Int]] {
            if offset == characters.count { return [[]] }
            if let cached = memo[offset] { return cached }

            var results: [[Int]] = []
            for end in (offset + 1)...characters.count {
                let token = String(characters[offset..<end])
                guard let id = tokenToId[token] else { continue }

                for suffix in collect(from: end) {
                    results.append([id] + suffix)
                    if results.count >= maxCount * 4 { break }
                }
                if results.count >= maxCount * 4 { break }
            }

            results.sort { lhs, rhs in
                if lhs.count != rhs.count { return lhs.count < rhs.count }
                return lhs.lexicographicallyPrecedes(rhs)
            }
            if results.count > maxCount * 4 {
                results = Array(results.prefix(maxCount * 4))
            }
            memo[offset] = results
            return results
        }

        var encodings = collect(from: 0)
        if let greedy {
            encodings.removeAll { $0 == greedy }
            encodings.insert(greedy, at: 0)
        }

        var deduped: [[Int]] = []
        var seen = Set<[Int]>()
        for encoding in encodings where !encoding.isEmpty {
            if seen.insert(encoding).inserted {
                deduped.append(encoding)
            }
            if deduped.count == maxCount { break }
        }
        return deduped
    }

    private func pieceTextForWordBoosting(_ phrase: String) -> String {
        let normalized = phrase
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        guard !normalized.isEmpty else { return "" }
        return "▁" + normalized.replacingOccurrences(of: " ", with: "▁")
    }

    /// Decode with per-word emission frames. Mirrors `decodeWords`: special
    /// tokens are skipped and a `▁`-prefixed piece starts a new word. Frames
    /// are whatever the caller provides (session-absolute encoder frames);
    /// `endFrame` is the frame of the word's last token, inclusive. The
    /// session converts pieces to seconds-based `TimedWord`s.
    public func decodeTimedWords(
        _ tokenIds: [Int], frames: [Int]
    ) -> [TimedWordPiece] {
        guard tokenIds.count == frames.count else { return [] }

        var words: [TimedWordPiece] = []
        var currentWord = ""
        var currentStart = 0
        var currentEnd = 0

        func flushCurrentWord() {
            let word = currentWord.replacingOccurrences(of: "▁", with: " ")
                .trimmingCharacters(in: .whitespaces)
            if !word.isEmpty {
                words.append(TimedWordPiece(
                    word: word, startFrame: currentStart, endFrame: currentEnd))
            }
        }

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }
            if Self.isSpecialToken(token) { continue }
            let isWordStart = token.hasPrefix("▁") && !currentWord.isEmpty

            if isWordStart {
                flushCurrentWord()
                currentWord = token
                currentStart = frames[i]
                currentEnd = frames[i]
            } else {
                if currentWord.isEmpty { currentStart = frames[i] }
                currentWord += token
                currentEnd = frames[i]
            }
        }
        flushCurrentWord()
        return words
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
