import AudioCommon
import Foundation

/// SentencePiece-based vocabulary for Parakeet TDT.
///
/// Loads a `vocab.json` mapping from token ID strings to token strings,
/// and decodes sequences using SentencePiece conventions (`▁` → space).
public struct ParakeetVocabulary: Sendable {
    /// Mapping from token ID to token string.
    private let idToToken: [Int: String]

    /// Token IDs used for model control rather than transcript text.
    ///
    /// Parakeet's tokenizer interleaves plain digit tokens with control tokens, so token IDs
    /// cannot be classified using a numeric boundary. Special tokens use angle-bracket syntax
    /// (`<unk>`, `<pad>`, `<|pnc|>`, `<|en|>`, and similar).
    private let specialTokenIds: Set<Int>

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
        self.specialTokenIds = Self.extractSpecialTokenIds(from: idToToken)
        self.languageTagIds = Self.extractLanguageTags(from: idToToken)
    }

    /// Whether an emitted token belongs in transcript text.
    ///
    /// Unknown IDs are excluded as well: a model/vocabulary mismatch should not contribute an
    /// undecodable token or confidence score to the returned transcription.
    func isTextToken(_ id: Int) -> Bool {
        idToToken[id] != nil && !specialTokenIds.contains(id)
    }

    private static func extractSpecialTokenIds(from idToToken: [Int: String]) -> Set<Int> {
        Set(idToToken.compactMap { id, token in
            token.hasPrefix("<") && token.hasSuffix(">") ? id : nil
        })
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

    /// Resolve the language-tag IDs that should be suppressed for a comma-separated allowlist.
    /// Unknown-only input leaves native auto-detection unchanged instead of masking every tag.
    func maskedLanguageTokenIds(allowing language: String?) -> Set<Int> {
        guard let language else { return [] }
        let requested = language.lowercased()
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        guard !requested.isEmpty else { return [] }

        let allowedIds = Set(requested.compactMap { languageTagIds[$0] })
        guard !allowedIds.isEmpty else { return [] }
        return Set(languageTagIds.values).subtracting(allowedIds)
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
    /// When `tokenStartTimes` (one absolute start time in seconds per token) and
    /// `frameDuration` (seconds one token's emission frame spans) are given, each
    /// word also carries `startTime` (its first token's start) and `endTime` (its
    /// last token's start plus `frameDuration`). If either is missing, or the
    /// start times do not match the tokens one to one, all times are nil.
    public func decodeWords(
        _ tokenIds: [Int], logProbs: [Float],
        tokenStartTimes: [Double]? = nil, frameDuration: Double? = nil
    ) -> [WordConfidence] {
        guard tokenIds.count == logProbs.count else {
            return [WordConfidence(word: decode(tokenIds), confidence: 0)]
        }
        let starts = tokenStartTimes?.count == tokenIds.count ? tokenStartTimes : nil

        var words = [WordConfidence]()
        var currentWord = ""
        var currentLogProbs = [Float]()
        var firstStart: Double?
        var lastStart: Double?

        func flush() {
            let meanLP = currentLogProbs.reduce(0, +) / Float(currentLogProbs.count)
            var startTime: Double?
            var endTime: Double?
            if let frameDuration, let firstStart, let lastStart {
                startTime = firstStart
                endTime = lastStart + frameDuration
            }
            words.append(WordConfidence(
                word: currentWord, confidence: min(1.0, exp(meanLP)),
                startTime: startTime, endTime: endTime))
            currentWord = ""
            currentLogProbs = []
            firstStart = nil
            lastStart = nil
        }

        for (i, id) in tokenIds.enumerated() {
            guard let token = idToToken[id] else { continue }

            let startsNewWord = token.hasPrefix("\u{2581}")
            let text = token.replacingOccurrences(of: "\u{2581}", with: "")

            if startsNewWord && !currentWord.isEmpty {
                flush()
            }

            currentWord += text
            currentLogProbs.append(logProbs[i])
            if let starts {
                if firstStart == nil { firstStart = starts[i] }
                lastStart = starts[i]
            }
        }

        if !currentWord.isEmpty {
            flush()
        }

        return words
    }
}
