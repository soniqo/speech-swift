import Foundation
import AudioCommon

/// SentencePiece Unigram tokenizer used to encode word-boosting phrases into
/// the exact token sequence the Nemotron RNN-T decoder emits.
///
/// Parses the SentencePiece model via the shared `AudioCommon.SentencePieceModel`
/// reader (also used by OmnilingualASR, PersonaPlex, SpeechWakeWord) — this
/// module owns only the encode-side Viterbi DP.
///
/// Why we ship this instead of greedy vocab segmentation: real SPM Unigram
/// scores spans by piece log-probability, not by left-to-right longest match.
/// On out-of-vocabulary brand/technical terms the two paths diverge on the
/// first token (e.g. "voxtral" → ▁ vo x tra l with real SPM, ▁v o x tra l
/// greedy), which would cause the boost trie to never fire.
struct NemotronSentencePieceUnigramTokenizer: Sendable {

    /// Hard cap on input phrase length (Unicode scalars).
    ///
    /// Unigram Viterbi is O(N²) over candidate spans, which is fine for the
    /// vocabulary terms this API expects (names, technical terms, short
    /// phrases). A clipboard-sized paste would otherwise spin the calling
    /// thread for tens of seconds and allocate hundreds of MB of DP state,
    /// so we refuse pathologically long inputs at the boundary.
    static let maxPhraseScalars = 256

    private let pieces: [AudioCommon.SentencePieceModel.Piece]
    private let tokenToId: [String: Int]

    init(modelURL: URL) throws {
        let model = try AudioCommon.SentencePieceModel(contentsOf: modelURL)
        try self.init(pieces: model.pieces)
    }

    init(model: AudioCommon.SentencePieceModel) throws {
        try self.init(pieces: model.pieces)
    }

    private init(pieces: [AudioCommon.SentencePieceModel.Piece]) throws {
        guard !pieces.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: "tokenizer.model",
                reason: "SentencePiece model did not contain vocabulary pieces"
            )
        }
        self.pieces = pieces
        self.tokenToId = Dictionary(
            uniqueKeysWithValues: pieces.enumerated().map { ($0.element.text, $0.offset) }
        )
    }

    /// Test-only initialiser that accepts raw `(text, score, type)` tuples.
    init(rawPieces: [(token: String, score: Float, type: Int)]) {
        let pieces = rawPieces.map {
            AudioCommon.SentencePieceModel.Piece(text: $0.token, score: $0.score, type: Int32($0.type))
        }
        self.pieces = pieces
        self.tokenToId = Dictionary(
            uniqueKeysWithValues: pieces.enumerated().map { ($0.element.text, $0.offset) }
        )
    }

    func encodeForWordBoosting(_ phrase: String) -> [Int]? {
        let normalized = pieceText(for: phrase)
        guard !normalized.isEmpty else { return nil }

        let scalars = Array(normalized.unicodeScalars)
        guard scalars.count <= Self.maxPhraseScalars else { return nil }

        // bestScores[i] = best log-prob of any segmentation reaching position i.
        // backPointer[i] = (start, pieceId) that achieved bestScores[i]. We
        // store back-pointers instead of copying the path array per accepted
        // span — this keeps memory O(N) instead of O(N²) and avoids the
        // O(N³) total path-copy cost the previous implementation incurred on
        // long inputs.
        var bestScores = [Float](repeating: -.infinity, count: scalars.count + 1)
        var backPointer = [(start: Int, id: Int)?](repeating: nil, count: scalars.count + 1)
        bestScores[0] = 0

        for start in 0..<scalars.count where bestScores[start].isFinite {
            for end in (start + 1)...scalars.count {
                let token = String(String.UnicodeScalarView(scalars[start..<end]))
                guard let id = tokenToId[token] else { continue }
                // Filter by piece type — never select <unk>, control, byte,
                // or unused pieces as a Viterbi span. The previous gate
                // `id != 0` assumed <unk> was always at index 0 (true for
                // Nemotron 3.5 only) and silently allowed other special
                // pieces to be selected.
                guard !pieces[id].isControlOrUnknown else { continue }

                let candidateScore = bestScores[start] + pieces[id].score
                let candidateBetter: Bool
                if candidateScore > bestScores[end] {
                    candidateBetter = true
                } else if candidateScore == bestScores[end] {
                    // Tie-break: prefer the path with the lexicographically
                    // smaller token-id sequence, matching the previous
                    // implementation's determinism guarantee. Comparing only
                    // the most-recent (start, id) is sufficient because
                    // bestScores[start] would already have selected the
                    // lex-min prefix.
                    if let existing = backPointer[end] {
                        candidateBetter =
                            id < existing.id ||
                            (id == existing.id && start > existing.start)
                    } else {
                        candidateBetter = true
                    }
                } else {
                    candidateBetter = false
                }

                if candidateBetter {
                    bestScores[end] = candidateScore
                    backPointer[end] = (start: start, id: id)
                }
            }
        }

        guard bestScores[scalars.count].isFinite else { return nil }

        // Walk back-pointers to reconstruct the path.
        var path: [Int] = []
        var cursor = scalars.count
        while cursor > 0, let bp = backPointer[cursor] {
            path.append(bp.id)
            cursor = bp.start
        }
        return path.isEmpty ? nil : path.reversed()
    }

    private func pieceText(for phrase: String) -> String {
        let normalized = (phrase as NSString)
            .precomposedStringWithCompatibilityMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
        guard !normalized.isEmpty else { return "" }
        return "▁" + normalized.replacingOccurrences(of: " ", with: "▁")
    }
}
