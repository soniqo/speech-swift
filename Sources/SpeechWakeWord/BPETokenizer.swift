import Foundation
import AudioCommon

/// SentencePiece-style BPE encoder for KWS keyword phrases.
///
/// Uses Viterbi decode (dynamic programming over the piece table with
/// per-piece log-probability scores from the shipped ``bpe.model``) so
/// the decomposition matches what SentencePiece would emit — no longer
/// requiring callers to supply ``KeywordSpec(tokens:)`` for common
/// phrases whose greedy decomposition differed from the training-time
/// one (e.g. ``"LIGHT UP"`` decomposed greedily as ``▁LI GHT ▁UP`` but
/// the model was trained on ``▁ L IGHT ▁UP``).
///
/// For edge cases or unusual phrases callers can still override via
/// ``KeywordSpec(tokens:)`` — that bypasses this encoder entirely.
public struct BPETokenizer: Sendable {
    public let pieceToId: [String: Int]
    public let idToPiece: [Int: String]
    public let pieceScores: [String: Float]
    public let unkId: Int
    /// Normalise input case before encoding. The icefall KWS vocab is
    /// uppercase — set ``.uppercase`` for that model, ``.none`` to
    /// preserve the caller's casing, ``.lowercase`` for all-lowercase
    /// vocabularies.
    public let caseHandling: CaseHandling

    public enum CaseHandling: Sendable, Equatable {
        case none
        case uppercase
        case lowercase
    }

    public init(
        model: SentencePieceModel,
        unkId: Int = 2,
        caseHandling: CaseHandling = .uppercase
    ) {
        var p2i = [String: Int]()
        var i2p = [Int: String]()
        var scores = [String: Float]()
        for (idx, piece) in model.pieces.enumerated() {
            p2i[piece.text] = idx
            i2p[idx] = piece.text
            scores[piece.text] = piece.score
        }
        self.pieceToId = p2i
        self.idToPiece = i2p
        self.pieceScores = scores
        self.unkId = unkId
        self.caseHandling = caseHandling
    }

    /// Encode a phrase like "HEY SONIQO" into BPE token ids via
    /// Viterbi decode. Tokens follow SentencePiece conventions: the
    /// word-start marker ``▁`` opens each word.
    public func encode(_ phrase: String) -> [Int] {
        let normalized: String
        switch caseHandling {
        case .uppercase: normalized = phrase.uppercased()
        case .lowercase: normalized = phrase.lowercased()
        case .none: normalized = phrase
        }
        let words = normalized.split(whereSeparator: { $0.isWhitespace })
        var ids: [Int] = []
        for word in words {
            let wordWithMarker = "\u{2581}\(word)"
            ids.append(contentsOf: encodeViterbi(wordWithMarker))
        }
        return ids
    }

    // MARK: - Viterbi decode

    /// Dynamic-programming Viterbi decode over the piece table.
    ///
    /// ``dp[i]`` = best cumulative log-score achievable decomposing
    /// ``chars[0..<i]`` into pieces. Recurrence:
    ///
    /// ```
    /// dp[i] = max over j < i of ( dp[j] + score(piece(chars[j..<i])) )
    /// ```
    ///
    /// Backtrack recovers the piece sequence. When a prefix has no
    /// winning path (i.e. the only completions that reach here go
    /// through unknown characters), we fall back to a single-character
    /// ``unkId`` insertion with a large negative score so real pieces
    /// are always preferred when available.
    private func encodeViterbi(_ text: String) -> [Int] {
        let chars = Array(text.unicodeScalars)
        let n = chars.count
        guard n > 0 else { return [] }

        // Cumulative best score at each boundary, 0..n.
        var dp = [Float](repeating: -.infinity, count: n + 1)
        // Back-pointer to the start of the piece that ends at i.
        var backStart = [Int](repeating: -1, count: n + 1)
        // Whether we had to insert a fallback unk at i.
        var backIsUnk = [Bool](repeating: false, count: n + 1)
        dp[0] = 0

        // Large negative penalty for fallback unks — bigger than any
        // realistic cumulative sum of piece log-probs so unks only win
        // when no valid covering exists.
        let unkPenalty: Float = -1e6

        for i in 1...n {
            // Try every piece-length ending at i.
            for j in 0..<i {
                guard dp[j] > -.infinity else { continue }
                let prefix = String(String.UnicodeScalarView(chars[j..<i]))
                if let score = pieceScores[prefix] {
                    let candidate = dp[j] + score
                    if candidate > dp[i] {
                        dp[i] = candidate
                        backStart[i] = j
                        backIsUnk[i] = false
                    }
                }
            }
            // Fallback: if nothing reached i, let a single-character
            // unk cover position i-1..i.
            if dp[i] == -.infinity, dp[i - 1] > -.infinity {
                dp[i] = dp[i - 1] + unkPenalty
                backStart[i] = i - 1
                backIsUnk[i] = true
            }
        }

        // Backtrack.
        var ids: [Int] = []
        var cursor = n
        while cursor > 0 {
            let start = backStart[cursor]
            if start < 0 {
                // Degenerate: shouldn't happen but guard anyway.
                ids.append(unkId)
                break
            }
            if backIsUnk[cursor] {
                ids.append(unkId)
            } else {
                let piece = String(String.UnicodeScalarView(chars[start..<cursor]))
                ids.append(pieceToId[piece] ?? unkId)
            }
            cursor = start
        }
        return ids.reversed()
    }
}
