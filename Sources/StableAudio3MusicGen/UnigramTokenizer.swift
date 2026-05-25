import Foundation
import AudioCommon

/// Minimal Unigram (SentencePiece) Viterbi tokenizer.
///
/// Operates over the piece array exposed by `AudioCommon.SentencePieceModel`.
/// Supports the subset of SP behaviour SA3 / T5Gemma needs:
///   * "▁"-prefixing: spaces become U+2581, an additional U+2581 is prepended.
///   * Forward-DP best-segmentation over UTF-8 byte positions of the input.
///   * Unknown pieces are emitted as `unkId` (T5Gemma SP model: id = 3).
///   * No byte-fallback (T5Gemma uses sufficient regular pieces for English).
///
/// This is enough for SA3 text prompts. It is not a general-purpose SP encoder.
public final class UnigramTokenizer: @unchecked Sendable {
    public let model: SentencePieceModel
    public let pieceToId: [String: Int]
    public let unkId: Int

    public init(model: SentencePieceModel, unkId: Int = 3) {
        self.model = model
        var map = [String: Int](minimumCapacity: model.pieces.count)
        for (i, p) in model.pieces.enumerated() {
            map[p.text] = i
        }
        self.pieceToId = map
        self.unkId = unkId
    }

    /// Encode a string to a sequence of piece ids.
    ///
    /// T5Gemma's `.model` is **BPE**, not Unigram — the per-piece "score" is
    /// a merge rank stored as a negative integer. Higher-priority merges
    /// have the **least-negative** (closest to zero) score and are applied
    /// first. The algorithm here:
    ///   1. Split the input on ASCII spaces. Each word but the first gets a
    ///      leading `▁` (U+2581) so that "house" inside a sentence resolves
    ///      to `▁house`.
    ///   2. Each word starts as a list of single Unicode-scalar pieces.
    ///   3. Repeatedly find the adjacent pair that, when concatenated, is in
    ///      the vocab with the most-negative score, and apply the merge.
    ///   4. Stop when no adjacent pair is in the vocab.
    public func encodeAsIds(_ text: String) -> [Int] {
        if text.isEmpty { return [] }
        var ids: [Int] = []
        let words = text.split(separator: " ", omittingEmptySubsequences: false)
        for (idx, w) in words.enumerated() {
            let segment = idx == 0 ? String(w) : "▁" + String(w)
            if segment.isEmpty { continue }
            bpeMergeAppend(&ids, segment: segment)
        }
        return ids
    }

    private func bpeMergeAppend(_ ids: inout [Int], segment: String) {
        // Start as a list of single-grapheme pieces.
        var pieces = segment.map { String($0) }
        if pieces.isEmpty { return }

        while pieces.count >= 2 {
            // Find the adjacent pair with the highest priority — least-negative
            // (closest to zero) score wins.
            var bestIdx = -1
            var bestScore = -Float.infinity
            for i in 0..<(pieces.count - 1) {
                let merged = pieces[i] + pieces[i + 1]
                guard let pid = pieceToId[merged] else { continue }
                let s = model.pieces[pid].score
                if s > bestScore {
                    bestScore = s
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }   // no more merges possible
            let merged = pieces[bestIdx] + pieces[bestIdx + 1]
            pieces.replaceSubrange(bestIdx...(bestIdx + 1), with: [merged])
        }

        for p in pieces {
            if let id = pieceToId[p] {
                ids.append(id)
            } else {
                ids.append(unkId)
            }
        }
    }
}

/// Upper bound on a single piece's UTF-8 length. T5Gemma's longest pieces are
/// around 24 bytes (multi-codepoint Asian-script words); 32 is comfortably safe.
private let MaxPieceUTF8Length = 32
