import Accelerate
import Foundation

/// Streaming state carried across Sortformer chunks.
///
/// The model is autoregressive in speaker identity: each chunk's predictions
/// depend on `spkcache` (long-term speaker memory) and `fifo` (recent context)
/// from prior chunks. To keep identities stable across chunks we have to
/// maintain those buffers *and* their predictions in sync with what the model
/// expects on the next call.
public struct SortformerStreamingState: Sendable {
    /// Flat `[length, fcDModel]` embeddings — long-term speaker memory.
    public var spkcache: [Float]
    public var spkcacheLength: Int
    /// Flat `[length, numSpeakers]` predictions tracked alongside `spkcache`.
    /// Used during compression to keep the highest-confidence frames per
    /// speaker. Lazily created the first time spkcache overflows.
    public var spkcachePreds: [Float]?

    /// Flat `[length, fcDModel]` embeddings — recent context being collected
    /// before it spills into `spkcache`.
    public var fifo: [Float]
    public var fifoLength: Int
    /// Flat `[length, numSpeakers]` predictions tracked alongside `fifo`.
    public var fifoPreds: [Float]?

    /// Running mean embedding for silence frames. Filled with the
    /// `[spkcacheSilFramesPerSpk × numSpeakers]` placeholder slots during
    /// spkcache compression so the model has a stable "no one is talking"
    /// anchor for the current acoustic environment.
    public var meanSilenceEmbedding: [Float]
    public var silenceFrameCount: Int

    public init(config: SortformerConfig) {
        self.spkcache = []
        self.spkcacheLength = 0
        self.spkcachePreds = nil

        self.fifo = []
        self.fifoLength = 0
        self.fifoPreds = nil

        let dim = config.fcDModel
        self.fifo.reserveCapacity(
            (config.fifoLen + Int(config.chunkLenSeconds)) * dim)
        self.spkcache.reserveCapacity(
            (config.spkcacheLen + config.spkcacheUpdatePeriod) * dim)
        self.meanSilenceEmbedding = [Float](repeating: 0, count: dim)
        self.silenceFrameCount = 0
    }

    public mutating func reset() {
        spkcache.removeAll(keepingCapacity: true)
        spkcacheLength = 0
        spkcachePreds = nil
        fifo.removeAll(keepingCapacity: true)
        fifoLength = 0
        fifoPreds = nil
        meanSilenceEmbedding = [Float](
            repeating: 0, count: meanSilenceEmbedding.count)
        silenceFrameCount = 0
    }
}

/// Streaming state manager for Sortformer. Ports NeMo's `streaming_update`
/// (Python) / FluidAudio's `SortformerStateUpdater` (Swift) so multi-chunk
/// runs of the High variant keep speaker identity stable.
///
/// Why this exists: each model call returns per-frame predictions for
/// `spkcache + fifo + chunk` frames plus encoder embeddings for the chunk
/// itself. The naive "discard oldest when overflow" approach drops the
/// speaker-identifying frames the model needs to keep speakers consistent
/// across chunks. The reference algorithm instead:
///   1. Tracks predictions alongside embeddings.
///   2. Folds silence frames into a running mean (acoustic anchor).
///   3. Compresses the spkcache via log-score top-K so the most distinctive
///      frame per speaker survives, not the most recent one.
struct SortformerStateUpdater {
    let config: SortformerConfig

    init(config: SortformerConfig) {
        self.config = config
    }

    /// Result of one streaming update — the per-chunk predictions in the
    /// "confirmed" region (core frames) and the "tentative" region
    /// (right-context frames whose interpretation may change once the next
    /// chunk lands).
    struct Update {
        let confirmed: [Float]  // [coreFrames * numSpeakers]
        let tentative: [Float]  // [rightCtx * numSpeakers]
    }

    /// Apply one chunk to the streaming state.
    ///
    /// - Parameters:
    ///   - state: streaming state, mutated in place
    ///   - chunkEmbs: per-frame encoder embeddings for the chunk, flat
    ///     `[chunkLeft + chunkCore + chunkRight, fcDModel]`
    ///   - preds: per-frame predictions for the full output region, flat
    ///     `[spkcacheLength + fifoLength + chunkFrames, numSpeakers]`
    ///   - leftContext: number of left-context encoder frames in this chunk
    ///     (varies between the first chunk and later ones)
    ///   - rightContext: number of right-context encoder frames in this chunk
    func update(
        state: inout SortformerStreamingState,
        chunkEmbs: [Float],
        preds: [Float],
        leftContext: Int,
        rightContext: Int
    ) -> Update {
        let dim = config.fcDModel
        let numSpeakers = config.maxSpeakers
        let fifoCapacity = config.fifoLen
        let spkcacheCapacity = config.spkcacheLen

        let prevSpkcacheLen = state.spkcacheLength
        let prevFifoLen = state.fifoLength

        // Capture FIFO predictions before we slice the new chunk in.
        if prevFifoLen > 0 {
            let start = prevSpkcacheLen * numSpeakers
            let end = (prevSpkcacheLen + prevFifoLen) * numSpeakers
            if end <= preds.count {
                state.fifoPreds = Array(preds[start..<end])
            }
        }

        // Slice out the CORE frames from the chunk embeddings (no context).
        let chunkTotalFrames = chunkEmbs.count / dim
        let coreFrames = max(0, chunkTotalFrames - leftContext - rightContext)
        let coreStart = leftContext * dim
        let coreEnd = (leftContext + coreFrames) * dim
        let coreEmbs = (coreEnd <= chunkEmbs.count)
            ? Array(chunkEmbs[coreStart..<coreEnd])
            : []

        // Per-chunk predictions live at:
        //   spkcache (prevSpkcacheLen) | fifo (prevFifoLen) | chunk (chunkTotalFrames)
        // Core slice: skip left context, take coreFrames.
        let chunkStart = prevSpkcacheLen + prevFifoLen + leftContext
        let chunkEnd = chunkStart + coreFrames
        let tentEnd = chunkEnd + rightContext
        let confirmed: [Float] = {
            let s = chunkStart * numSpeakers
            let e = chunkEnd * numSpeakers
            guard e <= preds.count else { return [] }
            return Array(preds[s..<e])
        }()
        let tentative: [Float] = {
            let s = chunkEnd * numSpeakers
            let e = tentEnd * numSpeakers
            guard e <= preds.count else { return [] }
            return Array(preds[s..<e])
        }()

        // Append core embeddings and predictions to FIFO.
        state.fifo.append(contentsOf: coreEmbs)
        state.fifoLength += coreFrames
        if state.fifoPreds != nil {
            state.fifoPreds!.append(contentsOf: confirmed)
        } else {
            state.fifoPreds = confirmed
        }

        // Flush oldest FIFO frames into spkcache when FIFO overflows.
        let context = coreFrames + prevFifoLen
        if context > fifoCapacity {
            guard let currentFifoPreds = state.fifoPreds else {
                return Update(confirmed: confirmed, tentative: tentative)
            }

            var popOut = config.spkcacheUpdatePeriod
            popOut = max(popOut, context - fifoCapacity)
            popOut = min(popOut, context)

            let popEmbs = Array(state.fifo.prefix(popOut * dim))
            let popPreds = Array(currentFifoPreds.prefix(popOut * numSpeakers))

            updateSilenceProfile(
                state: &state, embs: popEmbs, preds: popPreds,
                frameCount: popOut)

            state.fifo.removeFirst(popOut * dim)
            state.fifoLength -= popOut
            state.fifoPreds?.removeFirst(popOut * numSpeakers)

            state.spkcache.append(contentsOf: popEmbs)
            state.spkcacheLength += popOut
            if state.spkcachePreds != nil {
                state.spkcachePreds!.append(contentsOf: popPreds)
            }

            // Compress spkcache the first time AND every time it overflows.
            if state.spkcacheLength > spkcacheCapacity {
                if state.spkcachePreds == nil {
                    // First overflow — backfill predictions for the frames
                    // we already had before popOut arrived.
                    if prevSpkcacheLen > 0 {
                        let prefix = preds.prefix(prevSpkcacheLen * numSpeakers)
                        state.spkcachePreds = Array(prefix) + popPreds
                    } else {
                        state.spkcachePreds = popPreds
                    }
                }
                compressSpkcache(state: &state)
            }
        }

        return Update(confirmed: confirmed, tentative: tentative)
    }

    // MARK: - Silence profile

    private func updateSilenceProfile(
        state: inout SortformerStreamingState,
        embs: [Float],
        preds: [Float],
        frameCount: Int
    ) {
        let dim = config.fcDModel
        let numSpeakers = config.maxSpeakers
        let silenceThreshold = config.silenceThreshold

        for frame in 0..<frameCount {
            var probSum: Float = 0
            for spk in 0..<numSpeakers {
                let idx = frame * numSpeakers + spk
                if idx < preds.count { probSum += preds[idx] }
            }
            guard probSum < silenceThreshold else { continue }

            let n = Float(state.silenceFrameCount)
            let newN = n + 1
            for d in 0..<dim {
                let i = frame * dim + d
                guard i < embs.count else { continue }
                let old = state.meanSilenceEmbedding[d]
                state.meanSilenceEmbedding[d] = (old * n + embs[i]) / newN
            }
            state.silenceFrameCount += 1
        }
    }

    // MARK: - Spkcache compression

    /// Compress spkcache from `state.spkcacheLength` rows down to
    /// `config.spkcacheLen` by keeping the most distinctive frame per speaker.
    /// Mirrors NeMo's `_compress_spkcache`.
    private func compressSpkcache(state: inout SortformerStreamingState) {
        guard let spkcachePreds = state.spkcachePreds else { return }

        let dim = config.fcDModel
        let numSpeakers = config.maxSpeakers
        let spkcacheCapacity = config.spkcacheLen
        let silPerSpk = config.spkcacheSilFramesPerSpk
        let currentLen = state.spkcacheLength

        // Per-speaker budget after silence reservations.
        let perSpk = spkcacheCapacity / numSpeakers - silPerSpk
        let strongPerSpk = Int(Float(perSpk) * config.strongBoostRate)
        let weakPerSpk = Int(Float(perSpk) * config.weakBoostRate)
        let minPosPerSpk = Int(Float(perSpk) * config.minPosScoresRate)

        var scores = logPredScores(preds: spkcachePreds, frameCount: currentLen)
        scores = disableLowScores(
            preds: spkcachePreds, scores: scores,
            frameCount: currentLen, minPos: minPosPerSpk)

        // Latest frames (past the capacity threshold) get a small bonus —
        // breaks ties in favor of recency.
        if currentLen > spkcacheCapacity {
            for frame in spkcacheCapacity..<currentLen {
                for spk in 0..<numSpeakers {
                    scores[frame * numSpeakers + spk] += config.scoresBoostLatest
                }
            }
        }

        scores = boostTopK(
            scores: scores, frameCount: currentLen,
            k: strongPerSpk, scale: 2.0)
        scores = boostTopK(
            scores: scores, frameCount: currentLen,
            k: weakPerSpk, scale: 1.0)

        // Reserve `silPerSpk × numSpeakers` slots with +∞ score so they
        // always make the top-k cut and we can fill them with silence embeds.
        let totalFrames = currentLen + silPerSpk
        for _ in 0..<(silPerSpk * numSpeakers) {
            scores.append(.infinity)
        }

        let (indices, disabled) = topKIndices(
            scores: scores, frameCount: totalFrames, k: spkcacheCapacity)

        var newSpkcache = [Float](repeating: 0, count: spkcacheCapacity * dim)
        var newPreds = [Float](repeating: 0, count: spkcacheCapacity * numSpeakers)

        for (i, frameIdx) in indices.enumerated() {
            if disabled[i] {
                // Fill with running silence profile.
                for d in 0..<dim {
                    newSpkcache[i * dim + d] = state.meanSilenceEmbedding[d]
                }
                // Predictions left at zero.
            } else if frameIdx < currentLen {
                for d in 0..<dim {
                    let src = frameIdx * dim + d
                    if src < state.spkcache.count {
                        newSpkcache[i * dim + d] = state.spkcache[src]
                    }
                }
                for s in 0..<numSpeakers {
                    let src = frameIdx * numSpeakers + s
                    if src < spkcachePreds.count {
                        newPreds[i * numSpeakers + s] = spkcachePreds[src]
                    }
                }
            }
        }

        state.spkcache = newSpkcache
        state.spkcacheLength = spkcacheCapacity
        state.spkcachePreds = newPreds
    }

    // MARK: - Score computation

    /// log(p/(1-p)) + Σ_{j≠i} log(1-p_j) - log(0.5)
    private func logPredScores(preds: [Float], frameCount: Int) -> [Float] {
        let numSpeakers = config.maxSpeakers
        let threshold = config.predScoreThreshold
        var scores = [Float](repeating: 0, count: frameCount * numSpeakers)

        var tmp = [Float](repeating: 0, count: preds.count)
        var log1mP = [Float](repeating: 0, count: preds.count)

        vDSP.clip(preds, to: threshold...Float.greatestFiniteMagnitude, result: &tmp)
        vForce.log(tmp, result: &scores)

        vDSP.clip(preds, to: 0...(1 - threshold), result: &tmp)
        vDSP.negative(tmp, result: &tmp)
        vForce.log1p(tmp, result: &log1mP)
        vDSP.subtract(scores, log1mP, result: &scores)
        vDSP.add(logf(2), scores, result: &scores)

        scores.withUnsafeMutableBufferPointer { sBuf in
            log1mP.withUnsafeBufferPointer { lBuf in
                guard let s = sBuf.baseAddress, let l = lBuf.baseAddress else { return }
                let S = numSpeakers
                for frame in 0..<frameCount {
                    let base = frame &* S
                    var sum: Float = 0
                    for spk in 0..<S { sum += l[base + spk] }
                    for spk in 0..<S { s[base + spk] += sum }
                }
            }
        }
        return scores
    }

    /// Mark scores `-∞` for non-speech frames and for frames whose speaker
    /// already has enough positive scores.
    private func disableLowScores(
        preds: [Float],
        scores: [Float],
        frameCount: Int,
        minPos: Int
    ) -> [Float] {
        let numSpeakers = config.maxSpeakers
        var result = scores

        var posCounts = [Int](repeating: 0, count: numSpeakers)
        for frame in 0..<frameCount {
            for spk in 0..<numSpeakers {
                let idx = frame * numSpeakers + spk
                if preds[idx] > 0.5 && scores[idx] > 0 {
                    posCounts[spk] += 1
                }
            }
        }
        for spk in 0..<numSpeakers {
            for frame in 0..<frameCount {
                let idx = frame * numSpeakers + spk
                let p = preds[idx]
                if p <= 0.5 {
                    result[idx] = -.infinity
                    continue
                }
                if result[idx] <= 0 && posCounts[spk] >= minPos {
                    result[idx] = -.infinity
                }
            }
        }
        return result
    }

    /// Boost the top-k scores per speaker by `-scale * log(0.5)`.
    private func boostTopK(
        scores: [Float],
        frameCount: Int,
        k: Int,
        scale: Float
    ) -> [Float] {
        let S = config.maxSpeakers
        guard frameCount > 0, S > 0, k > 0 else { return scores }
        let boost: Float = -scale * logf(0.5)
        var result = scores
        let kEff = min(k, frameCount)

        result.withUnsafeMutableBufferPointer { resBuf in
            guard let base = resBuf.baseAddress else { return }
            for spk in 0..<S {
                // Insertion sort: arrays kept DESC by score.
                var topFrames = [Int](repeating: 0, count: kEff)
                var topScores = [Float](
                    repeating: -.greatestFiniteMagnitude, count: kEff)
                var count = 0
                for frame in 0..<frameCount {
                    let idx = frame &* S &+ spk
                    let v = base[idx]
                    if v == -.infinity { continue }
                    if count < kEff {
                        var pos = count
                        while pos > 0 && v > topScores[pos - 1] {
                            topScores[pos] = topScores[pos - 1]
                            topFrames[pos] = topFrames[pos - 1]
                            pos -= 1
                        }
                        topScores[pos] = v
                        topFrames[pos] = frame
                        count += 1
                    } else {
                        if v <= topScores[count - 1] { continue }
                        var pos = count - 1
                        while pos > 0 && v > topScores[pos - 1] {
                            topScores[pos] = topScores[pos - 1]
                            topFrames[pos] = topFrames[pos - 1]
                            pos -= 1
                        }
                        topScores[pos] = v
                        topFrames[pos] = frame
                    }
                }
                for i in 0..<count {
                    base[topFrames[i] &* S &+ spk] += boost
                }
            }
        }
        return result
    }

    /// Top-k frame indices over the (numSpeakers × frameCount) flattened
    /// scores. Matches NeMo's `_get_topk_indices` exactly: permutes to
    /// `[speaker, frame]` order, takes the top k, sorts ascending, then
    /// converts back to frame indices via `% frameCount`. Frames past the
    /// real content (`frameCount - silFramesPerSpk`) are marked disabled
    /// so the caller fills them with the silence profile.
    private func topKIndices(
        scores: [Float],
        frameCount: Int,
        k: Int
    ) -> (indices: [Int], disabled: [Bool]) {
        let S = config.maxSpeakers
        let silPerSpk = config.spkcacheSilFramesPerSpk
        let realFrames = frameCount - silPerSpk
        let maxIndex = config.maxIndex
        let N = frameCount * S
        guard k > 0 else { return ([], []) }

        let kEff = min(k, N)
        var bestIdx = [Int](repeating: 0, count: kEff)
        var bestVal = [Float](repeating: -.infinity, count: kEff)
        var count = 0

        for spk in 0..<S {
            for frame in 0..<frameCount {
                let permuted = spk * frameCount + frame
                let v = scores[frame * S + spk]
                if count < kEff {
                    var pos = count
                    while pos > 0 {
                        let pv = bestVal[pos - 1]
                        let pi = bestIdx[pos - 1]
                        if v > pv || (v == pv && permuted < pi) {
                            bestVal[pos] = pv
                            bestIdx[pos] = pi
                            pos -= 1
                        } else { break }
                    }
                    bestVal[pos] = v
                    bestIdx[pos] = permuted
                    count += 1
                } else {
                    let worstV = bestVal[kEff - 1]
                    let worstI = bestIdx[kEff - 1]
                    if v < worstV || (v == worstV && permuted >= worstI) { continue }
                    var pos = kEff - 1
                    while pos > 0 {
                        let pv = bestVal[pos - 1]
                        let pi = bestIdx[pos - 1]
                        if v > pv || (v == pv && permuted < pi) {
                            bestVal[pos] = pv
                            bestIdx[pos] = pi
                            pos -= 1
                        } else { break }
                    }
                    bestVal[pos] = v
                    bestIdx[pos] = permuted
                }
            }
        }

        var indices = [Int](repeating: maxIndex, count: k)
        for i in 0..<kEff {
            indices[i] = (bestVal[i] == -.infinity) ? maxIndex : bestIdx[i]
        }
        indices.sort()

        var disabled = [Bool](repeating: false, count: k)
        for i in 0..<k where indices[i] == maxIndex { disabled[i] = true }
        for i in 0..<k where !disabled[i] {
            indices[i] = indices[i] % frameCount
        }
        for i in 0..<k where !disabled[i] {
            if indices[i] >= realFrames { disabled[i] = true }
        }
        for i in 0..<k where disabled[i] { indices[i] = 0 }
        return (indices, disabled)
    }
}
