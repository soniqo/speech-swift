import Foundation
import MLX

// MARK: - On-device sampling

/// `ChatSampler` evaluated where the logits already are.
///
/// The host sampler needs the whole distribution in a Swift array. For Gemma 4's 262,144-entry
/// vocabulary that is a megabyte crossing the GPU boundary per generated token, copied twice more
/// on arrival (`prefix(vocabSize)`, then the sampler's own working copy) before any of the O(V)
/// host scans start. Every decision it makes — end-token suppression, repetition penalty,
/// temperature, top-K, top-P and the draw — is expressible as an MLX op, so this variant makes
/// them on device and brings back one Int32.
///
/// It is a second implementation of the same rules rather than a rearrangement of the first, so
/// nothing about the construction guarantees they agree; tests hold them together. That is why
/// ``ChatSampler/sample(logits:config:previousTokens:uniform:)`` takes the nucleus draw as a
/// parameter — with the randomness lifted out, both paths are pure functions of the same inputs
/// and can be compared token-for-token over a sweep of draws.
///
/// One divergence is inherent. Both paths rank candidates by probability and neither breaks ties
/// by index — the host sorts with `Array.sort`, this one with `argSort` — so on exactly equal
/// logits they may return different tokens *of equal probability*. Greedy is unaffected: `argMax`
/// and the host's strict `>` scan both take the lowest index.
extension ChatSampler {

    /// Sample one token from a device logits array.
    ///
    /// - Parameters:
    ///   - logits: `[vocab]`, `[T, vocab]` or `[B, T, vocab]`; the last position is sampled.
    ///   - config: the same knobs the host sampler reads.
    ///   - suppressing: token ids forced out of the running, e.g. the caller's end-of-turn ids.
    ///     Suppression happens before the repetition penalty, matching the host decode loop's
    ///     order.
    ///   - previousTokens: recent history for the repetition penalty.
    ///   - vocabSize: logits wider than the tokenizer's vocabulary are trimmed, matching the host
    ///     path's `prefix(vocabSize)`.
    ///   - uniform: the nucleus draw, in `[0, 1)`. Unread when sampling greedily.
    /// - Returns: a scalar Int32 `MLXArray`. Reading it (`.item(Int.self)`) is the one host/GPU
    ///   synchronisation a decode step needs — the model forward, the sampling, and nothing else.
    static func sampleOnDevice(
        logits: MLXArray,
        config: ChatSamplingConfig,
        suppressing: [Int] = [],
        previousTokens: [Int] = [],
        vocabSize: Int? = nil,
        uniform: Float
    ) -> MLXArray {
        var row = logits
        if row.ndim == 3 { row = row[0, row.dim(1) - 1] }
        else if row.ndim == 2 { row = row[row.dim(0) - 1] }
        var scaled = row.asType(.float32)
        if let vocabSize, vocabSize < scaled.dim(0) { scaled = scaled[0 ..< vocabSize] }
        let vocab = scaled.dim(0)
        if vocab == 0 { return MLXArray(Int32(0)) }

        // End-token suppression. `-greatestFiniteMagnitude` rather than `-infinity` because the
        // penalty below multiplies whatever it finds and the host path suppresses with the same
        // finite value; both then overflow to -inf together on a suppressed token that also
        // appears in the penalty window.
        let blocked = suppressing.filter { $0 >= 0 && $0 < vocab }
        if !blocked.isEmpty {
            scaled[MLXArray(blocked.map(Int32.init))] = MLXArray(-Float.greatestFiniteMagnitude)
        }

        // Repetition penalty over the same 64-token window, divided when positive and multiplied
        // when not, so both signs move away from being picked. Deduplicating first matters: the
        // host applies the penalty once per distinct id, and a scatter would otherwise depend on
        // which duplicate wrote last.
        if config.repetitionPenalty > 1.0 {
            let seen = Set(previousTokens.suffix(64)).filter { $0 >= 0 && $0 < vocab }
            if !seen.isEmpty {
                let ids = MLXArray(seen.map(Int32.init))
                let v = scaled[ids]
                let p = config.repetitionPenalty
                scaled[ids] = which(v .> 0, v / p, v * p)
            }
        }

        // Greedy. `argMax` takes the lowest index among equal maxima, like the host's strict `>`
        // scan — `ChatSamplerDeviceTests.testGreedyTieTakesLowestIndex` pins that.
        if config.temperature <= 0 { return argMax(scaled, axis: 0) }

        // Candidate set = the top-K by logit, which is the top-K by probability because softmax is
        // monotonic — the same equivalence the host sampler leans on to select on raw logits.
        // Ranking the whole vocabulary once and slicing is one kernel; everything after it is K
        // wide, and K is a few dozen.
        //
        // `argPartition` would express "the K largest, unordered" more directly, and was measured
        // against this: over a 262,144-entry vocabulary it cost 0.340 ms against `argSort`'s
        // 0.344 ms, because MLX's Metal backend reaches partition through a sort anyway. Selecting
        // and then sorting the selection came to 0.431 ms — one kernel more for no less work.
        let k = (config.topK > 0 && config.topK < vocab) ? config.topK : vocab
        var order = argSort(scaled, axis: 0)[.stride(by: -1)]        // best first
        if k < vocab { order = order[0 ..< k] }
        let ranked = take(scaled, order, axis: 0)

        let probs = softmax(ranked / config.temperature, axis: 0, precise: true)

        // Top-P (nucleus): keep the shortest prefix whose cumulative probability reaches topP.
        // A rank survives when everything ranked above it still sums to less than topP, which is
        // exactly the exclusive prefix sum — the host's "first rank at or above topP, inclusive"
        // written without a loop.
        var kept = probs
        if config.topP < 1.0 {
            kept = which(cumsum(probs, axis: 0, inclusive: false) .< config.topP,
                         probs, MLXArray(Float(0)))
        }

        // Draw by inverse CDF over the kept mass. Counting the ranks strictly below the target
        // gives the first rank at or above it without depending on how `argMax` orders the equal
        // entries of a boolean array. The clamp is the host's fall-through to the last kept rank.
        let keptSum = kept.sum()
        let target = which(keptSum .> 0, keptSum, MLXArray(Float(1))) * uniform
        let rank = (cumsum(kept, axis: 0, inclusive: true) .< target).asType(.int32).sum()
        return take(order, minimum(rank, MLXArray(Int32(order.dim(0) - 1))), axis: 0)
    }
}
