import Foundation

/// Standalone token sampler for text generation.
///
/// Supports temperature scaling, top-K filtering, top-P (nucleus) filtering,
/// and repetition penalty.
enum ChatSampler {

    /// Sample a token from logits using the given sampling config.
    ///
    /// - Parameters:
    ///   - logits: Raw logits array of size vocab_size
    ///   - config: Sampling parameters (temperature, topK, topP, repetitionPenalty)
    ///   - previousTokens: Recently generated tokens for repetition penalty
    /// - Returns: Sampled token index
    static func sample(
        logits: [Float],
        config: ChatSamplingConfig,
        previousTokens: [Int] = []
    ) -> Int {
        let vocab = logits.count
        if vocab == 0 { return 0 }

        // Repetition penalty — one O(V) copy + O(seen) in-place updates.
        var scaled = logits
        if config.repetitionPenalty > 1.0 {
            let p = config.repetitionPenalty
            for tid in Set(previousTokens.suffix(64)) where tid >= 0 && tid < vocab {
                scaled[tid] = scaled[tid] > 0 ? scaled[tid] / p : scaled[tid] * p
            }
        }

        // Greedy (argmax) when temperature is 0.
        if config.temperature <= 0 {
            var best = 0
            for i in 1..<vocab where scaled[i] > scaled[best] { best = i }
            return best
        }

        // Candidate set = top-K indices. A size-K min-heap finds them in O(V·log K), instead of
        // the previous O(V·log V) full sort of the entire vocab (~248K tokens) — which dominated
        // generation latency. top-K by logit == top-K by softmax prob (softmax is monotonic), so
        // selecting on logits is equivalent and lets everything after operate on just K elements.
        let k = (config.topK > 0 && config.topK < vocab) ? config.topK : vocab
        var cand: [Int]
        if k >= vocab {
            cand = Array(0..<vocab)
        } else {
            var heap = [Int]()
            heap.reserveCapacity(k)
            for i in 0..<vocab {
                if heap.count < k {
                    heap.append(i)
                    var c = heap.count - 1
                    while c > 0 {
                        let parent = (c - 1) / 2
                        if scaled[heap[c]] < scaled[heap[parent]] { heap.swapAt(c, parent); c = parent } else { break }
                    }
                } else if scaled[i] > scaled[heap[0]] {
                    heap[0] = i
                    var parent = 0
                    while true {
                        let l = 2 * parent + 1, r = 2 * parent + 2
                        var smallest = parent
                        if l < k && scaled[heap[l]] < scaled[heap[smallest]] { smallest = l }
                        if r < k && scaled[heap[r]] < scaled[heap[smallest]] { smallest = r }
                        if smallest == parent { break }
                        heap.swapAt(smallest, parent); parent = smallest
                    }
                }
            }
            cand = heap
        }

        // Softmax over the candidates (with temperature).
        let t = config.temperature
        var maxL = -Float.greatestFiniteMagnitude
        for i in cand { let v = scaled[i] / t; if v > maxL { maxL = v } }
        var probs = [Float](repeating: 0, count: cand.count)
        var sum: Float = 0
        for (j, i) in cand.enumerated() { let e = exp(scaled[i] / t - maxL); probs[j] = e; sum += e }
        if sum > 0 { for j in probs.indices { probs[j] /= sum } }

        // Sort candidates by prob desc (K is small) for top-P + sampling.
        var order = Array(0..<cand.count)
        order.sort { probs[$0] > probs[$1] }

        // Top-P (nucleus): keep the shortest prefix whose cumulative prob reaches topP.
        var cut = order.count
        if config.topP < 1.0 {
            var cum: Float = 0
            for (rank, j) in order.enumerated() { cum += probs[j]; if cum >= config.topP { cut = rank + 1; break } }
        }

        // Sample within the kept nucleus (renormalized).
        var keptSum: Float = 0
        for r in 0..<cut { keptSum += probs[order[r]] }
        let target = Float.random(in: 0..<1) * (keptSum > 0 ? keptSum : 1)
        var c: Float = 0
        for r in 0..<cut {
            c += probs[order[r]]
            if c >= target { return cand[order[r]] }
        }
        return cand[order[cut - 1]]
    }
}
