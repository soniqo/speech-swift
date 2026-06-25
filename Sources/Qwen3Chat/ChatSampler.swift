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
            for tokenId in Set(previousTokens.suffix(64)) where tokenId >= 0 && tokenId < vocab {
                scaled[tokenId] = scaled[tokenId] > 0 ? scaled[tokenId] / p : scaled[tokenId] * p
            }
        }

        // Greedy (argmax) when temperature is 0.
        if config.temperature <= 0 {
            var best = 0
            for i in 1..<vocab where scaled[i] > scaled[best] { best = i }
            return best
        }

        // Candidate set = top-K indices. A size-K min-heap finds them in O(V*log K), instead of
        // the previous O(V*log V) full sort of the entire vocab (~248K tokens) — which dominated
        // generation latency. Top-K by logit equals top-K by softmax probability because softmax is
        // monotonic, so everything after this operates on just K elements.
        let k = (config.topK > 0 && config.topK < vocab) ? config.topK : vocab
        var candidates: [Int]
        if k >= vocab {
            candidates = Array(0..<vocab)
        } else {
            var heap: [Int] = []
            heap.reserveCapacity(k)
            for i in 0..<vocab {
                if heap.count < k {
                    heap.append(i)
                    var child = heap.count - 1
                    while child > 0 {
                        let parent = (child - 1) / 2
                        if scaled[heap[child]] < scaled[heap[parent]] {
                            heap.swapAt(child, parent)
                            child = parent
                        } else {
                            break
                        }
                    }
                } else if scaled[i] > scaled[heap[0]] {
                    heap[0] = i
                    var parent = 0
                    while true {
                        let left = 2 * parent + 1
                        let right = 2 * parent + 2
                        var smallest = parent
                        if left < k && scaled[heap[left]] < scaled[heap[smallest]] { smallest = left }
                        if right < k && scaled[heap[right]] < scaled[heap[smallest]] { smallest = right }
                        if smallest == parent { break }
                        heap.swapAt(smallest, parent)
                        parent = smallest
                    }
                }
            }
            candidates = heap
        }

        // Softmax over candidates only, with temperature.
        let temperature = config.temperature
        var maxLogit = -Float.greatestFiniteMagnitude
        for i in candidates {
            let value = scaled[i] / temperature
            if value > maxLogit { maxLogit = value }
        }

        var probs = [Float](repeating: 0, count: candidates.count)
        var sum: Float = 0
        for (index, tokenId) in candidates.enumerated() {
            let p = exp(scaled[tokenId] / temperature - maxLogit)
            probs[index] = p
            sum += p
        }
        if sum > 0 {
            for i in probs.indices { probs[i] /= sum }
        }

        // Sort candidates by probability descending for top-P and sampling. K is normally small.
        var order = Array(0..<candidates.count)
        order.sort { probs[$0] > probs[$1] }

        // Top-P (nucleus): keep the shortest prefix whose cumulative probability reaches topP.
        var cutoff = order.count
        if config.topP < 1.0 {
            var cumulative: Float = 0
            for (rank, index) in order.enumerated() {
                cumulative += probs[index]
                if cumulative >= config.topP {
                    cutoff = rank + 1
                    break
                }
            }
        }

        // Sample within the kept nucleus, renormalized.
        var keptSum: Float = 0
        for rank in 0..<cutoff { keptSum += probs[order[rank]] }
        let target = Float.random(in: 0..<1) * (keptSum > 0 ? keptSum : 1)
        var cumulative: Float = 0
        for rank in 0..<cutoff {
            cumulative += probs[order[rank]]
            if cumulative >= target { return candidates[order[rank]] }
        }
        return candidates[order[cutoff - 1]]
    }
}
