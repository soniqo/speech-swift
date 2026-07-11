import Foundation

enum IndicMioSampler {
    static func sample(
        logits: [Float],
        config: IndicMioSamplingConfig,
        previousTokens: [Int] = []
    ) -> Int {
        let vocab = logits.count
        guard vocab > 0 else { return 0 }

        var scaled = logits
        if config.repetitionPenalty > 1 {
            let penalty = config.repetitionPenalty
            for token in Set(previousTokens.suffix(64)) where token >= 0 && token < vocab {
                scaled[token] = scaled[token] > 0 ? scaled[token] / penalty : scaled[token] * penalty
            }
        }

        if config.temperature <= 0 {
            return argmax(scaled)
        }

        let k = config.topK > 0 && config.topK < vocab ? config.topK : vocab
        var candidates: [Int]
        if k >= vocab {
            candidates = Array(0..<vocab)
        } else {
            candidates = topKIndices(scaled, k: k)
        }

        let temperature = max(config.temperature, 1e-5)
        var maxLogit = -Float.greatestFiniteMagnitude
        for index in candidates {
            maxLogit = max(maxLogit, scaled[index] / temperature)
        }

        var probs = [Float](repeating: 0, count: candidates.count)
        var total: Float = 0
        for (i, index) in candidates.enumerated() {
            let p = exp(scaled[index] / temperature - maxLogit)
            probs[i] = p
            total += p
        }
        if total > 0 {
            for i in probs.indices { probs[i] /= total }
        }

        var order = Array(probs.indices)
        order.sort { probs[$0] > probs[$1] }

        var cutoff = order.count
        if config.topP < 1 {
            var cumulative: Float = 0
            for (rank, index) in order.enumerated() {
                cumulative += probs[index]
                if cumulative >= config.topP {
                    cutoff = rank + 1
                    break
                }
            }
        }

        var keptSum: Float = 0
        for rank in 0..<cutoff {
            keptSum += probs[order[rank]]
        }

        let target = Float.random(in: 0..<1) * max(keptSum, 1e-12)
        var cumulative: Float = 0
        for rank in 0..<cutoff {
            let j = order[rank]
            cumulative += probs[j]
            if cumulative >= target {
                return candidates[j]
            }
        }
        return candidates[order[max(0, cutoff - 1)]]
    }

    private static func argmax(_ values: [Float]) -> Int {
        var best = 0
        for i in 1..<values.count where values[i] > values[best] {
            best = i
        }
        return best
    }

    private static func topKIndices(_ values: [Float], k: Int) -> [Int] {
        var heap: [Int] = []
        heap.reserveCapacity(k)

        func less(_ lhs: Int, _ rhs: Int) -> Bool {
            values[lhs] < values[rhs]
        }

        func siftUp(_ index: Int, _ heap: inout [Int]) {
            var child = index
            while child > 0 {
                let parent = (child - 1) / 2
                if less(heap[child], heap[parent]) {
                    heap.swapAt(child, parent)
                    child = parent
                } else {
                    break
                }
            }
        }

        func siftDown(_ index: Int, _ heap: inout [Int]) {
            var parent = index
            while true {
                let left = 2 * parent + 1
                let right = left + 1
                var smallest = parent
                if left < heap.count && less(heap[left], heap[smallest]) {
                    smallest = left
                }
                if right < heap.count && less(heap[right], heap[smallest]) {
                    smallest = right
                }
                if smallest == parent { break }
                heap.swapAt(parent, smallest)
                parent = smallest
            }
        }

        for index in values.indices {
            if heap.count < k {
                heap.append(index)
                siftUp(heap.count - 1, &heap)
            } else if values[index] > values[heap[0]] {
                heap[0] = index
                siftDown(0, &heap)
            }
        }
        return heap
    }
}
