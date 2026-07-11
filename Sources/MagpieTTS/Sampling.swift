import Foundation
import MLX
import MLXRandom

/// Top-k filter — set logits below the k-th highest to a very negative
/// constant along the last axis. ``k <= 0`` or ``k >= V`` returns ``logits``
/// unmodified (greedy and full distribution).
public func topKFilter(_ logits: MLXArray, k: Int) -> MLXArray {
    let V = logits.dim(-1)
    if k <= 0 || k >= V { return logits }
    // Sort ascending along last axis, then the (V - k)-th entry is the k-th
    // largest. Pull that slice out as a per-row threshold and zero anything
    // smaller with -1e30.
    let sortedAsc = MLX.sorted(logits, axis: -1)
    let threshold = sortedAsc.take(MLXArray(Int32(V - k)), axis: -1)
    let mask = logits .< threshold
    return MLX.where(mask, MLXArray(Float(-1e30)), logits)
}

/// Add ``-1e30`` to specific vocab IDs to forbid them (BOS/EOS).
public func forbidIds(_ logits: MLXArray, ids: [Int]) -> MLXArray {
    if ids.isEmpty { return logits }
    let V = logits.dim(-1)
    let arange = MLXArray(0..<Int32(V))
    var penalty = MLXArray(Array(repeating: Float(0), count: V))
    for i in ids {
        let hit = (arange .== MLXArray(Int32(i))).asType(Float.self)
        penalty = penalty + MLXArray(Float(-1e30)) * hit
    }
    return logits + penalty
}

/// Sample a single int32 token from ``logits`` (last axis). Greedy when
/// ``temperature <= 1e-3``; otherwise Gumbel-max over top-k filtered logits.
public func sampleTopK(_ logits: MLXArray, temperature: Float, k: Int) -> MLXArray {
    if temperature <= 1e-3 {
        return argMax(logits, axis: -1).asType(Int32.self)
    }
    let scaled = topKFilter(logits / MLXArray(temperature), k: k)
    let u = MLXRandom.uniform(low: Float(0), high: Float(1), scaled.shape)
    let gumbel = -log(-log(u + MLXArray(Float(1e-20))) + MLXArray(Float(1e-20)))
    return argMax(scaled + gumbel, axis: -1).asType(Int32.self)
}
