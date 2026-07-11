import Foundation
import MLX
import MLXRandom

/// Top-p (nucleus) sample one token per row from logits.
///
/// `logits: [..., card]`. Returns int32 `[..., 1]`. Faithful port of the
/// Python reference: sort ascending, cumsum from the largest, mask out tokens
/// whose cumulative probability exceeds `topP`, force-keep the single largest
/// token, then `categorical` over the masked probabilities (log space).
public func sampleTopP(_ logits: MLXArray, topP: Float, temperature: Float) -> MLXArray {
    let tempArr = MLXArray(max(temperature, 1e-2))
    let probs = softmax(logits / tempArr, axis: -1)
    let sortedIdx = argSort(probs, axis: -1)
    let sortedP = takeAlong(probs, sortedIdx, axis: -1)
    let lastAxis = sortedP.ndim - 1
    // Cumulative from largest down.
    let desc = sortedP[.ellipsis, .stride(by: -1)]
    let cum = cumsum(desc, axis: lastAxis)[.ellipsis, .stride(by: -1)]
    let keep = cum .<= MLXArray(topP)
    let N = sortedP.dim(lastAxis)
    let arange = MLXArray(0..<Int32(N))
    let lastCol = arange .== MLXArray(Int32(N - 1))
    let keepWithLast = MLX.logicalOr(keep, lastCol.reshaped(Array(repeating: 1, count: lastAxis) + [N]))
    let masked = MLX.where(keepWithLast, sortedP, MLXArray(Float(0)))
    let logits2 = log(masked + MLXArray(Float(1e-12)))
    let choice = categorical(logits2, axis: lastAxis)
    let choiceExpanded = choice.expandedDimensions(axis: lastAxis)
    return takeAlong(sortedIdx, choiceExpanded, axis: lastAxis)
}

/// Build a `[B, 1, N]` bool mask from `[B, 1, M]` integer indices via
/// broadcast equality + any-reduce (avoids needing a scatter primitive).
public func positionsToMask(indices: MLXArray, N: Int) -> MLXArray {
    let rng = MLXArray(0..<Int32(N)).reshaped([1, 1, 1, N])
    let idx = indices.expandedDimensions(axis: -1)
    let hits = rng .== idx
    return any(hits, axis: -2)
}

/// Write `stageSeq[B, 1, T]` into `gen[B, K, T]` at codebook index `stage`.
public func stageWrite(gen: MLXArray, stageSeq: MLXArray, stage: Int) -> MLXArray {
    let K = gen.dim(1)
    let selector = MLXArray(0..<Int32(K)).reshaped([1, K, 1]) .== MLXArray(Int32(stage))
    return MLX.where(selector, stageSeq, gen)
}
