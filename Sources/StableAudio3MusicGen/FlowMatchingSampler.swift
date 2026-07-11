import Foundation
import MLX
import MLXRandom

/// Build the pingpong sampling schedule used by SA3's rectified-flow denoiser.
/// Linear σ from `sigmaMax → 0` in `steps+1` points, then warped by LogSNR
/// shift (`anchor=-6.2, end=2.0`); endpoints clamped to (σmax, 0).
public func buildPingPongSchedule(steps: Int, sigmaMax: Float = 1.0,
                                   useLogSNRShift: Bool = true) -> MLXArray {
    let t = MLX.linspace(sigmaMax, 0.0, count: steps + 1).asType(.float32)
    if !useLogSNRShift {
        return t
    }
    let anchorLogSNR: Float = -6.2
    let logsnrEnd: Float = 2.0
    let logsnr = MLXArray(logsnrEnd) - t * (MLXArray(logsnrEnd) - MLXArray(anchorLogSNR))
    var warped = MLX.sigmoid(-logsnr)
    // Preserve endpoints exactly.
    let zeros = MLXArray.zeros(warped.shape, dtype: .float32)
    let ones  = MLXArray.ones(warped.shape, dtype: .float32)
    warped = MLX.where(t .<= MLXArray(Float(0.0)), zeros, warped)
    warped = MLX.where(t .>= MLXArray(Float(1.0)), ones, warped)
    // Re-anchor start to sigmaMax (preserves the literal sigma at i=0).
    let head = MLXArray([sigmaMax])
    return MLX.concatenated([head, warped[1...]], axis: 0)
}

/// Ping-pong sampler for rf_denoiser models. Per step:
///   denoised = x - σ_curr * v(x, σ_curr)
///   x_next   = (1 - σ_next) * denoised + σ_next * noise   (intermediate steps)
///   x_final  = denoised                                     (last step)
///
/// `modelFn(x, t)` returns velocity prediction `v` matching `x.dtype`. `t` is
/// shape `[B]` with the current sigma broadcast across the batch.
public func sampleFlowPingPong(
    modelFn: (MLXArray, MLXArray) -> MLXArray,
    initial: MLXArray,
    sigmas: MLXArray,
    seed: UInt64,
    onStep: ((Int, Int) -> Void)? = nil
) -> MLXArray {
    var x = initial
    let numSteps = sigmas.dim(0) - 1
    var key = MLXRandom.key(seed)
    for i in 0..<numSteps {
        let sigCurr = sigmas[i]
        let sigNext = sigmas[i + 1]
        let tTensor = MLXArray.ones([x.dim(0)], dtype: x.dtype) * sigCurr.asType(x.dtype)
        let v = modelFn(x, tTensor)
        let denoised = x - sigCurr.asType(x.dtype) * v
        if i < numSteps - 1 && sigNext.item(Float.self) > 0 {
            let split = MLXRandom.split(key: key)
            key = split.0
            let sub = split.1
            let noise = MLXRandom.normal(x.shape, dtype: x.dtype, key: sub)
            let mix = (MLXArray(Float(1.0)).asType(x.dtype) - sigNext.asType(x.dtype)) * denoised
            x = mix + sigNext.asType(x.dtype) * noise
        } else {
            x = denoised
        }
        eval(x)
        onStep?(i + 1, numSteps)
    }
    return x
}

/// Reverse of PatchedPretransform: `[B, 512, T*16] → [B, 2, T*4096]`.
/// einops: `rearrange("b (c h) l -> b c (l h)", h=patchSize)`.
public func patchedDecode(_ patches: MLXArray, patchSize: Int = 256, channels: Int = 2) -> MLXArray {
    let B = patches.dim(0)
    let CH = patches.dim(1)
    let L = patches.dim(2)
    precondition(CH == channels * patchSize, "Expected \(channels * patchSize) channels, got \(CH)")
    let x = patches.reshaped([B, channels, patchSize, L])
                    .transposed(0, 1, 3, 2)
                    .reshaped([B, channels, L * patchSize])
    return x
}
