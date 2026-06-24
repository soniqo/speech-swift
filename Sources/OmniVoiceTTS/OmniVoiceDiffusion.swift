import Foundation
import MLX
import MLXNN

extension OmniVoiceModel {
    /// N-step iterative-unmasking diffusion decode → audio token matrix `[1, C, T]`.
    ///
    /// Each step runs a cond forward (full text+ref+target context) and an uncond
    /// forward (target only), combines them with classifier-free guidance in
    /// log-prob space, scores each still-masked position by confidence minus a
    /// per-codebook layer penalty, and unmasks the top-k positions (assigning the
    /// argmax token). `positionTemperature == 0` here → deterministic (greedy)
    /// position selection, matching the deterministic oracle.
    ///
    /// - condInputIds: `[1, C, condLen]` — style+text+ref+target (target all-mask).
    /// - audioMask: `[1, condLen]` (0/1).
    /// - targetLen: number of audio frames to generate (the trailing positions).
    public func generateTokens(
        condInputIds: MLXArray, audioMask: MLXArray, targetLen: Int,
        numSteps: Int = 16, guidance: Float = 2.0, tShift: Float = 0.1,
        layerPenalty: Float = 5.0
    ) -> MLXArray {
        let C = cfg.numAudioCodebook
        let maskId = cfg.audioMaskId
        let condLen = condInputIds.dim(2)
        let tStart = condLen - targetLen

        var tokens = MLXArray.full([1, C, targetLen], values: MLXArray(Int32(maskId)))
        var cond = condInputIds
        let uncondMask = MLXArray.ones([1, targetLen]).asType(.int32)
        let layerIds = MLXArray((0 ..< C).map { Float($0) }).reshaped([1, C, 1])

        // Time-shifted schedule: how many of the C*T tokens to unmask per step.
        let ts: [Float] = (0 ... numSteps).map {
            let t = Float($0) / Float(numSteps)
            return tShift * t / (1 + (tShift - 1) * t)
        }
        let totalMask = targetLen * C
        var schedule = [Int](); var rem = totalMask
        for step in 0 ..< numSteps {
            let n = step == numSteps - 1
                ? rem
                : min(Int(ceil(Double(totalMask) * Double(ts[step + 1] - ts[step]))), rem)
            schedule.append(n); rem -= n
        }

        let negInf = MLXArray(-Float.greatestFiniteMagnitude)
        for step in 0 ..< numSteps {
            let k = schedule[step]
            if k <= 0 { continue }

            let condLogits = self(inputIds: cond, audioMask: audioMask)[
                0..., 0..., tStart ..< condLen, 0...]                       // [1, C, T, V]
            let uncondLogits = self(inputIds: tokens, audioMask: uncondMask) // [1, C, T, V]

            // CFG in log-prob space, then re-normalise.
            let cLog = logSoftmax(condLogits)
            let uLog = logSoftmax(uncondLogits)
            var logProbs = logSoftmax(cLog + guidance * (cLog - uLog))
            // Never emit the mask token.
            var maskCol = MLXArray.zeros([cfg.audioVocabSize])
            maskCol[maskId] = negInf
            logProbs = logProbs + maskCol.reshaped([1, 1, 1, cfg.audioVocabSize])

            let predTokens = logProbs.argMax(axis: -1).asType(.int32)        // [1, C, T]
            let confidence = logProbs.max(axis: -1)                          // [1, C, T]
            var scores = confidence - layerIds * layerPenalty
            // Only consider still-masked positions.
            scores = MLX.where(tokens .!= MLXArray(Int32(maskId)), negInf, scores)

            // Top-k positions to unmask this step (deterministic).
            let flat = scores.reshaped([C * targetLen])
            let order = MLX.argSort(-flat)                                   // descending
            let topk = order[0 ..< k]
            var pick = MLXArray.zeros([C * targetLen]).asType(.int32)
            pick[topk] = MLXArray.ones([k]).asType(.int32)
            let pickMask = (pick.reshaped([1, C, targetLen]) .!= MLXArray(Int32(0)))

            tokens = MLX.where(pickMask, predTokens, tokens)
            cond[0..., 0..., tStart ..< condLen] = tokens
            MLX.eval(tokens, cond)
        }
        return tokens
    }

    private func logSoftmax(_ x: MLXArray) -> MLXArray {
        x - MLX.logSumExp(x, axis: -1, keepDims: true)
    }
}
