import Foundation

/// Inverse Finite Scalar Quantisation for Magpie's nano-codec.
///
/// The codec takes continuous 32-dim latents per frame; the AR decoder
/// emits 8 codebook indices per frame. The mapping is fixed by NeMo:
///
///     NUM_LEVELS = [8, 7, 6, 6]
///     base       = cumprod([1, 8, 7, 6]) = [1, 8, 56, 336]
///
/// For each codebook value `i ∈ [0, 2016)`:
///   for j in 0..3:
///     d_j      = (i // base[j]) % level[j]
///     dequant  = (d_j - level[j]/2) / (level[j]/2)
///
/// 8 codebooks × 4 dequants = 32 latent dims per frame.
public enum MagpieCoreMLFSQ {

    /// Convert one frame of 8 codes to a 32-dim Float32 latent.
    public static func decodeFrame(codes: [Int32]) -> [Float] {
        precondition(codes.count == MagpieCoreMLConstants.fsqNumGroups,
                     "expected 8 codes, got \(codes.count)")
        var out = [Float](repeating: 0, count: MagpieCoreMLConstants.fsqLatentDim)
        for cb in 0..<MagpieCoreMLConstants.fsqNumGroups {
            let code = codes[cb]
            for j in 0..<MagpieCoreMLConstants.fsqDimPerGroup {
                let base = MagpieCoreMLConstants.fsqBase[j]
                let level = MagpieCoreMLConstants.fsqLevels[j]
                let nonneg = (code / base) % level
                let halfL = level / 2
                let dequant = Float(nonneg - halfL) / Float(halfL)
                // Layout matches NeMo: per-codebook 4-dim block, concatenated.
                out[cb * MagpieCoreMLConstants.fsqDimPerGroup + j] = dequant
            }
        }
        return out
    }

    /// Convert a sequence of frames `[T][8]` to a `(1, 32, T)` row-major
    /// Float32 buffer suitable for the nano-codec input. The model is
    /// traced with `T = nanocodecFramesPerWindow = 64`; callers should
    /// chunk longer sequences and concatenate the audio outputs.
    ///
    /// - Important: the codec was traced in **NCL layout** (channels axis 1,
    ///   time axis 2), so the returned buffer is `(1, 32, 64)` with values
    ///   indexed as `[1][channel][t]`. Each frame's 32-dim latent is
    ///   scattered across the channel axis at column `t`.
    public static func decodeWindow(frames: ArraySlice<[Int32]>) -> [Float] {
        let T = MagpieCoreMLConstants.nanocodecFramesPerWindow
        let C = MagpieCoreMLConstants.fsqLatentDim
        var out = [Float](repeating: 0, count: C * T)
        let count = min(frames.count, T)
        let base = frames.startIndex
        for tIdx in 0..<count {
            let frame = frames[base + tIdx]
            let latent = decodeFrame(codes: frame)
            // Write latent column-major into the (C, T) buffer: out[c, t] = latent[c]
            for c in 0..<C {
                out[c * T + tIdx] = latent[c]
            }
        }
        // Remaining time steps (when frames.count < T) stay zero — the
        // codec's causal HiFi-GAN will treat that as silence, and we trim
        // the audio output to the actual count anyway.
        return out
    }
}
