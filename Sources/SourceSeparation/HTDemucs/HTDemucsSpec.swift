import Foundation
import MLX

/// STFT / iSTFT + complex-as-channels (CaC) for Hybrid Transformer Demucs,
/// matching demucs `spec.py` (`spectro`/`ispectro`) and `htdemucs._spec`/`_ispec`.
///
/// The model normalizes the magnitude by its own per-input mean/std, so the
/// pipeline is invariant to the STFT's global scale — we only need an
/// invertible STFT/iSTFT pair (COLA Hann, hop = nfft/4) plus the exact demucs
/// pad/trim arithmetic so the frequency- and time-branch lengths align.
///
/// NOTE: numerical parity vs the reference is validated in Phase D (torch dump
/// → Swift compare). Conventions here follow torch.stft(center=True,
/// pad_mode="reflect", normalized=True) functionally up to the scale that the
/// mean/std normalization cancels.
struct HTDemucsSpec {
    let nfft: Int
    let hop: Int
    let window: MLXArray   // periodic Hann, length nfft

    init(nfft: Int = 4096, hop: Int = 1024) {
        self.nfft = nfft
        self.hop = hop
        let w = (0..<nfft).map {
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(nfft)))
        }
        self.window = MLXArray(w)
    }

    // MARK: - reflect padding (PadMode has no .reflect in MLX Swift)

    /// Reflect-pad the last axis without repeating the edge sample (torch
    /// "reflect"): left = x[left..1], right = x[L-2..L-1-right].
    private func reflectPad(_ x: MLXArray, _ left: Int, _ right: Int) -> MLXArray {
        let L = x.dim(-1)
        var parts: [MLXArray] = []
        if left > 0 {
            let idx = MLXArray((1...left).reversed().map { Int32($0) })
            parts.append(take(x, idx, axis: -1))
        }
        parts.append(x)
        if right > 0 {
            let idx = MLXArray((0..<right).map { Int32(L - 2 - $0) })
            parts.append(take(x, idx, axis: -1))
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: -1)
    }

    private func complexZeros(_ shape: [Int]) -> MLXArray {
        MLXArray.zeros(shape).asImaginary()  // 0 + 0i
    }

    // MARK: - core STFT / iSTFT

    /// torch.stft equivalent. Input `[N, T]` real → complex `[N, freqs, frames]`,
    /// `freqs = nfft/2 + 1`, center=True (reflect nfft/2 each side), Hann window.
    func stft(_ x: MLXArray) -> MLXArray {
        let pad = nfft / 2
        let xp = reflectPad(x, pad, pad)
        let N = xp.dim(0)
        let Lp = xp.dim(1)
        let frames = 1 + (Lp - nfft) / hop
        // strided frames: (n, f, k) -> xp[n, f*hop + k]
        let framed = asStrided(xp, [N, frames, nfft], strides: [Lp, hop, 1], offset: 0)
        let win = window.reshaped([1, 1, nfft])
        let z = rfft(framed * win, axis: -1)     // [N, frames, freqs] complex
        return z.transposed(0, 2, 1)             // [N, freqs, frames]
    }

    /// Inverse of `stft`. Complex `[N, freqs, frames]` → real `[N, length]`.
    /// Overlap-add (k = nfft/hop sub-windows) with window² normalization, then
    /// trim the center pad.
    func istft(_ z: MLXArray, length: Int) -> MLXArray {
        let k = nfft / hop
        let zt = z.transposed(0, 2, 1)                 // [N, frames, freqs]
        var frames = irfft(zt, n: nfft, axis: -1)      // [N, frames, nfft] real
        let N = frames.dim(0)
        let T = frames.dim(1)
        frames = frames * window.reshaped([1, 1, nfft])

        // overlap-add via reshape + pad-and-sum (valid since nfft % hop == 0)
        let sub = frames.reshaped([N, T, k, hop])
        var accum: MLXArray? = nil
        for j in 0..<k {
            let slice = sub[0..., 0..., j, 0...]        // [N, T, hop]
            let padded = MLX.padded(
                slice, widths: [IntOrPair((0, 0)), IntOrPair((j, (k - 1) - j)), IntOrPair((0, 0))])
            accum = accum.map { $0 + padded } ?? padded
        }
        let combined = accum!.reshaped([N, (T + k - 1) * hop])

        // window² overlap-add (depends only on T) for COLA normalization
        let w2 = (window * window).reshaped([k, hop])
        var w2Accum: MLXArray? = nil
        for j in 0..<k {
            let row = MLX.broadcast(w2[j, 0...].reshaped([1, hop]), to: [T, hop])
            let padded = MLX.padded(row, widths: [IntOrPair((j, (k - 1) - j)), IntOrPair((0, 0))])
            w2Accum = w2Accum.map { $0 + padded } ?? padded
        }
        let winSum = w2Accum!.reshaped([(T + k - 1) * hop])
        let normalized = combined / MLX.maximum(winSum, MLXArray(Float(1e-8)))

        let pad = nfft / 2
        return normalized[0..., pad ..< (pad + length)]
    }

    // MARK: - demucs _spec / _ispec wrappers

    /// `htdemucs._spec`: input `[B, C, T]` real → complex `[B, C, nfft/2, le]`,
    /// `le = ceil(T/hop)`. Extra reflect pad + drop top freq bin + trim 2 frames
    /// each side keeps `out_size == in_size/hop` and aligns with the time branch.
    func spec(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), C = x.dim(1), T = x.dim(2)
        let le = (T + hop - 1) / hop
        let pad = hop / 2 * 3
        let xp = reflectPad(x, pad, pad + le * hop - T)   // [B, C, (le+3)*hop]
        let Lp = xp.dim(2)
        let z = stft(xp.reshaped([B * C, Lp]))            // [B*C, nfft/2+1, le+4]
        // drop last freq bin (→ nfft/2) and 2 frames each side (→ le)
        let cropped = z[0..., 0 ..< (nfft / 2), 2 ..< (2 + le)]
        return cropped.reshaped([B, C, nfft / 2, le])
    }

    /// `htdemucs._ispec`: complex `[B, C, nfft/2, le]` → real `[B, C, length]`.
    func ispec(_ z: MLXArray, length: Int) -> MLXArray {
        let dims = z.shape
        let B = dims[0], C = dims[1], Fr = dims[2], le = dims[3]
        // pad freq nfft/2 -> nfft/2+1 (append zero bin)
        var zp = concatenated([z, complexZeros([B, C, 1, le])], axis: 2)
        // pad frames (2, 2) with zeros
        zp = concatenated(
            [complexZeros([B, C, Fr + 1, 2]), zp, complexZeros([B, C, Fr + 1, 2])], axis: 3)
        let pad = hop / 2 * 3
        let le2 = hop * ((length + hop - 1) / hop) + 2 * pad
        let n2 = zp.dim(-1)
        let wav = istft(zp.reshaped([B * C, nfft / 2 + 1, n2]), length: le2)  // [B*C, le2]
        let trimmed = wav[0..., pad ..< (pad + length)]
        return trimmed.reshaped([B, C, length])
    }

    // MARK: - complex-as-channels (cac=True)

    /// `_magnitude` for cac: complex `[B, C, Fr, T]` → real `[B, 2C, Fr, T]`,
    /// channel layout `[c0_re, c0_im, c1_re, c1_im, ...]`.
    func magnitudeCaC(_ z: MLXArray) -> MLXArray {
        let B = z.dim(0), C = z.dim(1), Fr = z.dim(2), T = z.dim(3)
        let re = z.realPart()                 // [B, C, Fr, T]
        let im = z.imaginaryPart()
        // stack -> [B, C, 2, Fr, T] -> [B, 2C, Fr, T]
        let stacked = MLX.stacked([re, im], axis: 2)
        return stacked.reshaped([B, 2 * C, Fr, T])
    }

    /// Inverse of `magnitudeCaC` over `S` sources: real `[B, S*2C, Fr, T]` →
    /// complex `[B, S, C, Fr, T]`. (cac mask = full spectrogram, `z` ignored.)
    func maskCaC(_ m: MLXArray, sources S: Int, channels C: Int) -> MLXArray {
        let B = m.dim(0), Fr = m.dim(2), T = m.dim(3)
        let r = m.reshaped([B, S, C, 2, Fr, T])
        let re = r[0..., 0..., 0..., 0, 0..., 0...]   // [B, S, C, Fr, T]
        let im = r[0..., 0..., 0..., 1, 0..., 0...]
        return re + im.asImaginary()
    }
}
