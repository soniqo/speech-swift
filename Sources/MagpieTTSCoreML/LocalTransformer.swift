// Adapted from FluidInference/FluidAudio (Apache-2.0)
// https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/Magpie/LocalTransformer/MagpieLocalTransformer.swift
//
// One-layer transformer (d=256, FFN=1024, 1 attention head) that produces
// the 8 codebook tokens per frame conditioned on the decoder hidden state.
// Pure Swift over Accelerate so the CoreML pipeline target carries no MLX
// dependency.

import Accelerate
import Foundation

public struct MagpieCoreMLLocalTransformer: Sendable {
    public let weights: MagpieCoreMLLocalTransformerWeights

    public init(weights: MagpieCoreMLLocalTransformerWeights) {
        self.weights = weights
    }

    /// Project a `[dModel]` decoder hidden through `inProj` into a
    /// `[localDim]` LT-space vector. Caller appends positional embedding
    /// inside ``forward(...)``.
    public func projectInput(hidden: [Float]) -> [Float] {
        precondition(hidden.count == weights.dModel)
        let D = weights.localDim
        let M = weights.dModel
        var out = weights.inProjBias  // copy bias as the GEMV accumulator
        weights.inProjWeight.withUnsafeBufferPointer { wPtr in
            hidden.withUnsafeBufferPointer { hPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(D), Int32(M),
                        1.0,
                        wPtr.baseAddress, Int32(M),
                        hPtr.baseAddress, 1,
                        1.0,
                        outPtr.baseAddress, 1)
                }
            }
        }
        return out
    }

    /// Forward pass over a sequence of length `T` (T ≤ maxPositions). Adds
    /// positional embeddings, runs pre-norm causal attention, then pre-norm
    /// FFN with tanh-GELU. Returns the same `[T * localDim]` shape.
    public func forward(sequence: [Float], length T: Int) -> [Float] {
        precondition(sequence.count >= T * weights.localDim, "sequence buffer too small")
        precondition(T <= weights.maxPositions, "sequence length exceeds maxPositions")

        let D = weights.localDim
        let ffnD = weights.ffnDim

        // x = sequence[:T*D] + posEmb[:T*D]
        var x = Swift.Array(sequence.prefix(T * D))
        addPositional(into: &x, length: T)

        // ── Pre-norm self-attention ──
        var xNorm = layerNorm(x, length: T, weight: weights.norm1Weight)

        // QKV = xNorm @ saQkvWeight.T  → (T, 3D)
        var qkv = Swift.Array<Float>(repeating: 0, count: T * 3 * D)
        matmulTransB(
            a: xNorm, aRows: T, aCols: D,
            b: weights.saQkvWeight, bRows: 3 * D, bCols: D,
            out: &qkv)

        // Split QKV → Q, K, V (each (T, D))
        var q = Swift.Array<Float>(repeating: 0, count: T * D)
        var k = Swift.Array<Float>(repeating: 0, count: T * D)
        var v = Swift.Array<Float>(repeating: 0, count: T * D)
        let bytesPerRow = D * MemoryLayout<Float>.size
        qkv.withUnsafeBufferPointer { src in
            q.withUnsafeMutableBufferPointer { qB in
                k.withUnsafeMutableBufferPointer { kB in
                    v.withUnsafeMutableBufferPointer { vB in
                        guard let s = src.baseAddress,
                              let qb = qB.baseAddress,
                              let kb = kB.baseAddress,
                              let vb = vB.baseAddress else { return }
                        for t in 0..<T {
                            let row = s.advanced(by: t * 3 * D)
                            let dst = t * D
                            memcpy(qb.advanced(by: dst), row, bytesPerRow)
                            memcpy(kb.advanced(by: dst), row.advanced(by: D), bytesPerRow)
                            memcpy(vb.advanced(by: dst), row.advanced(by: 2 * D), bytesPerRow)
                        }
                    }
                }
            }
        }

        // attn = Q @ K.T * (1/sqrt(D))   → (T, T)
        var attn = Swift.Array<Float>(repeating: 0, count: T * T)
        matmulTransB(
            a: q, aRows: T, aCols: D,
            b: k, bRows: T, bCols: D,
            out: &attn)
        let scale = Float(1.0 / sqrt(Double(D)))
        var scaleVar = scale
        vDSP_vsmul(attn, 1, &scaleVar, &attn, 1, vDSP_Length(T * T))

        // Causal mask + softmax (numerically stable: subtract row-max).
        for t in 0..<T {
            var maxVal: Float = -.infinity
            for j in 0...t where attn[t * T + j] > maxVal {
                maxVal = attn[t * T + j]
            }
            var denom: Float = 0
            for j in 0..<T {
                if j <= t {
                    let e = expf(attn[t * T + j] - maxVal)
                    attn[t * T + j] = e
                    denom += e
                } else {
                    attn[t * T + j] = 0
                }
            }
            if denom > 0 {
                let inv = 1.0 / denom
                for j in 0...t { attn[t * T + j] *= inv }
            }
        }

        // saOut = attn @ V             → (T, D)
        var saOut = Swift.Array<Float>(repeating: 0, count: T * D)
        matmul(
            a: attn, aRows: T, aCols: T,
            b: v,    bRows: T, bCols: D,
            out: &saOut)

        // saProj = saOut @ saOWeight.T → (T, D)
        var saProj = Swift.Array<Float>(repeating: 0, count: T * D)
        matmulTransB(
            a: saOut, aRows: T, aCols: D,
            b: weights.saOWeight, bRows: D, bCols: D,
            out: &saProj)

        // x += saProj
        vDSP_vadd(x, 1, saProj, 1, &x, 1, vDSP_Length(T * D))

        // ── Pre-norm FFN ──
        xNorm = layerNorm(x, length: T, weight: weights.norm2Weight)

        var h = Swift.Array<Float>(repeating: 0, count: T * ffnD)
        matmulTransB(
            a: xNorm, aRows: T, aCols: D,
            b: weights.ffnConv1Weight, bRows: ffnD, bCols: D,
            out: &h)
        applyGeluTanh(into: &h)

        var ffnOut = Swift.Array<Float>(repeating: 0, count: T * D)
        matmulTransB(
            a: h, aRows: T, aCols: ffnD,
            b: weights.ffnConv2Weight, bRows: D, bCols: ffnD,
            out: &ffnOut)
        vDSP_vadd(x, 1, ffnOut, 1, &x, 1, vDSP_Length(T * D))

        return x
    }

    /// Project the last LT hidden through ``outProjWeights[codebook]`` →
    /// `[numCodes]` logits for that codebook.
    public func codebookLogits(lastHidden: [Float], codebook: Int) -> [Float] {
        precondition(lastHidden.count == weights.localDim)
        let V = weights.numCodesPerCodebook
        let D = weights.localDim
        var out = weights.outProjBiases[codebook]  // (V,)
        weights.outProjWeights[codebook].withUnsafeBufferPointer { wPtr in
            lastHidden.withUnsafeBufferPointer { hPtr in
                out.withUnsafeMutableBufferPointer { oPtr in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans,
                        Int32(V), Int32(D),
                        1.0,
                        wPtr.baseAddress, Int32(D),
                        hPtr.baseAddress, 1,
                        1.0,
                        oPtr.baseAddress, 1)
                }
            }
        }
        return out
    }

    // MARK: - Internal kernels

    private func addPositional(into buffer: inout [Float], length T: Int) {
        let D = weights.localDim
        weights.posEmbedding.withUnsafeBufferPointer { posPtr in
            buffer.withUnsafeMutableBufferPointer { bufPtr in
                guard let pos = posPtr.baseAddress, let buf = bufPtr.baseAddress
                else { return }
                for t in 0..<T {
                    vDSP_vadd(
                        buf.advanced(by: t * D), 1,
                        pos.advanced(by: t * D), 1,
                        buf.advanced(by: t * D), 1,
                        vDSP_Length(D))
                }
            }
        }
    }

    private func layerNorm(_ x: [Float], length T: Int, weight: [Float]) -> [Float] {
        let D = weights.localDim
        var out = Swift.Array<Float>(repeating: 0, count: T * D)
        let eps: Float = 1e-5
        x.withUnsafeBufferPointer { xPtr in
            weight.withUnsafeBufferPointer { wPtr in
                out.withUnsafeMutableBufferPointer { outPtr in
                    guard let xBase = xPtr.baseAddress,
                          let wBase = wPtr.baseAddress,
                          let outBase = outPtr.baseAddress else { return }
                    for t in 0..<T {
                        let row = xBase.advanced(by: t * D)
                        let oRow = outBase.advanced(by: t * D)
                        var mean: Float = 0
                        vDSP_meanv(row, 1, &mean, vDSP_Length(D))
                        var negMean = -mean
                        vDSP_vsadd(row, 1, &negMean, oRow, 1, vDSP_Length(D))
                        var meanSq: Float = 0
                        vDSP_measqv(oRow, 1, &meanSq, vDSP_Length(D))
                        var invStd = 1.0 / sqrt(meanSq + eps)
                        vDSP_vsmul(oRow, 1, &invStd, oRow, 1, vDSP_Length(D))
                        vDSP_vmul(oRow, 1, wBase, 1, oRow, 1, vDSP_Length(D))
                    }
                }
            }
        }
        return out
    }

    private func matmul(
        a: [Float], aRows M: Int, aCols K: Int,
        b: [Float], bRows: Int, bCols N: Int,
        out: inout [Float]
    ) {
        precondition(K == bRows, "matmul inner dim mismatch")
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                out.withUnsafeMutableBufferPointer { op in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0,
                        ap.baseAddress, Int32(K),
                        bp.baseAddress, Int32(N),
                        0.0,
                        op.baseAddress, Int32(N))
                }
            }
        }
    }

    private func matmulTransB(
        a: [Float], aRows M: Int, aCols K: Int,
        b: [Float], bRows N: Int, bCols bk: Int,
        out: inout [Float]
    ) {
        precondition(K == bk, "matmulTransB inner dim mismatch")
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                out.withUnsafeMutableBufferPointer { op in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(M), Int32(N), Int32(K),
                        1.0,
                        ap.baseAddress, Int32(K),
                        bp.baseAddress, Int32(K),
                        0.0,
                        op.baseAddress, Int32(N))
                }
            }
        }
    }

    private func applyGeluTanh(into buffer: inout [Float]) {
        let n = buffer.count
        guard n > 0 else { return }
        var sqrt2pi: Float = 0.7978845608
        var coef: Float = 0.044715
        var half: Float = 0.5
        var one: Float = 1.0
        var inner = Swift.Array<Float>(repeating: 0, count: n)
        var tanhOut = Swift.Array<Float>(repeating: 0, count: n)
        buffer.withUnsafeMutableBufferPointer { buf in
            inner.withUnsafeMutableBufferPointer { innerBuf in
                tanhOut.withUnsafeMutableBufferPointer { tanhBuf in
                    guard let xPtr = buf.baseAddress,
                          let inPtr = innerBuf.baseAddress,
                          let tPtr = tanhBuf.baseAddress else { return }
                    vDSP_vsq(xPtr, 1, inPtr, 1, vDSP_Length(n))
                    vDSP_vmul(inPtr, 1, xPtr, 1, inPtr, 1, vDSP_Length(n))
                    vDSP_vsmul(inPtr, 1, &coef, inPtr, 1, vDSP_Length(n))
                    vDSP_vadd(inPtr, 1, xPtr, 1, inPtr, 1, vDSP_Length(n))
                    vDSP_vsmul(inPtr, 1, &sqrt2pi, inPtr, 1, vDSP_Length(n))
                    var nVar = Int32(n)
                    vvtanhf(tPtr, inPtr, &nVar)
                    vDSP_vsadd(tPtr, 1, &one, tPtr, 1, vDSP_Length(n))
                    vDSP_vmul(tPtr, 1, xPtr, 1, tPtr, 1, vDSP_Length(n))
                    vDSP_vsmul(tPtr, 1, &half, xPtr, 1, vDSP_Length(n))
                }
            }
        }
    }
}
