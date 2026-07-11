import Foundation
import MLX
import MLXNN

// Cross-domain transformer for HTDemucs (dense attention), translated from the
// MIT MLX reference + demucs transformer.py. htdemucs config: dim 512, 8 heads,
// FFN 2048, 5 layers, norm_first=true, norm_out=true (GroupNorm), layer_scale=true,
// sin positional embeddings, gelu. Layer pattern (cross_first=false): indices
// 0,2,4 = self-attention; 1,3 = cross-attention.
//
// Keys match the exported torch names: `self_attn`/`cross_attn` with packed
// `in_proj_weight`/`in_proj_bias` + `out_proj`, `linear1/2`, `norm1/2[/3]`,
// `norm_out`, `gamma_1/2`. NOT yet parity-validated (Phase D).

// MARK: - positional embeddings (parameter-free, computed in Swift)

/// `create_sin_embedding`: returns [T, 1, C]; first half cos, second half sin.
private func sinEmbedding1D(_ T: Int, _ C: Int, maxPeriod: Float) -> MLXArray {
    let half = C / 2
    var data = [Float](repeating: 0, count: T * C)
    for t in 0..<T {
        for j in 0..<half {
            let phase = Float(t) / pow(maxPeriod, Float(j) / Float(half - 1))
            data[t * C + j] = cos(phase)
            data[t * C + half + j] = sin(phase)
        }
    }
    return MLXArray(data, [T, 1, C])
}

/// `create_2d_sin_embedding`: returns [1, C, Fr, T1]. Channels 0..<C/2 encode
/// the width (time) position, C/2..<C encode the height (freq) position;
/// within each, even=sin / odd=cos over div_term.
private func sinEmbedding2D(_ C: Int, _ Fr: Int, _ T1: Int, maxPeriod: Float) -> MLXArray {
    let half = C / 2
    let n = half / 2
    var div = [Float](repeating: 0, count: n)
    for i in 0..<n { div[i] = exp(Float(2 * i) * -(log(maxPeriod) / Float(half))) }
    var data = [Float](repeating: 0, count: C * Fr * T1)
    for c in 0..<C {
        let isWidth = c < half
        let c2 = isWidth ? c : c - half
        let i = c2 / 2
        let useSin = (c2 % 2 == 0)
        for h in 0..<Fr {
            for w in 0..<T1 {
                let pos = Float(isWidth ? w : h)
                let ph = pos * div[i]
                data[(c * Fr + h) * T1 + w] = useSin ? sin(ph) : cos(ph)
            }
        }
    }
    return MLXArray(data, [1, C, Fr, T1])
}

// MARK: - GroupNorm over channels-last [B, seq, C] (demucs MyGroupNorm)

/// demucs `MyGroupNorm` (subclasses GroupNorm(1, C)): on [B, seq, C] it
/// normalizes group=1 stats over (seq, C) per sample, then per-channel affine.
/// Holds weight/bias directly so keys are `norm_out.weight`/`norm_out.bias`.
final class MyGroupNorm: Module {
    let eps: Float = 1e-5
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(_ channels: Int) {
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1), C = x.dim(2)
        let flat = x.reshaped([B, S * C])
        let mean = flat.mean(axis: 1, keepDims: true)
        let v = flat.variance(axis: 1, keepDims: true)
        let nrm = ((flat - mean) * MLX.rsqrt(v + eps)).reshaped([B, S, C])
        return nrm * weight + bias
    }
}

// MARK: - attention (packed torch in_proj layout)

/// Multi-head attention with torch-packed projections: `in_proj_weight`
/// `[3d, d]` (q|k|v stacked), `in_proj_bias` `[3d]`, `out_proj` Linear.
final class DemucsAttention: Module {
    let nhead: Int
    let dim: Int
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, nhead: Int) {
        self.dim = dim
        self.nhead = nhead
        self._inProjWeight.wrappedValue = MLXArray.zeros([3 * dim, dim])
        self._inProjBias.wrappedValue = MLXArray.zeros([3 * dim])
        self._outProj.wrappedValue = Linear(dim, dim)
    }

    private func project(_ x: MLXArray, _ slot: Int) -> MLXArray {
        let w = inProjWeight[(slot * dim) ..< ((slot + 1) * dim), 0...]   // [d, d]
        let b = inProjBias[(slot * dim) ..< ((slot + 1) * dim)]
        return matmul(x, w.transposed(1, 0)) + b
    }

    private func splitHeads(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        return x.reshaped([B, L, nhead, dim / nhead]).transposed(0, 2, 1, 3)  // [B, H, L, dh]
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let B = q.dim(0), Lq = q.dim(1)
        let qh = splitHeads(project(q, 0))
        let kh = splitHeads(project(k, 1))
        let vh = splitHeads(project(v, 2))
        let scale = Float(1.0 / sqrt(Double(dim / nhead)))
        var scores = matmul(qh, kh.transposed(0, 1, 3, 2)) * scale   // [B, H, Lq, Lk]
        scores = softmax(scores, axis: -1)
        let out = matmul(scores, vh)                                 // [B, H, Lq, dh]
        let merged = out.transposed(0, 2, 1, 3).reshaped([B, Lq, dim])
        return outProj(merged)
    }
}

// MARK: - encoder layers (norm_first=true path)

/// Self-attention encoder layer (idx 0,2,4).
final class SelfAttnLayer: Module {
    @ModuleInfo(key: "self_attn") var attn: DemucsAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: MyGroupNorm
    @ModuleInfo(key: "gamma_1") var gamma1: LayerScale
    @ModuleInfo(key: "gamma_2") var gamma2: LayerScale

    init(dim: Int, nhead: Int, ffn: Int) {
        self._attn.wrappedValue = DemucsAttention(dim: dim, nhead: nhead)
        self._linear1.wrappedValue = Linear(dim, ffn)
        self._linear2.wrappedValue = Linear(ffn, dim)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim)
        self._normOut.wrappedValue = MyGroupNorm(dim)
        self._gamma1.wrappedValue = LayerScale(dim, channelLast: true)
        self._gamma2.wrappedValue = LayerScale(dim, channelLast: true)
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        let n = norm1(x)
        x = x + gamma1(attn(n, n, n))
        x = x + gamma2(linear2(gelu(linear1(norm2(x)))))
        return normOut(x)
    }
}

/// Cross-attention encoder layer (idx 1,3): query stream attends to the other.
final class CrossAttnLayer: Module {
    @ModuleInfo(key: "cross_attn") var attn: DemucsAttention
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm
    @ModuleInfo(key: "norm_out") var normOut: MyGroupNorm
    @ModuleInfo(key: "gamma_1") var gamma1: LayerScale
    @ModuleInfo(key: "gamma_2") var gamma2: LayerScale

    init(dim: Int, nhead: Int, ffn: Int) {
        self._attn.wrappedValue = DemucsAttention(dim: dim, nhead: nhead)
        self._linear1.wrappedValue = Linear(dim, ffn)
        self._linear2.wrappedValue = Linear(ffn, dim)
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim)
        self._normOut.wrappedValue = MyGroupNorm(dim)
        self._gamma1.wrappedValue = LayerScale(dim, channelLast: true)
        self._gamma2.wrappedValue = LayerScale(dim, channelLast: true)
    }

    /// `q` stream attends to `k` stream (norm_first).
    func callAsFunction(_ q: MLXArray, _ k: MLXArray) -> MLXArray {
        let kn = norm2(k)
        var x = q + gamma1(attn(norm1(q), kn, kn))
        x = x + gamma2(linear2(gelu(linear1(norm3(x)))))
        return normOut(x)
    }
}

// MARK: - CrossTransformerEncoder

final class CrossTransformerEncoder: Module {
    let numLayers: Int
    let maxPeriod: Float
    let weightPosEmbed: Float
    @ModuleInfo(key: "norm_in") var normIn: LayerNorm
    @ModuleInfo(key: "norm_in_t") var normInT: LayerNorm
    // Heterogeneous per index (self vs cross) — stored as base Module, dispatched.
    @ModuleInfo(key: "layers") var layers: [Module]
    @ModuleInfo(key: "layers_t") var layersT: [Module]

    init(dim: Int, nhead: Int, ffn: Int, numLayers: Int, maxPeriod: Float, weightPosEmbed: Float) {
        self.numLayers = numLayers
        self.maxPeriod = maxPeriod
        self.weightPosEmbed = weightPosEmbed
        self._normIn.wrappedValue = LayerNorm(dimensions: dim)
        self._normInT.wrappedValue = LayerNorm(dimensions: dim)
        var ls: [Module] = []
        var lst: [Module] = []
        for idx in 0..<numLayers {
            if idx % 2 == 0 {
                ls.append(SelfAttnLayer(dim: dim, nhead: nhead, ffn: ffn))
                lst.append(SelfAttnLayer(dim: dim, nhead: nhead, ffn: ffn))
            } else {
                ls.append(CrossAttnLayer(dim: dim, nhead: nhead, ffn: ffn))
                lst.append(CrossAttnLayer(dim: dim, nhead: nhead, ffn: ffn))
            }
        }
        self._layers.wrappedValue = ls
        self._layersT.wrappedValue = lst
    }

    /// x: [B, C, Fr, T1] (spectral), xt: [B, C, T2] (temporal). Returns the pair.
    func callAsFunction(_ xIn: MLXArray, _ xtIn: MLXArray) -> (MLXArray, MLXArray) {
        let B = xIn.dim(0), C = xIn.dim(1), Fr = xIn.dim(2), T1 = xIn.dim(3)

        // spectral: add 2D pos-emb, flatten to [B, T1*Fr, C]
        var pe2d = sinEmbedding2D(C, Fr, T1, maxPeriod: maxPeriod)        // [1, C, Fr, T1]
        pe2d = MLX.broadcast(pe2d, to: [B, C, Fr, T1]).transposed(0, 3, 2, 1).reshaped([B, T1 * Fr, C])
        var x = xIn.transposed(0, 3, 2, 1).reshaped([B, T1 * Fr, C])
        x = normIn(x) + weightPosEmbed * pe2d

        // temporal: add 1D pos-emb
        let T2 = xtIn.dim(2)
        var xt = xtIn.transposed(0, 2, 1)                                 // [B, T2, C]
        let pe1d = sinEmbedding1D(T2, C, maxPeriod: maxPeriod).transposed(1, 0, 2)  // [1, T2, C]
        xt = normInT(xt) + weightPosEmbed * pe1d

        for idx in 0..<numLayers {
            if idx % 2 == 0 {
                x = (layers[idx] as! SelfAttnLayer)(x)
                xt = (layersT[idx] as! SelfAttnLayer)(xt)
            } else {
                let oldX = x
                x = (layers[idx] as! CrossAttnLayer)(x, xt)
                xt = (layersT[idx] as! CrossAttnLayer)(xt, oldX)
            }
        }

        let xOut = x.reshaped([B, T1, Fr, C]).transposed(0, 3, 2, 1)       // [B, C, Fr, T1]
        let xtOut = xt.transposed(0, 2, 1)                                 // [B, C, T2]
        return (xOut, xtOut)
    }
}
