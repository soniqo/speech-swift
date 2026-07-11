import Foundation
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Gemma-2-style RMSNorm: normalize in fp32, scale by (1 + weight), cast back.
/// Stored as a raw parameter (no MLXNN.Module) — matches the npz layout where
/// `(pre|post)_(self_attn|feedforward)_layernorm` are weight-only with no
/// surrounding module path.
@inline(__always)
private func gemmaRMSNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let varX = (x32 * x32).mean(axis: -1, keepDims: true)
    let n = x32 * MLX.rsqrt(varX + MLXArray(eps))
    let scaled = n * (MLXArray(Float(1.0)) + weight.asType(.float32))
    return scaled.asType(dtype)
}

/// Gemma RMSNorm wrapped as a Module so the checkpoint's `name.weight` key
/// loads through the standard `update(parameters:)` path. Behaviour is
/// `gamma * tanh-free RMSNorm(1 + weight)`.
public final class GemmaNorm: Module {
    @ParameterInfo public var weight: MLXArray
    public let eps: Float
    public init(dim: Int, eps: Float = T5GemmaDims.rmsNormEps) {
        self._weight.wrappedValue = MLXArray.zeros([dim])
        self.eps = eps
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        gemmaRMSNorm(x, weight: weight, eps: eps)
    }
}

@inline(__always)
private func ropeCosSin(seqLen: Int, headDim: Int, theta: Float) -> (cos: MLXArray, sin: MLXArray) {
    let half = headDim / 2
    let dimIdx = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
    let invFreq = MLXArray(Float(1.0)) / MLX.pow(MLXArray(theta), dimIdx / Float(headDim))
    let pos = MLXArray(0..<seqLen).asType(.float32)
    let freqs = pos.reshaped([seqLen, 1]) * invFreq.reshaped([1, half])
    let emb = MLX.concatenated([freqs, freqs], axis: -1)         // half-half layout
    return (MLX.cos(emb), MLX.sin(emb))
}

@inline(__always)
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let lower = x[.ellipsis, 0..<half]
    let upper = x[.ellipsis, half..<(2 * half)]
    return MLX.concatenated([-upper, lower], axis: -1)
}

@inline(__always)
private func applyRope(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let cosE = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)  // [1,1,S,D]
    let sinE = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    let qOut = (q * cosE.asType(q.dtype)) + (rotateHalf(q) * sinE.asType(q.dtype))
    let kOut = (k * cosE.asType(k.dtype)) + (rotateHalf(k) * sinE.asType(k.dtype))
    return (qOut, kOut)
}

// MARK: - T5Gemma encoder layer

public final class T5GemmaSelfAttention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public let numHeads: Int
    public let headDim: Int
    public let scaling: Float
    public let softcap: Float

    public override init() {
        let H = T5GemmaDims.numAttentionHeads
        let D = T5GemmaDims.headDim
        let HS = T5GemmaDims.hiddenSize
        self.numHeads = H
        self.headDim = D
        self.scaling = 1.0 / Float(T5GemmaDims.queryPreAttnScalar).squareRoot()
        self.softcap = T5GemmaDims.attnLogitSoftcapping

        self._qProj.wrappedValue = Linear(HS, H * D, bias: false)
        self._kProj.wrappedValue = Linear(HS, T5GemmaDims.numKeyValueHeads * D, bias: false)
        self._vProj.wrappedValue = Linear(HS, T5GemmaDims.numKeyValueHeads * D, bias: false)
        self._oProj.wrappedValue = Linear(H * D, HS, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, addMask: MLXArray?) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let H = numHeads, D = headDim
        var q = qProj(x).reshaped([B, S, H, D]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([B, S, H, D]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([B, S, H, D]).transposed(0, 2, 1, 3)
        (q, k) = applyRope(q, k, cos: cos, sin: sin)

        // Manual softcapped attention (MLXFast.scaledDotProductAttention has
        // no softcap arg). [B, H, S, D] · [B, H, D, S] = [B, H, S, S].
        var qk = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * MLXArray(scaling)
        qk = MLX.tanh(qk / MLXArray(softcap)) * MLXArray(softcap)
        if let m = addMask {
            qk = qk + m
        }
        let p = MLX.softmax(qk.asType(.float32), axis: -1).asType(v.dtype)
        let out = MLX.matmul(p, v).transposed(0, 2, 1, 3).reshaped([B, S, H * D])
        return oProj(out)
    }
}

public final class T5GemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public override init() {
        let HS = T5GemmaDims.hiddenSize
        let IS = T5GemmaDims.intermediateSize
        self._gateProj.wrappedValue = Linear(HS, IS, bias: false)
        self._upProj.wrappedValue = Linear(HS, IS, bias: false)
        self._downProj.wrappedValue = Linear(IS, HS, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

public final class T5GemmaEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: T5GemmaSelfAttention
    @ModuleInfo public var mlp: T5GemmaMLP

    // RMSNorm sub-modules — each loads `name.weight` directly via GemmaNorm.
    @ModuleInfo(key: "pre_self_attn_layernorm")  public var preSelfAttnLN: GemmaNorm
    @ModuleInfo(key: "post_self_attn_layernorm") public var postSelfAttnLN: GemmaNorm
    @ModuleInfo(key: "pre_feedforward_layernorm")  public var preFFLN: GemmaNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFFLN: GemmaNorm

    public override init() {
        let HS = T5GemmaDims.hiddenSize
        self._selfAttn.wrappedValue = T5GemmaSelfAttention()
        self._mlp.wrappedValue = T5GemmaMLP()
        self._preSelfAttnLN.wrappedValue = GemmaNorm(dim: HS)
        self._postSelfAttnLN.wrappedValue = GemmaNorm(dim: HS)
        self._preFFLN.wrappedValue = GemmaNorm(dim: HS)
        self._postFFLN.wrappedValue = GemmaNorm(dim: HS)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, addMask: MLXArray?) -> MLXArray {
        var h = preSelfAttnLN(x)
        h = selfAttn(h, cos: cos, sin: sin, addMask: addMask)
        h = postSelfAttnLN(h)
        var y = x + h
        h = preFFLN(y)
        h = mlp(h)
        h = postFFLN(h)
        y = y + h
        return y
    }
}

public final class T5GemmaEncoderModel: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [T5GemmaEncoderLayer]
    @ModuleInfo public var norm: GemmaNorm

    private let normalizer: Float

    public override init() {
        let HS = T5GemmaDims.hiddenSize
        self._embedTokens.wrappedValue = Embedding(embeddingCount: T5GemmaDims.vocabSize, dimensions: HS)
        self._layers.wrappedValue = (0..<T5GemmaDims.numLayers).map { _ in T5GemmaEncoderLayer() }
        self._norm.wrappedValue = GemmaNorm(dim: HS)
        self.normalizer = Float(HS).squareRoot()
        super.init()
    }

    /// `inputIds`  : [B, S] int32. `attentionMask` : [B, S] int32 (1=real, 0=pad).
    /// Returns `(hiddenStates [B, S, 768], same attention mask)` matching the
    /// upstream Python signature.
    public func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var x = embedTokens(inputIds).asType(.float16)
        x = x * MLXArray(normalizer).asType(.float16)
        let (cos, sin) = ropeCosSin(seqLen: x.dim(1), headDim: T5GemmaDims.headDim,
                                     theta: T5GemmaDims.ropeTheta)

        var addMask: MLXArray? = nil
        if let m = attentionMask {
            let keep = m.asType(.float32)
            let negInf = (MLXArray(Float(1.0)) - keep) * MLXArray(Float(-1e9))
            // [B, S] → [B, 1, 1, S]
            addMask = negInf.expandedDimensions(axis: 1).expandedDimensions(axis: 1).asType(x.dtype)
        }
        for layer in layers {
            x = layer(x, cos: cos, sin: sin, addMask: addMask)
        }
        return norm(x)
    }
}

// MARK: - Public T5Gemma wrapper

/// High-level wrapper: tokenize raw strings → MLX embeddings.
public final class T5GemmaText {
    public let encoder: T5GemmaEncoderModel
    public let tokenizer: UnigramTokenizer

    public init(encoder: T5GemmaEncoderModel, tokenizer: UnigramTokenizer) {
        self.encoder = encoder
        self.tokenizer = tokenizer
    }

    /// Tokenize a single prompt to (input_ids, attention_mask), padded/truncated
    /// to `maxLen`. Padding token id = 0.
    public func tokenize(_ prompt: String, maxLen: Int = 256) -> (ids: MLXArray, mask: MLXArray) {
        let raw = tokenizer.encodeAsIds(prompt)
        let toks = raw.prefix(maxLen).map { Int32($0) }
        var ids = [Int32](repeating: T5GemmaDims.padTokenId, count: maxLen)
        var mask = [Int32](repeating: 0, count: maxLen)
        for (i, t) in toks.enumerated() {
            ids[i] = t
            mask[i] = 1
        }
        return (MLXArray(ids, [1, maxLen]), MLXArray(mask, [1, maxLen]))
    }

    /// Encode one prompt and return (last_hidden_state [1, maxLen, 768] fp16,
    /// attention_mask [1, maxLen] int32). An all-pad row gets a single
    /// visible position before the forward pass to avoid an all-(-inf) row
    /// in the softmax; the returned mask still reads pad=0 everywhere so the
    /// caller's prompt-padding replacement works untouched.
    public func encode(_ prompt: String, maxLen: Int = 256) -> (embeds: MLXArray, mask: MLXArray) {
        let (ids, mask) = tokenize(prompt, maxLen: maxLen)
        let sum = mask.sum().item(Int32.self)
        let attn: MLXArray
        if sum == 0 {
            var fixed = [Int32](repeating: 0, count: maxLen)
            fixed[0] = 1
            attn = MLXArray(fixed, [1, maxLen])
        } else {
            attn = mask
        }
        let out = encoder(ids, attentionMask: attn)
        return (out, mask)
    }
}
