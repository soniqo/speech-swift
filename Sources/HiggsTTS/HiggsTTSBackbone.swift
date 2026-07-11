import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

// Standard Qwen3 dense decoder (GQA, per-head q/k RMSNorm before RoPE, SwiGLU,
// pre-norm residuals) in unquantized form for the bf16 Higgs checkpoint,
// adapted from Qwen3Chat's Qwen3DenseModel. Differences: float Linear /
// Embedding layers, a forward that accepts precomputed embeddings (the Higgs
// prompt interleaves text and audio embeddings), and no LM head — Higgs reads
// hidden states through the fused codebook head instead.

final class HiggsTTSAttention: Module {
    let numQHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE

    init(_ c: HiggsTTSTextConfig) {
        numQHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = 1.0 / sqrt(Float(c.headDim))
        let qDim = numQHeads * headDim
        let kvDim = numKVHeads * headDim
        _qProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, qDim, bias: false))
        _kProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, kvDim, bias: false))
        _vProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, kvDim, bias: false))
        _oProj = ModuleInfo(wrappedValue: Linear(qDim, c.hiddenSize, bias: false))
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: (MLXArray, MLXArray)?, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = x.dim(0)
        let t = x.dim(1)
        var q = qNorm(qProj(x).reshaped(b, t, numQHeads, headDim))
        var k = kNorm(kProj(x).reshaped(b, t, numKVHeads, headDim))
        var v = vProj(x).reshaped(b, t, numKVHeads, headDim)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let ropeOffset = cache?.0.dim(2) ?? offset
        q = rope(q, offset: ropeOffset)
        k = rope(k, offset: ropeOffset)

        var ck = k
        var cv = v
        if let (pk, pv) = cache {
            ck = concatenated([pk, k], axis: 2)
            cv = concatenated([pv, v], axis: 2)
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if t <= 1 && (cache != nil || offset > 0) {
            maskMode = .none
        } else {
            let kvLen = ck.dim(2)
            let pastLen = kvLen - t
            let causal = MLXArray.tri(t, m: kvLen, k: pastLen, type: Float.self) - 1
            let additive = causal * Float.greatestFiniteMagnitude
            maskMode = .array(additive.reshaped(1, 1, t, kvLen).asType(q.dtype))
        }

        let attnOut = SDPA.attendAndMerge(qHeads: q, kHeads: ck, vHeads: cv, scale: scale, mask: maskMode)
        return (oProj(attnOut), (ck, cv))
    }
}

final class HiggsTTSMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: HiggsTTSTextConfig) {
        _gateProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false))
        _upProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false))
        _downProj = ModuleInfo(wrappedValue: Linear(c.intermediateSize, c.hiddenSize, bias: false))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class HiggsTTSLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attn: HiggsTTSAttention
    @ModuleInfo var mlp: HiggsTTSMLP

    init(_ c: HiggsTTSTextConfig) {
        _inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _attn = ModuleInfo(wrappedValue: HiggsTTSAttention(c))
        _mlp = ModuleInfo(wrappedValue: HiggsTTSMLP(c))
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, cache: (MLXArray, MLXArray)?, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (a, newCache) = attn(inputLayerNorm(x), cache: cache, offset: offset)
        let h = x + a
        return (h + mlp(postAttentionLayerNorm(h)), newCache)
    }
}

/// The Higgs Qwen3 backbone: embeddings in, final-norm hidden states out.
final class HiggsTTSBackbone: Module {
    let config: HiggsTTSTextConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [HiggsTTSLayer]
    @ModuleInfo var norm: RMSNorm

    init(config c: HiggsTTSTextConfig) {
        self.config = c
        _embedTokens = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize))
        _layers = ModuleInfo(wrappedValue: (0..<c.numHiddenLayers).map { _ in HiggsTTSLayer(c) })
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        super.init()
    }

    struct InferenceState {
        var kvCaches: [(MLXArray, MLXArray)?]
        var position: Int

        static func initial(config c: HiggsTTSTextConfig) -> InferenceState {
            InferenceState(kvCaches: Array(repeating: nil, count: c.numHiddenLayers), position: 0)
        }
    }

    func embedText(_ ids: MLXArray) -> MLXArray {
        embedTokens(ids)
    }

    /// - Returns: (final-norm hidden `[B, T, hidden]`, updated state).
    func forward(embeddings: MLXArray, state: InferenceState) -> (MLXArray, InferenceState) {
        let t = embeddings.dim(1)
        var h = embeddings
        var caches = state.kvCaches
        for (i, layer) in layers.enumerated() {
            let (nh, nc) = layer(h, cache: state.kvCaches[i], offset: state.position)
            h = nh
            caches[i] = nc
        }
        return (norm(h), InferenceState(kvCaches: caches, position: state.position + t))
    }
}
