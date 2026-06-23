import Foundation
import MLX
import MLXFast
import MLXNN

/// Precompute the llama3-scaled RoPE frequencies `[headDim/2]`, matching the standard
/// `Llama3RoPE`: high frequencies unchanged, low frequencies divided by `factor`,
/// medium frequencies smoothly interpolated.
private func llama3Freqs(_ cfg: T3Config) -> MLXArray {
    let dims = cfg.headDim
    let base = cfg.ropeTheta
    let factor = cfg.ropeFactor
    let lowF = cfg.ropeLowFreqFactor
    let highF = cfg.ropeHighFreqFactor
    let oldLen = Float(cfg.ropeOrigMaxPos)
    let lowWavelen = oldLen / lowF
    let highWavelen = oldLen / highF

    let exps = MLXArray(stride(from: 0, to: dims, by: 2).map { Float($0) }) / Float(dims)
    let freqs = MLX.pow(MLXArray(base), exps)        // base ** (arange(0,dims,2)/dims)
    let wavelens = (2.0 * Float.pi) * freqs

    let scaled = MLX.which(MLX.greater(wavelens, lowWavelen), freqs * factor, freqs)
    let isMedium = MLX.logicalAnd(MLX.greater(wavelens, highWavelen), MLX.less(wavelens, lowWavelen))
    let smoothFactors = (MLXArray(oldLen) / wavelens - lowF) / (highF - lowF)
    let smoothFreqs = scaled / ((MLXArray(1.0) - smoothFactors) / factor + smoothFactors)
    return MLX.which(isMedium, smoothFreqs, scaled)
}

/// Minimal append-only KV cache for the T3 AR loop.
final class T3KVCache {
    var keys: MLXArray?
    var values: MLXArray?
    var offset = 0

    func update(_ k: MLXArray, _ v: MLXArray) -> (MLXArray, MLXArray) {
        if let keys, let values {
            self.keys = concatenated([keys, k], axis: 2)
            self.values = concatenated([values, v], axis: 2)
        } else {
            keys = k
            values = v
        }
        offset += k.dim(2)
        return (keys!, values!)
    }
}

/// One Llama attention block (full MHA, llama3 RoPE, KV cache).
final class T3Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let freqs: MLXArray

    init(_ cfg: T3Config, freqs: MLXArray) {
        nHeads = cfg.numHeads
        nKVHeads = cfg.numKVHeads
        headDim = cfg.headDim
        scale = pow(Float(cfg.headDim), -0.5)
        self.freqs = freqs
        let dim = cfg.hiddenSize
        _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: T3KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped([b, l, nHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped([b, l, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([b, l, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset, freqs: freqs)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: nil, scale: 1.0, offset: offset, freqs: freqs)
        if let cache { (k, v) = cache.update(k, v) }

        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([b, l, nHeads * headDim])
        return oProj(out)
    }
}

/// SwiGLU MLP.
final class T3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ cfg: T3Config) {
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class T3Block: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: T3Attention
    @ModuleInfo(key: "mlp") var mlp: T3MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ cfg: T3Config, freqs: MLXArray) {
        _selfAttn.wrappedValue = T3Attention(cfg, freqs: freqs)
        _mlp.wrappedValue = T3MLP(cfg)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: T3KVCache?) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Llama backbone (standard Llama transformer), driven by input embeddings.
final class T3Backbone: Module {
    // Vestigial in the checkpoint (a [8,1024] stub); the model is always driven by
    // input embeddings, so this is never used — declared only to match weight keys.
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [T3Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ cfg: T3Config) {
        let freqs = llama3Freqs(cfg)
        _embedTokens.wrappedValue = Embedding(embeddingCount: 8, dimensions: cfg.hiddenSize)
        _layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in T3Block(cfg, freqs: freqs) }
        _norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    /// `embeddings`: `[B, L, dim]`. `caches`: per-layer (nil for a fresh prefill mask).
    func callAsFunction(_ embeddings: MLXArray, caches: [T3KVCache]?) -> MLXArray {
        var h = embeddings
        let l = h.dim(1)
        // Causal mask only needed when processing >1 token (prefill); single-step
        // decode attends to all cached keys with no mask.
        let mask: MLXArray? = l > 1 ? MultiHeadAttention.createAdditiveCausalMask(l).asType(h.dtype) : nil
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches?[i])
        }
        return norm(h)
    }
}
