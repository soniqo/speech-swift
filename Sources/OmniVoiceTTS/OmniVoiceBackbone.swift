import Foundation
import MLX
import MLXNN
import MLXFast

// The OmniVoice backbone is a standard Qwen3 transformer driven by input
// embeddings and run **bidirectionally** (no causal mask) — it's a NAR
// diffusion model, so every position attends to every other. Checkpoint keys
// live under `llm.*`.

/// Holder for the precomputed RoPE so it isn't registered as a loadable param.
final class OVFreqs {
    let rope: MLXNN.RoPE
    init(_ cfg: OmniVoiceConfig) {
        rope = MLXNN.RoPE(dimensions: cfg.headDim, traditional: false, base: cfg.ropeTheta)
    }
}

/// Qwen3 attention: GQA (16 query / 8 KV heads), per-head RMSNorm on Q and K
/// before RoPE, full rotary, no attention bias. Bidirectional (no mask).
final class OVAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let freqs: OVFreqs

    init(_ cfg: OmniVoiceConfig, freqs: OVFreqs) {
        nHeads = cfg.numAttentionHeads
        nKVHeads = cfg.numKeyValueHeads
        headDim = cfg.headDim
        scale = pow(Float(cfg.headDim), -0.5)
        _qProj.wrappedValue = Linear(cfg.hiddenSize, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(cfg.hiddenSize, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, cfg.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps)
        self.freqs = freqs
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        // [B, L, H, D] → per-head RMSNorm → [B, H, L, D]
        var q = qNorm(qProj(x).reshaped([b, l, nHeads, headDim])).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped([b, l, nKVHeads, headDim])).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped([b, l, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        q = freqs.rope(q)
        k = freqs.rope(k)
        // Bidirectional: no mask. SDPA handles GQA (q heads > kv heads) directly.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped([b, l, nHeads * headDim]))
    }
}

/// SwiGLU MLP.
final class OVMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ cfg: OmniVoiceConfig) {
        _gateProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class OVBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: OVAttention
    @ModuleInfo(key: "mlp") var mlp: OVMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ cfg: OmniVoiceConfig, freqs: OVFreqs) {
        _selfAttn.wrappedValue = OVAttention(cfg, freqs: freqs)
        _mlp.wrappedValue = OVMLP(cfg)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x))
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// The Qwen3 backbone (`llm.*`): token embeddings + N bidirectional blocks +
/// final norm. Driven by precomputed input embeddings (text + audio fused).
final class OVBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OVBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ cfg: OmniVoiceConfig) {
        let freqs = OVFreqs(cfg)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: cfg.textVocabSize, dimensions: cfg.hiddenSize)
        _layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in OVBlock(cfg, freqs: freqs) }
        _norm.wrappedValue = RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps)
    }

    /// `embeddings`: `[B, L, hidden]` → `[B, L, hidden]` (post final norm).
    func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        var h = embeddings
        for layer in layers { h = layer(h) }
        return norm(h)
    }
}
