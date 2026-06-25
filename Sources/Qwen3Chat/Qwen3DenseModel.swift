import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

// Hand-written MLX implementation of the standard **Qwen3 dense** transformer (model_type "qwen3"),
// in the same style as Qwen35MLXModel but for the plain dense architecture used by the larger chat
// models (Qwen3-1.7B / 4B / 8B …). Reuses the shared building blocks (PreQuantizedEmbedding,
// QuantizedLinear, RMSNorm, RoPE, SDPA). No DeltaNet, no attention gate — every layer is a standard
// GQA block: q/k/v/o projections, per-head q_norm/k_norm, full RoPE, SwiGLU MLP, pre-norm residuals.

// MARK: - Config

/// Parsed from a standard Qwen3 `config.json`.
public struct Qwen3DenseConfig: Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let eosTokenId: Int
    public let quantGroupSize: Int
    public let quantBits: Int

    public static func load(from url: URL) throws -> Qwen3DenseConfig {
        let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        func int(_ k: String, _ d: Int) -> Int { (obj[k] as? NSNumber)?.intValue ?? d }
        func dbl(_ k: String, _ d: Double) -> Double { (obj[k] as? NSNumber)?.doubleValue ?? d }
        let hidden = int("hidden_size", 2560)
        let heads = int("num_attention_heads", 32)
        // head_dim may be omitted → hidden/heads
        let headDim = obj["head_dim"] != nil ? int("head_dim", 128) : hidden / heads
        // eos_token_id can be an int or a list
        let eos: Int
        if let n = obj["eos_token_id"] as? NSNumber { eos = n.intValue }
        else if let a = obj["eos_token_id"] as? [NSNumber], let f = a.first { eos = f.intValue }
        else { eos = 151645 }  // <|im_end|>
        let quant = obj["quantization"] as? [String: Any]
        return Qwen3DenseConfig(
            hiddenSize: hidden,
            numHiddenLayers: int("num_hidden_layers", 36),
            numAttentionHeads: heads,
            numKeyValueHeads: int("num_key_value_heads", 8),
            headDim: headDim,
            intermediateSize: int("intermediate_size", 9728),
            vocabSize: int("vocab_size", 151936),
            ropeTheta: Float(dbl("rope_theta", 5_000_000)),
            rmsNormEps: Float(dbl("rms_norm_eps", 1e-6)),
            tieWordEmbeddings: (obj["tie_word_embeddings"] as? Bool) ?? true,
            eosTokenId: eos,
            quantGroupSize: (quant?["group_size"] as? NSNumber)?.intValue ?? 64,
            quantBits: (quant?["bits"] as? NSNumber)?.intValue ?? 4
        )
    }
}

// MARK: - Attention (standard GQA)

public final class Qwen3DenseAttention: Module {
    let numQHeads: Int, numKVHeads: Int, headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") var vProj: QuantizedLinear
    @ModuleInfo(key: "o_proj") var oProj: QuantizedLinear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE

    public init(_ c: Qwen3DenseConfig) {
        numQHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = 1.0 / sqrt(Float(c.headDim))
        let gs = c.quantGroupSize, bits = c.quantBits
        let qDim = numQHeads * headDim, kvDim = numKVHeads * headDim
        _qProj = ModuleInfo(wrappedValue: QuantizedLinear(c.hiddenSize, qDim, bias: false, groupSize: gs, bits: bits))
        _kProj = ModuleInfo(wrappedValue: QuantizedLinear(c.hiddenSize, kvDim, bias: false, groupSize: gs, bits: bits))
        _vProj = ModuleInfo(wrappedValue: QuantizedLinear(c.hiddenSize, kvDim, bias: false, groupSize: gs, bits: bits))
        _oProj = ModuleInfo(wrappedValue: QuantizedLinear(qDim, c.hiddenSize, bias: false, groupSize: gs, bits: bits))
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
        super.init()
    }

    /// - Returns: (output [B,T,hidden], updated KV cache (keys, values) each [B, Hkv, S, D]).
    public func callAsFunction(
        _ x: MLXArray, cache: (MLXArray, MLXArray)?, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = x.dim(0), t = x.dim(1)
        // Project, reshape to heads, per-head q/k RMSNorm (Qwen3 normalises before RoPE).
        var q = qNorm(qProj(x).reshaped(b, t, numQHeads, headDim))
        var k = kNorm(kProj(x).reshaped(b, t, numKVHeads, headDim))
        var v = vProj(x).reshaped(b, t, numKVHeads, headDim)
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let ropeOffset = cache?.0.dim(2) ?? offset
        q = rope(q, offset: ropeOffset)
        k = rope(k, offset: ropeOffset)

        var ck = k, cv = v
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
            let additive = causal * Float.greatestFiniteMagnitude  // 0 attended, -inf masked
            maskMode = .array(additive.reshaped(1, 1, t, kvLen).asType(q.dtype))
        }

        let attnOut = SDPA.attendAndMerge(qHeads: q, kHeads: ck, vHeads: cv, scale: scale, mask: maskMode)
        return (oProj(attnOut), (ck, cv))
    }
}

// MARK: - SwiGLU MLP

public final class Qwen3DenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: QuantizedLinear
    @ModuleInfo(key: "up_proj") var upProj: QuantizedLinear
    @ModuleInfo(key: "down_proj") var downProj: QuantizedLinear

    public init(_ c: Qwen3DenseConfig) {
        let h = c.hiddenSize, i = c.intermediateSize, gs = c.quantGroupSize, bits = c.quantBits
        _gateProj = ModuleInfo(wrappedValue: QuantizedLinear(h, i, bias: false, groupSize: gs, bits: bits))
        _upProj = ModuleInfo(wrappedValue: QuantizedLinear(h, i, bias: false, groupSize: gs, bits: bits))
        _downProj = ModuleInfo(wrappedValue: QuantizedLinear(i, h, bias: false, groupSize: gs, bits: bits))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Transformer layer

public final class Qwen3DenseLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attn: Qwen3DenseAttention
    @ModuleInfo var mlp: Qwen3DenseMLP

    public init(_ c: Qwen3DenseConfig) {
        _inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _attn = ModuleInfo(wrappedValue: Qwen3DenseAttention(c))
        _mlp = ModuleInfo(wrappedValue: Qwen3DenseMLP(c))
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, cache: (MLXArray, MLXArray)?, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (a, newCache) = attn(inputLayerNorm(x), cache: cache, offset: offset)
        let h = x + a
        return (h + mlp(postAttentionLayerNorm(h)), newCache)
    }
}

// MARK: - Model

public final class Qwen3DenseModel: Module {
    public let config: Qwen3DenseConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: PreQuantizedEmbedding
    @ModuleInfo var layers: [Qwen3DenseLayer]
    @ModuleInfo var norm: RMSNorm
    // Present only when embeddings are untied.
    @ModuleInfo(key: "lm_head") var lmHead: QuantizedLinear?

    public init(config c: Qwen3DenseConfig) {
        self.config = c
        _embedTokens = ModuleInfo(wrappedValue: PreQuantizedEmbedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize,
            groupSize: c.quantGroupSize, bits: c.quantBits))
        _layers = ModuleInfo(wrappedValue: (0..<c.numHiddenLayers).map { _ in Qwen3DenseLayer(c) })
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _lmHead = ModuleInfo(wrappedValue: c.tieWordEmbeddings ? nil
            : QuantizedLinear(c.hiddenSize, c.vocabSize, bias: false, groupSize: c.quantGroupSize, bits: c.quantBits))
        super.init()
    }

    /// KV cache for every layer + the current sequence position (RoPE offset).
    public struct InferenceState {
        public var kvCaches: [(MLXArray, MLXArray)?]
        public var position: Int
        public static func initial(config c: Qwen3DenseConfig) -> InferenceState {
            InferenceState(kvCaches: Array(repeating: nil, count: c.numHiddenLayers), position: 0)
        }
    }

    /// - Returns: (logits [B,T,vocab], updated state).
    public func forward(inputIds: MLXArray, state: InferenceState) -> (MLXArray, InferenceState) {
        let t = inputIds.dim(1)
        var h = embedTokens(inputIds)
        var caches = state.kvCaches
        for (i, layer) in layers.enumerated() {
            let (nh, nc) = layer(h, cache: state.kvCaches[i], offset: state.position)
            h = nh
            caches[i] = nc
        }
        h = norm(h)
        let logits = lmHead?(h) ?? embedTokens.asLinear(h)  // tied → reuse embedding
        return (logits, InferenceState(kvCaches: caches, position: state.position + t))
    }
}
