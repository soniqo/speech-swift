import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

// Hand-written MLX implementation of the **Gemma 4 text** transformer (model_type "gemma4_text"),
// a 1:1 Swift port of `mlx_lm/models/gemma4_text.py` (the E2B / E4B dense, non-MoE configuration).
// Mirrors the structure of `Qwen3DenseModel` but implements every Gemma-4-specific detail:
//
//   • embed_scale = sqrt(hidden_size); Per-Layer Embeddings (PLE) projected per layer.
//   • Sandwich norms (input / post_attention / pre_feedforward / post_feedforward).
//   • Attention scale = 1.0 (NOT 1/sqrt(d)); q_norm/k_norm (RMSNorm) + v_norm (RMSNorm no-scale).
//   • head_dim differs by layer type: full_attention → 512, sliding_attention → 256.
//   • Dual RoPE: sliding = standard nn.RoPE(256, base 1e4); full = ProportionalRoPE(512, rotated 128, base 1e6).
//   • KV-sharing: the last `num_kv_shared_layers` layers reuse post-RoPE K/V from the last earlier
//     producing layer of the same layer_type (no k/v projections of their own).
//   • Double-wide MLP on KV-shared layers (use_double_wide_mlp).
//   • Per-layer input gating after the FFN block; per-layer layer_scalar.
//   • Final: norm → tied lm_head → logit_softcap(30).
//
// This is a parity port — a single forward pass over the prompt (no incremental KV cache object);
// shared layers reuse the (keys, values) computed by their producing layer in the same pass.

// MARK: - Config

/// Parsed from a Gemma 4 multimodal `config.json` (`text_config` block).
public struct Gemma4DenseConfig: Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let headDim: Int            // sliding head_dim
    public let globalHeadDim: Int      // full_attention head_dim
    public let numKeyValueHeads: Int
    public let numKVSharedLayers: Int
    public let hiddenSizePerLayerInput: Int
    public let vocabSize: Int
    public let vocabSizePerLayerInput: Int
    public let rmsNormEps: Float
    public let slidingWindow: Int
    public let maxPositionEmbeddings: Int
    public let fullRopeTheta: Float
    public let fullPartialRotaryFactor: Float
    public let slidingRopeTheta: Float
    public let finalLogitSoftcapping: Float
    public let useDoubleWideMLP: Bool
    public let tieWordEmbeddings: Bool
    public let layerTypes: [String]    // "full_attention" | "sliding_attention"
    public let eosTokenId: Int
    public let quantGroupSize: Int
    public let quantBits: Int

    public static func load(from url: URL) throws -> Gemma4DenseConfig {
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
        // Text fields live under "text_config" for the multimodal checkpoint; fall back to root.
        let tc = (root["text_config"] as? [String: Any]) ?? root
        func int(_ d: [String: Any], _ k: String, _ def: Int) -> Int { (d[k] as? NSNumber)?.intValue ?? def }
        func dbl(_ d: [String: Any], _ k: String, _ def: Double) -> Double { (d[k] as? NSNumber)?.doubleValue ?? def }

        let numLayers = int(tc, "num_hidden_layers", 35)
        let slidingWindowPattern = int(tc, "sliding_window_pattern", 5)

        // layer_types: explicit list, else derived from sliding_window_pattern (last of each block is full).
        let layerTypes: [String]
        if let lt = tc["layer_types"] as? [String] {
            layerTypes = lt
        } else {
            var pattern = Array(repeating: "sliding_attention", count: slidingWindowPattern - 1)
            pattern.append("full_attention")
            var derived: [String] = []
            while derived.count < numLayers { derived.append(contentsOf: pattern) }
            layerTypes = Array(derived.prefix(numLayers))
        }

        // rope_parameters: { full_attention: {partial_rotary_factor, rope_theta, rope_type},
        //                    sliding_attention: {rope_theta, rope_type} }
        let ropeParams = tc["rope_parameters"] as? [String: Any]
        let fullRope = ropeParams?["full_attention"] as? [String: Any]
        let slidingRope = ropeParams?["sliding_attention"] as? [String: Any]
        let fullTheta = (fullRope?["rope_theta"] as? NSNumber)?.doubleValue ?? 1_000_000.0
        let fullPRF = (fullRope?["partial_rotary_factor"] as? NSNumber)?.doubleValue ?? 0.25
        let slidingTheta = (slidingRope?["rope_theta"] as? NSNumber)?.doubleValue ?? 10_000.0

        // eos can be int or list (use first).
        let eos: Int
        if let n = tc["eos_token_id"] as? NSNumber { eos = n.intValue }
        else if let a = tc["eos_token_id"] as? [NSNumber], let f = a.first { eos = f.intValue }
        else if let n = root["eos_token_id"] as? NSNumber { eos = n.intValue }
        else if let a = root["eos_token_id"] as? [NSNumber], let f = a.first { eos = f.intValue }
        else { eos = 1 }

        // quantization lives at the root of the MLX config.
        let quant = (root["quantization"] as? [String: Any]) ?? (tc["quantization"] as? [String: Any])

        return Gemma4DenseConfig(
            hiddenSize: int(tc, "hidden_size", 1536),
            numHiddenLayers: numLayers,
            intermediateSize: int(tc, "intermediate_size", 6144),
            numAttentionHeads: int(tc, "num_attention_heads", 8),
            headDim: int(tc, "head_dim", 256),
            globalHeadDim: int(tc, "global_head_dim", 512),
            numKeyValueHeads: int(tc, "num_key_value_heads", 1),
            numKVSharedLayers: int(tc, "num_kv_shared_layers", 20),
            hiddenSizePerLayerInput: int(tc, "hidden_size_per_layer_input", 256),
            vocabSize: int(tc, "vocab_size", 262144),
            vocabSizePerLayerInput: int(tc, "vocab_size_per_layer_input", 262144),
            rmsNormEps: Float(dbl(tc, "rms_norm_eps", 1e-6)),
            slidingWindow: int(tc, "sliding_window", 512),
            maxPositionEmbeddings: int(tc, "max_position_embeddings", 131072),
            fullRopeTheta: Float(fullTheta),
            fullPartialRotaryFactor: Float(fullPRF),
            slidingRopeTheta: Float(slidingTheta),
            finalLogitSoftcapping: Float(dbl(tc, "final_logit_softcapping", 30.0)),
            useDoubleWideMLP: (tc["use_double_wide_mlp"] as? Bool) ?? true,
            tieWordEmbeddings: (tc["tie_word_embeddings"] as? Bool) ?? true,
            layerTypes: layerTypes,
            eosTokenId: eos,
            quantGroupSize: (quant?["group_size"] as? NSNumber)?.intValue ?? 64,
            quantBits: (quant?["bits"] as? NSNumber)?.intValue ?? 4
        )
    }

    /// Index of the first KV-shared layer.
    public var firstKVSharedLayer: Int { numHiddenLayers - numKVSharedLayers }

    public func isKVSharedLayer(_ i: Int) -> Bool {
        let first = firstKVSharedLayer
        return i >= first && first > 0
    }

    public func headDim(forLayer i: Int) -> Int {
        layerTypes[i] == "full_attention" ? globalHeadDim : headDim
    }

    /// Maps each layer index → the producing layer whose post-RoPE K/V it reuses.
    /// Producing layers map to themselves. Mirrors `Gemma4TextModel.previous_kvs`.
    public var previousKVs: [Int] {
        var prev = Array(0..<numHiddenLayers)
        guard numKVSharedLayers > 0 else { return prev }
        let m = firstKVSharedLayer
        var lastOfType: [String: Int] = [:]
        for i in 0..<m { lastOfType[layerTypes[i]] = i }
        for j in m..<numHiddenLayers { prev[j] = lastOfType[layerTypes[j]]! }
        return prev
    }
}

// MARK: - RMSNorm without scale (matches RMSNormNoScale → rms_norm(x, None, eps))

/// `mx.fast.rms_norm(x, None, eps)` — normalises by RMS with no learnable weight.
/// Implemented via the fast kernel with a frozen ones-weight (numerically identical).
public final class Gemma4RMSNormNoScale: Module {
    let eps: Float
    let ones: MLXArray
    public init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.ones = MLXArray.ones([dimensions], dtype: .float32)
        super.init()
        self.freeze()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: ones.asType(x.dtype), eps: eps)
    }
}

// MARK: - Proportional RoPE (full_attention layers)

/// Port of `rope_utils.ProportionalRoPE`. Rotates only the first `rotatedDims` of `dims`,
/// with frequencies computed over the FULL `dims` (the "proportional" detail), leaving the
/// rest unrotated (freqs set to +inf → zero rotation).
public final class Gemma4ProportionalRoPE {
    let dims: Int
    let freqs: MLXArray

    public init(dims: Int, rotatedDims: Int, base: Float, factor: Float = 1.0) {
        self.dims = dims
        precondition(rotatedDims <= dims, "rotatedDims must be <= dims")
        // exponents = arange(0, rotatedDims, 2) / dims   (NOTE: divide by `dims`, not rotatedDims)
        let nReal = (rotatedDims + 1) / 2   // count of arange(0, rotatedDims, 2)
        var freqVals = [Float](repeating: 0, count: nReal)
        for (idx, e) in stride(from: 0, to: rotatedDims, by: 2).enumerated() {
            freqVals[idx] = factor * pow(base, Float(e) / Float(dims))
        }
        let nInf = (dims - rotatedDims) / 2
        let infVals = [Float](repeating: Float.infinity, count: nInf)
        self.freqs = MLXArray(freqVals + infVals).asType(.float32)
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dims, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: freqs)
    }
}

// MARK: - Attention

public final class Gemma4Attention: Module {
    let layerIdx: Int
    let isSliding: Bool
    let hasKV: Bool
    let headDim: Int
    let nHeads: Int
    let nKVHeads: Int
    let scale: Float = 1.0

    @ModuleInfo(key: "q_proj") var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") var kProj: QuantizedLinear?
    @ModuleInfo(key: "v_proj") var vProj: QuantizedLinear?
    @ModuleInfo(key: "o_proj") var oProj: QuantizedLinear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    var vNorm: Gemma4RMSNormNoScale?

    // RoPE: one of the two paths is used per layer type.
    let slidingRope: RoPE?
    let fullRope: Gemma4ProportionalRoPE?

    public init(_ c: Gemma4DenseConfig, layerIdx: Int) {
        self.layerIdx = layerIdx
        let layerType = c.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.hasKV = layerIdx < c.numHiddenLayers - c.numKVSharedLayers
        self.headDim = c.headDim(forLayer: layerIdx)
        self.nHeads = c.numAttentionHeads
        self.nKVHeads = c.numKeyValueHeads

        let gs = c.quantGroupSize, bits = c.quantBits
        let dim = c.hiddenSize
        let qDim = nHeads * headDim
        let kvDim = nKVHeads * headDim

        _qProj = ModuleInfo(wrappedValue: QuantizedLinear(dim, qDim, bias: false, groupSize: gs, bits: bits))
        _oProj = ModuleInfo(wrappedValue: QuantizedLinear(qDim, dim, bias: false, groupSize: gs, bits: bits))
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))

        if hasKV {
            _kProj = ModuleInfo(wrappedValue: QuantizedLinear(dim, kvDim, bias: false, groupSize: gs, bits: bits))
            _vProj = ModuleInfo(wrappedValue: QuantizedLinear(dim, kvDim, bias: false, groupSize: gs, bits: bits))
            _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps))
            self.vNorm = Gemma4RMSNormNoScale(dimensions: headDim, eps: c.rmsNormEps)
        } else {
            _kProj = ModuleInfo(wrappedValue: nil)
            _vProj = ModuleInfo(wrappedValue: nil)
            _kNorm = ModuleInfo(wrappedValue: nil)
            self.vNorm = nil
        }

        if isSliding {
            self.slidingRope = RoPE(dimensions: headDim, traditional: false, base: c.slidingRopeTheta)
            self.fullRope = nil
        } else {
            let rotated = Int(Float(headDim) * c.fullPartialRotaryFactor)
            self.fullRope = Gemma4ProportionalRoPE(dims: headDim, rotatedDims: rotated, base: c.fullRopeTheta)
            self.slidingRope = nil
        }

        super.init()
    }

    private func applyRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let r = slidingRope { return r(x, offset: offset) }
        return fullRope!(x, offset: offset)
    }

    /// - Parameters:
    ///   - sharedKV: post-RoPE (keys, values) from the producing layer (KV-shared layers only).
    ///     When non-nil this is the FULL cache to attend over (no past is concatenated).
    ///   - pastKV: previously cached post-RoPE (keys, values) for a *producing* layer in incremental
    ///     decode; this step's new K/V are concatenated onto it. Ignored when `sharedKV` is set.
    ///   - mask: additive float mask [1,1,Tq,Tk] (already causal / windowed).
    ///   - offset: RoPE position offset for the *new* tokens (== number of cached positions).
    /// - Returns: (output [B,T,hidden], full (keys,values) attended over [B,Hkv,Tk,D]).
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray, sharedKV: (MLXArray, MLXArray)?,
        pastKV: (MLXArray, MLXArray)? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = x.dim(0), t = x.dim(1)

        var q = qNorm(qProj(x).reshaped(b, t, nHeads, headDim))   // q_norm before transpose

        let keys: MLXArray
        let values: MLXArray
        if let (sk, sv) = sharedKV {
            // KV-shared layer: attend over the producing layer's full cache, read-only.
            keys = sk
            values = sv
        } else {
            var k = kNorm!(kProj!(x).reshaped(b, t, nKVHeads, headDim))
            k = k.transposed(0, 2, 1, 3)
            k = applyRope(k, offset: offset)

            var v = vNorm!(vProj!(x).reshaped(b, t, nKVHeads, headDim))
            v = v.transposed(0, 2, 1, 3)

            // Producing layer: append this step's post-RoPE K/V to the running cache.
            if let (pk, pv) = pastKV {
                k = concatenated([pk, k], axis: 2)
                v = concatenated([pv, v], axis: 2)
            }
            keys = k
            values = v
        }

        q = q.transposed(0, 2, 1, 3)
        q = applyRope(q, offset: offset)

        let attnOut = SDPA.attendAndMerge(
            qHeads: q, kHeads: keys, vHeads: values, scale: scale, mask: mask)
        return (oProj(attnOut), (keys, values))
    }
}

// MARK: - MLP (GeGLU; double-wide on KV-shared layers)

public final class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: QuantizedLinear
    @ModuleInfo(key: "up_proj") var upProj: QuantizedLinear
    @ModuleInfo(key: "down_proj") var downProj: QuantizedLinear

    public init(_ c: Gemma4DenseConfig, layerIdx: Int) {
        let h = c.hiddenSize, gs = c.quantGroupSize, bits = c.quantBits
        let useDoubleWide = c.useDoubleWideMLP && c.isKVSharedLayer(layerIdx)
        let inter = c.intermediateSize * (useDoubleWide ? 2 : 1)
        _gateProj = ModuleInfo(wrappedValue: QuantizedLinear(h, inter, bias: false, groupSize: gs, bits: bits))
        _upProj = ModuleInfo(wrappedValue: QuantizedLinear(h, inter, bias: false, groupSize: gs, bits: bits))
        _downProj = ModuleInfo(wrappedValue: QuantizedLinear(inter, h, bias: false, groupSize: gs, bits: bits))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // geglu(gate, x) = gelu_approx(gate) * x   (gate = gate_proj(x), x = up_proj(x))
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

public final class Gemma4Layer: Module {
    @ModuleInfo(key: "self_attn") var attn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    // Per-layer input gating.
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: QuantizedLinear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: QuantizedLinear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm

    // NB: reassigning a `@ParameterInfo` in init() drops any explicit `key:`, so the parameter is
    // registered under the Swift property name `layerScalar`; the loader updates it by that name.
    @ParameterInfo var layerScalar: MLXArray

    public init(_ c: Gemma4DenseConfig, layerIdx: Int) {
        _attn = ModuleInfo(wrappedValue: Gemma4Attention(c, layerIdx: layerIdx))
        _mlp = ModuleInfo(wrappedValue: Gemma4MLP(c, layerIdx: layerIdx))
        _inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _preFeedforwardLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        _postFeedforwardLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))

        let ple = c.hiddenSizePerLayerInput
        _perLayerInputGate = ModuleInfo(wrappedValue: QuantizedLinear(
            c.hiddenSize, ple, bias: false, groupSize: c.quantGroupSize, bits: c.quantBits))
        _perLayerProjection = ModuleInfo(wrappedValue: QuantizedLinear(
            ple, c.hiddenSize, bias: false, groupSize: c.quantGroupSize, bits: c.quantBits))
        _postPerLayerInputNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))

        _layerScalar = ParameterInfo(wrappedValue: MLXArray.ones([1]))
        super.init()
    }

    /// - Returns: (hidden_out, (keys,values) this layer used) so producing layers' K/V can be shared.
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray, perLayerInput: MLXArray,
        sharedKV: (MLXArray, MLXArray)?, pastKV: (MLXArray, MLXArray)? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        var residual = x

        var h = inputLayerNorm(x)
        let (attnOut, kv) = attn(h, mask: mask, sharedKV: sharedKV, pastKV: pastKV, offset: offset)
        h = postAttentionLayerNorm(attnOut)   // post-attn norm applied BEFORE the residual add
        h = residual + h

        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        // Per-layer input gating.
        residual = h
        var gate = perLayerInputGate(h)
        gate = geluApproximate(gate)
        gate = gate * perLayerInput
        gate = perLayerProjection(gate)
        gate = postPerLayerInputNorm(gate)
        h = residual + gate

        h = h * layerScalar
        return (h, kv)
    }
}

// MARK: - Model

public final class Gemma4Model: Module {
    public let config: Gemma4DenseConfig
    let embedScale: Float
    let perLayerInputScale: Float        // 2^-0.5
    let perLayerProjectionScale: Float   // hidden^-0.5
    let embedTokensPerLayerScale: Float  // sqrt(hidden_size_per_layer_input)

    @ModuleInfo(key: "embed_tokens") var embedTokens: PreQuantizedEmbedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: PreQuantizedEmbedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: QuantizedLinear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm
    @ModuleInfo var layers: [Gemma4Layer]
    @ModuleInfo var norm: RMSNorm

    public init(config c: Gemma4DenseConfig) {
        self.config = c
        self.embedScale = sqrt(Float(c.hiddenSize))
        self.perLayerInputScale = pow(2.0, -0.5)
        self.perLayerProjectionScale = pow(Float(c.hiddenSize), -0.5)
        self.embedTokensPerLayerScale = sqrt(Float(c.hiddenSizePerLayerInput))

        _embedTokens = ModuleInfo(wrappedValue: PreQuantizedEmbedding(
            embeddingCount: c.vocabSize, dimensions: c.hiddenSize,
            groupSize: c.quantGroupSize, bits: c.quantBits))
        _embedTokensPerLayer = ModuleInfo(wrappedValue: PreQuantizedEmbedding(
            embeddingCount: c.vocabSizePerLayerInput,
            dimensions: c.numHiddenLayers * c.hiddenSizePerLayerInput,
            groupSize: c.quantGroupSize, bits: c.quantBits))
        _perLayerModelProjection = ModuleInfo(wrappedValue: QuantizedLinear(
            c.hiddenSize, c.numHiddenLayers * c.hiddenSizePerLayerInput,
            bias: false, groupSize: c.quantGroupSize, bits: c.quantBits))
        _perLayerProjectionNorm = ModuleInfo(wrappedValue: RMSNorm(
            dimensions: c.hiddenSizePerLayerInput, eps: c.rmsNormEps))
        _layers = ModuleInfo(wrappedValue: (0..<c.numHiddenLayers).map { Gemma4Layer(c, layerIdx: $0) })
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps))
        super.init()
    }

    // _get_per_layer_inputs: lookup PLE table → *sqrt(256) → reshape (B,T,L,256).
    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        let b = inputIds.dim(0), t = inputIds.dim(1)
        var result = embedTokensPerLayer(inputIds)              // [B,T, L*256]
        result = result * embedTokensPerLayerScale
        return result.reshaped(b, t, config.numHiddenLayers, config.hiddenSizePerLayerInput)
    }

    // _project_per_layer_inputs: project hidden → *hidden^-0.5 → reshape → RMSNorm(256) →
    //   (proj + per_layer_inputs) * 2^-0.5.
    private func projectPerLayerInputs(_ hidden: MLXArray, _ perLayerInputs: MLXArray) -> MLXArray {
        let b = hidden.dim(0), t = hidden.dim(1)
        var proj = perLayerModelProjection(hidden)             // [B,T, L*256]
        proj = proj * perLayerProjectionScale
        proj = proj.reshaped(b, t, config.numHiddenLayers, config.hiddenSizePerLayerInput)
        proj = perLayerProjectionNorm(proj)
        return (proj + perLayerInputs) * perLayerInputScale
    }

    /// Build an additive float mask [1,1,T,T]: causal, with an optional sliding window.
    private func makeMask(seqLen t: Int, windowSize: Int?, dtype: DType) -> MLXArray {
        makeMask(queryLen: t, keyLen: t, offset: 0, windowSize: windowSize, dtype: dtype)
    }

    /// Build an additive float mask [1,1,Tq,Tk] for incremental decode: the `Tq` new queries
    /// live at absolute positions `offset ..< offset+Tq`; the `Tk` keys at `0 ..< Tk`
    /// (`Tk == offset + Tq` for a contiguous cache). Causal, with an optional sliding window.
    private func makeMask(queryLen tq: Int, keyLen tk: Int, offset: Int,
                          windowSize: Int?, dtype: DType) -> MLXArray {
        // query absolute pos qi = offset + i ; key absolute pos kj = j.
        let qInds = (MLXArray(Int32(0)..<Int32(tq)) + Int32(offset)).reshaped(tq, 1)
        let kInds = MLXArray(Int32(0)..<Int32(tk)).reshaped(1, tk)
        var keep = qInds .>= kInds                       // causal
        if let w = windowSize {
            keep = logicalAnd(keep, qInds .< (kInds + w))  // sliding window
        }
        let zeros = MLXArray.zeros([tq, tk], dtype: dtype)
        let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(dtype)
        let mask = MLX.where(keep, zeros, neg)
        return mask.reshaped(1, 1, tq, tk)
    }

    /// Debug flag — prints per-stage tensor statistics matching the Python reference dump.
    public static var debugStats = false
    private func stat(_ name: String, _ x: MLXArray) {
        guard Gemma4Model.debugStats else { return }
        let xf = x.asType(.float32)
        eval(xf)
        let mean = xf.mean().item(Float.self)
        let variance = MLX.mean(MLX.square(xf - mean)).item(Float.self)
        let std = variance.squareRoot()
        let absmax = MLX.abs(xf).max().item(Float.self)
        let first3 = Array(xf.flattened().asArray(Float.self).prefix(3))
        print(String(format: "[swift] %@: mean=%.5f std=%.5f absmax=%.5f first3=%@",
                     name, mean, std, absmax, "\(first3.map { String(format: "%.5f", $0) })"))
    }

    /// Hidden states after the final norm — `[B,T,hidden]`.
    public func hidden(inputIds: MLXArray) -> MLXArray {
        let t = inputIds.dim(1)

        // Initial hidden state.
        var h = embedTokens(inputIds)
        h = h * embedScale
        let dtype = h.dtype
        stat("h_init", h)

        // Per-layer inputs.
        let pleRaw = getPerLayerInputs(inputIds)
        stat("ple_raw", pleRaw)
        let perLayer = projectPerLayerInputs(h, pleRaw)        // [B,T,L,256]
        stat("ple_proj", perLayer)

        // Masks per layer type.
        let fullMask = makeMask(seqLen: t, windowSize: nil, dtype: dtype)
        let slidingMask = makeMask(seqLen: t, windowSize: config.slidingWindow, dtype: dtype)

        let prevKVs = config.previousKVs
        var produced: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: layers.count)

        for (i, layer) in layers.enumerated() {
            let isFull = config.layerTypes[i] == "full_attention"
            let mask = isFull ? fullMask : slidingMask
            let perLayerInput = perLayer[0..., 0..., i, 0...]    // [B,T,256]
            let shared = (prevKVs[i] == i) ? nil : produced[prevKVs[i]]
            let (nh, kv) = layer(h, mask: mask, perLayerInput: perLayerInput,
                                 sharedKV: shared, offset: 0)
            h = nh
            produced[i] = kv
            if Gemma4Model.debugStats, i < 3 || i == 14 || i == 15 || i == 34 {
                stat("after_layer_\(i)", h)
            }
        }

        let hn = norm(h)
        stat("after_norm", hn)
        return hn
    }

    /// - Returns: logits `[B,T,vocab]` with the final logit softcap applied.
    public func forward(inputIds: MLXArray) -> MLXArray {
        let h = hidden(inputIds: inputIds)
        var logits = embedTokens.asLinear(h)   // tied lm_head
        let cap = config.finalLogitSoftcapping
        logits = MLX.tanh(logits / cap) * cap
        return logits
    }

    // MARK: - Incremental generation (KV-cache + cross-layer KV sharing)

    /// Per-conversation decode state. We keep a KV cache only for the M = (num_layers −
    /// num_kv_shared_layers) *producing* layers (slots for shared layers stay nil); each
    /// shared layer attends over the cache of its mapped producing layer (`previousKVs`).
    /// `position` is the number of cached token positions (== the RoPE offset for the next step).
    public struct InferenceState {
        public var kvCaches: [(MLXArray, MLXArray)?]   // indexed by layer; nil on shared layers
        public var position: Int

        public static func initial(config c: Gemma4DenseConfig) -> InferenceState {
            InferenceState(kvCaches: Array(repeating: nil, count: c.numHiddenLayers), position: 0)
        }
    }

    /// Incremental forward over `T` new tokens, updating `state`'s caches in place.
    ///
    /// Prompt prefill passes the whole prompt (T = prompt length, position 0); each decode step
    /// passes a single token (T = 1). Producing layers append their post-RoPE K/V to their cache;
    /// shared layers read the producing layer's (now-updated) cache read-only.
    ///
    /// NOTE: sliding-window layers use a simple growing cache (no eviction). For the short voice
    /// turns this backend targets, sequence length stays well under `slidingWindow`, so the
    /// windowed mask alone already reproduces the reference attention; eviction would only matter
    /// for very long contexts.
    public func forward(inputIds: MLXArray, state: inout InferenceState) -> MLXArray {
        let t = inputIds.dim(1)
        let offset = state.position

        var h = embedTokens(inputIds)
        h = h * embedScale
        let dtype = h.dtype

        let pleRaw = getPerLayerInputs(inputIds)
        let perLayer = projectPerLayerInputs(h, pleRaw)        // [B,T,L,256]

        // Masks: keys span [0, offset+t) — a contiguous growing cache. Tq = t, Tk = offset+t.
        let tk = offset + t
        let fullMask = makeMask(queryLen: t, keyLen: tk, offset: offset, windowSize: nil, dtype: dtype)
        let slidingMask = makeMask(queryLen: t, keyLen: tk, offset: offset,
                                   windowSize: config.slidingWindow, dtype: dtype)

        let prevKVs = config.previousKVs
        for (i, layer) in layers.enumerated() {
            let isFull = config.layerTypes[i] == "full_attention"
            let mask = isFull ? fullMask : slidingMask
            let perLayerInput = perLayer[0..., 0..., i, 0...]    // [B,T,256]

            let isProducing = (prevKVs[i] == i)
            let shared = isProducing ? nil : state.kvCaches[prevKVs[i]]
            let past = isProducing ? state.kvCaches[i] : nil
            let (nh, kv) = layer(h, mask: mask, perLayerInput: perLayerInput,
                                 sharedKV: shared, pastKV: past, offset: offset)
            h = nh
            if isProducing { state.kvCaches[i] = kv }
        }

        state.position = tk

        let hn = norm(h)
        var logits = embedTokens.asLinear(hn)   // tied lm_head
        let cap = config.finalLogitSoftcapping
        logits = MLX.tanh(logits / cap) * cap
        return logits
    }
}
