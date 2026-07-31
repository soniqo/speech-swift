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
// `hidden(inputIds:)` is the parity port: one forward pass over the prompt with nothing cached,
// where shared layers reuse the (keys, values) their producing layer computed in the same pass.
// Generation goes through `Gemma4KVCache` instead, and the two are held against each other by
// `E2EGemma4KVCacheTests`.

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

        // Quantization lives at the root of the MLX config; older exports put it beside
        // the text fields instead, so both are searched.
        let quant = ChatQuantization.resolve(searching: [root, tc])

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
            quantGroupSize: quant.groupSize,
            quantBits: quant.bits
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

// MARK: - Blocked attention geometry

/// One query block of a blocked attention pass: which of this call's queries it covers, which
/// slice of the key axis those queries can reach, and how that block is masked.
///
/// A sliding layer's query only reaches `sliding_window` keys back, so a block of `Q` consecutive
/// queries reaches at most `Q + sliding_window - 1` of them. Scoring the block against that slice
/// is what makes prefill near-linear in prompt length. The single `[1,1,Tq,Tk]` mask this replaced
/// let the window decide which scores *survived*, never how many were *computed*: all 42 layers
/// built the complete Tq×Tk matrix and then threw ~97% of it away.
struct Gemma4AttentionBlock {
    let queryStart: Int   // half-open range into this call's query axis
    let queryEnd: Int
    let keyStart: Int     // half-open range into the (cache-concatenated) key axis
    let keyEnd: Int
    let mask: MLXFast.ScaledDotProductAttentionMaskMode
}

// MARK: - KV cache

/// The key axis one attention pass runs over: `length` keys, the first of which holds absolute
/// token position `base`.
///
/// `base` is 0 for a cache that still starts where the sequence does, and only moves once a sliding
/// layer begins dropping what it can no longer read.
struct Gemma4KeyAxis {
    var base: Int
    var length: Int
}

/// One producing layer's post-RoPE K/V cache.
///
/// The arrays attention reads *are* this buffer, so a decode step writes its one token into spare
/// capacity instead of allocating a new array and copying every cached token into it to add one.
/// What that replaces is `concatenated([past, new])`, once per producing layer per step: 57 KB per
/// token of context across the 24 of them, so 975 MB copied on every step at 17k tokens and O(n²)
/// over a generation. MLX donates a slice assignment's input buffer when nothing else holds it, so
/// the write lands in place and the per-step cost stops depending on how much is cached.
///
/// `base` is the absolute token position of entry 0. A sliding layer's attention cannot read
/// further back than `slidingWindow`, so entries that fall out of it are dropped and `base` moves;
/// every key range in `attentionBlocks` is derived in absolute positions and translated through it.
/// Global layers keep everything, so their `base` stays 0.
///
/// A reference type deliberately. The write above is visible through every reference to the cache,
/// so a value type would promise a copy it does not make.
public final class Gemma4KVCache {
    /// `[B, Hkv, capacity, D]`, nil until the first append. Only `0 ..< count` along the token axis
    /// holds real entries; the rest is the room the next decode steps write into.
    private var storedKeys: MLXArray?
    private var storedValues: MLXArray?

    /// Live entries — the post-RoPE K/V for absolute positions `base ..< base + count`.
    public private(set) var count = 0
    /// Absolute token position of entry 0.
    public private(set) var base = 0

    /// How many of the newest entries a later query can still reach, or nil where none of them ever
    /// expire. A sliding layer's next query reaches `slidingWindow - 1` tokens behind itself and no
    /// further, so that is exactly what has to survive it; a global layer's reaches everything.
    private let retention: Int?

    /// Entries left spare at the end of the buffer when it is rebuilt — how many decode steps can
    /// then be written in place before the next rebuild. One window's worth, which costs a sliding
    /// layer as much again as it keeps: 21 MB of spare across the twenty of them beside the 21 MB
    /// they hold. That is bought against the global caches, which may drop nothing and so rebuild
    /// the whole 285 MB they reach at 17k tokens — amortised over 512 steps, half a megabyte a step
    /// against the 975 MB a step this replaced.
    private static let spareCapacity = 512

    init(retention: Int?, startingAt position: Int) {
        self.retention = retention
        self.base = position
    }

    private var capacity: Int { storedKeys?.dim(2) ?? 0 }

    /// How many of the oldest entries `append` will drop to fit `t` more.
    ///
    /// Only a rebuild can drop anything — while the buffer has room the write goes in place and
    /// nothing moves — and a rebuild drops only what has expired. Everything else is kept, so this
    /// is 0 far more often than not.
    private func dropped(appending t: Int) -> Int {
        guard storedKeys != nil, count + t > capacity, let retention else { return 0 }
        return count - min(count, retention)
    }

    /// The key axis the pass appending `t` entries will attend over.
    ///
    /// The pass has to ask before it appends, because it plans one geometry for every layer of a
    /// type and the append is what moves `base`. Asking the cache rather than re-deriving the rule
    /// keeps the answer and the act the same decision.
    func keyAxis(appending t: Int) -> Gemma4KeyAxis {
        let expired = dropped(appending: t)
        return Gemma4KeyAxis(base: base + expired, length: count - expired + t)
    }

    /// The keys and values attention reads: entries `0 ..< count`, with the spare room excluded.
    /// Valid only once this pass's tokens have been appended, which for a KV-shared layer its
    /// producing layer has always already done.
    func window() -> (MLXArray, MLXArray) {
        guard let storedKeys, let storedValues else {
            preconditionFailure("Gemma4KVCache read before anything was appended to it")
        }
        guard count < capacity else { return (storedKeys, storedValues) }
        return (storedKeys[0..., 0..., 0 ..< count, 0...],
                storedValues[0..., 0..., 0 ..< count, 0...])
    }

    /// Append this pass's post-RoPE K/V, in place wherever the buffer still has room for them.
    func append(keys k: MLXArray, values v: MLXArray) {
        let t = k.dim(2)
        guard let storedKeys, let storedValues else {
            // Nothing cached yet, so there is nothing to copy these onto: the projection's own
            // arrays become the buffer, and a prompt prefill costs no copy at all. `trim` hands a
            // sliding layer its spare room back immediately afterwards; a global layer, which may
            // drop nothing, grows on its first decode step instead.
            self.storedKeys = k
            self.storedValues = v
            count = t
            return
        }
        guard count + t <= capacity else {
            rebuild(keeping: count - dropped(appending: t), appending: (k, v))
            return
        }
        storedKeys[0..., 0..., count ..< (count + t), 0...] = k
        storedValues[0..., 0..., count ..< (count + t), 0...] = v
        count += t
    }

    /// Drop whatever no later query can reach, between passes.
    ///
    /// `append` already drops what has expired whenever it has to rebuild, so in steady decode this
    /// changes nothing. It is for the pass that stored a whole prompt into a window-bounded cache —
    /// its own early queries needed all of it — which would otherwise hold the entire prompt in
    /// twenty sliding layers until the next token asked for room.
    func trim() {
        guard let retention, count > retention + Self.spareCapacity else { return }
        rebuild(keeping: retention, appending: nil)
    }

    /// Move the newest `keep` entries, plus any new ones, into a buffer sized for them and the
    /// spare room, and advance `base` past what was left behind.
    private func rebuild(keeping keep: Int, appending new: (MLXArray, MLXArray)?) {
        guard let oldKeys = storedKeys, let oldValues = storedValues else { return }
        let dropped = count - keep
        let added = new?.0.dim(2) ?? 0
        let live = keep + added
        let capacity = live + Self.spareCapacity

        func rebuilt(_ old: MLXArray, _ appended: MLXArray?) -> MLXArray {
            let grown = MLXArray.zeros([old.dim(0), old.dim(1), capacity, old.dim(3)],
                                       dtype: old.dtype)
            if keep > 0 {
                grown[0..., 0..., 0 ..< keep, 0...] = old[0..., 0..., dropped ..< count, 0...]
            }
            if let appended { grown[0..., 0..., keep ..< live, 0...] = appended }
            return grown
        }

        storedKeys = rebuilt(oldKeys, new?.0)
        storedValues = rebuilt(oldValues, new?.1)
        base += dropped
        count = live
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
    ///     When non-nil this is the FULL cache to attend over (nothing of this call's is added).
    ///   - cache: a *producing* layer's running cache in incremental decode. This call's post-RoPE
    ///     K/V are appended to it and the keys attended over are the window it then holds. Ignored
    ///     when `sharedKV` is set, and nil for the whole-prompt forward, which caches nothing.
    ///   - offset: RoPE position offset for the *new* tokens (== number of positions already seen).
    /// - Returns: (q [B,Hq,T,D], full keys, full values [B,Hkv,Tk,D]).
    private func projectQKV(
        _ x: MLXArray, sharedKV: (MLXArray, MLXArray)?, cache: Gemma4KVCache?, offset: Int
    ) -> (MLXArray, MLXArray, MLXArray) {
        let b = x.dim(0), t = x.dim(1)

        var q = qNorm(qProj(x).reshaped(b, t, nHeads, headDim))   // q_norm before transpose

        let keys: MLXArray
        let values: MLXArray
        if let (sk, sv) = sharedKV {
            // KV-shared layer: attend over the producing layer's window, read-only.
            keys = sk
            values = sv
        } else {
            var k = kNorm!(kProj!(x).reshaped(b, t, nKVHeads, headDim))
            k = k.transposed(0, 2, 1, 3)
            k = applyRope(k, offset: offset)

            var v = vNorm!(vProj!(x).reshaped(b, t, nKVHeads, headDim))
            v = v.transposed(0, 2, 1, 3)

            // Producing layer: hand this step's post-RoPE K/V to the cache, which writes them into
            // spare capacity rather than rebuilding an array around them.
            if let cache {
                cache.append(keys: k, values: v)
                let window = cache.window()
                keys = window.0
                values = window.1
            } else {
                keys = k
                values = v
            }
        }

        q = q.transposed(0, 2, 1, 3)
        q = applyRope(q, offset: offset)
        return (q, keys, values)
    }

    /// Reference path: one SDPA call against the whole key axis with a precomputed additive
    /// `[1,1,Tq,Tk]` mask. Superseded by the blocked overload below and retained because it is the
    /// implementation the blocked one is measured against.
    ///
    /// - Returns: (output [B,T,hidden], full (keys,values) attended over [B,Hkv,Tk,D]).
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray, sharedKV: (MLXArray, MLXArray)?,
        cache: Gemma4KVCache? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (q, keys, values) = projectQKV(x, sharedKV: sharedKV, cache: cache, offset: offset)
        let attnOut = SDPA.attendAndMerge(
            qHeads: q, kHeads: keys, vHeads: values, scale: scale, mask: mask)
        return (oProj(attnOut), (keys, values))
    }

    /// Blocked path: each block scores its queries against only the keys they can reach, so a
    /// sliding layer never materialises a score matrix wider than its window. Q/K/V projections and
    /// RoPE still run once over the whole call — they are linear in length and cheap to share — and
    /// the full (keys, values) are returned unsliced so they can still be cached and shared.
    ///
    /// The per-block outputs are concatenated along the token axis before `o_proj`, so the output
    /// projection stays one matmul rather than one per block.
    func callAsFunction(
        _ x: MLXArray, blocks: [Gemma4AttentionBlock], sharedKV: (MLXArray, MLXArray)?,
        cache: Gemma4KVCache? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (q, keys, values) = projectQKV(x, sharedKV: sharedKV, cache: cache, offset: offset)

        let attnOut: MLXArray
        if blocks.count == 1 {
            let block = blocks[0]
            attnOut = SDPA.attendAndMerge(
                qHeads: q, kHeads: keySlice(keys, block), vHeads: keySlice(values, block),
                scale: scale, mask: block.mask)
        } else {
            var parts: [MLXArray] = []
            parts.reserveCapacity(blocks.count)
            for block in blocks {
                parts.append(SDPA.attendAndMerge(
                    qHeads: q[0..., 0..., block.queryStart ..< block.queryEnd, 0...],
                    kHeads: keySlice(keys, block), vHeads: keySlice(values, block),
                    scale: scale, mask: block.mask))
            }
            attnOut = concatenated(parts, axis: 1)
        }
        return (oProj(attnOut), (keys, values))
    }

    /// Key/value slice a block attends over, left alone when the block already spans everything
    /// (short prompts and single-block passes) so nothing pays for a no-op view.
    @inline(__always)
    private func keySlice(_ kv: MLXArray, _ block: Gemma4AttentionBlock) -> MLXArray {
        (block.keyStart == 0 && block.keyEnd == kv.dim(2))
            ? kv : kv[0..., 0..., block.keyStart ..< block.keyEnd, 0...]
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
        sharedKV: (MLXArray, MLXArray)?, cache: Gemma4KVCache? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (attnOut, kv) = attn(inputLayerNorm(x), mask: mask, sharedKV: sharedKV,
                                 cache: cache, offset: offset)
        return (feedForward(residual: x, attnOut: attnOut, perLayerInput: perLayerInput), kv)
    }

    /// Blocked attention variant — identical outside the attention call itself.
    func callAsFunction(
        _ x: MLXArray, blocks: [Gemma4AttentionBlock], perLayerInput: MLXArray,
        sharedKV: (MLXArray, MLXArray)?, cache: Gemma4KVCache? = nil, offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (attnOut, kv) = attn(inputLayerNorm(x), blocks: blocks, sharedKV: sharedKV,
                                 cache: cache, offset: offset)
        return (feedForward(residual: x, attnOut: attnOut, perLayerInput: perLayerInput), kv)
    }

    /// Everything after the attention call: post-attn norm, the sandwiched FFN, per-layer input
    /// gating, and the layer scalar. Shared by both attention paths so they can only differ in how
    /// the scores were computed.
    private func feedForward(
        residual attnResidual: MLXArray, attnOut: MLXArray, perLayerInput: MLXArray
    ) -> MLXArray {
        var h = postAttentionLayerNorm(attnOut)   // post-attn norm applied BEFORE the residual add
        h = attnResidual + h

        var residual = h
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

        return h * layerScalar
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

    /// Layers that own a KV cache — the ones `previousKVs` maps to themselves.
    private let producingLayers: [Int]
    /// The first producing layer of each type. Its cache answers for every cache of that type,
    /// since they are all appended to and trimmed together — see `hidden(inputIds:state:)`.
    private let fullProducer: Int?
    private let slidingProducer: Int?

    public init(config c: Gemma4DenseConfig) {
        self.config = c
        let previous = c.previousKVs
        self.producingLayers = (0..<c.numHiddenLayers).filter { previous[$0] == $0 }
        self.fullProducer = producingLayers.first { c.layerTypes[$0] == "full_attention" }
        self.slidingProducer = producingLayers.first { c.layerTypes[$0] != "full_attention" }
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

    // MARK: - Attention geometry

    /// Queries per attention block.
    ///
    /// A sliding layer's per-block key span is `queryBlock + sliding_window - 1`, so its whole pass
    /// costs `T · (queryBlock + window - 1)` key comparisons against the `T²` the full mask paid: at
    /// the checkpoint's 512-token window and a 17k prompt that is a 17× reduction, and the ratio
    /// keeps improving with length. Smaller blocks shave a little more arithmetic but launch more
    /// kernels; 512 keeps each block's matmul large (8 heads × 512 queries × ~1023 keys) while the
    /// cost is already within 2× of the `T · window` floor. The value is deliberately independent of
    /// `sliding_window` — every interior block has the same relative geometry whatever it is, so one
    /// mask array serves all of them and all 35 sliding layers.
    ///
    /// The seven global layers are blocked at the same size even though they have no band to
    /// exploit, which looks pointless and is not. Handing MLX a single end-aligned causal call over
    /// the whole prompt instead was measured at 17,408 tokens to raise peak memory from 7.03 GB to
    /// 11.30 GB — a figure that is deterministic and repeated exactly run to run, unlike the wall
    /// times beside it, which drifted further with the machine's thermal state than the two variants
    /// differed. So MLX's SDPA reserves against the whole Tq×Tk score matrix rather than against the
    /// half a causal mask keeps, and bounding Tk per block is what caps the reservation, exactly as
    /// it does on the sliding layers. Do not special-case global layers back to one call without
    /// re-measuring peak memory.
    private static let queryBlock = 512

    /// Geometry that decides a block's mask: how many queries, how many keys, and where the block's
    /// first query sits inside its key slice. Interior blocks agree on all three.
    struct BlockMaskKey: Hashable {
        let queries: Int
        let keys: Int
        let queryOffset: Int
    }

    /// Plan one attention pass: `tq` new queries at absolute positions `offset ..< offset+tq`, over
    /// `tk` keys whose first holds absolute position `keyBase`. `windowSize` nil means global.
    ///
    /// `keyBase` is what a window-bounded cache costs. A sliding layer drops the entries it can no
    /// longer read, so key `j` is token `keyBase + j` rather than token `j`; every bound below is
    /// derived in absolute positions and translated onto the key axis once, at the end. Leave it 0
    /// for a cache that still starts where the sequence does.
    ///
    /// Two block shapes need no mask array at all. A one-query block (every decode step) reaches
    /// every key in its own slice, and a block whose slice starts at the first cached key with no
    /// query's window biting is exactly the end-aligned causal case MLX derives itself — which is
    /// every block of a full-attention layer, so those allocate no mask whatever the length.
    static func attentionBlocks(queryLen tq: Int, keyLen tk: Int, offset: Int, keyBase: Int = 0,
                                windowSize: Int?, dtype: DType) -> [Gemma4AttentionBlock] {
        var blocks: [Gemma4AttentionBlock] = []
        var masks: [BlockMaskKey: MLXArray] = [:]
        var start = 0
        while start < tq {
            let end = min(start + queryBlock, tq)
            let firstQuery = offset + start
            let lastQuery = offset + end - 1
            // Causal: nothing above the block's last query. Windowed: nothing below the first
            // query's window floor, since a later query's floor is only ever higher. Both are
            // absolute positions, so both shift down by the position the key axis starts at.
            let keyEnd = min(lastQuery + 1 - keyBase, tk)
            let keyStart = windowSize.map { max(0, firstQuery - $0 + 1 - keyBase) } ?? 0

            let mask: MLXFast.ScaledDotProductAttentionMaskMode
            if end - start == 1 {
                mask = .none
            } else if keyStart == 0, windowSize.map({ lastQuery < $0 + keyBase }) ?? true {
                mask = .causal
            } else {
                let key = BlockMaskKey(queries: end - start, keys: keyEnd - keyStart,
                                       queryOffset: firstQuery - keyBase - keyStart)
                let array = masks[key] ?? blockMask(key, windowSize: windowSize, dtype: dtype)
                masks[key] = array
                mask = .array(array)
            }

            blocks.append(Gemma4AttentionBlock(queryStart: start, queryEnd: end,
                                               keyStart: keyStart, keyEnd: keyEnd, mask: mask))
            start = end
        }
        return blocks
    }

    /// Additive mask for one block — `[1,1,queries,keys]`, in the block's own coordinates: query `i`
    /// sits at key-slice position `queryOffset + i`, key `j` at position `j`.
    static func blockMask(_ key: BlockMaskKey, windowSize: Int?, dtype: DType) -> MLXArray {
        let qInds = (MLXArray(Int32(0)..<Int32(key.queries)) + Int32(key.queryOffset))
            .reshaped(key.queries, 1)
        let kInds = MLXArray(Int32(0)..<Int32(key.keys)).reshaped(1, key.keys)
        var keep = qInds .>= kInds                       // causal
        if let w = windowSize {
            keep = logicalAnd(keep, qInds .< (kInds + w)) // sliding window
        }
        let zeros = MLXArray.zeros([key.queries, key.keys], dtype: dtype)
        let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(dtype)
        return MLX.where(keep, zeros, neg).reshaped(1, 1, key.queries, key.keys)
    }

    /// How a pass drives its layers: the blocked geometry production uses, or the single full-size
    /// mask it replaced. The reference stays reachable so the parity test can hold one against the
    /// other on the same weights and the same prompt; nothing in production selects it.
    private enum AttentionPlan {
        case blocks(full: [Gemma4AttentionBlock], sliding: [Gemma4AttentionBlock])
        case referenceMasks(full: MLXArray, sliding: MLXArray)
    }

    /// Test-only: fall back to the full `[1,1,Tq,Tk]` mask for every layer.
    static var usesReferenceFullMask = false

    /// The two layer types no longer share a key axis: a sliding layer's cache is bounded to its
    /// window and starts wherever eviction has left it, while a global layer's still holds the whole
    /// sequence from position 0. So each is planned against the axis its own caches will present.
    private func attentionPlan(queryLen tq: Int, offset: Int, dtype: DType,
                               full: Gemma4KeyAxis, sliding: Gemma4KeyAxis) -> AttentionPlan {
        if Gemma4Model.usesReferenceFullMask {
            return .referenceMasks(
                full: Self.makeMask(queryLen: tq, keyLen: full.length, offset: offset,
                                    keyBase: full.base, windowSize: nil, dtype: dtype),
                sliding: Self.makeMask(queryLen: tq, keyLen: sliding.length, offset: offset,
                                       keyBase: sliding.base, windowSize: config.slidingWindow,
                                       dtype: dtype))
        }
        return .blocks(
            full: Self.attentionBlocks(queryLen: tq, keyLen: full.length, offset: offset,
                                       keyBase: full.base, windowSize: nil, dtype: dtype),
            sliding: Self.attentionBlocks(queryLen: tq, keyLen: sliding.length, offset: offset,
                                          keyBase: sliding.base, windowSize: config.slidingWindow,
                                          dtype: dtype))
    }

    /// Run one layer under the pass's plan.
    private func run(_ layer: Gemma4Layer, _ h: MLXArray, plan: AttentionPlan, isFull: Bool,
                     perLayerInput: MLXArray, sharedKV: (MLXArray, MLXArray)?,
                     cache: Gemma4KVCache?, offset: Int)
        -> (MLXArray, (MLXArray, MLXArray)) {
        switch plan {
        case .blocks(let full, let sliding):
            return layer(h, blocks: isFull ? full : sliding, perLayerInput: perLayerInput,
                         sharedKV: sharedKV, cache: cache, offset: offset)
        case .referenceMasks(let full, let sliding):
            return layer(h, mask: isFull ? full : sliding, perLayerInput: perLayerInput,
                         sharedKV: sharedKV, cache: cache, offset: offset)
        }
    }

    /// Build an additive float mask [1,1,Tq,Tk]: the `Tq` new queries live at absolute positions
    /// `offset ..< offset+Tq`; the `Tk` keys at `keyBase ..< keyBase+Tk`, `keyBase` being where the
    /// cache starts once a sliding layer has dropped what it cannot read. Causal, with an optional
    /// sliding window. Reference geometry only — see `AttentionPlan`.
    static func makeMask(queryLen tq: Int, keyLen tk: Int, offset: Int, keyBase: Int = 0,
                         windowSize: Int?, dtype: DType) -> MLXArray {
        // query absolute pos qi = offset + i ; key absolute pos kj = keyBase + j.
        let qInds = (MLXArray(Int32(0)..<Int32(tq)) + Int32(offset)).reshaped(tq, 1)
        let kInds = (MLXArray(Int32(0)..<Int32(tk)) + Int32(keyBase)).reshaped(1, tk)
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

        // Attention geometry per layer type — planned once and shared by every layer of that type.
        // Nothing is cached on this path, so both types' keys are this pass's own tokens.
        let plan = attentionPlan(queryLen: t, offset: 0, dtype: dtype,
                                 full: Gemma4KeyAxis(base: 0, length: t),
                                 sliding: Gemma4KeyAxis(base: 0, length: t))

        let prevKVs = config.previousKVs
        var produced: [(MLXArray, MLXArray)?] = Array(repeating: nil, count: layers.count)

        for (i, layer) in layers.enumerated() {
            let isFull = config.layerTypes[i] == "full_attention"
            let perLayerInput = perLayer[0..., 0..., i, 0...]    // [B,T,256]
            let shared = (prevKVs[i] == i) ? nil : produced[prevKVs[i]]
            let (nh, kv) = run(layer, h, plan: plan, isFull: isFull, perLayerInput: perLayerInput,
                               sharedKV: shared, cache: nil, offset: 0)
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
        softcapped(embedTokens.asLinear(hidden(inputIds: inputIds)))   // tied lm_head
    }

    /// `tanh(logits / cap) * cap` — Gemma 4's final logit softcap.
    private func softcapped(_ logits: MLXArray) -> MLXArray {
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }

    // MARK: - Incremental generation (KV-cache + cross-layer KV sharing)

    /// Per-conversation decode state. We keep a KV cache only for the M = (num_layers −
    /// num_kv_shared_layers) *producing* layers (slots for shared layers stay nil); each
    /// shared layer attends over the cache of its mapped producing layer (`previousKVs`).
    /// `position` is the number of token positions seen (== the RoPE offset for the next step).
    ///
    /// The caches are reference types that a step writes into, so this is a handle to them rather
    /// than a snapshot of them: copy it and both copies drive the same buffers. Start a second
    /// conversation with `initial` instead.
    public struct InferenceState {
        public var kvCaches: [Gemma4KVCache?]   // indexed by layer; nil on shared layers
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
    public func forward(inputIds: MLXArray, state: inout InferenceState) -> MLXArray {
        softcapped(embedTokens.asLinear(hidden(inputIds: inputIds, state: &state)))  // tied lm_head
    }

    /// Logits for the final position only — `[B,1,vocab]`, softcapped like `forward`'s.
    ///
    /// The lm_head projects onto a 262k vocabulary, so running it over every position turns a
    /// 17k-token prompt into a `[1,17408,262144]` bfloat16 array — 9 GB, and again as much for the
    /// softcap's intermediates — to produce the one row generation samples from. Prefill takes this
    /// path instead. The token axis is kept so callers can index the last position either way.
    ///
    /// Internal: `forward` remains the whole public surface, so nothing outside the module has to
    /// choose between two ways of prefilling.
    func lastTokenLogits(inputIds: MLXArray, state: inout InferenceState) -> MLXArray {
        let hn = hidden(inputIds: inputIds, state: &state)
        let t = hn.dim(1)
        return softcapped(embedTokens.asLinear(hn[0..., (t - 1) ..< t, 0...]))
    }

    /// Hidden states after the final norm for `T` new tokens, updating `state`'s caches in place.
    private func hidden(inputIds: MLXArray, state: inout InferenceState) -> MLXArray {
        let t = inputIds.dim(1)
        let offset = state.position

        var h = embedTokens(inputIds)
        h = h * embedScale
        let dtype = h.dtype

        let pleRaw = getPerLayerInputs(inputIds)
        let perLayer = projectPerLayerInputs(h, pleRaw)        // [B,T,L,256]

        // A sliding layer's next query reaches `slidingWindow - 1` tokens behind itself, so that is
        // all its cache has to keep; a global layer keeps everything.
        for i in producingLayers where state.kvCaches[i] == nil {
            state.kvCaches[i] = Gemma4KVCache(
                retention: config.layerTypes[i] == "full_attention"
                    ? nil : max(0, config.slidingWindow - 1),
                startingAt: offset)
        }

        // Every producing layer of one type appends the same tokens and drops by the same rule, so
        // one of them answers for all of them — and for the KV-shared layers reading their buffers.
        // Asked before anything is appended, because one geometry has to serve the whole pass.
        let plan = attentionPlan(
            queryLen: t, offset: offset, dtype: dtype,
            full: keyAxis(of: fullProducer, in: state, appending: t),
            sliding: keyAxis(of: slidingProducer, in: state, appending: t))

        let prevKVs = config.previousKVs
        for (i, layer) in layers.enumerated() {
            let isFull = config.layerTypes[i] == "full_attention"
            let perLayerInput = perLayer[0..., 0..., i, 0...]    // [B,T,256]

            let isProducing = (prevKVs[i] == i)
            let shared = isProducing ? nil : state.kvCaches[prevKVs[i]]?.window()
            let (nh, _) = run(layer, h, plan: plan, isFull: isFull, perLayerInput: perLayerInput,
                              sharedKV: shared, cache: isProducing ? state.kvCaches[i] : nil,
                              offset: offset)
            h = nh
        }

        // Drop what the sliding layers can no longer read, now that every layer that could still
        // read it has run — a KV-shared layer's queries are this pass's queries and reach exactly
        // as far back, so nothing may be dropped before the last of them.
        //
        // Trimming each producing layer the moment its own layer finished, which needs the two
        // whose cache is shared held back to the end anyway, was tried and measured at a 17k
        // prefill to change peak memory by nothing at all (7.10 GB either way). The peak is set by
        // the activations mid-stack, not by the caches at the end of it, so the extra state that
        // variant needed bought a rounding error.
        for i in producingLayers { state.kvCaches[i]?.trim() }

        state.position = offset + t
        return norm(h)
    }

    /// The key axis a layer type's caches will all present. A config with no producing layer of a
    /// type has no layer that would read the answer, so any axis will do.
    private func keyAxis(of producer: Int?, in state: InferenceState,
                         appending t: Int) -> Gemma4KeyAxis {
        producer.flatMap { state.kvCaches[$0]?.keyAxis(appending: t) }
            ?? Gemma4KeyAxis(base: 0, length: t)
    }
}
