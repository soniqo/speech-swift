import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen2Attention: Module {
    public let config: Qwen2Configuration
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var wq: Linear
    @ModuleInfo(key: "k_proj") public var wk: Linear
    @ModuleInfo(key: "v_proj") public var wv: Linear
    @ModuleInfo(key: "o_proj") public var wo: Linear

    public let rope: RoPE

    public init(_ config: Qwen2Configuration) {
        self.config = config

        let dim = config.hiddenSize
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: config.ropeTheta
        )

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .causal,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, config.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)

        if let cache = cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let effectiveMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let cache = cache {
            effectiveMask = createAttentionMask(h: x, cache: cache)
        } else {
            effectiveMask = mask
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: effectiveMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// Pure-functional single-step variant suitable for `MLX.compile(shapeless: true)`.
    ///
    /// Cache is passed/returned as explicit `(K, V)` MLXArray tuples instead of via
    /// `inout KVCacheSimple`, and `offset` is an `MLXArray` so `compile` treats it as
    /// a runtime input rather than baking the value as a constant. With shapeless
    /// compile, the same compiled graph handles every cache length — no per-shape
    /// recompile during the autoregressive loop.
    ///
    /// `attentionMask == nil` is the autoregressive-decode path (single new query
    /// attends to all cached keys; SDPA implicitly handles it). Multi-token prefill
    /// callers must supply an explicit causal mask.
    public func forwardStep(
        _ x: MLXArray,
        offset: MLXArray,
        attentionMask: MLXArray? = nil,
        cache: (MLXArray, MLXArray)? = nil
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let L = x.dim(1)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // Use -1 in batch dim so the compiled graph works for any batch size.
        // Only one -1 is allowed per reshape — head_dim is given explicitly.
        queries = queries.reshaped(-1, L, config.attentionHeads, config.headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(-1, L, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(-1, L, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)

        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let allKeys: MLXArray
        let allValues: MLXArray
        if let (prevK, prevV) = cache {
            allKeys = concatenated([prevK, keys], axis: 2)
            allValues = concatenated([prevV, values], axis: 2)
        } else {
            allKeys = keys
            allValues = values
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode = {
            if let m = attentionMask { return .array(m) }
            return .none
        }()

        let attended = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: allKeys,
            values: allValues,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(-1, L, config.attentionHeads * config.headDim)

        return (wo(attended), (allKeys, allValues))
    }
}
