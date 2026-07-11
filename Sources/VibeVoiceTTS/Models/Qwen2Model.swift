import Foundation
import MLX
import MLXFast
import MLXNN

public class Qwen2Model: Module {
    public let config: Qwen2Configuration

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    public let layers: [Qwen2TransformerBlock]
    public let norm: RMSNorm

    public init(_ config: Qwen2Configuration) {
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        var layersList: [Qwen2TransformerBlock] = []
        for _ in 0..<config.hiddenLayers {
            layersList.append(Qwen2TransformerBlock(config))
        }
        self.layers = layersList

        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCacheSimple]? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        if let cache = cache {
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache[i])
            }
        } else {
            for layer in layers {
                h = layer(h, mask: mask, cache: nil)
            }
        }

        return norm(h)
    }

    public func forwardWithEmbeddings(
        _ embeddings: MLXArray,
        cache: [KVCacheSimple]? = nil,
        applyFinalNorm: Bool = true
    ) -> MLXArray {
        var h = embeddings

        let mask = createAttentionMask(h: h, cache: cache?.first)

        if let cache = cache {
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache[i])
            }
        } else {
            for layer in layers {
                h = layer(h, mask: mask, cache: nil)
            }
        }

        if applyFinalNorm {
            return norm(h)
        } else {
            return h
        }
    }

    private func createAttentionMask(h: MLXArray, cache: KVCacheSimple?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let n = h.dim(1)

        if n == 1 {
            return .none
        }

        let offset = cache?.offset ?? 0

        if offset == 0 {
            return .causal
        }

        let mask = createCausalMask(n: n, offset: offset)
        return .array(mask)
    }

    private func createCausalMask(n: Int, offset: Int) -> MLXArray {
        var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
        var linds = MLXArray(Int32(offset) ..< Int32(offset + n))
        linds = linds[0..., .newAxis]
        rinds = rinds[.newAxis]
        let mask = linds .>= rinds
        return mask
    }

    public func newCache() -> [KVCacheSimple] {
        (0..<config.hiddenLayers).map { _ in KVCacheSimple() }
    }

    // MARK: - Compiled autoregressive step

    /// Compiled single-token forward step. Set up by `setupCompilation()`. When
    /// non-nil, `executeStep(...)` uses it instead of the eager path.
    private var compiledStep: (([MLXArray]) -> [MLXArray])?

    /// Pure-functional single-step transformer pass. Cache is passed/returned
    /// as flat `[(K, V)]` tuples and `offset` is an `MLXArray` so `MLX.compile`
    /// treats it as a runtime input. With shapeless compile, the same graph
    /// services every cache length — no per-shape recompile per token.
    ///
    /// `attentionMask == nil` is the autoregressive single-token case (one new
    /// query, no future positions to mask). Multi-token prefill callers must
    /// supply an explicit causal mask.
    public func forwardStep(
        embeddings: MLXArray,
        offset: MLXArray,
        cache: [(MLXArray, MLXArray)]? = nil,
        attentionMask: MLXArray? = nil,
        applyFinalNorm: Bool = true
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        var h = embeddings
        var newCache: [(MLXArray, MLXArray)] = []
        newCache.reserveCapacity(layers.count)
        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let (out, updated) = layer.forwardStep(
                h, offset: offset, attentionMask: attentionMask, cache: layerCache
            )
            h = out
            newCache.append(updated)
        }
        if applyFinalNorm { h = norm(h) }
        return (h, newCache)
    }

    /// Install a `MLX.compile(shapeless: true)` wrapper around `forwardStep`.
    /// Subsequent `executeStep(...)` calls fuse all layer kernels into a single
    /// compiled graph. Call once after model load.
    public func setupCompilation() {
        let selfRef = self
        let numLayers = config.hiddenLayers

        compiledStep = compile(
            inputs: [selfRef], outputs: [selfRef], shapeless: true
        ) { inputs in
            let embeds = inputs[0]
            let offset = inputs[1]
            var cache: [(MLXArray, MLXArray)] = []
            cache.reserveCapacity(numLayers)
            for i in 0..<numLayers {
                cache.append((inputs[2 + i * 2], inputs[3 + i * 2]))
            }
            let (hidden, newCache) = selfRef.forwardStep(
                embeddings: embeds, offset: offset, cache: cache, attentionMask: nil
            )
            var result: [MLXArray] = [hidden]
            for (k, v) in newCache { result.append(k); result.append(v) }
            return result
        }
    }

    /// Run one autoregressive step using the compiled graph when available,
    /// falling back to the eager path otherwise.
    public func executeStep(
        embeddings: MLXArray,
        offset: MLXArray,
        cache: [(MLXArray, MLXArray)]
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        guard let compiled = compiledStep else {
            return forwardStep(embeddings: embeddings, offset: offset, cache: cache)
        }
        var flat: [MLXArray] = [embeddings, offset]
        flat.reserveCapacity(2 + cache.count * 2)
        for (k, v) in cache { flat.append(k); flat.append(v) }
        let out = compiled(flat)
        var newCache: [(MLXArray, MLXArray)] = []
        newCache.reserveCapacity(cache.count)
        for i in 0..<cache.count {
            newCache.append((out[1 + i * 2], out[2 + i * 2]))
        }
        return (out[0], newCache)
    }
}

public class Qwen2ForCausalLM: Module {
    public let config: Qwen2Configuration
    public let model: Qwen2Model

    @ModuleInfo(key: "lm_head") public var lmHead: Linear?

    public init(_ config: Qwen2Configuration) {
        self.config = config
        self.model = Qwen2Model(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray, cache: [KVCacheSimple]? = nil) -> MLXArray {
        var out = model(inputIds, cache: cache)

        if let lmHead = lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }

        return out
    }

    public func newCache() -> [KVCacheSimple] {
        model.newCache()
    }
}

extension Embedding {
    func asLinear(_ x: MLXArray) -> MLXArray {
        matmul(x, weight.T)
    }
}
