import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

final class IndicMioQwen3Attention: Module {
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

    init(_ config: IndicMioModelConfig) {
        self.numQHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))

        let qDim = numQHeads * headDim
        let kvDim = numKVHeads * headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, qDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvDim, bias: false)
        self._oProj.wrappedValue = Linear(qDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: (MLXArray, MLXArray)?,
        offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = x.dim(0)
        let t = x.dim(1)

        var q = qNorm(qProj(x).reshaped(b, t, numQHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, t, numKVHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let ropeOffset = cache?.0.dim(2) ?? offset
        q = rope(q, offset: ropeOffset)
        k = rope(k, offset: ropeOffset)

        var cachedK = k
        var cachedV = v
        if let (previousK, previousV) = cache {
            cachedK = concatenated([previousK, k], axis: 2)
            cachedV = concatenated([previousV, v], axis: 2)
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if t <= 1 && (cache != nil || offset > 0) {
            maskMode = .none
        } else {
            let kvLen = cachedK.dim(2)
            let pastLen = kvLen - t
            let causal = MLXArray.tri(t, m: kvLen, k: pastLen, type: Float.self) - 1
            let additive = causal * Float.greatestFiniteMagnitude
            maskMode = .array(additive.reshaped(1, 1, t, kvLen).asType(q.dtype))
        }

        let attended = SDPA.attendAndMerge(
            qHeads: q,
            kHeads: cachedK,
            vHeads: cachedV,
            scale: scale,
            mask: maskMode)
        return (oProj(attended), (cachedK, cachedV))
    }
}

final class IndicMioQwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: IndicMioModelConfig) {
        self._gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class IndicMioQwen3Layer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: IndicMioQwen3Attention
    @ModuleInfo var mlp: IndicMioQwen3MLP

    init(_ config: IndicMioModelConfig) {
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)
        self._attention.wrappedValue = IndicMioQwen3Attention(config)
        self._mlp.wrappedValue = IndicMioQwen3MLP(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: (MLXArray, MLXArray)?,
        offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (attn, newCache) = attention(inputLayerNorm(x), cache: cache, offset: offset)
        let h = x + attn
        return (h + mlp(postAttentionLayerNorm(h)), newCache)
    }
}

final class IndicMioQwen3Model: Module {
    let config: IndicMioModelConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [IndicMioQwen3Layer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(config: IndicMioModelConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            IndicMioQwen3Layer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._lmHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    struct InferenceState {
        var kvCaches: [(MLXArray, MLXArray)?]
        var position: Int

        static func initial(config: IndicMioModelConfig) -> InferenceState {
            InferenceState(
                kvCaches: Array(repeating: nil, count: config.numHiddenLayers),
                position: 0)
        }
    }

    func forward(inputIds: MLXArray, state: InferenceState) -> (MLXArray, InferenceState) {
        let t = inputIds.dim(1)
        var hidden = embedTokens(inputIds)
        var caches = state.kvCaches

        for (index, layer) in layers.enumerated() {
            let (next, cache) = layer(hidden, cache: state.kvCaches[index], offset: state.position)
            hidden = next
            caches[index] = cache
        }

        hidden = norm(hidden)
        let logits = lmHead?(hidden) ?? embedTokens.asLinear(hidden)
        return (
            logits,
            InferenceState(kvCaches: caches, position: state.position + t)
        )
    }
}

enum IndicMioQwen3WeightLoader {
    static func loadWeights(
        into model: IndicMioQwen3Model,
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws {
        progressHandler?(0.05, "Loading Indic-Mio weights")
        let raw = try CommonWeightLoader.loadAllSafetensors(from: directory)
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(raw.count)
        for (key, value) in raw {
            if key.hasPrefix("model.") {
                weights[String(key.dropFirst("model.".count))] = value
            } else {
                weights[key] = value
            }
        }

        CommonWeightLoader.applyEmbeddingWeights(
            to: model.embedTokens,
            prefix: "embed_tokens",
            from: weights)
        CommonWeightLoader.applyRMSNormWeights(to: model.norm, prefix: "norm", from: weights)
        if let lmHead = model.lmHead {
            CommonWeightLoader.applyLinearWeights(to: lmHead, prefix: "lm_head", from: weights)
        }

        for index in 0..<model.config.numHiddenLayers {
            let prefix = "layers.\(index)"
            let layer = model.layers[index]
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.inputLayerNorm,
                prefix: "\(prefix).input_layernorm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.postAttentionLayerNorm,
                prefix: "\(prefix).post_attention_layernorm",
                from: weights)

            let attention = layer.attention
            CommonWeightLoader.applyLinearWeights(
                to: attention.qProj,
                prefix: "\(prefix).self_attn.q_proj",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attention.kProj,
                prefix: "\(prefix).self_attn.k_proj",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attention.vProj,
                prefix: "\(prefix).self_attn.v_proj",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: attention.oProj,
                prefix: "\(prefix).self_attn.o_proj",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attention.qNorm,
                prefix: "\(prefix).self_attn.q_norm",
                from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: attention.kNorm,
                prefix: "\(prefix).self_attn.k_norm",
                from: weights)

            let mlp = layer.mlp
            CommonWeightLoader.applyLinearWeights(
                to: mlp.gateProj,
                prefix: "\(prefix).mlp.gate_proj",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: mlp.upProj,
                prefix: "\(prefix).mlp.up_proj",
                from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: mlp.downProj,
                prefix: "\(prefix).mlp.down_proj",
                from: weights)
            progressHandler?(0.1 + 0.85 * Double(index + 1) / Double(model.config.numHiddenLayers),
                             "Loaded layer \(index + 1)/\(model.config.numHiddenLayers)")
        }

        eval(model)
        progressHandler?(1.0, "Indic-Mio weights ready")
    }
}
