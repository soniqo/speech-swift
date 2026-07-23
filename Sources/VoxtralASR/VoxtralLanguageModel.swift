import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

struct VoxtralLanguageModelState {
    var caches: [(MLXArray, MLXArray)?]
    var position: Int

    static func initial(layerCount: Int) -> VoxtralLanguageModelState {
        VoxtralLanguageModelState(
            caches: Array(repeating: nil, count: layerCount),
            position: 0)
    }
}

final class VoxtralLlamaAttention: Module {
    let queryHeads: Int
    let keyValueHeads: Int
    let headDimension: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    init(_ config: VoxtralTextConfig) {
        queryHeads = config.numAttentionHeads
        keyValueHeads = config.numKeyValueHeads
        headDimension = config.headDim
        scale = 1 / sqrt(Float(config.headDim))
        _queryProjection.wrappedValue = Linear(
            config.hiddenSize, queryHeads * headDimension, bias: false)
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize, keyValueHeads * headDimension, bias: false)
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize, keyValueHeads * headDimension, bias: false)
        _outputProjection.wrappedValue = Linear(
            queryHeads * headDimension, config.hiddenSize, bias: false)
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: (MLXArray, MLXArray)?,
        position: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let batch = input.dim(0)
        let length = input.dim(1)
        var query = queryProjection(input)
            .reshaped(batch, length, queryHeads, headDimension)
            .transposed(0, 2, 1, 3)
        var key = keyProjection(input)
            .reshaped(batch, length, keyValueHeads, headDimension)
            .transposed(0, 2, 1, 3)
        let value = valueProjection(input)
            .reshaped(batch, length, keyValueHeads, headDimension)
            .transposed(0, 2, 1, 3)
        let offset = cache?.0.dim(2) ?? position
        query = rope(query, offset: offset)
        key = rope(key, offset: offset)
        let cachedKey = cache.map { MLX.concatenated([$0.0, key], axis: 2) } ?? key
        let cachedValue = cache.map { MLX.concatenated([$0.1, value], axis: 2) } ?? value

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if length == 1 && (cache != nil || position > 0) {
            mask = .none
        } else {
            let keyLength = cachedKey.dim(2)
            let previousLength = keyLength - length
            let causal = MLXArray.tri(
                length, m: keyLength, k: previousLength, type: Float.self) - 1
            mask = .array(
                (causal * Float.greatestFiniteMagnitude)
                    .reshaped(1, 1, length, keyLength)
                    .asType(query.dtype))
        }
        let attended = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: cachedKey,
            vHeads: cachedValue,
            scale: scale,
            mask: mask)
        return (outputProjection(attended), (cachedKey, cachedValue))
    }
}

final class VoxtralLlamaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ config: VoxtralTextConfig) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        _upProjection.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        _downProjection.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

final class VoxtralLlamaLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: VoxtralLlamaAttention
    @ModuleInfo var mlp: VoxtralLlamaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: VoxtralTextConfig) {
        _attention.wrappedValue = VoxtralLlamaAttention(config)
        _mlp.wrappedValue = VoxtralLlamaMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: (MLXArray, MLXArray)?,
        position: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let attentionResult = attention(
            inputLayerNorm(input), cache: cache, position: position)
        let hidden = input + attentionResult.0
        return (
            hidden + mlp(postAttentionLayerNorm(hidden)),
            attentionResult.1)
    }
}

final class VoxtralLlamaModel: Module {
    let config: VoxtralTextConfig

    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo var layers: [VoxtralLlamaLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: VoxtralTextConfig) {
        self.config = config
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        _layers.wrappedValue = (0..<config.numHiddenLayers).map { _ in
            VoxtralLlamaLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func embed(_ tokenIDs: MLXArray) -> MLXArray { tokenEmbedding(tokenIDs) }

    func forward(
        embeddings: MLXArray,
        state: VoxtralLanguageModelState
    ) -> (MLXArray, VoxtralLanguageModelState) {
        var hidden = embeddings
        var caches = state.caches
        for index in layers.indices {
            let result = layers[index](
                hidden,
                cache: state.caches[index],
                position: state.position)
            hidden = result.0
            caches[index] = result.1
        }
        return (
            norm(hidden),
            VoxtralLanguageModelState(
                caches: caches,
                position: state.position + embeddings.dim(1)))
    }
}

final class VoxtralLanguageModel: Module {
    let config: VoxtralTextConfig
    @ModuleInfo var model: VoxtralLlamaModel
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear?

    init(_ config: VoxtralTextConfig) {
        self.config = config
        _model.wrappedValue = VoxtralLlamaModel(config)
        _languageModelHead.wrappedValue = config.tieWordEmbeddings
            ? nil
            : Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        languageModelHead?(hidden) ?? model.tokenEmbedding.asLinear(hidden)
    }
}
