import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN

final class MossMLXTextAttention: Module {
    let config: MossMLXConfiguration.Decoder
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm

    init(_ config: MossMLXConfiguration.Decoder) {
        self.config = config
        scale = 1 / sqrt(Float(config.headDimension))
        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * config.headDimension,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeads * config.headDimension,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.keyValueHeads * config.headDimension,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            config.attentionHeads * config.headDimension,
            config.hiddenSize,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(
            dimensions: config.headDimension,
            eps: config.rmsNormEpsilon
        )
        _keyNorm.wrappedValue = RMSNorm(
            dimensions: config.headDimension,
            eps: config.rmsNormEpsilon
        )
        rope = RoPE(
            dimensions: config.headDimension,
            traditional: false,
            base: config.ropeTheta
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        var query = queryNorm(
            queryProjection(input).reshaped(
                batch,
                length,
                config.attentionHeads,
                config.headDimension
            )
        ).transposed(0, 2, 1, 3)
        var key = keyNorm(
            keyProjection(input).reshaped(
                batch,
                length,
                config.keyValueHeads,
                config.headDimension
            )
        ).transposed(0, 2, 1, 3)
        let value = valueProjection(input).reshaped(
            batch,
            length,
            config.keyValueHeads,
            config.headDimension
        ).transposed(0, 2, 1, 3)

        query = rope(query, offset: cache.offset)
        key = rope(key, offset: cache.offset)
        let attended = attentionWithCacheUpdate(
            queries: query,
            keys: key,
            values: value,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(
            batch,
            length,
            config.attentionHeads * config.headDimension
        )
        return outputProjection(attended)
    }
}

final class MossMLXTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProjection: Linear
    @ModuleInfo(key: "up_proj") var upProjection: Linear
    @ModuleInfo(key: "down_proj") var downProjection: Linear

    init(_ config: MossMLXConfiguration.Decoder) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(
            silu(gateProjection(input)) * upProjection(input)
        )
    }
}

final class MossMLXTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MossMLXTextAttention
    @ModuleInfo var mlp: MossMLXTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var
        postAttentionLayerNorm: RMSNorm

    init(_ config: MossMLXConfiguration.Decoder) {
        _attention.wrappedValue = MossMLXTextAttention(config)
        _mlp.wrappedValue = MossMLXTextMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEpsilon
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEpsilon
        )
        super.init()
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache
    ) -> MLXArray {
        let hidden =
            input
            + attention(
                inputLayerNorm(input),
                mask: mask,
                cache: cache
            )
        return hidden + mlp(postAttentionLayerNorm(hidden))
    }
}

final class MossMLXTextModel: Module {
    let config: MossMLXConfiguration.Decoder

    @ModuleInfo(key: "embed_tokens") var tokenEmbedding: Embedding
    @ModuleInfo var layers: [MossMLXTextLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: MossMLXConfiguration.Decoder) {
        self.config = config
        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0..<config.layers).map { _ in
            MossMLXTextLayer(config)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEpsilon
        )
        super.init()
    }

    func embed(_ tokenIDs: MLXArray) -> MLXArray {
        tokenEmbedding(tokenIDs)
    }

    func forward(
        embeddings: MLXArray,
        cache: [KVCache]
    ) -> MLXArray {
        precondition(
            cache.count == layers.count,
            "MOSS requires one K/V cache per decoder layer"
        )
        let mask = createAttentionMask(
            h: embeddings,
            cache: cache.first
        )
        var hidden = embeddings
        for index in layers.indices {
            hidden = layers[index](
                hidden,
                mask: mask,
                cache: cache[index]
            )
        }
        return norm(hidden)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        tokenEmbedding.asLinear(hidden)
    }

    func makeCache(
        precision: MossMLXKVCachePrecision,
        step: Int
    ) -> [KVCache] {
        (0..<config.layers).map { _ in
            switch precision {
            case .float16:
                let cache = KVCacheSimple()
                cache.step = max(1, step)
                return cache
            case .int8:
                return QuantizedKVCache(groupSize: 64, bits: 8)
            }
        }
    }
}
