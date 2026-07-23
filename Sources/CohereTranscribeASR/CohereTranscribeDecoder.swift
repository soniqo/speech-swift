// Portions adapted from mlx-audio-swift (MIT License).
// Copyright (c) 2025 Blaizzy and contributors.

import Foundation
import MLX
import MLXNN
import MLXFast

struct CohereTranscribeDecoderLayerKVCache {
    var selfKeys: MLXArray?
    var selfValues: MLXArray?
    var crossKeys: MLXArray?
    var crossValues: MLXArray?
    var isCrossUpdated: Bool

    init(
        selfKeys: MLXArray? = nil,
        selfValues: MLXArray? = nil,
        crossKeys: MLXArray? = nil,
        crossValues: MLXArray? = nil,
        isCrossUpdated: Bool = false
    ) {
        self.selfKeys = selfKeys
        self.selfValues = selfValues
        self.crossKeys = crossKeys
        self.crossValues = crossValues
        self.isCrossUpdated = isCrossUpdated
    }
}
struct CohereTranscribeDecoderKVCache {
    var layers: [CohereTranscribeDecoderLayerKVCache]
    var sequenceLength: Int

    init(layerCount: Int, sequenceLength: Int = 0) {
        self.layers = Array(repeating: CohereTranscribeDecoderLayerKVCache(), count: layerCount)
        self.sequenceLength = sequenceLength
    }
}

final class FixedPositionalEncoding {
    let hiddenSize: Int
    let maxSequenceLength: Int
    let posEnc: MLXArray

    init(hiddenSize: Int, maxSequenceLength: Int = 512) {
        self.hiddenSize = hiddenSize
        self.maxSequenceLength = maxSequenceLength

        let position = MLXArray(0..<maxSequenceLength).asType(.float32).reshaped(maxSequenceLength, 1)
        let scale = -log(10000.0) / Float(hiddenSize)
        let evenIndex = MLXArray(stride(from: 0, to: hiddenSize, by: 2)).asType(.float32)
        let divTerm = MLX.exp(scale * evenIndex)

        let angles = position * divTerm.reshaped(1, -1)
        let sin = MLX.sin(angles)
        let cos = MLX.cos(angles)
        let interleaved = MLX.stacked([sin, cos], axis: -1).reshaped(maxSequenceLength, hiddenSize)
        self.posEnc = interleaved / Float(sqrt(Double(hiddenSize)))
    }

    func callAsFunction(_ positionIds: MLXArray) -> MLXArray {
        let flat = positionIds.reshaped(-1).asType(.int32)
        let gathered = MLX.take(posEnc, flat, axis: 0)
        return gathered.reshaped(positionIds.shape[0], positionIds.shape[1], hiddenSize)
    }
}

final class DecoderAttention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(hiddenSize: Int, numHeads: Int, layerIdx: Int) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.layerIdx = layerIdx
        self.headDim = hiddenSize / numHeads
        self.scale = pow(Float(headDim), -0.5)

        self._qkvProj.wrappedValue = Linear(hiddenSize, hiddenSize * 3)
        self._outProj.wrappedValue = Linear(hiddenSize, hiddenSize)
    }

    private func splitQKV(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let b = x.dim(0)
        let t = x.dim(1)
        let qkv = x.reshaped(b, t, 3, hiddenSize)
        let q = qkv[0..., 0..., 0, 0...]
        let k = qkv[0..., 0..., 1, 0...]
        let v = qkv[0..., 0..., 2, 0...]
        return (q, k, v)
    }

    private func reshapeForHeads(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        return x.reshaped(b, t, numHeads, headDim).transposed(0, 2, 1, 3)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        contextStates: MLXArray?,
        attentionMask: MLXArray?,
        cache: CohereTranscribeDecoderLayerKVCache?,
        isCrossAttention: Bool
    ) -> (MLXArray, CohereTranscribeDecoderLayerKVCache) {
        var nextCache = cache ?? CohereTranscribeDecoderLayerKVCache()

        let queryProjection = qkvProj(hiddenStates)
        let (qHidden, _, _) = splitQKV(queryProjection)
        let queries = reshapeForHeads(qHidden)

        var keys: MLXArray
        var values: MLXArray

        if isCrossAttention {
            if nextCache.isCrossUpdated, let cachedK = nextCache.crossKeys, let cachedV = nextCache.crossValues {
                keys = cachedK
                values = cachedV
            } else {
                let source = contextStates ?? hiddenStates
                let sourceProjection = qkvProj(source)
                let (_, kSource, vSource) = splitQKV(sourceProjection)
                keys = reshapeForHeads(kSource)
                values = reshapeForHeads(vSource)
                nextCache.crossKeys = keys
                nextCache.crossValues = values
                nextCache.isCrossUpdated = true
            }
        } else {
            let source = contextStates ?? hiddenStates
            let sourceProjection = qkvProj(source)
            let (_, kSource, vSource) = splitQKV(sourceProjection)
            let newK = reshapeForHeads(kSource)
            let newV = reshapeForHeads(vSource)

            if let cachedK = nextCache.selfKeys, let cachedV = nextCache.selfValues {
                keys = MLX.concatenated([cachedK, newK], axis: 2)
                values = MLX.concatenated([cachedV, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }

            nextCache.selfKeys = keys
            nextCache.selfValues = values
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = attentionMask != nil ? .array(attentionMask!) : .none
        let attn = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: maskMode
        )

        let b = hiddenStates.dim(0)
        let t = hiddenStates.dim(1)
        let merged = attn.transposed(0, 2, 1, 3).reshaped(b, t, hiddenSize)
        return (outProj(merged), nextCache)
    }
}

final class DecoderFeedForward: Module {
    let hiddenAct: String

    @ModuleInfo(key: "dense_in") var denseIn: Linear
    @ModuleInfo(key: "dense_out") var denseOut: Linear

    init(hiddenSize: Int, innerSize: Int, hiddenAct: String = "relu") {
        self.hiddenAct = hiddenAct.lowercased().replacingOccurrences(of: "swish", with: "silu")
        self._denseIn.wrappedValue = Linear(hiddenSize, innerSize)
        self._denseOut.wrappedValue = Linear(innerSize, hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = denseIn(x)
        let activated: MLXArray
        switch hiddenAct {
        case "relu":
            activated = relu(h)
        case "gelu":
            activated = gelu(h)
        case "silu":
            activated = silu(h)
        default:
            activated = relu(h)
        }
        return denseOut(activated)
    }
}

final class TransformerDecoderLayer: Module {
    @ModuleInfo(key: "layer_norm_1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "first_sub_layer") var firstSubLayer: DecoderAttention
    @ModuleInfo(key: "layer_norm_2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "second_sub_layer") var secondSubLayer: DecoderAttention
    @ModuleInfo(key: "layer_norm_3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "third_sub_layer") var thirdSubLayer: DecoderFeedForward

    init(
        hiddenSize: Int,
        innerSize: Int,
        numHeads: Int,
        layerIdx: Int,
        hiddenAct: String = "relu"
    ) {
        self._layerNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._firstSubLayer.wrappedValue = DecoderAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            layerIdx: layerIdx
        )

        self._layerNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._secondSubLayer.wrappedValue = DecoderAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            layerIdx: layerIdx
        )

        self._layerNorm3.wrappedValue = LayerNorm(dimensions: hiddenSize)
        self._thirdSubLayer.wrappedValue = DecoderFeedForward(
            hiddenSize: hiddenSize,
            innerSize: innerSize,
            hiddenAct: hiddenAct
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray?,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereTranscribeDecoderLayerKVCache?
    ) -> (MLXArray, CohereTranscribeDecoderLayerKVCache) {
        var layerCache = cache

        let selfInput = layerNorm1(hiddenStates)
        let selfOut = firstSubLayer(
            selfInput,
            contextStates: nil,
            attentionMask: selfAttentionMask,
            cache: layerCache,
            isCrossAttention: false
        )
        var h = hiddenStates + selfOut.0
        layerCache = selfOut.1

        let crossInput = layerNorm2(h)
        let crossOut = secondSubLayer(
            crossInput,
            contextStates: encoderHiddenStates,
            attentionMask: crossAttentionMask,
            cache: layerCache,
            isCrossAttention: true
        )
        h = h + crossOut.0
        layerCache = crossOut.1

        let ffnInput = layerNorm3(h)
        h = h + thirdSubLayer(ffnInput)

        return (h, layerCache ?? CohereTranscribeDecoderLayerKVCache())
    }
}

final class TransformerDecoderEmbedding: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    let positionEmbedding: FixedPositionalEncoding

    init(vocabSize: Int, hiddenSize: Int, maxSequenceLength: Int, paddingIdx _: Int = 2) {
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: hiddenSize)
        self.positionEmbedding = FixedPositionalEncoding(
            hiddenSize: hiddenSize,
            maxSequenceLength: maxSequenceLength
        )
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(_ inputIds: MLXArray, positions: MLXArray) -> MLXArray {
        layerNorm(tokenEmbedding(inputIds) + positionEmbedding(positions))
    }
}

final class TransformerDecoderCore: Module {
    @ModuleInfo(key: "layers") var layers: [TransformerDecoderLayer]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(
        hiddenSize: Int,
        innerSize: Int,
        numHeads: Int,
        numLayers: Int,
        hiddenAct: String = "relu"
    ) {
        self._layers.wrappedValue = (0..<numLayers).map { idx in
            TransformerDecoderLayer(
                hiddenSize: hiddenSize,
                innerSize: innerSize,
                numHeads: numHeads,
                layerIdx: idx,
                hiddenAct: hiddenAct
            )
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray?,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereTranscribeDecoderKVCache?
    ) -> (MLXArray, CohereTranscribeDecoderKVCache) {
        var h = hiddenStates
        var nextCache = cache ?? CohereTranscribeDecoderKVCache(layerCount: layers.count)

        for i in layers.indices {
            let layerOut = layers[i](
                h,
                encoderHiddenStates: encoderHiddenStates,
                selfAttentionMask: selfAttentionMask,
                crossAttentionMask: crossAttentionMask,
                cache: nextCache.layers[i]
            )
            h = layerOut.0
            nextCache.layers[i] = layerOut.1
        }

        if let firstLayerSelfK = nextCache.layers.first?.selfKeys {
            nextCache.sequenceLength = firstLayerSelfK.dim(2)
        }

        return (finalLayerNorm(h), nextCache)
    }
}

final class TransformerDecoderWrapper: Module {
    @ModuleInfo(key: "embedding") var embedding: TransformerDecoderEmbedding
    @ModuleInfo(key: "core") var core: TransformerDecoderCore

    init(config: CohereTranscribeConfig) {
        let decConfig = config.decoder

        self._embedding.wrappedValue = TransformerDecoderEmbedding(
            vocabSize: config.vocabSize,
            hiddenSize: decConfig.hiddenSize,
            maxSequenceLength: decConfig.maxSequenceLength,
            paddingIdx: 2
        )
        self._core.wrappedValue = TransformerDecoderCore(
            hiddenSize: decConfig.hiddenSize,
            innerSize: decConfig.innerSize,
            numHeads: decConfig.numAttentionHeads,
            numLayers: decConfig.numLayers,
            hiddenAct: "relu"
        )
    }

    func makeCache() -> CohereTranscribeDecoderKVCache {
        CohereTranscribeDecoderKVCache(layerCount: core.layers.count)
    }

    func callAsFunction(
        inputIds: MLXArray,
        positions: MLXArray,
        encoderHiddenStates: MLXArray?,
        selfAttentionMask: MLXArray?,
        crossAttentionMask: MLXArray?,
        cache: CohereTranscribeDecoderKVCache? = nil
    ) -> (MLXArray, CohereTranscribeDecoderKVCache) {
        let hiddenStates = embedding(inputIds, positions: positions)
        return core(
            hiddenStates,
            encoderHiddenStates: encoderHiddenStates,
            selfAttentionMask: selfAttentionMask,
            crossAttentionMask: crossAttentionMask,
            cache: cache
        )
    }
}
