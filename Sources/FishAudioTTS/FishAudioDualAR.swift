import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

public struct FishAudioKVState {
    var caches: [(MLXArray, MLXArray)?]
    var position: Int

    static func initial(layerCount: Int) -> FishAudioKVState {
        FishAudioKVState(caches: Array(repeating: nil, count: layerCount), position: 0)
    }
}

public struct FishAudioForwardResult {
    public let logits: MLXArray
    public let hiddenStates: MLXArray
}

struct FishAudioBlockConfig {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let ropeTheta: Float
    let rmsNormEps: Float
    let attentionQKNorm: Bool
    let attentionQKVBias: Bool
    let attentionOutputBias: Bool

    init(_ config: FishAudioTransformerConfig) {
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize
        self.numHiddenLayers = config.numHiddenLayers
        self.numAttentionHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.ropeTheta = config.ropeTheta
        self.rmsNormEps = config.rmsNormEps
        self.attentionQKNorm = config.attentionQKNorm
        self.attentionQKVBias = config.attentionQKVBias
        self.attentionOutputBias = config.attentionOutputBias
    }

    init(_ config: FishAudioDecoderConfig) {
        self.hiddenSize = config.hiddenSize
        self.intermediateSize = config.intermediateSize
        self.numHiddenLayers = config.numHiddenLayers
        self.numAttentionHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.ropeTheta = config.ropeTheta
        self.rmsNormEps = config.rmsNormEps
        self.attentionQKNorm = config.attentionQKNorm
        self.attentionQKVBias = config.attentionQKVBias
        self.attentionOutputBias = config.attentionOutputBias
    }
}

final class FishAudioAttention: Module {
    let numQHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let useQKNorm: Bool

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: FishAudioBlockConfig) {
        self.numQHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(config.headDim))
        self.useQKNorm = config.attentionQKNorm

        let qDim = numQHeads * headDim
        let kvDim = numKVHeads * headDim
        self._wqkv.wrappedValue = Linear(
            config.hiddenSize,
            qDim + 2 * kvDim,
            bias: config.attentionQKVBias)
        self._wo.wrappedValue = Linear(
            qDim,
            config.hiddenSize,
            bias: config.attentionOutputBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self.rope = RoPE(dimensions: headDim, traditional: true, base: config.ropeTheta)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: (MLXArray, MLXArray)?,
        offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let b = x.dim(0)
        let t = x.dim(1)
        let qDim = numQHeads * headDim
        let kvDim = numKVHeads * headDim

        let parts = split(wqkv(x), indices: [qDim, qDim + kvDim], axis: -1)
        var q = parts[0].reshaped(b, t, numQHeads, headDim)
        var k = parts[1].reshaped(b, t, numKVHeads, headDim)
        let v = parts[2].reshaped(b, t, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if useQKNorm {
            q = qNorm(q)
            k = kNorm(k)
        }

        q = rope(q.transposed(0, 2, 1, 3), offset: cache?.0.dim(2) ?? offset)
        k = rope(k.transposed(0, 2, 1, 3), offset: cache?.0.dim(2) ?? offset)

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
        return (wo(attended), (cachedK, cachedV))
    }
}

final class FishAudioFeedForward: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(_ config: FishAudioBlockConfig) {
        self._w1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._w2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._w3.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class FishAudioLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo var attention: FishAudioAttention
    @ModuleInfo(key: "feed_forward") var feedForward: FishAudioFeedForward

    init(_ config: FishAudioBlockConfig) {
        self._attentionNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._attention.wrappedValue = FishAudioAttention(config)
        self._feedForward.wrappedValue = FishAudioFeedForward(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        cache: (MLXArray, MLXArray)?,
        offset: Int
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (attn, newCache) = attention(attentionNorm(x), cache: cache, offset: offset)
        let h = x + attn
        return (h + feedForward(ffnNorm(h)), newCache)
    }
}

public final class FishAudioSlowModel: Module {
    let config: FishAudioConfig

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    @ModuleInfo var layers: [FishAudioLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "output") var output: Linear?

    public init(config: FishAudioConfig) {
        self.config = config
        let blockConfig = FishAudioBlockConfig(config.text)
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.text.vocabSize,
            dimensions: config.text.hiddenSize)
        self._codebookEmbeddings.wrappedValue = Embedding(
            embeddingCount: FishAudioDefaults.codebookSize * config.audioDecoder.numCodebooks,
            dimensions: config.text.hiddenSize)
        self._layers.wrappedValue = (0..<config.text.numHiddenLayers).map { _ in
            FishAudioLayer(blockConfig)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.text.hiddenSize,
            eps: config.text.rmsNormEps)
        self._output.wrappedValue = config.text.tieWordEmbeddings
            ? nil
            : Linear(config.text.hiddenSize, config.text.vocabSize, bias: false)
        super.init()
    }

    func initialState() -> FishAudioKVState {
        FishAudioKVState.initial(layerCount: config.text.numHiddenLayers)
    }

    public func embed(_ inputIds: MLXArray) -> MLXArray {
        let mainTokens = inputIds[0..., 0, 0...]
        let textEmbeddings = embeddings(mainTokens)

        var codebookSum: MLXArray?
        for index in 0..<config.audioDecoder.numCodebooks {
            let rawCodes = inputIds[0..., index + 1, 0...]
            let codebookIds = rawCodes + MLXArray(Int32(index * FishAudioDefaults.codebookSize))
            let embedded = codebookEmbeddings(codebookIds)
            codebookSum = codebookSum.map { $0 + embedded } ?? embedded
        }

        let semanticMask = (
            (mainTokens .>= MLXArray(Int32(config.semanticStartTokenId)))
                .&& (mainTokens .<= MLXArray(Int32(config.semanticEndTokenId)))
        )
        .asType(textEmbeddings.dtype)
        .reshaped(mainTokens.dim(0), mainTokens.dim(1), 1)

        let combined = textEmbeddings + (codebookSum ?? MLXArray(Float(0))).asType(textEmbeddings.dtype) * semanticMask
        guard config.scaleCodebookEmbeddings else {
            return combined
        }
        let scale = MLXArray(Float(1.0 / sqrt(Float(config.audioDecoder.numCodebooks + 1))))
            .asType(combined.dtype)
        return MLX.where(semanticMask .> MLXArray(Float(0)), combined * scale, combined)
    }

    func forward(inputIds: MLXArray, state: FishAudioKVState) -> (FishAudioForwardResult, FishAudioKVState) {
        let t = inputIds.dim(2)
        var hidden = embed(inputIds)
        var caches = state.caches

        for (index, layer) in layers.enumerated() {
            let (next, cache) = layer(hidden, cache: state.caches[index], offset: state.position)
            hidden = next
            caches[index] = cache
        }

        let normalized = norm(hidden)
        let hiddenForFastDecoder = config.normFastLayerInput ? normalized : hidden
        let logits = output?(normalized) ?? embeddings.asLinear(normalized)
        let newState = FishAudioKVState(
            caches: caches,
            position: state.position + t)
        return (FishAudioForwardResult(logits: logits, hiddenStates: hiddenForFastDecoder), newState)
    }
}

public final class FishAudioFastDecoder: Module {
    let config: FishAudioConfig

    @ModuleInfo(key: "fast_embeddings") var embeddings: Embedding
    @ModuleInfo(key: "fast_layers") var layers: [FishAudioLayer]
    @ModuleInfo(key: "fast_norm") var norm: RMSNorm
    @ModuleInfo(key: "fast_output") var output: Linear

    public init(config: FishAudioConfig) {
        self.config = config
        let blockConfig = FishAudioBlockConfig(config.audioDecoder)
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.audioDecoder.vocabSize,
            dimensions: config.audioDecoder.hiddenSize)
        self._layers.wrappedValue = (0..<config.audioDecoder.numHiddenLayers).map { _ in
            FishAudioLayer(blockConfig)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.audioDecoder.hiddenSize,
            eps: config.audioDecoder.rmsNormEps)
        self._output.wrappedValue = Linear(
            config.audioDecoder.hiddenSize,
            config.audioDecoder.vocabSize,
            bias: false)
        super.init()
    }

    func initialState() -> FishAudioKVState {
        FishAudioKVState.initial(layerCount: config.audioDecoder.numHiddenLayers)
    }

    public func embedCodebooks(_ codebookIds: MLXArray) -> MLXArray {
        embeddings(codebookIds)
    }

    func forward(inputEmbeddings: MLXArray, state: FishAudioKVState) -> (MLXArray, FishAudioKVState) {
        let t = inputEmbeddings.dim(1)
        var hidden = inputEmbeddings
        var caches = state.caches

        for (index, layer) in layers.enumerated() {
            let (next, cache) = layer(hidden, cache: state.caches[index], offset: state.position)
            hidden = next
            caches[index] = cache
        }

        let logits = output(norm(hidden))
        return (
            logits,
            FishAudioKVState(caches: caches, position: state.position + t)
        )
    }
}

public final class FishAudioDualARModel: Module {
    public let config: FishAudioConfig

    @ModuleInfo var slow: FishAudioSlowModel
    @ModuleInfo var fast: FishAudioFastDecoder

    public init(config: FishAudioConfig) {
        self.config = config
        self._slow.wrappedValue = FishAudioSlowModel(config: config)
        self._fast.wrappedValue = FishAudioFastDecoder(config: config)
        super.init()
    }

    public static func load(
        from directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> FishAudioDualARModel {
        let config = try FishAudioConfig.load(from: directory.appendingPathComponent("config.json"))
        let model = FishAudioDualARModel(config: config)
        try FishAudioWeightLoader.loadWeights(
            into: model,
            from: directory,
            progressHandler: progressHandler)
        return model
    }

    public func initialSlowState() -> FishAudioKVState {
        slow.initialState()
    }

    public func initialFastState() -> FishAudioKVState {
        fast.initialState()
    }

    public func forwardSlow(
        inputIds: MLXArray,
        state: FishAudioKVState
    ) -> (FishAudioForwardResult, FishAudioKVState) {
        slow.forward(inputIds: inputIds, state: state)
    }

    public func forwardFast(
        inputEmbeddings: MLXArray,
        state: FishAudioKVState
    ) -> (MLXArray, FishAudioKVState) {
        fast.forward(inputEmbeddings: inputEmbeddings, state: state)
    }
}
