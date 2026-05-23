import Foundation
import MLX
import MLXNN

public let MagpieNumCodebooks = 8
public let MagpieVocabPerCodebook = 2024
public let MagpieAudioBosId: Int32 = 2016
public let MagpieAudioEosId: Int32 = 2017

/// Bundle 2/3: 12-layer causal Transformer decoder with cross-attention to the
/// text encoder memory. Exposes ``prefill(...)`` to seed the KV cache with
/// the 110-frame baked speaker context + the BOS audio frame, and ``step(...)``
/// to consume a single AR frame embedding.
public final class MagpieDecoder: Module {
    @ModuleInfo(key: "audio_embeddings")            public var audioEmbeddings: [Embedding]
    @ModuleInfo(key: "baked_context_embedding")     public var bakedContextEmbedding: Embedding
    @ParameterInfo(key: "baked_context_embedding_len") public var bakedContextEmbeddingLen: MLXArray
    @ModuleInfo(key: "position_embeddings")         public var positionEmbeddings: Embedding
    @ModuleInfo public var layers: [MagpieTransformerLayer]
    @ModuleInfo(key: "norm_out")                    public var normOut: LayerNorm
    @ModuleInfo(key: "final_proj")                  public var finalProj: Linear

    public let config: MagpieDecoderConfig

    public init(config: MagpieDecoderConfig) {
        self.config = config

        self._audioEmbeddings = ModuleInfo(
            wrappedValue: (0..<config.numCodebooks).map { _ in
                Embedding(embeddingCount: config.vocabPerCodebook,
                          dimensions: config.dModel)
            },
            key: "audio_embeddings")

        // Baked speaker context, flattened (5 × (T·D)). Reshape on lookup.
        self._bakedContextEmbedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.numBakedSpeakers,
                                     dimensions: config.bakedT * config.dModel),
            key: "baked_context_embedding")
        self._bakedContextEmbeddingLen = ParameterInfo(
            wrappedValue: MLXArray.zeros([config.numBakedSpeakers], dtype: .int32),
            key: "baked_context_embedding_len")

        self._positionEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.maxLen, dimensions: config.dModel),
            key: "position_embeddings")

        self._layers = ModuleInfo(wrappedValue:
            (0..<config.nLayers).map { _ in
                MagpieTransformerLayer(
                    dModel: config.dModel, dFfn: config.dFfn,
                    nHeads: config.nHeads, kernelSize: config.kernelSize,
                    hasXattn: true, isCausal: true,
                    xaDMemory: config.xaDMemory,
                    xaNHeads: config.xaNHeads,
                    xaDHead: config.xaDHead,
                    applyNormToCond: true)
            })

        self._normOut = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.dModel, eps: 1e-5, affine: true, bias: false),
            key: "norm_out")

        // final_proj keeps a bias (NeMo: ``bias=True``).
        self._finalProj = ModuleInfo(
            wrappedValue: Linear(config.dModel,
                                  config.numCodebooks * config.vocabPerCodebook,
                                  bias: true),
            key: "final_proj")

        super.init()
    }

    // MARK: - Helpers

    /// Average the per-codebook embeddings for one or many frames.
    /// `codes: (B, T, K) int32` → `(B, T, d_model)`.
    public func embedAudioFrame(_ codes: MLXArray) -> MLXArray {
        var out = audioEmbeddings[0](codes[.ellipsis, 0])
        for k in 1..<config.numCodebooks {
            out = out + audioEmbeddings[k](codes[.ellipsis, k])
        }
        return out / MLXArray(Float(config.numCodebooks))
    }

    /// Look up the flattened baked context for one speaker → `(1, T, D)`.
    public func bakedContext(speakerIdx: Int) -> MLXArray {
        let flat = bakedContextEmbedding.weight[speakerIdx]   // (T·D,)
        return flat.reshaped([1, config.bakedT, config.dModel])
    }

    private func runLayers(
        _ x: MLXArray, mask: MLXArray,
        memory: MLXArray, memoryMask: MLXArray,
        caches: [MagpieKVCache], positionOffset: Int
    ) -> MLXArray {
        let T = x.dim(1)
        let pos = MLXArray(Int32(positionOffset)..<Int32(positionOffset + T))
            .reshaped([1, T])
        var h = x + positionEmbeddings(pos)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, memory: memory, memoryMask: memoryMask,
                      cache: caches[i])
        }
        return normOut(h)
    }

    // MARK: - Prefill: baked speaker context + BOS frame

    /// Seed the KV cache with the 110-frame baked speaker context plus the BOS
    /// audio frame. Returns the last hidden state (1, 1, d_model) — the seed
    /// for the very first LocalTransformer cycle — alongside the populated
    /// per-layer caches.
    public func prefill(
        speakerIdx: Int,
        encoderOutput: MLXArray,
        encoderMask: MLXArray
    ) -> (hLast: MLXArray, caches: [MagpieKVCache]) {
        let baked = bakedContext(speakerIdx: speakerIdx)                 // (1, 110, d)
        let bosCodes = MLXArray.full([1, 1, config.numCodebooks],
                                      values: MLXArray(MagpieAudioBosId))
        let bosEmb = embedAudioFrame(bosCodes)                          // (1, 1, d)
        let x = MLX.concatenated([baked, bosEmb], axis: 1)              // (1, 111, d)
        let mask = MLXArray.ones([1, x.dim(1)], dtype: .float32)
        let caches = magpieEmptyCache(layers: config.nLayers)
        let h = runLayers(x, mask: mask, memory: encoderOutput,
                          memoryMask: encoderMask, caches: caches,
                          positionOffset: 0)
        return (h[0..., (h.dim(1) - 1)..<h.dim(1), 0...], caches)
    }

    // MARK: - One AR step

    /// Forward one new audio frame embedding through the decoder.
    /// Returns:
    ///   - logits (1, 1, K, V) — the parallel ``final_proj`` head's per-codebook
    ///     logits over the V=2024 vocab (used for EOS detection).
    ///   - hLast  (1, 1, d_model) — hidden state, feeds the LocalTransformer.
    ///   - caches updated in-place.
    public func step(
        audioEmb: MLXArray,
        encoderOutput: MLXArray,
        encoderMask: MLXArray,
        caches: [MagpieKVCache],
        position: Int
    ) -> (logits: MLXArray, hLast: MLXArray, caches: [MagpieKVCache]) {
        let mask = MLXArray.ones([1, 1], dtype: .float32)
        let h = runLayers(audioEmb, mask: mask, memory: encoderOutput,
                          memoryMask: encoderMask, caches: caches,
                          positionOffset: position)
        let logits = finalProj(h).reshaped([1, 1, config.numCodebooks,
                                             config.vocabPerCodebook])
        return (logits, h, caches)
    }
}
