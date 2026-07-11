import Foundation
import MLX
import MLXNN

/// 1-layer AR transformer (d_model=256) that refines the per-codebook
/// distribution given the decoder hidden as the position-0 priming token.
/// At codebook k we project the previously-sampled token embedding and feed
/// it as the input at position k, with the per-position out_projections[k]
/// producing logits over the V=2024 vocab.
public final class MagpieLocalTransformer: Module {
    @ModuleInfo(key: "in_projection")        public var inProjection: Linear
    @ModuleInfo(key: "position_embeddings")  public var positionEmbeddings: Embedding
    @ModuleInfo public var layers: [MagpieTransformerLayer]
    @ModuleInfo(key: "out_projections")       public var outProjections: [Linear]

    public let dModel: Int
    public let numCodebooks: Int

    public init(dModelDec: Int = 768, dModel: Int = 256,
                nHeads: Int = 1, numCodebooks: Int = MagpieNumCodebooks,
                vocabPerCodebook: Int = MagpieVocabPerCodebook,
                maxPos: Int = 10) {
        self.dModel = dModel
        self.numCodebooks = numCodebooks
        self._inProjection = ModuleInfo(
            wrappedValue: Linear(dModelDec, dModel, bias: true),
            key: "in_projection")
        self._positionEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: maxPos, dimensions: dModel),
            key: "position_embeddings")
        self._layers = ModuleInfo(wrappedValue: [
            MagpieTransformerLayer(
                dModel: dModel, dFfn: dModel * 4, nHeads: nHeads,
                kernelSize: 1, hasXattn: false, isCausal: true)
        ])
        self._outProjections = ModuleInfo(
            wrappedValue: (0..<numCodebooks).map { _ in
                Linear(dModel, vocabPerCodebook, bias: true)
            },
            key: "out_projections")
        super.init()
    }

    /// One step. `x: (1, 1, d_model)` (already projected), `position: 0..K-1`.
    public func callAsFunction(_ x: MLXArray, position: Int,
                                cache: MagpieKVCache) -> MLXArray {
        let posTok = MLXArray([Int32(position)], [1, 1])
        let h = x + positionEmbeddings(posTok)
        let mask = MLXArray.ones([1, 1], dtype: .float32)
        return layers[0](h, mask: mask, memory: nil, memoryMask: nil,
                          cache: cache)
    }
}
