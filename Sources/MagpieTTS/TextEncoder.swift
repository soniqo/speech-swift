import Foundation
import MLX
import MLXNN

/// Bundle 1: 6-layer causal Transformer encoder over phoneme/byte token IDs.
/// Output: (B, T, d_model) memory consumed by the decoder via cross-attention.
public final class MagpieTextEncoder: Module {
    @ModuleInfo(key: "text_embedding")      public var textEmbedding: Embedding
    @ModuleInfo(key: "position_embeddings") public var positionEmbeddings: Embedding
    @ModuleInfo public var layers: [MagpieTransformerLayer]
    @ModuleInfo(key: "norm_out") public var normOut: LayerNorm

    public let config: MagpieTextEncoderConfig

    public init(config: MagpieTextEncoderConfig) {
        self.config = config
        self._textEmbedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.dModel),
            key: "text_embedding")
        self._positionEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.maxLen, dimensions: config.dModel),
            key: "position_embeddings")
        self._layers = ModuleInfo(wrappedValue: (0..<config.nLayers).map { _ in
            MagpieTransformerLayer(
                dModel: config.dModel, dFfn: config.dFfn, nHeads: config.nHeads,
                kernelSize: config.kernelSize,
                hasXattn: false, isCausal: true)
        })
        self._normOut = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.dModel, eps: 1e-5, affine: true, bias: false),
            key: "norm_out")
        super.init()
    }

    /// `tokens: (B, T) int32`, optional `mask: (B, T) {0,1}` float32.
    /// Returns memory `(B, T, d_model)`.
    public func callAsFunction(_ tokens: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let T = tokens.dim(1)
        let posIds = MLXArray(0..<Int32(T)).reshaped([1, T])
        var x = textEmbedding(tokens) + positionEmbeddings(posIds)
        let effMask = mask ?? MLXArray.ones(tokens.shape, dtype: .float32)
        let cache = magpieEmptyCache(layers: config.nLayers)
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: effMask, memory: nil, memoryMask: nil, cache: cache[i])
        }
        return normOut(x)
    }
}
