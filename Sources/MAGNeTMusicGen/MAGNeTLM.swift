import Foundation
import MLX
import MLXFast
import MLXNN
import MLXCommon

// MARK: - Sinusoidal position embedding

/// `[1, seqLen, dim]` Transformer sin embedding (concat[cos|sin]).
public func magnetSinEmbedding(seqLen: Int, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
    precondition(dim % 2 == 0)
    let half = dim / 2
    let pos = MLXArray(0..<Int32(seqLen)).asType(.float32).reshaped([seqLen, 1])
    let adim = MLXArray(0..<Int32(half)).asType(.float32).reshaped([1, half])
    let phase = pos / pow(MLXArray(maxPeriod), adim / Float(half - 1))
    let emb = concatenated([cos(phase), sin(phase)], axis: -1)
    return emb.expandedDimensions(axis: 0)
}

// MARK: - MAGNeT multi-head attention (quantized linears)

public final class MAGNeTMultiHeadAttention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: QuantizedLinear
    @ModuleInfo(key: "k_proj") public var kProj: QuantizedLinear
    @ModuleInfo(key: "v_proj") public var vProj: QuantizedLinear
    @ModuleInfo(key: "out_proj") public var outProj: QuantizedLinear
    public let numHeads: Int
    public let headDim: Int
    public let scale: Float

    public init(dim: Int, numHeads: Int, groupSize: Int, bits: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = Float(1.0 / sqrt(Double(self.headDim)))
        self._qProj = ModuleInfo(wrappedValue: QuantizedLinear(
            dim, dim, bias: false, groupSize: groupSize, bits: bits))
        self._kProj = ModuleInfo(wrappedValue: QuantizedLinear(
            dim, dim, bias: false, groupSize: groupSize, bits: bits))
        self._vProj = ModuleInfo(wrappedValue: QuantizedLinear(
            dim, dim, bias: false, groupSize: groupSize, bits: bits))
        self._outProj = ModuleInfo(wrappedValue: QuantizedLinear(
            dim, dim, bias: false, groupSize: groupSize, bits: bits))
        super.init()
    }

    public func callAsFunction(
        q qIn: MLXArray, k kIn: MLXArray, v vIn: MLXArray,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let q = qProj(qIn)
        let k = kProj(kIn)
        let v = vProj(vIn)
        let out = SDPA.multiHead(
            q: q, k: k, v: v,
            numHeads: numHeads, headDim: headDim, scale: scale,
            mask: mask)
        return outProj(out)
    }
}

// MARK: - MAGNeT transformer block

public final class MAGNeTTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: MAGNeTMultiHeadAttention
    @ModuleInfo(key: "cross_attn") public var crossAttn: MAGNeTMultiHeadAttention
    @ModuleInfo public var linear1: QuantizedLinear
    @ModuleInfo public var linear2: QuantizedLinear
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo(key: "norm_cross") public var normCross: LayerNorm
    @ModuleInfo public var norm2: LayerNorm

    public init(hiddenSize: Int, numHeads: Int, ffnDim: Int,
                groupSize: Int, bits: Int, eps: Float = 1e-5) {
        self._selfAttn = ModuleInfo(wrappedValue: MAGNeTMultiHeadAttention(
            dim: hiddenSize, numHeads: numHeads, groupSize: groupSize, bits: bits))
        self._crossAttn = ModuleInfo(wrappedValue: MAGNeTMultiHeadAttention(
            dim: hiddenSize, numHeads: numHeads, groupSize: groupSize, bits: bits))
        self._linear1 = ModuleInfo(wrappedValue: QuantizedLinear(
            hiddenSize, ffnDim, bias: false, groupSize: groupSize, bits: bits))
        self._linear2 = ModuleInfo(wrappedValue: QuantizedLinear(
            ffnDim, hiddenSize, bias: false, groupSize: groupSize, bits: bits))
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps))
        self._normCross = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps))
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize, eps: eps))
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, conditioning: MLXArray, selfMask: MLXArray? = nil
    ) -> MLXArray {
        let xn = norm1(x)
        let h1 = x + selfAttn(q: xn, k: xn, v: xn, mask: selfMask)
        let xc = normCross(h1)
        let h2 = h1 + crossAttn(q: xc, k: conditioning, v: conditioning, mask: nil)
        let xf = norm2(h2)
        let h3 = h2 + linear2(gelu(linear1(xf)))
        return h3
    }
}

// MARK: - MAGNeT LM

public final class MAGNeTLM: Module {
    @ModuleInfo public var emb: [Embedding]
    @ModuleInfo public var layers: [MAGNeTTransformerBlock]
    @ModuleInfo(key: "out_norm") public var outNorm: LayerNorm
    @ModuleInfo public var linears: [Linear]

    public let config: MAGNeTConfig
    private let seqLen: Int
    private let dim: Int
    private var stageMasks: [MLXArray?]

    public init(config: MAGNeTConfig) {
        self.config = config
        self.seqLen = config.seqLen
        self.dim = config.dim
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 8

        // Embeddings: card + 1 rows (vocab + mask token).
        self._emb = ModuleInfo(wrappedValue: (0..<config.nQ).map { _ in
            Embedding(embeddingCount: config.card + 1, dimensions: config.dim)
        })
        self._layers = ModuleInfo(wrappedValue: (0..<config.numLayers).map { _ in
            MAGNeTTransformerBlock(
                hiddenSize: config.dim, numHeads: config.numHeads,
                ffnDim: config.ffnDim, groupSize: groupSize, bits: bits)
        })
        self._outNorm = ModuleInfo(wrappedValue: LayerNorm(
            dimensions: config.dim))
        // Per-codebook output heads — FP (kept high precision).
        self._linears = ModuleInfo(wrappedValue: (0..<config.nQ).map { _ in
            Linear(config.dim, config.card, bias: false)
        })

        // Pre-build per-stage attention masks.
        var built: [MLXArray?] = []
        for s in 0..<config.nQ {
            built.append(Self.stageMask(stage: s, seqLen: config.seqLen,
                                         subcodesContext: config.subcodesContext))
        }
        self.stageMasks = built
        super.init()
    }

    // MARK: - Weight loading (per-leaf, with quantization)

    /// Apply weights from the MAGNeT bundle to this LM. The bundle uses keys
    /// like `lm.layers.<n>.self_attn.q_proj.{weight,scales,biases}`; the
    /// caller must strip the `lm.` prefix before passing.
    ///
    /// We load per-leaf-module rather than via `update(parameters:)` with a
    /// deeply-nested unflatten because mlx-swift's macro-generated
    /// @ModuleInfo(key: ...) wrappers don't reliably route nested keys
    /// through to inner `QuantizedLinear` submodules (silent miss → random
    /// projection weights, which is exactly what produced our noisy audio).
    public func loadWeights(_ weights: [String: MLXArray]) throws {
        // Per-codebook embeddings and output heads.
        for k in 0..<config.nQ {
            CommonWeightLoader.applyEmbeddingWeights(
                to: emb[k], prefix: "emb.\(k)", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: linears[k], prefix: "linears.\(k)", from: weights)
        }
        // Final output layer norm.
        CommonWeightLoader.applyLayerNormWeights(
            to: outNorm, prefix: "out_norm", from: weights)

        // Transformer blocks.
        for (i, layer) in layers.enumerated() {
            let p = "layers.\(i)"
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.norm1, prefix: "\(p).norm1", from: weights)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.norm2, prefix: "\(p).norm2", from: weights)
            CommonWeightLoader.applyLayerNormWeights(
                to: layer.normCross, prefix: "\(p).norm_cross", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.linear1, prefix: "\(p).linear1", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.linear2, prefix: "\(p).linear2", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.qProj, prefix: "\(p).self_attn.q_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.kProj, prefix: "\(p).self_attn.k_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.vProj, prefix: "\(p).self_attn.v_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.selfAttn.outProj, prefix: "\(p).self_attn.out_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.crossAttn.qProj, prefix: "\(p).cross_attn.q_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.crossAttn.kProj, prefix: "\(p).cross_attn.k_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.crossAttn.vProj, prefix: "\(p).cross_attn.v_proj", from: weights)
            CommonWeightLoader.applyQuantizedLinearWeights(
                to: layer.crossAttn.outProj, prefix: "\(p).cross_attn.out_proj", from: weights)
        }
    }

    private static func stageMask(stage: Int, seqLen: Int, subcodesContext: Int) -> MLXArray? {
        // Stage 0 (or unrestricted context) = full self-attention.
        if stage == 0 || subcodesContext < 0 { return nil }
        // Restricted local mask: |q - k| <= subcodesContext.
        let q = MLXArray(0..<Int32(seqLen)).reshaped([seqLen, 1])
        let k = MLXArray(0..<Int32(seqLen)).reshaped([1, seqLen])
        let valid = abs(q - k) .<= MLXArray(Int32(subcodesContext))
        let zero = MLXArray(Float(0.0))
        let negInf = MLXArray(-Float.infinity)
        let m = MLX.where(valid, zero, negInf).asType(.float32)
        return m.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    }

    /// `audioTokens: [B, T, K]` int, `conditioning: [B, L, D]`.
    /// Returns logits `[B, T, K, card]`.
    public func callAsFunction(
        _ audioTokens: MLXArray, conditioning: MLXArray, stage: Int
    ) -> MLXArray {
        // Sum per-codebook embeddings along K.
        var x: MLXArray = emb[0](audioTokens[0..., 0..., 0])
        for k in 1..<config.nQ {
            x = x + emb[k](audioTokens[0..., 0..., k])
        }
        let T = x.dim(1)
        let posEmb = magnetSinEmbedding(seqLen: T, dim: dim).asType(x.dtype)
        x = x + posEmb
        var mask = stageMasks[stage]
        if let m = mask, T != seqLen {
            // Slice down if runtime T smaller than pre-built mask (rare path).
            mask = m[0..., 0..., 0..<T, 0..<T]
        }
        for layer in layers {
            x = layer(x, conditioning: conditioning, selfMask: mask)
        }
        x = outNorm(x)
        // [B, T, K, card]
        var heads: [MLXArray] = []
        for k in 0..<config.nQ {
            heads.append(linears[k](x))
        }
        return stacked(heads, axis: -2)
    }
}
