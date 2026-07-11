import Foundation
import MLX
import MLXFast
import MLXNN
import MLXCommon

// MARK: - T5 helpers

/// T5 bidirectional relative-position bucketing (encoder self-attention).
///
/// Half the buckets go to the upper range (positive offsets), half to the
/// lower (negative). Within each half, the first `n/2` buckets cover exact
/// small distances, the rest are log-spaced up to `maxDistance`.
private func t5RelativeBucketsBidirectional(
    qLen: Int, kLen: Int, numBuckets: Int, maxDistance: Int
) -> MLXArray {
    var matrix = [Int32]()
    matrix.reserveCapacity(qLen * kLen)
    for q in 0..<qLen {
        for k in 0..<kLen {
            matrix.append(Int32(k - q))
        }
    }
    var rel = MLXArray(matrix, [qLen, kLen])
    var buckets = MLXArray.zeros([qLen, kLen], dtype: .int32)

    let n = numBuckets / 2
    let isPositive = (rel .> MLXArray(Int32(0))).asType(.int32) * MLXArray(Int32(n))
    buckets = buckets + isPositive
    rel = abs(rel)

    let maxExact = n / 2
    let isSmall = rel .< MLXArray(Int32(maxExact))
    let relF = rel.asType(.float32)
    let logRatio = log(relF / Float(maxExact)) / log(Float(maxDistance) / Float(maxExact))
    let large = MLXArray(Int32(maxExact)) + (logRatio * Float(n - maxExact)).asType(.int32)
    let largeClamped = minimum(large, MLXArray(Int32(n - 1)))
    return buckets + MLX.where(isSmall, rel, largeClamped)
}

// MARK: - T5 RelativePositionBias

public final class T5RelativeAttentionBias: Module {
    @ModuleInfo public var embeddings: Embedding
    public let numBuckets: Int
    public let maxDistance: Int
    public let numHeads: Int

    public init(numBuckets: Int, maxDistance: Int, numHeads: Int) {
        self._embeddings = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: numBuckets, dimensions: numHeads))
        self.numBuckets = numBuckets
        self.maxDistance = maxDistance
        self.numHeads = numHeads
        super.init()
    }

    /// Returns `[numHeads, queryLength, keyLength]`.
    public func callAsFunction(queryLength: Int, keyLength: Int) -> MLXArray {
        let bucketIds = t5RelativeBucketsBidirectional(
            qLen: queryLength, kLen: keyLength,
            numBuckets: numBuckets, maxDistance: maxDistance)
        let values = embeddings(bucketIds.flattened()).reshaped(queryLength, keyLength, numHeads)
        return values.transposed(2, 0, 1)
    }
}

// MARK: - T5 attention (self-attn only, encoder)

public final class T5SelfAttention: Module {
    @ModuleInfo(key: "query_proj") public var queryProj: Linear
    @ModuleInfo(key: "key_proj")   public var keyProj: Linear
    @ModuleInfo(key: "value_proj") public var valueProj: Linear
    @ModuleInfo(key: "out_proj")   public var outProj: Linear

    public let numHeads: Int
    public let headDim: Int

    public init(dModel: Int, dKv: Int, numHeads: Int) {
        let inner = dKv * numHeads
        self._queryProj = ModuleInfo(wrappedValue: Linear(dModel, inner, bias: false))
        self._keyProj   = ModuleInfo(wrappedValue: Linear(dModel, inner, bias: false))
        self._valueProj = ModuleInfo(wrappedValue: Linear(dModel, inner, bias: false))
        self._outProj   = ModuleInfo(wrappedValue: Linear(inner, dModel, bias: false))
        self.numHeads = numHeads
        self.headDim = dKv
        super.init()
    }

    /// `x: [B, T, dModel]`; `bias: [H, T, T]` (broadcast over batch). T5 uses
    /// unscaled scores (no 1/sqrt(d)) and softmax in fp32 — match the Python
    /// reference manually rather than using the fast SDPA kernel, which has
    /// produced slightly off values for additive biases of this shape in
    /// mlx-swift.
    public func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let q = queryProj(x)
        let k = keyProj(x)
        let v = valueProj(x)
        let B = q.dim(0); let L = q.dim(1); let S = k.dim(1)
        let H = numHeads
        let D = headDim
        // (B, L, H*D) → (B, H, L, D)
        let qH = q.reshaped([B, L, H, D]).transposed(0, 2, 1, 3)
        // (B, S, H*D) → (B, H, D, S)  — note last-two-axis swap for Q @ K
        let kH = k.reshaped([B, S, H, D]).transposed(0, 2, 3, 1)
        let vH = v.reshaped([B, S, H, D]).transposed(0, 2, 1, 3)

        // scores: (B, H, L, S)
        var scores = matmul(qH, kH)
        // bias: (H, L, S) → broadcasts to (B, H, L, S)
        let biasExpanded = bias.expandedDimensions(axis: 0).asType(scores.dtype)
        scores = scores + biasExpanded
        // softmax in fp32 then cast back to scores dtype (matches Python).
        let probs = softmax(scores.asType(.float32), axis: -1).asType(scores.dtype)
        // out: (B, H, L, D) → (B, L, H*D)
        let attn = matmul(probs, vH).transposed(0, 2, 1, 3).reshaped([B, L, H * D])
        return outProj(attn)
    }
}

// MARK: - T5 dense (vanilla, ReLU)

public final class T5Dense: Module {
    @ModuleInfo public var wi: Linear
    @ModuleInfo public var wo: Linear

    public init(dModel: Int, dFf: Int) {
        self._wi = ModuleInfo(wrappedValue: Linear(dModel, dFf, bias: false))
        self._wo = ModuleInfo(wrappedValue: Linear(dFf, dModel, bias: false))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return wo(relu(wi(x)))
    }
}

// MARK: - T5 encoder layer

public final class T5EncoderLayer: Module {
    @ModuleInfo public var attention: T5SelfAttention
    @ModuleInfo public var dense: T5Dense
    @ModuleInfo public var ln1: RMSNorm
    @ModuleInfo public var ln2: RMSNorm

    public init(config: T5ModelConfig) {
        self._attention = ModuleInfo(wrappedValue: T5SelfAttention(
            dModel: config.dModel, dKv: config.dKv, numHeads: config.numHeads))
        self._dense = ModuleInfo(wrappedValue: T5Dense(
            dModel: config.dModel, dFf: config.dFf))
        self._ln1 = ModuleInfo(wrappedValue: RMSNorm(
            dimensions: config.dModel, eps: config.layerNormEpsilon))
        self._ln2 = ModuleInfo(wrappedValue: RMSNorm(
            dimensions: config.dModel, eps: config.layerNormEpsilon))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, bias: MLXArray) -> MLXArray {
        let y = attention(ln1(x), bias: bias)
        let h = x + y
        let z = dense(ln2(h))
        return h + z
    }
}

// MARK: - T5 encoder

public final class T5Encoder: Module {
    @ModuleInfo public var wte: Embedding
    @ModuleInfo public var layers: [T5EncoderLayer]
    @ModuleInfo public var ln: RMSNorm
    @ModuleInfo(key: "relative_attention_bias") public var relativeAttentionBias: T5RelativeAttentionBias

    public let config: T5ModelConfig

    public init(config: T5ModelConfig) {
        self.config = config
        self._wte = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: config.vocabSize, dimensions: config.dModel))
        self._layers = ModuleInfo(wrappedValue: (0..<config.numLayers).map { _ in
            T5EncoderLayer(config: config)
        })
        self._ln = ModuleInfo(wrappedValue: RMSNorm(
            dimensions: config.dModel, eps: config.layerNormEpsilon))
        self._relativeAttentionBias = ModuleInfo(
            wrappedValue: T5RelativeAttentionBias(
                numBuckets: config.relativeAttentionNumBuckets,
                maxDistance: config.relativeAttentionMaxDistance,
                numHeads: config.numHeads),
            key: "relative_attention_bias")
        super.init()
    }

    /// `inputIds: [B, T]` int → `[B, T, dModel]`.
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var x = wte(inputIds)
        let T = x.dim(1)
        let bias = relativeAttentionBias(queryLength: T, keyLength: T)
        for layer in layers {
            x = layer(x, bias: bias)
        }
        return ln(x)
    }

    /// Apply sanitized weights to this encoder module-by-module via
    /// `CommonWeightLoader`. We bypass `module.update(parameters:)` with a
    /// deep unflatten because mlx-swift's @ModuleInfo(key:) wrappers silently
    /// fail to route nested keys to inner submodules (the bug that VoxCPM2 /
    /// MADLAD also hit — produces random projection weights with no error).
    public func loadSanitizedWeights(_ weights: [String: MLXArray]) throws {
        CommonWeightLoader.applyEmbeddingWeights(
            to: wte, prefix: "wte", from: weights)
        CommonWeightLoader.applyRMSNormWeights(
            to: ln, prefix: "ln", from: weights)
        CommonWeightLoader.applyEmbeddingWeights(
            to: relativeAttentionBias.embeddings,
            prefix: "relative_attention_bias.embeddings", from: weights)
        for (i, layer) in layers.enumerated() {
            let p = "layers.\(i)"
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.ln1, prefix: "\(p).ln1", from: weights)
            CommonWeightLoader.applyRMSNormWeights(
                to: layer.ln2, prefix: "\(p).ln2", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.queryProj,
                prefix: "\(p).attention.query_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.keyProj,
                prefix: "\(p).attention.key_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.valueProj,
                prefix: "\(p).attention.value_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.attention.outProj,
                prefix: "\(p).attention.out_proj", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.dense.wi, prefix: "\(p).dense.wi", from: weights)
            CommonWeightLoader.applyLinearWeights(
                to: layer.dense.wo, prefix: "\(p).dense.wo", from: weights)
        }
    }

    /// Convert HuggingFace `t5-base` safetensors keys to our module layout.
    /// Mirrors the Python `T5.sanitize` from the reference, encoder-only.
    ///
    /// Order matters: do `.block.→.layers.` and the SelfAttention/DenseReluDense
    /// renames BEFORE stripping the `encoder.` prefix, otherwise the leading-
    /// dot anchors in `.block.` / `.layer.0.SelfAttention.` fail to match.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let renames: [(String, String)] = [
            (".block.", ".layers."),
            (".k.", ".key_proj."),
            (".o.", ".out_proj."),
            (".q.", ".query_proj."),
            (".v.", ".value_proj."),
            (".layer.0.layer_norm.", ".ln1."),
            (".layer.1.layer_norm.", ".ln2."),
            (".final_layer_norm.", ".ln."),
            (".layer.0.SelfAttention.", ".attention."),
            (".layer.1.DenseReluDense.", ".dense."),
            // After `.block.→.layers.` this match works at the head.
            (
                "encoder.layers.0.attention.relative_attention_bias.",
                "encoder.relative_attention_bias.embeddings."
            ),
        ]
        var out: [String: MLXArray] = [:]
        for (k, v) in weights {
            // Only encoder + shared embedding; skip decoder & lm_head entirely.
            guard k.hasPrefix("encoder.") || k.hasPrefix("shared.") else { continue }
            var key = k
            for (old, new) in renames { key = key.replacingOccurrences(of: old, with: new) }
            // Strip the "encoder." prefix once all replacements are done — our
            // module IS the encoder, so there is no nested namespace.
            if key.hasPrefix("encoder.") { key = String(key.dropFirst("encoder.".count)) }
            if key.hasPrefix("shared.") {
                key = "wte." + String(key.dropFirst("shared.".count))
            }
            out[key] = v
        }
        return out
    }
}
