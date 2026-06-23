import Foundation
import MLX
import MLXFast
import MLXNN

/// T3 (text→speech-token) configuration. Multilingual variant.
public struct T3Config {
    public var textTokensDictSize = 2352
    public var startTextToken = 255
    public var stopTextToken = 0
    public var maxTextTokens = 2048
    public var speechTokensDictSize = 8194
    public var startSpeechToken = 6561
    public var stopSpeechToken = 6562
    public var maxSpeechTokens = 4096
    public var nChannels = 1024
    public var speakerEmbedSize = 256
    public var usePerceiverResampler = true
    public var emotionAdv = true

    // Llama_520M backbone.
    public var numLayers = 30
    public var hiddenSize = 1024
    public var numHeads = 16
    public var numKVHeads = 16
    public var headDim = 64
    public var intermediateSize = 4096
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 500_000.0
    // llama3 rope scaling
    public var ropeFactor: Float = 8.0
    public var ropeHighFreqFactor: Float = 4.0
    public var ropeLowFreqFactor: Float = 1.0
    public var ropeOrigMaxPos = 8192

    public init() {}
}

/// Learned position embeddings (GPT-2 style table), used for both text and speech.
final class LearnedPositionEmbeddings: Module {
    @ModuleInfo(key: "emb") var emb: Embedding

    init(seqLen: Int, dim: Int) {
        self._emb.wrappedValue = Embedding(embeddingCount: seqLen, dimensions: dim)
    }

    /// Position embeddings for indices `0 ..< x.dim(1)` → `[T, dim]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        emb(MLXArray(0 ..< x.dim(1)))
    }

    /// Embedding(s) at a fixed index → `[1, 1, dim]`.
    func fixed(_ idx: Int) -> MLXArray {
        emb(MLXArray([idx]).reshaped([1, 1]))
    }
}

/// Cross-attention block (LayerNorm + separate Q/K/V + proj_out, residual on the
/// query). Shared by the Perceiver for both its cross- and self-attention passes.
final class PerceiverAttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(channels: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = channels / numHeads
        self.scale = pow(Float(channels / numHeads), -0.5)
        self._norm.wrappedValue = LayerNorm(dimensions: channels)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._projOut.wrappedValue = Linear(channels, channels)
    }

    private func split(_ x: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        return x.reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
    }

    /// Cross-attention from `x1` (query) to `x2` (key/value), residual on `x1`.
    func callAsFunction(_ x1: MLXArray, _ x2: MLXArray) -> MLXArray {
        let q = split(toQ(norm(x1)))
        let k = split(toK(norm(x2)))
        let v = split(toV(norm(x2)))
        var h = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        let (b, t) = (h.dim(0), h.dim(2))
        h = h.transposed(0, 2, 1, 3).reshaped([b, t, numHeads * headDim])
        return x1 + projOut(h)
    }
}

/// Perceiver resampler: 32 learnable queries attend to the variable-length input,
/// then to themselves, producing a fixed `[B, 32, dim]` conditioning.
final class Perceiver: Module {
    @ParameterInfo(key: "pre_attention_query") var preAttentionQuery: MLXArray
    @ModuleInfo(key: "attn") var attn: PerceiverAttentionBlock

    init(queryTokens: Int = 32, dim: Int = 1024, numHeads: Int = 4) {
        self._preAttentionQuery.wrappedValue = MLXArray.zeros([1, queryTokens, dim])
        self._attn.wrappedValue = PerceiverAttentionBlock(channels: dim, numHeads: numHeads)
    }

    func callAsFunction(_ h: MLXArray) -> MLXArray {
        let b = h.dim(0)
        let query = broadcast(preAttentionQuery, to: [b, preAttentionQuery.dim(1), preAttentionQuery.dim(2)])
        let preAtt = attn(query, h)   // cross-attention
        return attn(preAtt, preAtt)   // self-attention
    }
}

/// T3 conditioning encoder: speaker projection + optional emotion control +
/// perceiver-resampled prompt-speech embedding → `[B, condLen, dim]`.
final class T3CondEnc: Module {
    @ModuleInfo(key: "spkr_enc") var spkrEnc: Linear
    @ModuleInfo(key: "emotion_adv_fc") var emotionAdvFC: Linear?
    @ModuleInfo(key: "perceiver") var perceiver: Perceiver?

    let speakerEmbedSize: Int

    init(_ cfg: T3Config) {
        self.speakerEmbedSize = cfg.speakerEmbedSize
        self._spkrEnc.wrappedValue = Linear(cfg.speakerEmbedSize, cfg.nChannels)
        self._emotionAdvFC.wrappedValue = cfg.emotionAdv ? Linear(1, cfg.nChannels, bias: false) : nil
        self._perceiver.wrappedValue = cfg.usePerceiverResampler ? Perceiver() : nil
    }

    /// - speakerEmb: `[B, speakerEmbedSize]`
    /// - promptSpeechEmb: optional `[B, T, dim]` (already embedded prompt-speech)
    /// - emotionAdv: scalar emotion exaggeration
    func callAsFunction(
        speakerEmb: MLXArray, promptSpeechEmb: MLXArray? = nil, emotionAdv: Float = 0.5
    ) -> MLXArray {
        let b = speakerEmb.dim(0)
        let condSpkr = spkrEnc(speakerEmb.reshaped([b, speakerEmbedSize])).expandedDimensions(axis: 1)

        var parts: [MLXArray] = [condSpkr]
        if let perceiver, let promptSpeechEmb {
            parts.append(perceiver(promptSpeechEmb))
        }
        if let emotionAdvFC {
            let e = MLXArray(emotionAdv).reshaped([1, 1, 1])
            parts.append(emotionAdvFC(broadcast(e, to: [b, 1, 1])))
        }
        return concatenated(parts, axis: 1)
    }
}
