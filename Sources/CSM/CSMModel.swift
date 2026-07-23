import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import PersonaPlex   // reuse RMSNormF32, KVCacheSimple, makeLinear/applyLinear

// MARK: - Config
//
// Sesame CSM-1B (Conversational Speech Model, Apache-2.0). Moshi-family codec-token
// model: a Llama-1B backbone predicts Mimi codebook 0, a small Llama-100M decoder
// predicts codebooks 1..31. Reuses PersonaPlex's Mimi codec and primitives.
//
// Loads OUR exported format (aufklarer/CSM-1B-MLX-*): a single model.safetensors
// with `model.` HF-style keys, plus our own config.json.

public struct CSMTowerConfig {
    public let dim: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int      // GQA
    public let headDim: Int
    public let intermediateDim: Int
    public let ropeBase: Float
    public let normEps: Float
}

public struct CSMConfig {
    public let backbone: CSMTowerConfig
    public let decoder: CSMTowerConfig
    public let textVocabSize: Int
    public let audioVocabSize: Int
    public let audioNumCodebooks: Int
    public let quantBits: Int?      // nil = fp16 (no quantization)
    public let quantGroupSize: Int

    public static func load(from directory: URL) throws -> CSMConfig {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        func tower(_ d: [String: Any]) -> CSMTowerConfig {
            CSMTowerConfig(
                dim: d["dim"] as! Int, numLayers: d["num_layers"] as! Int,
                numHeads: d["num_heads"] as! Int, numKVHeads: d["num_kv_heads"] as! Int,
                headDim: d["head_dim"] as! Int, intermediateDim: d["intermediate_dim"] as! Int,
                ropeBase: Float((d["rope_base"] as? Double) ?? 500000.0),
                normEps: Float((d["norm_eps"] as? Double) ?? 1e-5))
        }
        let quant = json["quantization"] as? [String: Any]
        return CSMConfig(
            backbone: tower(json["backbone"] as! [String: Any]),
            decoder: tower(json["decoder"] as! [String: Any]),
            textVocabSize: json["text_vocab_size"] as! Int,
            audioVocabSize: json["audio_vocab_size"] as! Int,
            audioNumCodebooks: json["audio_num_codebooks"] as! Int,
            quantBits: quant?["bits"] as? Int,
            quantGroupSize: (quant?["group_size"] as? Int) ?? 64)
    }
}

// MARK: - Attention (separate Q/K/V/O projections, grouped-query)

public final class CSMAttention: Module {
    @ModuleInfo(key: "q_proj") public var qProj: Module
    @ModuleInfo(key: "k_proj") public var kProj: Module
    @ModuleInfo(key: "v_proj") public var vProj: Module
    @ModuleInfo(key: "o_proj") public var oProj: Module
    @ModuleInfo public var rope: RoPE

    private let numHeads: Int
    private let numKVHeads: Int
    private let headDim: Int
    private let scale: Float

    public init(_ c: CSMTowerConfig, bits: Int?, groupSize: Int) {
        numHeads = c.numHeads; numKVHeads = c.numKVHeads; headDim = c.headDim
        scale = 1.0 / Float(Double(c.headDim).squareRoot())
        _qProj = ModuleInfo(wrappedValue: makeLinear(c.dim, c.numHeads * c.headDim, bias: false, groupSize: groupSize, bits: bits))
        _kProj = ModuleInfo(wrappedValue: makeLinear(c.dim, c.numKVHeads * c.headDim, bias: false, groupSize: groupSize, bits: bits))
        _vProj = ModuleInfo(wrappedValue: makeLinear(c.dim, c.numKVHeads * c.headDim, bias: false, groupSize: groupSize, bits: bits))
        _oProj = ModuleInfo(wrappedValue: makeLinear(c.numHeads * c.headDim, c.dim, bias: false, groupSize: groupSize, bits: bits))
        _rope = ModuleInfo(wrappedValue: RoPE(dimensions: c.headDim, traditional: true, base: c.ropeBase))
    }

    public func callAsFunction(_ xs: MLXArray, cache: any KVCache, offset: Int) -> MLXArray {
        let b = xs.shape[0], t = xs.shape[1]
        var q = applyLinear(qProj, xs).reshaped([b, t, numHeads, headDim]).swappedAxes(1, 2)
        var k = applyLinear(kProj, xs).reshaped([b, t, numKVHeads, headDim]).swappedAxes(1, 2)
        let v = applyLinear(vProj, xs).reshaped([b, t, numKVHeads, headDim]).swappedAxes(1, 2)
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        let (kk, vv) = cache.update(keys: k, values: v)
        let kvLen = kk.shape[2]
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode
        if t <= 1 {
            maskMode = .none
        } else {
            let causal = MLXArray.tri(t, m: kvLen, k: kvLen - t, type: Float.self) * 1e9 - 1e9
            maskMode = .array(causal.reshaped([1, 1, t, kvLen]).asType(q.dtype))
        }
        // MLXFast SDPA repeats KV heads internally for GQA when q has more heads.
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: kk, values: vv, scale: scale, mask: maskMode)
        let merged = out.swappedAxes(1, 2).reshaped([b, t, numHeads * headDim])
        return applyLinear(oProj, merged)
    }
}

// MARK: - FFN (SwiGLU, separate gate/up/down)

public final class CSMFFN: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Module
    @ModuleInfo(key: "up_proj") public var upProj: Module
    @ModuleInfo(key: "down_proj") public var downProj: Module

    public init(_ c: CSMTowerConfig, bits: Int?, groupSize: Int) {
        _gateProj = ModuleInfo(wrappedValue: makeLinear(c.dim, c.intermediateDim, bias: false, groupSize: groupSize, bits: bits))
        _upProj = ModuleInfo(wrappedValue: makeLinear(c.dim, c.intermediateDim, bias: false, groupSize: groupSize, bits: bits))
        _downProj = ModuleInfo(wrappedValue: makeLinear(c.intermediateDim, c.dim, bias: false, groupSize: groupSize, bits: bits))
    }

    public func callAsFunction(_ xs: MLXArray) -> MLXArray {
        applyLinear(downProj, silu(applyLinear(gateProj, xs)) * applyLinear(upProj, xs))
    }
}

// MARK: - Transformer layer + tower

public final class CSMLayer: Module {
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: RMSNormF32
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: RMSNormF32
    @ModuleInfo(key: "self_attn") public var selfAttn: CSMAttention
    @ModuleInfo public var mlp: CSMFFN

    public init(_ c: CSMTowerConfig, bits: Int?, groupSize: Int) {
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNormF32(dimensions: c.dim, eps: c.normEps))
        _postAttentionLayernorm = ModuleInfo(wrappedValue: RMSNormF32(dimensions: c.dim, eps: c.normEps))
        _selfAttn = ModuleInfo(wrappedValue: CSMAttention(c, bits: bits, groupSize: groupSize))
        _mlp = ModuleInfo(wrappedValue: CSMFFN(c, bits: bits, groupSize: groupSize))
    }

    public func callAsFunction(_ xs: MLXArray, cache: any KVCache, offset: Int) -> MLXArray {
        var x = xs + selfAttn(inputLayernorm(xs), cache: cache, offset: offset)
        x = x + mlp(postAttentionLayernorm(x))
        return x
    }
}

public final class CSMTower: Module {
    @ModuleInfo public var layers: [CSMLayer]
    @ModuleInfo public var norm: RMSNormF32
    public private(set) var cache: [any KVCache]

    public init(_ c: CSMTowerConfig, bits: Int?, groupSize: Int) {
        _layers = ModuleInfo(wrappedValue: (0..<c.numLayers).map { _ in CSMLayer(c, bits: bits, groupSize: groupSize) })
        _norm = ModuleInfo(wrappedValue: RMSNormF32(dimensions: c.dim, eps: c.normEps))
        cache = (0..<c.numLayers).map { _ in KVCacheSimple() }
    }

    public func resetCache() { cache = layers.map { _ in KVCacheSimple() } }

    public func callAsFunction(_ embeds: MLXArray, offset: Int) -> MLXArray {
        var h = embeds
        for (layer, c) in zip(layers, cache) { h = layer(h, cache: c, offset: offset) }
        return norm(h)
    }
}

// MARK: - CSM model (backbone + decoder + heads)

public final class CSMModel: Module {
    public let cfg: CSMConfig

    @ModuleInfo public var backbone: CSMTower
    @ModuleInfo public var decoder: CSMTower
    @ModuleInfo(key: "text_embeddings") public var textEmbeddings: Module   // (Quantized)Embedding
    @ModuleInfo(key: "audio_embeddings") public var audioEmbeddings: Module
    @ModuleInfo(key: "codebook0_head") public var codebook0Head: Module     // backbone dim -> audio vocab
    @ModuleInfo public var projection: Module                                // backbone dim -> decoder dim
    // audio_head: [numCodebooks-1, decoderDim, audioVocab] weights for codebooks 1..31.
    // Keyed by property name (audioHead); the loader's snake→camel maps our
    // exported `audio_head` onto it, consistent with the rest of the tree.
    @ParameterInfo public var audioHead: MLXArray

    public init(_ cfg: CSMConfig) {
        self.cfg = cfg
        let bits = cfg.quantBits, gs = cfg.quantGroupSize
        _backbone = ModuleInfo(wrappedValue: CSMTower(cfg.backbone, bits: bits, groupSize: gs))
        _decoder = ModuleInfo(wrappedValue: CSMTower(cfg.decoder, bits: bits, groupSize: gs))
        _textEmbeddings = ModuleInfo(wrappedValue:
            makeEmbedding(cfg.textVocabSize, cfg.backbone.dim, bits: bits, groupSize: gs))
        _audioEmbeddings = ModuleInfo(wrappedValue:
            makeEmbedding(cfg.audioVocabSize * cfg.audioNumCodebooks, cfg.backbone.dim, bits: bits, groupSize: gs))
        _codebook0Head = ModuleInfo(wrappedValue:
            makeLinear(cfg.backbone.dim, cfg.audioVocabSize, bias: false, groupSize: gs, bits: bits))
        _projection = ModuleInfo(wrappedValue:
            makeLinear(cfg.backbone.dim, cfg.decoder.dim, bias: false, groupSize: gs, bits: bits))
        _audioHead = ParameterInfo(
            wrappedValue: MLXArray.zeros([cfg.audioNumCodebooks - 1, cfg.decoder.dim, cfg.audioVocabSize]))
    }
}

/// Build a (quantized) embedding matching how our export stored the table.
func makeEmbedding(_ count: Int, _ dim: Int, bits: Int?, groupSize: Int) -> Module {
    if let bits {
        return QuantizedEmbedding(embeddingCount: count, dimensions: dim, groupSize: groupSize, bits: bits)
    }
    return Embedding(embeddingCount: count, dimensions: dim)
}
