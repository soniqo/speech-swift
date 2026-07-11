import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon

// MARK: - Mimi codec encoder (Qwen3-TTS 12 Hz speech tokenizer)
//
// The Qwen3-TTS reference encoder is HuggingFace `MimiModel`
// (`class Qwen3TTSTokenizerV2Encoder(MimiModel)`). The encode path is:
//
//   xs = seanet_encoder(xs)          # conv stack, 24 kHz → 80 frames @ 512ch
//   xs = encoder_transformer(xs)     # 8-layer causal transformer
//   xs = downsample(xs)              # extra ×2 → 40 frames @ 12.5 Hz
//   codes = quantizer.encode(xs)     # split RVQ → [B, 16, T] indices
//
// This is a faithful MLX port of mlx-audio's Mimi modules, using the
// HuggingFace checkpoint's weight layout (q/k/v/o_proj, fc1/fc2,
// input/post_attention_layernorm, EMA codebooks via embed_sum/cluster_usage).
//
// Everything runs in NLC layout ([B, T, C]); the golden PyTorch tensors are
// NCL ([B, C, T]) and are transposed for comparison in the validation harness.

/// Fixed config for the 1.7B/0.6B Qwen3-TTS 12 Hz tokenizer (config.json
/// `encoder_config`). These are model constants, not per-checkpoint knobs.
public struct MimiEncoderConfig {
    public let channels = 1
    public let nFilters = 64
    public let ratios = [8, 6, 5, 4]          // applied reversed in the encoder
    public let kSize = 7
    public let lastKSize = 3
    public let residualKSize = 3
    public let dilationBase = 2
    public let nResidualLayers = 1
    public let compress = 2                    // extra downsample stride
    public let dimension = 512                 // SEANet output / transformer dim
    public let numLayers = 8
    public let numHeads = 8
    public let headDim = 64
    public let intermediateSize = 2048
    public let ropeTheta: Float = 10000
    public let codebookDim = 256
    public let codebookSize = 2048
    public let numSemanticQuantizers = 1
    public let validNumQuantizers = 16         // 1 semantic + 15 acoustic
    public init() {}
}

// MARK: - Causal streamable conv (Mimi padding)

/// Conv1d with Mimi's causal padding: pad_total = k_eff - stride on the left,
/// plus the "extra padding" that makes the framing exact, with a configurable
/// pad mode (zeros for SEANet convs, edge/replicate for the ×2 downsample).
public class MimiConv1d: Module {
    @ModuleInfo var conv: Conv1d
    let kSize: Int
    let stride: Int
    let dilation: Int
    let padEdge: Bool

    public init(inC: Int, outC: Int, kSize: Int, stride: Int = 1,
                dilation: Int = 1, bias: Bool = true, padEdge: Bool = false) {
        self.kSize = kSize
        self.stride = stride
        self.dilation = dilation
        self.padEdge = padEdge
        self._conv.wrappedValue = Conv1d(
            inputChannels: inC, outputChannels: outC, kernelSize: kSize,
            stride: stride, padding: 0, dilation: dilation, groups: 1, bias: bias)
        super.init()
    }

    /// Input/Output: [B, T, C]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let len = x.dim(1)
        let kEff = (kSize - 1) * dilation + 1
        let paddingTotal = kEff - stride
        // extra padding so the last (possibly partial) frame is covered
        let nframes = Double(max(len + paddingTotal - kEff, 0)) / Double(stride) + 1.0
        let idealLen = (Int(ceil(nframes)) - 1) * stride + kEff - paddingTotal
        let extra = max(0, idealLen - len)
        // causal: all padding on the left, plus the extra on the right
        let padded = MLX.padded(
            x,
            widths: [.init((low: 0, high: 0)),
                     .init((low: paddingTotal, high: extra)),
                     .init((low: 0, high: 0))],
            mode: padEdge ? .edge : .constant)
        return conv(padded)
    }
}

// MARK: - SEANet encoder

/// Residual unit: x + conv2(elu(conv1(elu(x)))). true_skip ⇒ no shortcut conv.
public class MimiResnetBlock: Module {
    @ModuleInfo var convA: MimiConv1d   // checkpoint block.1 (k=residual, dilation)
    @ModuleInfo var convB: MimiConv1d   // checkpoint block.3 (k=1)

    public init(dim: Int, cfg: MimiEncoderConfig, dilation: Int) {
        let hidden = dim / cfg.compress
        self._convA.wrappedValue = MimiConv1d(
            inC: dim, outC: hidden, kSize: cfg.residualKSize, dilation: dilation)
        self._convB.wrappedValue = MimiConv1d(inC: hidden, outC: dim, kSize: 1)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convA(elu(x, alpha: 1.0))
        h = convB(elu(h, alpha: 1.0))
        return h + x
    }
}

/// One downsample stage: residual block(s) then elu + strided conv.
public class MimiEncoderLayer: Module {
    @ModuleInfo var residuals: [MimiResnetBlock]
    @ModuleInfo var downsample: MimiConv1d

    public init(dim: Int, ratio: Int, cfg: MimiEncoderConfig) {
        var dilation = 1
        var blocks: [MimiResnetBlock] = []
        for _ in 0..<cfg.nResidualLayers {
            blocks.append(MimiResnetBlock(dim: dim, cfg: cfg, dilation: dilation))
            dilation *= cfg.dilationBase
        }
        self._residuals.wrappedValue = blocks
        self._downsample.wrappedValue = MimiConv1d(
            inC: dim, outC: dim * 2, kSize: ratio * 2, stride: ratio)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in residuals { h = r(h) }
        return downsample(elu(h, alpha: 1.0))
    }
}

public class MimiSeanetEncoder: Module {
    @ModuleInfo var initConv: MimiConv1d
    @ModuleInfo var layers: [MimiEncoderLayer]
    @ModuleInfo var finalConv: MimiConv1d

    public init(cfg: MimiEncoderConfig) {
        self._initConv.wrappedValue = MimiConv1d(
            inC: cfg.channels, outC: cfg.nFilters, kSize: cfg.kSize)
        var mult = 1
        var ls: [MimiEncoderLayer] = []
        for ratio in cfg.ratios.reversed() {
            ls.append(MimiEncoderLayer(dim: mult * cfg.nFilters, ratio: ratio, cfg: cfg))
            mult *= 2
        }
        self._layers.wrappedValue = ls
        self._finalConv.wrappedValue = MimiConv1d(
            inC: mult * cfg.nFilters, outC: cfg.dimension, kSize: cfg.lastKSize)
        super.init()
    }

    /// [B, T_samples, 1] → [B, T/960, 512]
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = initConv(x)
        for l in layers { h = l(h) }
        h = elu(h, alpha: 1.0)
        return finalConv(h)
    }
}

// MARK: - Encoder transformer (HF Mimi layout)

public class MimiTransformerLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnLayernorm: LayerNorm
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ParameterInfo(key: "self_attn_scale") var selfAttnScale: MLXArray
    @ParameterInfo(key: "mlp_scale") var mlpScale: MLXArray
    let rope: RoPE
    let numHeads: Int
    let headDim: Int
    let scale: Float

    public init(cfg: MimiEncoderConfig) {
        self._inputLayernorm.wrappedValue = LayerNorm(dimensions: cfg.dimension, eps: 1e-5)
        self._postAttnLayernorm.wrappedValue = LayerNorm(dimensions: cfg.dimension, eps: 1e-5)
        self._qProj.wrappedValue = Linear(cfg.dimension, cfg.numHeads * cfg.headDim, bias: false)
        self._kProj.wrappedValue = Linear(cfg.dimension, cfg.numHeads * cfg.headDim, bias: false)
        self._vProj.wrappedValue = Linear(cfg.dimension, cfg.numHeads * cfg.headDim, bias: false)
        self._oProj.wrappedValue = Linear(cfg.numHeads * cfg.headDim, cfg.dimension, bias: false)
        self._fc1.wrappedValue = Linear(cfg.dimension, cfg.intermediateSize, bias: false)
        self._fc2.wrappedValue = Linear(cfg.intermediateSize, cfg.dimension, bias: false)
        self._selfAttnScale.wrappedValue = MLXArray.ones([cfg.dimension])
        self._mlpScale.wrappedValue = MLXArray.ones([cfg.dimension])
        self.rope = RoPE(dimensions: cfg.headDim, traditional: false, base: cfg.ropeTheta)
        self.numHeads = cfg.numHeads
        self.headDim = cfg.headDim
        self.scale = pow(Float(cfg.headDim), -0.5)
        super.init()
    }

    private func attention(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        func heads(_ p: MLXArray) -> MLXArray {
            p.reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)  // [B,H,T,Hd]
        }
        var q = heads(qProj(x))
        var k = heads(kProj(x))
        let v = heads(vProj(x))
        q = rope(q)
        k = rope(k)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, numHeads * headDim])
        return oProj(merged)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var h = x + selfAttnScale * attention(inputLayernorm(x), mask: mask)
        h = h + mlpScale * fc2(gelu(fc1(postAttnLayernorm(h))))
        return h
    }
}

public class MimiEncoderTransformer: Module {
    @ModuleInfo var layers: [MimiTransformerLayer]

    public init(cfg: MimiEncoderConfig) {
        self._layers.wrappedValue = (0..<cfg.numLayers).map { _ in MimiTransformerLayer(cfg: cfg) }
        super.init()
    }

    /// [B, T, 512] → [B, T, 512] (causal self-attention)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = x.dim(1)
        // additive causal mask [T, T]: -inf above the diagonal
        let rows = MLXArray(Int32(0)..<Int32(t)).reshaped([t, 1])
        let cols = MLXArray(Int32(0)..<Int32(t)).reshaped([1, t])
        let mask = MLX.which(cols .> rows, MLXArray(-Float.greatestFiniteMagnitude), MLXArray(Float(0)))
        var h = x
        for l in layers { h = l(h, mask: mask) }
        return h
    }
}

// MARK: - Split residual vector quantizer (encode only)

/// EMA codebook: embedding = embed_sum / max(cluster_usage, eps); nearest by
/// argmin(‖e‖²/2 − x·e).
public class MimiEuclideanCodebook: Module {
    @ParameterInfo(key: "embed_sum") var embedSum: MLXArray
    @ParameterInfo(key: "cluster_usage") var clusterUsage: MLXArray
    let dim: Int

    public init(size: Int, dim: Int) {
        self.dim = dim
        self._embedSum.wrappedValue = MLXArray.zeros([size, dim])
        self._clusterUsage.wrappedValue = MLXArray.zeros([size])
        super.init()
    }

    /// embedding [size, dim] from EMA stats.
    func embedding() -> MLXArray {
        let denom = maximum(clusterUsage, MLXArray(Float(1e-5))).reshaped([clusterUsage.dim(0), 1])
        return embedSum / denom
    }

    /// x: [B, T, dim] → indices [B, T]
    func encode(_ x: MLXArray, embedding e: MLXArray, c2: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1)
        let flat = x.reshaped([b * t, dim])
        let dot = matmul(flat, e.transposed(1, 0))          // [B*T, size]
        let idx = argMin(c2 - dot, axis: -1)                // [B*T]
        return idx.reshaped([b, t])
    }

    /// indices [B, T] → vectors [B, T, dim]
    func decode(_ idx: MLXArray, embedding e: MLXArray) -> MLXArray {
        let b = idx.dim(0), t = idx.dim(1)
        return e[idx.reshaped([b * t])].reshaped([b, t, dim])
    }
}

/// One residual vector quantizer: 512→256 input_proj, then nq residual codebooks.
public class MimiResidualVQ: Module {
    @ModuleInfo(key: "input_proj") var inputProj: MimiConv1d
    @ModuleInfo var codebooks: [MimiEuclideanCodebook]

    public init(nq: Int, cfg: MimiEncoderConfig) {
        self._inputProj.wrappedValue = MimiConv1d(
            inC: cfg.dimension, outC: cfg.codebookDim, kSize: 1, bias: false)
        self._codebooks.wrappedValue = (0..<nq).map { _ in
            MimiEuclideanCodebook(size: cfg.codebookSize, dim: cfg.codebookDim)
        }
        super.init()
    }

    /// h: [B, T, 512] → codes [B, nq, T]
    public func encode(_ h: MLXArray) -> MLXArray {
        var residual = inputProj(h)                          // [B, T, 256]
        var codes: [MLXArray] = []
        for cb in codebooks {
            let e = cb.embedding()
            let c2 = (e * e).sum(axis: -1) / 2
            let idx = cb.encode(residual, embedding: e, c2: c2)   // [B, T]
            let quant = cb.decode(idx, embedding: e)              // [B, T, 256]
            residual = residual - quant
            codes.append(idx.expandedDimensions(axis: 1))         // [B, 1, T]
        }
        return concatenated(codes, axis: 1)                       // [B, nq, T]
    }
}

public class MimiSplitRVQ: Module {
    @ModuleInfo(key: "semantic_rvq") var semanticRVQ: MimiResidualVQ
    @ModuleInfo(key: "acoustic_rvq") var acousticRVQ: MimiResidualVQ

    public init(cfg: MimiEncoderConfig) {
        self._semanticRVQ.wrappedValue = MimiResidualVQ(nq: cfg.numSemanticQuantizers, cfg: cfg)
        self._acousticRVQ.wrappedValue = MimiResidualVQ(
            nq: cfg.validNumQuantizers - cfg.numSemanticQuantizers, cfg: cfg)
        super.init()
    }

    /// h: [B, T, 512] → codes [B, 16, T]
    public func encode(_ h: MLXArray) -> MLXArray {
        let sem = semanticRVQ.encode(h)        // [B, 1, T]
        let aco = acousticRVQ.encode(h)        // [B, 15, T]
        return concatenated([sem, aco], axis: 1)
    }
}

// MARK: - Top-level encoder

/// Mimi codec encoder. Public surface (`encode(samples:) -> [1, 16, T]`) is
/// unchanged so the ICL caller is untouched.
public class SpeechTokenizerEncoder: Module {
    @ModuleInfo(key: "seanet") var seanet: MimiSeanetEncoder
    @ModuleInfo(key: "transformer") var transformer: MimiEncoderTransformer
    @ModuleInfo(key: "downsample") var downsample: MimiConv1d
    @ModuleInfo(key: "quantizer") var quantizer: MimiSplitRVQ
    public let cfg: MimiEncoderConfig

    public init(config: SpeechTokenizerDecoderConfig) {
        let c = MimiEncoderConfig()
        self.cfg = c
        self._seanet.wrappedValue = MimiSeanetEncoder(cfg: c)
        self._transformer.wrappedValue = MimiEncoderTransformer(cfg: c)
        // The extra ×2 downsample uses edge padding and no bias.
        self._downsample.wrappedValue = MimiConv1d(
            inC: c.dimension, outC: c.dimension, kSize: c.compress * 2,
            stride: c.compress, bias: false, padEdge: true)
        self._quantizer.wrappedValue = MimiSplitRVQ(cfg: c)
        super.init()
    }

    /// Encode a 24 kHz mono PCM buffer to codec indices [1, 16, T_frames].
    public func encode(samples: [Float]) -> MLXArray {
        let audio = MLXArray(samples).reshaped([1, samples.count, 1])  // [B, T, 1]
        var h = seanet(audio)                       // [1, T/960, 512]
        h = transformer(h)                          // [1, T/960, 512]
        h = downsample(h)                           // [1, T/1920, 512]
        return quantizer.encode(h)                  // [1, 16, T/1920]
    }
}
