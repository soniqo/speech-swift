import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Differential attention (DiT-Medium uses 5-way QKV split)

/// `to_qkv(x) → q, k, v, q_diff, k_diff`. Two SDPAs, subtract. RoPE on first
/// `ropeDims` of each head. QK-norm per-head (RMSNorm over head dim).
/// Linears are INT-quantised when `bits ∈ {4,8}`, plain FP16 when `bits == 0`.
public final class DiffSelfAttentionMedium: Module {
    @ModuleInfo(key: "to_qkv") public var toQKV: QuantizedLinear
    @ModuleInfo(key: "to_out") public var toOut: QuantizedLinear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public let scale: Float

    public init(bits: Int, groupSize: Int = 64) {
        let E = DiTMediumDims.embedDim
        let D = DiTMediumDims.headDim
        self._toQKV.wrappedValue = QuantizedLinear(E, 5 * E, bias: false,
                                                    groupSize: groupSize, bits: bits)
        self._toOut.wrappedValue = QuantizedLinear(E, E, bias: false,
                                                    groupSize: groupSize, bits: bits)
        self._qNorm.wrappedValue = RMSNorm(dimensions: D, eps: DiTMediumDims.qkNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: D, eps: DiTMediumDims.qkNormEps)
        self.scale = 1.0 / Float(D).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let H = DiTMediumDims.numHeads, D = DiTMediumDims.headDim
        let qkv = toQKV(x)                                       // [B, T, 5E]
        let parts = MLX.split(qkv, parts: 5, axis: -1)
        var q = parts[0], k = parts[1], v = parts[2], qDiff = parts[3], kDiff = parts[4]

        func toHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, T, H, D]).transposed(0, 2, 1, 3)
        }
        q = toHeads(q); k = toHeads(k); v = toHeads(v)
        qDiff = toHeads(qDiff); kDiff = toHeads(kDiff)

        q     = qNorm(q);     k     = kNorm(k)
        qDiff = qNorm(qDiff); kDiff = kNorm(kDiff)

        q     = MLXFast.RoPE(q,     dimensions: DiTMediumDims.ropeDims, traditional: false,
                              base: DiTMediumDims.ropeBase, scale: 1.0, offset: 0)
        k     = MLXFast.RoPE(k,     dimensions: DiTMediumDims.ropeDims, traditional: false,
                              base: DiTMediumDims.ropeBase, scale: 1.0, offset: 0)
        qDiff = MLXFast.RoPE(qDiff, dimensions: DiTMediumDims.ropeDims, traditional: false,
                              base: DiTMediumDims.ropeBase, scale: 1.0, offset: 0)
        kDiff = MLXFast.RoPE(kDiff, dimensions: DiTMediumDims.ropeDims, traditional: false,
                              base: DiTMediumDims.ropeBase, scale: 1.0, offset: 0)

        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: v,
                                                         scale: scale, mask: nil)
        let outF = (outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, T, DiTMediumDims.embedDim])
        return toOut(outF)
    }
}

public final class DiffCrossAttentionMedium: Module {
    @ModuleInfo(key: "to_q") public var toQ: QuantizedLinear        // E → 2E
    @ModuleInfo(key: "to_kv") public var toKV: QuantizedLinear      // E → 3E
    @ModuleInfo(key: "to_out") public var toOut: QuantizedLinear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm
    public let scale: Float

    public init(bits: Int, groupSize: Int = 64) {
        let E = DiTMediumDims.embedDim
        let D = DiTMediumDims.headDim
        self._toQ.wrappedValue = QuantizedLinear(E, 2 * E, bias: false,
                                                  groupSize: groupSize, bits: bits)
        self._toKV.wrappedValue = QuantizedLinear(E, 3 * E, bias: false,
                                                   groupSize: groupSize, bits: bits)
        self._toOut.wrappedValue = QuantizedLinear(E, E, bias: false,
                                                    groupSize: groupSize, bits: bits)
        self._qNorm.wrappedValue = RMSNorm(dimensions: D, eps: DiTMediumDims.qkNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: D, eps: DiTMediumDims.qkNormEps)
        self.scale = 1.0 / Float(D).squareRoot()
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let B = x.dim(0), Tx = x.dim(1), Tc = context.dim(1)
        let H = DiTMediumDims.numHeads, D = DiTMediumDims.headDim
        let qParts = MLX.split(toQ(x), parts: 2, axis: -1)
        var q = qParts[0], qDiff = qParts[1]
        let kvParts = MLX.split(toKV(context), parts: 3, axis: -1)
        var k = kvParts[0], kDiff = kvParts[1]
        let v = kvParts[2]

        func qHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, Tx, H, D]).transposed(0, 2, 1, 3)
        }
        func cHeads(_ t: MLXArray) -> MLXArray {
            t.reshaped([B, Tc, H, D]).transposed(0, 2, 1, 3)
        }
        q = qHeads(q); qDiff = qHeads(qDiff)
        k = cHeads(k); kDiff = cHeads(kDiff)
        let vh = cHeads(v)

        q = qNorm(q); k = kNorm(k)
        qDiff = qNorm(qDiff); kDiff = kNorm(kDiff)

        let outMain = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: vh,
                                                         scale: scale, mask: nil)
        let outDiff = MLXFast.scaledDotProductAttention(queries: qDiff, keys: kDiff, values: vh,
                                                         scale: scale, mask: nil)
        let outF = (outMain - outDiff).transposed(0, 2, 1, 3).reshaped([B, Tx, DiTMediumDims.embedDim])
        return toOut(outF)
    }
}

// MARK: - GeGLU feed-forward

/// `ff.0.proj(x) → split → silu(gate) * value` (FF inner = 6144).
public final class GLUWrapMedium: Module {
    @ModuleInfo public var proj: QuantizedLinear
    public init(bits: Int, groupSize: Int = 64) {
        self._proj.wrappedValue = QuantizedLinear(
            DiTMediumDims.embedDim, 2 * DiTMediumDims.ffInner,
            bias: true, groupSize: groupSize, bits: bits)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let p = proj(x)
        let parts = MLX.split(p, parts: 2, axis: -1)
        return parts[0] * silu(parts[1])
    }
}

/// FeedForward path with `ff.0 = GLU`, `ff.2 = QuantizedLinear(ffInner, embed)`.
/// We model the indexed list via custom @ModuleInfo keys "0" / "2". Both halves
/// are INT-quantised (ffInner = 6144 / embed = 1536 — both divisible by 64).
public final class FeedForwardMedium: Module {
    /// Inner container — loader rewrites `ff.ff.{0,2}` → `ff.ff.{glu,out}`.
    public final class Inner: Module {
        @ModuleInfo public var glu: GLUWrapMedium
        @ModuleInfo public var out: QuantizedLinear
        public init(bits: Int, groupSize: Int = 64) {
            self._glu.wrappedValue = GLUWrapMedium(bits: bits, groupSize: groupSize)
            self._out.wrappedValue = QuantizedLinear(
                DiTMediumDims.ffInner, DiTMediumDims.embedDim,
                bias: true, groupSize: groupSize, bits: bits)
            super.init()
        }
    }
    @ModuleInfo public var ff: Inner
    public init(bits: Int) {
        self._ff.wrappedValue = Inner(bits: bits)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        ff.out(ff.glu(x))
    }
}

/// Per-layer local-conditioning MLP: 257 → embed → embed with SiLU between.
/// Checkpoint keys: `to_local_embed.seq.0.{weight,bias}`, `to_local_embed.seq.2.{weight,bias}`.
/// `.0` has input 257 (not divisible by 64) so stays plain Linear.
/// `.2` is embed→embed (1536→1536) — INT-quantised.
public final class LocalEmbedSeqMedium: Module {
    public final class Seq: Module {
        @ModuleInfo public var inProj: Linear
        @ModuleInfo public var outProj: QuantizedLinear
        public init(bits: Int, groupSize: Int = 64) {
            self._inProj.wrappedValue = Linear(DiTMediumDims.localAddCondDim, DiTMediumDims.embedDim, bias: true)
            self._outProj.wrappedValue = QuantizedLinear(
                DiTMediumDims.embedDim, DiTMediumDims.embedDim,
                bias: true, groupSize: groupSize, bits: bits)
            super.init()
        }
    }
    @ModuleInfo public var seq: Seq
    public init(bits: Int) {
        self._seq.wrappedValue = Seq(bits: bits)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        seq.outProj(silu(seq.inProj(x)))
    }
}

// MARK: - Transformer block (Medium)

public final class TransformerBlockMedium: Module {
    @ModuleInfo(key: "pre_norm") public var preNorm: RMSNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: DiffSelfAttentionMedium
    @ModuleInfo(key: "cross_attend_norm") public var crossAttendNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") public var crossAttn: DiffCrossAttentionMedium
    @ModuleInfo(key: "ff_norm") public var ffNorm: RMSNorm
    @ModuleInfo public var ff: FeedForwardMedium
    @ModuleInfo(key: "to_local_embed") public var toLocalEmbed: LocalEmbedSeqMedium
    @ParameterInfo(key: "to_scale_shift_gate") public var toScaleShiftGate: MLXArray  // (6E,)

    public init(bits: Int) {
        let E = DiTMediumDims.embedDim
        self._preNorm.wrappedValue = RMSNorm(dimensions: E, eps: DiTMediumDims.normEps)
        self._selfAttn.wrappedValue = DiffSelfAttentionMedium(bits: bits)
        self._crossAttendNorm.wrappedValue = RMSNorm(dimensions: E, eps: DiTMediumDims.normEps)
        self._crossAttn.wrappedValue = DiffCrossAttentionMedium(bits: bits)
        self._ffNorm.wrappedValue = RMSNorm(dimensions: E, eps: DiTMediumDims.normEps)
        self._ff.wrappedValue = FeedForwardMedium(bits: bits)
        self._toLocalEmbed.wrappedValue = LocalEmbedSeqMedium(bits: bits)
        self._toScaleShiftGate.wrappedValue = MLXArray.zeros([6 * E])
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, context: MLXArray,
                                globalCond: MLXArray, localEmbPadded: MLXArray) -> MLXArray {
        // (to_scale_shift_gate + global_cond)[:, None, :] split into 6 chunks
        let ssBase = toScaleShiftGate.asType(globalCond.dtype) + globalCond
        let ss = ssBase.expandedDimensions(axis: 1)              // [B, 1, 6E]
        let parts = MLX.split(ss, parts: 6, axis: -1)
        let scaleSelf = parts[0], shiftSelf = parts[1], gateSelf = parts[2]
        let scaleFF = parts[3],    shiftFF = parts[4],    gateFF = parts[5]
        let one = MLXArray(Float(1.0)).asType(scaleSelf.dtype)

        var x = xIn
        var residual = x
        var h = preNorm(x)
        h = h * (one + scaleSelf) + shiftSelf
        h = selfAttn(h)
        h = h * MLX.sigmoid(one - gateSelf)
        x = h + residual

        x = x + crossAttn(crossAttendNorm(x), context: context)
        x = x + localEmbPadded

        residual = x
        h = ffNorm(x)
        h = h * (one + scaleFF) + shiftFF
        h = ff(h)
        h = h * MLX.sigmoid(one - gateFF)
        x = h + residual
        return x
    }
}

// MARK: - Top transformer + DiT

public final class GlobalCondEmbedderMedium: Module {
    @ModuleInfo public var inProj: QuantizedLinear
    @ModuleInfo public var outProj: QuantizedLinear
    public init(bits: Int, groupSize: Int = 64) {
        let E = DiTMediumDims.embedDim
        self._inProj.wrappedValue = QuantizedLinear(
            E, E, bias: true, groupSize: groupSize, bits: bits)
        self._outProj.wrappedValue = QuantizedLinear(
            E, 6 * E, bias: true, groupSize: groupSize, bits: bits)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        outProj(silu(inProj(x)))
    }
}

public final class ContinuousTransformerMedium: Module {
    @ModuleInfo(key: "project_in") public var projectIn: QuantizedLinear
    @ModuleInfo(key: "project_out") public var projectOut: QuantizedLinear
    @ParameterInfo(key: "memory_tokens") public var memoryTokens: MLXArray   // (M, E)
    @ModuleInfo(key: "global_cond_embedder") public var globalCondEmbedder: GlobalCondEmbedderMedium
    @ModuleInfo public var layers: [TransformerBlockMedium]

    public init(bits: Int) {
        let E = DiTMediumDims.embedDim
        let groupSize = 64
        self._projectIn.wrappedValue = QuantizedLinear(
            DiTMediumDims.ioChannels, E, bias: false,
            groupSize: groupSize, bits: bits)
        self._projectOut.wrappedValue = QuantizedLinear(
            E, DiTMediumDims.ioChannels, bias: false,
            groupSize: groupSize, bits: bits)
        self._memoryTokens.wrappedValue = MLXArray.zeros([DiTMediumDims.numMemoryTokens, E])
        self._globalCondEmbedder.wrappedValue = GlobalCondEmbedderMedium(bits: bits)
        self._layers.wrappedValue = (0..<DiTMediumDims.depth).map { _ in TransformerBlockMedium(bits: bits) }
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, context: MLXArray,
                                globalEmbed: MLXArray, localAddCondZeros: MLXArray) -> MLXArray {
        let B = xIn.dim(0)
        let E = DiTMediumDims.embedDim
        let M = DiTMediumDims.numMemoryTokens
        var x = projectIn(xIn)
        let mem = MLX.broadcast(
            memoryTokens.expandedDimensions(axis: 0).asType(x.dtype),
            to: [B, M, E])
        x = MLX.concatenated([mem, x], axis: 1)

        let g = globalCondEmbedder(globalEmbed)              // [B, 6E]

        let pad = MLXArray.zeros([B, M, E], dtype: x.dtype)
        for layer in layers {
            let localEmb = layer.toLocalEmbed(localAddCondZeros)        // [B, T, E]
            let localEmbPadded = MLX.concatenated([pad, localEmb], axis: 1)
            x = layer(x, context: context, globalCond: g, localEmbPadded: localEmbPadded)
        }
        x = x[0..., M..., 0...]
        return projectOut(x)
    }
}

public final class CondEmbed2Medium: Module {
    @ModuleInfo public var inProj: QuantizedLinear
    @ModuleInfo public var outProj: QuantizedLinear
    public init(inDim: Int, outDim: Int, bias: Bool, bits: Int, groupSize: Int = 64) {
        self._inProj.wrappedValue = QuantizedLinear(
            inDim, outDim, bias: bias, groupSize: groupSize, bits: bits)
        self._outProj.wrappedValue = QuantizedLinear(
            outDim, outDim, bias: bias, groupSize: groupSize, bits: bits)
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        outProj(silu(inProj(x)))
    }
}

public final class DiTMedium: Module {
    @ModuleInfo(key: "preprocess_conv") public var preprocessConv: Conv1d
    @ModuleInfo(key: "postprocess_conv") public var postprocessConv: Conv1d
    @ModuleInfo(key: "to_cond_embed") public var toCondEmbed: CondEmbed2Medium
    @ModuleInfo(key: "to_global_embed") public var toGlobalEmbed: CondEmbed2Medium
    @ModuleInfo(key: "to_timestep_embed") public var toTimestepEmbed: CondEmbed2Medium
    @ModuleInfo public var transformer: ContinuousTransformerMedium

    public let tLat: Int
    public let bits: Int

    public init(tLat: Int, bits: Int) {
        self.tLat = tLat
        self.bits = bits
        let IO = DiTMediumDims.ioChannels
        let E = DiTMediumDims.embedDim
        let CT = DiTMediumDims.condTokenDim
        let GC = DiTMediumDims.globalCondDim
        let TF = DiTMediumDims.timestepFeatDim

        self._preprocessConv.wrappedValue  = Conv1d(inputChannels: IO, outputChannels: IO, kernelSize: 1, bias: false)
        self._postprocessConv.wrappedValue = Conv1d(inputChannels: IO, outputChannels: IO, kernelSize: 1, bias: false)
        self._toCondEmbed.wrappedValue     = CondEmbed2Medium(inDim: CT, outDim: E, bias: false, bits: bits)
        self._toGlobalEmbed.wrappedValue   = CondEmbed2Medium(inDim: GC, outDim: E, bias: false, bits: bits)
        self._toTimestepEmbed.wrappedValue = CondEmbed2Medium(inDim: TF, outDim: E, bias: true,  bits: bits)
        self._transformer.wrappedValue     = ContinuousTransformerMedium(bits: bits)
        super.init()
    }

    /// `x`: [B, 256, T_lat] channels-first. `t`: [B] (current sigma).
    /// `crossAttnCondRaw`: [B, 257, 768]. `globalCondRaw`: [B, 768].
    /// `localAddCond`: optional [B, T_lat, 257] for inpainting (nil otherwise).
    /// Returns velocity prediction in [B, 256, T_lat].
    public func callAsFunction(_ x: MLXArray, t: MLXArray,
                                crossAttnCondRaw: MLXArray, globalCondRaw: MLXArray,
                                localAddCond: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0)
        let context = toCondEmbed(crossAttnCondRaw)
        let globalPre = toGlobalEmbed(globalCondRaw)
        let tf = expoFourierFeatures(t, dim: DiTMediumDims.timestepFeatDim,
                                     minFreq: 0.5, maxFreq: 10_000)
        let tEmbed = toTimestepEmbed(tf.asType(globalPre.dtype))
        let globalEmbed = globalPre + tEmbed

        // Conv1d in MLX-Swift takes [B, T, C]. Input is [B, C, T] → transpose.
        let xLC = x.transposed(0, 2, 1)
        let xPP = preprocessConv(xLC) + xLC

        let local: MLXArray
        if let lc = localAddCond {
            local = lc
        } else {
            local = MLXArray.zeros([B, tLat, DiTMediumDims.localAddCondDim], dtype: xPP.dtype)
        }

        let h = transformer(xPP, context: context, globalEmbed: globalEmbed,
                             localAddCondZeros: local)
        let out = postprocessConv(h) + h
        return out.transposed(0, 2, 1)
    }
}
