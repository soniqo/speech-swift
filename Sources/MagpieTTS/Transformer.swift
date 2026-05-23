import Foundation
import MLX
import MLXNN

// MARK: - KV cache containers

/// Per-layer self-attention cache. `nil` means uninitialised (prefill seed).
public final class MagpieSACache {
    public var k: MLXArray?  // (B, T_past, n_heads, d_head)
    public var v: MLXArray?
    public init() { self.k = nil; self.v = nil }
}

/// Per-layer cross-attention cache (encoder-derived K/V; cached forever).
public final class MagpieXACache {
    public var k: MLXArray?  // (B, T_mem, n_heads, d_head)
    public var v: MLXArray?
    public init() { self.k = nil; self.v = nil }
}

public final class MagpieKVCache {
    public let sa: MagpieSACache
    public let xa: MagpieXACache
    public init() { self.sa = MagpieSACache(); self.xa = MagpieXACache() }
}

public func magpieEmptyCache(layers: Int) -> [MagpieKVCache] {
    (0..<layers).map { _ in MagpieKVCache() }
}

// MARK: - Causal 1-D conv FFN (NLC layout)

/// Conv1d with left-only causal padding (NeMo's transformer_2501 layout).
/// Weight stored as `(out_channels, kernel_size, in_channels)` (MLX NLC).
public final class MagpieCausalConv1d: Module {
    public var weight: MLXArray
    public var bias: MLXArray?
    public let kernelSize: Int
    public let isCausal: Bool
    public let leftPad: Int
    public let samePad: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                isCausal: Bool, bias: Bool = false) {
        self.kernelSize = kernelSize
        self.isCausal = isCausal
        if isCausal {
            self.leftPad = kernelSize - 1
            self.samePad = 0
        } else {
            precondition(kernelSize % 2 == 1, "non-causal conv expects odd kernel")
            self.leftPad = 0
            self.samePad = (kernelSize - 1) / 2
        }
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if isCausal && leftPad > 0 {
            h = MLX.padded(h, widths: [.init((0, 0)), .init((leftPad, 0)), .init((0, 0))])
        }
        var y = MLX.conv1d(h, weight, stride: 1, padding: samePad)
        if let b = bias { y = y + b }
        return y
    }
}

/// Conv1d(d, d_ff, k) → GELU(approx) → Conv1d(d_ff, d, k). No bias.
public final class MagpiePositionwiseConvFF: Module {
    @ModuleInfo public var proj: MagpieCausalConv1d
    @ModuleInfo(key: "o_net") public var oNet: MagpieCausalConv1d

    public init(dModel: Int, dFfn: Int, kernelSize: Int, isCausal: Bool) {
        self._proj = ModuleInfo(wrappedValue:
            MagpieCausalConv1d(inChannels: dModel, outChannels: dFfn,
                               kernelSize: kernelSize, isCausal: isCausal, bias: false))
        self._oNet = ModuleInfo(
            wrappedValue: MagpieCausalConv1d(inChannels: dFfn, outChannels: dModel,
                                              kernelSize: kernelSize, isCausal: isCausal, bias: false),
            key: "o_net")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        if let m = mask { h = h * m[.ellipsis, .newAxis] }
        h = proj(h)
        if let m = mask { h = h * m[.ellipsis, .newAxis] }
        h = geluApproximate(h)
        h = oNet(h)
        if let m = mask { h = h * m[.ellipsis, .newAxis] }
        return h
    }
}

// MARK: - Attention with explicit KV cache I/O

/// Causal multi-head self-attention. Always returns the updated cache.
public final class MagpieSelfAttention: Module {
    @ModuleInfo(key: "qkv_net") public var qkvNet: Linear
    @ModuleInfo(key: "o_net") public var oNet: Linear
    public let nHeads: Int
    public let dHead: Int
    public let scale: Float
    public let isCausal: Bool

    public init(dModel: Int, nHeads: Int, isCausal: Bool) {
        precondition(dModel % nHeads == 0, "d_model must be divisible by n_heads")
        self.nHeads = nHeads
        self.dHead = dModel / nHeads
        self.scale = Float(1.0 / sqrt(Double(self.dHead)))
        self.isCausal = isCausal
        self._qkvNet = ModuleInfo(
            wrappedValue: Linear(dModel, 3 * nHeads * dHead, bias: false),
            key: "qkv_net")
        self._oNet = ModuleInfo(
            wrappedValue: Linear(nHeads * dHead, dModel, bias: false),
            key: "o_net")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: MagpieSACache) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let qkv = qkvNet(x).reshaped([B, T, 3, nHeads, dHead])
        let q  = qkv[0..., 0..., 0]
        let nk = qkv[0..., 0..., 1]
        let nv = qkv[0..., 0..., 2]

        let k: MLXArray
        let v: MLXArray
        if let prevK = cache.k, let prevV = cache.v {
            k = MLX.concatenated([prevK, nk], axis: 1)
            v = MLX.concatenated([prevV, nv], axis: 1)
        } else {
            k = nk
            v = nv
        }
        let Tkv = k.dim(1)

        // (B, T, nH, dH) -> (B, nH, T, dH)
        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        var scores = MLX.matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let m = mask {
            // m: (B, T_kv) → (B, 1, 1, T_kv); broadcast across heads + queries.
            let km = m[0..., .newAxis, .newAxis, 0..<Tkv]
            scores = MLX.where(km .== MLXArray(Float(0)),
                               MLXArray(Float(-1e30)), scores)
        }
        if isCausal {
            // allow[i,j] = j <= (T_kv - T) + i
            let qIdx = MLXArray(0..<Int32(T)).reshaped([T, 1])
            let kIdx = MLXArray(0..<Int32(Tkv)).reshaped([1, Tkv])
            let allow = kIdx .<= (qIdx + MLXArray(Int32(Tkv - T)))
            let block = MLX.logicalNot(allow).reshaped([1, 1, T, Tkv])
            scores = MLX.where(block, MLXArray(Float(-1e30)), scores)
        }
        let probs = MLX.softmax(scores, axis: -1)
        var y = MLX.matmul(probs, vT)                                 // (B, nH, T, dH)
        y = y.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * dHead])
        cache.k = k
        cache.v = v
        return oNet(y)
    }
}

/// Cross-attention over an encoder memory. Cached K/V skip the kv projection
/// on every step after the first.
public final class MagpieCrossAttention: Module {
    @ModuleInfo(key: "q_net") public var qNet: Linear
    @ModuleInfo(key: "kv_net") public var kvNet: Linear
    @ModuleInfo(key: "o_net") public var oNet: Linear
    public let nHeads: Int
    public let dHead: Int
    public let scale: Float

    public init(dModel: Int, dMemory: Int, nHeads: Int, dHead: Int) {
        self.nHeads = nHeads
        self.dHead = dHead
        self.scale = Float(1.0 / sqrt(Double(dHead)))
        self._qNet  = ModuleInfo(
            wrappedValue: Linear(dModel, nHeads * dHead, bias: false),
            key: "q_net")
        self._kvNet = ModuleInfo(
            wrappedValue: Linear(dMemory, 2 * nHeads * dHead, bias: false),
            key: "kv_net")
        self._oNet  = ModuleInfo(
            wrappedValue: Linear(nHeads * dHead, dModel, bias: false),
            key: "o_net")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, memory: MLXArray,
                                memoryMask: MLXArray?, cache: MagpieXACache) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let q = qNet(x).reshaped([B, T, nHeads, dHead])

        let k: MLXArray
        let v: MLXArray
        if let pk = cache.k, let pv = cache.v {
            k = pk
            v = pv
        } else {
            let Bm = memory.dim(0)
            let Tm = memory.dim(1)
            let kv = kvNet(memory).reshaped([Bm, Tm, 2, nHeads, dHead])
            k = kv[0..., 0..., 0]
            v = kv[0..., 0..., 1]
            cache.k = k
            cache.v = v
        }
        let Tm = k.dim(1)

        let qT = q.transposed(0, 2, 1, 3)
        let kT = k.transposed(0, 2, 1, 3)
        let vT = v.transposed(0, 2, 1, 3)

        var scores = MLX.matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)
        if let m = memoryMask {
            let mm = m[0..., .newAxis, .newAxis, 0..<Tm]
            scores = MLX.where(mm .== MLXArray(Float(0)),
                               MLXArray(Float(-1e30)), scores)
        }
        let probs = MLX.softmax(scores, axis: -1)
        var y = MLX.matmul(probs, vT)
        y = y.transposed(0, 2, 1, 3).reshaped([B, T, nHeads * dHead])
        return oNet(y)
    }
}

// MARK: - One transformer layer (encoder + decoder share this)

public final class MagpieTransformerLayer: Module {
    public let hasXattn: Bool
    @ModuleInfo(key: "norm_self") public var normSelf: LayerNorm
    @ModuleInfo(key: "self_attention") public var selfAttention: MagpieSelfAttention
    @ModuleInfo(key: "norm_xattn_query") public var normXattnQuery: LayerNorm?
    @ModuleInfo(key: "cross_attention") public var crossAttention: MagpieCrossAttention?
    @ModuleInfo(key: "norm_xattn_memory") public var normXattnMemory: LayerNorm?
    @ModuleInfo(key: "norm_pos_ff") public var normPosFf: LayerNorm
    @ModuleInfo(key: "pos_ff") public var posFf: MagpiePositionwiseConvFF

    public init(dModel: Int, dFfn: Int, nHeads: Int, kernelSize: Int,
                hasXattn: Bool, isCausal: Bool,
                xaDMemory: Int? = nil, xaNHeads: Int? = nil, xaDHead: Int? = nil,
                applyNormToCond: Bool = true) {
        self.hasXattn = hasXattn
        self._normSelf = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: dModel, eps: 1e-5, affine: true, bias: false),
            key: "norm_self")
        self._selfAttention = ModuleInfo(
            wrappedValue: MagpieSelfAttention(dModel: dModel, nHeads: nHeads, isCausal: isCausal),
            key: "self_attention")
        if hasXattn {
            guard let dMem = xaDMemory, let nH = xaNHeads, let dH = xaDHead else {
                fatalError("xattn layer needs xa_d_memory, xa_n_heads, xa_d_head")
            }
            self._normXattnQuery = ModuleInfo(
                wrappedValue: LayerNorm(dimensions: dModel, eps: 1e-5, affine: true, bias: false),
                key: "norm_xattn_query")
            self._crossAttention = ModuleInfo(
                wrappedValue: MagpieCrossAttention(dModel: dModel, dMemory: dMem, nHeads: nH, dHead: dH),
                key: "cross_attention")
            if applyNormToCond {
                self._normXattnMemory = ModuleInfo(
                    wrappedValue: LayerNorm(dimensions: dMem, eps: 1e-5, affine: true, bias: false),
                    key: "norm_xattn_memory")
            } else {
                self._normXattnMemory = ModuleInfo(wrappedValue: nil, key: "norm_xattn_memory")
            }
        } else {
            self._normXattnQuery = ModuleInfo(wrappedValue: nil, key: "norm_xattn_query")
            self._crossAttention = ModuleInfo(wrappedValue: nil, key: "cross_attention")
            self._normXattnMemory = ModuleInfo(wrappedValue: nil, key: "norm_xattn_memory")
        }
        self._normPosFf = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: dModel, eps: 1e-5, affine: true, bias: false),
            key: "norm_pos_ff")
        self._posFf = ModuleInfo(
            wrappedValue: MagpiePositionwiseConvFF(dModel: dModel, dFfn: dFfn,
                                                   kernelSize: kernelSize, isCausal: isCausal),
            key: "pos_ff")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray?,
                                memory: MLXArray?, memoryMask: MLXArray?,
                                cache: MagpieKVCache) -> MLXArray {
        var h = x
        if let m = mask { h = h * m[.ellipsis, .newAxis] }
        let saOut = selfAttention(normSelf(h), mask: mask, cache: cache.sa)
        h = h + saOut
        if hasXattn, let mem = memory, let xa = crossAttention,
           let qNorm = normXattnQuery {
            let qIn = qNorm(h)
            let memIn: MLXArray
            if cache.xa.k == nil {
                memIn = normXattnMemory?.callAsFunction(mem) ?? mem
            } else {
                memIn = mem
            }
            h = h + xa(qIn, memory: memIn, memoryMask: memoryMask, cache: cache.xa)
        }
        let ffOut = posFf(normPosFf(h), mask: mask)
        h = h + ffOut
        if let m = mask { h = h * m[.ellipsis, .newAxis] }
        return h
    }
}
