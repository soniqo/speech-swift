import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Helpers

@inline(__always)
func flashSilu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

func flashUNetGroupNorm(_ channels: Int, eps: Float = 1e-5) -> GroupNorm {
    // See VAE.swift / flashGroupNorm — pytorchCompatible: true is required.
    GroupNorm(groupCount: 32, dimensions: channels, eps: eps, affine: true,
              pytorchCompatible: true)
}

/// Sinusoidal timestep embedding. `t: (B,) → (B, dim)`. Same formula as upstream.
public func flashTimestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000) -> MLXArray {
    let half = dim / 2
    let exponents = MLXArray(0..<Int32(half)).asType(.float32) / Float(half)
    let freqs = exp(-log(MLXArray(maxPeriod)) * exponents)                              // [half]
    let args = t.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    var emb = concatenated([cos(args), sin(args)], axis: -1)
    if dim % 2 != 0 {
        emb = MLX.padded(emb,
                         widths: [.init((0, 0)), .init((0, 1))],
                         value: MLXArray(Float(0)))
    }
    return emb
}

// MARK: - Indexed slot containers (PyTorch nn.Sequential key compatibility)

/// Empty placeholder Module for parameterless slots (SiLU, Dropout). mlx-swift
/// flatten skips it cleanly because it owns no @ModuleInfo / @ParameterInfo.
public final class FlashSRNoop: Module {}

/// Wrap a PyTorch `nn.Sequential`-style group as a flat `[Module]` array so
/// the loader's `update(parameters:)` path receives the slots as a list
/// (matching how mlx-swift's unflatten interprets `<parent>.<i>.*` keys).
///
/// Parameterless slots (SiLU, Dropout) are filled with `FlashSRNoop()` so the
/// indices match the PyTorch ordering. Callers access by absolute index.
public final class FlashSRSeqLayers: Module {
    @ModuleInfo public var layers: [Module]

    public init(_ layers: [Module]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }
}

// MARK: - Up / Down

public final class FlashSRUNetUpsample: Module {
    @ModuleInfo public var conv: Conv2d?
    public let useConv: Bool

    public init(channels: Int, useConv: Bool, outChannels: Int? = nil) {
        let outC = outChannels ?? channels
        self.useConv = useConv
        if useConv {
            self._conv = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: outC,
                kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)))
        } else {
            self._conv = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0); let H = x.dim(1); let W = x.dim(2); let C = x.dim(3)
        let expanded = x.reshaped([B, H, 1, W, 1, C])
        let tiled = MLX.broadcast(expanded, to: [B, H, 2, W, 2, C])
        var out = tiled.reshaped([B, H * 2, W * 2, C])
        if let c = conv { out = c(out) }
        return out
    }
}

public final class FlashSRUNetDownsample: Module {
    @ModuleInfo public var op: Conv2d?

    public init(channels: Int, useConv: Bool, outChannels: Int? = nil) {
        let outC = outChannels ?? channels
        if useConv {
            self._op = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: outC,
                kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1)))
        } else {
            self._op = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let c = op { return c(x) }
        let B = x.dim(0); let H = x.dim(1); let W = x.dim(2); let C = x.dim(3)
        return x.reshaped([B, H / 2, 2, W / 2, 2, C]).mean(axes: [2, 4])
    }
}

// MARK: - ResBlock (with time embedding)

public final class FlashSRResBlock: Module {
    @ModuleInfo(key: "in_layers") public var inLayers: FlashSRSeqLayers
    @ModuleInfo(key: "emb_layers") public var embLayers: FlashSRSeqLayers
    @ModuleInfo(key: "out_layers") public var outLayers: FlashSRSeqLayers
    @ModuleInfo(key: "skip_connection") public var skipConnection: Conv2d?

    public let channels: Int
    public let outChannels: Int

    public init(channels: Int, embChannels: Int, outChannels: Int? = nil, useConvSkip: Bool = false) {
        let out = outChannels ?? channels
        self.channels = channels
        self.outChannels = out
        // in_layers = [GroupNorm@0, SiLU@1, Conv2d@2]
        self._inLayers = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            flashUNetGroupNorm(channels),
            FlashSRNoop(),
            Conv2d(inputChannels: channels, outputChannels: out,
                   kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
        ]), key: "in_layers")
        // emb_layers = [SiLU@0, Linear@1]
        self._embLayers = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            FlashSRNoop(),
            Linear(embChannels, out),
        ]), key: "emb_layers")
        // out_layers = [GroupNorm@0, SiLU@1, Dropout@2, Conv2d@3]
        self._outLayers = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            flashUNetGroupNorm(out),
            FlashSRNoop(),
            FlashSRNoop(),
            Conv2d(inputChannels: out, outputChannels: out,
                   kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
        ]), key: "out_layers")
        if out == channels {
            self._skipConnection = ModuleInfo(wrappedValue: nil, key: "skip_connection")
        } else if useConvSkip {
            self._skipConnection = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: out,
                kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
                key: "skip_connection")
        } else {
            self._skipConnection = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: out,
                kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
                key: "skip_connection")
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        let inNorm = inLayers.layers[0] as! GroupNorm
        let inConv = inLayers.layers[2] as! Conv2d
        let embLin = embLayers.layers[1] as! Linear
        let outNorm = outLayers.layers[0] as! GroupNorm
        let outConv = outLayers.layers[3] as! Conv2d

        var h = inNorm(x)
        h = flashSilu(h)
        h = inConv(h)
        var e = flashSilu(emb)
        e = embLin(e)
        h = h + e.expandedDimensions(axis: 1).expandedDimensions(axis: 1)
        h = outNorm(h)
        h = flashSilu(h)
        h = outConv(h)
        var residual = x
        if let sc = skipConnection { residual = sc(x) }
        return residual + h
    }
}

// MARK: - Attention (cross-attention + feed-forward) for SpatialTransformer

public final class FlashSRCrossAttention: Module {
    @ModuleInfo(key: "to_q") public var toQ: Linear
    @ModuleInfo(key: "to_k") public var toK: Linear
    @ModuleInfo(key: "to_v") public var toV: Linear
    /// `to_out = [Linear@0, Dropout@1]` (Dropout is FlashSRNoop placeholder).
    @ModuleInfo(key: "to_out") public var toOut: FlashSRSeqLayers
    public let heads: Int
    public let dimHead: Int
    public let scale: Float

    public init(queryDim: Int, contextDim: Int? = nil, heads: Int, dimHead: Int) {
        let inner = dimHead * heads
        let ctx = contextDim ?? queryDim
        self.heads = heads
        self.dimHead = dimHead
        self.scale = 1.0 / Float(dimHead).squareRoot()
        self._toQ = ModuleInfo(wrappedValue: Linear(queryDim, inner, bias: false), key: "to_q")
        self._toK = ModuleInfo(wrappedValue: Linear(ctx, inner, bias: false), key: "to_k")
        self._toV = ModuleInfo(wrappedValue: Linear(ctx, inner, bias: false), key: "to_v")
        self._toOut = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            Linear(inner, queryDim),
            FlashSRNoop(),
        ]), key: "to_out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let ctx = context ?? x
        let B = x.dim(0); let N = x.dim(1); let M = ctx.dim(1)
        let q = toQ(x).reshaped([B, N, heads, dimHead]).transposed(0, 2, 1, 3)
        let k = toK(ctx).reshaped([B, M, heads, dimHead]).transposed(0, 2, 1, 3)
        let v = toV(ctx).reshaped([B, M, heads, dimHead]).transposed(0, 2, 1, 3)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, N, heads * dimHead])
        return (toOut.layers[0] as! Linear)(merged)
    }
}

/// `proj(x)` → split into (gate, value), return `value * gelu(gate)`.
public final class FlashSRGEGLU: Module {
    @ModuleInfo public var proj: Linear
    public let dimOut: Int

    public init(dimIn: Int, dimOut: Int) {
        self.dimOut = dimOut
        self._proj = ModuleInfo(wrappedValue: Linear(dimIn, dimOut * 2))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = proj(x)
        let val = projected[.ellipsis, 0..<dimOut]
        let gate = projected[.ellipsis, dimOut..<(2 * dimOut)]
        return val * gelu(gate)
    }
}

public final class FlashSRFeedForward: Module {
    /// `net = [GEGLU@0, Dropout@1, Linear@2]`.
    @ModuleInfo public var net: FlashSRSeqLayers

    public init(dim: Int, mult: Int = 4) {
        let inner = dim * mult
        self._net = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            FlashSRGEGLU(dimIn: dim, dimOut: inner),
            FlashSRNoop(),
            Linear(inner, dim),
        ]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let geglu = net.layers[0] as! FlashSRGEGLU
        let lin = net.layers[2] as! Linear
        return lin(geglu(x))
    }
}

public final class FlashSRBasicTransformerBlock: Module {
    @ModuleInfo public var attn1: FlashSRCrossAttention
    @ModuleInfo public var ff: FlashSRFeedForward
    @ModuleInfo public var attn2: FlashSRCrossAttention
    @ModuleInfo public var norm1: LayerNorm
    @ModuleInfo public var norm2: LayerNorm
    @ModuleInfo public var norm3: LayerNorm

    public init(dim: Int, nHeads: Int, dHead: Int, contextDim: Int? = nil) {
        self._attn1 = ModuleInfo(wrappedValue: FlashSRCrossAttention(
            queryDim: dim, heads: nHeads, dimHead: dHead))
        self._ff = ModuleInfo(wrappedValue: FlashSRFeedForward(dim: dim))
        self._attn2 = ModuleInfo(wrappedValue: FlashSRCrossAttention(
            queryDim: dim, contextDim: contextDim, heads: nHeads, dimHead: dHead))
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim))
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim))
        self._norm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: dim))
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, context: MLXArray? = nil) -> MLXArray {
        var x = xIn
        x = attn1(norm1(x)) + x
        x = attn2(norm2(x), context: context) + x
        x = ff(norm3(x)) + x
        return x
    }
}

public final class FlashSRSpatialTransformer: Module {
    @ModuleInfo public var norm: GroupNorm
    @ModuleInfo(key: "proj_in") public var projIn: Conv2d
    @ModuleInfo(key: "transformer_blocks") public var transformerBlocks: [FlashSRBasicTransformerBlock]
    @ModuleInfo(key: "proj_out") public var projOut: Conv2d

    public let inChannels: Int

    public init(inChannels: Int, nHeads: Int, dHead: Int, depth: Int = 1, contextDim: Int? = nil) {
        let innerDim = nHeads * dHead
        self.inChannels = inChannels
        self._norm = ModuleInfo(wrappedValue: GroupNorm(
            groupCount: 32, dimensions: inChannels, eps: 1e-6, affine: true,
            pytorchCompatible: true))
        self._projIn = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: innerDim,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            key: "proj_in")
        self._transformerBlocks = ModuleInfo(wrappedValue: (0..<depth).map { _ in
            FlashSRBasicTransformerBlock(dim: innerDim, nHeads: nHeads, dHead: dHead, contextDim: contextDim)
        }, key: "transformer_blocks")
        self._projOut = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: innerDim, outputChannels: inChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            key: "proj_out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, context: MLXArray? = nil) -> MLXArray {
        let xIn = x
        let B = x.dim(0); let H = x.dim(1); let W = x.dim(2)
        var h = norm(x)
        h = projIn(h)
        let innerDim = h.dim(3)
        h = h.reshaped([B, H * W, innerDim])
        for block in transformerBlocks {
            h = block(h, context: context)
        }
        h = h.reshaped([B, H, W, innerDim])
        h = projOut(h)
        return h + xIn
    }
}

// MARK: - UNet block sequence (mixed ResBlock + SpatialTransformer + Down/Up)

/// PyTorch `TimestepEmbedSequential` — a list of heterogeneous modules where
/// `ResBlock` takes `(x, emb)` and others take `(x)`. Stored as a flat
/// `[Module]` so the saved keys match `<parent>.<i>.layers.<j>.*` (we insert
/// the `.layers.` segment in the loader to bridge from the bundle's
/// `<parent>.<i>.<j>.*` layout).
public final class FlashSRTSeq: Module {
    @ModuleInfo public var layers: [Module]

    public init(layers: [Module]) {
        self._layers = ModuleInfo(wrappedValue: layers)
        super.init()
    }

    public func callAsFunction(_ xIn: MLXArray, emb: MLXArray) -> MLXArray {
        var x = xIn
        for m in layers {
            x = applyOne(m, x: x, emb: emb)
        }
        return x
    }

    private func applyOne(_ m: Module, x: MLXArray, emb: MLXArray) -> MLXArray {
        if let r = m as? FlashSRResBlock { return r(x, emb: emb) }
        if let c = m as? Conv2d { return c(x) }
        if let u = m as? FlashSRUNetUpsample { return u(x) }
        if let d = m as? FlashSRUNetDownsample { return d(x) }
        if let s = m as? FlashSRSpatialTransformer { return s(x, context: nil) }
        fatalError("FlashSRTSeq: unknown module type \(type(of: m))")
    }
}

// MARK: - AudioSRUnet

public final class FlashSRAudioSRUnet: Module {
    /// `time_embed = [Linear@0, SiLU@1, Linear@2]`.
    @ModuleInfo(key: "time_embed") public var timeEmbed: FlashSRSeqLayers
    @ModuleInfo(key: "input_blocks") public var inputBlocks: [FlashSRTSeq]
    @ModuleInfo(key: "middle_block") public var middleBlock: FlashSRTSeq
    @ModuleInfo(key: "output_blocks") public var outputBlocks: [FlashSRTSeq]
    /// `out = [GroupNorm@0, SiLU@1, Conv2d@2]`.
    @ModuleInfo public var out: FlashSRSeqLayers

    public let cfg: FlashSRUNetConfig

    public init(cfg: FlashSRUNetConfig) {
        self.cfg = cfg
        let timeDim = cfg.modelChannels * 4
        self._timeEmbed = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            Linear(cfg.modelChannels, timeDim),
            FlashSRNoop(),
            Linear(timeDim, timeDim),
        ]), key: "time_embed")

        var inputBlocks: [FlashSRTSeq] = []
        // index 0: initial conv (single module)
        inputBlocks.append(FlashSRTSeq(layers: [Conv2d(
            inputChannels: cfg.inChannels, outputChannels: cfg.modelChannels,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1))]))
        var inputBlockChans: [Int] = [cfg.modelChannels]
        var ch = cfg.modelChannels
        var ds = 1
        let attnSet = Set(cfg.attentionResolutions)
        for (lvl, mult) in cfg.channelMult.enumerated() {
            for _ in 0..<cfg.numResBlocks {
                var layers: [Module] = [
                    FlashSRResBlock(channels: ch, embChannels: timeDim, outChannels: mult * cfg.modelChannels)
                ]
                ch = mult * cfg.modelChannels
                if attnSet.contains(ds) {
                    let numHeads = ch / cfg.numHeadChannels
                    let dHead = cfg.numHeadChannels
                    if cfg.extraSALayer {
                        layers.append(FlashSRSpatialTransformer(
                            inChannels: ch, nHeads: numHeads, dHead: dHead,
                            depth: cfg.transformerDepth, contextDim: nil))
                    }
                    layers.append(FlashSRSpatialTransformer(
                        inChannels: ch, nHeads: numHeads, dHead: dHead,
                        depth: cfg.transformerDepth, contextDim: nil))
                }
                inputBlocks.append(FlashSRTSeq(layers: layers))
                inputBlockChans.append(ch)
            }
            if lvl != cfg.channelMult.count - 1 {
                inputBlocks.append(FlashSRTSeq(layers: [FlashSRUNetDownsample(
                    channels: ch, useConv: true, outChannels: ch)]))
                inputBlockChans.append(ch)
                ds *= 2
            }
        }
        self._inputBlocks = ModuleInfo(wrappedValue: inputBlocks, key: "input_blocks")

        // Middle: ResBlock + (optional extra SA) SpatialTransformer + SpatialTransformer + ResBlock
        var midLayers: [Module] = [FlashSRResBlock(channels: ch, embChannels: timeDim)]
        let midHeads = ch / cfg.numHeadChannels
        let midDHead = cfg.numHeadChannels
        if cfg.extraSALayer {
            midLayers.append(FlashSRSpatialTransformer(
                inChannels: ch, nHeads: midHeads, dHead: midDHead,
                depth: cfg.transformerDepth, contextDim: nil))
        }
        midLayers.append(FlashSRSpatialTransformer(
            inChannels: ch, nHeads: midHeads, dHead: midDHead,
            depth: cfg.transformerDepth, contextDim: nil))
        midLayers.append(FlashSRResBlock(channels: ch, embChannels: timeDim))
        self._middleBlock = ModuleInfo(wrappedValue: FlashSRTSeq(layers: midLayers), key: "middle_block")

        // Output blocks
        var outputBlocks: [FlashSRTSeq] = []
        for (lvl, mult) in cfg.channelMult.enumerated().reversed() {
            for i in 0..<(cfg.numResBlocks + 1) {
                let ich = inputBlockChans.removeLast()
                var layers: [Module] = [
                    FlashSRResBlock(channels: ch + ich, embChannels: timeDim,
                                    outChannels: mult * cfg.modelChannels)
                ]
                ch = mult * cfg.modelChannels
                if attnSet.contains(ds) {
                    let numHeads = ch / cfg.numHeadChannels
                    let dHead = cfg.numHeadChannels
                    if cfg.extraSALayer {
                        layers.append(FlashSRSpatialTransformer(
                            inChannels: ch, nHeads: numHeads, dHead: dHead,
                            depth: cfg.transformerDepth, contextDim: nil))
                    }
                    layers.append(FlashSRSpatialTransformer(
                        inChannels: ch, nHeads: numHeads, dHead: dHead,
                        depth: cfg.transformerDepth, contextDim: nil))
                }
                if lvl != 0 && i == cfg.numResBlocks {
                    layers.append(FlashSRUNetUpsample(channels: ch, useConv: true, outChannels: ch))
                    ds /= 2
                }
                outputBlocks.append(FlashSRTSeq(layers: layers))
            }
        }
        self._outputBlocks = ModuleInfo(wrappedValue: outputBlocks, key: "output_blocks")

        self._out = ModuleInfo(wrappedValue: FlashSRSeqLayers([
            flashUNetGroupNorm(ch),
            FlashSRNoop(),
            Conv2d(inputChannels: ch, outputChannels: cfg.outChannels,
                   kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
        ]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, cond: MLXArray, timesteps: MLXArray) -> MLXArray {
        var h = concatenated([x, cond], axis: -1)                                    // (B, H, W, 32)
        let tEmb = flashTimestepEmbedding(timesteps, dim: cfg.modelChannels)
        let teLin0 = timeEmbed.layers[0] as! Linear
        let teLin2 = timeEmbed.layers[2] as! Linear
        var emb = teLin0(tEmb)
        emb = flashSilu(emb)
        emb = teLin2(emb)
        var hs: [MLXArray] = []
        for block in inputBlocks {
            h = block(h, emb: emb)
            hs.append(h)
        }
        h = middleBlock(h, emb: emb)
        for block in outputBlocks {
            let skip = hs.removeLast()
            h = concatenated([h, skip], axis: -1)
            h = block(h, emb: emb)
        }
        let outNorm = out.layers[0] as! GroupNorm
        let outConv = out.layers[2] as! Conv2d
        h = outNorm(h)
        h = flashSilu(h)
        h = outConv(h)
        return h
    }
}
