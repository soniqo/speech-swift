import Foundation
import MLX
import MLXNN

// MARK: - Helpers

@inline(__always)
func flashSwish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

func flashGroupNorm(_ channels: Int, eps: Float = 1e-6) -> GroupNorm {
    // `pytorchCompatible: true` is required for parity. MLX's default groups-
    // and-normalizes per-channel only, but PyTorch normalizes per-group across
    // (channels, H, W). Without this flag the encoder/decoder produce wildly
    // different activations and the upsampled audio is full of HF noise.
    GroupNorm(groupCount: 32, dimensions: channels, eps: eps, affine: true,
              pytorchCompatible: true)
}

// MARK: - ResnetBlock (no time embedding)

/// `norm1 → swish → conv1 → norm2 → swish → conv2 + residual`.
/// Residual is either identity (in==out), `conv_shortcut` (3×3), or
/// `nin_shortcut` (1×1).
public final class FlashSRVAEResnetBlock: Module {
    @ModuleInfo public var norm1: GroupNorm
    @ModuleInfo public var conv1: Conv2d
    @ModuleInfo public var norm2: GroupNorm
    @ModuleInfo public var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") public var convShortcut: Conv2d?
    @ModuleInfo(key: "nin_shortcut") public var ninShortcut: Conv2d?

    public let inChannels: Int
    public let outChannels: Int

    public init(inChannels: Int, outChannels: Int? = nil, useConvShortcut: Bool = false) {
        let out = outChannels ?? inChannels
        self.inChannels = inChannels
        self.outChannels = out

        self._norm1 = ModuleInfo(wrappedValue: flashGroupNorm(inChannels))
        self._conv1 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: out,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)))
        self._norm2 = ModuleInfo(wrappedValue: flashGroupNorm(out))
        self._conv2 = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: out, outputChannels: out,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)))

        if inChannels != out {
            if useConvShortcut {
                self._convShortcut = ModuleInfo(wrappedValue: Conv2d(
                    inputChannels: inChannels, outputChannels: out,
                    kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
                    key: "conv_shortcut")
                self._ninShortcut = ModuleInfo(wrappedValue: nil, key: "nin_shortcut")
            } else {
                self._ninShortcut = ModuleInfo(wrappedValue: Conv2d(
                    inputChannels: inChannels, outputChannels: out,
                    kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
                    key: "nin_shortcut")
                self._convShortcut = ModuleInfo(wrappedValue: nil, key: "conv_shortcut")
            }
        } else {
            self._convShortcut = ModuleInfo(wrappedValue: nil, key: "conv_shortcut")
            self._ninShortcut = ModuleInfo(wrappedValue: nil, key: "nin_shortcut")
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        h = flashSwish(h)
        h = conv1(h)
        h = norm2(h)
        h = flashSwish(h)
        h = conv2(h)
        var residual = x
        if let cs = convShortcut { residual = cs(x) }
        else if let nin = ninShortcut { residual = nin(x) }
        return residual + h
    }
}

// MARK: - AttnBlock (self-attention over spatial dims)

public final class FlashSRVAEAttnBlock: Module {
    @ModuleInfo public var norm: GroupNorm
    @ModuleInfo public var q: Conv2d
    @ModuleInfo public var k: Conv2d
    @ModuleInfo public var v: Conv2d
    @ModuleInfo(key: "proj_out") public var projOut: Conv2d
    public let inChannels: Int

    public init(inChannels: Int) {
        self.inChannels = inChannels
        self._norm = ModuleInfo(wrappedValue: flashGroupNorm(inChannels))
        self._q = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: inChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)))
        self._k = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: inChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)))
        self._v = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: inChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)))
        self._projOut = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: inChannels, outputChannels: inChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            key: "proj_out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h0 = norm(x)
        let qOut = q(h0)
        let kOut = k(h0)
        let vOut = v(h0)
        let B = qOut.dim(0); let H = qOut.dim(1); let W = qOut.dim(2); let C = qOut.dim(3)
        let qf = qOut.reshaped([B, H * W, C])
        let kf = kOut.reshaped([B, H * W, C]).transposed(0, 2, 1)
        let vf = vOut.reshaped([B, H * W, C])
        let scale = MLXArray(Float(1.0) / Float(C).squareRoot())
        var attn = matmul(qf, kf) * scale
        attn = softmax(attn, axis: -1)
        let outAttn = matmul(attn, vf).reshaped([B, H, W, C])
        return x + projOut(outAttn)
    }
}

// MARK: - Upsample / Downsample

public final class FlashSRVAEUpsample: Module {
    @ModuleInfo public var conv: Conv2d?
    public let withConv: Bool

    public init(channels: Int, withConv: Bool = true) {
        self.withConv = withConv
        if withConv {
            self._conv = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)))
        } else {
            self._conv = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0); let H = x.dim(1); let W = x.dim(2); let C = x.dim(3)
        // Nearest-neighbor 2× upsample by broadcast-and-reshape.
        let expanded = x.reshaped([B, H, 1, W, 1, C])
        let tiled = MLX.broadcast(expanded, to: [B, H, 2, W, 2, C])
        var out = tiled.reshaped([B, H * 2, W * 2, C])
        if let c = conv { out = c(out) }
        return out
    }
}

public final class FlashSRVAEDownsample: Module {
    @ModuleInfo public var conv: Conv2d?
    public let withConv: Bool

    public init(channels: Int, withConv: Bool = true) {
        self.withConv = withConv
        if withConv {
            self._conv = ModuleInfo(wrappedValue: Conv2d(
                inputChannels: channels, outputChannels: channels,
                kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(0)))
        } else {
            self._conv = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let c = conv {
            // Pad (0, 1) on H and W (asymmetric, right/bottom only) to match PyTorch path.
            let padded = MLX.padded(
                x,
                widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))],
                value: MLXArray(Float(0))
            )
            return c(padded)
        }
        // Avg pool fallback
        let B = x.dim(0); let H = x.dim(1); let W = x.dim(2); let C = x.dim(3)
        return x.reshaped([B, H / 2, 2, W / 2, 2, C]).mean(axes: [2, 4])
    }
}

// MARK: - Encoder / Decoder

public final class FlashSRVAEDownLevel: Module {
    @ModuleInfo public var block: [FlashSRVAEResnetBlock]
    @ModuleInfo public var attn: [FlashSRVAEAttnBlock]
    @ModuleInfo public var downsample: FlashSRVAEDownsample?

    public init(inCh: Int, outCh: Int, numBlocks: Int, downsample: Bool, withAttn: Bool) {
        var blocks: [FlashSRVAEResnetBlock] = []
        var attns: [FlashSRVAEAttnBlock] = []
        var blockIn = inCh
        for _ in 0..<numBlocks {
            blocks.append(FlashSRVAEResnetBlock(inChannels: blockIn, outChannels: outCh))
            blockIn = outCh
            if withAttn {
                attns.append(FlashSRVAEAttnBlock(inChannels: outCh))
            }
        }
        self._block = ModuleInfo(wrappedValue: blocks)
        self._attn = ModuleInfo(wrappedValue: attns)
        if downsample {
            self._downsample = ModuleInfo(wrappedValue: FlashSRVAEDownsample(channels: outCh, withConv: true))
        } else {
            self._downsample = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }
}

public final class FlashSRVAEUpLevel: Module {
    @ModuleInfo public var block: [FlashSRVAEResnetBlock]
    @ModuleInfo public var attn: [FlashSRVAEAttnBlock]
    @ModuleInfo public var upsample: FlashSRVAEUpsample?

    public init(inCh: Int, outCh: Int, numBlocks: Int, upsample: Bool, withAttn: Bool) {
        var blocks: [FlashSRVAEResnetBlock] = []
        var attns: [FlashSRVAEAttnBlock] = []
        var blockIn = inCh
        for _ in 0..<(numBlocks + 1) {
            blocks.append(FlashSRVAEResnetBlock(inChannels: blockIn, outChannels: outCh))
            blockIn = outCh
            if withAttn {
                attns.append(FlashSRVAEAttnBlock(inChannels: outCh))
            }
        }
        self._block = ModuleInfo(wrappedValue: blocks)
        self._attn = ModuleInfo(wrappedValue: attns)
        if upsample {
            self._upsample = ModuleInfo(wrappedValue: FlashSRVAEUpsample(channels: outCh, withConv: true))
        } else {
            self._upsample = ModuleInfo(wrappedValue: nil)
        }
        super.init()
    }
}

/// Container for the middle (bottleneck) block of the VAE encoder/decoder.
/// Matches the PyTorch layout where `mid.block_1`, `mid.attn_1`, `mid.block_2`
/// are nested under a `mid` submodule.
public final class FlashSRVAEMid: Module {
    @ModuleInfo public var block_1: FlashSRVAEResnetBlock
    @ModuleInfo public var attn_1: FlashSRVAEAttnBlock
    @ModuleInfo public var block_2: FlashSRVAEResnetBlock

    public init(channels: Int) {
        self._block_1 = ModuleInfo(wrappedValue: FlashSRVAEResnetBlock(inChannels: channels, outChannels: channels))
        self._attn_1 = ModuleInfo(wrappedValue: FlashSRVAEAttnBlock(inChannels: channels))
        self._block_2 = ModuleInfo(wrappedValue: FlashSRVAEResnetBlock(inChannels: channels, outChannels: channels))
        super.init()
    }
}

public final class FlashSRVAEEncoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo public var down: [FlashSRVAEDownLevel]
    @ModuleInfo public var mid: FlashSRVAEMid
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public let cfg: FlashSRVAEConfig

    public init(cfg: FlashSRVAEConfig) {
        self.cfg = cfg
        let numLevels = cfg.chMult.count
        self._convIn = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: cfg.inChannels, outputChannels: cfg.ch,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            key: "conv_in")

        let inChMult = [1] + cfg.chMult
        var levels: [FlashSRVAEDownLevel] = []
        var currRes = cfg.resolution
        for i in 0..<numLevels {
            let inC = cfg.ch * inChMult[i]
            let outC = cfg.ch * cfg.chMult[i]
            levels.append(FlashSRVAEDownLevel(
                inCh: inC, outCh: outC,
                numBlocks: cfg.numResBlocks,
                downsample: (i != numLevels - 1),
                withAttn: cfg.attnResolutions.contains(currRes)))
            if i != numLevels - 1 { currRes /= 2 }
        }
        self._down = ModuleInfo(wrappedValue: levels)

        let midCh = cfg.ch * cfg.chMult[numLevels - 1]
        self._mid = ModuleInfo(wrappedValue: FlashSRVAEMid(channels: midCh))

        self._normOut = ModuleInfo(wrappedValue: flashGroupNorm(midCh), key: "norm_out")
        let outZ = cfg.doubleZ ? 2 * cfg.zChannels : cfg.zChannels
        self._convOut = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: midCh, outputChannels: outZ,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            key: "conv_out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for lvl in down {
            for (i, blk) in lvl.block.enumerated() {
                h = blk(h)
                if !lvl.attn.isEmpty { h = lvl.attn[i](h) }
            }
            if let ds = lvl.downsample { h = ds(h) }
        }
        h = mid.block_1(h)
        h = mid.attn_1(h)
        h = mid.block_2(h)
        h = normOut(h)
        h = flashSwish(h)
        h = convOut(h)
        return h
    }
}

public final class FlashSRVAEDecoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo public var mid: FlashSRVAEMid
    @ModuleInfo public var up: [FlashSRVAEUpLevel]
    @ModuleInfo(key: "norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d

    public let cfg: FlashSRVAEConfig

    public init(cfg: FlashSRVAEConfig) {
        self.cfg = cfg
        let numLevels = cfg.chMult.count
        var blockIn = cfg.ch * cfg.chMult[numLevels - 1]

        self._convIn = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: cfg.zChannels, outputChannels: blockIn,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            key: "conv_in")

        self._mid = ModuleInfo(wrappedValue: FlashSRVAEMid(channels: blockIn))

        // Build up-levels in the same index order as PyTorch (i=0 is the outermost level
        // — highest resolution — which is built LAST in the loop but inserted at index 0).
        var levels: [FlashSRVAEUpLevel] = []
        var currRes = cfg.resolution / (1 << (numLevels - 1))
        for i in stride(from: numLevels - 1, through: 0, by: -1) {
            let outC = cfg.ch * cfg.chMult[i]
            let lvl = FlashSRVAEUpLevel(
                inCh: blockIn, outCh: outC,
                numBlocks: cfg.numResBlocks,
                upsample: (i != 0),
                withAttn: cfg.attnResolutions.contains(currRes))
            levels.insert(lvl, at: 0)
            blockIn = outC
            if i != 0 { currRes *= 2 }
        }
        self._up = ModuleInfo(wrappedValue: levels)

        self._normOut = ModuleInfo(wrappedValue: flashGroupNorm(blockIn), key: "norm_out")
        self._convOut = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: blockIn, outputChannels: cfg.outCh,
            kernelSize: IntOrPair(3), stride: IntOrPair(1), padding: IntOrPair(1)),
            key: "conv_out")
        super.init()
    }

    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = convIn(z)
        h = mid.block_1(h)
        h = mid.attn_1(h)
        h = mid.block_2(h)
        for lvl in up.reversed() {
            for (i, blk) in lvl.block.enumerated() {
                h = blk(h)
                if !lvl.attn.isEmpty { h = lvl.attn[i](h) }
            }
            if let us = lvl.upsample { h = us(h) }
        }
        h = normOut(h)
        h = flashSwish(h)
        h = convOut(h)
        return h
    }
}

// MARK: - AutoencoderKL

public struct FlashSRVAEPosterior {
    public let mean: MLXArray
    public let logvar: MLXArray
    public let std: MLXArray

    public init(params: MLXArray) {
        let c = params.dim(-1) / 2
        let mean = params[.ellipsis, 0..<c]
        let logvar = MLX.clip(params[.ellipsis, c..<(2 * c)], min: -30.0, max: 20.0)
        self.mean = mean
        self.logvar = logvar
        self.std = exp(0.5 * logvar)
    }
}

public final class FlashSRAutoencoderKL: Module {
    @ModuleInfo public var encoder: FlashSRVAEEncoder
    @ModuleInfo public var decoder: FlashSRVAEDecoder
    @ModuleInfo(key: "quant_conv") public var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") public var postQuantConv: Conv2d

    public let cfg: FlashSRVAEConfig

    public init(cfg: FlashSRVAEConfig) {
        self.cfg = cfg
        self._encoder = ModuleInfo(wrappedValue: FlashSRVAEEncoder(cfg: cfg))
        self._decoder = ModuleInfo(wrappedValue: FlashSRVAEDecoder(cfg: cfg))
        self._quantConv = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: 2 * cfg.zChannels, outputChannels: 2 * cfg.embedDim,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            key: "quant_conv")
        self._postQuantConv = ModuleInfo(wrappedValue: Conv2d(
            inputChannels: cfg.embedDim, outputChannels: cfg.zChannels,
            kernelSize: IntOrPair(1), stride: IntOrPair(1), padding: IntOrPair(0)),
            key: "post_quant_conv")
        super.init()
    }

    public func encode(_ x: MLXArray) -> FlashSRVAEPosterior {
        let h = encoder(x)
        let moments = quantConv(h)
        return FlashSRVAEPosterior(params: moments)
    }

    public func decode(_ z: MLXArray) -> MLXArray {
        return decoder(postQuantConv(z))
    }
}
