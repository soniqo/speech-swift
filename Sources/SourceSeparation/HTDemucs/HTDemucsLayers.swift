import Foundation
import MLX
import MLXNN

// Conv/encoder/decoder building blocks for Hybrid Transformer Demucs,
// translated from the MIT MLX reference (ssmall256/demucs-mlx) and demucs
// hdemucs.py. For htdemucs `norm_starts == depth`, so HEncLayer/HDecLayer run
// with norm disabled (no GroupNorm there); only DConv carries GroupNorm.
//
// Module/parameter keys mirror the PyTorch names so the exported safetensors
// load directly (e.g. `encoder.0.conv.weight`, `encoder.0.dconv.layers.0.0.weight`).
// NOT yet parity-validated — see Phase D.

/// `x * scale` per channel. `scale` shape [C]. `channelLast` selects the
/// broadcast axis: false → channels-first [B, C, T] (DConv); true → channels-
/// last [B, ..., C] (transformer).
final class LayerScale: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray
    let channelLast: Bool
    init(_ channels: Int, channelLast: Bool = false) {
        self.channelLast = channelLast
        self._scale.wrappedValue = MLXArray.zeros([channels])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        channelLast ? x * scale : x * scale.reshaped([1, scale.dim(0), 1])
    }
}

/// GroupNorm over channels-first [B, C, L], stats over (C/G, L) — matches
/// PyTorch `nn.GroupNorm` and the reference `GroupNormNCL`.
final class GroupNormNCL: Module {
    let groups: Int
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(groups: Int, channels: Int, eps: Float = 1e-5) {
        self.groups = groups
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([channels])
        self._bias.wrappedValue = MLXArray.zeros([channels])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), C = x.dim(1), L = x.dim(2)
        let r = x.reshaped([B, groups, C / groups, L])
        let mean = r.mean(axes: [2, 3], keepDims: true)
        let v = r.variance(axes: [2, 3], keepDims: true)
        let nrm = ((r - mean) * MLX.rsqrt(v + eps)).reshaped([B, C, L])
        return nrm * weight.reshaped([1, C, 1]) + bias.reshaped([1, C, 1])
    }
}

/// Frequency embedding: stored `embedding.weight`, output scaled by `scale`.
final class ScaledEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    let scale: Float
    init(_ numEmbeddings: Int, _ dim: Int, scale: Float) {
        self._embedding.wrappedValue = Embedding(embeddingCount: numEmbeddings, dimensions: dim)
        self.scale = scale
    }
    /// `frs`: Int32 indices [F]; returns [F, dim].
    func callAsFunction(_ frs: MLXArray) -> MLXArray { embedding(frs) * scale }
}

/// Residual dilated-conv branch. `depth` blocks, each a 7-slot torch Sequential
/// [conv, GN, gelu, conv1x1, GN, glu, LayerScale] — residual-added. Slots 2 & 5
/// (gelu/glu) are parameterless `Identity` so the loaded list aligns with the
/// exported `dconv.layers.{d}.{i}` keys (two integer-indexed levels → [[Module]]).
/// Operates channels-first [B, C, T].
final class DConv: Module {
    @ModuleInfo(key: "layers") var layers: [[Module]]   // [depth][7]

    init(channels: Int, depth: Int, compress: Int, kernel: Int = 3) {
        let hidden = channels / compress
        self._layers.wrappedValue = (0..<depth).map { d -> [Module] in
            let dilation = 1 << d
            return [
                Conv1d(inputChannels: channels, outputChannels: hidden,
                       kernelSize: kernel, stride: 1,
                       padding: dilation * (kernel / 2), dilation: dilation),
                GroupNormNCL(groups: 1, channels: hidden),
                Identity(),
                Conv1d(inputChannels: hidden, outputChannels: 2 * channels, kernelSize: 1),
                GroupNormNCL(groups: 1, channels: 2 * channels),
                Identity(),
                LayerScale(channels),
            ]
        }
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for b in layers {
            var y = applyConv1dNCL(x, b[0] as! Conv1d)   // [B, hidden, T]
            y = gelu((b[1] as! GroupNormNCL)(y))
            y = applyConv1dNCL(y, b[3] as! Conv1d)        // [B, 2C, T]
            y = (b[4] as! GroupNormNCL)(y)
            y = glu(y, axis: 1)                            // [B, C, T]
            y = (b[6] as! LayerScale)(y)
            x = x + y
        }
        return x
    }
}

/// Hybrid encoder layer. `freq` → Conv2d over (freq, time); else Conv1d over time.
final class HEncLayer: Module {
    let freq: Bool
    let empty: Bool
    let stride: Int
    @ModuleInfo(key: "conv") var conv: Module        // Conv2d (freq) | Conv1d (time)
    @ModuleInfo(key: "rewrite") var rewrite: Module? // Conv2d | Conv1d (1x1 → GLU)
    @ModuleInfo(key: "dconv") var dconv: DConv?

    init(chin: Int, chout: Int, kernelSize: Int, stride: Int, pad: Int,
         freq: Bool, empty: Bool, rewrite: Bool, context: Int, dconv: DConv?) {
        self.freq = freq
        self.empty = empty
        self.stride = stride
        if freq {
            self._conv.wrappedValue = Conv2d(
                inputChannels: chin, outputChannels: chout,
                kernelSize: IntOrPair((kernelSize, 1)), stride: IntOrPair((stride, 1)),
                padding: IntOrPair((pad, 0)))
        } else {
            self._conv.wrappedValue = Conv1d(
                inputChannels: chin, outputChannels: chout,
                kernelSize: kernelSize, stride: stride, padding: pad)
        }
        if empty {
            self._rewrite.wrappedValue = nil
            self._dconv.wrappedValue = nil
            super.init()
            return
        }
        if rewrite {
            let k = 1 + 2 * context
            self._rewrite.wrappedValue = freq
                ? Conv2d(inputChannels: chout, outputChannels: 2 * chout,
                         kernelSize: IntOrPair((k, 1)), stride: IntOrPair(1),
                         padding: IntOrPair((context, 0)))
                : Conv1d(inputChannels: chout, outputChannels: 2 * chout,
                         kernelSize: k, stride: 1, padding: context)
        } else {
            self._rewrite.wrappedValue = nil
        }
        self._dconv.wrappedValue = dconv
        super.init()
    }

    private func applyConv(_ m: Module, _ x: MLXArray) -> MLXArray {
        if let c = m as? Conv2d { return applyConv2dNCHW(x, c) }
        return applyConv1dNCL(x, m as! Conv1d)
    }

    func callAsFunction(_ x0: MLXArray, inject: MLXArray? = nil) -> MLXArray {
        var x = x0
        if !freq && x.ndim == 4 {
            let B = x.dim(0), T = x.dim(3)
            x = x.reshaped([B, -1, T])
        }
        if !freq {
            let le = x.dim(-1)
            if le % stride != 0 {
                x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)),
                                       IntOrPair((0, stride - (le % stride)))])
            }
        }
        var y = applyConv(conv, x)
        if empty { return y }
        if var inj = inject {
            if inj.ndim == 3 && y.ndim == 4 {
                inj = inj.reshaped([inj.dim(0), inj.dim(1), 1, inj.dim(2)])
            }
            y = y + inj
        }
        y = gelu(y)                              // norm1 == Identity for htdemucs
        if let dconv {
            if freq {
                let B = y.dim(0), C = y.dim(1), Fr = y.dim(2), T = y.dim(3)
                y = y.transposed(0, 2, 1, 3).reshaped([B * Fr, C, T])
                y = dconv(y)
                y = y.reshaped([B, Fr, C, T]).transposed(0, 2, 1, 3)
            } else {
                y = dconv(y)
            }
        }
        if let rewrite {
            return glu(applyConv(rewrite, y), axis: 1)   // norm2 == Identity
        }
        return y
    }
}

/// Hybrid decoder layer. Mirrors HEncLayer with transposed convolutions.
final class HDecLayer: Module {
    let freq: Bool
    let empty: Bool
    let last: Bool
    let pad: Int
    let chin: Int
    @ModuleInfo(key: "conv_tr") var convTr: Module       // ConvTransposed2d | 1d
    @ModuleInfo(key: "rewrite") var rewrite: Module?
    @ModuleInfo(key: "dconv") var dconv: DConv?

    init(chin: Int, chout: Int, kernelSize: Int, stride: Int, pad: Int, last: Bool,
         freq: Bool, empty: Bool, rewrite: Bool, context: Int, dconv: DConv?) {
        self.freq = freq
        self.empty = empty
        self.last = last
        self.pad = pad
        self.chin = chin
        if freq {
            self._convTr.wrappedValue = ConvTransposed2d(
                inputChannels: chin, outputChannels: chout,
                kernelSize: IntOrPair((kernelSize, 1)), stride: IntOrPair((stride, 1)))
        } else {
            self._convTr.wrappedValue = ConvTransposed1d(
                inputChannels: chin, outputChannels: chout, kernelSize: kernelSize, stride: stride)
        }
        if empty {
            self._rewrite.wrappedValue = nil
            self._dconv.wrappedValue = nil
            super.init()
            return
        }
        if rewrite {
            let k = 1 + 2 * context
            // Decoder rewrite uses a SQUARE (k, k) conv with (context, context)
            // padding (context_freq=True; scalar kernel_size in demucs).
            self._rewrite.wrappedValue = freq
                ? Conv2d(inputChannels: chin, outputChannels: 2 * chin,
                         kernelSize: IntOrPair(k), stride: IntOrPair(1),
                         padding: IntOrPair(context))
                : Conv1d(inputChannels: chin, outputChannels: 2 * chin,
                         kernelSize: k, stride: 1, padding: context)
        } else {
            self._rewrite.wrappedValue = nil
        }
        self._dconv.wrappedValue = dconv
        super.init()
    }

    private func applyConv(_ m: Module, _ x: MLXArray) -> MLXArray {
        if let c = m as? Conv2d { return applyConv2dNCHW(x, c) }
        return applyConv1dNCL(x, m as! Conv1d)
    }
    private func applyConvTr(_ x: MLXArray) -> MLXArray {
        if let c = convTr as? ConvTransposed2d { return applyConvT2dNCHW(x, c) }
        return applyConvT1dNCL(x, convTr as! ConvTransposed1d)
    }

    /// Returns (output, pre) — `pre` is the input to the transposed conv,
    /// used when the freq/time branches separate in the decoder.
    func callAsFunction(_ x0: MLXArray, skip: MLXArray?, length: Int) -> (MLXArray, MLXArray) {
        var x = x0
        if freq && x.ndim == 3 {
            let B = x.dim(0), T = x.dim(2)
            x = x.reshaped([B, chin, -1, T])
        }
        var y = x
        if !empty {
            if let skip { x = x + skip }
            if let rewrite {
                y = glu(applyConv(rewrite, x), axis: 1)   // norm1 == Identity
            } else {
                y = x
            }
            if let dconv {
                if freq {
                    let B = y.dim(0), C = y.dim(1), Fr = y.dim(2), T = y.dim(3)
                    y = y.transposed(0, 2, 1, 3).reshaped([B * Fr, C, T])
                    y = dconv(y)
                    y = y.reshaped([B, Fr, C, T]).transposed(0, 2, 1, 3)
                } else {
                    y = dconv(y)
                }
            }
        }
        var z = applyConvTr(y)                       // norm2 == Identity
        if freq {
            if pad > 0 { z = z[0..., 0..., pad ..< (z.dim(2) - pad), 0...] }
        } else {
            z = z[0..., 0..., pad ..< (pad + length)]
        }
        if !last { z = gelu(z) }
        return (z, y)
    }
}
