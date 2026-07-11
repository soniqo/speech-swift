import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Activation helpers

/// Mish activation: `x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`.
private func mish(_ x: MLXArray) -> MLXArray {
    x * tanh(log1p(exp(x)))
}

// MARK: - Precomputed-constant holder

/// Non-Module holder so precomputed timestep / noise arrays aren't reflected as
/// loadable parameters (mirrors the `T3Freqs` trick in `T3.swift`).
final class CFMConst {
    let array: MLXArray
    init(_ a: MLXArray) { array = a }
}

// MARK: - SinusoidalPosEmb

/// Parameter-free sinusoidal timestep embedding.
///
/// Maps a scalar (or `[B]`) timestep to a `dim`-dimensional embedding. Matches the
/// Python `SinusoidalPosEmb` exactly: `scale=1000`, `log(10000)/(half-1)` spacing,
/// concatenation `[sin, cos]`.
final class CFMSinusoidalPosEmb {
    let dim: Int

    init(dim: Int) {
        precondition(dim % 2 == 0, "SinusoidalPosEmb requires even dim")
        self.dim = dim
    }

    /// - Parameter t: `[B]` float timestep values.
    /// - Returns: `[B, dim]` embedding.
    func callAsFunction(_ t: MLXArray, scale: Float = 1000) -> MLXArray {
        var t = t
        if t.ndim < 1 {
            t = t.expandedDimensions(axis: 0)
        }
        let halfDim = dim / 2
        let factor = Float(log(10000.0)) / Float(halfDim - 1)
        let emb = exp(MLXArray(0 ..< Int32(halfDim)).asType(.float32) * (-factor))  // [halfDim]
        // scale * t[:, None] * emb[None, :]
        let angles = MLXArray(scale) * t.expandedDimensions(axis: 1) * emb.expandedDimensions(axis: 0)
        return concatenated([sin(angles), cos(angles)], axis: -1)  // [B, dim]
    }
}

// MARK: - TimestepEmbedding (time_mlp)

/// MLP projecting the sinusoidal timestep embedding: `linear_1 -> SiLU -> linear_2`.
final class CFMTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inChannels: Int, timeEmbedDim: Int) {
        self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
        self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = linear1(x)
        h = silu(h)
        h = linear2(h)
        return h
    }
}

// MARK: - Causal convolutions / blocks

/// Causal 1D convolution (stride=1). Left-pads by `kernelSize - 1` so the output
/// depends only on current and past positions. Operates on NCL `[B, C, T]` input
/// (matching the Python decoder's channel-first convention), transposing to NLC
/// internally for `MLXNN.Conv1d`.
final class CFMCausalConv1d: Module {
    let causalPadding: Int
    @ModuleInfo var conv: MLXNN.Conv1d

    init(inChannels: Int, outChannels: Int, kernelSize: Int) {
        self.causalPadding = kernelSize - 1
        self._conv.wrappedValue = MLXNN.Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: 1, padding: 0, bias: true)
        super.init()
    }

    /// - Parameter x: `[B, C, T]` (NCL). - Returns: `[B, C, T]` (NCL).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.swappedAxes(1, 2)  // [B, T, C]
        h = padded(h, widths: [.init((low: 0, high: 0)),
                               .init((low: causalPadding, high: 0)),
                               .init((low: 0, high: 0))])
        h = conv(h)
        return h.swappedAxes(1, 2)  // [B, C, T]
    }
}

/// Causal conv block: `CausalConv1d -> LayerNorm -> Mish`, masked at the boundaries.
/// The norm is a `LayerNorm` over the channel dim (matches the causal-path weights
/// `block*.norm.{weight,bias}` of shape `[dim_out]`).
final class CFMCausalBlock1D: Module {
    @ModuleInfo var conv: CFMCausalConv1d
    @ModuleInfo var norm: LayerNorm

    init(dim: Int, dimOut: Int) {
        self._conv.wrappedValue = CFMCausalConv1d(inChannels: dim, outChannels: dimOut, kernelSize: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dimOut, eps: 1e-5, affine: true)
        super.init()
    }

    /// - Parameters: x `[B, C, T]`, mask `[B, 1, T]`. - Returns: `[B, C, T]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        var out = conv(x * mask)               // [B, C, T]
        out = out.swappedAxes(1, 2)            // [B, T, C]
        out = norm(out)                        // LayerNorm over channels
        out = out.swappedAxes(1, 2)            // [B, C, T]
        out = mish(out)
        return out * mask
    }
}

/// Causal ResNet block with timestep conditioning.
///
/// `block1 -> (+ mlp_linear(mish(t_emb))) -> block2 -> (+ res_conv(x))`.
/// `mlp_linear` projects the time embedding; `res_conv` is a 1x1 conv shortcut.
final class CFMResnetBlock1D: Module {
    @ModuleInfo(key: "mlp_linear") var mlpLinear: Linear
    @ModuleInfo var block1: CFMCausalBlock1D
    @ModuleInfo var block2: CFMCausalBlock1D
    @ModuleInfo(key: "res_conv") var resConv: MLXNN.Conv1d

    init(dim: Int, dimOut: Int, timeEmbDim: Int) {
        self._mlpLinear.wrappedValue = Linear(timeEmbDim, dimOut, bias: true)
        self._block1.wrappedValue = CFMCausalBlock1D(dim: dim, dimOut: dimOut)
        self._block2.wrappedValue = CFMCausalBlock1D(dim: dimOut, dimOut: dimOut)
        // 1x1 conv shortcut: in NCL we still go through NLC for MLXNN.Conv1d.
        self._resConv.wrappedValue = MLXNN.Conv1d(
            inputChannels: dim, outputChannels: dimOut, kernelSize: 1, bias: true)
        super.init()
    }

    /// - Parameters: x `[B, C, T]`, mask `[B, 1, T]`, timeEmb `[B, timeEmbDim]`.
    /// - Returns: `[B, dimOut, T]`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
        var h = block1(x, mask: mask)
        // h += mlp(time_emb) where mlp = Mish() -> Linear()
        let tProj = mlpLinear(mish(timeEmb))                    // [B, dimOut]
        h = h + tProj.expandedDimensions(axis: -1)              // broadcast over T: [B, dimOut, 1]
        h = block2(h, mask: mask)
        // res_conv shortcut on the (masked) input.
        var res = (x * mask).swappedAxes(1, 2)                 // [B, T, C]
        res = resConv(res)
        res = res.swappedAxes(1, 2)                            // [B, dimOut, T]
        return h + res
    }
}

// MARK: - Transformer (BasicTransformerBlock)

/// Diffusers-style attention used by the Matcha transformer blocks.
///
/// `inner_dim = heads * dim_head` (e.g. 8*64=512). q/k/v project `query_dim -> inner_dim`
/// (no bias); out projects `inner_dim -> query_dim` (with bias). Weight keys:
/// `query_proj`, `key_proj`, `value_proj`, `out_proj`.
final class CFMAttention: Module {
    let heads: Int
    let dimHead: Int
    let innerDim: Int
    let scale: Float

    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(queryDim: Int, heads: Int, dimHead: Int) {
        self.heads = heads
        self.dimHead = dimHead
        self.innerDim = heads * dimHead
        self.scale = pow(Float(dimHead), -0.5)
        self._queryProj.wrappedValue = Linear(queryDim, innerDim, bias: false)
        self._keyProj.wrappedValue = Linear(queryDim, innerDim, bias: false)
        self._valueProj.wrappedValue = Linear(queryDim, innerDim, bias: false)
        self._outProj.wrappedValue = Linear(innerDim, queryDim, bias: true)
        super.init()
    }

    /// - Parameters:
    ///   - hiddenStates: `[B, T, queryDim]`
    ///   - attentionBias: additive bias `[B, T, T]` (0 = attend, large-negative = mask), or nil.
    /// - Returns: `[B, T, queryDim]`
    func callAsFunction(_ hiddenStates: MLXArray, attentionBias: MLXArray?) -> MLXArray {
        let B = hiddenStates.dim(0)
        let T = hiddenStates.dim(1)

        var q = queryProj(hiddenStates)
        var k = keyProj(hiddenStates)
        var v = valueProj(hiddenStates)

        q = q.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)  // [B, H, T, d]
        k = k.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)
        v = v.reshaped(B, T, heads, dimHead).transposed(0, 2, 1, 3)

        // Additive bias path: bias is [B, T_q, T_kv] -> [B, 1, T_q, T_kv].
        let mask4: MLXArray? = attentionBias.map { $0.expandedDimensions(axis: 1) }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask4)

        let merged = out.transposed(0, 2, 1, 3).reshaped(B, T, innerDim)
        return outProj(merged)
    }
}

/// GELU feedforward: `layers[0] -> GELU -> layers[1]`.
/// `layers` is a dot-indexed module list, matching checkpoint keys
/// `ff.layers.0.{weight,bias}` and `ff.layers.1.{weight,bias}`.
final class CFMFeedForward: Module {
    @ModuleInfo var layers: [Linear]

    init(dim: Int, innerDim: Int) {
        self._layers.wrappedValue = [
            Linear(dim, innerDim, bias: true),   // 256 -> 1024
            Linear(innerDim, dim, bias: true),   // 1024 -> 256
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = layers[0](x)
        h = gelu(h)
        h = layers[1](h)
        return h
    }
}

/// Basic transformer block: pre-norm self-attention + pre-norm feedforward, both
/// with residuals. `norm1`/`norm3` are affine `LayerNorm`. No cross-attention.
final class CFMTransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm3: LayerNorm
    @ModuleInfo var attn: CFMAttention
    @ModuleInfo var ff: CFMFeedForward

    init(dim: Int, numHeads: Int, attentionHeadDim: Int) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5, affine: true)
        self._norm3.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5, affine: true)
        self._attn.wrappedValue = CFMAttention(
            queryDim: dim, heads: numHeads, dimHead: attentionHeadDim)
        self._ff.wrappedValue = CFMFeedForward(dim: dim, innerDim: dim * 4)
        super.init()
    }

    /// - Parameters: hiddenStates `[B, T, dim]`, attentionBias `[B, T, T]` or nil.
    /// - Returns: `[B, T, dim]`.
    func callAsFunction(_ hiddenStates: MLXArray, attentionBias: MLXArray?) -> MLXArray {
        var h = hiddenStates
        h = h + attn(norm1(h), attentionBias: attentionBias)
        h = h + ff(norm3(h))
        return h
    }
}

// MARK: - U-Net blocks
//
// The Python source builds the indexed sub-blocks with
// `setattr(self, f"transformer_{i}", ...)` and `setattr(self, f"mid_blocks_{i}", ...)`,
// producing UNDERSCORE-indexed keys (`transformer_0`, `mid_blocks_11`) directly under
// the parent — not the dot-indexed keys (`transformer.0`) that an MLX `[Module]` array
// emits. So every indexed child is declared with an explicit `@ModuleInfo(key:)`.

/// One down block: a ResNet + 4 transformer blocks + a (causal-conv) downsample.
/// For the single-channel-level Chatterbox decoder the downsample is a stride-1
/// causal conv (no length change).
final class CFMDownBlock: Module {
    @ModuleInfo var resnet: CFMResnetBlock1D
    @ModuleInfo(key: "transformer_0") var transformer0: CFMTransformerBlock
    @ModuleInfo(key: "transformer_1") var transformer1: CFMTransformerBlock
    @ModuleInfo(key: "transformer_2") var transformer2: CFMTransformerBlock
    @ModuleInfo(key: "transformer_3") var transformer3: CFMTransformerBlock
    @ModuleInfo var downsample: CFMCausalConv1d

    init(inChannels: Int, outChannels: Int, timeEmbDim: Int,
         numHeads: Int, attentionHeadDim: Int) {
        self._resnet.wrappedValue = CFMResnetBlock1D(
            dim: inChannels, dimOut: outChannels, timeEmbDim: timeEmbDim)
        func mk() -> CFMTransformerBlock {
            CFMTransformerBlock(dim: outChannels, numHeads: numHeads, attentionHeadDim: attentionHeadDim)
        }
        self._transformer0.wrappedValue = mk(); self._transformer1.wrappedValue = mk()
        self._transformer2.wrappedValue = mk(); self._transformer3.wrappedValue = mk()
        self._downsample.wrappedValue = CFMCausalConv1d(
            inChannels: outChannels, outChannels: outChannels, kernelSize: 3)
        super.init()
    }

    var transformers: [CFMTransformerBlock] { [transformer0, transformer1, transformer2, transformer3] }
}

/// One mid block: a ResNet + 4 transformer blocks (no resampling).
final class CFMMidBlock: Module {
    @ModuleInfo var resnet: CFMResnetBlock1D
    @ModuleInfo(key: "transformer_0") var transformer0: CFMTransformerBlock
    @ModuleInfo(key: "transformer_1") var transformer1: CFMTransformerBlock
    @ModuleInfo(key: "transformer_2") var transformer2: CFMTransformerBlock
    @ModuleInfo(key: "transformer_3") var transformer3: CFMTransformerBlock

    init(channels: Int, timeEmbDim: Int, numHeads: Int, attentionHeadDim: Int) {
        self._resnet.wrappedValue = CFMResnetBlock1D(
            dim: channels, dimOut: channels, timeEmbDim: timeEmbDim)
        func mk() -> CFMTransformerBlock {
            CFMTransformerBlock(dim: channels, numHeads: numHeads, attentionHeadDim: attentionHeadDim)
        }
        self._transformer0.wrappedValue = mk(); self._transformer1.wrappedValue = mk()
        self._transformer2.wrappedValue = mk(); self._transformer3.wrappedValue = mk()
        super.init()
    }

    var transformers: [CFMTransformerBlock] { [transformer0, transformer1, transformer2, transformer3] }
}

/// One up block: a ResNet (consuming the concat of x + skip) + 4 transformer blocks
/// + a (causal-conv) upsample. For the single-channel-level decoder the upsample is
/// a stride-1 causal conv (no length change).
final class CFMUpBlock: Module {
    @ModuleInfo var resnet: CFMResnetBlock1D
    @ModuleInfo(key: "transformer_0") var transformer0: CFMTransformerBlock
    @ModuleInfo(key: "transformer_1") var transformer1: CFMTransformerBlock
    @ModuleInfo(key: "transformer_2") var transformer2: CFMTransformerBlock
    @ModuleInfo(key: "transformer_3") var transformer3: CFMTransformerBlock
    @ModuleInfo var upsample: CFMCausalConv1d

    init(inChannels: Int, outChannels: Int, timeEmbDim: Int,
         numHeads: Int, attentionHeadDim: Int) {
        self._resnet.wrappedValue = CFMResnetBlock1D(
            dim: inChannels, dimOut: outChannels, timeEmbDim: timeEmbDim)
        func mk() -> CFMTransformerBlock {
            CFMTransformerBlock(dim: outChannels, numHeads: numHeads, attentionHeadDim: attentionHeadDim)
        }
        self._transformer0.wrappedValue = mk(); self._transformer1.wrappedValue = mk()
        self._transformer2.wrappedValue = mk(); self._transformer3.wrappedValue = mk()
        self._upsample.wrappedValue = CFMCausalConv1d(
            inChannels: outChannels, outChannels: outChannels, kernelSize: 3)
        super.init()
    }

    var transformers: [CFMTransformerBlock] { [transformer0, transformer1, transformer2, transformer3] }
}

// MARK: - ConditionalDecoder (the U-Net estimator)

/// Matcha-style U-Net velocity estimator for the flow-matching ODE.
///
/// The estimator predicts `dphi/dt` given the noised mel `x`, the encoder output
/// `mu`, the timestep `t`, and the (broadcast) speaker / prompt conditioning. The
/// Chatterbox decoder uses a single channel level (`channels=[256]`), so neither
/// the down nor up "resamplers" change the sequence length — they are stride-1
/// causal convolutions. The skip connection still concatenates the single
/// down-block output into the up block (256 + 256 = 512 input channels).
///
/// Weight keys (after stripping `s3gen.flow.`): `decoder.estimator.*`.
final class CFMConditionalDecoder: Module {
    let inChannels: Int
    let outChannels: Int
    let staticChunkSize: Int

    // Parameter-free sinusoidal timestep embedding (NOT a loadable param).
    let timeEmbeddings: CFMSinusoidalPosEmb

    @ModuleInfo(key: "time_mlp") var timeMlp: CFMTimestepEmbedding
    @ModuleInfo(key: "down_blocks_0") var downBlock0: CFMDownBlock
    @ModuleInfo(key: "mid_blocks_0") var midBlock0: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_1") var midBlock1: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_2") var midBlock2: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_3") var midBlock3: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_4") var midBlock4: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_5") var midBlock5: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_6") var midBlock6: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_7") var midBlock7: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_8") var midBlock8: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_9") var midBlock9: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_10") var midBlock10: CFMMidBlock
    @ModuleInfo(key: "mid_blocks_11") var midBlock11: CFMMidBlock
    @ModuleInfo(key: "up_blocks_0") var upBlock0: CFMUpBlock
    @ModuleInfo(key: "final_block") var finalBlock: CFMCausalBlock1D
    @ModuleInfo(key: "final_proj") var finalProj: MLXNN.Conv1d

    var midBlocks: [CFMMidBlock] {
        [midBlock0, midBlock1, midBlock2, midBlock3, midBlock4, midBlock5,
         midBlock6, midBlock7, midBlock8, midBlock9, midBlock10, midBlock11]
    }

    init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        channels: [Int] = [256],
        attentionHeadDim: Int = 64,
        numHeads: Int = 8,
        staticChunkSize: Int = 50
    ) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.staticChunkSize = staticChunkSize

        self.timeEmbeddings = CFMSinusoidalPosEmb(dim: inChannels)
        let timeEmbedDim = channels[0] * 4
        let ch = channels[0]

        self._timeMlp.wrappedValue = CFMTimestepEmbedding(
            inChannels: inChannels, timeEmbedDim: timeEmbedDim)

        // Single down block: in_channels(320) -> channels[0](256).
        self._downBlock0.wrappedValue = CFMDownBlock(
            inChannels: inChannels, outChannels: ch, timeEmbDim: timeEmbedDim,
            numHeads: numHeads, attentionHeadDim: attentionHeadDim)

        // 12 mid blocks: channels[0] -> channels[0].
        func mkMid() -> CFMMidBlock {
            CFMMidBlock(channels: ch, timeEmbDim: timeEmbedDim,
                        numHeads: numHeads, attentionHeadDim: attentionHeadDim)
        }
        self._midBlock0.wrappedValue = mkMid(); self._midBlock1.wrappedValue = mkMid()
        self._midBlock2.wrappedValue = mkMid(); self._midBlock3.wrappedValue = mkMid()
        self._midBlock4.wrappedValue = mkMid(); self._midBlock5.wrappedValue = mkMid()
        self._midBlock6.wrappedValue = mkMid(); self._midBlock7.wrappedValue = mkMid()
        self._midBlock8.wrappedValue = mkMid(); self._midBlock9.wrappedValue = mkMid()
        self._midBlock10.wrappedValue = mkMid(); self._midBlock11.wrappedValue = mkMid()

        // Single up block: input = channels[0]*2 (x ++ skip) -> channels[0].
        self._upBlock0.wrappedValue = CFMUpBlock(
            inChannels: ch * 2, outChannels: ch, timeEmbDim: timeEmbedDim,
            numHeads: numHeads, attentionHeadDim: attentionHeadDim)

        self._finalBlock.wrappedValue = CFMCausalBlock1D(dim: ch, dimOut: ch)
        self._finalProj.wrappedValue = MLXNN.Conv1d(
            inputChannels: ch, outputChannels: outChannels, kernelSize: 1, bias: true)

        super.init()
    }

    /// Build a full-attention additive bias from a `[B, 1, T]` float mask.
    /// Valid positions (mask==1) -> 0, padding -> -1e10. Result: `[B, T, T]`.
    private func attentionBias(fromMask mask: MLXArray, T: Int) -> MLXArray {
        // mask: [B, 1, T] -> keys valid along last axis. Broadcast to [B, T, T].
        let keysValid = mask.reshaped(mask.dim(0), 1, T)              // [B, 1, T]
        let bias = (MLXArray(1.0) - keysValid) * MLXArray(-1.0e10)    // [B, 1, T]
        return broadcast(bias, to: [mask.dim(0), T, T])              // [B, T, T]
    }

    /// Velocity estimator forward pass.
    ///
    /// - Parameters:
    ///   - x: `[B, in_feats, T]` current ODE state (noised mel).
    ///   - mask: `[B, 1, T]` validity mask (1 = valid).
    ///   - mu: `[B, in_feats, T]` encoder output conditioning.
    ///   - t: `[B]` timestep in [0, 1].
    ///   - spks: `[B, spk_emb_dim]` projected speaker embedding (broadcast over T).
    ///   - cond: `[B, in_feats, T]` prompt-mel conditioning.
    /// - Returns: `[B, out_feats, T]` predicted velocity.
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, mu: MLXArray, t: MLXArray,
        spks: MLXArray?, cond: MLXArray?
    ) -> MLXArray {
        // Timestep embedding.
        var tEmb = timeEmbeddings(t)   // [B, in_channels]
        tEmb = timeMlp(tEmb)           // [B, time_embed_dim]

        // Concatenate conditioning along the channel axis (NCL).
        var h = concatenated([x, mu], axis: 1)
        if let spks = spks {
            let spksExpanded = broadcast(
                spks.expandedDimensions(axis: -1),
                to: [spks.dim(0), spks.dim(1), h.dim(2)])
            h = concatenated([h, spksExpanded], axis: 1)
        }
        if let cond = cond {
            h = concatenated([h, cond], axis: 1)
        }

        let maskDown = mask

        // ---- Down block ----
        h = downBlock0.resnet(h, mask: maskDown, timeEmb: tEmb)
        var hT = h.swappedAxes(1, 2)                            // [B, T, C]
        var bias = attentionBias(fromMask: maskDown, T: hT.dim(1))
        for tb in downBlock0.transformers {
            hT = tb(hT, attentionBias: bias)
        }
        h = hT.swappedAxes(1, 2)                                // [B, C, T]
        let skip = h                                            // single down-block hidden
        h = downBlock0.downsample(h * maskDown)

        let maskMid = maskDown

        // ---- Mid blocks ----
        for mid in midBlocks {
            h = mid.resnet(h, mask: maskMid, timeEmb: tEmb)
            hT = h.swappedAxes(1, 2)
            bias = attentionBias(fromMask: maskMid, T: hT.dim(1))
            for tb in mid.transformers {
                hT = tb(hT, attentionBias: bias)
            }
            h = hT.swappedAxes(1, 2)
        }

        // ---- Up block ----
        let maskUp = maskDown
        // Truncate x to match the skip length, then concat along channels.
        let skipLen = skip.dim(2)
        h = concatenated([h[0..., 0..., 0 ..< skipLen], skip], axis: 1)   // [B, 2C, T]
        h = upBlock0.resnet(h, mask: maskUp, timeEmb: tEmb)
        hT = h.swappedAxes(1, 2)
        bias = attentionBias(fromMask: maskUp, T: hT.dim(1))
        for tb in upBlock0.transformers {
            hT = tb(hT, attentionBias: bias)
        }
        h = hT.swappedAxes(1, 2)
        h = upBlock0.upsample(h * maskUp)

        // ---- Final layers ----
        h = finalBlock(h, mask: maskUp)
        var proj = (h * maskUp).swappedAxes(1, 2)               // [B, T, C]
        proj = finalProj(proj)                                  // [B, T, out_feats]
        let output = proj.swappedAxes(1, 2)                     // [B, out_feats, T]
        return output * mask
    }
}

// MARK: - MatchaCFM (estimator + ODE solver)

/// Causal Conditional Flow Matching for the Chatterbox S3Gen flow decoder.
///
/// Wraps the Matcha U-Net velocity estimator (`decoder.estimator.*`) and the
/// flow-matching ODE solver. `solve`/`inference` integrate the velocity field from
/// pure noise (`t=0`) to the target mel (`t=1`) using a fixed-step Euler method with
/// classifier-free guidance (CFG), conditioned on the encoder output `mu`, the
/// projected speaker embedding `spks`, and the prompt-mel `cond`.
///
/// Weight keys (after stripping `s3gen.flow.`): all under `decoder.estimator.*`.
public final class MatchaCFM: Module {
    /// Mel channel count (the estimator output dim).
    public static let melChannels = 80

    public let nFeats: Int
    public let spkEmbDim: Int
    /// CFG strength for inference. `v = (1+cfg)*v_cond - cfg*v_uncond`.
    public let inferenceCfgRate: Float
    /// Default ODE step count (from `CausalMaskedDiffWithXvec.n_timesteps`).
    public let defaultNTimesteps: Int

    @ModuleInfo(key: "estimator") var estimator: CFMConditionalDecoder

    /// Pre-seeded noise buffer, matching the Python `CausalConditionalCFM.rand_noise`
    /// (`mx.random.seed(0); mx.random.normal((1, 80, 50*300))`). Sliced to the mel
    /// length at inference. Held in a non-Module box so it isn't a loadable param.
    private let randNoise: CFMConst

    public init(
        inChannels: Int = 320,
        outChannels: Int = 80,
        spkEmbDim: Int = 80,
        inferenceCfgRate: Float = 0.7,
        defaultNTimesteps: Int = 10
    ) {
        self.nFeats = outChannels
        self.spkEmbDim = spkEmbDim
        self.inferenceCfgRate = inferenceCfgRate
        self.defaultNTimesteps = defaultNTimesteps

        self._estimator.wrappedValue = CFMConditionalDecoder(
            inChannels: inChannels, outChannels: outChannels)

        // Deterministic pre-seeded noise (seed 0), shape [1, 80, 50*300]. Uses an
        // explicit PRNG key so the global RNG stream is left untouched.
        self.randNoise = CFMConst(
            MLXRandom.normal([1, MatchaCFM.melChannels, 50 * 300], key: MLXRandom.key(0)))

        super.init()
    }

    /// Cosine-mapped time schedule of length `nTimesteps + 1`.
    /// Python: `t_span = linspace(0,1,n+1); t_span = 1 - cos(t_span * 0.5 * pi)`.
    private func cosineSchedule(_ nTimesteps: Int) -> [Float] {
        (0 ... nTimesteps).map { i in
            let t = Float(i) / Float(nTimesteps)
            return 1.0 - cos(t * 0.5 * .pi)
        }
    }

    /// Euler ODE solver with classifier-free guidance.
    ///
    /// - Parameters:
    ///   - x: `[B, n_feats, T]` initial noise.
    ///   - tSpan: time schedule `[nTimesteps + 1]`.
    ///   - mu: `[B, n_feats, T]` encoder conditioning.
    ///   - mask: `[B, 1, T]` validity mask.
    ///   - spks: `[B, spk_emb_dim]` projected speaker embedding, or nil.
    ///   - cond: `[B, n_feats, T]` prompt-mel conditioning, or nil.
    /// - Returns: `[B, n_feats, T]` solution at the final timestep.
    public func solveEuler(
        _ x: MLXArray, tSpan: [Float], mu: MLXArray, mask: MLXArray,
        spks: MLXArray?, cond: MLXArray?
    ) -> MLXArray {
        var x = x
        let B = x.dim(0)
        let T = x.dim(2)
        let dtype = mu.dtype

        // Unconditioned-branch defaults when spks/cond are nil (match Python:
        // zeros are concatenated as the uncond half regardless).
        let spksZeros = MLXArray.zeros([B, spkEmbDim]).asType(dtype)
        let condZeros = MLXArray.zeros([B, nFeats, T]).asType(dtype)

        let cfg = MLXArray(inferenceCfgRate).asType(dtype)

        for step in 0 ..< (tSpan.count - 1) {
            let tVal = tSpan[step]
            let dt = tSpan[step + 1] - tSpan[step]
            let dtScalar = MLXArray(dt).asType(dtype)

            // CFG batch doubling: [cond | uncond].
            let xIn = concatenated([x, x], axis: 0)
            let maskIn = concatenated([mask, mask], axis: 0)
            let muIn = concatenated([mu, MLXArray.zeros(like: mu)], axis: 0)
            let tIn = MLXArray([Float](repeating: tVal, count: B * 2)).asType(dtype)

            let spksCondHalf = spks ?? spksZeros
            let spksIn = concatenated([spksCondHalf, MLXArray.zeros(like: spksCondHalf)], axis: 0)
            let condCondHalf = cond ?? condZeros
            let condIn = concatenated([condCondHalf, MLXArray.zeros(like: condCondHalf)], axis: 0)

            let dphiDt = estimator(
                xIn, mask: maskIn, mu: muIn, t: tIn, spks: spksIn, cond: condIn)

            let dphiCond = dphiDt[0 ..< B]
            let dphiUncond = dphiDt[B...]
            let v = (1.0 + cfg) * dphiCond - cfg * dphiUncond

            x = x + dtScalar * v
            eval(x)
        }

        return x
    }

    /// Run the CFM ODE to generate a mel spectrogram (causal variant — uses the
    /// pre-seeded deterministic noise buffer, matching `CausalConditionalCFM`).
    ///
    /// - Parameters:
    ///   - mu: `[B, n_feats, T]` encoder output conditioning.
    ///   - mask: `[B, 1, T]` validity mask (1 = valid, 0 = padding).
    ///   - nTimesteps: ODE step count (default `defaultNTimesteps`, typically 10).
    ///   - temperature: noise scaling factor.
    ///   - spks: `[B, spk_emb_dim]` projected speaker embedding, or nil.
    ///   - cond: `[B, n_feats, T]` prompt-mel conditioning, or nil.
    /// - Returns: `[B, n_feats, T]` generated mel spectrogram.
    public func solve(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int? = nil,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        let steps = nTimesteps ?? defaultNTimesteps
        let T = mu.dim(2)
        // z = rand_noise[:, :, :T] * temperature, broadcast to mu's batch.
        var z = randNoise.array[0..., 0..., 0 ..< T] * MLXArray(temperature)
        z = z.asType(mu.dtype)
        if mu.dim(0) != z.dim(0) {
            z = broadcast(z, to: [mu.dim(0), z.dim(1), z.dim(2)])
        }
        let tSpan = cosineSchedule(steps)
        return solveEuler(z, tSpan: tSpan, mu: mu, mask: mask, spks: spks, cond: cond)
    }

    /// Alias for `solve`, mirroring the Python decoder `__call__` entry point used by
    /// `CausalMaskedDiffWithXvec.inference`.
    public func inference(
        mu: MLXArray,
        mask: MLXArray,
        nTimesteps: Int? = nil,
        temperature: Float = 1.0,
        spks: MLXArray? = nil,
        cond: MLXArray? = nil
    ) -> MLXArray {
        solve(mu: mu, mask: mask, nTimesteps: nTimesteps,
              temperature: temperature, spks: spks, cond: cond)
    }
}
