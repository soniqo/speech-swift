import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - S3Gen flow encoder (UpsampleConformerEncoder)
//
// Port of the Chatterbox S3Gen flow encoder: a relative-positional conformer with
// a linear input embed, a pre-lookahead causal conv, an upsampling stage, and a
// second post-upsample conformer stack, followed by a final LayerNorm. The flow
// invokes it as: input_embedding(tokens) -> encoder -> encoder_proj, which is what
// this file exposes via `S3GenConformer.encode(...)`.
//
// Layout convention: MLX-Swift `Conv1d` is NLC (PyTorch `[out, in, k]` weights are
// stored as `[out, k, in]`); the bundle is already in that layout. Tensors flow as
// `[B, T, D]` everywhere except inside the conv blocks, which transpose as needed.
//
// Weight keys (after stripping the `s3gen.flow.` prefix) match exactly so that
// `update(parameters: ModuleParameters.unflattened(weights), verify: .all)` loads
// cleanly. The bundle at /tmp/cbx-fp16/model.safetensors carries only the
// embed / pre_lookahead / up / after_norm pieces of the encoder (the upstream
// loader uses `strict=False` and leaves the conformer blocks at their init); those
// blocks are therefore held outside the reflected parameter tree so `verify: .all`
// stays green against this bundle, and a `loadConformerBlocks` hook is provided for
// a full checkpoint that does carry them.

// MARK: - Espnet relative positional encoding

/// Precompute the Espnet-style bidirectional relative positional encoding table.
///
/// Builds `[1, 2*size-1, d]`: reversed positive positions `[..., 2, 1, 0]` followed
/// by negative positions `[1, 2, 3, ...]`. `position_encoding(size)` slices the
/// centered `2*size-1` window out of it for use in relative attention.
private func espnetRelPosTable(_ dModel: Int, maxLen: Int) -> MLXArray {
    let position = MLXArray((0 ..< maxLen).map { Float($0) }).reshaped([maxLen, 1])
    let divExp = MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
        * (-(Foundation.log(10000.0) / Double(dModel)))
    let divTerm = MLX.exp(divExp.asType(.float32))                 // [d/2]

    let half = divTerm.dim(0)

    func interleave(_ s: MLXArray, _ c: MLXArray) -> MLXArray {
        // build [maxLen, d] with sin at even indices, cos at odd indices
        let stacked = MLX.stacked([s, c], axis: -1)                // [maxLen, d/2, 2]
        return stacked.reshaped([maxLen, half * 2])
    }

    let posSin = MLX.sin(position * divTerm)
    let posCos = MLX.cos(position * divTerm)
    let pePositive = interleave(posSin, posCos)                    // [maxLen, d]

    let negSin = MLX.sin(MLXArray(Float(-1.0)) * position * divTerm)
    let negCos = MLX.cos(MLXArray(Float(-1.0)) * position * divTerm)
    let peNegative = interleave(negSin, negCos)                    // [maxLen, d]

    // reverse positive along time, drop the duplicated center (index 0) of negative
    let posRev = pePositive[.stride(by: -1)]                       // [maxLen, d] reversed
    let negTail = peNegative[1...]                                 // [maxLen-1, d]

    let pe = MLX.concatenated([posRev, negTail], axis: 0)          // [2*maxLen-1, d]
    return pe.expandedDimensions(axis: 0)                          // [1, 2*maxLen-1, d]
}

/// Non-Module holder so the precomputed rel-pos table is not reflected as a
/// loadable parameter (same trick as `T3Freqs`).
final class S3GenRelPos {
    let table: MLXArray   // [1, 2*maxLen-1, d]
    let dModel: Int
    let xscale: Float
    init(dModel: Int, maxLen: Int) {
        self.dModel = dModel
        self.xscale = Foundation.sqrt(Float(dModel))
        self.table = espnetRelPosTable(dModel, maxLen: maxLen)
    }

    /// Returns the centered `[1, 2*T-1, d]` slice for a sequence of length `T`.
    func posEmb(_ size: Int) -> MLXArray {
        let center = table.dim(1) / 2
        let start = center - size + 1
        let end = center + size
        return table[0..., start ..< end, 0...]
    }
}

// MARK: - LinearNoSubsampling embed

/// `linear -> LayerNorm`, returning the scaled hidden and the rel-pos table slice.
/// Matches `LinearNoSubsampling` + `EspnetRelPositionalEncoding`.
final class S3GenLinearEmbed: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    private let relpos: S3GenRelPos

    init(idim: Int, odim: Int, relpos: S3GenRelPos) {
        self.relpos = relpos
        _linear.wrappedValue = Linear(idim, odim, bias: true)
        _norm.wrappedValue = LayerNorm(dimensions: odim, eps: 1e-5, affine: true)
        super.init()
    }

    /// `x: [B, T, idim]` -> (`x: [B, T, odim]` scaled by sqrt(d), `posEmb: [1, 2*T-1, odim]`).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = linear(x)
        h = norm(h)
        h = h * MLXArray(relpos.xscale).asType(h.dtype)
        let pe = relpos.posEmb(h.dim(1)).asType(h.dtype)
        return (h, pe)
    }
}

// MARK: - Pre-lookahead layer

/// Two Conv1d layers with explicit right/left padding and a residual, matching
/// `PreLookaheadLayer` (conv1 kernel = pre_lookahead_len+1, conv2 kernel = 3).
final class S3GenPreLookahead: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    let preLookaheadLen: Int

    init(channels: Int, preLookaheadLen: Int) {
        self.preLookaheadLen = preLookaheadLen
        _conv1.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: preLookaheadLen + 1, stride: 1, padding: 0, bias: true)
        _conv2.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: 3, stride: 1, padding: 0, bias: true)
        super.init()
    }

    /// `inputs: [B, T, C]` (NLC) -> `[B, T, C]`.
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // right-pad on time for the look-ahead conv
        var h = MLX.padded(inputs, widths: [.init((0, 0)), .init((0, preLookaheadLen)), .init((0, 0))])
        h = leakyRelu(conv1(h))
        // left-pad for the causal conv2
        h = MLX.padded(h, widths: [.init((0, 0)), .init((2, 0)), .init((0, 0))])
        h = conv2(h)
        return h + inputs
    }
}

// MARK: - Upsample1D

/// Nearest-neighbour temporal upsample (repeat each frame `stride` times) followed
/// by a stride-1 Conv1d with left padding `stride*2`, matching `Upsample1D`.
final class S3GenUpsample: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    let stride: Int

    init(channels: Int, outChannels: Int, stride: Int) {
        self.stride = stride
        _conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: outChannels,
            kernelSize: stride * 2 + 1, stride: 1, padding: 0, bias: true)
        super.init()
    }

    /// `inputs: [B, C, T]` (NCL) -> `[B, C, T*stride]` (NCL).
    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // repeat each timestep `stride` times along the time axis (axis 2 in NCL)
        var h = MLX.repeated(inputs, count: stride, axis: 2)        // [B, C, T*stride]
        // left-pad by stride*2 on time
        h = MLX.padded(h, widths: [.init((0, 0)), .init((0, 0)), .init((stride * 2, 0))])
        h = h.transposed(0, 2, 1)                                   // NLC
        h = conv(h)
        return h.transposed(0, 2, 1)                                // back to NCL
    }
}

// MARK: - Positionwise feed forward

final class S3GenFeedForward: Module {
    @ModuleInfo(key: "w_1") var w1: Linear
    @ModuleInfo(key: "w_2") var w2: Linear

    init(idim: Int, hidden: Int) {
        _w1.wrappedValue = Linear(idim, hidden, bias: true)
        _w2.wrappedValue = Linear(hidden, idim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)))
    }
}

// MARK: - Relative-position multi-head attention

/// Multi-head self-attention with Transformer-XL style relative positional
/// encoding (`pos_bias_u`/`pos_bias_v`, `linear_q/k/v/out`, `linear_pos`).
final class S3GenRelPosAttention: Module {
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear

    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    let h: Int
    let dK: Int

    init(nHead: Int, nFeat: Int, keyBias: Bool) {
        self.h = nHead
        self.dK = nFeat / nHead
        _linearQ.wrappedValue = Linear(nFeat, nFeat, bias: true)
        _linearK.wrappedValue = Linear(nFeat, nFeat, bias: keyBias)
        _linearV.wrappedValue = Linear(nFeat, nFeat, bias: true)
        _linearOut.wrappedValue = Linear(nFeat, nFeat, bias: true)
        _linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        _posBiasU.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead])
        _posBiasV.wrappedValue = MLXArray.zeros([nHead, nFeat / nHead])
        super.init()
    }

    /// Relative shift: `[B, h, T1, 2*T1-1]` -> `[B, h, T1, T1]`.
    private func relShift(_ x: MLXArray) -> MLXArray {
        let (b, hh, t1, n) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let zeroPad = MLXArray.zeros([b, hh, t1, 1]).asType(x.dtype)
        var xPadded = MLX.concatenated([zeroPad, x], axis: -1)      // [b, h, t1, n+1]
        xPadded = xPadded.reshaped([b, hh, n + 1, t1])
        let xView = xPadded[0..., 0..., 1..., 0...].reshaped([b, hh, t1, n])
        return xView[0..., 0..., 0..., 0 ..< (n / 2 + 1)]
    }

    /// `x: [B, T, D]`, `posEmb: [1, 2*T-1, D]`, `mask: [B, 1, T]` or nil -> `[B, T, D]`.
    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)

        let q = linearQ(x).reshaped([b, t, h, dK])                 // [B, T, h, dK]
        let k = linearK(x).reshaped([b, t, h, dK]).transposed(0, 2, 1, 3)  // [B, h, T, dK]
        let v = linearV(x).reshaped([b, t, h, dK]).transposed(0, 2, 1, 3)  // [B, h, T, dK]

        let nPos = posEmb.dim(1)
        let p = linearPos(posEmb).reshaped([posEmb.dim(0), nPos, h, dK]).transposed(0, 2, 1, 3)  // [1, h, 2T-1, dK]

        // q + bias, then move heads forward: [B, h, T, dK]
        let qU = (q + posBiasU.asType(q.dtype)).transposed(0, 2, 1, 3)
        let qV = (q + posBiasV.asType(q.dtype)).transposed(0, 2, 1, 3)

        let scale = MLXArray(1.0 / Foundation.sqrt(Float(dK))).asType(x.dtype)

        let matrixAC = MLX.matmul(qU, k.swappedAxes(-2, -1))        // [B, h, T, T]
        var matrixBD = MLX.matmul(qV, p.swappedAxes(-2, -1))        // [B, h, T, 2T-1]
        if matrixBD.dim(-1) != matrixAC.dim(-1) {
            matrixBD = relShift(matrixBD)                          // [B, h, T, T]
        }

        var scores = (matrixAC + matrixBD) * scale                 // [B, h, T, T]

        if let mask {
            // mask: [B, 1, T] (1 = keep) -> [B, 1, 1, T]
            let m = mask.expandedDimensions(axis: 1)               // [B, 1, 1, T]
            let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(scores.dtype)
            scores = MLX.where(m .== MLXArray(0).asType(m.dtype), neg, scores)
        }

        var attn = MLX.softmax(scores.asType(.float32), axis: -1).asType(x.dtype)
        if let mask {
            let m = mask.expandedDimensions(axis: 1)
            attn = MLX.where(m .== MLXArray(0).asType(m.dtype), MLXArray(0).asType(attn.dtype), attn)
        }

        var out = MLX.matmul(attn, v)                              // [B, h, T, dK]
        out = out.transposed(0, 2, 1, 3).reshaped([b, t, h * dK])  // [B, T, D]
        return linearOut(out)
    }
}

// MARK: - Conformer encoder layer

/// One conformer block. With the S3Gen config (`macaron_style=False`,
/// `use_cnn_module=False`) this reduces to: pre-norm MHA with rel-pos, residual;
/// pre-norm feed-forward, residual. Matches `ConformerEncoderLayer`.
final class S3GenConformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: S3GenRelPosAttention
    @ModuleInfo(key: "feed_forward") var feedForward: S3GenFeedForward
    @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
    @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm

    init(size: Int, nHead: Int, linearUnits: Int, keyBias: Bool) {
        _selfAttn.wrappedValue = S3GenRelPosAttention(nHead: nHead, nFeat: size, keyBias: keyBias)
        _feedForward.wrappedValue = S3GenFeedForward(idim: size, hidden: linearUnits)
        _normFF.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12, affine: true)
        _normMHA.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12, affine: true)
        super.init()
    }

    /// `x: [B, T, D]`, `posEmb: [1, 2*T-1, D]`, `mask: [B, 1, T]` or nil -> `[B, T, D]`.
    func callAsFunction(_ x: MLXArray, posEmb: MLXArray, mask: MLXArray?) -> MLXArray {
        // pre-norm self-attention + residual
        var h = x + selfAttn(normMHA(x), posEmb: posEmb, mask: mask)
        // pre-norm feed-forward + residual
        h = h + feedForward(normFF(h))
        return h
    }
}

// MARK: - S3GenConformer (UpsampleConformerEncoder)

/// Config for the S3Gen flow encoder. Defaults match `s3gen.py`'s
/// `UpsampleConformerEncoder(output_size=512, attention_heads=8, linear_units=2048,
/// num_blocks=6, ...)` with the encoder defaults `num_up_blocks=4,
/// pre_lookahead_len=3, upsample_stride=2`.
public struct S3GenConformerConfig: Sendable {
    public var inputSize: Int = 512
    public var outputSize: Int = 512
    public var attentionHeads: Int = 8
    public var linearUnits: Int = 2048
    public var numBlocks: Int = 6
    public var numUpBlocks: Int = 4
    public var preLookaheadLen: Int = 3
    public var upsampleStride: Int = 2
    public var keyBias: Bool = true
    public var maxLen: Int = 5000

    // Flow-level wrappers around the encoder.
    public var vocabSize: Int = 6561
    public var spkEmbedDim: Int = 192
    public var melDim: Int = 80          // encoder_proj / spk_embed_affine_layer out dim

    public init() {}
}

/// The S3Gen flow encoder plus the flow-level pieces the encoder path needs:
/// `input_embedding`, `spk_embed_affine_layer`, and `encoder_proj`.
///
/// The reflected (loadable) parameters match the bundle keys (after stripping
/// `s3gen.flow.`):
///   encoder.embed.{linear,norm}.{weight,bias}
///   encoder.pre_lookahead_layer.conv{1,2}.{weight,bias}
///   encoder.up_embed.{linear,norm}.{weight,bias}
///   encoder.up_layer.conv.{weight,bias}
///   encoder.after_norm.{weight,bias}
///   input_embedding.weight
///   spk_embed_affine_layer.{weight,bias}
///   encoder_proj.{weight,bias}
///
/// The conformer blocks (encoders / up_encoders) are held outside the reflected
/// parameter tree (`encoders` / `upEncoders` plain arrays) so that
/// `update(parameters:verify: .all)` succeeds against the fp16 bundle, which omits
/// them (the upstream loader uses `strict=False`). Use `loadConformerBlocks` to
/// load them from a checkpoint that does carry them.
public final class S3GenConformer: Module {
    public let config: S3GenConformerConfig

    // Flow-level pieces (keys: input_embedding.*, spk_embed_affine_layer.*, encoder_proj.*)
    @ModuleInfo(key: "input_embedding") var inputEmbedding: Embedding
    @ModuleInfo(key: "spk_embed_affine_layer") var spkEmbedAffineLayer: Linear
    @ModuleInfo(key: "encoder_proj") var encoderProj: Linear

    // Encoder pieces present in the bundle (keys under encoder.*)
    @ModuleInfo(key: "encoder") var encoder: S3GenEncoderCore

    // Conformer blocks held out of the reflected tree (no weights in this bundle).
    let encoders: [S3GenConformerLayer]
    let upEncoders: [S3GenConformerLayer]

    private let relpos: S3GenRelPos
    private let upRelpos: S3GenRelPos

    public init(config: S3GenConformerConfig = S3GenConformerConfig()) {
        self.config = config
        self.relpos = S3GenRelPos(dModel: config.outputSize, maxLen: config.maxLen)
        self.upRelpos = S3GenRelPos(dModel: config.outputSize, maxLen: config.maxLen)

        _inputEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.inputSize)
        _spkEmbedAffineLayer.wrappedValue = Linear(config.spkEmbedDim, config.melDim, bias: true)
        _encoderProj.wrappedValue = Linear(config.outputSize, config.melDim, bias: true)
        _encoder.wrappedValue = S3GenEncoderCore(config: config, relpos: relpos, upRelpos: upRelpos)

        self.encoders = (0 ..< config.numBlocks).map { _ in
            S3GenConformerLayer(
                size: config.outputSize, nHead: config.attentionHeads,
                linearUnits: config.linearUnits, keyBias: config.keyBias)
        }
        self.upEncoders = (0 ..< config.numUpBlocks).map { _ in
            S3GenConformerLayer(
                size: config.outputSize, nHead: config.attentionHeads,
                linearUnits: config.linearUnits, keyBias: config.keyBias)
        }
        super.init()
    }

    /// Load the conformer blocks from a full checkpoint that carries them.
    ///
    /// `weights` keys are expected after stripping `s3gen.flow.encoder.`, e.g.
    /// `encoders.0.self_attn.linear_q.weight`. Both `encoders.<i>.*` and the
    /// upstream `encoders_<i>.*` flat naming are accepted.
    public func loadConformerBlocks(_ weights: [String: MLXArray]) {
        func loadGroup(_ layers: [S3GenConformerLayer], prefix: String, flatPrefix: String) {
            for (i, layer) in layers.enumerated() {
                var sub: [String: MLXArray] = [:]
                let p1 = "\(prefix).\(i)."
                let p2 = "\(flatPrefix)\(i)."
                for (k, v) in weights {
                    if k.hasPrefix(p1) { sub[String(k.dropFirst(p1.count))] = v }
                    else if k.hasPrefix(p2) { sub[String(k.dropFirst(p2.count))] = v }
                }
                if !sub.isEmpty {
                    layer.update(parameters: ModuleParameters.unflattened(sub))
                }
            }
        }
        loadGroup(encoders, prefix: "encoders", flatPrefix: "encoders_")
        loadGroup(upEncoders, prefix: "up_encoders", flatPrefix: "up_encoders_")
    }

    /// Project a raw speaker embedding to the mel dim (L2-normalize then affine),
    /// matching the `spk_embed_affine_layer` step in `flow.inference`.
    /// `embedding: [B, spkEmbedDim]` -> `[B, melDim]`.
    public func projectSpeaker(_ embedding: MLXArray) -> MLXArray {
        let norm = MLX.sqrt(MLX.sum(embedding * embedding, axis: 1, keepDims: true)) + MLXArray(Float(1e-8))
        return spkEmbedAffineLayer(embedding / norm)
    }

    /// Run the encoder over speech-token embeddings and project to the mel dim.
    ///
    /// Mirrors the encoder path of `flow.inference`:
    ///   token embeddings -> encoder (embed -> pre_lookahead -> 6 conformer blocks ->
    ///   upsample -> up_embed -> 4 conformer blocks -> after_norm) -> encoder_proj.
    ///
    /// - Parameter tokenEmbeddings: `[B, T, inputSize]` — the output of
    ///   `inputEmbedding(tokens)` (optionally masked, as in `flow.inference`).
    /// - Returns: `mu = [B, T*upsampleStride, melDim]`, the conditioning the CFM
    ///   decoder consumes (it transposes to `[B, melDim, T']`).
    public func encode(tokenEmbeddings: MLXArray) -> MLXArray {
        let h = encoder(tokenEmbeddings, encoders: encoders, upEncoders: upEncoders)
        return encoderProj(h)                                       // [B, T', melDim]
    }

    /// Convenience: embed token ids then encode. `tokens: [B, T]` (Int) ->
    /// `mu: [B, T*upsampleStride, melDim]`.
    public func encode(tokens: MLXArray) -> MLXArray {
        encode(tokenEmbeddings: inputEmbedding(tokens))
    }
}

// MARK: - Encoder core (bundle-present submodules)

/// The `encoder.*` submodules that exist in the bundle. The conformer block stacks
/// are passed in by the owning `S3GenConformer` (they live outside the reflected
/// tree). Output is the encoded hidden `[B, T', outputSize]` (pre encoder_proj).
public final class S3GenEncoderCore: Module {
    @ModuleInfo(key: "embed") var embed: S3GenLinearEmbed
    @ModuleInfo(key: "pre_lookahead_layer") var preLookahead: S3GenPreLookahead
    @ModuleInfo(key: "up_embed") var upEmbed: S3GenLinearEmbed
    @ModuleInfo(key: "up_layer") var upLayer: S3GenUpsample
    @ModuleInfo(key: "after_norm") var afterNorm: LayerNorm

    init(config: S3GenConformerConfig, relpos: S3GenRelPos, upRelpos: S3GenRelPos) {
        _embed.wrappedValue = S3GenLinearEmbed(
            idim: config.inputSize, odim: config.outputSize, relpos: relpos)
        _preLookahead.wrappedValue = S3GenPreLookahead(
            channels: config.outputSize, preLookaheadLen: config.preLookaheadLen)
        _upEmbed.wrappedValue = S3GenLinearEmbed(
            idim: config.inputSize, odim: config.outputSize, relpos: upRelpos)
        _upLayer.wrappedValue = S3GenUpsample(
            channels: config.outputSize, outChannels: config.outputSize,
            stride: config.upsampleStride)
        _afterNorm.wrappedValue = LayerNorm(dimensions: config.outputSize, eps: 1e-5, affine: true)
        super.init()
    }

    /// `xs: [B, T, inputSize]` -> `[B, T*stride, outputSize]`.
    func callAsFunction(
        _ xs: MLXArray,
        encoders: [S3GenConformerLayer],
        upEncoders: [S3GenConformerLayer]
    ) -> MLXArray {
        // Full-context inference: no padding mask (single sequence, all valid).
        var (h, posEmb) = embed(xs)

        h = preLookahead(h)
        for layer in encoders {
            h = layer(h, posEmb: posEmb, mask: nil)
        }

        // upsample: NLC -> NCL -> upsample -> NLC
        h = upLayer(h.transposed(0, 2, 1)).transposed(0, 2, 1)

        let upd = upEmbed(h)
        h = upd.0
        posEmb = upd.1
        for layer in upEncoders {
            h = layer(h, posEmb: posEmb, mask: nil)
        }

        return afterNorm(h)
    }
}
