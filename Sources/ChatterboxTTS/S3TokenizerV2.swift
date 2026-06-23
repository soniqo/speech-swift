import Foundation
import MLX
import MLXCommon
import MLXNN

// S3TokenizerV2 — supervised semantic speech tokenizer (25 Hz, codebook 6561).
//
// Encodes a 16 kHz reference waveform into discrete speech-token codes used to
// build the reference `prompt_token` for zero-shot voice cloning. Ported to
// MLX-Swift to match the reference implementation; the module/parameter keys
// below map 1:1 to the converted bundle's tokenizer safetensors keys, so
// `module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)`
// (or `load_weights`) loads cleanly.
//
// Pipeline (single reference clip, < 30 s — the only case cloning needs):
//   16 kHz samples
//     → 128-bin Whisper-style log-mel front-end          (S3Mel, below)
//     → conv1 (k3 s2 p1) + GELU, conv2 (k3 s2 p1) + GELU  (downsample ×4)
//     → 6 × FSMN residual-attention blocks (RoPE + depthwise memory)
//     → FSQ quantizer (project_down → tanh → round → base-3 index)
//   ⇒ one integer code per ~40 ms (mel_frames // 4).
//
// Layout convention: conv weights in the converted bundle are already in MLX
// Conv1d layout `[out, k, in]` (and the FSMN depthwise weight is `[out, k, 1]`),
// so no transpose happens at load time. Tensors flow channels-last `(B, T, C)`
// throughout, matching the MLX `Conv1d`/`Linear`/`LayerNorm` calls.
//
// Masking note: the reference masks padded frames before each conv and inside
// attention. For a single, unpadded clip every frame is valid, so the non-pad
// mask is all-ones and `mask_to_bias` is all-zero — the masking is a no-op and
// is omitted here. The codes are therefore identical to the reference's
// `quantize` / `quantize_simple` on a clip under 30 s.

// MARK: - log-mel front-end

/// 128-bin Whisper-style log-mel front-end for the tokenizer.
///
/// Matches the reference `log_mel_spectrogram`: `n_fft = 400`, `hop = 160`,
/// Hann window, slaney mel scale + slaney filterbank norm at 16 kHz, power
/// spectrogram. It reproduces `torch.stft(center=True)` (one extra frame) and
/// then drops the trailing frame, so `T_mel = samples // hop`. The log step is
/// `log10` (not natural log) followed by the tokenizer's `max(x, max-8)` clamp
/// and `(x + 4) / 4` affine, which `SlaneyMel` does not apply — so the power
/// mel comes from `SlaneyMel` and this code finishes the log + normalization.
enum S3Mel {
    static let sampleRate = 16000
    static let nFft = 400
    static let hop = 160
    static let win = 400
    static let nMels = 128

    /// Power-spectrogram (no log) slaney mel config; `centerPad` reproduces
    /// `torch.stft(center=True)`. We drop the trailing frame after the call.
    static let melConfig = SlaneyMelConfig(
        sampleRate: sampleRate, nFft: nFft, hop: hop, win: win, nMels: nMels,
        fmin: 0, fmax: Float(sampleRate) / 2, power: 2.0, logMel: false, centerPad: true)

    /// 16 kHz mono samples → log-mel `(nMels, T)` MLX float32, where
    /// `T = samples.count / hop`.
    static func logMel(samples: [Float]) -> MLXArray {
        // (frames, nMels) power mel; SlaneyMel/centerPad yields `samples/hop + 1`
        // frames, of which we drop the last to match the reference drop-last.
        var mel = SlaneyMel.melSpec(samples: samples, config: melConfig)  // [frames, nMels]
        let frames = mel.dim(0)
        if frames > 1 { mel = mel[0 ..< (frames - 1), 0...] }              // drop last frame
        mel = mel.transposed(1, 0)                                        // (nMels, T)

        // Tokenizer log + normalization (global, over the whole spectrogram).
        var logSpec = MLX.log10(maximum(mel, MLXArray(Float(1e-10))))
        logSpec = maximum(logSpec, logSpec.max() - 8.0)
        logSpec = (logSpec + 4.0) / 4.0
        return logSpec
    }
}

// MARK: - RoPE (precomputed, non-loadable)

/// Non-Module holder so the precomputed RoPE cos/sin tables aren't reflected as
/// loadable parameters (mirrors `T3Freqs` in T3.swift).
final class S3Freqs {
    let cos: MLXArray  // [end, headDim]
    let sin: MLXArray  // [end, headDim]
    init(cos: MLXArray, sin: MLXArray) { self.cos = cos; self.sin = sin }
}

/// Precompute GPT-NeoX RoPE cos/sin tables, matching the reference
/// `precompute_freqs_cis(dim=headDim, end, theta=10000)`: half-dim inverse
/// frequencies, outer with positions, then cos/sin duplicated to full headDim.
private func s3PrecomputeFreqs(headDim: Int, end: Int, theta: Float = 10000.0) -> S3Freqs {
    let half = headDim / 2
    let exps = MLXArray((0 ..< half).map { Float(2 * $0) / Float(headDim) })  // arange(0,dim,2)/dim
    let invFreqs = MLXArray(Float(1.0)) / MLX.pow(MLXArray(theta), exps)      // [half]
    let t = MLXArray((0 ..< end).map { Float($0) })                          // [end]
    let freqs = t.reshaped([end, 1]) * invFreqs.reshaped([1, half])          // [end, half]
    let cosH = cos(freqs)
    let sinH = sin(freqs)
    return S3Freqs(
        cos: concatenated([cosH, cosH], axis: -1),   // [end, headDim]
        sin: concatenated([sinH, sinH], axis: -1))
}

/// Apply GPT-NeoX RoPE to `x` of shape `[B, T, H, headDim]` (rotate-half).
private func s3ApplyRoPE(_ x: MLXArray, freqs: S3Freqs) -> MLXArray {
    let t = x.dim(1)
    let headDim = x.dim(3)
    let half = headDim / 2
    let cosA = freqs.cos[0 ..< t, 0...].reshaped([1, t, 1, headDim]).asType(x.dtype)
    let sinA = freqs.sin[0 ..< t, 0...].reshaped([1, t, 1, headDim]).asType(x.dtype)
    let xl = x[0..., 0..., 0..., 0 ..< half]
    let xr = x[0..., 0..., 0..., half ..< headDim]
    let rotated = concatenated([-xr, xl], axis: -1)  // rotate_half
    return x * cosA + rotated * sinA
}

// MARK: - FSMN multi-head attention

/// Multi-head self-attention with a parallel FSMN (depthwise memory over `v`),
/// RoPE on q/k, and a residual add of the memory onto the attention output.
final class S3FSMNAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d

    let nHead: Int
    let nState: Int
    let headDim: Int
    let scale: Float
    let leftPad: Int
    let rightPad: Int
    let freqs: S3Freqs

    init(nState: Int, nHead: Int, kernelSize: Int = 31, freqs: S3Freqs) {
        self.nHead = nHead
        self.nState = nState
        self.headDim = nState / nHead
        // (headDim)**-0.25 applied to both q and k → effective (headDim)**-0.5.
        self.scale = pow(Float(headDim), -0.25)
        self.leftPad = (kernelSize - 1) / 2
        self.rightPad = kernelSize - 1 - leftPad
        self.freqs = freqs
        // key has no bias in the reference; query/value/out do.
        self._query.wrappedValue = Linear(nState, nState, bias: true)
        self._key.wrappedValue = Linear(nState, nState, bias: false)
        self._value.wrappedValue = Linear(nState, nState, bias: true)
        self._out.wrappedValue = Linear(nState, nState, bias: true)
        // Depthwise (groups == channels) Conv1d, no bias, "same" via explicit pad.
        self._fsmnBlock.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState,
            kernelSize: kernelSize, stride: 1, padding: 0, groups: nState, bias: false)
        super.init()
    }

    /// Depthwise FSMN memory over `v` reshaped to `(B, T, D)`, plus residual.
    private func forwardFSMN(_ v: MLXArray) -> MLXArray {
        let b = v.dim(0)
        let t = v.dim(1)
        let inputs = v.reshaped([b, t, nState])  // (B, T, D)
        let padL = MLXArray.zeros([b, leftPad, nState], dtype: inputs.dtype)
        let padR = MLXArray.zeros([b, rightPad, nState], dtype: inputs.dtype)
        let padded = concatenated([padL, inputs, padR], axis: 1)
        let x = fsmnBlock(padded)  // channels-last Conv1d → (B, T, D)
        return x + inputs
    }

    /// x: `(B, T, D)` → `(B, T, D)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        var q = query(x).reshaped([b, t, nHead, headDim])
        var k = key(x).reshaped([b, t, nHead, headDim])
        let v = value(x).reshaped([b, t, nHead, headDim])

        q = s3ApplyRoPE(q, freqs: freqs)
        k = s3ApplyRoPE(k, freqs: freqs)

        let fsmMemory = forwardFSMN(v)  // (B, T, D)

        // (B, H, T, headDim) with the (headDim)**-0.25 scale on q and k.
        let qh = (q * scale).transposed(0, 2, 1, 3)
        let kh = (k * scale).transposed(0, 2, 1, 3)
        let vh = v.transposed(0, 2, 1, 3)

        var scores = matmul(qh, kh.transposed(0, 1, 3, 2))  // [B,H,T,T]
        scores = MLX.softmax(scores.asType(.float32), axis: -1).asType(qh.dtype)
        var o = matmul(scores, vh)                           // [B,H,T,headDim]
        o = o.transposed(0, 2, 1, 3).reshaped([b, t, nState])
        return out(o) + fsmMemory
    }
}

// MARK: - Residual attention block

/// FSMN residual-attention block: `x + attn(ln(x))`, then `x + mlp(ln(x))`.
final class S3ResidualAttentionBlock: Module {
    @ModuleInfo(key: "attn") var attn: S3FSMNAttention
    @ModuleInfo(key: "attn_ln") var attnLN: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: Sequential
    @ModuleInfo(key: "mlp_ln") var mlpLN: LayerNorm

    init(nState: Int, nHead: Int, freqs: S3Freqs) {
        self._attn.wrappedValue = S3FSMNAttention(nState: nState, nHead: nHead, freqs: freqs)
        // attn_ln uses eps 1e-6 in the reference; mlp_ln uses the default (1e-5).
        self._attnLN.wrappedValue = LayerNorm(dimensions: nState, eps: 1e-6, affine: true)
        let nMLP = nState * 4
        self._mlp.wrappedValue = Sequential(layers: [
            Linear(nState, nMLP), GELU(), Linear(nMLP, nState),
        ])
        self._mlpLN.wrappedValue = LayerNorm(dimensions: nState, eps: 1e-5, affine: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attn(attnLN(x))
        return h + mlp(mlpLN(h))
    }
}

// MARK: - Audio encoder

/// Conv subsampling (×4) + FSMN residual-attention stack. Produces the hidden
/// states `(B, T', n_state)` quantized to codes.
final class S3AudioEncoderV2: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d
    @ModuleInfo(key: "blocks") var blocks: [S3ResidualAttentionBlock]

    init(nMels: Int, nState: Int, nHead: Int, nLayer: Int) {
        let freqs = s3PrecomputeFreqs(headDim: nState / nHead, end: 1024 * 2)
        // V2 always strides 2 at conv1 (and conv2), for ×4 total downsampling.
        self._conv1.wrappedValue = Conv1d(
            inputChannels: nMels, outputChannels: nState,
            kernelSize: 3, stride: 2, padding: 1, bias: true)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: nState, outputChannels: nState,
            kernelSize: 3, stride: 2, padding: 1, bias: true)
        self._blocks.wrappedValue = (0 ..< nLayer).map { _ in
            S3ResidualAttentionBlock(nState: nState, nHead: nHead, freqs: freqs)
        }
        super.init()
    }

    /// mel: `(B, n_mels, T)` → hidden `(B, T', n_state)`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel.transposed(0, 2, 1)  // (B, T, n_mels)
        x = MLXNN.gelu(conv1(x))         // (B, T1, n_state)
        x = MLXNN.gelu(conv2(x))         // (B, T2, n_state)
        for block in blocks { x = block(x) }
        return x
    }
}

// MARK: - FSQ quantizer

/// Finite Scalar Quantization codebook: `project_down` → tanh → round to 3
/// levels → base-3 index. Holds only the `project_down` linear weights.
final class S3FSQCodebook: Module {
    @ModuleInfo(key: "project_down") var projectDown: Linear

    let level: Int

    init(dim: Int, level: Int = 3) {
        self.level = level
        self._projectDown.wrappedValue = Linear(dim, 8, bias: true)  // 8 = 2 * level
        super.init()
    }

    /// hidden `(B, T, dim)` → integer codes `(B, T)`.
    func encode(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let flat = x.reshaped([b * t, x.dim(2)])
        var h = projectDown(flat).asType(.float32)          // [(B*T), 8]
        h = MLX.tanh(h)
        h = h * Float(0.9990000128746033)
        h = MLX.round(h) + 1.0                               // levels {0,1,2}
        // base-3 mixed-radix: sum(h[i] * 3^i).
        let powers = MLXArray((0 ..< 8).map { Float(pow(Double(level), Double($0))) })  // [8]
        let mu = MLX.sum(h * powers.reshaped([1, 8]), axis: -1)  // [(B*T)]
        return mu.reshaped([b, t]).asType(.int32)
    }
}

/// Wraps the FSQ codebook under the `fsq_codebook` key to match the bundle.
final class S3FSQVectorQuantization: Module {
    @ModuleInfo(key: "fsq_codebook") var fsqCodebook: S3FSQCodebook

    init(dim: Int) {
        self._fsqCodebook.wrappedValue = S3FSQCodebook(dim: dim)
        super.init()
    }

    func encode(_ x: MLXArray) -> MLXArray { fsqCodebook.encode(x) }
}

// MARK: - Tokenizer

/// S3TokenizerV2 — encodes a 16 kHz reference clip into discrete speech tokens.
public final class S3TokenizerV2: Module {
    @ModuleInfo(key: "encoder") var encoder: S3AudioEncoderV2
    @ModuleInfo(key: "quantizer") var quantizer: S3FSQVectorQuantization

    public static let sampleRate = 16000
    public static let tokenRate = 25
    public static let codebookSize = 6561  // 3^8
    public static let nMels = 128
    public static let nState = 1280
    public static let nHead = 20
    public static let nLayer = 6

    public override init() {
        self._encoder.wrappedValue = S3AudioEncoderV2(
            nMels: Self.nMels, nState: Self.nState, nHead: Self.nHead, nLayer: Self.nLayer)
        self._quantizer.wrappedValue = S3FSQVectorQuantization(dim: Self.nState)
        super.init()
    }

    /// Quantize a precomputed log-mel `(B, n_mels, T)` to codes `(B, T')` and the
    /// per-item token length `(B,)`. Single-clip path only (clips under 30 s).
    public func quantize(mel: MLXArray) -> (codes: MLXArray, codeLen: MLXArray) {
        let hidden = encoder(mel)            // (B, T', n_state)
        let codes = quantizer.encode(hidden) // (B, T')
        let b = codes.dim(0)
        let codeLen = MLXArray(Array(repeating: Int32(codes.dim(1)), count: b))
        return (codes, codeLen)
    }

    /// Encode 16 kHz mono samples → integer speech-token codes for the reference.
    /// Runs the log-mel front-end, the encoder, and the FSQ quantizer. The token
    /// count is `mel_frames // 4` (`mel_frames = samples.count / hop`).
    public func encode(_ wav16k: [Float]) -> [Int] {
        let mel = S3Mel.logMel(samples: wav16k)             // (n_mels, T)
        let melB = mel.expandedDimensions(axis: 0)          // (1, n_mels, T)
        let (codes, _) = quantize(mel: melB)                // (1, T')
        eval(codes)
        let flat = codes.reshaped([codes.size]).asArray(Int32.self)
        return flat.map { Int($0) }
    }
}
