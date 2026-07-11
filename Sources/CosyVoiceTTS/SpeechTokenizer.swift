import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Configuration

/// S3-Tokenizer-v3 architecture spec — matches s3tokenizer.model_v3.ModelConfigV3.
public struct SpeechTokenizerConfig: Sendable {
    public var nMels: Int = 128
    public var nAudioState: Int = 1280
    public var nAudioHead: Int = 20
    public var nAudioLayer: Int = 12
    public var codebookLevels: Int = 3         // FSQ rounds tanh output to {-1, 0, 1}
    public var codebookChannels: Int = 8       // 3^8 = 6561 = vocab
    public var fsmnKernelSize: Int = 31        // depth-wise positional conv
    public var subsampleStride1: Int = 2       // conv1 stride (raw mel → )
    public var subsampleStride2: Int = 2       // conv2 stride (→ 25 Hz at 16 kHz/160-hop input)
    public var ropeBase: Float = 10_000
    public var ropeMaxSeqLen: Int = 2_048

    public var headDim: Int { nAudioState / nAudioHead }       // 64
    public var totalSubsample: Int { subsampleStride1 * subsampleStride2 }  // 4
    public var codebookSize: Int {                              // 6561
        var v = 1
        for _ in 0..<codebookChannels { v *= codebookLevels }
        return v
    }
    public var mlpDim: Int { nAudioState * 4 }                  // 5120

    public init() {}
}

// MARK: - FSQ vector quantizer (inference-only)
//
// Mirrors `s3tokenizer.model_v2.FSQCodebook.encode`:
//   h = tanh(project_down(x)) * 0.999
//   h = round(h) + 1                        # values in {0, 1, 2}
//   idx = sum_k h[k] * 3^k                  # base-3 → flat index in [0, 6561)
public class FSQVectorQuantizer: Module {
    @ModuleInfo(key: "_codebook") var codebook: FSQCodebook

    public init(config: SpeechTokenizerConfig) {
        self._codebook.wrappedValue = FSQCodebook(config: config)
        super.init()
    }

    /// Quantize hidden states to FSQ code indices.
    /// - Parameter x: `[B, T, n_state]` encoder hidden states (bf16/fp32)
    /// - Returns: `[B, T]` integer codes in `[0, codebookSize)`
    public func encode(_ x: MLXArray) -> MLXArray {
        codebook.encode(x)
    }
}

/// Inner codebook — owns the projection. Mirrors upstream's `_codebook.project_down`.
public class FSQCodebook: Module {
    @ModuleInfo(key: "project_down") var projectDown: Linear
    let level: Int
    let channels: Int

    public init(config: SpeechTokenizerConfig) {
        self.level = config.codebookLevels
        self.channels = config.codebookChannels
        self._projectDown.wrappedValue = Linear(config.nAudioState, config.codebookChannels)
        super.init()
    }

    /// `x: [B, T, n_state]` → projection to `[B, T, channels]` → tanh → round → base-3 packing.
    public func encode(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)

        // Project down to FSQ channels and convert to fp32 for the rounding stage —
        // bf16 can land integers ±1 off when the pre-round value sits exactly on a
        // half-integer, which would shift the entire base-3 index.
        var h = projectDown(x).asType(.float32)    // [B, T, channels]
        h = tanh(h)
        h = h * MLXArray(Float(0.9990000128746033))
        h = round(h) + MLXArray(Float(1.0))         // values in {0, 1, 2}

        // Base-`level` packing: idx = sum_k h[..., k] * level^k.
        // powers[k] = level^k, k = 0..<channels.
        var powers = [Float](repeating: 0, count: channels)
        var p: Float = 1.0
        for k in 0..<channels {
            powers[k] = p
            p *= Float(level)
        }
        let powersArr = MLXArray(powers)            // [channels]
        let codes = sum(h * powersArr, axis: -1)    // [B, T]

        return codes.asType(.int32).reshaped([B, T])
    }
}

// MARK: - FSMN attention block
//
// Mirrors `s3tokenizer.model_v2.FSMNMultiHeadAttention`:
//   q = Linear(x), k = Linear(x, bias=False), v = Linear(x)
//   apply split-half RoPE to (q, k)
//   fsm_memory = depthwise Conv1d(v) over T with kernel=31, with v residual    (= forward_fsmn)
//   attn = softmax(q @ k^T / sqrt(d)) @ v
//   return Linear(attn) + fsm_memory
public class FSMNMultiHeadAttention: Module {
    let nHead: Int
    let headDim: Int
    let scale: Float
    let kernelSize: Int
    let leftPad: Int
    let rightPad: Int

    @ModuleInfo var query: Linear
    @ModuleInfo var key: Linear
    @ModuleInfo var value: Linear
    @ModuleInfo var out: Linear

    // Depth-wise positional conv: groups = n_state, in/group = 1, kernel = 31.
    // Safetensors stored as [n_state, 31, 1] (MLX Conv1d weight layout).
    @ModuleInfo(key: "fsmn_block") var fsmnBlock: Conv1d

    public init(config: SpeechTokenizerConfig) {
        self.nHead = config.nAudioHead
        self.headDim = config.headDim
        self.scale = 1.0 / sqrt(Float(headDim))
        self.kernelSize = config.fsmnKernelSize
        self.leftPad = (config.fsmnKernelSize - 1) / 2
        self.rightPad = config.fsmnKernelSize - 1 - leftPad

        let n = config.nAudioState
        self._query.wrappedValue = Linear(n, n)
        self._key.wrappedValue = Linear(n, n, bias: false)   // upstream's only no-bias linear
        self._value.wrappedValue = Linear(n, n)
        self._out.wrappedValue = Linear(n, n)

        self._fsmnBlock.wrappedValue = Conv1d(
            inputChannels: n,
            outputChannels: n,
            kernelSize: config.fsmnKernelSize,
            stride: 1,
            padding: 0,                     // we pad manually so left/right are explicit
            groups: n,
            bias: false
        )
        super.init()
    }

    /// - Parameters:
    ///   - x: `[B, T, n_state]`
    ///   - rope: RoPE module (split-half, 64-dim)
    /// - Returns: `[B, T, n_state]`
    public func callAsFunction(_ x: MLXArray, rope: MLXNN.RoPE) -> MLXArray {
        let B = x.dim(0)
        let T = x.dim(1)
        let D = x.dim(2)

        // 1. Project q, k, v.
        let q0 = query(x)
        let k0 = key(x)
        let v0 = value(x)

        // 2. FSMN over v (NLC -> NCL -> manual constant pad -> depthwise conv -> NLC -> residual).
        // Upstream applies pad_fn then conv (which expects channels-first). Our MLX Conv1d
        // operates in NLC so we keep things in NLC and pad along the time axis.
        var fsm = v0                                          // [B, T, D]
        let padL = MLXArray.zeros([B, leftPad, D]).asType(fsm.dtype)
        let padR = MLXArray.zeros([B, rightPad, D]).asType(fsm.dtype)
        fsm = concatenated([padL, fsm, padR], axis: 1)        // [B, T + 30, D]
        fsm = fsmnBlock(fsm)                                  // [B, T, D] (groups=D depthwise)
        let fsmMemory = fsm + v0                              // residual

        // 3. RoPE + multi-head attention (per-head shape: [B, n_head, T, head_dim]).
        let qHeads = q0.reshaped([B, T, nHead, headDim])
        let kHeads = k0.reshaped([B, T, nHead, headDim])
        let vHeads = v0.reshaped([B, T, nHead, headDim])
            .transposed(0, 2, 1, 3)                            // [B, n_head, T, head_dim]

        // MLX RoPE expects (..., T, head_dim) with last two dims being [seq, dim].
        // We compute on shape [B*n_head, T, head_dim] then reshape back.
        let qFlat = qHeads.transposed(0, 2, 1, 3)              // [B, n_head, T, head_dim]
            .reshaped([B * nHead, T, headDim])
        let kFlat = kHeads.transposed(0, 2, 1, 3)
            .reshaped([B * nHead, T, headDim])
        let qRot = rope(qFlat).reshaped([B, nHead, T, headDim])
        let kRot = rope(kFlat).reshaped([B, nHead, T, headDim])

        // 4. Scaled dot-product attention (MLXFast fused kernel).
        let attn = MLXFast.scaledDotProductAttention(
            queries: qRot,
            keys: kRot,
            values: vHeads,
            scale: scale,
            mask: nil as MLXArray?
        )                                                       // [B, n_head, T, head_dim]

        // 5. Merge heads + out projection + FSMN residual.
        let merged = attn.transposed(0, 2, 1, 3).reshaped([B, T, D])
        return out(merged) + fsmMemory
    }
}

// MARK: - Pre-norm transformer block

/// Pre-norm transformer block from s3tokenizer-v3:
///   x = x + attn(LN(x))
///   x = x + Linear(GELU(Linear(LN(x))))
///
/// Upstream stores the MLP as `nn.Sequential(Linear, GELU(), Linear)`, so the
/// safetensors keys are `mlp.0.weight/bias` (in→hidden) and `mlp.2.weight/bias`
/// (hidden→out). We expose them as `mlpFc1` / `mlpFc2` and let the weight
/// loader translate the names — Sequential isn't used anywhere else in this
/// package and would just add ceremony for two Linears.
public class ResidualAttentionBlockV3: Module {
    @ModuleInfo var attn: FSMNMultiHeadAttention
    @ModuleInfo(key: "attn_ln") var attnLN: LayerNorm
    @ModuleInfo var mlpFc1: Linear
    @ModuleInfo var mlpFc2: Linear
    @ModuleInfo(key: "mlp_ln") var mlpLN: LayerNorm

    public init(config: SpeechTokenizerConfig) {
        self._attn.wrappedValue = FSMNMultiHeadAttention(config: config)
        self._attnLN.wrappedValue = LayerNorm(dimensions: config.nAudioState, eps: 1e-5)
        self._mlpFc1.wrappedValue = Linear(config.nAudioState, config.mlpDim)
        self._mlpFc2.wrappedValue = Linear(config.mlpDim, config.nAudioState)
        self._mlpLN.wrappedValue = LayerNorm(dimensions: config.nAudioState, eps: 1e-5)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, rope: MLXNN.RoPE) -> MLXArray {
        var h = x + attn(attnLN(x), rope: rope)
        h = h + mlpFc2(gelu(mlpFc1(mlpLN(h))))
        return h
    }
}

// MARK: - Audio encoder

/// Two-stage stride-2 Conv1d subsampler + 12 transformer blocks. Input is `[B, n_mels, T]`
/// (the same channels-first layout `s3tokenizer.AudioEncoderV3` expects); output is
/// `[B, T / 4, n_state]` so a 30 s clip (3000 mel frames at 100 Hz) becomes 750 hidden frames
/// (25 Hz).
public class AudioEncoderV3: Module {
    public let config: SpeechTokenizerConfig

    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var blocks: [ResidualAttentionBlockV3]

    private let rope: MLXNN.RoPE

    public init(config: SpeechTokenizerConfig) {
        self.config = config

        self._conv1.wrappedValue = Conv1d(
            inputChannels: config.nMels, outputChannels: config.nAudioState,
            kernelSize: 3, stride: config.subsampleStride1, padding: 1, bias: true
        )
        self._conv2.wrappedValue = Conv1d(
            inputChannels: config.nAudioState, outputChannels: config.nAudioState,
            kernelSize: 3, stride: config.subsampleStride2, padding: 1, bias: true
        )
        self._blocks.wrappedValue = (0..<config.nAudioLayer).map { _ in
            ResidualAttentionBlockV3(config: config)
        }

        // Split-half RoPE matching `s3tokenizer.model_v2.apply_rotary_emb`. The upstream
        // concatenates `freqs_cis` with itself along dim=-1 — i.e. the same 32 frequencies
        // serve both halves of the 64-dim head, which is exactly what
        // `MLXNN.RoPE(dimensions: 64, traditional: false)` produces.
        self.rope = MLXNN.RoPE(
            dimensions: config.headDim, traditional: false, base: config.ropeBase)

        super.init()
    }

    /// - Parameter mel: `[B, n_mels, T]` log-mel spectrogram (Whisper conventions, T at 100 Hz).
    /// - Returns: `[B, T / 4, n_state]` hidden states at 25 Hz.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // MLX Conv1d wants NLC. Upstream's input is NCL, transpose once on the way in.
        var h = mel.transposed(0, 2, 1)                  // [B, T, n_mels]
        h = gelu(conv1(h))                                // [B, T / s1, n_state]
        h = gelu(conv2(h))                                // [B, T / s1 / s2, n_state]
        for block in blocks {
            h = block(h, rope: rope)
        }
        return h
    }
}

// MARK: - End-to-end speech tokenizer

/// The full S3-Tokenizer-v3 (encoder + FSQ quantizer). Used in CosyVoice 3 zero-shot
/// voice cloning to encode a reference audio clip into the FSQ-code stream the flow
/// model consumes as `prompt_token`.
public final class SpeechTokenizerModel: Module {
    public let config: SpeechTokenizerConfig

    @ModuleInfo var encoder: AudioEncoderV3
    @ModuleInfo var quantizer: FSQVectorQuantizer

    public init(config: SpeechTokenizerConfig = SpeechTokenizerConfig()) {
        self.config = config
        self._encoder.wrappedValue = AudioEncoderV3(config: config)
        self._quantizer.wrappedValue = FSQVectorQuantizer(config: config)
        super.init()
    }

    /// Encode a log-mel spectrogram to FSQ codes.
    /// - Parameter mel: `[B, n_mels=128, T_mel]` Whisper-style log-mel
    /// - Returns: `[B, T_mel / 4]` integer FSQ codes in `[0, 6561)` (25 Hz at 16 kHz / 160-hop input)
    public func encode(mel: MLXArray) -> MLXArray {
        let hidden = encoder(mel)              // [B, T/4, n_state]
        return quantizer.encode(hidden)         // [B, T/4]
    }

    /// Load weights from `speech_tokenizer.safetensors` produced by the conversion
    /// script. Pure convenience over `CosyVoiceWeightLoader.loadSpeechTokenizer`.
    public static func fromSafetensors(at url: URL) throws -> SpeechTokenizerModel {
        let model = SpeechTokenizerModel()
        try CosyVoiceWeightLoader.loadSpeechTokenizer(model, from: url)
        return model
    }
}
