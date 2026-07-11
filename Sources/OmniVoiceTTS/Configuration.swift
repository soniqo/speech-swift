import Foundation

/// OmniVoice — massively multilingual (600+ languages) zero-shot TTS.
///
/// A single-stage NAR **discrete-diffusion** model: a Qwen3 transformer backbone,
/// run **bidirectionally** (no causal mask), maps interleaved text + 8-codebook
/// audio tokens and predicts the masked acoustic tokens. Generation iteratively
/// unmasks over N steps with classifier-free guidance; a Higgs-audio v2 codec
/// (semantic wav2vec2 + acoustic DAC RVQ) encodes the reference and decodes the
/// predicted tokens to a 24 kHz waveform.
///
/// Source: `k2-fsa/OmniVoice` (paper arXiv:2604.00688). MLX weights:
/// `aufklarer/OmniVoice-MLX-fp16`.
public struct OmniVoiceConfig: Sendable {
    // MARK: Qwen3 backbone (`llm.*` in the checkpoint)
    public var numLayers: Int = 28
    public var hiddenSize: Int = 1024
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 8          // GQA: 16 query / 8 KV heads
    public var headDim: Int = 128
    public var intermediateSize: Int = 3072        // SwiGLU
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 1_000_000
    public var maxPositionEmbeddings: Int = 40960
    /// Qwen3 applies RMSNorm to per-head Q and K before RoPE.
    public var useQKNorm: Bool = true
    public var textVocabSize: Int = 151_676        // `llm.embed_tokens` rows

    // MARK: Audio tokens
    /// 8 acoustic codebooks; each token in `0..<1024`, with `1024` = mask.
    public var numAudioCodebook: Int = 8
    public var audioVocabSize: Int = 1025          // 1024 codes + mask
    public var audioMaskId: Int = 1024
    /// Per-codebook confidence weights used by the unmasking schedule.
    public var audioCodebookWeights: [Float] = [8, 8, 6, 6, 4, 4, 2, 2]
    /// `audio_embeddings` / `audio_heads` are sized `numAudioCodebook * audioVocabSize`.
    public var audioMatrixRows: Int { numAudioCodebook * audioVocabSize }   // 8200

    // MARK: Generation (diffusion unmasking)
    /// 32 steps is the reference default; 16 is near-identical quality at ~2x
    /// speed (paper Table 9), so we default to 16.
    public var numInferenceSteps: Int = 16
    public var guidanceScale: Float = 2.0
    /// Temperature applied to confidence scores when picking which positions to
    /// unmask each step.
    public var confidenceTemperature: Float = 5.0
    /// Time-shift schedule parameter for the cumulative unmask ratio.
    public var scheduleShift: Float = 0.1
    public var eosTokenId: Int = 151_645

    // MARK: Codec
    public var sampleRate: Int = 24000

    public init() {}

    /// HF repo id of the published fp16 MLX bundle.
    public static let defaultModelId = "aufklarer/OmniVoice-MLX-fp16"
}
