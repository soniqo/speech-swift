import Foundation
import MagpieTTS

/// Constants for the soniqo CoreML Magpie bundle
/// (``aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit``). Most values
/// match the MLX runtime exactly; the codec-window cap and the FSQ levels
/// are CoreML-bundle-specific.
public enum MagpieCoreMLConstants {
    public static let sampleRate: Int = 22_050
    public static let samplesPerFrame: Int = 1_024
    public static let framesPerSecond: Double = 22_050.0 / 1_024.0  // ~21.5

    public static let dModel: Int = 768
    public static let numDecoderLayers: Int = 12
    /// Self-attention cache window baked into ``decoder_step.mlmodelc``
    /// (110-frame speaker context + 1 BOS + 500 max generated frames - 1).
    public static let saCacheLength: Int = 600
    /// Cross-attention K/V is precomputed once by `decoder_prefill` and
    /// reused at every step — shape `(1, 256, 1, 128)` per layer.
    public static let xaContextLength: Int = 256
    public static let xaInnerDim: Int = 128
    public static let maxTextTokens: Int = 256
    public static let speakerContextLength: Int = 110
    public static let numSpeakers: Int = 5
    public static let saSelfHeads: Int = 12
    public static let saHeadDim: Int = 64

    public static let numCodebooks: Int = 8
    public static let numCodesPerCodebook: Int = 2_024
    public static let audioBosId: Int32 = 2_016
    public static let audioEosId: Int32 = 2_017
    public static let forbiddenAudioIds: [Int32] = [2_016, 2_018, 2_019, 2_020, 2_021, 2_022, 2_023]

    /// 1-layer LocalTransformer (NeMo's codebook sampling head). d=256,
    /// FFN=1024, 1 attention head; positional embedding slots for the 8
    /// codebooks + spare BOS slots.
    public static let localTransformerDim: Int = 256
    public static let localTransformerFfnDim: Int = 1_024
    public static let localTransformerMaxPositions: Int = 10

    /// FSQ inverse: each codebook value `i ∈ [0, 2016)` decodes to 4
    /// dequant scalars via `(i // base[j]) % level[j]`, mapped to
    /// `(d_j - L_j/2) / (L_j/2)`. 8 codebooks × 4 scalars = 32 latent dims.
    public static let fsqLevels: [Int32] = [8, 7, 6, 6]
    public static let fsqBase:   [Int32] = [1, 8, 56, 336]
    public static let fsqDimPerGroup: Int = 4
    public static let fsqNumGroups: Int = 8
    public static let fsqLatentDim: Int = 32   // 4 × 8

    /// Window the nano-codec consumes per call. The model was traced at 64
    /// frames; output is `(1, 64 * 1024) = (1, 65536)` samples ≈ 2.97 s.
    public static let nanocodecFramesPerWindow: Int = 64
    public static let nanocodecAudioPerWindow: Int = 65_536  // 64 * 1024

    /// Hard upper bound on AR frames per utterance. Matches MLX defaults so
    /// callers don't see a difference in `--magpie-max-frames`.
    public static let maxARSteps: Int = 500

    /// HuggingFace repo holding the 4 compiled CoreML packages.
    public static let huggingFaceRepo = "aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit"
}

/// CoreML bundle's baked speaker order:
///   0=John, 1=Sofia, 2=Aria, 3=Jason, 4=Leo
/// — must match the speaker_idx the decoder_prefill model selects on.
public enum MagpieCoreMLSpeaker: Int, Sendable, CaseIterable {
    case john  = 0
    case sofia = 1
    case aria  = 2
    case jason = 3
    case leo   = 4

    public var displayName: String {
        switch self {
        case .john:  return "John"
        case .sofia: return "Sofia"
        case .aria:  return "Aria"
        case .jason: return "Jason"
        case .leo:   return "Leo"
        }
    }

    public init?(named: String) {
        switch named.lowercased() {
        case "john", "john van stan", "johnvanstan", "john_van_stan": self = .john
        case "sofia": self = .sofia
        case "aria":  self = .aria
        case "jason": self = .jason
        case "leo":   self = .leo
        default: return nil
        }
    }

    /// MLX-bundle equivalent. Used by the CLI when `--language ja` is
    /// requested with `--engine magpie-coreml` — the CoreML bundle has no
    /// JA tokenizer JSON, so we route to the MLX backend and the speaker
    /// identity has to survive the handoff.
    public var mlxSpeaker: MagpieSpeaker {
        switch self {
        case .sofia: return .sofia
        case .aria:  return .aria
        case .jason: return .jason
        case .leo:   return .leo
        case .john:  return .johnVanStan
        }
    }
}

/// CoreML-supported languages. Japanese is excluded only because we don't
/// ship a JA tokenizer JSON inside the CoreML bundle — the model itself
/// supports JA. The CLI routes JA requests to the MLX backend.
public enum MagpieCoreMLLanguage: String, Sendable, CaseIterable {
    case english    = "en"
    case spanish    = "es"
    case german     = "de"
    case french     = "fr"
    case italian    = "it"
    case vietnamese = "vi"
    case chinese    = "zh"
    case hindi      = "hi"

    public var mlx: MagpieLanguage {
        switch self {
        case .english:    return .english
        case .spanish:    return .spanish
        case .german:     return .german
        case .french:     return .french
        case .italian:    return .italian
        case .vietnamese: return .vietnamese
        case .chinese:    return .chinese
        case .hindi:      return .hindi
        }
    }

    public init?(mlx: MagpieLanguage) {
        switch mlx {
        case .english:    self = .english
        case .spanish:    self = .spanish
        case .german:     self = .german
        case .french:     self = .french
        case .italian:    self = .italian
        case .vietnamese: self = .vietnamese
        case .chinese:    self = .chinese
        case .hindi:      self = .hindi
        case .japanese:   return nil
        }
    }
}

/// Sampling parameters for the parallel codebook head. The CoreML bundle's
/// `decoder_step` exposes `logits: (1, 1, 8, 2024)` directly — we sample 8
/// codebooks in parallel rather than running a per-frame LocalTransformer
/// AR loop (see ``MagpieTTSCoreML`` for context).
public struct MagpieCoreMLParams: Sendable {
    public var temperature: Float
    public var topK: Int
    public var maxSteps: Int
    public var minFrames: Int
    public var seed: UInt64?

    public init(
        temperature: Float = 0.6,
        topK: Int = 80,
        maxSteps: Int = MagpieCoreMLConstants.maxARSteps,
        minFrames: Int = 4,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.maxSteps = min(maxSteps, MagpieCoreMLConstants.maxARSteps)
        self.minFrames = minFrames
        self.seed = seed
    }
}
