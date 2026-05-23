import Foundation
import MagpieTTS

/// Constants for the FluidInference CoreML bundle. Most of these match the MLX
/// module's runtime values exactly; differences (speaker ordering, hard frame
/// cap from the codec's fixed-window export) are called out below.
public enum MagpieCoreMLConstants {
    public static let sampleRate: Int = 22_050
    public static let samplesPerFrame: Int = 1_024
    public static let framesPerSecond: Double = 22_050.0 / 1_024.0  // ~21.5

    public static let dModel: Int = 768
    public static let numDecoderLayers: Int = 12
    public static let numHeads: Int = 12
    public static let headDim: Int = 64
    /// `decoder_step` was exported with a 512-frame KV cache window.
    public static let maxCacheFrames: Int = 512
    /// `text_encoder` was exported with a fixed 256-token input.
    public static let maxTextTokens: Int = 256

    public static let numCodebooks: Int = 8
    public static let numCodesPerCodebook: Int = 2_024
    public static let audioBosId: Int32 = 2_016
    public static let audioEosId: Int32 = 2_017
    public static let forbiddenAudioIds: [Int32] = [2_016, 2_018, 2_019, 2_020, 2_021, 2_022, 2_023]

    public static let speakerContextLength: Int = 110
    public static let numSpeakers: Int = 5

    public static let localTransformerDim: Int = 256
    public static let localTransformerFfnDim: Int = 1_024
    public static let localTransformerMaxPositions: Int = 10

    /// Hard upper bound on codec frames per call. The FluidInference NanoCodec
    /// model is exported as a fixed-window batch decoder; their manifest
    /// documents that overlap-window streaming yields <15 dB SNR and is not a
    /// viable workaround, so we hard-cap generation rather than chunk.
    public static let maxNanocodecFrames: Int = 256
    public static let maxAudioSeconds: Double = Double(maxNanocodecFrames) / framesPerSecond

    public static let huggingFaceRepo = "FluidInference/magpie-tts-multilingual-357m-coreml"
}

/// CoreML bundle exposes 5 baked speakers, but in a different order than the
/// MLX bundle. We expose a CoreML-flavoured enum so the speaker index passed
/// into `decoder_prefill`/`decoder_step` is always the bundle-native index,
/// while ``mlxSpeaker`` lets the CLI's auto-fallback handoff pick the right
/// MLX-side speaker when the engine swaps to MLX for Japanese.
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

    /// MLX-bundle equivalent. Used by the CLI to swap to the MLX engine when
    /// `--language ja` is requested with `--engine magpie-coreml`.
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

/// CoreML bundle does not ship a Japanese G2P / tokenizer. The CLI handles JA
/// by falling back to the MLX engine.
public enum MagpieCoreMLLanguage: String, Sendable, CaseIterable {
    case english    = "en"
    case spanish    = "es"
    case german     = "de"
    case french     = "fr"
    case italian    = "it"
    case vietnamese = "vi"
    case chinese    = "zh"
    case hindi      = "hi"

    /// Bridge to the MLX module's language enum (reuses Tokenizer + G2P).
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

    /// Convert from the MLX enum. Returns nil for Japanese (caller must route
    /// to the MLX backend).
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

/// Sampling parameters. Mirrors ``MagpieTTSParams`` from the MLX module so
/// callers can use either backend with the same defaults.
public struct MagpieCoreMLParams: Sendable {
    public var temperature: Float
    public var topK: Int
    public var maxSteps: Int
    public var minFrames: Int
    /// Classifier-free guidance scale. The CoreML pipeline supports CFG via a
    /// separate unconditional decoder pass; `1.0` disables it (single
    /// `decoder_step` call per frame). The reference bundle ships `2.5` as its
    /// quality preset but doubles wall time.
    public var cfgScale: Float
    public var seed: UInt64?

    public init(
        temperature: Float = 0.6,
        topK: Int = 80,
        maxSteps: Int = MagpieCoreMLConstants.maxNanocodecFrames,
        minFrames: Int = 4,
        cfgScale: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        // The MLX pipeline allows up to 500 frames; the CoreML codec is
        // window-limited to 256. We default to the codec's hard cap so callers
        // get full-quality batch output without surprise truncation.
        self.maxSteps = min(maxSteps, MagpieCoreMLConstants.maxNanocodecFrames)
        self.minFrames = minFrames
        self.cfgScale = cfgScale
        self.seed = seed
    }
}

/// `constants/constants.json` shipped with the FluidInference bundle. Only the
/// subset we actually read at runtime.
public struct MagpieCoreMLBundleConstants: Codable, Sendable {
    public let embeddingDim: Int
    public let numAudioCodebooks: Int
    public let codebookSize: Int
    public let numAllTokensPerCodebook: Int
    public let sampleRate: Int
    public let codecSamplesPerFrame: Int
    public let specialTokens: SpecialTokens
    public let inference: InferenceDefaults

    public struct SpecialTokens: Codable, Sendable {
        public let audioBosId: Int
        public let audioEosId: Int
        public let textBosId: Int
        public let textEosId: Int
        enum CodingKeys: String, CodingKey {
            case audioBosId = "audio_bos_id"
            case audioEosId = "audio_eos_id"
            case textBosId  = "text_bos_id"
            case textEosId  = "text_eos_id"
        }
    }

    public struct InferenceDefaults: Codable, Sendable {
        public let temperature: Float
        public let topk: Int
        public let cfgScale: Float
        public let maxDecoderSteps: Int
        public let minGeneratedFrames: Int
        enum CodingKeys: String, CodingKey {
            case temperature, topk
            case cfgScale           = "cfg_scale"
            case maxDecoderSteps    = "max_decoder_steps"
            case minGeneratedFrames = "min_generated_frames"
        }
    }

    enum CodingKeys: String, CodingKey {
        case embeddingDim            = "embedding_dim"
        case numAudioCodebooks       = "num_audio_codebooks"
        case codebookSize            = "codebook_size"
        case numAllTokensPerCodebook = "num_all_tokens_per_codebook"
        case sampleRate              = "sample_rate"
        case codecSamplesPerFrame    = "codec_samples_per_frame"
        case specialTokens           = "special_tokens"
        case inference
    }
}

/// `constants/speaker_info.json` shipped with the FluidInference bundle.
public struct MagpieCoreMLSpeakerInfo: Codable, Sendable {
    public let numSpeakers: Int
    public let t: Int  // context length
    public let d: Int  // model dim
    public let names: [String: String]

    enum CodingKeys: String, CodingKey {
        case numSpeakers = "num_speakers"
        case t = "T"
        case d = "D"
        case names
    }
}
