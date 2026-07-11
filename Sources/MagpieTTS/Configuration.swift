import Foundation

/// Magpie-TTS Multilingual 357M variant. Weights are dequantised to FP at load
/// time — runtime is FP32 everywhere. int4 was decommissioned for TTS (too much
/// quality loss); INT8 is the floor (no bf16 bundle exists for Magpie).
public enum MagpieTTSVariant: String, Sendable, CaseIterable {
    case int8

    public var huggingFaceRepoId: String {
        switch self {
        case .int8: return "aufklarer/Magpie-TTS-Multilingual-357M-MLX-8bit"
        }
    }

    public var bits: Int {
        switch self {
        case .int8: return 8
        }
    }
}

/// Baked speaker identity. The model checkpoint embeds five speaker contexts
/// (110 frames × 768 dim) used as the prefix of every AR decode.
public enum MagpieSpeaker: Int, Sendable, CaseIterable {
    case sofia      = 0
    case aria       = 1
    case jason      = 2
    case leo        = 3
    case johnVanStan = 4

    public var displayName: String {
        switch self {
        case .sofia:       return "Sofia"
        case .aria:        return "Aria"
        case .jason:       return "Jason"
        case .leo:         return "Leo"
        case .johnVanStan: return "John Van Stan"
        }
    }

    public init?(named: String) {
        switch named.lowercased() {
        case "sofia":                       self = .sofia
        case "aria":                        self = .aria
        case "jason":                       self = .jason
        case "leo":                         self = .leo
        case "john", "john van stan", "johnvanstan", "john_van_stan":
            self = .johnVanStan
        default: return nil
        }
    }
}

/// Magpie supports nine on-device languages. The first eight ship a JSON
/// tokenizer in the model bundle; Japanese is tokenised via Apple's
/// `CFStringTokenizer` (no shipped dictionary).
public enum MagpieLanguage: String, Sendable, CaseIterable {
    case english    = "en"
    case spanish    = "es"
    case german     = "de"
    case french     = "fr"
    case italian    = "it"
    case vietnamese = "vi"
    case chinese    = "zh"
    case hindi      = "hi"
    case japanese   = "ja"

    public var displayName: String {
        switch self {
        case .english:    return "English"
        case .spanish:    return "Spanish"
        case .german:     return "German"
        case .french:     return "French"
        case .italian:    return "Italian"
        case .vietnamese: return "Vietnamese"
        case .chinese:    return "Chinese (Mandarin)"
        case .hindi:      return "Hindi"
        case .japanese:   return "Japanese"
        }
    }

    public init?(code: String) {
        let c = code.lowercased()
        for lang in MagpieLanguage.allCases where lang.rawValue == c {
            self = lang
            return
        }
        switch c {
        case "english":               self = .english
        case "spanish", "castellano": self = .spanish
        case "german", "deutsch":     self = .german
        case "french", "français":    self = .french
        case "italian", "italiano":   self = .italian
        case "vietnamese":            self = .vietnamese
        case "chinese", "mandarin", "cmn": self = .chinese
        case "hindi":                 self = .hindi
        case "japanese":              self = .japanese
        default: return nil
        }
    }
}

// MARK: - Bundle config.json structs

public struct MagpieQuantization: Codable, Sendable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String
    enum CodingKeys: String, CodingKey {
        case bits, mode
        case groupSize = "group_size"
    }
}

public struct MagpieTextEncoderConfig: Codable, Sendable {
    public let vocabSize: Int
    public let dModel: Int
    public let dFfn: Int
    public let nLayers: Int
    public let nHeads: Int
    public let kernelSize: Int
    public let maxLen: Int
    public let quantization: MagpieQuantization?
    public let quantizedShapes: [String: [Int]]?
    enum CodingKeys: String, CodingKey {
        case vocabSize       = "vocab_size"
        case dModel          = "d_model"
        case dFfn            = "d_ffn"
        case nLayers         = "n_layers"
        case nHeads          = "n_heads"
        case kernelSize      = "kernel_size"
        case maxLen          = "max_len"
        case quantization
        case quantizedShapes = "quantized_shapes"
    }
}

public struct MagpieLocalTransformerConfig: Codable, Sendable {
    public let nLayers: Int
    public let nHeads: Int
    public let dModel: Int
    enum CodingKeys: String, CodingKey {
        case nLayers = "n_layers"
        case nHeads  = "n_heads"
        case dModel  = "d_model"
    }
}

public struct MagpieDecoderConfig: Codable, Sendable {
    public let dModel: Int
    public let dFfn: Int
    public let nLayers: Int
    public let nHeads: Int
    public let kernelSize: Int
    public let xaDMemory: Int
    public let xaNHeads: Int
    public let xaDHead: Int
    public let maxLen: Int
    public let numCodebooks: Int
    public let vocabPerCodebook: Int
    public let audioBosId: Int
    public let audioEosId: Int
    public let numBakedSpeakers: Int
    public let bakedT: Int
    public let localTransformer: MagpieLocalTransformerConfig
    public let quantization: MagpieQuantization?
    public let quantizedShapes: [String: [Int]]?
    enum CodingKeys: String, CodingKey {
        case dModel           = "d_model"
        case dFfn             = "d_ffn"
        case nLayers          = "n_layers"
        case nHeads           = "n_heads"
        case kernelSize       = "kernel_size"
        case xaDMemory        = "xa_d_memory"
        case xaNHeads         = "xa_n_heads"
        case xaDHead          = "xa_d_head"
        case maxLen           = "max_len"
        case numCodebooks     = "num_codebooks"
        case vocabPerCodebook = "vocab_per_codebook"
        case audioBosId       = "audio_bos_id"
        case audioEosId       = "audio_eos_id"
        case numBakedSpeakers = "num_baked_speakers"
        case bakedT           = "baked_T"
        case localTransformer = "local_transformer"
        case quantization
        case quantizedShapes  = "quantized_shapes"
    }
}

public struct MagpieNanoCodecConfig: Codable, Sendable {
    public let sampleRate: Int
    public let samplesPerFrame: Int
    public let numCodebooks: Int
    public let vocabPerCodebook: Int
    public let fsqNumLevels: [Int]
    public let upSampleRates: [Int]
    public let baseChannels: Int
    public let quantization: MagpieQuantization?
    public let quantizedShapes: [String: [Int]]?
    enum CodingKeys: String, CodingKey {
        case sampleRate       = "sample_rate"
        case samplesPerFrame  = "samples_per_frame"
        case numCodebooks     = "num_codebooks"
        case vocabPerCodebook = "vocab_per_codebook"
        case fsqNumLevels     = "fsq_num_levels"
        case upSampleRates    = "up_sample_rates"
        case baseChannels     = "base_channels"
        case quantization
        case quantizedShapes  = "quantized_shapes"
    }
}

// MARK: - Sampling parameters

public struct MagpieTTSParams: Sendable {
    public var temperature: Float
    public var topK: Int
    /// Hard cap on AR frames (1 frame = 1/21.5 s ≈ 46 ms; 500 ≈ 23 s).
    public var maxSteps: Int
    /// Frames forced before EOS is allowed (avoids zero-length outputs).
    public var minFrames: Int
    public var seed: UInt64?

    public init(
        temperature: Float = 0.6,
        topK: Int = 80,
        maxSteps: Int = 500,
        minFrames: Int = 4,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.maxSteps = maxSteps
        self.minFrames = minFrames
        self.seed = seed
    }
}

// MARK: - Errors

public enum MagpieTTSError: Error, LocalizedError {
    case missingFile(String)
    case missingBundleDir(String)
    case weightLoadFailed(String)
    case unsupportedLanguage(String)
    case unsupportedSpeaker(Int)
    case textEncodingFailed(String)
    case invalidConfig(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f):       return "Magpie: missing file \(f)"
        case .missingBundleDir(let d):  return "Magpie: missing bundle directory \(d)"
        case .weightLoadFailed(let m):  return "Magpie: weight load failed: \(m)"
        case .unsupportedLanguage(let l): return "Magpie: language \(l) not supported"
        case .unsupportedSpeaker(let s): return "Magpie: invalid baked speaker index \(s) (valid: 0..4)"
        case .textEncodingFailed(let m): return "Magpie: text encoding failed: \(m)"
        case .invalidConfig(let m):     return "Magpie: invalid config: \(m)"
        }
    }
}
