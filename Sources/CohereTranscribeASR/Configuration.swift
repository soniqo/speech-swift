import Foundation

public struct CohereMLXQuantization: Codable, Sendable, Equatable {
    public let bits: Int
    public let groupSize: Int
    public let mode: String?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case mode
    }

    public init(bits: Int, groupSize: Int, mode: String? = "affine") {
        self.bits = bits
        self.groupSize = groupSize
        self.mode = mode
    }
}

public struct CohereTranscribeAudioEncoderConfig: Codable, Sendable, Equatable {
    public let dModel: Int
    public let ffExpansionFactor: Int
    public let nHeads: Int
    public let convKernelSize: Int
    public let nLayers: Int
    public let posEmbMaxLen: Int
    public let subsamplingConvChannels: Int
    public let subsamplingFactor: Int
    public let featIn: Int

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case ffExpansionFactor = "ff_expansion_factor"
        case nHeads = "n_heads"
        case convKernelSize = "conv_kernel_size"
        case nLayers = "n_layers"
        case posEmbMaxLen = "pos_emb_max_len"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case subsamplingFactor = "subsampling_factor"
        case featIn = "feat_in"
    }
}

public struct CohereTranscribeTextDecoderConfig: Codable, Sendable, Equatable {
    public let hiddenSize: Int
    public let innerSize: Int
    public let numAttentionHeads: Int
    public let numLayers: Int
    public let maxSequenceLength: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case innerSize = "inner_size"
        case numAttentionHeads = "num_attention_heads"
        case numLayers = "num_layers"
        case maxSequenceLength = "max_sequence_length"
    }
}

private struct CohereTranscribeHeadConfig: Decodable {
    let numClasses: Int

    enum CodingKeys: String, CodingKey {
        case numClasses = "num_classes"
    }
}

public struct CohereTranscribeConfig: Decodable, Sendable {
    public let modelType: String
    public let vocabSize: Int
    public let sampleRate: Int
    public let maxAudioClipS: Int
    public let overlapChunkSecond: Double
    public let minEnergyWindowSamples: Int
    public let encoder: CohereTranscribeAudioEncoderConfig
    public let decoder: CohereTranscribeTextDecoderConfig
    public let quantization: CohereMLXQuantization?

    private struct DecoderWrapper: Decodable {
        let configDict: CohereTranscribeTextDecoderConfig

        enum CodingKeys: String, CodingKey {
            case configDict = "config_dict"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case sampleRate = "sample_rate"
        case maxAudioClipS = "max_audio_clip_s"
        case overlapChunkSecond = "overlap_chunk_second"
        case minEnergyWindowSamples = "min_energy_window_samples"
        case encoder
        case transfDecoder = "transf_decoder"
        case head
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        if let topLevel = try container.decodeIfPresent(Int.self, forKey: .vocabSize) {
            vocabSize = topLevel
        } else {
            vocabSize = try container.decode(CohereTranscribeHeadConfig.self, forKey: .head).numClasses
        }
        sampleRate = try container.decode(Int.self, forKey: .sampleRate)
        maxAudioClipS = try container.decode(Int.self, forKey: .maxAudioClipS)
        overlapChunkSecond = try container.decodeIfPresent(
            Double.self, forKey: .overlapChunkSecond) ?? 5
        minEnergyWindowSamples = try container.decodeIfPresent(
            Int.self, forKey: .minEnergyWindowSamples) ?? 1_600
        encoder = try container.decode(CohereTranscribeAudioEncoderConfig.self, forKey: .encoder)
        self.decoder = try container.decode(DecoderWrapper.self, forKey: .transfDecoder).configDict
        quantization = try container.decodeIfPresent(CohereMLXQuantization.self, forKey: .quantization)
            ?? container.decodeIfPresent(CohereMLXQuantization.self, forKey: .quantizationConfig)
    }
}

public enum CohereTranscribeVariant: String, CaseIterable, Sendable {
    case fp16
    case int5
    case int8

    public var modelId: String {
        switch self {
        case .fp16: return "aufklarer/Cohere-Transcribe-2B-MLX-FP16"
        case .int5: return "aufklarer/Cohere-Transcribe-2B-MLX-5bit"
        case .int8: return "aufklarer/Cohere-Transcribe-2B-MLX-8bit"
        }
    }
}

public struct CohereTranscribeDecodingOptions: Sendable {
    public var maxTokens: Int
    public var language: String
    public var chunkDuration: Double

    public init(maxTokens: Int = 512, language: String = "en", chunkDuration: Double = 120) {
        self.maxTokens = maxTokens
        self.language = language
        self.chunkDuration = chunkDuration
    }
}
