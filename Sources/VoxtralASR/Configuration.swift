import Foundation

public struct VoxtralMLXQuantization: Codable, Sendable, Equatable {
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

public struct VoxtralAudioConfig: Decodable, Sendable, Equatable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let headDim: Int
    public let ropeTheta: Float
    public let vocabSize: Int
    public let numMelBins: Int
    public let maxSourcePositions: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1_280
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5_120
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 20
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 20
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 51_866
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1_500
    }
}

public struct VoxtralTextConfig: Decodable, Sendable, Equatable {
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let headDim: Int
    public let tieWordEmbeddings: Bool
    public let bosTokenID: Int
    public let eosTokenID: Int
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case ropeTheta = "rope_theta"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 131_072
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3_072
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8_192
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 1
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 2
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 100_000_000
    }
}

public struct VoxtralConfig: Decodable, Sendable {
    public let modelType: String
    public let audioConfig: VoxtralAudioConfig
    public let textConfig: VoxtralTextConfig
    public let audioTokenID: Int
    public let projectorHiddenAct: String
    public let vocabSize: Int
    public let hiddenSize: Int
    public let quantization: VoxtralMLXQuantization?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case audioTokenID = "audio_token_id"
        case projectorHiddenAct = "projector_hidden_act"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case quantization
        case quantizationConfig = "quantization_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral"
        audioConfig = try c.decode(VoxtralAudioConfig.self, forKey: .audioConfig)
        textConfig = try c.decode(VoxtralTextConfig.self, forKey: .textConfig)
        audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 24
        projectorHiddenAct = try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? textConfig.vocabSize
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? textConfig.hiddenSize
        quantization = try c.decodeIfPresent(VoxtralMLXQuantization.self, forKey: .quantization)
            ?? c.decodeIfPresent(VoxtralMLXQuantization.self, forKey: .quantizationConfig)
    }
}

public enum VoxtralVariant: String, CaseIterable, Sendable {
    case fp16
    case int5
    case int8

    public var modelId: String {
        switch self {
        case .fp16: return "aufklarer/Voxtral-Mini-3B-2507-MLX-FP16"
        case .int5: return "aufklarer/Voxtral-Mini-3B-2507-MLX-5bit"
        case .int8: return "aufklarer/Voxtral-Mini-3B-2507-MLX-8bit"
        }
    }
}

public struct VoxtralDecodingOptions: Sendable {
    public var maxTokens: Int
    public var language: String?

    public init(maxTokens: Int = 128, language: String? = nil) {
        self.maxTokens = maxTokens
        self.language = language
    }
}
