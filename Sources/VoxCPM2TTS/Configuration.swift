import Foundation

public struct RopeScalingConfig: Codable, Sendable {
    public var type: String = "longrope"
    public var shortFactor: [Float] = []
    public var longFactor: [Float] = []
    public var originalMaxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case type
        case shortFactor = "short_factor"
        case longFactor = "long_factor"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.type = try container.decodeIfPresent(String.self, forKey: .type) ?? "longrope"
        self.shortFactor = try container.decodeIfPresent([Float].self, forKey: .shortFactor) ?? []
        self.longFactor = try container.decodeIfPresent([Float].self, forKey: .longFactor) ?? []
        self.originalMaxPositionEmbeddings = try container.decodeIfPresent(
            Int.self,
            forKey: .originalMaxPositionEmbeddings
        ) ?? 32768
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(type, forKey: .type)
        try container.encode(shortFactor, forKey: .shortFactor)
        try container.encode(longFactor, forKey: .longFactor)
        try container.encode(originalMaxPositionEmbeddings, forKey: .originalMaxPositionEmbeddings)
    }
}

public struct LMConfig: Codable, Sendable {
    public var hiddenSize: Int = 2048
    public var numHiddenLayers: Int = 28
    public var numAttentionHeads: Int = 16
    public var numKeyValueHeads: Int = 2
    public var intermediateSize: Int = 6144
    public var vocabSize: Int = 73448
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 10000.0
    public var ropeScaling: RopeScalingConfig? = RopeScalingConfig()
    public var scaleEmb: Int = 12
    public var dimModelBase: Int = 256
    public var scaleDepth: Float = 1.4
    public var originalMaxPositionEmbeddings: Int = 32768
    public var maxPositionEmbeddings: Int = 32768
    public var bosTokenId: Int = 1
    public var eosTokenId: Int = 2
    public var useMup: Bool = false
    public var kvChannels: Int? = 128
    public var noRope: Bool = false

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case scaleEmb = "scale_emb"
        case dimModelBase = "dim_model_base"
        case scaleDepth = "scale_depth"
        case maxPositionEmbeddings = "max_position_embeddings"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case useMup = "use_mup"
        case kvChannels = "kv_channels"
    }

    public init() {}
}

public struct EncoderConfig: Codable, Sendable {
    public var hiddenDim: Int = 1024
    public var ffnDim: Int = 4096
    public var numHeads: Int = 16
    public var numLayers: Int = 12
    public var kvChannels: Int? = 128

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
    }

    public init() {}
}

public struct CFMConfig: Codable, Sendable {
    public var sigmaMin: Float = 1e-6
    public var solver: String = "euler"
    public var tScheduler: String = "log-norm"
    public var inferenceCfgRate: Float = 2.0

    enum CodingKeys: String, CodingKey {
        case sigmaMin = "sigma_min"
        case solver
        case tScheduler = "t_scheduler"
        case inferenceCfgRate = "inference_cfg_rate"
    }

    public init() {}
}

public struct DiTConfig: Codable, Sendable {
    public var hiddenDim: Int = 1024
    public var ffnDim: Int = 4096
    public var numHeads: Int = 16
    public var numLayers: Int = 12
    public var kvChannels: Int? = 128
    public var ditMeanMode: Bool = false
    public var cfmConfig: CFMConfig = CFMConfig()

    enum CodingKeys: String, CodingKey {
        case hiddenDim = "hidden_dim"
        case ffnDim = "ffn_dim"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case kvChannels = "kv_channels"
        case ditMeanMode = "mean_mode"
        case cfmConfig = "cfm_config"
    }

    public init() {}
}

public struct AudioVAEConfig: Codable, Sendable {
    public var encoderDim: Int = 128
    public var encoderRates: [Int] = [2, 5, 8, 8]
    public var latentDim: Int = 64
    public var decoderDim: Int = 2048
    public var decoderRates: [Int] = [8, 6, 5, 2, 2, 2]
    public var depthwise: Bool = true
    public var sampleRate: Int = 16000
    public var outSampleRate: Int = 48000
    public var useNoiseBlock: Bool = false
    public var srBinBoundaries: [Int] = [20000, 30000, 40000]
    public var condType: String = "scale_bias"
    public var condDim: Int = 128
    public var condOutLayer: Bool = false

    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case encoderRates = "encoder_rates"
        case latentDim = "latent_dim"
        case decoderDim = "decoder_dim"
        case decoderRates = "decoder_rates"
        case depthwise
        case sampleRate = "sample_rate"
        case outSampleRate = "out_sample_rate"
        case useNoiseBlock = "use_noise_block"
        case srBinBoundaries = "sr_bin_boundaries"
        case condType = "cond_type"
        case condDim = "cond_dim"
        case condOutLayer = "cond_out_layer"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.encoderDim = try container.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 128
        self.encoderRates = try container.decodeIfPresent([Int].self, forKey: .encoderRates) ?? [2, 5, 8, 8]
        self.latentDim = try container.decodeIfPresent(Int.self, forKey: .latentDim) ?? 64
        self.decoderDim = try container.decodeIfPresent(Int.self, forKey: .decoderDim) ?? 2048
        self.decoderRates = try container.decodeIfPresent([Int].self, forKey: .decoderRates) ?? [8, 6, 5, 2, 2, 2]
        self.depthwise = try container.decodeIfPresent(Bool.self, forKey: .depthwise) ?? true
        self.sampleRate = try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 16000
        self.outSampleRate = try container.decodeIfPresent(Int.self, forKey: .outSampleRate) ?? 48000
        self.useNoiseBlock = try container.decodeIfPresent(Bool.self, forKey: .useNoiseBlock) ?? false
        self.srBinBoundaries = try container.decodeIfPresent([Int].self, forKey: .srBinBoundaries) ?? [20000, 30000, 40000]
        self.condType = try container.decodeIfPresent(String.self, forKey: .condType) ?? "scale_bias"
        self.condDim = try container.decodeIfPresent(Int.self, forKey: .condDim) ?? 128
        self.condOutLayer = try container.decodeIfPresent(Bool.self, forKey: .condOutLayer) ?? false
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(encoderDim, forKey: .encoderDim)
        try container.encode(encoderRates, forKey: .encoderRates)
        try container.encode(latentDim, forKey: .latentDim)
        try container.encode(decoderDim, forKey: .decoderDim)
        try container.encode(decoderRates, forKey: .decoderRates)
        try container.encode(depthwise, forKey: .depthwise)
        try container.encode(sampleRate, forKey: .sampleRate)
        try container.encode(outSampleRate, forKey: .outSampleRate)
        try container.encode(useNoiseBlock, forKey: .useNoiseBlock)
        try container.encode(srBinBoundaries, forKey: .srBinBoundaries)
        try container.encode(condType, forKey: .condType)
        try container.encode(condDim, forKey: .condDim)
        try container.encode(condOutLayer, forKey: .condOutLayer)
    }
}

public struct QuantizationConfig: Codable, Sendable {
    public var bits: Int = 16
    public var groupSize: Int = 64
    public var variant: String?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
        case variant
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.bits = try container.decodeIfPresent(Int.self, forKey: .bits) ?? 16
        self.groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        self.variant = try container.decodeIfPresent(String.self, forKey: .variant)
    }

    public var isQuantized: Bool { bits == 4 || bits == 8 }
}

public struct ModelArgs: Codable, Sendable {
    public var lmConfig: LMConfig = LMConfig()
    public var encoderConfig: EncoderConfig = EncoderConfig()
    public var ditConfig: DiTConfig = DiTConfig()
    public var audioVAEConfig: AudioVAEConfig = AudioVAEConfig()
    public var patchSize: Int = 4
    public var featDim: Int = 64
    public var scalarQuantizationLatentDim: Int = 512
    public var scalarQuantizationScale: Int = 9
    public var residualLMNumLayers: Int = 8
    public var residualLMNoRope: Bool = true
    public var maxLength: Int = 8192
    public var quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case lmConfig = "lm_config"
        case encoderConfig = "encoder_config"
        case ditConfig = "dit_config"
        case audioVAEConfig = "audio_vae_config"
        case patchSize = "patch_size"
        case featDim = "feat_dim"
        case scalarQuantizationLatentDim = "scalar_quantization_latent_dim"
        case scalarQuantizationScale = "scalar_quantization_scale"
        case residualLMNumLayers = "residual_lm_num_layers"
        case residualLMNoRope = "residual_lm_no_rope"
        case maxLength = "max_length"
        case quantization
    }

    public init() {}

    public static func load(from directory: URL) throws -> ModelArgs {
        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let decoder = JSONDecoder()
        return try decoder.decode(ModelArgs.self, from: data)
    }
}
