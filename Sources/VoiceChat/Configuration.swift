import Foundation

/// Configuration for the NemotronLabs VoiceChat 11B perception bundle.
///
/// Decoded directly from the `encoder/config.json` emitted by the MLX export,
/// so the Swift runtime and the Python converter cannot drift apart.
///
/// Three fields below are load-bearing, and a stock Conformer gets all three
/// wrong in ways that produce a model which loads cleanly and computes
/// nonsense rather than raising:
///
/// - `useBias` is false. NeMo has no bias on the feed-forward, self-attention
///   or convolution linears. Building them with biases adds 264 parameters
///   that do not exist in the checkpoint.
/// - `preEncodeFreqOut` is 17, not 16. Causal subsampling pads (K-1, S-1) per
///   spatial axis, so 128 mel bins reduce to 17 and `pre_encode.out` is
///   [d_model, 256 * 17] = [1024, 4352].
/// - `convNormType` is `layer_norm`. The module named `batch_norm` in the
///   checkpoint is really a LayerNorm and carries no running statistics.
///
/// A fourth, `attContextSize`, is the one that fails most quietly: the encoder
/// was trained with 70 frames of left context and none to the right. Omitting
/// the mask gives full bidirectional attention, which a duplex model never
/// has, and changes every output without any error.
public struct VoiceChatEncoderConfig: Codable, Sendable {
    public let dModel: Int
    public let nLayers: Int
    public let nHeads: Int
    public let featIn: Int
    public let ffExpansionFactor: Int
    public let convKernelSize: Int
    public let subsamplingFactor: Int
    public let subsamplingConvChannels: Int
    public let preEncodeFreqOut: Int
    public let causalConvIndices: [Int]
    public let convNormType: String
    public let selfAttentionModel: String
    public let attContextSize: [Int]
    public let attContextStyle: String
    public let posEmbMaxLen: Int
    public let useBias: Bool
    public let xscaling: Bool

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case nLayers = "n_layers"
        case nHeads = "n_heads"
        case featIn = "feat_in"
        case ffExpansionFactor = "ff_expansion_factor"
        case convKernelSize = "conv_kernel_size"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case preEncodeFreqOut = "pre_encode_freq_out"
        case causalConvIndices = "causal_conv_indices"
        case convNormType = "conv_norm_type"
        case selfAttentionModel = "self_attention_model"
        case attContextSize = "att_context_size"
        case attContextStyle = "att_context_style"
        case posEmbMaxLen = "pos_emb_max_len"
        case useBias = "use_bias"
        case xscaling = "xscaling"
    }

    public var dFF: Int { dModel * ffExpansionFactor }
    public var leftContext: Int { attContextSize.first ?? 70 }
    public var rightContext: Int { attContextSize.count > 1 ? attContextSize[1] : 0 }
}

/// The single Linear that bridges encoder output into the language model's
/// hidden size. The NeMo config calls the modality adapter an
/// `IdentityConnector`, so this projection is the entire bridge.
public struct ModalityProjectionConfig: Codable, Sendable {
    public let inFeatures: Int
    public let outFeatures: Int

    enum CodingKeys: String, CodingKey {
        case inFeatures = "in_features"
        case outFeatures = "out_features"
    }
}

public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

public struct VoiceChatPerceptionConfig: Codable, Sendable {
    public let modelType: String
    public let encoder: VoiceChatEncoderConfig
    public let modalityProj: ModalityProjectionConfig
    public let sampleRate: Int
    public let frameLength: Double
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoder
        case modalityProj = "modality_proj"
        case sampleRate = "sample_rate"
        case frameLength = "frame_length"
        case quantization
    }

    public static func load(from directory: URL) throws -> VoiceChatPerceptionConfig {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        return try JSONDecoder().decode(VoiceChatPerceptionConfig.self, from: data)
    }
}
