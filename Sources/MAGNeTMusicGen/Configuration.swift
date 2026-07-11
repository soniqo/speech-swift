import Foundation

/// MAGNeT model variant. Pairs (300M small / 1.5B medium) × (int4 / int8) on
/// the soniqo-side `aufklarer/MAGNeT-*-30secs-MLX-*bit` bundles.
public enum MAGNeTVariant: String, Sendable, CaseIterable {
    case smallInt4 = "small-int4"
    case smallInt8 = "small-int8"
    case mediumInt4 = "medium-int4"
    case mediumInt8 = "medium-int8"

    public var huggingFaceRepoId: String {
        switch self {
        case .smallInt4:  return "aufklarer/MAGNeT-Small-30secs-MLX-4bit"
        case .smallInt8:  return "aufklarer/MAGNeT-Small-30secs-MLX-8bit"
        case .mediumInt4: return "aufklarer/MAGNeT-Medium-30secs-MLX-4bit"
        case .mediumInt8: return "aufklarer/MAGNeT-Medium-30secs-MLX-8bit"
        }
    }

    public var bits: Int {
        switch self {
        case .smallInt4, .mediumInt4: return 4
        case .smallInt8, .mediumInt8: return 8
        }
    }
}

/// `config.json` quantization block.
public struct MAGNeTQuantizationConfig: Codable, Sendable {
    public let mode: String
    public let bits: Int
    public let groupSize: Int
    public let targets: [String]

    enum CodingKeys: String, CodingKey {
        case mode, bits
        case groupSize = "group_size"
        case targets
    }
}

/// Full config decoded from the bundle's `config.json`.
public struct MAGNeTConfig: Codable, Sendable {
    /// Number of RVQ codebooks (4).
    public let nQ: Int
    /// Vocabulary size per codebook (2048). MAGNeT reserves `card` as the mask
    /// token id, so embeddings have `card + 1` rows.
    public let card: Int
    /// Transformer hidden size (1024 small / 1536 medium).
    public let dim: Int
    public let numHeads: Int
    public let numLayers: Int
    public let ffnDim: Int
    /// Output clip length in seconds (30).
    public let segmentDuration: Int
    /// EnCodec frame rate Hz (50).
    public let frameRate: Int
    /// Local attention window for stages > 0 (5).
    public let subcodesContext: Int
    /// Span length for non-overlapping chunk masking (3).
    public let spanLen: Int
    /// EnCodec sample rate (32000).
    public let sampleRate: Int
    /// Upstream T5 hf repo, e.g. "t5-base".
    public let t5Name: String
    /// T5 hidden size (768 for t5-base).
    public let t5Dim: Int
    /// EnCodec variant tag (used to build "mlx-community/encodec-32khz-float32").
    public let encodecName: String
    public let quantization: MAGNeTQuantizationConfig?
    public let format: String

    enum CodingKeys: String, CodingKey {
        case nQ = "n_q"
        case card, dim
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case ffnDim = "ffn_dim"
        case segmentDuration = "segment_duration"
        case frameRate = "frame_rate"
        case subcodesContext = "subcodes_context"
        case spanLen = "span_len"
        case sampleRate = "sample_rate"
        case t5Name = "t5_name"
        case t5Dim = "t5_dim"
        case encodecName = "encodec_name"
        case quantization, format
    }

    /// Output token count per codebook (frame_rate × segment_duration).
    public var seqLen: Int { frameRate * segmentDuration }

    /// Token id used to mark "still masked, predict me this stage".
    public var maskTokenId: Int { card }

    public static func load(from path: URL) throws -> MAGNeTConfig {
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(MAGNeTConfig.self, from: data)
    }
}

/// EnCodec (HuggingFace `transformers`-style) config.
public struct EncodecModelConfig: Codable, Sendable {
    public let audioChannels: Int
    public let chunkLengthS: Double?
    public let codebookDim: Int
    public let codebookSize: Int
    public let compress: Int
    public let dilationGrowthRate: Int
    public let hiddenSize: Int
    public let kernelSize: Int
    public let lastKernelSize: Int
    public let normType: String
    public let normalize: Bool
    public let numFilters: Int
    public let numLstmLayers: Int
    public let numResidualLayers: Int
    public let overlap: Double?
    public let padMode: String
    public let residualKernelSize: Int
    public let samplingRate: Int
    public let targetBandwidths: [Double]
    public let trimRightRatio: Double
    public let upsamplingRatios: [Int]
    public let useCausalConv: Bool
    public let useConvShortcut: Bool

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case chunkLengthS = "chunk_length_s"
        case codebookDim = "codebook_dim"
        case codebookSize = "codebook_size"
        case compress
        case dilationGrowthRate = "dilation_growth_rate"
        case hiddenSize = "hidden_size"
        case kernelSize = "kernel_size"
        case lastKernelSize = "last_kernel_size"
        case normType = "norm_type"
        case normalize
        case numFilters = "num_filters"
        case numLstmLayers = "num_lstm_layers"
        case numResidualLayers = "num_residual_layers"
        case overlap
        case padMode = "pad_mode"
        case residualKernelSize = "residual_kernel_size"
        case samplingRate = "sampling_rate"
        case targetBandwidths = "target_bandwidths"
        case trimRightRatio = "trim_right_ratio"
        case upsamplingRatios = "upsampling_ratios"
        case useCausalConv = "use_causal_conv"
        case useConvShortcut = "use_conv_shortcut"
    }

    public static func load(from path: URL) throws -> EncodecModelConfig {
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(EncodecModelConfig.self, from: data)
    }
}

/// T5 config subset needed by the encoder.
public struct T5ModelConfig: Codable, Sendable {
    public let dModel: Int
    public let dKv: Int
    public let dFf: Int
    public let numHeads: Int
    public let numLayers: Int
    public let vocabSize: Int
    public let layerNormEpsilon: Float
    public let relativeAttentionNumBuckets: Int
    public let relativeAttentionMaxDistance: Int
    public let feedForwardProj: String?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case dKv = "d_kv"
        case dFf = "d_ff"
        case numHeads = "num_heads"
        case numLayers = "num_layers"
        case vocabSize = "vocab_size"
        case layerNormEpsilon = "layer_norm_epsilon"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case relativeAttentionMaxDistance = "relative_attention_max_distance"
        case feedForwardProj = "feed_forward_proj"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.dModel = try c.decode(Int.self, forKey: .dModel)
        self.dKv = try c.decode(Int.self, forKey: .dKv)
        self.dFf = try c.decode(Int.self, forKey: .dFf)
        self.numHeads = try c.decode(Int.self, forKey: .numHeads)
        self.numLayers = try c.decode(Int.self, forKey: .numLayers)
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        self.relativeAttentionNumBuckets = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionNumBuckets) ?? 32
        self.relativeAttentionMaxDistance = try c.decodeIfPresent(Int.self, forKey: .relativeAttentionMaxDistance) ?? 128
        self.feedForwardProj = try c.decodeIfPresent(String.self, forKey: .feedForwardProj)
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    public static func load(from path: URL) throws -> T5ModelConfig {
        let data = try Data(contentsOf: path)
        return try JSONDecoder().decode(T5ModelConfig.self, from: data)
    }
}
