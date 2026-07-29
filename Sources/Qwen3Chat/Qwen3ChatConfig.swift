import Foundation

/// Model architecture type.
public enum ChatModelArch: String, Codable, Sendable {
    /// Qwen3.5 hybrid (DeltaNet linear attention + GatedAttention)
    case qwen35 = "qwen3_5_text"
}

/// Configuration for Qwen3.5 chat model.
public struct Qwen3ChatConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxSeqLen: Int
    public let ropeTheta: Double
    public let rmsNormEps: Double
    public let eosTokenId: Int
    public let padTokenId: Int
    /// Quantization label as written by the exporter, e.g. `"int4"` / `"int8"`.
    /// Kept for display and round-tripping; `quantBits` is what the model layers read.
    public let quantization: String
    /// Bit width per weight, when the checkpoint states it explicitly (`quantization_bits`).
    public let quantizationBits: Int?
    /// Quantization group size, when the checkpoint states it explicitly
    /// (`quantization_group_size`).
    public let quantizationGroupSize: Int?

    // Qwen3.5-specific fields
    public let modelType: ChatModelArch?
    /// Per-layer type: "linear_attention" (DeltaNet) or "full_attention" (GatedAttention)
    public let layerTypes: [String]?
    /// How often a full_attention layer appears (e.g., 4 = every 4th layer)
    public let fullAttentionInterval: Int?
    /// DeltaNet linear attention head config
    public let linearNumKeyHeads: Int?
    public let linearKeyHeadDim: Int?
    public let linearNumValueHeads: Int?
    public let linearValueHeadDim: Int?
    /// Causal conv1d kernel size for DeltaNet
    public let linearConvKernelDim: Int?
    /// Partial RoPE factor for GatedAttention (e.g., 0.25)
    public let partialRotaryFactor: Double?
    /// Whether embeddings are tied (lm_head = embed_tokens)
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxSeqLen = "max_seq_len"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case quantization
        case quantizationBits = "quantization_bits"
        case quantizationGroupSize = "quantization_group_size"
        case modelType = "model_type"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        intermediateSize: Int,
        vocabSize: Int,
        maxSeqLen: Int,
        ropeTheta: Double,
        rmsNormEps: Double,
        eosTokenId: Int,
        padTokenId: Int,
        quantization: String,
        quantizationBits: Int? = nil,
        quantizationGroupSize: Int? = nil,
        modelType: ChatModelArch? = nil,
        layerTypes: [String]? = nil,
        fullAttentionInterval: Int? = nil,
        linearNumKeyHeads: Int? = nil,
        linearKeyHeadDim: Int? = nil,
        linearNumValueHeads: Int? = nil,
        linearValueHeadDim: Int? = nil,
        linearConvKernelDim: Int? = nil,
        partialRotaryFactor: Double? = nil,
        tieWordEmbeddings: Bool? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.intermediateSize = intermediateSize
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.eosTokenId = eosTokenId
        self.padTokenId = padTokenId
        self.quantization = quantization
        self.quantizationBits = quantizationBits
        self.quantizationGroupSize = quantizationGroupSize
        self.modelType = modelType
        self.layerTypes = layerTypes
        self.fullAttentionInterval = fullAttentionInterval
        self.linearNumKeyHeads = linearNumKeyHeads
        self.linearKeyHeadDim = linearKeyHeadDim
        self.linearNumValueHeads = linearNumValueHeads
        self.linearValueHeadDim = linearValueHeadDim
        self.linearConvKernelDim = linearConvKernelDim
        self.partialRotaryFactor = partialRotaryFactor
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    /// Nested `"quantization": {"group_size": 64, "bits": 4}` object used by
    /// mlx-community / `mlx_lm` exports (the same shape `Qwen3DenseConfig` reads).
    private struct NestedQuantization: Decodable {
        let bits: Int?
        let groupSize: Int?

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
        }
    }

    /// Decoding is deliberately tolerant about how a checkpoint states its quantization,
    /// because three generations of exports are in circulation:
    ///   - `quantization_bits` / `quantization_group_size` — current, explicit and unambiguous
    ///   - `"quantization": "int8"` — older label-only form; carries no group size
    ///   - `"quantization": {"bits": 8, "group_size": 32}` — mlx-community object form
    /// and the oldest checkpoints state nothing at all. Everything else stays required:
    /// a config missing `hidden_size` is a broken config, not an old one.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        maxSeqLen = try container.decode(Int.self, forKey: .maxSeqLen)
        ropeTheta = try container.decode(Double.self, forKey: .ropeTheta)
        rmsNormEps = try container.decode(Double.self, forKey: .rmsNormEps)
        eosTokenId = try container.decode(Int.self, forKey: .eosTokenId)
        padTokenId = try container.decode(Int.self, forKey: .padTokenId)

        var bits = try container.decodeIfPresent(Int.self, forKey: .quantizationBits)
        var groupSize = try container.decodeIfPresent(Int.self, forKey: .quantizationGroupSize)
        var label = try? container.decode(String.self, forKey: .quantization)
        if label == nil,
           let nested = try? container.decode(NestedQuantization.self, forKey: .quantization) {
            bits = bits ?? nested.bits
            groupSize = groupSize ?? nested.groupSize
        }
        if label == nil, let bits { label = "int\(bits)" }
        quantization = label ?? ChatQuantization.defaultLabel
        quantizationBits = bits
        quantizationGroupSize = groupSize

        modelType = try container.decodeIfPresent(ChatModelArch.self, forKey: .modelType)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        fullAttentionInterval = try container.decodeIfPresent(
            Int.self, forKey: .fullAttentionInterval)
        linearNumKeyHeads = try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads)
        linearKeyHeadDim = try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim)
        linearNumValueHeads = try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads)
        linearValueHeadDim = try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim)
        partialRotaryFactor = try container.decodeIfPresent(
            Double.self, forKey: .partialRotaryFactor)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
    }

    // MARK: - Quantization

    /// Bit width and group size the quantized layers must be built with. The nested
    /// object form is folded into the explicit fields at decode time, so only those and
    /// the label are left to resolve here. See ``ChatQuantization`` for the precedence.
    public var resolvedQuantization: ChatQuantization {
        ChatQuantization.resolve(
            explicitBits: quantizationBits,
            explicitGroupSize: quantizationGroupSize,
            label: quantization)
    }

    public var quantBits: Int { resolvedQuantization.bits }
    public var quantGroupSize: Int { resolvedQuantization.groupSize }

    /// Whether this is a Qwen3.5 hybrid model.
    public var isQwen35: Bool {
        modelType == .qwen35 || layerTypes != nil
    }

    /// Number of full-attention layers (that need KV cache).
    public var numFullAttentionLayers: Int {
        guard let types = layerTypes else { return numHiddenLayers }
        return types.filter { $0 == "full_attention" }.count
    }

    /// Default config for Qwen3.5-0.8B.
    public static let qwen35_08B = Qwen3ChatConfig(
        hiddenSize: 1024,
        numHiddenLayers: 24,
        numAttentionHeads: 8,
        numKeyValueHeads: 2,
        headDim: 256,
        intermediateSize: 3584,
        vocabSize: 248320,
        maxSeqLen: 2048,
        ropeTheta: 10_000_000.0,
        rmsNormEps: 1e-6,
        eosTokenId: 248046,  // <|im_end|> — stops generation at end of assistant turn
        padTokenId: 248044,  // <|endoftext|>
        quantization: "int4",
        modelType: .qwen35,
        layerTypes: [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        fullAttentionInterval: 4,
        linearNumKeyHeads: 16,
        linearKeyHeadDim: 128,
        linearNumValueHeads: 16,
        linearValueHeadDim: 128,
        linearConvKernelDim: 4,
        partialRotaryFactor: 0.25,
        tieWordEmbeddings: true
    )

    /// Load config from a JSON file.
    public static func load(from url: URL) throws -> Qwen3ChatConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Qwen3ChatConfig.self, from: data)
    }
}

/// Sampling parameters for text generation.
public struct ChatSamplingConfig: Sendable {
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var maxTokens: Int
    public var repetitionPenalty: Float

    public init(
        temperature: Float = 0.7,
        topK: Int = 50,
        topP: Float = 0.9,
        maxTokens: Int = 256,
        repetitionPenalty: Float = 1.1
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.maxTokens = maxTokens
        self.repetitionPenalty = repetitionPenalty
    }

    public static let `default` = ChatSamplingConfig()
    public static let creative = ChatSamplingConfig(temperature: 0.9, topP: 0.95)
    public static let precise = ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8)
}
