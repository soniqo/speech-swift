#if canImport(CoreML)
import Foundation

public struct ChatterboxFlashCoreMLBundleConfig: Decodable, Sendable {
    public struct BundleInfo: Decodable, Sendable {
        public let format: String
        public let sourceModel: String
        public let sourceRepo: String

        public init(format: String, sourceModel: String, sourceRepo: String) {
            self.format = format
            self.sourceModel = sourceModel
            self.sourceRepo = sourceRepo
        }

        enum CodingKeys: String, CodingKey {
            case format
            case sourceModel = "source_model"
            case sourceRepo = "source_repo"
        }
    }

    public struct Components: Decodable, Sendable {
        public let t3: ChatterboxFlashT3Config
        public let audio: ChatterboxFlashAudioConfig
    }

    public let bundle: BundleInfo
    public let components: Components

    enum CodingKeys: String, CodingKey {
        case bundle
        case components
        case baseModel = "base_model"
        case format
        case sourceRepo = "source_repo"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        if let bundle = try container.decodeIfPresent(BundleInfo.self, forKey: .bundle) {
            self.bundle = bundle
        } else {
            self.bundle = BundleInfo(
                format: try container.decodeIfPresent(String.self, forKey: .format) ?? "coreml",
                sourceModel: try container.decodeIfPresent(String.self, forKey: .baseModel) ?? "",
                sourceRepo: try container.decodeIfPresent(String.self, forKey: .sourceRepo) ?? ""
            )
        }
        self.components = try container.decode(Components.self, forKey: .components)
    }

    public static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

public struct ChatterboxFlashT3Config: Decodable, Sendable {
    public let textLen: Int
    public let maxSeq: Int
    public let speechLen: Int
    public let blockSize: Int
    public let condLen: Int
    public let promptSpeechLen: Int
    public let hiddenSize: Int
    public let numLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let kvCacheShape: [Int]
    public let condDim: Int
    public let textVocabSize: Int
    public let speechVocabSize: Int
    public let maskTokenId: Int
    public let startSpeechToken: Int
    public let stopSpeechToken: Int
    public let startTextToken: Int
    public let stopTextToken: Int
    public let startToken: Int
    public let stopToken: Int
    public let maskToken: Int
    public let paddingToken: Int
    public let emotionAdvDefault: Float

    public var prefixLen: Int { condLen + textLen + 1 }

    enum CodingKeys: String, CodingKey {
        case textLen = "text_len"
        case maxSeq = "max_seq"
        case speechLen = "speech_len"
        case blockSize = "block_size"
        case condLen = "cond_len"
        case promptSpeechLen = "prompt_speech_len"
        case hiddenSize = "hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case kvCacheShape = "kv_cache_shape"
        case condDim = "cond_dim"
        case textVocabSize = "text_vocab_size"
        case speechVocabSize = "speech_vocab_size"
        case maskTokenId = "mask_token_id"
        case startSpeechToken = "start_speech_token"
        case stopSpeechToken = "stop_speech_token"
        case startTextToken = "start_text_token"
        case stopTextToken = "stop_text_token"
        case startToken = "start_token"
        case stopToken = "stop_token"
        case maskToken = "mask_token"
        case paddingToken = "padding_token"
        case emotionAdvDefault = "emotion_adv_default"
    }

    public init(
        textLen: Int,
        maxSeq: Int,
        speechLen: Int? = nil,
        blockSize: Int,
        condLen: Int,
        promptSpeechLen: Int,
        hiddenSize: Int,
        numLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        kvCacheShape: [Int],
        condDim: Int? = nil,
        textVocabSize: Int,
        speechVocabSize: Int,
        maskTokenId: Int,
        startSpeechToken: Int,
        stopSpeechToken: Int,
        startTextToken: Int,
        stopTextToken: Int,
        startToken: Int? = nil,
        stopToken: Int? = nil,
        maskToken: Int? = nil,
        paddingToken: Int? = nil,
        emotionAdvDefault: Float = 0.5
    ) {
        self.textLen = textLen
        self.maxSeq = maxSeq
        self.speechLen = speechLen ?? maxSeq
        self.blockSize = blockSize
        self.condLen = condLen
        self.promptSpeechLen = promptSpeechLen
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.kvCacheShape = kvCacheShape
        self.condDim = condDim ?? hiddenSize
        self.textVocabSize = textVocabSize
        self.speechVocabSize = speechVocabSize
        self.maskTokenId = maskTokenId
        self.startSpeechToken = startSpeechToken
        self.stopSpeechToken = stopSpeechToken
        self.startTextToken = startTextToken
        self.stopTextToken = stopTextToken
        self.startToken = startToken ?? startSpeechToken
        self.stopToken = stopToken ?? stopSpeechToken
        self.maskToken = maskToken ?? maskTokenId
        self.paddingToken = paddingToken ?? stopTextToken
        self.emotionAdvDefault = emotionAdvDefault
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = Self.fallback
        let textLen = try container.decodeIfPresent(Int.self, forKey: .textLen) ?? fallback.textLen
        let maxSeq = try container.decodeIfPresent(Int.self, forKey: .maxSeq)
            ?? container.decodeIfPresent(Int.self, forKey: .speechLen)
            ?? fallback.maxSeq
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? container.decodeIfPresent(Int.self, forKey: .condDim)
            ?? fallback.hiddenSize
        let startSpeech = try container.decodeIfPresent(Int.self, forKey: .startSpeechToken)
            ?? container.decodeIfPresent(Int.self, forKey: .startToken)
            ?? fallback.startSpeechToken
        let stopSpeech = try container.decodeIfPresent(Int.self, forKey: .stopSpeechToken)
            ?? container.decodeIfPresent(Int.self, forKey: .stopToken)
            ?? fallback.stopSpeechToken
        let mask = try container.decodeIfPresent(Int.self, forKey: .maskTokenId)
            ?? container.decodeIfPresent(Int.self, forKey: .maskToken)
            ?? fallback.maskTokenId

        self.init(
            textLen: textLen,
            maxSeq: maxSeq,
            speechLen: try container.decodeIfPresent(Int.self, forKey: .speechLen) ?? maxSeq,
            blockSize: try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? fallback.blockSize,
            condLen: try container.decodeIfPresent(Int.self, forKey: .condLen) ?? fallback.condLen,
            promptSpeechLen: try container.decodeIfPresent(Int.self, forKey: .promptSpeechLen)
                ?? fallback.promptSpeechLen,
            hiddenSize: hiddenSize,
            numLayers: try container.decodeIfPresent(Int.self, forKey: .numLayers) ?? fallback.numLayers,
            numAttentionHeads: try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
                ?? fallback.numAttentionHeads,
            numKeyValueHeads: try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
                ?? fallback.numKeyValueHeads,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim) ?? fallback.headDim,
            kvCacheShape: try container.decodeIfPresent([Int].self, forKey: .kvCacheShape)
                ?? fallback.kvCacheShape,
            condDim: try container.decodeIfPresent(Int.self, forKey: .condDim) ?? hiddenSize,
            textVocabSize: try container.decodeIfPresent(Int.self, forKey: .textVocabSize)
                ?? fallback.textVocabSize,
            speechVocabSize: try container.decodeIfPresent(Int.self, forKey: .speechVocabSize)
                ?? fallback.speechVocabSize,
            maskTokenId: mask,
            startSpeechToken: startSpeech,
            stopSpeechToken: stopSpeech,
            startTextToken: try container.decodeIfPresent(Int.self, forKey: .startTextToken)
                ?? fallback.startTextToken,
            stopTextToken: try container.decodeIfPresent(Int.self, forKey: .stopTextToken)
                ?? fallback.stopTextToken,
            startToken: startSpeech,
            stopToken: stopSpeech,
            maskToken: mask,
            paddingToken: try container.decodeIfPresent(Int.self, forKey: .paddingToken)
                ?? fallback.paddingToken,
            emotionAdvDefault: try container.decodeIfPresent(Float.self, forKey: .emotionAdvDefault)
                ?? fallback.emotionAdvDefault
        )
    }

    public static let fallback = Self(
        textLen: 256,
        maxSeq: 1024,
        blockSize: 16,
        condLen: 34,
        promptSpeechLen: 150,
        hiddenSize: 1024,
        numLayers: 30,
        numAttentionHeads: 16,
        numKeyValueHeads: 16,
        headDim: 64,
        kvCacheShape: [1, 30720, 1, 1024],
        textVocabSize: 704,
        speechVocabSize: 8194,
        maskTokenId: 8194,
        startSpeechToken: 6561,
        stopSpeechToken: 6562,
        startTextToken: 255,
        stopTextToken: 0,
        emotionAdvDefault: 0.5
    )

    public static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("t3/config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

public struct ChatterboxFlashAudioConfig: Decodable, Sendable {
    public let sampleRate: Int
    public let tokenLen: Int
    public let melLen: Int
    public let tokenMelRatio: Int
    public let samplesPerMelFrame: Int
    public let refEmbeddingDim: Int
    public let speakerProjectionDim: Int

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case tokenLen = "token_len"
        case melLen = "mel_len"
        case tokenMelRatio = "token_mel_ratio"
        case samplesPerMelFrame = "samples_per_mel_frame"
        case refEmbeddingDim = "ref_embedding_dim"
        case speakerProjectionDim = "speaker_projection_dim"
    }

    public init(
        sampleRate: Int,
        tokenLen: Int,
        melLen: Int,
        tokenMelRatio: Int,
        samplesPerMelFrame: Int,
        refEmbeddingDim: Int,
        speakerProjectionDim: Int
    ) {
        self.sampleRate = sampleRate
        self.tokenLen = tokenLen
        self.melLen = melLen
        self.tokenMelRatio = tokenMelRatio
        self.samplesPerMelFrame = samplesPerMelFrame
        self.refEmbeddingDim = refEmbeddingDim
        self.speakerProjectionDim = speakerProjectionDim
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let fallback = Self.fallback
        self.init(
            sampleRate: try container.decodeIfPresent(Int.self, forKey: .sampleRate) ?? fallback.sampleRate,
            tokenLen: try container.decodeIfPresent(Int.self, forKey: .tokenLen) ?? fallback.tokenLen,
            melLen: try container.decodeIfPresent(Int.self, forKey: .melLen) ?? fallback.melLen,
            tokenMelRatio: try container.decodeIfPresent(Int.self, forKey: .tokenMelRatio)
                ?? fallback.tokenMelRatio,
            samplesPerMelFrame: try container.decodeIfPresent(Int.self, forKey: .samplesPerMelFrame)
                ?? fallback.samplesPerMelFrame,
            refEmbeddingDim: try container.decodeIfPresent(Int.self, forKey: .refEmbeddingDim)
                ?? fallback.refEmbeddingDim,
            speakerProjectionDim: try container.decodeIfPresent(Int.self, forKey: .speakerProjectionDim)
                ?? fallback.speakerProjectionDim
        )
    }

    public static let fallback = Self(
        sampleRate: 24_000,
        tokenLen: 192,
        melLen: 384,
        tokenMelRatio: 2,
        samplesPerMelFrame: 480,
        refEmbeddingDim: 192,
        speakerProjectionDim: 80
    )

    public static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("audio/audio_config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(Self.self, from: data)
    }
}

public enum ChatterboxFlashCoreMLError: Error, LocalizedError {
    case missingFile(String)
    case invalidShape(String)
    case missingOutput(String)
    case unsupportedConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let path):
            return "Missing Chatterbox Flash Core ML file: \(path)"
        case .invalidShape(let message):
            return "Invalid Chatterbox Flash Core ML tensor shape: \(message)"
        case .missingOutput(let name):
            return "Core ML graph did not return output '\(name)'"
        case .unsupportedConfiguration(let message):
            return "Unsupported Chatterbox Flash Core ML configuration: \(message)"
        }
    }
}

public struct ChatterboxFlashGenerationOptions: Sendable {
    public var maxSpeechTokens: Int?
    public var numSteps: Int
    public var temperature: Float
    public var timeShiftTau: Float
    public var omnivoiceScheduleTShift: Float
    public var cfgScale: Float
    public var positionTemperature: Float
    public var seed: UInt64

    public init(
        maxSpeechTokens: Int? = nil,
        numSteps: Int = 10,
        temperature: Float = 0.6,
        timeShiftTau: Float = 0.1,
        omnivoiceScheduleTShift: Float = 0.5,
        cfgScale: Float = 0.0,
        positionTemperature: Float = 5.0,
        seed: UInt64 = 0
    ) {
        self.maxSpeechTokens = maxSpeechTokens
        self.numSteps = numSteps
        self.temperature = temperature
        self.timeShiftTau = timeShiftTau
        self.omnivoiceScheduleTShift = omnivoiceScheduleTShift
        self.cfgScale = cfgScale
        self.positionTemperature = positionTemperature
        self.seed = seed
    }
}

public struct ChatterboxFlashConditioning: Sendable {
    public let t3: ChatterboxFlashT3Conditioning
    public let audio: ChatterboxFlashS3GenReference

    public init(t3: ChatterboxFlashT3Conditioning, audio: ChatterboxFlashS3GenReference) {
        self.t3 = t3
        self.audio = audio
    }
}
#endif
