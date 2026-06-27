import Foundation

public struct IndicMioModelConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let eosTokenId: Int
    public let padTokenId: Int

    public static func load(from url: URL) throws -> IndicMioModelConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw IndicMioError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw IndicMioError.invalidConfig("config.json is not a JSON object")
        }

        func int(_ key: String, _ defaultValue: Int) -> Int {
            (obj[key] as? NSNumber)?.intValue ?? defaultValue
        }
        func double(_ key: String, _ defaultValue: Double) -> Double {
            (obj[key] as? NSNumber)?.doubleValue ?? defaultValue
        }
        func tokenId(_ key: String, _ defaultValue: Int) -> Int {
            if let n = obj[key] as? NSNumber { return n.intValue }
            if let a = obj[key] as? [NSNumber], let first = a.first { return first.intValue }
            return defaultValue
        }

        let hidden = int("hidden_size", 1024)
        let heads = int("num_attention_heads", 16)
        let headDim = obj["head_dim"] != nil ? int("head_dim", 128) : hidden / max(1, heads)

        return IndicMioModelConfig(
            hiddenSize: hidden,
            numHiddenLayers: int("num_hidden_layers", 28),
            numAttentionHeads: heads,
            numKeyValueHeads: int("num_key_value_heads", 8),
            headDim: headDim,
            intermediateSize: int("intermediate_size", 3072),
            vocabSize: int("vocab_size", 164_480),
            ropeTheta: Float(double("rope_theta", 1_000_000)),
            rmsNormEps: Float(double("rms_norm_eps", 1e-6)),
            tieWordEmbeddings: (obj["tie_word_embeddings"] as? Bool) ?? true,
            eosTokenId: tokenId("eos_token_id", IndicMioPrompt.imEndTokenId),
            padTokenId: tokenId("pad_token_id", IndicMioPrompt.endOfTextTokenId)
        )
    }
}

public struct IndicMioBundleConfig: Sendable, Equatable {
    public let modelId: String
    public let codecPath: String
    public let sampleRate: Int
    public let emotionMarkers: [String]

    public static let defaultModelId = "aufklarer/Indic-Mio-MLX-fp16"

    public static func load(from directory: URL) throws -> IndicMioBundleConfig {
        let url = directory.appendingPathComponent("bundle_config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            return IndicMioBundleConfig(
                modelId: defaultModelId,
                codecPath: "miocodec",
                sampleRate: 24_000,
                emotionMarkers: IndicMioPrompt.indianLanguageEmotionMarkers
            )
        }

        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw IndicMioError.invalidConfig("bundle_config.json is not a JSON object")
        }
        let primary = root["primary_model"] as? [String: Any] ?? [:]
        let codec = root["codec"] as? [String: Any] ?? [:]
        let markers = primary["emotion_markers"] as? [String] ?? IndicMioPrompt.indianLanguageEmotionMarkers
        let sampleRate = (codec["sample_rate_hz"] as? NSNumber)?.intValue ?? 24_000
        let codecPath = (codec["path"] as? String)?.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            ?? "miocodec"

        return IndicMioBundleConfig(
            modelId: defaultModelId,
            codecPath: codecPath.isEmpty ? "miocodec" : codecPath,
            sampleRate: sampleRate,
            emotionMarkers: markers
        )
    }
}

public struct MioCodecConfig: Sendable, Equatable {
    public let sampleRate: Int
    public let nFFT: Int
    public let hopLength: Int
    public let downsampleFactor: Int
    public let contentCodebookSize: Int
    public let contentEmbeddingDim: Int
    public let globalEmbeddingDim: Int
    public let waveDecoderDim: Int

    public var samplesPerToken: Int { hopLength * downsampleFactor }

    public static let `default` = MioCodecConfig(
        sampleRate: 24_000,
        nFFT: 1_920,
        hopLength: 480,
        downsampleFactor: 2,
        contentCodebookSize: 12_800,
        contentEmbeddingDim: 768,
        globalEmbeddingDim: 128,
        waveDecoderDim: 512
    )
}
