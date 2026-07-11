import Foundation

/// Qwen3 backbone hyperparameters, parsed from the upstream `text_config`.
public struct HiggsTTSTextConfig: Equatable, Sendable {
    public var hiddenSize: Int = 2560
    public var numHiddenLayers: Int = 36
    public var intermediateSize: Int = 9728
    public var numAttentionHeads: Int = 32
    public var numKeyValueHeads: Int = 8
    public var maxPositionEmbeddings: Int = 32768
    public var ropeTheta: Float = 1_000_000
    public var headDim: Int = 128
    public var rmsNormEps: Float = 1e-6
    public var vocabSize: Int = 151_936
    public var tieWordEmbeddings: Bool = true

    init(json: [String: Any]) {
        if let v = json["hidden_size"] as? Int { hiddenSize = v }
        if let v = json["num_hidden_layers"] as? Int { numHiddenLayers = v }
        if let v = json["intermediate_size"] as? Int { intermediateSize = v }
        if let v = json["num_attention_heads"] as? Int { numAttentionHeads = v }
        if let v = json["num_key_value_heads"] as? Int { numKeyValueHeads = v }
        if let v = json["max_position_embeddings"] as? Int { maxPositionEmbeddings = v }
        if let v = json["rope_theta"] as? Double { ropeTheta = Float(v) }
        if let v = json["head_dim"] as? Int { headDim = v }
        if let v = json["rms_norm_eps"] as? Double { rmsNormEps = Float(v) }
        if let v = json["vocab_size"] as? Int { vocabSize = v }
        if let v = json["tie_word_embeddings"] as? Bool { tieWordEmbeddings = v }
    }

    public init() {}
}

/// Higgs TTS 3 model configuration, parsed from the upstream `config.json`.
///
/// Field sourcing mirrors the reference implementations: codebook count and
/// vocab fall back to `audio_encoder_config`, and BOC/EOC default to the last
/// two ids of the codebook vocab (1024/1025 for vocab 1026).
public struct HiggsTTSConfig: Equatable, Sendable {
    public var modelType: String = "higgs_multimodal_qwen3"
    public var textConfig = HiggsTTSTextConfig()
    public var audioNumCodebooks: Int = 8
    public var audioCodebookSize: Int = 1026
    public var audioBOCTokenId: Int32 = 1024
    public var audioEOCTokenId: Int32 = 1025
    public var useDelayPattern: Bool = true
    public var sampleRate: Int = 24_000

    public static func load(from url: URL) throws -> HiggsTTSConfig {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw HiggsTTSError.invalidConfig(url, error.localizedDescription)
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw HiggsTTSError.invalidConfig(url, "not a JSON object")
        }
        return try HiggsTTSConfig(json: json, url: url)
    }

    init(json: [String: Any], url: URL) throws {
        modelType = json["model_type"] as? String ?? modelType
        guard modelType == "higgs_multimodal_qwen3" || modelType == "higgs_audio_v3" else {
            throw HiggsTTSError.unexpectedModelType(modelType)
        }
        if let text = json["text_config"] as? [String: Any] {
            textConfig = HiggsTTSTextConfig(json: text)
        }
        let encoder = json["audio_encoder_config"] as? [String: Any] ?? [:]
        audioNumCodebooks = json["audio_num_codebooks"] as? Int
            ?? encoder["num_codebooks"] as? Int ?? audioNumCodebooks
        audioCodebookSize = json["audio_codebook_size"] as? Int
            ?? encoder["vocab_size"] as? Int ?? audioCodebookSize
        audioBOCTokenId = Int32(json["audio_boc_token_id"] as? Int ?? audioCodebookSize - 2)
        audioEOCTokenId = Int32(json["audio_eoc_token_id"] as? Int ?? audioCodebookSize - 1)
        useDelayPattern = json["use_delay_pattern"] as? Bool
            ?? encoder["use_delay_pattern"] as? Bool ?? useDelayPattern
        sampleRate = json["sample_rate"] as? Int ?? sampleRate
    }

    public init() {}
}

public enum HiggsTTSError: Error, LocalizedError, Equatable {
    case invalidConfig(URL, String)
    case unexpectedModelType(String)
    case missingRequiredFile(String)
    case invalidCodes(String)
    case unloaded

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let url, let reason):
            return "Invalid Higgs TTS config at \(url.path): \(reason)"
        case .unexpectedModelType(let type):
            return "Unexpected Higgs TTS model_type '\(type)'"
        case .missingRequiredFile(let file):
            return "Higgs TTS bundle is missing required file: \(file)"
        case .invalidCodes(let reason):
            return "Higgs TTS invalid audio codes: \(reason)"
        case .unloaded:
            return "Higgs TTS model has been unloaded."
        }
    }
}
