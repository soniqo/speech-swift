import Foundation

public struct FishAudioTransformerConfig: Sendable, Equatable {
    public let vocabSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let maxSequenceLength: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let attentionQKNorm: Bool
    public let attentionQKVBias: Bool
    public let attentionOutputBias: Bool
}

public struct FishAudioDecoderConfig: Sendable, Equatable {
    public let vocabSize: Int
    public let numCodebooks: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenSize: Int
    public let textHiddenSize: Int
    public let intermediateSize: Int
    public let maxSequenceLength: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let attentionQKNorm: Bool
    public let attentionQKVBias: Bool
    public let attentionOutputBias: Bool
}

public struct FishAudioConfig: Sendable, Equatable {
    public static let defaultModelId = "aufklarer/Fish-Audio-S2-Pro-MLX-fp16"

    public let modelType: String
    public let dtype: String
    public let eosTokenId: Int
    public let padTokenId: Int
    public let audioPadTokenId: Int
    public let semanticStartTokenId: Int
    public let semanticEndTokenId: Int
    public let text: FishAudioTransformerConfig
    public let audioDecoder: FishAudioDecoderConfig

    public var semanticTokenCount: Int {
        max(0, semanticEndTokenId - semanticStartTokenId + 1)
    }

    public var scaleCodebookEmbeddings: Bool {
        modelType == "fish_qwen3_omni"
    }

    public var normFastLayerInput: Bool {
        modelType == "fish_qwen3_omni"
    }

    public static func load(from url: URL) throws -> FishAudioConfig {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FishAudioError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FishAudioError.invalidConfig("config.json is not a JSON object")
        }
        return try parse(root)
    }

    static func parse(_ root: [String: Any]) throws -> FishAudioConfig {
        let textRoot = try object(root, "text_config")
        let decoderRoot = try object(root, "audio_decoder_config")
        let text = FishAudioTransformerConfig(
            vocabSize: int(textRoot, "vocab_size", 155_776),
            numHiddenLayers: int(textRoot, "n_layer", 36),
            numAttentionHeads: int(textRoot, "n_head", 32),
            numKeyValueHeads: int(textRoot, "n_local_heads", 8),
            headDim: int(textRoot, "head_dim", 128),
            hiddenSize: int(textRoot, "dim", 2_560),
            intermediateSize: int(textRoot, "intermediate_size", 9_728),
            maxSequenceLength: int(textRoot, "max_seq_len", 32_768),
            ropeTheta: Float(double(textRoot, "rope_base", 1_000_000)),
            rmsNormEps: Float(double(textRoot, "norm_eps", 1e-6)),
            tieWordEmbeddings: bool(textRoot, "tie_word_embeddings", true),
            attentionQKNorm: bool(textRoot, "attention_qk_norm", true),
            attentionQKVBias: bool(textRoot, "attention_qkv_bias", false),
            attentionOutputBias: bool(textRoot, "attention_o_bias", false)
        )
        let decoder = FishAudioDecoderConfig(
            vocabSize: int(decoderRoot, "vocab_size", 4_096),
            numCodebooks: int(decoderRoot, "num_codebooks", 10),
            numHiddenLayers: int(decoderRoot, "n_layer", 4),
            numAttentionHeads: int(decoderRoot, "n_head", 32),
            numKeyValueHeads: int(decoderRoot, "n_local_heads", 8),
            headDim: int(decoderRoot, "head_dim", 128),
            hiddenSize: int(decoderRoot, "dim", 2_560),
            textHiddenSize: int(decoderRoot, "text_dim", text.hiddenSize),
            intermediateSize: int(decoderRoot, "intermediate_size", 9_728),
            maxSequenceLength: int(decoderRoot, "max_seq_len", 11),
            ropeTheta: Float(double(decoderRoot, "rope_base", 1_000_000)),
            rmsNormEps: Float(double(decoderRoot, "norm_eps", 1e-6)),
            attentionQKNorm: bool(decoderRoot, "attention_qk_norm", false),
            attentionQKVBias: bool(decoderRoot, "attention_qkv_bias", false),
            attentionOutputBias: bool(decoderRoot, "attention_o_bias", false)
        )
        return FishAudioConfig(
            modelType: string(root, "model_type", "fish_qwen3_omni"),
            dtype: string(root, "dtype", "bfloat16"),
            eosTokenId: int(root, "eos_token_id", 151_645),
            padTokenId: int(root, "pad_token_id", 151_669),
            audioPadTokenId: int(root, "audio_pad_token_id", 151_677),
            semanticStartTokenId: int(root, "semantic_start_token_id", 151_678),
            semanticEndTokenId: int(root, "semantic_end_token_id", 155_773),
            text: text,
            audioDecoder: decoder
        )
    }
}

private func object(_ root: [String: Any], _ key: String) throws -> [String: Any] {
    guard let value = root[key] as? [String: Any] else {
        throw FishAudioError.invalidConfig("\(key) is missing or not an object")
    }
    return value
}

private func int(_ root: [String: Any], _ key: String, _ defaultValue: Int) -> Int {
    (root[key] as? NSNumber)?.intValue ?? defaultValue
}

private func double(_ root: [String: Any], _ key: String, _ defaultValue: Double) -> Double {
    (root[key] as? NSNumber)?.doubleValue ?? defaultValue
}

private func bool(_ root: [String: Any], _ key: String, _ defaultValue: Bool) -> Bool {
    (root[key] as? Bool) ?? defaultValue
}

private func string(_ root: [String: Any], _ key: String, _ defaultValue: String) -> String {
    (root[key] as? String) ?? defaultValue
}
