import Foundation

extension Qwen3TTSModel {
    /// Resolve allocation-critical settings before constructing MLX modules.
    ///
    /// Published Qwen3-TTS bundles describe the talker in `config.json`, while the
    /// speech-tokenizer decoder currently uses the package's fixed architecture. A caller
    /// fallback may therefore supply decoder settings, but it must never override model size
    /// or quantization declared by the weights bundle.
    static func resolveLocalConfiguration(
        from configURL: URL,
        fallback: Qwen3TTSConfig? = nil
    ) throws -> Qwen3TTSConfig {
        let metadata: LocalBundleMetadata
        do {
            let data = try Data(contentsOf: configURL)
            metadata = try JSONDecoder().decode(LocalBundleMetadata.self, from: data)
        } catch {
            throw Qwen3TTSLoadingError.invalidConfiguration(
                configURL,
                reason: "could not decode config.json: \(error.localizedDescription)")
        }

        let declaredSize = try metadata.modelSize.map {
            try LocalModelSize(modelSize: $0, configURL: configURL)
        }
        let dimensionSize = try metadata.talker?.hiddenSize.map {
            try LocalModelSize(hiddenSize: $0, configURL: configURL)
        }
        if let declaredSize, let dimensionSize, declaredSize != dimensionSize {
            throw Qwen3TTSLoadingError.invalidConfiguration(
                configURL,
                reason: "model_size conflicts with talker_config.hidden_size")
        }

        let metadataSize = declaredSize ?? dimensionSize
        let fallbackSize = try metadataSize == nil
            ? fallback.map {
                try LocalModelSize(hiddenSize: $0.talker.hiddenSize, configURL: configURL)
            }
            : nil
        guard let modelSize = metadataSize ?? fallbackSize else {
            throw Qwen3TTSLoadingError.invalidConfiguration(
                configURL,
                reason: "model_size or talker_config.hidden_size is required")
        }

        let bits: Int
        if let quantization = metadata.quantization {
            guard quantization.bits == 4 || quantization.bits == 8 else {
                throw Qwen3TTSLoadingError.invalidConfiguration(
                    configURL,
                    reason: "quantization_config.bits must be 4 or 8")
            }
            bits = quantization.bits
        } else {
            // A missing quantization_config denotes an unquantized bf16/fp checkpoint.
            bits = 0
        }

        var resolved = Qwen3TTSConfig.config(for: modelSize.ttsModelSize, bits: bits)
        if let fallback {
            resolved.speechTokenizerDecoder = fallback.speechTokenizerDecoder
        }

        if let groupSize = metadata.quantization?.groupSize {
            guard groupSize > 0 else {
                throw Qwen3TTSLoadingError.invalidConfiguration(
                    configURL,
                    reason: "quantization_config.group_size must be positive")
            }
            resolved.talker.groupSize = groupSize
            resolved.codePredictor.groupSize = groupSize
        }

        if let talker = metadata.talker {
            try talker.apply(to: &resolved.talker, configURL: configURL)
            if let codePredictor = talker.codePredictor {
                try codePredictor.apply(to: &resolved.codePredictor, configURL: configURL)
            }
        }

        return resolved
    }
}

private enum LocalModelSize: Equatable {
    case small
    case large

    init(modelSize: String, configURL: URL) throws {
        switch modelSize.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "0.6b": self = .small
        case "1.7b": self = .large
        default:
            throw Qwen3TTSLoadingError.invalidConfiguration(
                configURL,
                reason: "unsupported model_size: \(modelSize)")
        }
    }

    init(hiddenSize: Int, configURL: URL) throws {
        switch hiddenSize {
        case 1024: self = .small
        case 2048: self = .large
        default:
            throw Qwen3TTSLoadingError.invalidConfiguration(
                configURL,
                reason: "unsupported talker_config.hidden_size: \(hiddenSize)")
        }
    }

    var ttsModelSize: TTSModelSize {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }
}

private struct LocalBundleMetadata: Decodable {
    let modelSize: String?
    let quantization: QuantizationMetadata?
    let talker: TalkerMetadata?

    enum CodingKeys: String, CodingKey {
        case modelSize = "model_size"
        case quantization = "quantization_config"
        case talker = "talker_config"
    }
}

private struct QuantizationMetadata: Decodable {
    let bits: Int
    let groupSize: Int?

    enum CodingKeys: String, CodingKey {
        case bits
        case groupSize = "group_size"
    }
}

private struct TalkerMetadata: Decodable {
    let hiddenSize: Int?
    let numLayers: Int?
    let numHeads: Int?
    let numKVHeads: Int?
    let headDim: Int?
    let intermediateSize: Int?
    let ropeTheta: Float?
    let rmsNormEps: Float?
    let textVocabSize: Int?
    let textHiddenSize: Int?
    let codecVocabSize: Int?
    let ropeScaling: RopeScalingMetadata?
    let codePredictor: CodePredictorMetadata?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_hidden_layers"
        case numHeads = "num_attention_heads"
        case numKVHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case textVocabSize = "text_vocab_size"
        case textHiddenSize = "text_hidden_size"
        case codecVocabSize = "vocab_size"
        case ropeScaling = "rope_scaling"
        case codePredictor = "code_predictor_config"
    }

    func apply(to config: inout TalkerConfig, configURL: URL) throws {
        try validatePositiveDimensions(
            [
                "talker_config.hidden_size": hiddenSize,
                "talker_config.num_hidden_layers": numLayers,
                "talker_config.num_attention_heads": numHeads,
                "talker_config.num_key_value_heads": numKVHeads,
                "talker_config.head_dim": headDim,
                "talker_config.intermediate_size": intermediateSize,
                "talker_config.text_vocab_size": textVocabSize,
                "talker_config.text_hidden_size": textHiddenSize,
                "talker_config.vocab_size": codecVocabSize,
            ],
            configURL: configURL)

        if let hiddenSize { config.hiddenSize = hiddenSize }
        if let numLayers { config.numLayers = numLayers }
        if let numHeads { config.numHeads = numHeads }
        if let numKVHeads { config.numKVHeads = numKVHeads }
        if let headDim { config.headDim = headDim }
        if let intermediateSize { config.intermediateSize = intermediateSize }
        if let ropeTheta { config.ropeTheta = ropeTheta }
        if let rmsNormEps { config.rmsNormEps = rmsNormEps }
        if let textVocabSize { config.textVocabSize = textVocabSize }
        if let textHiddenSize { config.textHiddenSize = textHiddenSize }
        if let codecVocabSize { config.codecVocabSize = codecVocabSize }
        if let sections = ropeScaling?.mropeSections {
            guard !sections.isEmpty, sections.allSatisfy({ $0 > 0 }) else {
                throw Qwen3TTSLoadingError.invalidConfiguration(
                    configURL,
                    reason: "talker_config.rope_scaling.mrope_section must contain positive values")
            }
            config.mropeSections = sections
        }
    }
}

private struct RopeScalingMetadata: Decodable {
    let mropeSections: [Int]?

    enum CodingKeys: String, CodingKey {
        case mropeSections = "mrope_section"
    }
}

private struct CodePredictorMetadata: Decodable {
    let hiddenSize: Int?
    let numLayers: Int?
    let numHeads: Int?
    let numKVHeads: Int?
    let headDim: Int?
    let intermediateSize: Int?
    let ropeTheta: Float?
    let rmsNormEps: Float?
    let vocabSize: Int?
    let numCodeGroups: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numLayers = "num_hidden_layers"
        case numHeads = "num_attention_heads"
        case numKVHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case numCodeGroups = "num_code_groups"
    }

    func apply(to config: inout CodePredictorConfig, configURL: URL) throws {
        try validatePositiveDimensions(
            [
                "talker_config.code_predictor_config.hidden_size": hiddenSize,
                "talker_config.code_predictor_config.num_hidden_layers": numLayers,
                "talker_config.code_predictor_config.num_attention_heads": numHeads,
                "talker_config.code_predictor_config.num_key_value_heads": numKVHeads,
                "talker_config.code_predictor_config.head_dim": headDim,
                "talker_config.code_predictor_config.intermediate_size": intermediateSize,
                "talker_config.code_predictor_config.vocab_size": vocabSize,
                "talker_config.code_predictor_config.num_code_groups": numCodeGroups,
            ],
            configURL: configURL)

        if let hiddenSize { config.hiddenSize = hiddenSize }
        if let numLayers { config.numLayers = numLayers }
        if let numHeads { config.numHeads = numHeads }
        if let numKVHeads { config.numKVHeads = numKVHeads }
        if let headDim { config.headDim = headDim }
        if let intermediateSize { config.intermediateSize = intermediateSize }
        if let ropeTheta { config.ropeTheta = ropeTheta }
        if let rmsNormEps { config.rmsNormEps = rmsNormEps }
        if let vocabSize { config.vocabSize = vocabSize }
        if let numCodeGroups { config.numCodeGroups = numCodeGroups }
    }
}

private func validatePositiveDimensions(
    _ dimensions: [String: Int?],
    configURL: URL
) throws {
    if let invalid = dimensions.first(where: {
        $0.value.map { $0 <= 0 } ?? false
    }) {
        throw Qwen3TTSLoadingError.invalidConfiguration(
            configURL,
            reason: "\(invalid.key) must be positive")
    }
}
