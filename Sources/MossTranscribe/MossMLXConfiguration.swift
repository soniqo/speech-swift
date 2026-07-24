import Foundation

struct MossMLXConfiguration: Decodable, Sendable, Equatable {
    struct Audio: Decodable, Sendable, Equatable {
        let hiddenSize: Int
        let intermediateSize: Int
        let layerNormEpsilon: Float
        let maximumSourcePositions: Int
        let mergeSize: Int
        let attentionHeads: Int
        let layers: Int
        let melBins: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layerNormEpsilon = "layer_norm_eps"
            case maximumSourcePositions = "max_source_positions"
            case mergeSize = "merge_size"
            case attentionHeads = "num_heads"
            case layers = "num_layers"
            case melBins = "num_mel_bins"
        }
    }

    struct Decoder: Decodable, Sendable, Equatable {
        let headDimension: Int
        let hiddenSize: Int
        let intermediateSize: Int
        let attentionHeads: Int
        let keyValueHeads: Int
        let layers: Int
        let rmsNormEpsilon: Float
        let ropeTheta: Float
        let vocabularySize: Int

        enum CodingKeys: String, CodingKey {
            case headDimension = "head_dim"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_heads"
            case keyValueHeads = "num_kv_heads"
            case layers = "num_layers"
            case rmsNormEpsilon = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case vocabularySize = "vocab_size"
        }
    }

    struct Files: Decodable, Sendable, Equatable {
        let audioEncoder: String
        let decoder: String

        enum CodingKeys: String, CodingKey {
            case audioEncoder = "audio_encoder"
            case decoder
        }
    }

    struct Quantization: Decodable, Sendable, Equatable {
        let bits: Int
        let groupSize: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    static let maximumContextTokens = 131_072
    static let supportedQuantizationBits = Set([2, 3, 4, 5, 6, 8])

    let audio: Audio
    let audioTokenID: Int
    let backend: String
    let decoder: Decoder
    let files: Files
    let maximumContextTokens: Int?
    let modelType: String
    let precision: String
    let quantization: Quantization

    enum CodingKeys: String, CodingKey {
        case audio = "audio_config"
        case audioTokenID = "audio_token_id"
        case backend
        case decoder = "decoder_config"
        case files
        case maximumContextTokens = "max_context_tokens"
        case modelType = "model_type"
        case precision
        case quantization = "quantization_config"
    }

    func validate() throws {
        guard backend == "mlx" else {
            throw MossTranscribeError.invalidConfiguration(
                "MLX bundle backend must be mlx"
            )
        }
        guard modelType == "moss-transcribe-diarize-mlx" else {
            throw MossTranscribeError.invalidConfiguration(
                "unsupported MLX model_type \(modelType)"
            )
        }
        guard
            audio.hiddenSize > 0,
            audio.intermediateSize > 0,
            audio.maximumSourcePositions
                == MossWhisperFeatureExtractor.timeFrames / 2,
            audio.mergeSize == 4,
            audio.attentionHeads > 0,
            audio.hiddenSize.isMultiple(of: audio.attentionHeads),
            audio.layers > 0,
            audio.melBins == MossWhisperFeatureExtractor.melBins
        else {
            throw MossTranscribeError.invalidConfiguration(
                "audio geometry does not match the MOSS Whisper frontend"
            )
        }
        guard
            decoder.headDimension > 0,
            decoder.hiddenSize > 0,
            decoder.intermediateSize > 0,
            decoder.attentionHeads > 0,
            decoder.keyValueHeads > 0,
            decoder.attentionHeads.isMultiple(
                of: decoder.keyValueHeads
            ),
            decoder.layers > 0,
            audioTokenID >= 0,
            decoder.vocabularySize > audioTokenID,
            decoder.ropeTheta > 0
        else {
            throw MossTranscribeError.invalidConfiguration(
                "invalid Qwen3 decoder geometry"
            )
        }
        guard decoder.hiddenSize == audio.hiddenSize else {
            throw MossTranscribeError.invalidConfiguration(
                "audio and decoder hidden sizes must match"
            )
        }
        guard
            maximumContextTokens == nil
                || maximumContextTokens == Self.maximumContextTokens
        else {
            throw MossTranscribeError.invalidConfiguration(
                "MLX context must be \(Self.maximumContextTokens) tokens"
            )
        }
        guard
            Self.supportedQuantizationBits.contains(quantization.bits),
            quantization.groupSize > 0,
            quantization.mode == "affine"
        else {
            throw MossTranscribeError.invalidConfiguration(
                "unsupported decoder quantization: "
                    + "\(quantization.bits)-bit group-"
                    + "\(quantization.groupSize) \(quantization.mode)"
            )
        }
        guard
            precision
                == "fp16-audio-int\(quantization.bits)-decoder"
        else {
            throw MossTranscribeError.invalidConfiguration(
                "precision does not match quantization_config"
            )
        }
        guard
            files.audioEncoder == "audio_encoder.safetensors",
            files.decoder == "decoder.safetensors"
        else {
            throw MossTranscribeError.invalidConfiguration(
                "MLX weight files must use the published bundle names"
            )
        }
    }
}
