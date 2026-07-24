import Foundation

public enum NemotronMLXVariant: String, CaseIterable, Sendable {
    case int5
    case int8

    public var modelId: String {
        switch self {
        case .int5:
            return "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-MLX-5bit"
        case .int8:
            return "aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-MLX-8bit"
        }
    }

    public var quantizationBits: Int {
        switch self {
        case .int5: return 5
        case .int8: return 8
        }
    }
}

public struct NemotronMLXQuantization: Codable, Equatable, Sendable {
    public let mode: String
    public let bits: Int
    public let groupSize: Int

    enum CodingKeys: String, CodingKey {
        case mode
        case bits
        case groupSize = "group_size"
    }
}

public struct NemotronMLXConfiguration: Decodable, Equatable, Sendable {
    public struct Preprocessor: Decodable, Equatable, Sendable {
        public let sampleRate: Int
        public let features: Int
        public let nFFT: Int
        public let windowSize: Double
        public let windowStride: Double
        public let preEmphasis: Float

        enum CodingKeys: String, CodingKey {
            case sampleRate = "sample_rate"
            case features
            case nFFT = "n_fft"
            case windowSize = "window_size"
            case windowStride = "window_stride"
            case preEmphasis = "preemph"
        }
    }

    public struct Encoder: Decodable, Equatable, Sendable {
        public let featureInput: Int
        public let layers: Int
        public let hiddenSize: Int
        public let attentionHeads: Int
        public let feedForwardExpansion: Int
        public let subsamplingFactor: Int
        public let convolutionKernelSize: Int
        public let subsamplingChannels: Int
        public let useBias: Bool
        public let convolutionNormalization: String

        enum CodingKeys: String, CodingKey {
            case featureInput = "feat_in"
            case layers = "n_layers"
            case hiddenSize = "d_model"
            case attentionHeads = "n_heads"
            case feedForwardExpansion = "ff_expansion_factor"
            case subsamplingFactor = "subsampling_factor"
            case convolutionKernelSize = "conv_kernel_size"
            case subsamplingChannels = "subsampling_conv_channels"
            case useBias = "use_bias"
            case convolutionNormalization = "conv_norm_type"
        }
    }

    public struct Decoder: Decodable, Equatable, Sendable {
        public struct PredictionNetwork: Decodable, Equatable, Sendable {
            public let hiddenSize: Int
            public let layers: Int

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "pred_hidden"
                case layers = "pred_rnn_layers"
            }
        }

        public let blankAsPadding: Bool
        public let vocabularySize: Int
        public let predictionNetwork: PredictionNetwork

        enum CodingKeys: String, CodingKey {
            case blankAsPadding = "blank_as_pad"
            case vocabularySize = "vocab_size"
            case predictionNetwork = "prednet"
        }
    }

    public struct Joint: Decodable, Equatable, Sendable {
        public struct Network: Decodable, Equatable, Sendable {
            public let hiddenSize: Int
            public let activation: String
            public let encoderHiddenSize: Int
            public let predictionHiddenSize: Int

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "joint_hidden"
                case activation
                case encoderHiddenSize = "encoder_hidden"
                case predictionHiddenSize = "pred_hidden"
            }
        }

        public let classes: Int
        public let network: Network

        enum CodingKeys: String, CodingKey {
            case classes = "num_classes"
            case network = "jointnet"
        }
    }

    public struct PromptKernel: Decodable, Equatable, Sendable {
        public let promptCount: Int
        public let hiddenSize: Int
        public let modelSize: Int

        enum CodingKeys: String, CodingKey {
            case promptCount = "num_prompts"
            case hiddenSize = "hidden"
            case modelSize = "d_model"
        }
    }

    public struct Streaming: Decodable, Equatable, Sendable {
        public let chunkMilliseconds: Int
        public let melFrames: Int
        public let preCacheSize: Int
        public let outputFrames: Int
        public let attentionLeftContext: Int
        public let convolutionCacheSize: Int

        enum CodingKeys: String, CodingKey {
            case chunkMilliseconds = "chunk_ms"
            case melFrames = "mel_frames"
            case preCacheSize = "pre_cache_size"
            case outputFrames = "output_frames"
            case attentionLeftContext = "attention_left_context"
            case convolutionCacheSize = "conv_cache_size"
        }
    }

    public let modelType: String
    public let sampleRate: Int
    public let vocabularySize: Int
    public let preprocessor: Preprocessor
    public let encoder: Encoder
    public let decoder: Decoder
    public let joint: Joint
    public let promptKernel: PromptKernel
    public let streaming: Streaming
    public let quantization: NemotronMLXQuantization

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case vocabularySize = "vocab_size"
        case preprocessor
        case encoder
        case decoder
        case joint
        case promptKernel = "prompt_kernel"
        case streaming
        case quantization
    }

    public func validate() throws {
        guard modelType == "nemotron_streaming_rnnt_mlx" else {
            throw NemotronMLXError.invalidConfiguration(
                "unexpected model_type \(modelType)")
        }
        guard sampleRate == 16_000,
              preprocessor.sampleRate == sampleRate,
              preprocessor.features == 128,
              preprocessor.nFFT == 512,
              abs(preprocessor.windowSize - 0.025) < 1e-9,
              abs(preprocessor.windowStride - 0.01) < 1e-9,
              abs(preprocessor.preEmphasis - 0.97) < 1e-6
        else {
            throw NemotronMLXError.invalidConfiguration(
                "unsupported audio frontend geometry")
        }
        guard vocabularySize == 13_087,
              decoder.vocabularySize == vocabularySize,
              joint.classes == vocabularySize,
              encoder.featureInput == 128,
              encoder.layers == 24,
              encoder.hiddenSize == 1_024,
              encoder.attentionHeads == 8,
              encoder.feedForwardExpansion == 4,
              encoder.subsamplingFactor == 8,
              encoder.convolutionKernelSize == 9,
              encoder.subsamplingChannels == 256,
              !encoder.useBias,
              encoder.convolutionNormalization == "layer_norm",
              decoder.blankAsPadding,
              decoder.predictionNetwork.hiddenSize == 640,
              decoder.predictionNetwork.layers == 2,
              joint.network.hiddenSize == 640,
              joint.network.activation == "relu",
              joint.network.encoderHiddenSize == 1_024,
              joint.network.predictionHiddenSize == 640,
              promptKernel.promptCount == 128,
              promptKernel.hiddenSize == 2_048,
              promptKernel.modelSize == 1_024
        else {
            throw NemotronMLXError.invalidConfiguration(
                "unsupported Nemotron network geometry")
        }
        guard streaming.chunkMilliseconds == 320,
              streaming.melFrames == 32,
              streaming.preCacheSize == 9,
              streaming.outputFrames == 4,
              streaming.attentionLeftContext == 56,
              streaming.convolutionCacheSize == 8
        else {
            throw NemotronMLXError.invalidConfiguration(
                "unsupported streaming cache geometry")
        }
        guard quantization.mode == "affine",
              [5, 8].contains(quantization.bits),
              quantization.groupSize == 64
        else {
            throw NemotronMLXError.invalidConfiguration(
                "only affine group-64 INT5 and INT8 bundles are supported")
        }
    }
}

public enum NemotronMLXError: LocalizedError {
    case missingModelFile(String)
    case invalidConfiguration(String)
    case inferenceFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingModelFile(let file):
            return "Missing Nemotron MLX model file: \(file)"
        case .invalidConfiguration(let reason):
            return "Invalid Nemotron MLX configuration: \(reason)"
        case .inferenceFailed(let reason):
            return "Nemotron MLX inference failed: \(reason)"
        }
    }
}
