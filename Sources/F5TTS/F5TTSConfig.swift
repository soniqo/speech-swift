import Foundation

public struct F5TTSArchitectureConfig: Codable, Equatable, Sendable {
    public let backbone: String
    public let dim: Int
    public let depth: Int
    public let heads: Int
    public let ffMult: Int
    public let textDim: Int
    public let textMaskPadding: Bool
    public let qkNorm: String?
    public let convLayers: Int
    public let peAttnHead: Int?
    public let attnBackend: String
    public let attnMaskEnabled: Bool
    public let checkpointActivations: Bool

    enum CodingKeys: String, CodingKey {
        case backbone
        case dim
        case depth
        case heads
        case ffMult = "ff_mult"
        case textDim = "text_dim"
        case textMaskPadding = "text_mask_padding"
        case qkNorm = "qk_norm"
        case convLayers = "conv_layers"
        case peAttnHead = "pe_attn_head"
        case attnBackend = "attn_backend"
        case attnMaskEnabled = "attn_mask_enabled"
        case checkpointActivations = "checkpoint_activations"
    }
}

public struct F5TTSMelSpecConfig: Codable, Equatable, Sendable {
    public let targetSampleRate: Int
    public let nMelChannels: Int
    public let hopLength: Int
    public let winLength: Int
    public let nFFT: Int
    public let melSpecType: String

    enum CodingKeys: String, CodingKey {
        case targetSampleRate = "target_sample_rate"
        case nMelChannels = "n_mel_channels"
        case hopLength = "hop_length"
        case winLength = "win_length"
        case nFFT = "n_fft"
        case melSpecType = "mel_spec_type"
    }
}

public struct F5TTSBundleFiles: Codable, Equatable, Sendable {
    public let model: String
    public let vocoder: String
    public let vocoderConfig: String
    public let vocab: String
    public let pinyinLexicon: String?

    enum CodingKeys: String, CodingKey {
        case model
        case vocoder
        case vocoderConfig = "vocoder_config"
        case vocab
        case pinyinLexicon = "pinyin_lexicon"
    }
}

public struct F5TTSConversionStats: Codable, Equatable, Sendable {
    public struct Component: Codable, Equatable, Sendable {
        public let sourceTensors: Int?
        public let savedTensors: Int
        public let skipped: [String]?
        public let sourceDtypeCounts: [String: Int]?
        public let sizeMB: Double

        enum CodingKeys: String, CodingKey {
            case sourceTensors = "source_tensors"
            case savedTensors = "saved_tensors"
            case skipped
            case sourceDtypeCounts = "source_dtype_counts"
            case sizeMB = "size_mb"
        }
    }

    public let f5: Component
    public let vocos: Component
}

public struct F5TTSConfig: Codable, Equatable, Sendable {
    public let modelType: String
    public let modelName: String
    public let sourceRepo: String
    public let sourceCheckpoint: String
    public let vocoderRepo: String
    public let license: String
    public let commercialUse: Bool
    public let precision: String
    public let architecture: F5TTSArchitectureConfig
    public let melSpec: F5TTSMelSpecConfig
    public let files: F5TTSBundleFiles
    public let conversion: F5TTSConversionStats?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case modelName = "model_name"
        case sourceRepo = "source_repo"
        case sourceCheckpoint = "source_checkpoint"
        case vocoderRepo = "vocoder_repo"
        case license
        case commercialUse = "commercial_use"
        case precision
        case architecture
        case melSpec = "mel_spec"
        case files
        case conversion
    }

    public static func load(from url: URL) throws -> F5TTSConfig {
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(F5TTSConfig.self, from: data)
        } catch {
            throw F5TTSError.invalidConfig(url, error.localizedDescription)
        }
    }
}

public struct F5TTSBundleInfo: Equatable, Sendable {
    public let directory: URL
    public let config: F5TTSConfig
    public let weightMemory: Int

    public var sampleRate: Int { config.melSpec.targetSampleRate }
}

public enum F5TTSError: Error, LocalizedError, Equatable {
    case invalidConfig(URL, String)
    case unexpectedModelType(String)
    case missingRequiredFile(String)
    case emptyWeightSet(URL)
    case unsupportedText(String)
    case missingTensor(component: String, key: String)
    case invalidTensorShape(component: String, key: String, expected: [Int], actual: [Int])
    case unloaded

    public var errorDescription: String? {
        switch self {
        case .invalidConfig(let url, let reason):
            return "Invalid F5-TTS config at \(url.path): \(reason)"
        case .unexpectedModelType(let type):
            return "Unexpected F5-TTS model_type '\(type)'"
        case .missingRequiredFile(let file):
            return "F5-TTS bundle is missing required file: \(file)"
        case .emptyWeightSet(let url):
            return "F5-TTS bundle has no safetensors weights: \(url.path)"
        case .unsupportedText(let reason):
            return "F5-TTS unsupported text: \(reason)"
        case .missingTensor(let component, let key):
            return "F5-TTS \(component) weights are missing tensor: \(key)"
        case .invalidTensorShape(let component, let key, let expected, let actual):
            return "F5-TTS \(component) tensor \(key) has shape \(actual), expected \(expected)"
        case .unloaded:
            return "F5-TTS model has been unloaded."
        }
    }
}

public enum F5TTSBundleLoader {
    public static let configFileName = "config.json"

    public static func load(from directory: URL) throws -> F5TTSBundleInfo {
        let configURL = directory.appendingPathComponent(configFileName)
        let config = try F5TTSConfig.load(from: configURL)
        guard config.modelType == "f5-tts" else {
            throw F5TTSError.unexpectedModelType(config.modelType)
        }

        let required = [
            "README.md",
            config.files.model,
            config.files.vocoder,
            config.files.vocoderConfig,
            config.files.vocab,
            configFileName,
        ]
        for path in required {
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(path).path) else {
                throw F5TTSError.missingRequiredFile(path)
            }
        }

        let weightFiles = [config.files.model, config.files.vocoder]
        let memory = weightFiles.reduce(0) { total, path in
            let fileURL = directory.appendingPathComponent(path)
            let size = ((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            return total + size
        }
        guard memory > 0 else {
            throw F5TTSError.emptyWeightSet(directory)
        }

        return F5TTSBundleInfo(directory: directory, config: config, weightMemory: memory)
    }
}
