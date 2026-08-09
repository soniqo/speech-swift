import AudioCommon
import Foundation

public enum SpeechLanguageIDError: Error, LocalizedError, Sendable {
    case invalidAudio(String)
    case invalidConfiguration(String)
    case invalidLabels(String)
    case unsupportedEngine(String)

    public var errorDescription: String? {
        switch self {
        case .invalidAudio(let reason):
            return "Invalid language-identification audio: \(reason)"
        case .invalidConfiguration(let reason):
            return "Invalid language-identification model configuration: \(reason)"
        case .invalidLabels(let reason):
            return "Invalid language-identification labels: \(reason)"
        case .unsupportedEngine(let reason):
            return "Unsupported language-identification engine: \(reason)"
        }
    }
}

/// Execution backend for SpeechBrain ECAPA language identification.
public enum LanguageIDEngine: String, Codable, Sendable, CaseIterable {
    /// MLX on Apple Silicon GPU.
    case mlx
    /// Compiled Core ML model using all available Apple compute units.
    case coreML = "coreml"
}

/// One label in the exact upstream VoxLingua107 classifier order.
public struct SpokenLanguageLabel: Codable, Sendable, Equatable, Hashable {
    public let id: Int
    public let code: String
    public let name: String
    public let upstreamLabel: String

    enum CodingKeys: String, CodingKey {
        case id
        case code
        case name
        case upstreamLabel = "upstream_label"
    }

    public init(id: Int, code: String, name: String, upstreamLabel: String) {
        self.id = id
        self.code = code
        self.name = name
        self.upstreamLabel = upstreamLabel
    }
}

/// Ranked closed-set prediction from the 107-language classifier.
public struct LanguageIDPrediction: Codable, Sendable, Equatable {
    public let label: SpokenLanguageLabel
    public let probability: Float
    public let logProbability: Float

    public init(
        label: SpokenLanguageLabel,
        probability: Float,
        logProbability: Float
    ) {
        self.label = label
        self.probability = probability
        self.logProbability = logProbability
    }
}

/// Language-ID output, including how a long recording was aggregated.
public struct LanguageIDResult: Codable, Sendable, Equatable {
    public let predictions: [LanguageIDPrediction]
    public let analyzedDuration: TimeInterval
    public let windowCount: Int

    public var best: LanguageIDPrediction? { predictions.first }

    public init(
        predictions: [LanguageIDPrediction],
        analyzedDuration: TimeInterval,
        windowCount: Int
    ) {
        self.predictions = predictions
        self.analyzedDuration = analyzedDuration
        self.windowCount = windowCount
    }
}

/// Minimal subset of the reproducible export contract consumed by the runtime.
public struct LanguageIDModelConfiguration: Codable, Sendable, Equatable {
    public let modelType: String
    public let task: String
    public let format: String
    public let sampleRate: Int
    public let nFFT: Int
    public let winLength: Int
    public let hopLength: Int
    public let nMels: Int
    public let minimumMelFrames: Int
    public let maximumMelFrames: Int
    public let embeddingDimension: Int
    public let classCount: Int
    public let outputName: String
    public let artifact: String
    public let sourceModel: String
    public let sourceRevision: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case task
        case format
        case sampleRate = "sample_rate"
        case nFFT = "n_fft"
        case winLength = "win_length"
        case hopLength = "hop_length"
        case nMels = "n_mels"
        case minimumMelFrames = "minimum_mel_frames"
        case maximumMelFrames = "maximum_mel_frames"
        case embeddingDimension = "embedding_dimension"
        case classCount = "class_count"
        case outputName = "output_name"
        case artifact
        case sourceModel = "source_model"
        case sourceRevision = "source_revision"
    }

    public func validate(for engine: LanguageIDEngine) throws {
        guard modelType == "speechbrain-ecapa-voxlingua107-language-id" else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "unexpected model_type '\(modelType)'"
            )
        }
        guard task == "audio-classification" else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "unexpected task '\(task)'"
            )
        }
        guard format == engine.rawValue else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "artifact format '\(format)' does not match engine '\(engine.rawValue)'"
            )
        }
        guard sampleRate == SpeechBrainFbank.sampleRate,
              nFFT == SpeechBrainFbank.fftSize,
              winLength == SpeechBrainFbank.windowLength,
              hopLength == SpeechBrainFbank.hopLength,
              nMels == 60,
              embeddingDimension == 256,
              classCount == 107,
              outputName == "log_probabilities",
              minimumMelFrames == 10,
              maximumMelFrames == 3_001,
              !artifact.isEmpty,
              !sourceModel.isEmpty,
              !sourceRevision.isEmpty
        else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "frontend or classifier dimensions do not match VoxLingua107 ECAPA"
            )
        }
    }

    public static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        do {
            return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: directory.path,
                reason: "could not decode config.json",
                underlying: error
            )
        }
    }
}

enum SpokenLanguageLabelLoader {
    static func load(from directory: URL, expectedCount: Int) throws -> [SpokenLanguageLabel] {
        let url = directory.appendingPathComponent("labels.json")
        let labels: [SpokenLanguageLabel]
        do {
            labels = try JSONDecoder().decode(
                [SpokenLanguageLabel].self,
                from: Data(contentsOf: url)
            )
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: directory.path,
                reason: "could not decode labels.json",
                underlying: error
            )
        }

        guard labels.count == expectedCount else {
            throw SpeechLanguageIDError.invalidLabels(
                "expected \(expectedCount), found \(labels.count)"
            )
        }
        guard labels.map(\.id) == Array(0..<expectedCount) else {
            throw SpeechLanguageIDError.invalidLabels(
                "ids must be contiguous and preserve the upstream classifier order"
            )
        }
        guard Set(labels.map(\.code)).count == expectedCount else {
            throw SpeechLanguageIDError.invalidLabels("language codes are not unique")
        }
        guard labels.allSatisfy({ label in
            !label.code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !label.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                && !label.upstreamLabel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }) else {
            throw SpeechLanguageIDError.invalidLabels(
                "language code, name, and upstream label must not be empty"
            )
        }
        return labels
    }
}
