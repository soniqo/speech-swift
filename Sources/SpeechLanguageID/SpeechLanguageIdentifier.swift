import AudioCommon
import CoreML
import Foundation
import MLX
import MLXCommon
import MLXNN

/// Native Apple runtime for SpeechBrain's ECAPA VoxLingua107 classifier.
///
/// This is a closed-set model. A high-ranked result is not proof that the
/// language belongs to the 107-label inventory; applications should calibrate
/// their own unknown-language rejection policy.
public final class SpeechLanguageIdentifier {
    public static let defaultMLXModelID =
        "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-MLX"
    public static let defaultCoreMLModelID =
        "aufklarer/SpeechBrain-ECAPA-VoxLingua107-21M-CoreML"

    public let engine: LanguageIDEngine
    public let configuration: LanguageIDModelConfiguration
    public let labels: [SpokenLanguageLabel]

    private let frontend: SpeechBrainFbank
    private let network: SpeechBrainLanguageIDNetwork?

    private let coreMLModel: MLModel?

    private init(
        engine: LanguageIDEngine,
        configuration: LanguageIDModelConfiguration,
        labels: [SpokenLanguageLabel],
        network: SpeechBrainLanguageIDNetwork?,
        coreMLModel: MLModel?
    ) {
        self.engine = engine
        self.configuration = configuration
        self.labels = labels
        self.frontend = SpeechBrainFbank(melBinCount: configuration.nMels)
        self.network = network
        self.coreMLModel = coreMLModel
    }

    /// Download and load a published `aufklarer` export.
    public static func fromPretrained(
        modelID: String? = nil,
        engine: LanguageIDEngine = .mlx,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SpeechLanguageIdentifier {
        let resolvedModelID = modelID ?? {
            switch engine {
            case .mlx: return defaultMLXModelID
            case .coreML: return defaultCoreMLModelID
            }
        }()
        let directory = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: resolvedModelID)

        let additionalFiles: [String]
        switch engine {
        case .mlx:
            additionalFiles = ["labels.json"]
        case .coreML:
            additionalFiles = [
                "SpeechBrainECAPAVoxLingua107.mlmodelc/**",
                "labels.json",
            ]
        }

        progressHandler?(0, "Downloading language-identification model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: resolvedModelID,
            to: directory,
            additionalFiles: additionalFiles,
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.85, "Downloading model files...")
            }
        )
        progressHandler?(0.85, "Loading language-identification model...")
        let model = try fromLocal(directory: directory, engine: engine)
        progressHandler?(1, "Ready")
        return model
    }

    /// Load a complete local export directory containing config, labels, and
    /// either MLX safetensors or a compiled Core ML bundle.
    public static func fromLocal(
        directory: URL,
        engine: LanguageIDEngine
    ) throws -> SpeechLanguageIdentifier {
        let configuration = try LanguageIDModelConfiguration.load(from: directory)
        try configuration.validate(for: engine)
        let labels = try SpokenLanguageLabelLoader.load(
            from: directory,
            expectedCount: configuration.classCount
        )
        let artifactURL = try validatedArtifactURL(
            configuration.artifact,
            in: directory,
            engine: engine
        )

        switch engine {
        case .mlx:
            let network = SpeechBrainLanguageIDNetwork()
            do {
                let weights = try MLX.loadArrays(url: artifactURL)
                try network.update(
                    parameters: ModuleParameters.unflattened(weights),
                    verify: .all
                )
                network.train(false)
                MLX.eval(network.parameters())
            } catch {
                throw AudioModelError.weightLoadingFailed(
                    path: artifactURL.path,
                    underlying: error
                )
            }
            return SpeechLanguageIdentifier(
                engine: engine,
                configuration: configuration,
                labels: labels,
                network: network,
                coreMLModel: nil
            )

        case .coreML:
            let model: MLModel
            do {
                model = try CoreMLLoader.load(
                    url: artifactURL,
                    computeUnits: .all,
                    name: "speechbrain-voxlingua107"
                )
            } catch {
                throw AudioModelError.modelLoadFailed(
                    modelId: directory.path,
                    reason: "failed to load compiled Core ML model",
                    underlying: error
                )
            }
            return SpeechLanguageIdentifier(
                engine: engine,
                configuration: configuration,
                labels: labels,
                network: nil,
                coreMLModel: model
            )
        }
    }

    /// Rank the model's known labels. Recordings longer than the artifact's
    /// maximum frame count are classified in non-overlapping windows and their
    /// probabilities are duration-weighted.
    public func identify(
        audio: [Float],
        sampleRate: Int,
        topK: Int = 5
    ) throws -> LanguageIDResult {
        guard sampleRate > 0 else {
            throw SpeechLanguageIDError.invalidAudio("sample rate must be positive")
        }
        guard (1...labels.count).contains(topK) else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "topK must be between 1 and \(labels.count)"
            )
        }

        let samples = sampleRate == configuration.sampleRate
            ? audio
            : AudioFileLoader.resample(
                audio,
                from: sampleRate,
                to: configuration.sampleRate
            )
        guard samples.allSatisfy(\.isFinite) else {
            throw SpeechLanguageIDError.invalidAudio("samples contain NaN or infinity")
        }

        let minimumSamples = max(
            1,
            (configuration.minimumMelFrames - 1) * configuration.hopLength
        )
        let maximumSamples = max(
            minimumSamples,
            (configuration.maximumMelFrames - 1) * configuration.hopLength
        )
        guard samples.count >= minimumSamples else {
            throw SpeechLanguageIDError.invalidAudio(
                "at least \(minimumSamples) samples "
                    + "(\(String(format: "%.2f", Double(minimumSamples) / Double(configuration.sampleRate))) s) are required"
            )
        }

        var weightedProbabilities = [Double](
            repeating: 0,
            count: labels.count
        )
        var analyzedSamples = 0
        var windowCount = 0
        var start = 0
        while start < samples.count {
            let end = min(start + maximumSamples, samples.count)
            let count = end - start
            if count < minimumSamples { break }

            let chunk = Array(samples[start..<end])
            let logProbabilities = try inferLogProbabilities(chunk)
            guard logProbabilities.count == labels.count else {
                throw AudioModelError.inferenceFailed(
                    operation: "LanguageIdentification",
                    reason: "expected \(labels.count) scores, found \(logProbabilities.count)"
                )
            }
            guard logProbabilities.allSatisfy(\.isFinite) else {
                throw AudioModelError.inferenceFailed(
                    operation: "LanguageIdentification",
                    reason: "model returned NaN or infinite scores"
                )
            }
            for index in weightedProbabilities.indices {
                weightedProbabilities[index] += Double(count)
                    * exp(Double(logProbabilities[index]))
            }
            analyzedSamples += count
            windowCount += 1
            start = end
        }

        guard analyzedSamples > 0 else {
            throw SpeechLanguageIDError.invalidAudio("no complete inference window")
        }
        let denominator = Double(analyzedSamples)
        let probabilities = weightedProbabilities.map { Float($0 / denominator) }
        let ranked = probabilities.indices.sorted {
            if probabilities[$0] == probabilities[$1] { return $0 < $1 }
            return probabilities[$0] > probabilities[$1]
        }
        let predictions = ranked.prefix(topK).map { index in
            let probability = probabilities[index]
            return LanguageIDPrediction(
                label: labels[index],
                probability: probability,
                logProbability: log(max(probability, Float.leastNonzeroMagnitude))
            )
        }
        return LanguageIDResult(
            predictions: predictions,
            analyzedDuration: Double(analyzedSamples) / Double(configuration.sampleRate),
            windowCount: windowCount
        )
    }

    private func inferLogProbabilities(_ samples: [Float]) throws -> [Float] {
        let features = try frontend.extract(samples)
        guard features.frameCount >= configuration.minimumMelFrames,
              features.frameCount <= configuration.maximumMelFrames
        else {
            throw SpeechLanguageIDError.invalidAudio(
                "frontend produced unsupported frame count \(features.frameCount)"
            )
        }

        switch engine {
        case .mlx:
            guard let network else {
                throw AudioModelError.inferenceFailed(
                    operation: "LanguageIdentification",
                    reason: "MLX network is not loaded"
                )
            }
            let input = MLXArray(
                features.values,
                [1, features.frameCount, features.melBinCount]
            )
            let output = network(input)
            MLX.eval(output)
            return output[0].asArray(Float.self)

        case .coreML:
            guard let coreMLModel else {
                throw AudioModelError.inferenceFailed(
                    operation: "LanguageIdentification",
                    reason: "Core ML model is not loaded"
                )
            }
            let input = try MLMultiArray(
                shape: [1, NSNumber(value: features.frameCount), 60],
                dataType: .float32
            )
            let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
            features.values.withUnsafeBufferPointer { source in
                pointer.update(from: source.baseAddress!, count: source.count)
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "mel_features": MLFeatureValue(multiArray: input),
            ])
            let prediction = try coreMLModel.prediction(from: provider)
            guard let output = prediction.featureValue(
                for: configuration.outputName
            )?.multiArrayValue else {
                throw AudioModelError.inferenceFailed(
                    operation: "LanguageIdentification",
                    reason: "missing '\(configuration.outputName)' output"
                )
            }
            return (0..<configuration.classCount).map { output[$0].floatValue }
        }
    }

    private static func validatedArtifactURL(
        _ artifact: String,
        in directory: URL,
        engine: LanguageIDEngine
    ) throws -> URL {
        guard !artifact.isEmpty,
              (artifact as NSString).lastPathComponent == artifact,
              artifact != ".",
              artifact != ".."
        else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "artifact must be a safe top-level file name"
            )
        }
        let expectedExtension = engine == .mlx ? "safetensors" : "mlmodelc"
        guard (artifact as NSString).pathExtension == expectedExtension else {
            throw SpeechLanguageIDError.invalidConfiguration(
                "artifact '\(artifact)' is not a .\(expectedExtension) bundle"
            )
        }
        let url = directory.appendingPathComponent(artifact)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: directory.path,
                reason: "missing artifact \(artifact)"
            )
        }
        return url
    }
}
