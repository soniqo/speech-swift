import CoreML
import Foundation
import AudioCommon

extension CanaryASRModel {
    /// Download (or reuse the cache) and load a Canary bundle.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model identifier
    ///   - language: source language for the decode prompt
    ///   - progressHandler: optional `(fraction, status)` callback
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        language: String = "en",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CanaryASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading Canary model: \(effectiveModelId)")

        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: resolvedCacheDir,
                additionalFiles: [
                    "CanaryEncoder.mlmodelc/**",
                    "CanaryPrefill.mlmodelc/**",
                    "CanaryDecoder.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        // config.json is required, not a convenience: the decode prompt, cache
        // dimensions and end-of-text id live there. Falling back to a built-in
        // default for a bundle that ships a different template would decode
        // fluent text off the wrong prompt.
        progressHandler?(0.70, "Loading configuration...")
        let configURL = resolvedCacheDir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Bundle has no config.json — it carries the decode contract",
                underlying: nil)
        }
        let config: CanaryConfig
        do {
            config = try JSONDecoder().decode(CanaryConfig.self, from: Data(contentsOf: configURL))
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to parse config.json", underlying: error)
        }

        progressHandler?(0.75, "Loading vocabulary...")
        let vocabulary: CanaryVocabulary
        do {
            vocabulary = try CanaryVocabulary.load(
                from: resolvedCacheDir.appendingPathComponent("vocab.json"))
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to load vocabulary", underlying: error)
        }

        progressHandler?(0.80, "Loading CoreML models...")
        let models = try loadModels(from: resolvedCacheDir, modelId: effectiveModelId)

        progressHandler?(1.0, "Ready")
        return CanaryASRModel(
            config: config,
            encoder: models.encoder,
            prefill: models.prefill,
            step: models.step,
            vocabulary: vocabulary,
            language: language
        )
    }

    private static func loadModels(
        from directory: URL, modelId: String
    ) throws -> (encoder: MLModel, prefill: MLModel, step: MLModel) {
        // `.cpuAndNeuralEngine` measured fastest end to end (147 ms for a 2.9 s
        // utterance, against 181 ms on CPU). `.all` is deliberately not offered:
        // it pays a large GPU planning cost on the first call and does not win
        // afterwards.
        let configuration = MLModelConfiguration()
        #if targetEnvironment(simulator)
        configuration.computeUnits = .cpuOnly
        #else
        configuration.computeUnits = .cpuAndNeuralEngine
        #endif

        func load(_ name: String) throws -> MLModel {
            let url = directory.appendingPathComponent(name)
            do {
                return try MLModel(contentsOf: url, configuration: configuration)
            } catch {
                // Fall back to CPU rather than failing the load outright.
                let cpu = MLModelConfiguration()
                cpu.computeUnits = .cpuOnly
                do {
                    let model = try MLModel(contentsOf: url, configuration: cpu)
                    AudioLog.modelLoading.warning("\(name) fell back to .cpuOnly: \(error)")
                    return model
                } catch {
                    throw AudioModelError.modelLoadFailed(
                        modelId: modelId, reason: "Failed to load \(name)", underlying: error)
                }
            }
        }

        return (try load("CanaryEncoder.mlmodelc"),
                try load("CanaryPrefill.mlmodelc"),
                try load("CanaryDecoder.mlmodelc"))
    }
}
