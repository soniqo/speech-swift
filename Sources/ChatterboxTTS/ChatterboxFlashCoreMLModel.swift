#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

public final class ChatterboxFlashCoreMLModel: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Chatterbox-Flash-CoreML"

    public let directory: URL
    public let bundleConfig: ChatterboxFlashCoreMLBundleConfig?
    public let t3: ChatterboxFlashT3Graphs
    public let audio: ChatterboxFlashAudioGraphs
    public var sampleRate: Int { audio.config.sampleRate }

    public init(directory: URL, computeUnits: MLComputeUnits = .all) throws {
        self.directory = directory
        self.bundleConfig = try? ChatterboxFlashCoreMLBundleConfig.load(from: directory)

        let t3Config = (try? ChatterboxFlashT3Config.load(from: directory))
            ?? bundleConfig?.components.t3
            ?? ChatterboxFlashT3Config.fallback
        let audioConfig = try ChatterboxFlashAudioConfig.load(from: directory)
        self.t3 = try ChatterboxFlashT3Graphs(directory: directory, config: t3Config, computeUnits: computeUnits)
        self.audio = try ChatterboxFlashAudioGraphs(directory: directory, config: audioConfig, computeUnits: computeUnits)
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        localPath: URL? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ChatterboxFlashCoreMLModel {
        let directory: URL
        if let localPath {
            directory = localPath
        } else {
            directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
            progressHandler?(0.0, "Downloading Chatterbox Flash Core ML...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: directory,
                additionalFiles: [
                    "README.md",
                    "t3/config.json",
                    "t3/tokenizer.json",
                    "t3/uncond_block_prior.npy",
                    "t3/ConditioningEncoder.mlmodelc/**",
                    "t3/TextPrefill.mlmodelc/**",
                    "t3/BlockDecoder.mlmodelc/**",
                    "audio/audio_config.json",
                    "audio/FlowSpeakerProjector.mlmodelc/**",
                    "audio/FlowEncoder.mlmodelc/**",
                    "audio/FlowEstimator.mlmodelc/**",
                    "audio/HiFTVocoder.mlmodelc/**",
                ],
                offlineMode: offlineMode
            ) { progress in
                progressHandler?(progress * 0.8, "Downloading Chatterbox Flash Core ML...")
            }
        }

        progressHandler?(0.8, "Loading Chatterbox Flash Core ML...")
        let model = try ChatterboxFlashCoreMLModel(directory: directory, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return model
    }

    public func synthesizeAudio(
        speechTokens: [Int],
        reference: ChatterboxFlashS3GenReference,
        seed: UInt64 = 0
    ) throws -> [Float] {
        try audio.synthesize(speechTokens: speechTokens, reference: reference, seed: seed)
    }

    public func generateSpeechTokens(
        text: String,
        conditioning: ChatterboxFlashT3Conditioning,
        options: ChatterboxFlashGenerationOptions = ChatterboxFlashGenerationOptions()
    ) throws -> [Int] {
        try t3.generateSpeechTokens(text: text, conditioning: conditioning, options: options)
    }

    public func generate(
        text: String,
        conditioning: ChatterboxFlashConditioning,
        options: ChatterboxFlashGenerationOptions = ChatterboxFlashGenerationOptions()
    ) throws -> [Float] {
        var boundedOptions = options
        let capacity = try speechTokenCapacity(for: conditioning.audio)
        if let requested = boundedOptions.maxSpeechTokens {
            boundedOptions.maxSpeechTokens = min(requested, capacity)
        } else {
            boundedOptions.maxSpeechTokens = capacity
        }
        guard (boundedOptions.maxSpeechTokens ?? 0) > 0 else {
            throw ChatterboxFlashCoreMLError.invalidShape("reference conditioning leaves no room for speech tokens")
        }
        let speechTokens = try generateSpeechTokens(
            text: text,
            conditioning: conditioning.t3,
            options: boundedOptions
        )
        return try synthesizeAudio(speechTokens: speechTokens, reference: conditioning.audio, seed: boundedOptions.seed)
    }

    private func speechTokenCapacity(for reference: ChatterboxFlashS3GenReference) throws -> Int {
        guard reference.promptFeatureFrames <= audio.config.melLen else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "prompt feature frames \(reference.promptFeatureFrames) exceed audio mel limit \(audio.config.melLen)"
            )
        }
        let maxByTokenBuffer = audio.config.tokenLen - reference.promptToken.count
        let maxTotalTokensByMel = audio.config.melLen / audio.config.tokenMelRatio
        let maxByMel = maxTotalTokensByMel - reference.promptToken.count
        let maxByT3Cache = t3.config.maxSeq - t3.config.condLen - 1
        return max(0, min(maxByTokenBuffer, maxByMel, maxByT3Cache))
    }
}
#endif
