import AudioCommon
import Foundation
import VoiceCloneTTSCommon

public final class F5TTSModel: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/F5-TTS-v1-Base-MLX-fp16"
    public static let modelKey = "f5-tts-v1"

    private static let requiredFiles = [
        "soniqo_manifest.json",
        "README.md",
        "config.json",
        "F5TTS_v1_Base/model_1250000.safetensors",
        "F5TTS_v1_Base/vocab.txt",
    ]

    public let bundleDirectory: URL
    public let manifest: VoiceCloneExportManifest
    public let sampleRate: Int

    private let bundleInfo: VoiceCloneBundleInfo
    private var loaded = true

    private init(bundleInfo: VoiceCloneBundleInfo) {
        self.bundleInfo = bundleInfo
        self.bundleDirectory = bundleInfo.directory
        self.manifest = bundleInfo.manifest
        self.sampleRate = bundleInfo.sampleRate
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> F5TTSModel {
        let directory = try await VoiceCloneBundleLoader.download(
            modelId: modelId,
            requiredFiles: requiredFiles,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            progressHandler: progressHandler)
        return try await fromBundle(directory) { progress, message in
            progressHandler?(0.85 + progress * 0.15, message)
        }
    }

    public static func fromBundle(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> F5TTSModel {
        let info = try VoiceCloneBundleLoader.load(
            from: directory,
            expectedModelKey: modelKey,
            progressHandler: progressHandler)
        return F5TTSModel(bundleInfo: info)
    }

    public var isLoaded: Bool {
        loaded
    }

    public var memoryFootprint: Int {
        loaded ? bundleInfo.weightMemory : 0
    }

    public func unload() {
        loaded = false
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try ensureLoaded()
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "F5-TTS v1")
    }

    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String,
        language: String? = nil
    ) async throws -> [Float] {
        try ensureLoaded()
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "F5-TTS v1")
    }

    private func ensureLoaded() throws {
        guard loaded else {
            throw AudioModelError.inferenceFailed(
                operation: "F5-TTS v1 synthesis",
                reason: "Model has been unloaded.")
        }
    }
}
