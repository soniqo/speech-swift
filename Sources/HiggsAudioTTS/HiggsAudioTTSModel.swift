import AudioCommon
import Foundation
import VoiceCloneTTSCommon

public final class HiggsAudioTTSModel: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Higgs-Audio-v3-TTS-4B-MLX-fp16"
    public static let modelKey = "higgs-audio-v3"

    private static let requiredFiles = [
        "soniqo_manifest.json",
        "README.md",
        "LICENSE",
        "PROMPTING.md",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors",
        "model.safetensors.index.json",
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
    ) async throws -> HiggsAudioTTSModel {
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
    ) async throws -> HiggsAudioTTSModel {
        let info = try VoiceCloneBundleLoader.load(
            from: directory,
            expectedModelKey: modelKey,
            progressHandler: progressHandler)
        return HiggsAudioTTSModel(bundleInfo: info)
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
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "Higgs Audio v3")
    }

    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String? = nil,
        voicePrompt: String? = nil,
        language: String? = nil
    ) async throws -> [Float] {
        try ensureLoaded()
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "Higgs Audio v3")
    }

    private func ensureLoaded() throws {
        guard loaded else {
            throw AudioModelError.inferenceFailed(
                operation: "Higgs Audio v3 synthesis",
                reason: "Model has been unloaded.")
        }
    }
}
