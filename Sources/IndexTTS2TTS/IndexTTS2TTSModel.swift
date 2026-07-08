import AudioCommon
import Foundation
import VoiceCloneTTSCommon

public final class IndexTTS2TTSModel: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/IndexTTS2-MLX-fp16"
    public static let modelKey = "indextts2"

    public struct AuxiliaryModel: Equatable, Sendable {
        public let name: String
        public let repository: String
        public let files: [String]
        public let purpose: String

        public init(name: String, repository: String, files: [String], purpose: String) {
            self.name = name
            self.repository = repository
            self.files = files
            self.purpose = purpose
        }
    }

    public static let auxiliaryModels: [AuxiliaryModel] = [
        AuxiliaryModel(
            name: "w2v-bert-2.0",
            repository: "facebook/w2v-bert-2.0",
            files: [
                "aux/w2v-bert-2.0/config.json",
                "aux/w2v-bert-2.0/model.safetensors",
                "aux/w2v-bert-2.0/preprocessor_config.json",
            ],
            purpose: "SeamlessM4T feature extraction and semantic hidden states for reference audio"),
        AuxiliaryModel(
            name: "MaskGCT semantic codec",
            repository: "amphion/MaskGCT",
            files: [
                "aux/maskgct/config.json",
                "aux/maskgct/README.md",
                "aux/maskgct/semantic_codec/model.safetensors",
            ],
            purpose: "Quantizes semantic reference features and maps generated codes to embeddings"),
        AuxiliaryModel(
            name: "CAMPPlus",
            repository: "funasr/campplus",
            files: [
                "aux/campplus/campplus_cn_common.safetensors",
                "aux/campplus/configuration.json",
                "aux/campplus/README.md",
            ],
            purpose: "Extracts the 192-d global style vector from the speaker reference"),
        AuxiliaryModel(
            name: "BigVGAN",
            repository: "nvidia/bigvgan_v2_22khz_80band_256x",
            files: [
                "aux/bigvgan/bigvgan_generator.safetensors",
                "aux/bigvgan/config.json",
                "aux/bigvgan/LICENSE",
                "aux/bigvgan/README.md",
            ],
            purpose: "Decodes generated 80-band mel spectrograms to waveform"),
    ]

    private static let baseRequiredFiles = [
        "soniqo_manifest.json",
        "README.md",
        "LICENSE.txt",
        "LICENSE_ZH.txt",
        "config.json",
        "config.yaml",
        "bpe.model",
        "gpt.safetensors",
        "s2mel.safetensors",
        "feat1.safetensors",
        "feat2.safetensors",
        "wav2vec2bert_stats.safetensors",
        "qwen0.6bemo4-merge/config.json",
        "qwen0.6bemo4-merge/generation_config.json",
        "qwen0.6bemo4-merge/tokenizer.json",
        "qwen0.6bemo4-merge/tokenizer_config.json",
        "qwen0.6bemo4-merge/special_tokens_map.json",
        "qwen0.6bemo4-merge/added_tokens.json",
        "qwen0.6bemo4-merge/vocab.json",
        "qwen0.6bemo4-merge/merges.txt",
        "qwen0.6bemo4-merge/chat_template.jinja",
        "qwen0.6bemo4-merge/model.safetensors",
    ]

    private static let requiredFiles =
        baseRequiredFiles + auxiliaryModels.flatMap(\.files)

    public let bundleDirectory: URL
    public let manifest: VoiceCloneExportManifest
    public let sampleRate: Int
    public let tokenizer: IndexTTS2Tokenizer?

    private let bundleInfo: VoiceCloneBundleInfo
    private var loaded = true

    private init(bundleInfo: VoiceCloneBundleInfo) {
        self.bundleInfo = bundleInfo
        self.bundleDirectory = bundleInfo.directory
        self.manifest = bundleInfo.manifest
        self.sampleRate = bundleInfo.sampleRate
        self.tokenizer = try? IndexTTS2Tokenizer(
            modelURL: bundleInfo.directory.appendingPathComponent("bpe.model"))
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> IndexTTS2TTSModel {
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
    ) async throws -> IndexTTS2TTSModel {
        let info = try VoiceCloneBundleLoader.load(
            from: directory,
            expectedModelKey: modelKey,
            progressHandler: progressHandler)
        return IndexTTS2TTSModel(bundleInfo: info)
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
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "IndexTTS2")
    }

    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String? = nil,
        emotionReferenceAudio: URL? = nil,
        language: String? = nil
    ) async throws -> [Float] {
        try ensureLoaded()
        throw VoiceCloneBundleLoader.runtimeNotImplemented(modelName: "IndexTTS2")
    }

    private func ensureLoaded() throws {
        guard loaded else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Model has been unloaded.")
        }
    }
}
