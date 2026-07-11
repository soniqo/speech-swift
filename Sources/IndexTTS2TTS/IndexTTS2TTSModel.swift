import AudioCommon
import Foundation

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
    public let manifest: IndexTTS2ExportManifest
    public let sampleRate: Int
    public let runtimeConfig: IndexTTS2RuntimeConfig?
    public let tokenizer: IndexTTS2Tokenizer?

    private let bundleInfo: IndexTTS2BundleInfo
    private var nativeRuntime: IndexTTS2NativeRuntime?
    private var loaded = true

    private init(bundleInfo: IndexTTS2BundleInfo) {
        self.bundleInfo = bundleInfo
        self.bundleDirectory = bundleInfo.directory
        self.manifest = bundleInfo.manifest
        self.runtimeConfig = try? IndexTTS2RuntimeConfig.load(from: bundleInfo.directory)
        self.sampleRate = runtimeConfig?.outputSampleRate ?? bundleInfo.sampleRate
        self.tokenizer = try? IndexTTS2Tokenizer(
            modelURL: bundleInfo.directory.appendingPathComponent("bpe.model"))
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> IndexTTS2TTSModel {
        let directory = try await IndexTTS2BundleLoader.download(
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
        let info = try IndexTTS2BundleLoader.load(
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
        nativeRuntime = nil
        loaded = false
    }

    @discardableResult
    public func prepareRuntime(
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2RuntimeInventory {
        try ensureLoaded()
        return try runtime(progressHandler: progressHandler).inventory
    }

    public func prepareReferenceConditioning(
        referenceAudio: URL,
        emotionReferenceAudio: URL? = nil,
        emotionControl: IndexTTS2EmotionControl? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2ReferenceConditioning {
        try ensureLoaded()
        return try runtime { progress, message in
            progressHandler?(progress * 0.65, message)
        }
        .prepareReferenceConditioning(
            referenceAudio: referenceAudio,
            emotionReferenceAudio: emotionReferenceAudio,
            emotionControl: emotionControl) { progress, message in
                progressHandler?(0.65 + progress * 0.35, message)
            }
    }

    public func generateSemanticCodes(
        text: String,
        referenceAudio: URL,
        emotionReferenceAudio: URL? = nil,
        emotionControl: IndexTTS2EmotionControl? = nil,
        options: IndexTTS2SemanticGenerationOptions = IndexTTS2SemanticGenerationOptions(),
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2SemanticGeneration {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 semantic generation",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        let textTokens = try tokenizer.encode(text)
        let runtime = try runtime { progress, message in
            progressHandler?(progress * 0.45, message)
        }
        let conditioning = try runtime.prepareReferenceConditioning(
            referenceAudio: referenceAudio,
            emotionReferenceAudio: emotionReferenceAudio,
            emotionControl: emotionControl) { progress, message in
                progressHandler?(0.45 + progress * 0.35, message)
            }
        progressHandler?(0.82, "Generating IndexTTS2 semantic tokens")
        let generated = try runtime.semanticGPT.generateSemanticCodes(
            textTokens: textTokens,
            conditioning: conditioning,
            options: options)
        progressHandler?(1.0, "IndexTTS2 semantic tokens ready")
        return generated
    }

    public func generateSemanticCodes(
        text: String,
        conditioning: IndexTTS2ReferenceConditioning,
        options: IndexTTS2SemanticGenerationOptions = IndexTTS2SemanticGenerationOptions()
    ) throws -> IndexTTS2SemanticGeneration {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 semantic generation",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        return try runtime().semanticGPT.generateSemanticCodes(
            textTokens: try tokenizer.encode(text),
            conditioning: conditioning,
            options: options)
    }

    public func synthesize(
        text: String,
        conditioning: IndexTTS2ReferenceConditioning,
        semanticOptions: IndexTTS2SemanticGenerationOptions = IndexTTS2SemanticGenerationOptions(),
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        return try runtime().synthesize(
            textTokens: try tokenizer.encode(text),
            conditioning: conditioning,
            semanticOptions: semanticOptions,
            synthesisOptions: synthesisOptions)
    }

    public func synthesize(
        text: String,
        conditioning: IndexTTS2ReferenceConditioning,
        semantic: IndexTTS2SemanticGeneration,
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        return try runtime().synthesize(
            textTokens: try tokenizer.encode(text),
            conditioning: conditioning,
            semantic: semantic,
            synthesisOptions: synthesisOptions)
    }

    public func synthesize(
        text: String,
        conditioning: IndexTTS2ReferenceConditioning,
        semanticCodes: [Int32],
        synthesisOptions: IndexTTS2SynthesisOptions = .default
    ) throws -> [Float] {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        return try runtime().synthesize(
            textTokens: try tokenizer.encode(text),
            semanticCodes: semanticCodes,
            conditioning: conditioning,
            synthesisOptions: synthesisOptions)
    }

    private func runtime(
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> IndexTTS2NativeRuntime {
        if let nativeRuntime {
            progressHandler?(1.0, "IndexTTS2 runtime weights ready")
            return nativeRuntime
        }

        let runtime = try IndexTTS2NativeRuntime.load(
            from: bundleDirectory,
            config: runtimeConfig ?? .fallback,
            progressHandler: progressHandler)
        nativeRuntime = runtime
        return runtime
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try ensureLoaded()
        throw AudioModelError.inferenceFailed(
            operation: "IndexTTS2 synthesis",
            reason: "IndexTTS2 requires a reference audio sample. Use generate(text:referenceAudio:referenceText:emotionReferenceAudio:language:).")
    }

    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String? = nil,
        emotionReferenceAudio: URL? = nil,
        emotionControl: IndexTTS2EmotionControl? = nil,
        synthesisOptions: IndexTTS2SynthesisOptions = .default,
        language: String? = nil
    ) async throws -> [Float] {
        try ensureLoaded()
        guard let tokenizer else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Tokenizer could not be initialized from bpe.model.")
        }
        let textTokens = try tokenizer.encode(text)
        var stageStart = CFAbsoluteTimeGetCurrent()
        let runtime = try runtime()
        IndexTTS2StageTimer.report("runtime-load", since: &stageStart)
        let conditioning = try runtime.prepareReferenceConditioning(
            referenceAudio: referenceAudio,
            emotionReferenceAudio: emotionReferenceAudio,
            emotionControl: emotionControl)
        IndexTTS2StageTimer.report("reference-conditioning", since: &stageStart)
        return try runtime.synthesize(
            textTokens: textTokens,
            conditioning: conditioning,
            synthesisOptions: synthesisOptions)
    }

    private func ensureLoaded() throws {
        guard loaded else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 synthesis",
                reason: "Model has been unloaded.")
        }
    }
}
