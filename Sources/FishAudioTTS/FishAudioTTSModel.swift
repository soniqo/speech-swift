import AudioCommon
import Foundation
import MLX
import MLXCommon

public final class FishAudioTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public static let defaultModelId = FishAudioConfig.defaultModelId

    public let sampleRate: Int
    public let bundleDirectory: URL
    public let config: FishAudioConfig

    private let tokenizer: FishAudioTokenizer
    private let languageModel: FishAudioDualARModel
    private let codec: FishAudioCodec

    private init(
        bundleDirectory: URL,
        config: FishAudioConfig,
        tokenizer: FishAudioTokenizer,
        languageModel: FishAudioDualARModel,
        codec: FishAudioCodec
    ) {
        self.bundleDirectory = bundleDirectory
        self.config = config
        self.sampleRate = FishAudioCodecDefaults.sampleRate
        self.tokenizer = tokenizer
        self.languageModel = languageModel
        self.codec = codec
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> FishAudioTTSModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: [
                "LICENSE.md",
                "README.md",
                "chat_template.jinja",
                "codec.safetensors",
                "config.json",
                "model.safetensors.index.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "soniqo_manifest.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.35, "Downloading Fish Audio S2 Pro") }
        )
        return try await fromBundle(directory) { progress, message in
            progressHandler?(0.35 + progress * 0.65, message)
        }
    }

    public static func fromBundle(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> FishAudioTTSModel {
        progressHandler?(0.05, "Loading Fish Audio config")
        let config = try FishAudioConfig.load(from: directory.appendingPathComponent("config.json"))

        progressHandler?(0.10, "Loading Fish Audio tokenizer")
        let tokenizer = try await FishAudioTokenizer.load(from: directory)

        progressHandler?(0.18, "Loading Fish Audio language model")
        let languageModel = try FishAudioDualARModel.load(from: directory) { progress, message in
            progressHandler?(0.18 + progress * 0.62, message)
        }

        progressHandler?(0.82, "Loading Fish Audio codec")
        let codec = try FishAudioCodec.load(from: directory)

        progressHandler?(1.0, "Fish Audio runtime ready")
        return FishAudioTTSModel(
            bundleDirectory: directory,
            config: config,
            tokenizer: tokenizer,
            languageModel: languageModel,
            codec: codec)
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try await generate(text: text, references: [], sampling: .default)
    }

    public func generate(
        text: String,
        references: [FishAudioReferencePrompt] = [],
        sampling: FishAudioSamplingConfig = .default
    ) async throws -> [Float] {
        let codebooks = try generateCodebooks(
            text: text,
            references: references,
            sampling: sampling)
        return try decode(codebooks)
    }

    public func generate(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int,
        referenceText: String,
        sampling: FishAudioSamplingConfig = .default
    ) async throws -> [Float] {
        let reference = try encodeReferencePrompt(
            audio: referenceAudio,
            sampleRate: referenceSampleRate,
            text: referenceText)
        return try await generate(text: text, references: [reference], sampling: sampling)
    }

    public func generate(
        text: String,
        referenceAudioURL: URL,
        referenceText: String,
        sampling: FishAudioSamplingConfig = .default
    ) async throws -> [Float] {
        let samples = try AudioFileLoader.load(
            url: referenceAudioURL,
            targetSampleRate: FishAudioCodecDefaults.sampleRate)
        return try await generate(
            text: text,
            referenceAudio: samples,
            referenceSampleRate: FishAudioCodecDefaults.sampleRate,
            referenceText: referenceText,
            sampling: sampling)
    }

    public func generateCodebooks(
        text: String,
        references: [FishAudioReferencePrompt] = [],
        sampling: FishAudioSamplingConfig = .default
    ) throws -> FishAudioGeneratedCodebooks {
        let input = try FishAudioInputBuilder.build(
            text: text,
            references: references,
            tokenizer: tokenizer,
            config: config)
        return try languageModel.generateCodebooks(from: input, sampling: sampling)
    }

    public func encodeReferencePrompt(
        audio samples: [Float],
        sampleRate: Int,
        text: String
    ) throws -> FishAudioReferencePrompt {
        let codes = try codec.encode(audio: samples, sampleRate: sampleRate)
        return try FishAudioReferencePrompt(text: text, codes: codes.codes)
    }

    public func decode(_ codebooks: FishAudioGeneratedCodebooks) throws -> [Float] {
        try codec.decode(codebooks)
    }
}
