import AudioCommon
import Foundation
import MLX
import MLXCommon

public final class F5TTSModel: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/F5TTS-v1-Base-MLX-fp16"
    public static let modelKey = "f5-tts"

    private static let requiredFiles = [
        "README.md",
        "config.json",
        "model.safetensors",
        "vocos.safetensors",
        "vocos_config.yaml",
        "vocab.txt",
    ]

    public let bundleDirectory: URL
    public let config: F5TTSConfig
    public let sampleRate: Int
    public let tokenizer: F5TTSTokenizer

    private let bundleInfo: F5TTSBundleInfo
    private var flow: F5TTSFlow?
    private var vocoder: F5TTSVocos?
    private var loaded = true

    private init(bundleInfo: F5TTSBundleInfo, tokenizer: F5TTSTokenizer) {
        self.bundleInfo = bundleInfo
        self.bundleDirectory = bundleInfo.directory
        self.config = bundleInfo.config
        self.sampleRate = bundleInfo.sampleRate
        self.tokenizer = tokenizer
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> F5TTSModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(
            modelId: modelId,
            to: directory,
            files: requiredFiles,
            offlineMode: offlineMode
        ) { progress in
            progressHandler?(progress * 0.85, "Downloading F5-TTS")
        }
        return try await fromBundle(directory) { progress, message in
            progressHandler?(0.85 + progress * 0.15, message)
        }
    }

    public static func fromBundle(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> F5TTSModel {
        progressHandler?(0.05, "Loading F5-TTS bundle")
        let info = try F5TTSBundleLoader.load(from: directory)
        let tokenizer = try F5TTSTokenizer(
            vocabURL: directory.appendingPathComponent(info.config.files.vocab))
        progressHandler?(1.0, "F5-TTS bundle ready")
        return F5TTSModel(bundleInfo: info, tokenizer: tokenizer)
    }

    public var isLoaded: Bool { loaded }

    public var memoryFootprint: Int {
        loaded ? bundleInfo.weightMemory : 0
    }

    public func unload() {
        flow = nil
        vocoder = nil
        loaded = false
    }

    public func prepareRuntime(progressHandler: ((Double, String) -> Void)? = nil) throws {
        try ensureLoaded()
        _ = try runtime(progressHandler: progressHandler)
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try ensureLoaded()
        throw F5TTSError.unsupportedText(
            "F5-TTS requires reference audio and a reference transcript. Use generate(text:referenceAudio:referenceText:options:).")
    }

    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String,
        options: F5TTSSynthesisOptions = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> [Float] {
        let samples = try AudioFileLoader.load(
            url: referenceAudio,
            targetSampleRate: sampleRate)
        return try generate(
            text: text,
            referenceAudio: samples,
            referenceSampleRate: sampleRate,
            referenceText: referenceText,
            options: options,
            progressHandler: progressHandler)
    }

    public func generate(
        text: String,
        referenceAudio samples: [Float],
        referenceSampleRate: Int,
        referenceText: String,
        options: F5TTSSynthesisOptions = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> [Float] {
        try ensureLoaded()
        let resampled: [Float]
        if referenceSampleRate == sampleRate {
            resampled = samples
        } else {
            resampled = AudioFileLoader.resample(samples, from: referenceSampleRate, to: sampleRate)
        }

        let (flow, vocoder) = try runtime(progressHandler: progressHandler)
        progressHandler?(0.72, "Preparing F5-TTS reference mel")
        let reference = flow.prepareReference(samples: resampled, options: options)
        progressHandler?(0.76, "Sampling F5-TTS mel")
        let mel = try flow.sampleMel(
            reference: reference,
            referenceText: referenceText,
            targetText: text,
            tokenizer: tokenizer,
            options: options)
        progressHandler?(0.94, "Decoding F5-TTS waveform")
        var wav = vocoder.decode(mel: mel).asType(.float32)
        if reference.rms > 0, reference.rms < options.targetRMS {
            wav = wav * MLXArray(reference.rms / options.targetRMS).asType(wav.dtype)
        }
        eval(wav)
        progressHandler?(1.0, "F5-TTS synthesis complete")
        return wav.asArray(Float.self)
    }

    private func runtime(
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> (F5TTSFlow, F5TTSVocos) {
        if let flow, let vocoder {
            progressHandler?(1.0, "F5-TTS runtime ready")
            return (flow, vocoder)
        }

        progressHandler?(0.10, "Loading F5-TTS DiT weights")
        let modelWeights = try CommonWeightLoader.loadSafetensors(
            url: bundleDirectory.appendingPathComponent(config.files.model))
        try F5TTSFlow.validate(modelWeights, config: config)
        let flow = F5TTSFlow(weights: modelWeights, config: config)

        progressHandler?(0.48, "Loading F5-TTS Vocos weights")
        let vocoderWeights = try CommonWeightLoader.loadSafetensors(
            url: bundleDirectory.appendingPathComponent(config.files.vocoder))
        try F5TTSVocos.validate(vocoderWeights)
        let vocoder = F5TTSVocos(weights: vocoderWeights)

        self.flow = flow
        self.vocoder = vocoder
        progressHandler?(0.70, "F5-TTS runtime weights ready")
        return (flow, vocoder)
    }

    private func ensureLoaded() throws {
        if !loaded {
            throw F5TTSError.unloaded
        }
    }
}
