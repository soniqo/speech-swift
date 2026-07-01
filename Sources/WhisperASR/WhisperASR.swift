import AudioCommon
import Foundation
import WhisperKit

/// Whisper large-v3 turbo ASR via WhisperKit/CoreML.
///
/// The default model repo is staged with the CoreML bundle at the repository
/// root, so this wrapper downloads the root files with `HuggingFaceDownloader`
/// and passes the local folder directly to WhisperKit.
public final class WhisperASRModel: SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Whisper-Large-v3-Turbo-CoreML"
    public static let defaultModelVariant = "large-v3-v20240930_turbo"

    public let inputSampleRate = WhisperKit.sampleRate
    public let modelId: String
    public let modelFolder: URL

    private let pipeline: WhisperKit

    private init(modelId: String, modelFolder: URL, pipeline: WhisperKit) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.pipeline = pipeline
    }

    /// Load Whisper large-v3 turbo from the published CoreML bundle.
    public static func fromPretrained(
        modelId: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> WhisperASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        let resolvedCacheDir: URL
        do {
            resolvedCacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Failed to resolve cache directory",
                underlying: error)
        }

        progressHandler?(0.0, "Downloading Whisper CoreML bundle...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: resolvedCacheDir,
                additionalFiles: [
                    "MelSpectrogram.mlmodelc/**",
                    "AudioEncoder.mlmodelc/**",
                    "TextDecoder.mlmodelc/**",
                    "TextDecoderContextPrefill.mlmodelc/**",
                    "generation_config.json",
                    "manifest.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "normalizer.json",
                    "preprocessor_config.json",
                ],
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(fraction * 0.8, "Downloading Whisper CoreML bundle...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Download failed",
                underlying: error)
        }

        progressHandler?(0.80, "Loading WhisperKit...")
        let config = WhisperKitConfig(
            modelFolder: resolvedCacheDir.path,
            verbose: false,
            logLevel: .error,
            prewarm: false,
            load: true,
            download: false
        )
        let pipeline = try await WhisperKit(config)

        progressHandler?(1.0, "Model loaded")
        return WhisperASRModel(modelId: effectiveModelId, modelFolder: resolvedCacheDir, pipeline: pipeline)
    }

    /// Transcribe PCM Float32 audio. Input is resampled to 16 kHz when needed.
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) async throws -> String {
        let normalizedAudio: [Float]
        if sampleRate == inputSampleRate {
            normalizedAudio = audio
        } else {
            normalizedAudio = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        let languageHint = language?.isEmpty == true ? nil : language
        let options = DecodingOptions(task: .transcribe, language: languageHint, temperatureFallbackCount: 0)
        let results = try await pipeline.transcribe(audioArray: normalizedAudio, decodeOptions: options)
        return results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Transcribe and return WhisperKit's detected language when available.
    public func transcribeWithLanguageAsync(
        audio: [Float],
        sampleRate: Int,
        language: String? = nil
    ) async throws -> AudioCommon.TranscriptionResult {
        let normalizedAudio: [Float]
        if sampleRate == inputSampleRate {
            normalizedAudio = audio
        } else {
            normalizedAudio = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        let languageHint = language?.isEmpty == true ? nil : language
        let options = DecodingOptions(task: .transcribe, language: languageHint, temperatureFallbackCount: 0)
        let results = try await pipeline.transcribe(audioArray: normalizedAudio, decodeOptions: options)
        let text = results.map(\.text).joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines)
        let detectedLanguage = results.first?.language
        return AudioCommon.TranscriptionResult(text: text, language: detectedLanguage, confidence: 0)
    }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try runBlocking {
                try await self.transcribeAudio(audio, sampleRate: sampleRate, language: language)
            }
        } catch {
            AudioLog.inference.error("Whisper transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> AudioCommon.TranscriptionResult {
        do {
            return try runBlocking {
                try await self.transcribeWithLanguageAsync(audio: audio, sampleRate: sampleRate, language: language)
            }
        } catch {
            AudioLog.inference.error("Whisper transcription failed: \(error)")
            return AudioCommon.TranscriptionResult(text: "")
        }
    }
}

private func runBlocking<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?

    Task {
        do {
            result = .success(try await operation())
        } catch {
            result = .failure(error)
        }
        semaphore.signal()
    }

    semaphore.wait()
    return try result!.get()
}
