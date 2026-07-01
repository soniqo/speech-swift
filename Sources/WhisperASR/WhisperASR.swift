import AudioCommon
import Foundation

/// Whisper large-v3 turbo ASR through speech-swift's native CoreML runtime.
public final class WhisperASRModel: SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Whisper-Large-v3-Turbo-CoreML"
    public static let defaultModelVariant = "large-v3-v20240930_turbo"
    public static let tokenizerModelId = "openai/whisper-large-v3"

    public let inputSampleRate = WhisperCoreMLRuntime.sampleRate
    public let modelId: String
    public let modelFolder: URL

    private let runtime: WhisperCoreMLRuntime

    private init(modelId: String, modelFolder: URL, runtime: WhisperCoreMLRuntime) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.runtime = runtime
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
                progressHandler?(fraction * 0.75, "Downloading Whisper CoreML bundle...")
            }

            try await ensureTokenizer(
                in: resolvedCacheDir,
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(0.75 + fraction * 0.10, "Downloading Whisper tokenizer...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Download failed",
                underlying: error)
        }

        progressHandler?(0.90, "Loading Whisper CoreML runtime...")
        do {
            let runtime = try WhisperCoreMLRuntime(modelFolder: resolvedCacheDir)
            progressHandler?(1.0, "Model loaded")
            return WhisperASRModel(modelId: effectiveModelId, modelFolder: resolvedCacheDir, runtime: runtime)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId,
                reason: "Failed to initialize native Whisper runtime",
                underlying: error)
        }
    }

    /// Transcribe PCM Float32 audio. Input is resampled to 16 kHz when needed.
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) async throws -> String {
        let normalizedAudio: [Float]
        if sampleRate == inputSampleRate {
            normalizedAudio = audio
        } else {
            normalizedAudio = AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        }

        return try runtime.transcribe(audio: normalizedAudio, languageHint: normalizedLanguage(language)).text
    }

    /// Transcribe and return the detected or hinted Whisper language code when available.
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

        let result = try runtime.transcribe(audio: normalizedAudio, languageHint: normalizedLanguage(language))
        return AudioCommon.TranscriptionResult(text: result.text, language: result.language, confidence: 0)
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

    private static func ensureTokenizer(
        in modelFolder: URL,
        offlineMode: Bool,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        let files = ["tokenizer.json", "tokenizer_config.json"]
        let missingFiles = files.filter {
            !FileManager.default.fileExists(atPath: modelFolder.appendingPathComponent($0).path)
        }
        if missingFiles.isEmpty {
            progressHandler?(1.0)
            return
        }
        if offlineMode {
            throw AudioModelError.modelLoadFailed(
                modelId: tokenizerModelId,
                reason: "Offline tokenizer cache miss: \(missingFiles.joined(separator: ", "))")
        }

        try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)
        for (index, file) in missingFiles.enumerated() {
            let escaped = file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file
            let endpoint = (HuggingFaceDownloader.resolvedEndpoint() ?? "https://huggingface.co")
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            guard let url = URL(string: "\(endpoint)/\(tokenizerModelId)/resolve/main/\(escaped)") else {
                throw AudioModelError.modelLoadFailed(modelId: tokenizerModelId, reason: "Invalid tokenizer URL for \(file)")
            }

            var request = URLRequest(url: url)
            let env = ProcessInfo.processInfo.environment
            if let token = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"],
               !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            }

            let (data, response) = try await URLSession.shared.data(for: request)
            if let http = response as? HTTPURLResponse,
               !(200..<300).contains(http.statusCode) {
                throw AudioModelError.modelLoadFailed(
                    modelId: tokenizerModelId,
                    reason: "\(file): HTTP \(http.statusCode)")
            }
            try data.write(to: modelFolder.appendingPathComponent(file), options: .atomic)
            progressHandler?(Double(index + 1) / Double(missingFiles.count))
        }
    }

    private func normalizedLanguage(_ language: String?) -> String? {
        guard let language else { return nil }
        let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return trimmed
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
