import AudioCommon
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

/// Long-context MLX runtime for MOSS-Transcribe-Diarize 0.9B.
///
/// The Whisper encoder processes non-overlapping 30-second chunks in bounded
/// batches. All adapted audio embeddings then enter one Qwen3 prompt, so
/// transcription and speaker attribution retain global recording context.
/// This remains an offline autoregressive model; it does not emit streaming
/// partials while audio is being recorded.
public final class MossMLXModel:
    SpeechRecognitionModel,
    @unchecked Sendable
{
    public static let defaultModelId = MossMLXVariant.int5.modelId
    public static let defaultInstruction =
        MossPromptProcessor.defaultInstruction
    public static let inputSampleRate =
        MossWhisperFeatureExtractor.sampleRate
    public static let maximumContextTokens =
        MossMLXConfiguration.maximumContextTokens

    static let requiredBundleFiles = [
        "config.json",
        "audio_encoder.safetensors",
        "decoder.safetensors",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    static let downloadAdditionalFiles = [
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "source_config.json",
        "export_config.json",
        "validation.json",
    ]

    public let modelId: String
    public let modelFolder: URL
    public var inputSampleRate: Int { Self.inputSampleRate }
    public var quantizationBits: Int {
        configuration.quantization.bits
    }

    private let configuration: MossMLXConfiguration
    private let audioModel: MossMLXAudioModel
    private let textModel: MossMLXTextModel
    private let tokenizer: any Tokenizers.Tokenizer
    private let promptProcessor: MossPromptProcessor
    private let featureExtractor = MossWhisperFeatureExtractor()
    private let inferenceLock = NSLock()

    private init(
        modelId: String,
        modelFolder: URL,
        configuration: MossMLXConfiguration,
        audioModel: MossMLXAudioModel,
        textModel: MossMLXTextModel,
        tokenizer: any Tokenizers.Tokenizer,
        promptProcessor: MossPromptProcessor
    ) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.configuration = configuration
        self.audioModel = audioModel
        self.textModel = textModel
        self.tokenizer = tokenizer
        self.promptProcessor = promptProcessor
    }

    public static func fromPretrained(
        variant: MossMLXVariant = .int5,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossMLXModel {
        try await fromPretrained(
            modelId: variant.modelId,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            progressHandler: progressHandler
        )
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossMLXModel {
        let directory: URL
        do {
            directory = try cacheDir
                ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to resolve the MOSS MLX cache directory",
                underlying: error
            )
        }

        progressHandler?(0, "Preparing MOSS MLX bundle...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: directory,
                additionalFiles: downloadAdditionalFiles,
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(
                    fraction * 0.75,
                    "Downloading MOSS MLX bundle..."
                )
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to download the MOSS MLX bundle",
                underlying: error
            )
        }
        return try await fromDirectory(
            directory,
            modelId: modelId
        ) { fraction, message in
            progressHandler?(0.75 + fraction * 0.25, message)
        }
    }

    /// Load an exported MLX bundle without accessing the network.
    public static func fromDirectory(
        _ directory: URL,
        modelId: String = "local/MOSS-Transcribe-Diarize-MLX",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossMLXModel {
        guard hasCompleteCachedBundle(in: directory) else {
            let missing = requiredBundleFiles.first {
                !FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent($0).path
                )
            } ?? "unknown file"
            throw MossTranscribeError.missingModelFile(missing)
        }

        let configuration = try JSONDecoder().decode(
            MossMLXConfiguration.self,
            from: Data(
                contentsOf: directory.appendingPathComponent(
                    "config.json"
                )
            )
        )
        try configuration.validate()
        for file in [
            configuration.files.audioEncoder,
            configuration.files.decoder,
        ] where !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(file).path
        ) {
            throw MossTranscribeError.missingModelFile(file)
        }

        let preprocessor = try JSONDecoder().decode(
            MossPreprocessorConfiguration.self,
            from: Data(
                contentsOf: directory.appendingPathComponent(
                    "preprocessor_config.json"
                )
            )
        )
        try preprocessor.validate()
        let processor = try JSONDecoder().decode(
            MossProcessorConfiguration.self,
            from: Data(
                contentsOf: directory.appendingPathComponent(
                    "processor_config.json"
                )
            )
        )
        try validateProcessorConfiguration(processor)

        progressHandler?(0.05, "Loading MOSS tokenizer...")
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: directory
        )
        let promptProcessor = try MossPromptProcessor(
            tokenizer: tokenizer,
            configuration: processor
        )
        guard promptProcessor.audioTokenID == configuration.audioTokenID else {
            throw MossTranscribeError.invalidConfiguration(
                "config audio_token_id does not match the tokenizer"
            )
        }

        progressHandler?(0.1, "Loading MOSS MLX audio encoder...")
        let audioModel = MossMLXAudioModel(configuration)
        do {
            let audioWeights = try MLX.loadArrays(
                url: directory.appendingPathComponent(
                    configuration.files.audioEncoder
                )
            )
            try audioModel.update(
                parameters: ModuleParameters.unflattened(audioWeights),
                verify: .all
            )
            audioModel.train(false)
            eval(audioModel)
        } catch {
            throw MossTranscribeError.invalidConfiguration(
                "incompatible MLX audio weights: "
                    + error.localizedDescription
            )
        }

        progressHandler?(0.45, "Loading quantized MOSS decoder...")
        let textModel = MossMLXTextModel(configuration.decoder)
        do {
            let decoderWeights = try MLX.loadArrays(
                url: directory.appendingPathComponent(
                    configuration.files.decoder
                )
            )
            MLXNN.quantize(model: textModel) { path, _ in
                decoderWeights["\(path).scales"] == nil
                    ? nil
                    : (
                        configuration.quantization.groupSize,
                        configuration.quantization.bits,
                        .affine
                    )
            }
            try textModel.update(
                parameters: ModuleParameters.unflattened(decoderWeights),
                verify: .all
            )
            textModel.train(false)
            eval(textModel)
        } catch {
            throw MossTranscribeError.invalidConfiguration(
                "incompatible quantized MLX decoder weights: "
                    + error.localizedDescription
            )
        }
        Memory.clearCache()
        progressHandler?(1, "MOSS MLX ready")

        return MossMLXModel(
            modelId: modelId,
            modelFolder: directory,
            configuration: configuration,
            audioModel: audioModel,
            textModel: textModel,
            tokenizer: tokenizer,
            promptProcessor: promptProcessor
        )
    }

    /// Compile representative encoder, prefill, and decode paths.
    public func warmUp() throws {
        var options = MossMLXDecodingOptions(maxTokens: 1)
        options.encoderBatchSize = 1
        _ = try transcribeDetailed(
            audio: [Float](repeating: 0, count: 1_600),
            sampleRate: Self.inputSampleRate,
            options: options
        )
    }

    public func transcribeDetailed(
        audio: [Float],
        sampleRate: Int,
        options: MossMLXDecodingOptions = MossMLXDecodingOptions(),
        instruction: String = MossMLXModel.defaultInstruction
    ) throws -> MossTranscription {
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        return try autoreleasepool {
            defer { Memory.clearCache() }
            return try transcribeLocked(
                audio: audio,
                sampleRate: sampleRate,
                options: options,
                instruction: instruction
            )
        }
    }

    public func transcribeDetailed(
        audio: [Float],
        sampleRate: Int,
        maxNewTokens: Int,
        instruction: String = MossMLXModel.defaultInstruction
    ) throws -> MossTranscription {
        try transcribeDetailed(
            audio: audio,
            sampleRate: sampleRate,
            options: MossMLXDecodingOptions(maxTokens: maxNewTokens),
            instruction: instruction
        )
    }

    public func transcribe(
        audio: [Float],
        sampleRate: Int,
        language: String?
    ) -> String {
        do {
            return try transcribeDetailed(
                audio: audio,
                sampleRate: sampleRate
            ).text
        } catch {
            AudioLog.inference.error(
                "MOSS MLX transcription failed: \(error.localizedDescription)"
            )
            return ""
        }
    }

    static func hasCompleteCachedBundle(in directory: URL) -> Bool {
        requiredBundleFiles.allSatisfy {
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent($0).path
            )
        }
    }

    private static func validateProcessorConfiguration(
        _ processor: MossProcessorConfiguration
    ) throws {
        let expectedTokenRate =
            Double(MossWhisperFeatureExtractor.sampleRate)
            / Double(
                MossWhisperFeatureExtractor.encoderStrideSamples
            )
        guard
            abs(
                processor.audioTokensPerSecond - expectedTokenRate
            ) < 1e-9,
            processor.audioMergeSize == 4,
            processor.timeMarkerEverySeconds > 0
        else {
            throw MossTranscribeError.invalidConfiguration(
                "processor settings do not match the native MOSS runtime"
            )
        }
    }

    private func transcribeLocked(
        audio: [Float],
        sampleRate: Int,
        options: MossMLXDecodingOptions,
        instruction: String
    ) throws -> MossTranscription {
        guard !audio.isEmpty else {
            throw MossTranscribeError.invalidAudio(
                "the waveform is empty"
            )
        }
        guard sampleRate > 0 else {
            throw MossTranscribeError.invalidAudio(
                "sample rate must be positive"
            )
        }
        guard
            options.maxTokens > 0,
            options.encoderBatchSize > 0,
            options.prefillChunkSize > 0
        else {
            throw MossTranscribeError.invalidConfiguration(
                "maxTokens, encoderBatchSize, and prefillChunkSize "
                    + "must be positive"
            )
        }

        let totalStarted = CFAbsoluteTimeGetCurrent()
        var preprocessingSeconds = 0.0
        var audioEncoderSeconds = 0.0

        var phaseStarted = CFAbsoluteTimeGetCurrent()
        let waveform =
            sampleRate == Self.inputSampleRate
            ? audio
            : AudioFileLoader.resample(
                audio,
                from: sampleRate,
                to: Self.inputSampleRate
            )
        preprocessingSeconds +=
            CFAbsoluteTimeGetCurrent() - phaseStarted
        guard !waveform.isEmpty else {
            throw MossTranscribeError.invalidAudio(
                "resampling produced an empty waveform"
            )
        }

        var featureChunks: [MossLogMelFeatures] = []
        var tokenCounts: [Int] = []
        for start in stride(
            from: 0,
            to: waveform.count,
            by: MossWhisperFeatureExtractor.chunkSamples
        ) {
            let end = min(
                start + MossWhisperFeatureExtractor.chunkSamples,
                waveform.count
            )
            let chunk = Array(waveform[start..<end])
            phaseStarted = CFAbsoluteTimeGetCurrent()
            featureChunks.append(
                try featureExtractor.extractPaddedChunk(chunk)
            )
            preprocessingSeconds +=
                CFAbsoluteTimeGetCurrent() - phaseStarted
            tokenCounts.append(
                MossWhisperFeatureExtractor.audioTokenCount(
                    sampleCount: chunk.count
                )
            )
        }

        var embeddingBatches: [MLXArray] = []
        embeddingBatches.reserveCapacity(
            (featureChunks.count + options.encoderBatchSize - 1)
                / options.encoderBatchSize
        )
        for start in stride(
            from: 0,
            to: featureChunks.count,
            by: options.encoderBatchSize
        ) {
            let end = min(
                start + options.encoderBatchSize,
                featureChunks.count
            )
            let features = featureChunks[start..<end]
            let values = features.flatMap(\.data)
            let featureArray = MLXArray(
                values,
                [
                    features.count,
                    MossWhisperFeatureExtractor.melBins,
                    MossWhisperFeatureExtractor.timeFrames,
                ]
            ).transposed(0, 2, 1)

            phaseStarted = CFAbsoluteTimeGetCurrent()
            let embeddings = try audioModel.encode(
                features: featureArray,
                tokenCounts: Array(tokenCounts[start..<end])
            )
            eval(embeddings)
            embeddingBatches.append(embeddings)
            audioEncoderSeconds +=
                CFAbsoluteTimeGetCurrent() - phaseStarted
            Memory.clearCache()
        }
        let audioEmbeddings = concatenated(
            embeddingBatches,
            axis: 0
        )
        eval(audioEmbeddings)
        let audioTokenCount = tokenCounts.reduce(0, +)

        phaseStarted = CFAbsoluteTimeGetCurrent()
        let prompt = try promptProcessor.prepare(
            audioTokenCount: audioTokenCount,
            instruction: instruction
        )
        preprocessingSeconds +=
            CFAbsoluteTimeGetCurrent() - phaseStarted
        guard
            prompt.inputIDs.count
                < Self.maximumContextTokens
        else {
            throw MossTranscribeError.promptTooLong(
                actual: prompt.inputIDs.count,
                maximum: Self.maximumContextTokens
            )
        }
        guard
            prompt.audioPlaceholderCount == audioTokenCount,
            audioEmbeddings.dim(0) == audioTokenCount
        else {
            throw MossTranscribeError.audioEmbeddingMismatch(
                placeholders: prompt.audioPlaceholderCount,
                embeddings: audioEmbeddings.dim(0)
            )
        }

        let prefillStarted = CFAbsoluteTimeGetCurrent()
        let cache = textModel.makeCache(
            precision: options.kvCachePrecision,
            step: options.prefillChunkSize
        )
        var audioIndex = 0
        var finalHidden: MLXArray?
        for start in stride(
            from: 0,
            to: prompt.inputIDs.count,
            by: options.prefillChunkSize
        ) {
            let end = min(
                start + options.prefillChunkSize,
                prompt.inputIDs.count
            )
            let ids = Array(prompt.inputIDs[start..<end])
            let tokenRows = textModel.embed(
                MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
            )[0]
            var audioIndices = [Int32](
                repeating: 0,
                count: ids.count
            )
            var audioMask = [Float](
                repeating: 0,
                count: ids.count
            )
            for index in ids.indices
            where ids[index] == promptProcessor.audioTokenID {
                audioIndices[index] = Int32(audioIndex)
                audioMask[index] = 1
                audioIndex += 1
            }
            let replacementRows = audioEmbeddings[
                MLXArray(audioIndices)
            ].asType(tokenRows.dtype)
            let mask = MLXArray(audioMask)
                .expandedDimensions(axis: 1)
                .asType(tokenRows.dtype)
            let mixedRows =
                tokenRows * (1 - mask) + replacementRows * mask
            finalHidden = textModel.forward(
                embeddings: mixedRows.expandedDimensions(axis: 0),
                cache: cache
            )
            asyncEval(cache)
        }
        eval(cache)
        guard
            audioIndex == audioTokenCount,
            let finalHidden
        else {
            throw MossTranscribeError.audioEmbeddingMismatch(
                placeholders: audioIndex,
                embeddings: audioTokenCount
            )
        }
        var logits = textModel.logits(finalHidden[0, -1])
        eval(logits)
        let decoderPrefillSeconds =
            CFAbsoluteTimeGetCurrent() - prefillStarted

        let decodeStarted = CFAbsoluteTimeGetCurrent()
        let availableContext =
            Self.maximumContextTokens - prompt.inputIDs.count
        let generationLimit = min(
            options.maxTokens,
            availableContext
        )
        var generated: [Int] = []
        generated.reserveCapacity(generationLimit)
        var stopReason: MossGenerationStopReason =
            options.maxTokens > availableContext
            ? .contextLimit
            : .maximumTokens

        while generated.count < generationLimit {
            let candidate = logits.argMax(axis: -1).item(Int.self)
            generated.append(candidate)
            if candidate == prompt.eosTokenID {
                stopReason = .endOfSequence
                break
            }
            if generated.count == generationLimit {
                break
            }
            let embedding = textModel.embed(
                MLXArray([Int32(candidate)]).reshaped(1, 1)
            )
            let hidden = textModel.forward(
                embeddings: embedding,
                cache: cache
            )
            logits = textModel.logits(hidden[0, -1])
            eval(logits)
        }
        let tokenDecodeSeconds =
            CFAbsoluteTimeGetCurrent() - decodeStarted

        let raw = tokenizer.decode(
            tokens: generated,
            skipSpecialTokens: true
        ).trimmingCharacters(
            in: CharacterSet.whitespacesAndNewlines
        )
        let parsed = MossTranscriptParser.plainText(from: raw)
        let metrics = MossTranscriptionMetrics(
            preprocessingSeconds: preprocessingSeconds,
            audioEncoderSeconds: audioEncoderSeconds,
            decoderPrefillSeconds: decoderPrefillSeconds,
            tokenDecodeSeconds: tokenDecodeSeconds,
            totalSeconds:
                CFAbsoluteTimeGetCurrent() - totalStarted,
            audioDurationSeconds:
                Double(waveform.count)
                / Double(Self.inputSampleRate),
            promptTokens: prompt.inputIDs.count,
            generatedTokens: generated.count,
            stopReason: stopReason
        )
        return MossTranscription(
            rawText: raw,
            text: parsed.text,
            segments: parsed.segments,
            metrics: metrics
        )
    }
}
