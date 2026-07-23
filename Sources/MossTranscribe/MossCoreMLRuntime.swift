import AudioCommon
import CoreML
import Foundation
import Tokenizers

struct MossDecoderConfiguration: Decodable, Sendable {
    struct Multifunction: Decodable, Sendable {
        let decoderFunction: String
        let embeddingFunction: String
        let file: String

        enum CodingKeys: String, CodingKey {
            case decoderFunction = "decoder_function"
            case embeddingFunction = "embedding_function"
            case file
        }
    }

    let hiddenSize: Int
    let maxSequenceLength: Int
    let vocabularySize: Int
    let enumeratedTokenLengths: [Int]
    let shapeMode: String
    let ioPrecision: String
    let multifunction: Multifunction

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case maxSequenceLength = "max_seq_length"
        case vocabularySize = "vocab_size"
        case enumeratedTokenLengths = "enumerated_t"
        case shapeMode = "shape_mode"
        case ioPrecision = "io_precision"
        case multifunction
    }

    func validate() throws {
        guard hiddenSize > 0 else {
            throw MossTranscribeError.invalidConfiguration(
                "hidden_size must be positive"
            )
        }
        guard maxSequenceLength > 0 else {
            throw MossTranscribeError.invalidConfiguration(
                "max_seq_length must be positive"
            )
        }
        guard vocabularySize > 0 else {
            throw MossTranscribeError.invalidConfiguration(
                "vocab_size must be positive"
            )
        }
        guard
            !enumeratedTokenLengths.isEmpty,
            enumeratedTokenLengths.allSatisfy({ $0 > 0 })
        else {
            throw MossTranscribeError.invalidConfiguration(
                "enumerated_t must contain positive token lengths"
            )
        }
        guard shapeMode == "range" || shapeMode == "enumerated" else {
            throw MossTranscribeError.invalidConfiguration(
                "unsupported shape_mode \(shapeMode)"
            )
        }
        guard ioPrecision == "float16" else {
            throw MossTranscribeError.invalidConfiguration(
                "native runtime requires float16 decoder I/O, found \(ioPrecision)"
            )
        }
    }

    func prefillChunks(tokenCount: Int) throws -> [Int] {
        guard tokenCount > 0 else { return [] }
        let supported = Array(Set(enumeratedTokenLengths)).sorted()
        guard let maximum = supported.last else { return [] }

        if shapeMode == "range" {
            var remaining = tokenCount
            var chunks: [Int] = []
            while remaining > 0 {
                let size = min(remaining, maximum)
                chunks.append(size)
                remaining -= size
            }
            return chunks
        }

        guard supported.first == 1 else {
            throw MossTranscribeError.invalidConfiguration(
                "enumerated_t must include 1 for exact prompt prefill"
            )
        }
        var remaining = tokenCount
        var chunks: [Int] = []
        while remaining > 0 {
            guard let size = supported.last(where: { $0 <= remaining }) else {
                throw MossTranscribeError.invalidConfiguration(
                    "cannot represent prefill length \(tokenCount) with enumerated_t"
                )
            }
            chunks.append(size)
            remaining -= size
        }
        return chunks
    }
}

struct MossBundleConfiguration: Decodable, Sendable {
    struct HostContract: Decodable, Sendable {
        let audioChunkSamples: Int
        let audioTokensPerSecond: Double
        let decoderCacheLength: Int
        let sampleRate: Int

        enum CodingKeys: String, CodingKey {
            case audioChunkSamples = "audio_chunk_samples"
            case audioTokensPerSecond = "audio_tokens_per_second"
            case decoderCacheLength = "decoder_cache_length"
            case sampleRate = "sample_rate"
        }
    }

    let backend: String
    let modelType: String
    let hostContract: HostContract

    enum CodingKeys: String, CodingKey {
        case backend
        case modelType = "model_type"
        case hostContract = "host_contract"
    }

    func validate(decoder: MossDecoderConfiguration) throws {
        guard backend == "coreml" else {
            throw MossTranscribeError.invalidConfiguration(
                "bundle backend must be coreml"
            )
        }
        guard modelType == "moss-transcribe-diarize-coreml" else {
            throw MossTranscribeError.invalidConfiguration(
                "unsupported bundle model_type \(modelType)"
            )
        }
        guard
            hostContract.sampleRate
                == MossWhisperFeatureExtractor.sampleRate,
            hostContract.audioChunkSamples
                == MossWhisperFeatureExtractor.chunkSamples,
            hostContract.decoderCacheLength
                == decoder.maxSequenceLength
        else {
            throw MossTranscribeError.invalidConfiguration(
                "bundle host contract does not match the native runtime"
            )
        }
        let expectedTokenRate =
            Double(MossWhisperFeatureExtractor.sampleRate)
            / Double(MossWhisperFeatureExtractor.encoderStrideSamples)
        guard
            abs(
                hostContract.audioTokensPerSecond - expectedTokenRate
            ) < 1e-9
        else {
            throw MossTranscribeError.invalidConfiguration(
                "bundle audio token rate does not match the encoder stride"
            )
        }
    }
}

struct MossPreprocessorConfiguration: Decodable, Sendable {
    let featureSize: Int
    let hopLength: Int
    let fftSize: Int
    let sampleCount: Int
    let maximumFrames: Int
    let sampleRate: Int

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case hopLength = "hop_length"
        case fftSize = "n_fft"
        case sampleCount = "n_samples"
        case maximumFrames = "nb_max_frames"
        case sampleRate = "sampling_rate"
    }

    func validate() throws {
        guard
            featureSize == MossWhisperFeatureExtractor.melBins,
            hopLength == MossWhisperFeatureExtractor.hopLength,
            fftSize == MossWhisperFeatureExtractor.fftSize,
            sampleCount == MossWhisperFeatureExtractor.chunkSamples,
            maximumFrames == MossWhisperFeatureExtractor.timeFrames,
            sampleRate == MossWhisperFeatureExtractor.sampleRate
        else {
            throw MossTranscribeError.invalidConfiguration(
                "preprocessor contract does not match the native Whisper frontend"
            )
        }
    }
}

/// Native Core ML runtime for MOSS-Transcribe-Diarize 0.9B.
///
/// The audio encoder and stateful Qwen3 decoder run through Core ML. Prompt
/// rendering, Whisper preprocessing, audio-embedding injection, MLState cache
/// management, greedy generation, and transcript parsing are implemented in
/// Swift; Python and ONNX Runtime are not required.
@available(macOS 15.0, iOS 18.0, *)
public final class MossTranscribeModel: SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = MossModelVariant.int8.modelId
    public static let defaultInstruction =
        MossPromptProcessor.defaultInstruction
    public static let inputSampleRate = MossWhisperFeatureExtractor.sampleRate

    static let requiredBundleFiles = [
        "config.json",
        "audio_encoder.mlmodelc/metadata.json",
        "audio_encoder.mlmodelc/model.mil",
        "audio_encoder.mlmodelc/coremldata.bin",
        "audio_encoder.mlmodelc/weights/weight.bin",
        "decoder/decoder.mlmodelc/metadata.json",
        "decoder/decoder.mlmodelc/model.mil",
        "decoder/decoder.mlmodelc/coremldata.bin",
        "decoder/decoder.mlmodelc/weights/weight.bin",
        "decoder/config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    private static let maximumCachedTokenEmbeddings = 4_096

    public let modelId: String
    public let modelFolder: URL
    public var inputSampleRate: Int { Self.inputSampleRate }

    private let audioEncoder: MLModel
    private let embeddingModel: MLModel
    private let decoderModel: MLModel
    private let tokenizer: Tokenizer
    private let promptProcessor: MossPromptProcessor
    private let decoderConfiguration: MossDecoderConfiguration
    private let featureExtractor = MossWhisperFeatureExtractor()
    private let inferenceLock = NSLock()
    private var embeddingCache: [Int: [Float16]] = [:]

    private init(
        modelId: String,
        modelFolder: URL,
        audioEncoder: MLModel,
        embeddingModel: MLModel,
        decoderModel: MLModel,
        tokenizer: Tokenizer,
        promptProcessor: MossPromptProcessor,
        decoderConfiguration: MossDecoderConfiguration
    ) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.audioEncoder = audioEncoder
        self.embeddingModel = embeddingModel
        self.decoderModel = decoderModel
        self.tokenizer = tokenizer
        self.promptProcessor = promptProcessor
        self.decoderConfiguration = decoderConfiguration
    }

    public static func fromPretrained(
        variant: MossModelVariant = .int8,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossTranscribeModel {
        try await fromPretrained(
            modelId: variant.modelId,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            computeUnits: computeUnits,
            progressHandler: progressHandler
        )
    }

    public static func fromPretrained(
        modelId: String,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossTranscribeModel {
        let directory: URL
        do {
            directory = try cacheDir
                ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to resolve the MOSS cache directory",
                underlying: error
            )
        }

        progressHandler?(0, "Preparing MOSS Core ML bundle...")
        do {
            if !hasCompleteCachedBundle(in: directory) {
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: modelId,
                    to: directory,
                    additionalFiles: [
                        "config.json",
                        "audio_encoder.mlmodelc/**",
                        "decoder/decoder.mlmodelc/**",
                        "decoder/config.json",
                        "processor_config.json",
                        "preprocessor_config.json",
                        "generation_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "vocab.json",
                        "merges.txt",
                        "chat_template.jinja",
                    ],
                    offlineMode: offlineMode
                ) { fraction in
                    progressHandler?(
                        fraction * 0.8,
                        "Downloading MOSS Core ML bundle..."
                    )
                }
            } else {
                progressHandler?(0.8, "Using cached MOSS Core ML bundle")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to download the MOSS bundle",
                underlying: error
            )
        }

        progressHandler?(0.82, "Loading MOSS tokenizer...")
        return try await fromDirectory(
            directory,
            modelId: modelId,
            computeUnits: computeUnits
        ) { fraction, message in
            progressHandler?(0.82 + fraction * 0.18, message)
        }
    }

    /// Load a downloaded or freshly exported Core ML bundle.
    public static func fromDirectory(
        _ directory: URL,
        modelId: String = "local/MOSS-Transcribe-Diarize",
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> MossTranscribeModel {
        guard hasCompleteCachedBundle(in: directory) else {
            let missing = requiredBundleFiles.first {
                !FileManager.default.fileExists(
                    atPath: directory.appendingPathComponent($0).path
                )
            } ?? "unknown file"
            throw MossTranscribeError.missingModelFile(missing)
        }

        let decoderConfigURL = directory
            .appendingPathComponent("decoder/config.json")
        let bundleConfigURL = directory
            .appendingPathComponent("config.json")
        let processorConfigURL = directory
            .appendingPathComponent("processor_config.json")
        let preprocessorConfigURL = directory
            .appendingPathComponent("preprocessor_config.json")
        let decoderConfiguration = try JSONDecoder().decode(
            MossDecoderConfiguration.self,
            from: Data(contentsOf: decoderConfigURL)
        )
        try decoderConfiguration.validate()
        let bundleConfiguration = try JSONDecoder().decode(
            MossBundleConfiguration.self,
            from: Data(contentsOf: bundleConfigURL)
        )
        try bundleConfiguration.validate(decoder: decoderConfiguration)
        let preprocessorConfiguration = try JSONDecoder().decode(
            MossPreprocessorConfiguration.self,
            from: Data(contentsOf: preprocessorConfigURL)
        )
        try preprocessorConfiguration.validate()
        let processorConfiguration = try JSONDecoder().decode(
            MossProcessorConfiguration.self,
            from: Data(contentsOf: processorConfigURL)
        )
        guard processorConfiguration.audioTokensPerSecond > 0 else {
            throw MossTranscribeError.invalidConfiguration(
                "audio_tokens_per_second must be positive"
            )
        }
        guard processorConfiguration.audioMergeSize == 4 else {
            throw MossTranscribeError.invalidConfiguration(
                "native audio encoder requires audio_merge_size=4"
            )
        }
        let expectedTokenRate =
            Double(MossWhisperFeatureExtractor.sampleRate)
            / Double(MossWhisperFeatureExtractor.encoderStrideSamples)
        guard abs(
            processorConfiguration.audioTokensPerSecond
                - expectedTokenRate
        ) < 1e-9 else {
            throw MossTranscribeError.invalidConfiguration(
                "audio_tokens_per_second does not match the encoder stride"
            )
        }

        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let promptProcessor = try MossPromptProcessor(
            tokenizer: tokenizer,
            configuration: processorConfiguration
        )

        let modelURL = directory
            .appendingPathComponent("decoder")
            .appendingPathComponent(
                decoderConfiguration.multifunction.file
            )
        let audioURL = directory
            .appendingPathComponent("audio_encoder.mlmodelc")

        func configuration(functionName: String? = nil)
            -> MLModelConfiguration
        {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            configuration.functionName = functionName
            configuration.optimizationHints.specializationStrategy =
                .fastPrediction
            return configuration
        }

        progressHandler?(0.15, "Loading MOSS audio encoder...")
        let audioEncoder = try CoreMLLoader.load(
            url: audioURL,
            configuration: configuration(),
            name: "moss-audio-encoder"
        )
        progressHandler?(0.5, "Loading MOSS token embedding...")
        let embeddingModel = try CoreMLLoader.load(
            url: modelURL,
            configuration: configuration(
                functionName:
                    decoderConfiguration.multifunction.embeddingFunction
            ),
            name: "moss-token-embedding"
        )
        progressHandler?(0.7, "Loading MOSS stateful decoder...")
        let decoderModel = try CoreMLLoader.load(
            url: modelURL,
            configuration: configuration(
                functionName:
                    decoderConfiguration.multifunction.decoderFunction
            ),
            name: "moss-stateful-decoder"
        )
        progressHandler?(1, "MOSS ready")

        return MossTranscribeModel(
            modelId: modelId,
            modelFolder: directory,
            audioEncoder: audioEncoder,
            embeddingModel: embeddingModel,
            decoderModel: decoderModel,
            tokenizer: tokenizer,
            promptProcessor: promptProcessor,
            decoderConfiguration: decoderConfiguration
        )
    }

    /// Compile and specialize both the 128-token prefill and one-token decode
    /// paths without generating a transcript.
    public func warmUp() throws {
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        try autoreleasepool {
            let features = try Self.makeMultiArray(
                shape: [1, MossWhisperFeatureExtractor.melBins,
                        MossWhisperFeatureExtractor.timeFrames],
                dataType: .float32
            )
            features.dataPointer.bindMemory(
                to: Float.self,
                capacity:
                    MossWhisperFeatureExtractor.melBins
                    * MossWhisperFeatureExtractor.timeFrames
            ).initialize(
                repeating: 0,
                count:
                    MossWhisperFeatureExtractor.melBins
                    * MossWhisperFeatureExtractor.timeFrames
            )
            let audioInput = try MLDictionaryFeatureProvider(dictionary: [
                "input_features": MLFeatureValue(multiArray: features)
            ])
            _ = try audioEncoder.prediction(from: audioInput)

            let warmPrompt = try promptProcessor.prepare(
                audioTokenCount:
                    MossWhisperFeatureExtractor.audioTokenCount(
                        sampleCount:
                            MossWhisperFeatureExtractor.chunkSamples
                    )
            )
            for token in Set(warmPrompt.inputIDs)
            where token != promptProcessor.audioTokenID {
                _ = try embedding(for: token)
            }

            let token = promptProcessor.eosTokenID
            let tokenEmbedding = try embedding(for: token)
            let maximum = decoderConfiguration
                .enumeratedTokenLengths.max() ?? 1
            let prefillLength = min(128, maximum)
            var embeddings = [Float16](
                repeating: 0,
                count: prefillLength * decoderConfiguration.hiddenSize
            )
            for position in 0..<prefillLength {
                let rowStart =
                    position * decoderConfiguration.hiddenSize
                let rowEnd =
                    (position + 1) * decoderConfiguration.hiddenSize
                embeddings.replaceSubrange(
                    rowStart..<rowEnd,
                    with: tokenEmbedding
                )
            }
            let state = decoderModel.makeState()
            _ = try runDecoder(
                embeddings: embeddings,
                positions: Array(0..<prefillLength),
                state: state
            )
            _ = try runDecoder(
                embeddings: tokenEmbedding,
                positions: [prefillLength],
                state: state
            )
        }
    }

    /// Run transcription, speaker attribution, and timestamp generation.
    public func transcribeDetailed(
        audio: [Float],
        sampleRate: Int,
        maxNewTokens: Int = 512,
        instruction: String = MossTranscribeModel.defaultInstruction
    ) throws -> MossTranscription {
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        return try autoreleasepool {
            try transcribeLocked(
                audio: audio,
                sampleRate: sampleRate,
                maxNewTokens: maxNewTokens,
                instruction: instruction
            )
        }
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
                "MOSS transcription failed: \(error.localizedDescription)"
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

    private func transcribeLocked(
        audio: [Float],
        sampleRate: Int,
        maxNewTokens: Int,
        instruction: String
    ) throws -> MossTranscription {
        guard !audio.isEmpty else {
            throw MossTranscribeError.invalidAudio("the waveform is empty")
        }
        guard sampleRate > 0 else {
            throw MossTranscribeError.invalidAudio(
                "sample rate must be positive"
            )
        }
        guard maxNewTokens > 0 else {
            throw MossTranscribeError.invalidConfiguration(
                "maxNewTokens must be positive"
            )
        }

        let totalStarted = CFAbsoluteTimeGetCurrent()
        var preprocessingSeconds = 0.0
        var audioEncoderSeconds = 0.0

        var phaseStarted = CFAbsoluteTimeGetCurrent()
        let waveform: [Float]
        if sampleRate == Self.inputSampleRate {
            waveform = audio
        } else {
            waveform = AudioFileLoader.resample(
                audio,
                from: sampleRate,
                to: Self.inputSampleRate
            )
        }
        preprocessingSeconds +=
            CFAbsoluteTimeGetCurrent() - phaseStarted
        guard !waveform.isEmpty else {
            throw MossTranscribeError.invalidAudio(
                "resampling produced an empty waveform"
            )
        }

        var audioEmbeddings: [Float16] = []
        var audioTokenCount = 0
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
            let realTokens =
                MossWhisperFeatureExtractor.audioTokenCount(
                    sampleCount: chunk.count
                )

            phaseStarted = CFAbsoluteTimeGetCurrent()
            let features = try featureExtractor.extractPaddedChunk(chunk)
            preprocessingSeconds +=
                CFAbsoluteTimeGetCurrent() - phaseStarted

            phaseStarted = CFAbsoluteTimeGetCurrent()
            let encoded = try encodeAudio(
                features,
                realTokenCount: realTokens
            )
            audioEncoderSeconds +=
                CFAbsoluteTimeGetCurrent() - phaseStarted
            audioEmbeddings.append(contentsOf: encoded)
            audioTokenCount += realTokens
        }

        phaseStarted = CFAbsoluteTimeGetCurrent()
        let prompt = try promptProcessor.prepare(
            audioTokenCount: audioTokenCount,
            instruction: instruction
        )
        preprocessingSeconds +=
            CFAbsoluteTimeGetCurrent() - phaseStarted
        guard prompt.inputIDs.count < decoderConfiguration.maxSequenceLength
        else {
            throw MossTranscribeError.promptTooLong(
                actual: prompt.inputIDs.count,
                maximum: decoderConfiguration.maxSequenceLength
            )
        }
        guard prompt.audioPlaceholderCount == audioTokenCount else {
            throw MossTranscribeError.audioEmbeddingMismatch(
                placeholders: prompt.audioPlaceholderCount,
                embeddings: audioTokenCount
            )
        }

        let prefillStarted = CFAbsoluteTimeGetCurrent()
        var mixedEmbeddings = [Float16](
            repeating: 0,
            count: prompt.inputIDs.count
                * decoderConfiguration.hiddenSize
        )
        var audioIndex = 0
        for (position, token) in prompt.inputIDs.enumerated() {
            let source: ArraySlice<Float16>
            if token == promptProcessor.audioTokenID {
                let start = audioIndex * decoderConfiguration.hiddenSize
                source = audioEmbeddings[
                    start..<(start + decoderConfiguration.hiddenSize)
                ]
                audioIndex += 1
            } else {
                source = try embedding(for: token)[...]
            }
            let rowStart =
                position * decoderConfiguration.hiddenSize
            let rowEnd =
                (position + 1) * decoderConfiguration.hiddenSize
            mixedEmbeddings.replaceSubrange(
                rowStart..<rowEnd,
                with: source
            )
        }

        let state = decoderModel.makeState()
        var offset = 0
        var nextToken: Int?
        for size in try decoderConfiguration.prefillChunks(
            tokenCount: prompt.inputIDs.count
        ) {
            let scalarStart = offset * decoderConfiguration.hiddenSize
            let scalarEnd =
                (offset + size) * decoderConfiguration.hiddenSize
            nextToken = try runDecoder(
                embeddings: Array(
                    mixedEmbeddings[scalarStart..<scalarEnd]
                ),
                positions: Array(offset..<(offset + size)),
                state: state
            )
            offset += size
        }
        guard var candidate = nextToken else {
            throw MossTranscribeError.inferenceFailed(
                "decoder received an empty prompt"
            )
        }
        let decoderPrefillSeconds =
            CFAbsoluteTimeGetCurrent() - prefillStarted

        let decodeStarted = CFAbsoluteTimeGetCurrent()
        let availableContext =
            decoderConfiguration.maxSequenceLength - prompt.inputIDs.count
        let generationLimit = min(maxNewTokens, availableContext)
        var generated: [Int] = []
        generated.reserveCapacity(generationLimit)
        var position = prompt.inputIDs.count
        var stopReason: MossGenerationStopReason =
            maxNewTokens > availableContext ? .contextLimit : .maximumTokens

        while generated.count < generationLimit {
            generated.append(candidate)
            if candidate == prompt.eosTokenID {
                stopReason = .endOfSequence
                break
            }
            guard position < decoderConfiguration.maxSequenceLength else {
                stopReason = .contextLimit
                break
            }
            candidate = try runDecoder(
                embeddings: try embedding(for: candidate),
                positions: [position],
                state: state
            )
            position += 1
        }
        let tokenDecodeSeconds =
            CFAbsoluteTimeGetCurrent() - decodeStarted

        let raw = tokenizer.decode(
            tokens: generated,
            skipSpecialTokens: true
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        let parsed = MossTranscriptParser.plainText(from: raw)
        let totalSeconds =
            CFAbsoluteTimeGetCurrent() - totalStarted
        let metrics = MossTranscriptionMetrics(
            preprocessingSeconds: preprocessingSeconds,
            audioEncoderSeconds: audioEncoderSeconds,
            decoderPrefillSeconds: decoderPrefillSeconds,
            tokenDecodeSeconds: tokenDecodeSeconds,
            totalSeconds: totalSeconds,
            audioDurationSeconds:
                Double(waveform.count) / Double(Self.inputSampleRate),
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

    private func encodeAudio(
        _ features: MossLogMelFeatures,
        realTokenCount: Int
    ) throws -> [Float16] {
        let input = try Self.makeMultiArray(
            shape: [1, features.melBins, features.timeFrames],
            dataType: .float32
        )
        let pointer = input.dataPointer.bindMemory(
            to: Float.self,
            capacity: features.data.count
        )
        features.data.withUnsafeBufferPointer { source in
            pointer.update(
                from: source.baseAddress!,
                count: source.count
            )
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: input)
        ])
        let output = try audioEncoder.prediction(from: provider)
        guard let embeddings = output
            .featureValue(for: "audio_embeds")?
            .multiArrayValue
        else {
            throw MossTranscribeError.missingModelOutput("audio_embeds")
        }
        let shape = embeddings.shape.map(\.intValue)
        guard
            shape.count == 3,
            shape[0] == 1,
            shape[1] >= realTokenCount,
            shape[2] == decoderConfiguration.hiddenSize
        else {
            throw MossTranscribeError.inferenceFailed(
                "unexpected audio_embeds shape \(shape)"
            )
        }
        return try Self.copyRows(
            from: embeddings,
            rowCount: realTokenCount,
            width: decoderConfiguration.hiddenSize
        )
    }

    private func embedding(for token: Int) throws -> [Float16] {
        if let cached = embeddingCache[token] {
            return cached
        }
        let tokenArray = try Self.makeMultiArray(
            shape: [1, 1],
            dataType: .int32
        )
        tokenArray.dataPointer.bindMemory(
            to: Int32.self,
            capacity: 1
        )[0] = Int32(token)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "token_id": MLFeatureValue(multiArray: tokenArray)
        ])
        let output = try embeddingModel.prediction(from: provider)
        guard let array = output
            .featureValue(for: "embedding")?
            .multiArrayValue
        else {
            throw MossTranscribeError.missingModelOutput("embedding")
        }
        let copied = try Self.copyRows(
            from: array,
            rowCount: 1,
            width: decoderConfiguration.hiddenSize
        )
        if embeddingCache.count < Self.maximumCachedTokenEmbeddings {
            embeddingCache[token] = copied
        }
        return copied
    }

    private func runDecoder(
        embeddings: [Float16],
        positions: [Int],
        state: MLState
    ) throws -> Int {
        let tokenCount = positions.count
        guard
            tokenCount > 0,
            embeddings.count
                == tokenCount * decoderConfiguration.hiddenSize
        else {
            throw MossTranscribeError.inferenceFailed(
                "decoder embedding/position shape mismatch"
            )
        }

        let embeddingArray = try Self.makeMultiArray(
            shape: [1, tokenCount, decoderConfiguration.hiddenSize],
            dataType: .float16
        )
        let embeddingPointer = embeddingArray.dataPointer.bindMemory(
            to: Float16.self,
            capacity: embeddings.count
        )
        embeddings.withUnsafeBufferPointer { source in
            embeddingPointer.update(
                from: source.baseAddress!,
                count: source.count
            )
        }

        let positionArray = try Self.makeMultiArray(
            shape: [tokenCount],
            dataType: .int32
        )
        let positionPointer = positionArray.dataPointer.bindMemory(
            to: Int32.self,
            capacity: tokenCount
        )
        for index in positions.indices {
            positionPointer[index] = Int32(positions[index])
        }

        let maskCount =
            tokenCount * decoderConfiguration.maxSequenceLength
        let attentionMask = try Self.makeMultiArray(
            shape: [
                1,
                1,
                tokenCount,
                decoderConfiguration.maxSequenceLength,
            ],
            dataType: .float16
        )
        let maskPointer = attentionMask.dataPointer.bindMemory(
            to: Float16.self,
            capacity: maskCount
        )
        let blocked = Float16(-10_000)
        for row in 0..<tokenCount {
            let visibleThrough = positions[row]
            let base = row * decoderConfiguration.maxSequenceLength
            for column in 0..<decoderConfiguration.maxSequenceLength {
                maskPointer[base + column] =
                    column <= visibleThrough ? 0 : blocked
            }
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "input_embeds": MLFeatureValue(multiArray: embeddingArray),
            "positions": MLFeatureValue(multiArray: positionArray),
            "attention_mask": MLFeatureValue(multiArray: attentionMask),
        ])
        let output = try decoderModel.prediction(
            from: provider,
            using: state
        )
        guard let logits = output
            .featureValue(for: "logits")?
            .multiArrayValue
        else {
            throw MossTranscribeError.missingModelOutput("logits")
        }
        return try Self.argmax(logits)
    }

    private static func makeMultiArray(
        shape: [Int],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType
        )
    }

    private static func copyRows(
        from array: MLMultiArray,
        rowCount: Int,
        width: Int
    ) throws -> [Float16] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        guard
            shape.count == 3,
            rowCount <= shape[1],
            width == shape[2]
        else {
            throw MossTranscribeError.inferenceFailed(
                "cannot copy \(rowCount)x\(width) rows from shape \(shape)"
            )
        }

        var result = [Float16](
            repeating: 0,
            count: rowCount * width
        )
        switch array.dataType {
        case .float16:
            let source = array.dataPointer.assumingMemoryBound(
                to: Float16.self
            )
            for row in 0..<rowCount {
                for column in 0..<width {
                    result[row * width + column] =
                        source[row * strides[1] + column * strides[2]]
                }
            }
        case .float32:
            let source = array.dataPointer.assumingMemoryBound(
                to: Float.self
            )
            for row in 0..<rowCount {
                for column in 0..<width {
                    result[row * width + column] = Float16(
                        source[row * strides[1] + column * strides[2]]
                    )
                }
            }
        default:
            throw MossTranscribeError.unsupportedArrayType(
                "\(array.dataType.rawValue)"
            )
        }
        return result
    }

    private static func argmax(_ array: MLMultiArray) throws -> Int {
        guard
            let vocabularySize = array.shape.last?.intValue,
            vocabularySize > 0,
            let vocabularyStride = array.strides.last?.intValue
        else {
            throw MossTranscribeError.inferenceFailed(
                "logits has no vocabulary axis"
            )
        }

        var bestIndex = 0
        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(
                to: Float16.self
            )
            var best = pointer[0]
            for index in 1..<vocabularySize {
                let value = pointer[index * vocabularyStride]
                if value > best {
                    best = value
                    bestIndex = index
                }
            }
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(
                to: Float.self
            )
            var best = pointer[0]
            for index in 1..<vocabularySize {
                let value = pointer[index * vocabularyStride]
                if value > best {
                    best = value
                    bestIndex = index
                }
            }
        default:
            throw MossTranscribeError.unsupportedArrayType(
                "\(array.dataType.rawValue)"
            )
        }
        return bestIndex
    }
}
