import AudioCommon
import Foundation
import MLX
import MLXNN

/// Native MLX runtime for Nemotron-3.5 Streaming ASR.
///
/// The encoder, convolution, and RNN-T predictor caches advance in 320 ms
/// increments, so partial text arrives without reprocessing the recording.
/// INT5 and INT8 use the same BF16 recurrent state and streaming geometry.
public final class NemotronStreamingASRMLXModel: @unchecked Sendable {
    public static let defaultModelId = NemotronMLXVariant.int5.modelId
    public static let inputSampleRate = 16_000

    static let requiredBundleFiles = [
        "config.json",
        "model.safetensors",
        "vocab.json",
    ]

    static let downloadAdditionalFiles = [
        "model.safetensors",
        "vocab.json",
        "languages.json",
        "lang2slot.json",
        "tokenizer.model",
        "speech_models_export.json",
    ]

    public let modelId: String
    public let modelFolder: URL
    public let configuration: NemotronMLXConfiguration
    public let languages: NemotronLanguages
    public var quantizationBits: Int {
        configuration.quantization.bits
    }

    let network: NemotronMLXNetwork
    let vocabulary: NemotronVocabulary
    let melPreprocessor: StreamingMelPreprocessor
    let inferenceLock = NSLock()

    private init(
        modelId: String,
        modelFolder: URL,
        configuration: NemotronMLXConfiguration,
        languages: NemotronLanguages,
        network: NemotronMLXNetwork,
        vocabulary: NemotronVocabulary
    ) {
        self.modelId = modelId
        self.modelFolder = modelFolder
        self.configuration = configuration
        self.languages = languages
        self.network = network
        self.vocabulary = vocabulary
        melPreprocessor = StreamingMelPreprocessor(
            config: Self.coreStreamingConfiguration
        )
    }

    public static func fromPretrained(
        variant: NemotronMLXVariant = .int5,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRMLXModel {
        let model = try await fromPretrained(
            modelId: variant.modelId,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            progressHandler: progressHandler
        )
        guard model.quantizationBits == variant.quantizationBits else {
            throw NemotronMLXError.invalidConfiguration(
                "\(variant.modelId) declares INT\(model.quantizationBits); "
                    + "expected INT\(variant.quantizationBits)"
            )
        }
        return model
    }

    public static func fromPretrained(
        modelId: String,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRMLXModel {
        let directory: URL
        do {
            directory = try cacheDir
                ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to resolve the Nemotron MLX cache directory",
                underlying: error
            )
        }

        progressHandler?(0, "Preparing Nemotron MLX bundle...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: directory,
                additionalFiles: downloadAdditionalFiles,
                offlineMode: offlineMode
            ) { fraction in
                progressHandler?(
                    fraction * 0.75,
                    "Downloading Nemotron MLX bundle..."
                )
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to download the Nemotron MLX bundle",
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

    public static func fromDirectory(
        _ directory: URL,
        modelId: String = "local/Nemotron-3.5-Streaming-MLX",
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> NemotronStreamingASRMLXModel {
        for file in requiredBundleFiles where !FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(file).path
        ) {
            throw NemotronMLXError.missingModelFile(file)
        }

        progressHandler?(0.05, "Loading Nemotron MLX configuration...")
        let configuration = try JSONDecoder().decode(
            NemotronMLXConfiguration.self,
            from: Data(
                contentsOf: directory.appendingPathComponent("config.json")
            )
        )
        try configuration.validate()

        progressHandler?(0.1, "Loading Nemotron vocabulary...")
        let vocabulary = try NemotronVocabulary.load(
            from: directory.appendingPathComponent("vocab.json")
        )
        guard vocabulary.count == configuration.vocabularySize else {
            throw NemotronMLXError.invalidConfiguration(
                "vocabulary contains \(vocabulary.count) entries; expected "
                    + "\(configuration.vocabularySize)")
        }

        let wrappedLanguages =
            directory.appendingPathComponent("languages.json")
        let flatLanguages = directory.appendingPathComponent("lang2slot.json")
        let languageURL: URL
        if FileManager.default.fileExists(atPath: wrappedLanguages.path) {
            languageURL = wrappedLanguages
        } else if FileManager.default.fileExists(atPath: flatLanguages.path) {
            languageURL = flatLanguages
        } else {
            throw NemotronMLXError.missingModelFile(
                "languages.json or lang2slot.json")
        }
        let languages = try NemotronLanguages.load(from: languageURL)
        guard
            languages.promptDictionary["auto"] != nil,
            languages.promptDictionary.values.allSatisfy({
                (0..<configuration.promptKernel.promptCount).contains($0)
            })
        else {
            throw NemotronMLXError.invalidConfiguration(
                "language map must contain auto and use prompt slots 0..<"
                    + "\(configuration.promptKernel.promptCount)"
            )
        }

        progressHandler?(0.2, "Loading quantized Nemotron MLX weights...")
        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("model.safetensors")
        )
        let network = NemotronMLXNetwork(configuration)
        do {
            try network.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: .all
            )
        } catch {
            throw NemotronMLXError.invalidConfiguration(
                "incompatible MLX weights: \(error.localizedDescription)")
        }
        network.train(false)
        MLX.eval(network.parameters())
        Memory.clearCache()
        progressHandler?(1, "Nemotron MLX ready")

        return NemotronStreamingASRMLXModel(
            modelId: modelId,
            modelFolder: directory,
            configuration: configuration,
            languages: languages,
            network: network,
            vocabulary: vocabulary
        )
    }

    public func createSession(
        language: String? = nil
    ) throws -> NemotronMLXStreamingSession {
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        return try NemotronMLXStreamingSession(
            model: self,
            languageSlot: languages.slot(for: language)
        )
    }

    public func transcribeStream(
        audio: [Float],
        sampleRate: Int,
        language: String? = nil
    ) -> AsyncStream<NemotronStreamingASRModel.PartialTranscript> {
        AsyncStream { continuation in
            Task {
                do {
                    let samples =
                        sampleRate == Self.inputSampleRate
                        ? audio
                        : AudioFileLoader.resample(
                            audio,
                            from: sampleRate,
                            to: Self.inputSampleRate
                        )
                    let session = try self.createSession(language: language)
                    let chunkSamples =
                        self.configuration.streaming.melFrames
                        * Int(
                            self.configuration.preprocessor.windowStride
                                * Double(Self.inputSampleRate)
                        )
                    var offset = 0
                    while offset < samples.count {
                        let end = min(offset + chunkSamples, samples.count)
                        for partial in try session.pushAudio(
                            Array(samples[offset..<end])
                        ) {
                            continuation.yield(partial)
                        }
                        offset = end
                    }
                    for final in try session.finalize() {
                        continuation.yield(final)
                    }
                    continuation.finish()
                } catch {
                    AudioLog.inference.error(
                        "Nemotron MLX streaming failed: \(error)")
                    continuation.finish()
                }
            }
        }
    }

    public func transcribeAudio(
        _ audio: [Float],
        sampleRate: Int,
        language: String? = nil
    ) throws -> String {
        let samples =
            sampleRate == Self.inputSampleRate
            ? audio
            : AudioFileLoader.resample(
                audio,
                from: sampleRate,
                to: Self.inputSampleRate
            )
        let session = try createSession(language: language)
        var results = try session.pushAudio(samples)
        results.append(contentsOf: try session.finalize())
        return results.last?.text ?? ""
    }

    public func warmUp() throws {
        let session = try createSession(language: "en-US")
        _ = try session.pushAudio(
            [Float](
                repeating: 0,
                count: configuration.streaming.melFrames
                    * Self.coreStreamingConfiguration.hopLength
            )
        )
        _ = try session.finalize()
    }

    static let coreStreamingConfiguration = NemotronStreamingConfig.default
}

public final class NemotronMLXStreamingSession {
    private let model: NemotronStreamingASRMLXModel
    private let configuration: NemotronMLXConfiguration
    private let languageMask: MLXArray
    private let caches: [NemotronMLXLayerCache]

    private var preCache: MLXArray
    private var decoderHidden: MLXArray
    private var decoderCell: MLXArray
    private var decoderOutput: MLXArray
    private var sampleBuffer: [Float] = []
    private var audioTail = [Float](repeating: 0, count: 480)
    private var allTokens: [Int] = []
    private var allLogProbabilities: [Float] = []
    private var allTokenFrames: [Int] = []
    private var encoderFrameOffset = 0
    private var lastPublishedText = ""

    init(
        model: NemotronStreamingASRMLXModel,
        languageSlot: Int
    ) throws {
        self.model = model
        configuration = model.configuration
        guard
            (0..<configuration.promptKernel.promptCount)
                .contains(languageSlot)
        else {
            throw NemotronMLXError.invalidConfiguration(
                "language slot \(languageSlot) is outside the prompt kernel"
            )
        }
        caches = (0..<configuration.encoder.layers).map { _ in
            NemotronMLXLayerCache()
        }
        preCache = MLXArray.zeros(
            [
                1,
                configuration.streaming.preCacheSize,
                configuration.preprocessor.features,
            ],
            dtype: .bfloat16
        )
        var mask = [Float](
            repeating: 0,
            count: configuration.promptKernel.promptCount
        )
        mask[languageSlot] = 1
        languageMask = MLXArray(mask)
            .reshaped(1, configuration.promptKernel.promptCount)
            .asType(.bfloat16)

        let initial = model.network.decoder.step(
            token: nil,
            hidden: nil,
            cell: nil
        )
        decoderOutput = initial.output
        decoderHidden = initial.hidden
        decoderCell = initial.cell
        MLX.eval(decoderOutput, decoderHidden, decoderCell)
    }

    public var frameDurationSeconds: Double {
        Double(configuration.encoder.subsamplingFactor)
            * configuration.preprocessor.windowStride
    }

    public func pushAudio(
        _ samples: [Float]
    ) throws -> [NemotronStreamingASRModel.PartialTranscript] {
        model.inferenceLock.lock()
        defer { model.inferenceLock.unlock() }
        sampleBuffer.append(contentsOf: samples)
        let samplesPerChunk =
            configuration.streaming.melFrames
            * NemotronStreamingASRMLXModel.coreStreamingConfiguration.hopLength
        var results: [NemotronStreamingASRModel.PartialTranscript] = []
        var consumedSamples = 0
        defer {
            if consumedSamples > 0 {
                sampleBuffer.removeFirst(consumedSamples)
            }
        }
        while sampleBuffer.count - consumedSamples >= samplesPerChunk {
            let end = consumedSamples + samplesPerChunk
            let chunk = Array(sampleBuffer[consumedSamples..<end])
            consumedSamples = end
            if let partial = try processChunk(chunk) {
                results.append(partial)
            }
        }
        return results
    }

    public func finalize(
    ) throws -> [NemotronStreamingASRModel.PartialTranscript] {
        model.inferenceLock.lock()
        defer { model.inferenceLock.unlock() }
        if !sampleBuffer.isEmpty {
            let samplesPerChunk =
                configuration.streaming.melFrames
                * NemotronStreamingASRMLXModel
                    .coreStreamingConfiguration.hopLength
            let padded = sampleBuffer
                + [Float](
                    repeating: 0,
                    count: max(0, samplesPerChunk - sampleBuffer.count)
                )
            sampleBuffer.removeAll(keepingCapacity: false)
            _ = try processChunk(Array(padded.prefix(samplesPerChunk)))
        }
        guard !allTokens.isEmpty else { return [] }
        return [makePartial(isFinal: true)]
    }

    private func processChunk(
        _ audio: [Float]
    ) throws -> NemotronStreamingASRModel.PartialTranscript? {
        let chunkMel = try makeMelArray(audio)
        let fullMel = MLX.concatenated([preCache, chunkMel], axis: 1)
        preCache = fullMel[
            0...,
            (fullMel.dim(1) - configuration.streaming.preCacheSize)...,
            0...
        ]

        var encoded = model.network.encoder.stream(
            mel: fullMel,
            caches: caches
        )
        encoded = model.network.promptKernel(
            encoded,
            languageMask: languageMask
        )
        var evaluated = [encoded, preCache]
        evaluated.append(contentsOf: caches.flatMap(\.evaluatedArrays))
        MLX.eval(evaluated)

        for frameIndex in 0..<encoded.dim(1) {
            let frame = encoded[0..., frameIndex..<(frameIndex + 1), 0...]
            for _ in 0..<10 {
                let logits = model.network.joint(
                    encoder: frame,
                    prediction: decoderOutput
                )
                MLX.eval(logits)
                let token = logits.argMax(axis: -1).item(Int.self)
                if token == configuration.vocabularySize {
                    break
                }

                let flattened = logits.reshaped(-1).asType(.float32)
                let logProbability = logSoftmax(
                    flattened,
                    axis: -1
                )[token].item(Float.self)
                let next = model.network.decoder.step(
                    token: token,
                    hidden: decoderHidden,
                    cell: decoderCell
                )
                MLX.eval(next.output, next.hidden, next.cell)
                decoderOutput = next.output
                decoderHidden = next.hidden
                decoderCell = next.cell
                allTokens.append(token)
                allLogProbabilities.append(logProbability)
                allTokenFrames.append(
                    encoderFrameOffset + frameIndex
                )
            }
        }
        encoderFrameOffset += encoded.dim(1)
        Memory.clearCache()

        let text = model.vocabulary.decode(allTokens)
        guard !text.isEmpty, text != lastPublishedText else {
            return nil
        }
        lastPublishedText = text
        return makePartial(isFinal: false)
    }

    private func makeMelArray(_ audio: [Float]) throws -> MLXArray {
        // Keep three STFT hops of raw context. Computing every 320 ms chunk
        // independently would reflect-pad each internal boundary and gradually
        // diverge from the continuous-audio frontend used for export testing.
        let tailFrames =
            audioTail.count
            / NemotronStreamingASRMLXModel.coreStreamingConfiguration.hopLength
        let contextualAudio = audioTail + audio
        audioTail = Array(
            contextualAudio.suffix(audioTail.count)
        )
        let extracted = try model.melPreprocessor.extractRaw(contextualAudio)
        let source = extracted.mel
        let targetFrames = configuration.streaming.melFrames
        let sourceFrames = source.shape[2].intValue
        let bins = configuration.preprocessor.features
        let pointer = source.dataPointer.assumingMemoryBound(to: Float.self)
        var values = [Float](
            repeating: 0,
            count: targetFrames * bins
        )
        let copiedFrames = min(
            max(0, sourceFrames - tailFrames),
            targetFrames
        )
        for frame in 0..<copiedFrames {
            for bin in 0..<bins {
                values[frame * bins + bin] =
                    pointer[
                        bin * sourceFrames
                            + tailFrames
                            + frame
                    ]
            }
        }
        return MLXArray(values)
            .reshaped(1, targetFrames, bins)
            .asType(.bfloat16)
    }

    private func makePartial(
        isFinal: Bool
    ) -> NemotronStreamingASRModel.PartialTranscript {
        let confidence: Float
        if allLogProbabilities.isEmpty {
            confidence = 0
        } else {
            confidence = min(
                1,
                exp(
                    allLogProbabilities.reduce(0, +)
                        / Float(allLogProbabilities.count)
                )
            )
        }
        let words = model.vocabulary.decodeTimedWords(
            allTokens,
            frames: allTokenFrames
        ).map {
            TimedWord(
                text: $0.word,
                startTime:
                    Double($0.startFrame) * frameDurationSeconds,
                endTime:
                    Double($0.endFrame + 1) * frameDurationSeconds
            )
        }
        return NemotronStreamingASRModel.PartialTranscript(
            text: model.vocabulary.decode(allTokens),
            isFinal: isFinal,
            confidence: confidence,
            segmentIndex: 0,
            wordBoostingChangedDecisions: 0,
            words: words
        )
    }
}
