import AudioCommon
import Foundation
import MLX
import MLXCommon

public final class IndicMioTTSModel: SpeechGenerationModel, @unchecked Sendable {
    public static let defaultModelId = IndicMioBundleConfig.defaultModelId

    public let sampleRate: Int
    public let bundleDirectory: URL
    public let bundleConfig: IndicMioBundleConfig
    public let modelConfig: IndicMioModelConfig

    private let tokenizer: IndicMioTokenizer
    private let languageModel: IndicMioQwen3Model
    private let contentDecoder: MioCodecContentDecoder
    private let globalEncoder: MioCodecGlobalEncoder
    private let waveDecoder: MioCodecWaveDecoder
    private var state: IndicMioQwen3Model.InferenceState
    private var wavLMReferenceEncoder: IndicMioWavLMFeatureModel?

    private init(
        bundleDirectory: URL,
        bundleConfig: IndicMioBundleConfig,
        modelConfig: IndicMioModelConfig,
        tokenizer: IndicMioTokenizer,
        languageModel: IndicMioQwen3Model,
        contentDecoder: MioCodecContentDecoder,
        globalEncoder: MioCodecGlobalEncoder,
        waveDecoder: MioCodecWaveDecoder
    ) {
        self.bundleDirectory = bundleDirectory
        self.bundleConfig = bundleConfig
        self.modelConfig = modelConfig
        self.sampleRate = bundleConfig.sampleRate
        self.tokenizer = tokenizer
        self.languageModel = languageModel
        self.contentDecoder = contentDecoder
        self.globalEncoder = globalEncoder
        self.waveDecoder = waveDecoder
        self.state = .initial(config: modelConfig)
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> IndicMioTTSModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: [
                "config.json",
                "generation_config.json",
                "bundle_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "model.safetensors",
                "miocodec/*",
            ],
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.45, "Downloading Indic-Mio") }
        )
        return try await fromBundle(directory) { progress, message in
            progressHandler?(0.45 + progress * 0.55, message)
        }
    }

    public static func fromBundle(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> IndicMioTTSModel {
        let configURL = directory.appendingPathComponent("config.json")
        let bundleConfig = try IndicMioBundleConfig.load(from: directory)
        let modelConfig = try IndicMioModelConfig.load(from: configURL)

        progressHandler?(0.05, "Loading Indic-Mio tokenizer")
        let tokenizer = try await IndicMioTokenizer.load(from: directory)

        progressHandler?(0.10, "Initializing Indic-Mio Qwen3")
        let languageModel = IndicMioQwen3Model(config: modelConfig)
        try IndicMioQwen3WeightLoader.loadWeights(
            into: languageModel,
            from: directory,
            progressHandler: { progress, message in
                progressHandler?(0.10 + progress * 0.75, message)
            })

        progressHandler?(0.88, "Loading MioCodec decoder")
        let codecDirectory = directory.appendingPathComponent(bundleConfig.codecPath, isDirectory: true)
        let codecWeights = try CommonWeightLoader.loadAllSafetensors(from: codecDirectory)
        let contentDecoder = MioCodecContentDecoder()
        contentDecoder.loadWeights(from: codecWeights)

        progressHandler?(0.91, "Loading MioCodec global encoder")
        let globalEncoder = MioCodecGlobalEncoder()
        globalEncoder.loadWeights(from: codecWeights)

        progressHandler?(0.94, "Loading MioCodec wave decoder")
        let waveDecoder = MioCodecWaveDecoder()
        waveDecoder.loadWeights(from: codecWeights)

        progressHandler?(1.0, "Indic-Mio runtime ready")
        return IndicMioTTSModel(
            bundleDirectory: directory,
            bundleConfig: bundleConfig,
            modelConfig: modelConfig,
            tokenizer: tokenizer,
            languageModel: languageModel,
            contentDecoder: contentDecoder,
            globalEncoder: globalEncoder,
            waveDecoder: waveDecoder)
    }

    public func resetState() {
        state = .initial(config: modelConfig)
    }

    /// Generate MioCodec content-token ids (`0..<12800`) from Hindi/Indic text.
    public func generateSpeechTokens(
        text: String,
        language: String? = nil,
        sampling: IndicMioSamplingConfig = .default
    ) async throws -> [Int] {
        let promptTokens = try tokenizer.encodeChatPrompt(text: text)
        return try generateSpeechTokens(promptTokens: promptTokens, sampling: sampling)
    }

    /// Decode content-token ids into the first MioCodec latent representation.
    public func decodeContentEmbeddings(contentTokens: [Int]) throws -> MLXArray {
        let ids = try contentTokenIds(contentTokens)
        let embeddings = contentDecoder.decodeContentEmbeddings(ids)
        eval(embeddings)
        return embeddings
    }

    /// Decode MioCodec content-token ids to waveform audio.
    ///
    /// - Parameters:
    ///   - contentTokens: MioCodec token ids in `0..<12800`.
    ///   - globalEmbedding: Optional speaker/acoustic embedding of length 128.
    ///     When omitted, the decoder uses a deterministic zero embedding. For
    ///     cloned-voice output, pass an embedding extracted by MioCodec's global
    ///     branch.
    ///   - targetAudioLength: Optional target length in samples. Defaults to
    ///     `contentTokens.count * 960`, matching the 25 Hz token rate at 24 kHz.
    public func decodeWaveform(
        contentTokens: [Int],
        globalEmbedding: [Float]? = nil,
        targetAudioLength: Int? = nil
    ) throws -> [Float] {
        let ids = try contentTokenIds(contentTokens)
        let embeddings = contentDecoder.decodeContentEmbeddings(ids)
        let speaker = try globalEmbeddingArray(globalEmbedding)
        let waveform = waveDecoder.decode(
            contentEmbeddings: embeddings,
            globalEmbedding: speaker,
            targetAudioLength: targetAudioLength)
        let flat = waveform.squeezed().asType(.float32)
        eval(flat)
        return flat.asArray(Float.self)
    }

    public func generate(
        text: String,
        language: String?,
        globalEmbedding: [Float]? = nil,
        sampling: IndicMioSamplingConfig
    ) async throws -> [Float] {
        let tokens = try await generateSpeechTokens(text: text, language: language, sampling: sampling)
        return try decodeWaveform(contentTokens: tokens, globalEmbedding: globalEmbedding)
    }

    public func generate(
        text: String,
        language: String?,
        referenceAudio: [Float],
        referenceSampleRate: Int,
        sampling: IndicMioSamplingConfig = .default
    ) async throws -> [Float] {
        let globalEmbedding = try await extractGlobalEmbedding(
            referenceAudio: referenceAudio,
            referenceSampleRate: referenceSampleRate)
        return try await generate(
            text: text,
            language: language,
            globalEmbedding: globalEmbedding,
            sampling: sampling)
    }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try await generate(text: text, language: language, sampling: .default)
    }

    public func generate(
        text: String,
        language: String?,
        globalEmbedding: [Float]
    ) async throws -> [Float] {
        try await generate(text: text, language: language, globalEmbedding: globalEmbedding, sampling: .default)
    }

    public func extractGlobalEmbedding(
        referenceAudio: [Float],
        referenceSampleRate: Int
    ) async throws -> [Float] {
        guard !referenceAudio.isEmpty else {
            throw IndicMioError.invalidConfig("reference audio must not be empty")
        }

        let wavlm = try await referenceEncoder()
        let prepared = prepareReferenceAudioForWavLM(
            referenceAudio,
            sourceSampleRate: referenceSampleRate,
            wavlm: wavlm)
        let sslFeatures = wavlm.averagedGlobalFeatures(audio16k: prepared)
        let embedding = globalEncoder(sslFeatures).asType(.float32).squeezed()
        eval(embedding)
        return embedding.asArray(Float.self)
    }

    private func referenceEncoder() async throws -> IndicMioWavLMFeatureModel {
        if let wavLMReferenceEncoder {
            return wavLMReferenceEncoder
        }

        if let bundle = ProcessInfo.processInfo.environment["INDIC_MIO_WAVLM_BUNDLE"], !bundle.isEmpty {
            let model = try IndicMioWavLMFeatureModel.fromBundle(
                URL(fileURLWithPath: bundle, isDirectory: true))
            wavLMReferenceEncoder = model
            return model
        }

        let model = try await IndicMioWavLMFeatureModel.fromPretrained()
        wavLMReferenceEncoder = model
        return model
    }

    private func prepareReferenceAudioForWavLM(
        _ audio: [Float],
        sourceSampleRate: Int,
        wavlm: IndicMioWavLMFeatureModel
    ) -> [Float] {
        let audio24k = sourceSampleRate == IndicMioReferenceConfig.codecSampleRate
            ? audio
            : AudioFileLoader.resample(
                audio,
                from: sourceSampleRate,
                to: IndicMioReferenceConfig.codecSampleRate)

        let padding = referencePaddingSamples24k(
            audioLength: audio24k.count,
            wavlm: wavlm)
        let padded: [Float]
        if padding > 0 {
            padded = [Float](repeating: 0, count: padding)
                + audio24k
                + [Float](repeating: 0, count: padding)
        } else {
            padded = audio24k
        }

        return AudioFileLoader.resample(
            padded,
            from: IndicMioReferenceConfig.codecSampleRate,
            to: IndicMioReferenceConfig.wavLMSampleRate)
    }

    private func referencePaddingSamples24k(
        audioLength: Int,
        wavlm: IndicMioWavLMFeatureModel
    ) -> Int {
        let resampledLength = Double(audioLength)
            / Double(IndicMioReferenceConfig.codecSampleRate)
            * Double(IndicMioReferenceConfig.wavLMSampleRate)
        let expectedFrames = Int(ceil(resampledLength / Double(IndicMioReferenceConfig.wavLMHopSize)))
        let required16k = wavlm.minimumWavLMInputLength(forFeatureFrames: max(1, expectedFrames))
        let required24k = Double(required16k)
            / Double(IndicMioReferenceConfig.wavLMSampleRate)
            * Double(IndicMioReferenceConfig.codecSampleRate)
        return max(0, Int(ceil((required24k - Double(audioLength)) / 2.0)))
    }

    private func generateSpeechTokens(
        promptTokens: [Int],
        sampling: IndicMioSamplingConfig
    ) throws -> [Int] {
        resetState()
        let prompt = MLXArray(promptTokens.map(Int32.init)).expandedDimensions(axis: 0)
        let (prefill, prefillState) = languageModel.forward(inputIds: prompt, state: state)
        eval(prefill)
        state = prefillState

        var logits = lastLogits(prefill)
        var generated: [Int] = []
        var speechCodes: [Int] = []

        for _ in 0..<sampling.maxNewTokens {
            if speechCodes.isEmpty {
                suppressEndTokens(&logits)
            }
            let next = IndicMioSampler.sample(
                logits: logits,
                config: sampling,
                previousTokens: promptTokens + generated)

            if isEndToken(next) {
                break
            }

            generated.append(next)
            if let speechCode = IndicMioSpeechTokens.speechCode(from: next) {
                speechCodes.append(speechCode)
            }

            let stepInput = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
            let (stepLogits, nextState) = languageModel.forward(inputIds: stepInput, state: state)
            eval(stepLogits)
            state = nextState
            logits = lastLogits(stepLogits)
        }

        guard !speechCodes.isEmpty else { throw IndicMioError.noSpeechTokensGenerated }
        return speechCodes
    }

    private func contentTokenIds(_ contentTokens: [Int]) throws -> MLXArray {
        guard !contentTokens.isEmpty else { throw IndicMioError.noSpeechTokensGenerated }
        let invalid = contentTokens.first { $0 < 0 || $0 >= MioCodecFSQ.codebookSize }
        if let invalid {
            throw IndicMioError.invalidConfig("content token out of range: \(invalid)")
        }
        return MLXArray(contentTokens.map(Int32.init)).expandedDimensions(axis: 0)
    }

    private func globalEmbeddingArray(_ embedding: [Float]?) throws -> MLXArray {
        guard let embedding else {
            return MLXArray.zeros([1, MioCodecConfig.default.globalEmbeddingDim])
        }
        guard embedding.count == MioCodecConfig.default.globalEmbeddingDim else {
            throw IndicMioError.invalidConfig(
                "global embedding must contain \(MioCodecConfig.default.globalEmbeddingDim) floats, got \(embedding.count)")
        }
        return MLXArray(embedding).expandedDimensions(axis: 0)
    }

    private func lastLogits(_ logits: MLXArray) -> [Float] {
        let t = logits.dim(1)
        let last = logits[0, t - 1].asType(.float32)
        eval(last)
        return Array(last.asArray(Float.self).prefix(modelConfig.vocabSize))
    }

    private func suppressEndTokens(_ logits: inout [Float]) {
        let minusInf = -Float.greatestFiniteMagnitude
        for token in [
            modelConfig.eosTokenId,
            modelConfig.padTokenId,
            IndicMioPrompt.imEndTokenId,
            IndicMioPrompt.endOfTextTokenId,
        ] where token >= 0 && token < logits.count {
            logits[token] = minusInf
        }
    }

    private func isEndToken(_ token: Int) -> Bool {
        token == modelConfig.eosTokenId
            || token == modelConfig.padTokenId
            || token == IndicMioPrompt.imEndTokenId
            || token == IndicMioPrompt.endOfTextTokenId
    }
}
