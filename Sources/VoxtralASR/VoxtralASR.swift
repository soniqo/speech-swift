import AudioCommon
import Foundation
import MLX
import MLXNN

public enum VoxtralError: Error, LocalizedError {
    case invalidQuantization(bits: Int, groupSize: Int)
    case invalidAudioPacking(hiddenSize: Int, intermediateSize: Int)
    case noWeights(URL)
    case incompatibleCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .invalidQuantization(let bits, let groupSize):
            return "Unsupported Voxtral MLX quantization: bits=\(bits), group_size=\(groupSize)."
        case .invalidAudioPacking(let hiddenSize, let intermediateSize):
            return "Voxtral audio packing requires intermediate_size to be divisible by hidden_size; got \(intermediateSize)/\(hiddenSize)."
        case .noWeights(let directory):
            return "No Voxtral safetensors found in \(directory.path)."
        case .incompatibleCheckpoint(let reason):
            return "Incompatible Voxtral checkpoint: \(reason)"
        }
    }
}

public final class VoxtralModel: Module, SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = VoxtralVariant.int5.modelId
    static let supportedQuantizationBits = Set([2, 3, 4, 5, 6, 8])
    static let downloadAdditionalFiles = ["tekken.json", "preprocessor_config.json"]
    public let config: VoxtralConfig
    public let inputSampleRate = VoxtralAudioFrontend.sampleRate

    @ModuleInfo(key: "language_model") var languageModel: VoxtralLanguageModel
    @ModuleInfo(key: "audio_tower") var audioTower: VoxtralAudioEncoder
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: VoxtralMultiModalProjector

    private let tokenizer: VoxtralTokenizer
    private let frontend: VoxtralAudioFrontend
    private let endTokenIDs = Set([2, 4, 32_000])

    public init(config: VoxtralConfig, tokenizer: VoxtralTokenizer) throws {
        guard config.audioConfig.intermediateSize % config.audioConfig.hiddenSize == 0 else {
            throw VoxtralError.invalidAudioPacking(
                hiddenSize: config.audioConfig.hiddenSize,
                intermediateSize: config.audioConfig.intermediateSize)
        }
        self.config = config
        self.tokenizer = tokenizer
        self.frontend = VoxtralAudioFrontend(melBins: config.audioConfig.numMelBins)
        _languageModel.wrappedValue = VoxtralLanguageModel(config.textConfig)
        _audioTower.wrappedValue = VoxtralAudioEncoder(config.audioConfig)
        _multiModalProjector.wrappedValue = VoxtralMultiModalProjector(config)
        super.init()
    }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        transcribe(
            audio: audio,
            sampleRate: sampleRate,
            options: VoxtralDecodingOptions(language: language))
    }

    public func transcribeWithLanguage(
        audio: [Float], sampleRate: Int, language: String?
    ) -> TranscriptionResult {
        TranscriptionResult(
            text: transcribe(audio: audio, sampleRate: sampleRate, language: language),
            language: language)
    }

    public func transcribe(
        audio: [Float],
        sampleRate: Int,
        options: VoxtralDecodingOptions
    ) -> String {
        let features = frontend.process(audio, inputSampleRate: sampleRate)
        let encoderOutput = audioTower(features.values)
        let audioEmbeddings = multiModalProjector(
            encoderOutput.reshaped(-1, config.audioConfig.intermediateSize))
        let prompt = tokenizer.transcriptionPrompt(
            audioTokenCount: features.audioTokenCount,
            language: options.language)
        let audioStart = prompt.firstIndex(of: config.audioTokenID)
            ?? prompt.endIndex
        let audioEnd = prompt[audioStart...].firstIndex(where: { $0 != config.audioTokenID })
            ?? prompt.endIndex
        precondition(
            audioEnd - audioStart == audioEmbeddings.dim(0),
            "Voxtral prompt/audio embedding count mismatch")

        let prefix = Array(prompt[..<audioStart])
        let suffix = Array(prompt[audioEnd...])
        let prefixEmbeddings = languageModel.model.embed(
            MLXArray(prefix.map(Int32.init)).reshaped(1, prefix.count))
        let suffixEmbeddings = languageModel.model.embed(
            MLXArray(suffix.map(Int32.init)).reshaped(1, suffix.count))
        let promptEmbeddings = MLX.concatenated([
            prefixEmbeddings,
            audioEmbeddings.expandedDimensions(axis: 0),
            suffixEmbeddings,
        ], axis: 1)

        var state = VoxtralLanguageModelState.initial(
            layerCount: config.textConfig.numHiddenLayers)
        let prefill = languageModel.model.forward(
            embeddings: promptEmbeddings,
            state: state)
        state = prefill.1
        // Only the final prompt state predicts the first generated token.
        // Slicing before the 131k-vocabulary projection avoids materializing
        // logits for every audio/prompt position during prefill.
        var logits = languageModel.logits(prefill.0[0, -1])
        eval(logits)

        var generated: [Int] = []
        generated.reserveCapacity(max(0, options.maxTokens))
        for _ in 0..<max(0, options.maxTokens) {
            let token = logits.argMax(axis: -1).item(Int.self)
            if endTokenIDs.contains(token) { break }
            generated.append(token)
            let embedding = languageModel.model.embed(
                MLXArray([Int32(token)]).reshaped(1, 1))
            let next = languageModel.model.forward(embeddings: embedding, state: state)
            state = next.1
            logits = languageModel.logits(next.0[0, -1])
            eval(logits)
        }
        Memory.clearCache()
        return tokenizer.decode(generated)
    }
}

public extension VoxtralModel {
    static func validateQuantization(_ quantization: VoxtralMLXQuantization?) throws {
        guard let quantization else { return }
        guard supportedQuantizationBits.contains(quantization.bits),
              quantization.groupSize > 0 else {
            throw VoxtralError.invalidQuantization(
                bits: quantization.bits, groupSize: quantization.groupSize)
        }
    }

    static func fromDirectory(_ directory: URL) throws -> VoxtralModel {
        let config = try JSONDecoder().decode(
            VoxtralConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try validateQuantization(config.quantization)
        let tokenizer = try VoxtralTokenizer(
            tekkenURL: directory.appendingPathComponent("tekken.json"))
        let model = try VoxtralModel(config: config, tokenizer: tokenizer)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw VoxtralError.noWeights(directory) }

        var weights: [String: MLXArray] = [:]
        for file in files {
            weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        if let quantization = config.quantization {
            MLXNN.quantize(model: model) { path, _ in
                weights["\(path).scales"] == nil
                    ? nil
                    : (quantization.groupSize, quantization.bits, .affine)
            }
        }
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: .all)
        } catch {
            throw VoxtralError.incompatibleCheckpoint(error.localizedDescription)
        }
        model.train(false)
        eval(model)
        return model
    }

    static func load(
        _ modelPath: String = defaultModelId,
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> VoxtralModel {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDirectory),
           isDirectory.boolValue {
            return try fromDirectory(URL(fileURLWithPath: modelPath, isDirectory: true))
        }
        return try await fromPretrained(
            modelPath,
            offlineMode: offlineMode,
            progressHandler: progressHandler)
    }

    static func fromPretrained(
        _ modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> VoxtralModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: downloadAdditionalFiles,
            offlineMode: offlineMode,
            progressHandler: progressHandler)
        return try fromDirectory(directory)
    }
}
