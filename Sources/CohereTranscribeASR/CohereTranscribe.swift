// Portions adapted from mlx-audio-swift (MIT License).
// Copyright (c) 2025 Blaizzy and contributors.

import AudioCommon
import Foundation
import MLX
import MLXNN

private struct CoherePrefillContext {
    let encoderStates: MLXArray
    let promptLength: Int
    var logits: MLXArray
    var cache: CohereTranscribeDecoderKVCache
}

public enum CohereTranscribeError: Error, LocalizedError {
    case invalidQuantization(bits: Int, groupSize: Int)
    case missingTokenizerPrompt(String)
    case noWeights(URL)
    case incompatibleCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .invalidQuantization(let bits, let groupSize):
            return "Unsupported Cohere MLX quantization: bits=\(bits), group_size=\(groupSize)."
        case .missingTokenizerPrompt(let language):
            return "Cohere tokenizer is missing prompt tokens for language \(language)."
        case .noWeights(let directory):
            return "No Cohere safetensors found in \(directory.path)."
        case .incompatibleCheckpoint(let reason):
            return "Incompatible Cohere checkpoint: \(reason)"
        }
    }
}

public final class CohereTranscribeModel: Module, SpeechRecognitionModel, @unchecked Sendable {
    public static let defaultModelId = CohereTranscribeVariant.int5.modelId
    static let supportedQuantizationBits = Set([2, 3, 4, 5, 6, 8])
    static let downloadAdditionalFiles = ["tokenizer.model", "tokenizer_config.json"]
    public let config: CohereTranscribeConfig
    public let inputSampleRate: Int

    @ModuleInfo(key: "encoder") var encoder: ConformerEncoder
    @ModuleInfo(key: "decoder") var decoder: TransformerDecoderWrapper
    @ModuleInfo(key: "bridge_proj") var bridgeProjection: Linear?
    @ModuleInfo(key: "lm_head") var languageModelHead: Linear

    private let tokenizer: CohereTranscribeTokenizer
    private let frontend: CohereAudioFrontend

    public init(config: CohereTranscribeConfig, tokenizer: CohereTranscribeTokenizer) throws {
        self.config = config
        self.inputSampleRate = config.sampleRate
        self.tokenizer = tokenizer
        self.frontend = CohereAudioFrontend(
            sampleRate: config.sampleRate,
            melBins: config.encoder.featIn)
        self._encoder.wrappedValue = ConformerEncoder(config.encoder)
        self._decoder.wrappedValue = TransformerDecoderWrapper(config: config)
        self._bridgeProjection.wrappedValue = config.encoder.dModel == config.decoder.hiddenSize
            ? nil
            : Linear(config.encoder.dModel, config.decoder.hiddenSize)
        self._languageModelHead.wrappedValue = Linear(config.decoder.hiddenSize, config.vocabSize)
        super.init()
    }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        let options = CohereTranscribeDecodingOptions(language: language ?? "en")
        return transcribe(audio: audio, sampleRate: sampleRate, options: options)
    }

    public func transcribeWithLanguage(
        audio: [Float], sampleRate: Int, language: String?
    ) -> TranscriptionResult {
        let resolvedLanguage = language ?? "en"
        return TranscriptionResult(
            text: transcribe(audio: audio, sampleRate: sampleRate, language: resolvedLanguage),
            language: resolvedLanguage)
    }

    public func transcribe(
        audio: [Float],
        sampleRate: Int,
        options: CohereTranscribeDecodingOptions
    ) -> String {
        let resampled = sampleRate == inputSampleRate
            ? audio
            : AudioFileLoader.resample(audio, from: sampleRate, to: inputSampleRate)
        let maximumDuration = min(Double(config.maxAudioClipS), options.chunkDuration)
        let ranges = Self.energyChunkRanges(
            audio: resampled,
            sampleRate: inputSampleRate,
            maximumDuration: maximumDuration,
            boundaryContext: config.overlapChunkSecond,
            energyWindowSamples: config.minEnergyWindowSamples)
        let texts = ranges.map { range in
            let chunk = range.isEmpty ? [Float(0)] : Array(resampled[range])
            return transcribeChunk(
                chunk, language: options.language, maxTokens: options.maxTokens)
        }
        .filter { !$0.isEmpty }
        let code = options.language.lowercased().split(separator: "-").first.map(String.init)
        return texts.joined(separator: ["ja", "zh"].contains(code ?? "") ? "" : " ")
    }

    static func energyChunkRanges(
        audio: [Float],
        sampleRate: Int,
        maximumDuration: Double,
        boundaryContext: Double,
        energyWindowSamples: Int
    ) -> [Range<Int>] {
        let totalSamples = audio.count
        guard totalSamples > 0 else { return [0..<0] }
        let chunkSize = max(1, Int((maximumDuration * Double(sampleRate)).rounded()))
        let boundarySamples = max(1, Int((boundaryContext * Double(sampleRate)).rounded()))
        let energyWindow = max(1, energyWindowSamples)
        guard totalSamples > chunkSize else { return [0..<totalSamples] }

        var ranges: [Range<Int>] = []
        var start = 0
        while start < totalSamples {
            if start + chunkSize >= totalSamples {
                ranges.append(start..<totalSamples)
                break
            }
            let searchStart = max(start, start + chunkSize - boundarySamples)
            let searchEnd = min(start + chunkSize, totalSamples)
            let searchLength = searchEnd - searchStart
            var splitPoint = (searchStart + searchEnd) / 2
            if searchLength > energyWindow {
                let usableSamples = (searchLength / energyWindow) * energyWindow
                if usableSamples > 0 {
                    var quietestIndex = 0
                    var quietestEnergy = Float.greatestFiniteMagnitude
                    for windowIndex in 0..<(usableSamples / energyWindow) {
                        let windowStart = searchStart + windowIndex * energyWindow
                        var energy: Float = 0
                        for sample in audio[windowStart..<(windowStart + energyWindow)] {
                            energy += sample * sample
                        }
                        energy /= Float(energyWindow)
                        if energy < quietestEnergy {
                            quietestEnergy = energy
                            quietestIndex = windowIndex
                        }
                    }
                    splitPoint = searchStart + quietestIndex * energyWindow
                }
            }
            splitPoint = max(start + 1, min(splitPoint, totalSamples))
            ranges.append(start..<splitPoint)
            start = splitPoint
        }
        return ranges
    }

    private func transcribeChunk(_ audio: [Float], language: String, maxTokens: Int) -> String {
        var context = prefill(audio: audio, language: language)
        let maximum = min(
            max(0, maxTokens),
            max(0, config.decoder.maxSequenceLength - context.promptLength))
        var generated: [Int] = []
        generated.reserveCapacity(maximum)

        for position in context.promptLength..<(context.promptLength + maximum) {
            let token = context.logits.argMax(axis: -1).item(Int.self)
            if token == tokenizer.eosTokenID { break }
            generated.append(token)

            let next = decoder(
                inputIds: MLXArray([Int32(token)]).reshaped(1, 1),
                positions: MLXArray([Int32(position)]).reshaped(1, 1),
                encoderHiddenStates: context.encoderStates,
                selfAttentionMask: nil,
                crossAttentionMask: nil,
                cache: context.cache)
            context.cache = next.1
            context.logits = languageModelHead(next.0[0, -1])
            eval(context.logits)
        }
        Memory.clearCache()
        return tokenizer.decode(tokens: generated)
    }

    private func prefill(audio: [Float], language: String) -> CoherePrefillContext {
        let features = frontend.extract(audio)
        let encoded = encoder(features).0
        let encoderStates = bridgeProjection?(encoded) ?? encoded
        let prompt = tokenizer.buildPromptTokens(language: language)
        precondition(!prompt.isEmpty, "Cohere prompt tokens are missing")
        let length = prompt.count
        let inputIDs = MLXArray(prompt.map(Int32.init)).reshaped(1, length)
        let positions = MLXArray((0..<length).map(Int32.init)).reshaped(1, length)
        let mask = MultiHeadAttention.createAdditiveCausalMask(length).asType(encoderStates.dtype)
        let result = decoder(
            inputIds: inputIDs,
            positions: positions,
            encoderHiddenStates: encoderStates,
            selfAttentionMask: mask,
            crossAttentionMask: nil,
            cache: nil)
        let logits = languageModelHead(result.0[0, -1])
        var arrays = [logits]
        for layer in result.1.layers {
            arrays.append(contentsOf: [layer.selfKeys, layer.selfValues, layer.crossKeys, layer.crossValues].compactMap { $0 })
        }
        eval(arrays)
        return CoherePrefillContext(
            encoderStates: encoderStates,
            promptLength: length,
            logits: logits,
            cache: result.1)
    }
}

public extension CohereTranscribeModel {
    static func validateQuantization(_ quantization: CohereMLXQuantization?) throws {
        guard let quantization else { return }
        guard supportedQuantizationBits.contains(quantization.bits),
              quantization.groupSize > 0 else {
            throw CohereTranscribeError.invalidQuantization(
                bits: quantization.bits, groupSize: quantization.groupSize)
        }
    }

    static func fromDirectory(_ directory: URL) throws -> CohereTranscribeModel {
        let config = try JSONDecoder().decode(
            CohereTranscribeConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try validateQuantization(config.quantization)
        let tokenizer = try CohereTranscribeTokenizer(modelDirectory: directory)
        let model = try CohereTranscribeModel(config: config, tokenizer: tokenizer)
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw CohereTranscribeError.noWeights(directory) }

        var weights: [String: MLXArray] = [:]
        for file in files {
            weights.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        let normalized = normalizeCohereWeightKeys(weights)
        if let quantization = config.quantization {
            MLXNN.quantize(model: model) { path, _ in
                normalized["\(path).scales"] == nil
                    ? nil
                    : (quantization.groupSize, quantization.bits, .affine)
            }
        }
        do {
            try model.update(
                parameters: ModuleParameters.unflattened(normalized),
                verify: .all)
        } catch {
            throw CohereTranscribeError.incompatibleCheckpoint(error.localizedDescription)
        }
        model.train(false)
        eval(model)
        return model
    }

    static func load(
        _ modelPath: String = defaultModelId,
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> CohereTranscribeModel {
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
    ) async throws -> CohereTranscribeModel {
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
