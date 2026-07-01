import AudioCommon
import CoreML
import Foundation

struct WhisperTranscription: Sendable {
    let text: String
    let language: String?
}

final class WhisperCoreMLRuntime: @unchecked Sendable {
    static let sampleRate = 16_000

    private let modelFolder: URL
    private let melModel: MLModel
    private let encoderModel: MLModel
    private let decoderPrefillModel: MLModel
    private let decoderModel: MLModel
    private let tokenizer: WhisperByteLevelTokenizer
    private let generationConfig: WhisperGenerationConfig

    private let maxAudioSamples = 480_000
    private let cacheDim: Int
    private let cacheLength: Int
    private let vocabSize: Int

    init(modelFolder: URL) throws {
        self.modelFolder = modelFolder
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .all

        self.melModel = try Self.loadModel("MelSpectrogram.mlmodelc", from: modelFolder, configuration: mlConfig)
        self.encoderModel = try Self.loadModel("AudioEncoder.mlmodelc", from: modelFolder, configuration: mlConfig)
        self.decoderPrefillModel = try Self.loadModel("TextDecoderContextPrefill.mlmodelc", from: modelFolder, configuration: mlConfig)
        self.decoderModel = try Self.loadModel("TextDecoder.mlmodelc", from: modelFolder, configuration: mlConfig)
        self.tokenizer = try WhisperByteLevelTokenizer(modelFolder: modelFolder)
        self.generationConfig = try WhisperGenerationConfig.load(from: modelFolder)

        let decoderInputs = decoderModel.modelDescription.inputDescriptionsByName
        let decoderOutputs = decoderModel.modelDescription.outputDescriptionsByName
        self.cacheDim = decoderInputs["key_cache"]?.multiArrayConstraint?.shape[safe: 1]?.intValue ?? 5_120
        self.cacheLength = decoderInputs["key_cache"]?.multiArrayConstraint?.shape[safe: 3]?.intValue ?? 224
        self.vocabSize = decoderOutputs["logits"]?.multiArrayConstraint?.shape.last?.intValue ?? 51_866
    }

    func transcribe(audio: [Float], languageHint: String?) throws -> WhisperTranscription {
        guard !audio.isEmpty else {
            return WhisperTranscription(text: "", language: languageHint)
        }

        var offset = 0
        var texts: [String] = []
        var resolvedLanguage: String?

        while offset < audio.count {
            let end = min(offset + maxAudioSamples, audio.count)
            let chunk = Array(audio[offset..<end])
            let result = try transcribeChunk(audio: chunk, languageHint: languageHint ?? resolvedLanguage)
            if !result.text.isEmpty {
                texts.append(result.text)
            }
            if resolvedLanguage == nil {
                resolvedLanguage = result.language
            }
            offset = end
        }

        return WhisperTranscription(
            text: texts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            language: resolvedLanguage ?? languageHint)
    }

    private func transcribeChunk(audio: [Float], languageHint: String?) throws -> WhisperTranscription {
        let audioArray = try makeAudioArray(audio)
        let melOutput = try melModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: audioArray),
        ]))
        guard let mel = melOutput.featureValue(for: "melspectrogram_features")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(operation: "Whisper mel spectrogram", reason: "Missing melspectrogram_features")
        }

        let encoderOutput = try encoderModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "melspectrogram_features": MLFeatureValue(multiArray: mel),
        ]))
        guard let encoderEmbeds = encoderOutput.featureValue(for: "encoder_output_embeds")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(operation: "Whisper audio encoder", reason: "Missing encoder_output_embeds")
        }

        let languageToken = try resolveLanguageToken(languageHint, encoderEmbeds: encoderEmbeds)
        let language = generationConfig.languageCode(forToken: languageToken)
        let state = try makeDecoderState(languageToken: languageToken)
        let generated = try decodeGreedy(encoderEmbeds: encoderEmbeds, state: state)

        return WhisperTranscription(
            text: tokenizer.decode(generated).trimmingCharacters(in: .whitespacesAndNewlines),
            language: language)
    }

    private func resolveLanguageToken(_ languageHint: String?, encoderEmbeds: MLMultiArray) throws -> Int {
        if let languageHint,
           let token = generationConfig.languageToken(for: languageHint) {
            return token
        }
        return try detectLanguageToken(encoderEmbeds: encoderEmbeds)
    }

    private func detectLanguageToken(encoderEmbeds: MLMultiArray) throws -> Int {
        let state = try makeEmptyDecoderState()
        state.inputIds[0] = NSNumber(value: Int32(generationConfig.startOfTranscriptToken))
        state.cacheLength[0] = NSNumber(value: Int32(0))

        setFloat16(state.kvCacheUpdateMask, at: 0, to: 1)
        setFloat16(state.decoderKeyPaddingMask, at: 0, to: 0)

        let output = try predictDecoder(encoderEmbeds: encoderEmbeds, state: state)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(operation: "Whisper language detection", reason: "Missing logits")
        }

        return argmax(logits: logits, allowed: generationConfig.languageTokenSet)
            ?? generationConfig.englishToken
    }

    private func makeDecoderState(languageToken: Int) throws -> DecoderState {
        let state = try makeEmptyDecoderState()

        let prefillInputs = try MLDictionaryFeatureProvider(dictionary: [
            "language": MLFeatureValue(multiArray: scalarInt32(languageToken)),
            "task": MLFeatureValue(multiArray: scalarInt32(generationConfig.transcribeTaskIndex)),
        ])
        let prefillOutput = try decoderPrefillModel.prediction(from: prefillInputs)
        guard let keyPrefill = prefillOutput.featureValue(for: "key_cache_prefill")?.multiArrayValue,
              let valuePrefill = prefillOutput.featureValue(for: "value_cache_prefill")?.multiArrayValue
        else {
            throw AudioModelError.inferenceFailed(operation: "Whisper decoder prefill", reason: "Missing prefill cache outputs")
        }

        copyCacheSlice(keyPrefill, into: state.keyCache, tokenCount: generationConfig.prefillCacheTokenCount)
        copyCacheSlice(valuePrefill, into: state.valueCache, tokenCount: generationConfig.prefillCacheTokenCount)

        for index in 0...generationConfig.prefillCacheTokenCount {
            setFloat16(state.decoderKeyPaddingMask, at: index, to: 0)
        }
        setFloat16(state.kvCacheUpdateMask, at: generationConfig.prefillCacheTokenCount, to: 1)
        return state
    }

    private func decodeGreedy(encoderEmbeds: MLMultiArray, state: DecoderState) throws -> [Int] {
        var generated: [Int] = []
        var nextToken = generationConfig.noTimestampsToken
        var tokenIndex = generationConfig.prefillCacheTokenCount
        let maxTokenIndex = cacheLength - 1

        while tokenIndex < maxTokenIndex {
            state.inputIds[0] = NSNumber(value: Int32(nextToken))
            state.cacheLength[0] = NSNumber(value: Int32(tokenIndex))

            let output = try predictDecoder(encoderEmbeds: encoderEmbeds, state: state)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                  let keyUpdate = output.featureValue(for: "key_cache_updates")?.multiArrayValue,
                  let valueUpdate = output.featureValue(for: "value_cache_updates")?.multiArrayValue
            else {
                throw AudioModelError.inferenceFailed(operation: "Whisper decoder", reason: "Missing decoder outputs")
            }

            let sampled = argmax(logits: logits, generatedTokenCount: generated.count)
            if sampled == generationConfig.endToken {
                break
            }

            if sampled < generationConfig.specialTokenBegin {
                generated.append(sampled)
            }

            copyCacheUpdate(keyUpdate, into: state.keyCache, at: tokenIndex)
            copyCacheUpdate(valueUpdate, into: state.valueCache, at: tokenIndex)

            setFloat16(state.decoderKeyPaddingMask, at: tokenIndex + 1, to: 0)
            setFloat16(state.kvCacheUpdateMask, at: tokenIndex, to: 0)
            setFloat16(state.kvCacheUpdateMask, at: tokenIndex + 1, to: 1)

            nextToken = sampled
            tokenIndex += 1
        }

        return generated
    }

    private func predictDecoder(encoderEmbeds: MLMultiArray, state: DecoderState) throws -> MLFeatureProvider {
        try decoderModel.prediction(from: MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: state.inputIds),
            "cache_length": MLFeatureValue(multiArray: state.cacheLength),
            "key_cache": MLFeatureValue(multiArray: state.keyCache),
            "value_cache": MLFeatureValue(multiArray: state.valueCache),
            "kv_cache_update_mask": MLFeatureValue(multiArray: state.kvCacheUpdateMask),
            "encoder_output_embeds": MLFeatureValue(multiArray: encoderEmbeds),
            "decoder_key_padding_mask": MLFeatureValue(multiArray: state.decoderKeyPaddingMask),
        ]))
    }

    private func makeEmptyDecoderState() throws -> DecoderState {
        let keyCache = try MLMultiArray(shape: [1, cacheDim as NSNumber, 1, cacheLength as NSNumber], dataType: .float16)
        let valueCache = try MLMultiArray(shape: [1, cacheDim as NSNumber, 1, cacheLength as NSNumber], dataType: .float16)
        let kvCacheUpdateMask = try MLMultiArray(shape: [1, cacheLength as NSNumber], dataType: .float16)
        let decoderKeyPaddingMask = try MLMultiArray(shape: [1, cacheLength as NSNumber], dataType: .float16)
        let inputIds = try MLMultiArray(shape: [1], dataType: .int32)
        let cacheLengthArray = try MLMultiArray(shape: [1], dataType: .int32)

        fillFloat16(keyCache, with: 0)
        fillFloat16(valueCache, with: 0)
        fillFloat16(kvCacheUpdateMask, with: 0)
        fillFloat16(decoderKeyPaddingMask, with: -10_000)
        inputIds[0] = NSNumber(value: Int32(0))
        cacheLengthArray[0] = NSNumber(value: Int32(0))

        return DecoderState(
            inputIds: inputIds,
            cacheLength: cacheLengthArray,
            keyCache: keyCache,
            valueCache: valueCache,
            kvCacheUpdateMask: kvCacheUpdateMask,
            decoderKeyPaddingMask: decoderKeyPaddingMask)
    }

    private func makeAudioArray(_ audio: [Float]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [maxAudioSamples as NSNumber], dataType: .float16)
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        ptr.update(repeating: Float16(0), count: array.count)
        for i in 0..<min(audio.count, maxAudioSamples) {
            ptr[i] = Float16(max(-1, min(1, audio[i])))
        }
        return array
    }

    private func scalarInt32(_ value: Int) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: Int32(value))
        return array
    }

    private func argmax(logits: MLMultiArray, generatedTokenCount: Int) -> Int {
        argmax(logits: logits) { token in
            if token == generationConfig.endToken {
                return generatedTokenCount == 0 && generationConfig.beginSuppressTokens.contains(token)
            }
            if token >= generationConfig.specialTokenBegin {
                return true
            }
            if generatedTokenCount == 0 && generationConfig.beginSuppressTokens.contains(token) {
                return true
            }
            return generationConfig.suppressTokens.contains(token)
        } ?? generationConfig.endToken
    }

    private func argmax(logits: MLMultiArray, allowed: Set<Int>) -> Int? {
        argmax(logits: logits) { token in
            !allowed.contains(token)
        }
    }

    private func argmax(logits: MLMultiArray, shouldSkip: (Int) -> Bool) -> Int? {
        let count = min(vocabSize, logits.shape.last?.intValue ?? logits.count)
        let stride = logits.strides.last?.intValue ?? 1
        var bestToken: Int?
        var bestValue = -Float.infinity

        switch logits.dataType {
        case .float16:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float16.self)
            for token in 0..<count where !shouldSkip(token) {
                let value = Float(ptr[token * stride])
                if value.isNaN { continue }
                if value > bestValue {
                    bestValue = value
                    bestToken = token
                }
            }
        case .float32:
            let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
            for token in 0..<count where !shouldSkip(token) {
                let value = ptr[token * stride]
                if value.isNaN { continue }
                if value > bestValue {
                    bestValue = value
                    bestToken = token
                }
            }
        default:
            for token in 0..<count where !shouldSkip(token) {
                let value = logits[token * stride].floatValue
                if value.isNaN { continue }
                if value > bestValue {
                    bestValue = value
                    bestToken = token
                }
            }
        }

        return bestToken
    }

    private func copyCacheSlice(_ source: MLMultiArray, into destination: MLMultiArray, tokenCount: Int) {
        let src = source.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = destination.dataPointer.assumingMemoryBound(to: Float16.self)
        let srcStrides = source.strides.map(\.intValue)
        let dstStrides = destination.strides.map(\.intValue)
        let dim = min(source.shape[safe: 1]?.intValue ?? cacheDim, cacheDim)
        let count = min(tokenCount, source.shape[safe: 3]?.intValue ?? tokenCount, cacheLength)

        for channel in 0..<dim {
            for position in 0..<count {
                let srcIndex = channel * srcStrides[1] + position * srcStrides[3]
                let dstIndex = channel * dstStrides[1] + position * dstStrides[3]
                dst[dstIndex] = src[srcIndex]
            }
        }
    }

    private func copyCacheUpdate(_ source: MLMultiArray, into destination: MLMultiArray, at position: Int) {
        let src = source.dataPointer.assumingMemoryBound(to: Float16.self)
        let dst = destination.dataPointer.assumingMemoryBound(to: Float16.self)
        let srcStrides = source.strides.map(\.intValue)
        let dstStrides = destination.strides.map(\.intValue)
        let dim = min(source.shape[safe: 1]?.intValue ?? cacheDim, cacheDim)

        for channel in 0..<dim {
            let srcIndex = channel * srcStrides[1]
            let dstIndex = channel * dstStrides[1] + position * dstStrides[3]
            dst[dstIndex] = src[srcIndex]
        }
    }

    private func fillFloat16(_ array: MLMultiArray, with value: Float16) {
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        ptr.update(repeating: value, count: array.count)
    }

    private func setFloat16(_ array: MLMultiArray, at index: Int, to value: Float16) {
        guard index >= 0, index < array.count else { return }
        let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
        ptr[index] = value
    }

    private static func loadModel(_ name: String, from folder: URL, configuration: MLModelConfiguration) throws -> MLModel {
        let url = folder.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw AudioModelError.modelLoadFailed(modelId: folder.lastPathComponent, reason: "Missing \(name)")
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }
}

private final class DecoderState {
    let inputIds: MLMultiArray
    let cacheLength: MLMultiArray
    let keyCache: MLMultiArray
    let valueCache: MLMultiArray
    let kvCacheUpdateMask: MLMultiArray
    let decoderKeyPaddingMask: MLMultiArray

    init(
        inputIds: MLMultiArray,
        cacheLength: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        kvCacheUpdateMask: MLMultiArray,
        decoderKeyPaddingMask: MLMultiArray
    ) {
        self.inputIds = inputIds
        self.cacheLength = cacheLength
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.kvCacheUpdateMask = kvCacheUpdateMask
        self.decoderKeyPaddingMask = decoderKeyPaddingMask
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
