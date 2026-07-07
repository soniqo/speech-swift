#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation
import MLX

enum ChatterboxFlashCoreMLBridge {
    private static let int8DataTypeRawValue = 0x20000 | 8

    static func loadCompiledModel(
        relativePath: String,
        in directory: URL,
        computeUnits: MLComputeUnits,
        name: String
    ) throws -> MLModel {
        let url = directory.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ChatterboxFlashCoreMLError.missingFile(relativePath)
        }
        return try CoreMLLoader.load(url: url, computeUnits: computeUnits, name: name)
    }

    static func float32(_ values: [Float], shape: [Int], label: String) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw ChatterboxFlashCoreMLError.invalidShape("\(label) expected \(count) values, got \(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
        ptr.update(from: values, count: count)
        return arr
    }

    static func int32(_ values: [Int32], shape: [Int], label: String) throws -> MLMultiArray {
        let count = shape.reduce(1, *)
        guard values.count == count else {
            throw ChatterboxFlashCoreMLError.invalidShape("\(label) expected \(count) values, got \(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
        ptr.update(from: values, count: count)
        return arr
    }

    static func scalarFloat32(_ value: Float) throws -> MLMultiArray {
        try float32([value], shape: [1], label: "scalar")
    }

    static func scalarInt32(_ value: Int32) throws -> MLMultiArray {
        try int32([value], shape: [1], label: "scalar")
    }

    static func toFloat32(_ arr: MLMultiArray) throws -> [Float] {
        let count = arr.count
        if arr.dataType.rawValue == int8DataTypeRawValue {
            let ptr = arr.dataPointer.bindMemory(to: Int8.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        }

        switch arr.dataType {
        case .float32:
            let ptr = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: ptr, count: count))
        case .float16:
            let ptr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Float(Float16(bitPattern: ptr[$0])) }
        case .double:
            let ptr = arr.dataPointer.bindMemory(to: Double.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        case .int32:
            let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: count)
            return (0..<count).map { Float(ptr[$0]) }
        default:
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("Unsupported MLMultiArray data type \(arr.dataType)")
        }
    }

    static func predict(_ model: MLModel, inputs: [String: MLMultiArray]) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        return try model.prediction(from: provider)
    }

    static func output(_ provider: MLFeatureProvider, named name: String) throws -> MLMultiArray {
        guard let arr = provider.featureValue(for: name)?.multiArrayValue else {
            throw ChatterboxFlashCoreMLError.missingOutput(name)
        }
        return arr
    }
}

public struct ChatterboxFlashS3GenReference: Sendable {
    public let embedding: [Float]
    public let promptToken: [Int32]
    public let promptFeature: [Float]
    public let promptFeatureFrames: Int

    public init(
        embedding: [Float],
        promptToken: [Int32],
        promptFeature: [Float],
        promptFeatureFrames: Int
    ) {
        self.embedding = embedding
        self.promptToken = promptToken
        self.promptFeature = promptFeature
        self.promptFeatureFrames = promptFeatureFrames
    }

    public init(_ reference: ChatterboxS3GenRef) {
        let embeddingArray = reference.xVector.reshaped([reference.xVector.size]).asType(.float32)
        let featureArray = reference.promptFeat.reshaped([reference.promptFeat.size]).asType(.float32)
        eval(embeddingArray, featureArray)
        self.embedding = embeddingArray.asArray(Float.self)
        self.promptToken = reference.promptToken.map { Int32($0) }
        self.promptFeature = featureArray.asArray(Float.self)
        self.promptFeatureFrames = reference.promptFeat.dim(1)
    }
}

public struct ChatterboxFlashT3Conditioning: Sendable {
    public let speakerEmbedding: [Float]
    public let promptSpeechTokens: [Int32]
    public let emotionAdv: Float

    public init(speakerEmbedding: [Float], promptSpeechTokens: [Int32], emotionAdv: Float = 0.5) {
        self.speakerEmbedding = speakerEmbedding
        self.promptSpeechTokens = promptSpeechTokens
        self.emotionAdv = emotionAdv
    }

    public init(speakerEmbedding: [Float], promptSpeechTokens: [Int], emotionAdv: Float = 0.5) {
        self.init(
            speakerEmbedding: speakerEmbedding,
            promptSpeechTokens: promptSpeechTokens.map(Int32.init),
            emotionAdv: emotionAdv
        )
    }
}

public final class ChatterboxFlashT3Graphs: @unchecked Sendable {
    public let config: ChatterboxFlashT3Config
    private let conditioningEncoder: MLModel
    private let textPrefill: MLModel
    private let blockDecoder: MLModel
    public let tokenizerURL: URL
    public let unconditionalBlockPriorURL: URL
    public let tokenizer: ChatterboxFlashTokenizer
    private let unconditionalBlockPrior: [Float]

    public init(
        directory: URL,
        config: ChatterboxFlashT3Config,
        computeUnits: MLComputeUnits
    ) throws {
        self.config = config
        self.conditioningEncoder = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "t3/ConditioningEncoder.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.ConditioningEncoder"
        )
        self.textPrefill = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "t3/TextPrefill.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.TextPrefill"
        )
        self.blockDecoder = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "t3/BlockDecoder.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.BlockDecoder"
        )

        self.tokenizerURL = directory.appendingPathComponent("t3/tokenizer.json")
        self.unconditionalBlockPriorURL = directory.appendingPathComponent("t3/uncond_block_prior.npy")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw ChatterboxFlashCoreMLError.missingFile("t3/tokenizer.json")
        }
        guard FileManager.default.fileExists(atPath: unconditionalBlockPriorURL.path) else {
            throw ChatterboxFlashCoreMLError.missingFile("t3/uncond_block_prior.npy")
        }
        self.tokenizer = try ChatterboxFlashTokenizer(tokenizerURL: tokenizerURL)
        let prior = try ChatterboxFlashNumpy.loadFloat32Vector(from: unconditionalBlockPriorURL)
        guard prior.count == config.speechVocabSize else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "unconditional prior must contain \(config.speechVocabSize) values, got \(prior.count)"
            )
        }
        self.unconditionalBlockPrior = prior
    }

    public func encodeConditioning(_ conditioning: ChatterboxFlashT3Conditioning) throws -> MLMultiArray {
        guard conditioning.speakerEmbedding.count == 256 else {
            throw ChatterboxFlashCoreMLError.invalidShape("speaker embedding must contain 256 floats")
        }
        guard conditioning.promptSpeechTokens.count == 150 else {
            throw ChatterboxFlashCoreMLError.invalidShape("prompt speech tokens must contain 150 ids")
        }

        let provider = try ChatterboxFlashCoreMLBridge.predict(
            conditioningEncoder,
            inputs: [
                "speaker_emb": ChatterboxFlashCoreMLBridge.float32(
                    conditioning.speakerEmbedding,
                    shape: [1, 256],
                    label: "speaker_emb"
                ),
                "prompt_speech_tokens": ChatterboxFlashCoreMLBridge.int32(
                    conditioning.promptSpeechTokens,
                    shape: [1, 150],
                    label: "prompt_speech_tokens"
                ),
                "emotion_adv": ChatterboxFlashCoreMLBridge.float32(
                    [conditioning.emotionAdv],
                    shape: [1, 1, 1],
                    label: "emotion_adv"
                ),
            ]
        )
        return try ChatterboxFlashCoreMLBridge.output(provider, named: "cond_emb")
    }

    public func prefill(textTokenIds: [Int32], conditioningEmbedding: MLMultiArray) throws -> MLFeatureProvider {
        guard textTokenIds.count == config.textLen else {
            throw ChatterboxFlashCoreMLError.invalidShape("text tokens must contain \(config.textLen) ids")
        }
        return try ChatterboxFlashCoreMLBridge.predict(
            textPrefill,
            inputs: [
                "text_token_ids": ChatterboxFlashCoreMLBridge.int32(
                    textTokenIds,
                    shape: [1, config.textLen],
                    label: "text_token_ids"
                ),
                "cond_emb": conditioningEmbedding,
            ]
        )
    }

    public func decodeBlock(
        speechTokenIds: [Int32],
        speechPositionIds: [Int32],
        shiftContext: MLMultiArray,
        keyCache: MLMultiArray,
        valueCache: MLMultiArray,
        keyPaddingMask: [Float],
        blockCacheMap: [Float]
    ) throws -> MLFeatureProvider {
        guard speechTokenIds.count == config.blockSize else {
            throw ChatterboxFlashCoreMLError.invalidShape("speech tokens must contain \(config.blockSize) ids")
        }
        guard speechPositionIds.count == config.blockSize else {
            throw ChatterboxFlashCoreMLError.invalidShape("speech positions must contain \(config.blockSize) ids")
        }
        guard keyPaddingMask.count == config.maxSeq else {
            throw ChatterboxFlashCoreMLError.invalidShape("key padding mask must contain \(config.maxSeq) values")
        }
        guard blockCacheMap.count == config.blockSize * config.maxSeq else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "block cache map must contain \(config.blockSize * config.maxSeq) values"
            )
        }

        return try ChatterboxFlashCoreMLBridge.predict(
            blockDecoder,
            inputs: [
                "speech_token_ids": ChatterboxFlashCoreMLBridge.int32(
                    speechTokenIds,
                    shape: [1, config.blockSize],
                    label: "speech_token_ids"
                ),
                "speech_pos_ids": ChatterboxFlashCoreMLBridge.int32(
                    speechPositionIds,
                    shape: [1, config.blockSize],
                    label: "speech_pos_ids"
                ),
                "shift_ctx": shiftContext,
                "key_cache": keyCache,
                "value_cache": valueCache,
                "key_padding_mask": ChatterboxFlashCoreMLBridge.float32(
                    keyPaddingMask,
                    shape: [1, config.maxSeq],
                    label: "key_padding_mask"
                ),
                "block_cache_map": ChatterboxFlashCoreMLBridge.float32(
                    blockCacheMap,
                    shape: [config.blockSize, config.maxSeq],
                    label: "block_cache_map"
                ),
            ]
        )
    }

    public func generateSpeechTokens(
        text: String,
        conditioning: ChatterboxFlashT3Conditioning,
        options: ChatterboxFlashGenerationOptions = ChatterboxFlashGenerationOptions()
    ) throws -> [Int] {
        try validateGenerationOptions(options)
        let textTokenIds = try tokenizer.encodePadded(text, config: config)
        let encodedTextCount = tokenizer.encode(text, addSpecialTokens: true, config: config).count
        let maxByCache = max(0, config.maxSeq - config.prefixLen)
        let requested = options.maxSpeechTokens ?? max(300, encodedTextCount * 6)
        let totalSpeechLen = min(maxByCache, requested)
        guard totalSpeechLen > 0 else { return [] }

        let conditioningEmbedding = try encodeConditioning(conditioning)
        let prefillProvider = try prefill(textTokenIds: textTokenIds, conditioningEmbedding: conditioningEmbedding)
        var shiftContext = try ChatterboxFlashCoreMLBridge.output(prefillProvider, named: "shift_ctx")
        var keyCache = try ChatterboxFlashCoreMLBridge.output(prefillProvider, named: "key_cache")
        var valueCache = try ChatterboxFlashCoreMLBridge.output(prefillProvider, named: "value_cache")

        var rng = ChatterboxFlashGaussianRNG(seed: options.seed)
        var speech = Array(repeating: config.maskTokenId, count: totalSpeechLen)

        var blockStart = 0
        while blockStart < totalSpeechLen {
            let blockLen = min(config.blockSize, totalSpeechLen - blockStart)
            let schedule = Self.omnivoiceUnmaskSchedule(
                nTotalMask: blockLen,
                numSteps: options.numSteps,
                tShift: options.omnivoiceScheduleTShift
            )

            var rowHitEOS = false
            for step in 0..<options.numSteps {
                let provider = try decodeBlock(
                    speechTokenIds: blockTokenIds(speech, start: blockStart, length: blockLen),
                    speechPositionIds: blockPositionIds(start: blockStart),
                    shiftContext: shiftContext,
                    keyCache: keyCache,
                    valueCache: valueCache,
                    keyPaddingMask: keyPaddingMask(blockStart: blockStart, blockLen: blockLen),
                    blockCacheMap: blockCacheMap(blockStart: blockStart, blockLen: blockLen)
                )
                let logits = try ChatterboxFlashCoreMLBridge.toFloat32(
                    ChatterboxFlashCoreMLBridge.output(provider, named: "logits")
                )
                guard logits.count == config.blockSize * config.speechVocabSize else {
                    throw ChatterboxFlashCoreMLError.invalidShape(
                        "logits expected \(config.blockSize * config.speechVocabSize) values, got \(logits.count)"
                    )
                }
                let result = try denoiseBlockStep(
                    logits: logits,
                    tokens: Array(speech[blockStart ..< blockStart + blockLen]),
                    blockLen: blockLen,
                    step: step,
                    numSteps: options.numSteps,
                    scheduleCount: schedule[step],
                    temperature: options.temperature,
                    timeShiftTau: options.timeShiftTau,
                    positionTemperature: options.positionTemperature,
                    rng: &rng
                )
                for offset in 0..<blockLen {
                    speech[blockStart + offset] = result.tokens[offset]
                }
                rowHitEOS = result.hitEOS
                if rowHitEOS || result.remainingMasks == 0 {
                    break
                }
            }

            if rowHitEOS {
                if let eosOffset = speech[blockStart ..< blockStart + blockLen]
                    .firstIndex(of: config.stopSpeechToken) {
                    return filterSpeechTokens(Array(speech[..<eosOffset]))
                }
                return filterSpeechTokens(speech)
            }

            let isLastBlock = blockStart + blockLen >= totalSpeechLen
            if !isLastBlock {
                let finalizeProvider = try decodeBlock(
                    speechTokenIds: blockTokenIds(speech, start: blockStart, length: blockLen),
                    speechPositionIds: blockPositionIds(start: blockStart),
                    shiftContext: shiftContext,
                    keyCache: keyCache,
                    valueCache: valueCache,
                    keyPaddingMask: keyPaddingMask(blockStart: blockStart, blockLen: blockLen),
                    blockCacheMap: blockCacheMap(blockStart: blockStart, blockLen: blockLen)
                )
                keyCache = try ChatterboxFlashCoreMLBridge.output(finalizeProvider, named: "new_key_cache")
                valueCache = try ChatterboxFlashCoreMLBridge.output(finalizeProvider, named: "new_value_cache")
                let blockHidden = try ChatterboxFlashCoreMLBridge.output(finalizeProvider, named: "block_hidden")
                shiftContext = try sliceHidden(blockHidden, row: blockLen - 1)
            }

            blockStart += blockLen
        }

        return filterSpeechTokens(speech)
    }

    private func validateGenerationOptions(_ options: ChatterboxFlashGenerationOptions) throws {
        guard options.numSteps > 0 else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("numSteps must be greater than zero")
        }
        guard options.temperature >= 0 else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("temperature must be non-negative")
        }
        guard options.cfgScale == 0 else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration(
                "cfgScale > 0 requires a zero-text null prefill graph that is not included in this export"
            )
        }
    }

    private func blockTokenIds(_ speech: [Int], start: Int, length: Int) -> [Int32] {
        var ids = Array(repeating: Int32(config.maskTokenId), count: config.blockSize)
        for offset in 0..<length {
            ids[offset] = Int32(speech[start + offset])
        }
        return ids
    }

    private func blockPositionIds(start: Int) -> [Int32] {
        (0..<config.blockSize).map { Int32(start + $0) }
    }

    private func keyPaddingMask(blockStart: Int, blockLen: Int) -> [Float] {
        let validCount = min(config.maxSeq, config.prefixLen + blockStart + blockLen)
        var mask = Array(repeating: Float(-1.0e4), count: config.maxSeq)
        for index in 0..<validCount {
            mask[index] = 0
        }
        return mask
    }

    private func blockCacheMap(blockStart: Int, blockLen: Int) -> [Float] {
        var cacheMap = Array(repeating: Float(0), count: config.blockSize * config.maxSeq)
        for row in 0..<blockLen {
            let column = config.prefixLen + blockStart + row
            if column < config.maxSeq {
                cacheMap[row * config.maxSeq + column] = 1
            }
        }
        return cacheMap
    }

    private struct DenoiseResult {
        let tokens: [Int]
        let remainingMasks: Int
        let hitEOS: Bool
    }

    private func denoiseBlockStep(
        logits: [Float],
        tokens: [Int],
        blockLen: Int,
        step: Int,
        numSteps: Int,
        scheduleCount: Int,
        temperature: Float,
        timeShiftTau: Float,
        positionTemperature: Float,
        rng: inout ChatterboxFlashGaussianRNG
    ) throws -> DenoiseResult {
        var sampled = Array(repeating: 0, count: blockLen)
        var probabilities = Array(repeating: Array<Float>(), count: blockLen)

        for row in 0..<blockLen {
            let start = row * config.speechVocabSize
            let end = start + config.speechVocabSize
            let rowLogits = Array(logits[start..<end])
            probabilities[row] = Self.softmax(rowLogits)
            sampled[row] = Self.sample(logits: rowLogits, temperature: temperature, rng: &rng)
        }

        let maskedIndexes = (0..<blockLen).filter { tokens[$0] == config.maskTokenId }
        if maskedIndexes.isEmpty {
            return DenoiseResult(tokens: tokens, remainingMasks: 0, hitEOS: tokens.contains(config.stopSpeechToken))
        }

        var pmi = Array(repeating: -Float.greatestFiniteMagnitude, count: blockLen)
        for row in maskedIndexes {
            let token = sampled[row]
            let probability = probabilities[row][token]
            let prior = unconditionalBlockPrior[token]
            pmi[row] = log(max(probability, 1.0e-8)) - log(max(prior, 1.0e-8))
            if positionTemperature > 0 {
                pmi[row] += Self.gumbel(rng: &rng)
            }
        }

        var shouldUnmask = Array(repeating: false, count: blockLen)
        if step == numSteps - 1 {
            for row in maskedIndexes {
                shouldUnmask[row] = true
            }
        } else {
            let currentMaskCount = maskedIndexes.count
            let schedule = min(max(0, scheduleCount), currentMaskCount)
            let quantileCount: Int
            if timeShiftTau > 0 {
                let q = max(Float(0), Float(1) - timeShiftTau * Float(step + 1) / Float(numSteps))
                let threshold = Self.quantile(maskedIndexes.map { pmi[$0] }, q: q)
                quantileCount = maskedIndexes.filter { pmi[$0] >= threshold }.count
            } else {
                quantileCount = 0
            }
            let unmaskCount = min(max(schedule, quantileCount), currentMaskCount)
            if unmaskCount >= currentMaskCount {
                for row in maskedIndexes {
                    shouldUnmask[row] = true
                }
            } else if unmaskCount > 0 {
                for row in maskedIndexes.sorted(by: { pmi[$0] > pmi[$1] }).prefix(unmaskCount) {
                    shouldUnmask[row] = true
                }
            }
        }

        var next = tokens
        for row in 0..<blockLen where shouldUnmask[row] {
            next[row] = sampled[row]
        }
        let remaining = next.filter { $0 == config.maskTokenId }.count
        return DenoiseResult(tokens: next, remainingMasks: remaining, hitEOS: next.contains(config.stopSpeechToken))
    }

    private func filterSpeechTokens(_ tokens: [Int]) -> [Int] {
        let prefix = if let stop = tokens.firstIndex(of: config.stopSpeechToken) {
            Array(tokens[..<stop])
        } else {
            tokens
        }
        return prefix.filter { $0 < config.startSpeechToken }
    }

    private func sliceHidden(_ hidden: MLMultiArray, row: Int) throws -> MLMultiArray {
        let values = try ChatterboxFlashCoreMLBridge.toFloat32(hidden)
        let start = row * config.hiddenSize
        let end = start + config.hiddenSize
        guard row >= 0, end <= values.count else {
            throw ChatterboxFlashCoreMLError.invalidShape("block_hidden does not contain row \(row)")
        }
        return try ChatterboxFlashCoreMLBridge.float32(
            Array(values[start..<end]),
            shape: [1, 1, config.hiddenSize],
            label: "shift_ctx"
        )
    }

    private static func omnivoiceUnmaskSchedule(nTotalMask: Int, numSteps: Int, tShift: Float) -> [Int] {
        let steps = max(1, numSteps)
        let total = max(0, nTotalMask)
        if total == 0 { return Array(repeating: 0, count: steps) }
        if steps == 1 { return [total] }
        if tShift <= 0 {
            var output = Array(repeating: total / steps, count: steps)
            output[steps - 1] += total - output.reduce(0, +)
            return output
        }

        var counts: [Int] = []
        counts.reserveCapacity(steps)
        var cumulative = 0
        var remaining = total
        for step in 0..<steps {
            let count: Int
            if step == steps - 1 {
                count = remaining
            } else {
                let s = Float(step + 1) / Float(steps)
                let shifted = tShift * s / (1 + (tShift - 1) * s)
                let target = Int(round(Float(total) * shifted))
                count = min(max(0, target - cumulative), remaining)
            }
            counts.append(count)
            cumulative += count
            remaining -= count
        }
        return counts
    }

    private static func softmax(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        var total = Float(0)
        var values = Array(repeating: Float(0), count: logits.count)
        for index in logits.indices {
            let value = exp(logits[index] - maxLogit)
            values[index] = value
            total += value
        }
        guard total > 0, total.isFinite else {
            return Array(repeating: Float(1) / Float(max(1, logits.count)), count: logits.count)
        }
        for index in values.indices {
            values[index] /= total
        }
        return values
    }

    private static func sample(
        logits: [Float],
        temperature: Float,
        rng: inout ChatterboxFlashGaussianRNG
    ) -> Int {
        if temperature <= 0 {
            return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
        }
        let probabilities = softmax(logits.map { $0 / temperature })
        let target = rng.uniform()
        var cumulative = Float(0)
        for index in probabilities.indices {
            cumulative += probabilities[index]
            if target <= cumulative {
                return index
            }
        }
        return probabilities.count - 1
    }

    private static func quantile(_ values: [Float], q: Float) -> Float {
        guard !values.isEmpty else { return 0 }
        if values.count == 1 { return values[0] }
        let sorted = values.sorted()
        let clamped = min(max(q, 0), 1)
        let position = Float(sorted.count - 1) * clamped
        let lower = Int(floor(position))
        let upper = Int(ceil(position))
        if lower == upper { return sorted[lower] }
        let fraction = position - Float(lower)
        return sorted[lower] * (1 - fraction) + sorted[upper] * fraction
    }

    private static func gumbel(rng: inout ChatterboxFlashGaussianRNG) -> Float {
        let u = min(max(rng.uniform(), 1.0e-10), 1 - 1.0e-10)
        return -log(-log(u))
    }
}

public final class ChatterboxFlashAudioGraphs: @unchecked Sendable {
    public let config: ChatterboxFlashAudioConfig
    private let speakerProjector: MLModel
    private let flowEncoder: MLModel
    private let flowEstimator: MLModel
    private let vocoder: MLModel

    public init(
        directory: URL,
        config: ChatterboxFlashAudioConfig,
        computeUnits: MLComputeUnits
    ) throws {
        self.config = config
        self.speakerProjector = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "audio/FlowSpeakerProjector.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.FlowSpeakerProjector"
        )
        self.flowEncoder = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "audio/FlowEncoder.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.FlowEncoder"
        )
        self.flowEstimator = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "audio/FlowEstimator.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.FlowEstimator"
        )
        self.vocoder = try ChatterboxFlashCoreMLBridge.loadCompiledModel(
            relativePath: "audio/HiFTVocoder.mlmodelc",
            in: directory,
            computeUnits: computeUnits,
            name: "ChatterboxFlash.HiFTVocoder"
        )
    }

    public func synthesize(
        speechTokens: [Int],
        reference: ChatterboxFlashS3GenReference,
        seed: UInt64 = 0
    ) throws -> [Float] {
        try validate(speechTokens: speechTokens, reference: reference)

        let totalTokenCount = reference.promptToken.count + speechTokens.count
        let totalMelFrames = totalTokenCount * config.tokenMelRatio
        let speechMelFrames = totalMelFrames - reference.promptFeatureFrames
        let tokenIds = paddedTokenIds(prompt: reference.promptToken, speech: speechTokens)

        let speakerProvider = try ChatterboxFlashCoreMLBridge.predict(
            speakerProjector,
            inputs: [
                "embedding": ChatterboxFlashCoreMLBridge.float32(
                    reference.embedding,
                    shape: [1, config.refEmbeddingDim],
                    label: "embedding"
                )
            ]
        )
        let speaker = try ChatterboxFlashCoreMLBridge.output(speakerProvider, named: "spks")

        let encoderProvider = try ChatterboxFlashCoreMLBridge.predict(
            flowEncoder,
            inputs: [
                "token_ids": ChatterboxFlashCoreMLBridge.int32(
                    tokenIds,
                    shape: [1, config.tokenLen],
                    label: "token_ids"
                ),
                "token_len": ChatterboxFlashCoreMLBridge.scalarInt32(Int32(totalTokenCount)),
            ]
        )
        let mu = try ChatterboxFlashCoreMLBridge.output(encoderProvider, named: "mu")
        let mask = try ChatterboxFlashCoreMLBridge.output(encoderProvider, named: "mask")

        let cond = try ChatterboxFlashCoreMLBridge.float32(
            conditioningMel(reference: reference),
            shape: [1, 80, config.melLen],
            label: "cond"
        )
        var rng = ChatterboxFlashGaussianRNG(seed: seed)
        var x = rng.normal(count: 80 * config.melLen)

        for step in [(t: Float(0.0), r: Float(0.5), dt: Float(0.5)), (t: Float(0.5), r: Float(1.0), dt: Float(0.5))] {
            let provider = try ChatterboxFlashCoreMLBridge.predict(
                flowEstimator,
                inputs: [
                    "x": ChatterboxFlashCoreMLBridge.float32(x, shape: [1, 80, config.melLen], label: "x"),
                    "mask": mask,
                    "mu": mu,
                    "t": ChatterboxFlashCoreMLBridge.scalarFloat32(step.t),
                    "spks": speaker,
                    "cond": cond,
                    "r": ChatterboxFlashCoreMLBridge.scalarFloat32(step.r),
                ]
            )
            let dxdt = try ChatterboxFlashCoreMLBridge.toFloat32(
                ChatterboxFlashCoreMLBridge.output(provider, named: "dxdt")
            )
            guard dxdt.count == x.count else {
                throw ChatterboxFlashCoreMLError.invalidShape("dxdt expected \(x.count) values, got \(dxdt.count)")
            }
            for i in x.indices {
                x[i] += step.dt * dxdt[i]
            }
        }

        let generatedMel = croppedGeneratedMel(x, startFrame: reference.promptFeatureFrames, frameCount: speechMelFrames)
        let vocoderProvider = try ChatterboxFlashCoreMLBridge.predict(
            vocoder,
            inputs: [
                "mel": ChatterboxFlashCoreMLBridge.float32(
                    generatedMel,
                    shape: [1, 80, config.melLen],
                    label: "mel"
                )
            ]
        )
        let audio = try ChatterboxFlashCoreMLBridge.toFloat32(
            ChatterboxFlashCoreMLBridge.output(vocoderProvider, named: "audio")
        )
        let sampleCount = min(audio.count, max(0, speechMelFrames * config.samplesPerMelFrame))
        return Array(audio.prefix(sampleCount))
    }

    private func validate(speechTokens: [Int], reference: ChatterboxFlashS3GenReference) throws {
        guard reference.embedding.count == config.refEmbeddingDim else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "reference embedding must contain \(config.refEmbeddingDim) floats"
            )
        }
        guard reference.promptFeatureFrames >= 0 else {
            throw ChatterboxFlashCoreMLError.invalidShape("prompt feature frame count must be non-negative")
        }
        guard reference.promptFeature.count == reference.promptFeatureFrames * 80 else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "prompt features must contain promptFeatureFrames * 80 values"
            )
        }
        let totalTokenCount = reference.promptToken.count + speechTokens.count
        guard totalTokenCount <= config.tokenLen else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "total token count \(totalTokenCount) exceeds exported limit \(config.tokenLen)"
            )
        }
        let totalMelFrames = totalTokenCount * config.tokenMelRatio
        guard totalMelFrames <= config.melLen else {
            throw ChatterboxFlashCoreMLError.invalidShape(
                "total mel frames \(totalMelFrames) exceeds exported limit \(config.melLen)"
            )
        }
        guard totalMelFrames >= reference.promptFeatureFrames else {
            throw ChatterboxFlashCoreMLError.invalidShape("prompt feature frames exceed generated mel length")
        }
    }

    private func paddedTokenIds(prompt: [Int32], speech: [Int]) -> [Int32] {
        var tokenIds = Array(repeating: Int32(0), count: config.tokenLen)
        var offset = 0
        for token in prompt {
            tokenIds[offset] = token
            offset += 1
        }
        for token in speech {
            tokenIds[offset] = Int32(token)
            offset += 1
        }
        return tokenIds
    }

    private func conditioningMel(reference: ChatterboxFlashS3GenReference) -> [Float] {
        var cond = Array(repeating: Float(0), count: 80 * config.melLen)
        for frame in 0..<reference.promptFeatureFrames {
            for channel in 0..<80 {
                cond[channel * config.melLen + frame] = reference.promptFeature[frame * 80 + channel]
            }
        }
        return cond
    }

    private func croppedGeneratedMel(_ mel: [Float], startFrame: Int, frameCount: Int) -> [Float] {
        var cropped = Array(repeating: Float(0), count: 80 * config.melLen)
        guard frameCount > 0 else { return cropped }
        let framesToCopy = min(frameCount, config.melLen)
        for channel in 0..<80 {
            for frame in 0..<framesToCopy {
                cropped[channel * config.melLen + frame] = mel[channel * config.melLen + startFrame + frame]
            }
        }
        return cropped
    }
}

struct ChatterboxFlashGaussianRNG {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    mutating func normal(count: Int) -> [Float] {
        var values: [Float] = []
        values.reserveCapacity(count)
        while values.count < count {
            let u1 = max(Float.leastNonzeroMagnitude, uniform())
            let u2 = uniform()
            let radius = sqrt(-2.0 * log(u1))
            let angle = 2.0 * Float.pi * u2
            values.append(radius * cos(angle))
            if values.count < count {
                values.append(radius * sin(angle))
            }
        }
        return values
    }

    mutating func uniform() -> Float {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return Float(z >> 40) / Float(1 << 24)
    }
}
#endif
