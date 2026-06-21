import CoreML
import Foundation
import Hub
import Tokenizers

/// On-device FunctionGemma 270M tool-calling LLM, driving the unified
/// ANE-shaped CoreML model published as
/// `aufklarer/FunctionGemma-270M-CoreML-Palettize8` (or `-CoreML` fp16).
///
/// Pipeline:
///   1. Tokenize a prompt formatted with
///      ``FunctionGemmaPrompt.formatToolCallPrompt``, which mirrors the exact
///      developer / user / model-prefix layout the model was trained on
///      (including the `<start_function_declaration>` block).
///   2. Right-pad to ``cacheSeqLen`` (128), build prefill inputs, prefill once
///      against an `MLState` cache.
///   3. Decode greedily one token per step; each step writes to a one-hot
///      `write_mask` slot in the cache.
///   4. Stop on the first `<end_function_call>` / `<end_of_turn>` / EOS
///      token, then return the decoded text. Use ``FunctionGemmaParser``
///      to extract structured calls from it.
@available(iOS 18.0, macOS 15.0, *)
public final class FunctionGemma: @unchecked Sendable {

    public static let defaultModelId = "aufklarer/FunctionGemma-270M-CoreML-Palettize8"

    // MARK: - Architecture constants (must match the published model)

    public let cacheSeqLen: Int = 128
    public let headDim: Int = 256
    public let hiddenSize: Int = 640
    public let numLayers: Int = 18
    public static let fullRopeTheta: Double = 1_000_000
    public static let slidingRopeTheta: Double = 10_000
    public static let maskValue: Float16 = -1.0e4

    // MARK: - Loaded state

    private let model: MLModel
    public let tokenizer: Tokenizer

    private let cosFullAll: MLMultiArray
    private let sinFullAll: MLMultiArray
    private let cosSlidingAll: MLMultiArray
    private let sinSlidingAll: MLMultiArray

    private let stopIds: Set<Int>

    public struct Metrics {
        public var prefillMs: Double = 0
        public var decodeMs: Double = 0
        public var decodeTokens: Int = 0
        public var tokensPerSecond: Double {
            decodeMs > 0 ? Double(decodeTokens) / (decodeMs / 1000.0) : 0
        }
    }

    public private(set) var lastMetrics: Metrics = Metrics()

    // MARK: - Loading

    /// Load from a directory that holds the unzipped HuggingFace repo
    /// (`*.mlmodelc/`, `tokenizer.json`, `tokenizer_config.json`,
    /// `chat_template.jinja`, `config.json`).
    public static func load(from directory: URL,
                            modelFileName: String = "FunctionGemmaANEUnifiedStateful.mlmodelc",
                            computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> FunctionGemma {
        let modelURL = directory.appendingPathComponent(modelFileName)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw FunctionGemmaError.modelLoadFailed(
                "no compiled model at \(modelURL.path)")
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits
        let model: MLModel
        do {
            model = try MLModel(contentsOf: modelURL, configuration: cfg)
        } catch {
            throw FunctionGemmaError.modelLoadFailed("\(error)")
        }
        let tok: Tokenizer
        do {
            tok = try await AutoTokenizer.from(modelFolder: directory)
        } catch {
            throw FunctionGemmaError.tokenizerLoadFailed("\(error)")
        }
        return try FunctionGemma(model: model, tokenizer: tok)
    }

    /// Download the published HuggingFace repo into a local Hub cache and
    /// load it. Defaults to ``defaultModelId``.
    public static func loadFromHub(_ repoId: String = defaultModelId,
                                   computeUnits: MLComputeUnits = .cpuAndNeuralEngine) async throws -> FunctionGemma {
        let hub = HubApi()
        let folder = try await hub.snapshot(from: repoId)
        return try await load(from: folder, computeUnits: computeUnits)
    }

    private init(model: MLModel, tokenizer: Tokenizer) throws {
        self.model = model
        self.tokenizer = tokenizer
        do {
            self.cosFullAll = try Self.buildRopeTable(theta: Self.fullRopeTheta,
                                                     seqLen: 128, headDim: 256, useCos: true)
            self.sinFullAll = try Self.buildRopeTable(theta: Self.fullRopeTheta,
                                                     seqLen: 128, headDim: 256, useCos: false)
            self.cosSlidingAll = try Self.buildRopeTable(theta: Self.slidingRopeTheta,
                                                        seqLen: 128, headDim: 256, useCos: true)
            self.sinSlidingAll = try Self.buildRopeTable(theta: Self.slidingRopeTheta,
                                                        seqLen: 128, headDim: 256, useCos: false)
        } catch {
            throw FunctionGemmaError.modelLoadFailed("rope table init failed: \(error)")
        }
        // The function-call grammar terminates on `<end_function_call>` (id 49
        // in the public tokenizer) and `<end_of_turn>` (id 106). Token id 1
        // is the EOS the base model emits. We also stop on the BOS / pad.
        var ids = Set<Int>([1, 50, 106])
        if let eos = (tokenizer as? PreTrainedTokenizer)?.eosTokenId { ids.insert(eos) }
        self.stopIds = ids
    }

    // MARK: - Public API

    /// Greedy prefill + decode. Returns the decoded text **excluding** the
    /// prompt. Pass the prompt as the bare user text — the chat template is
    /// applied internally.
    public func generate(prompt: String,
                          tools: [FunctionDeclaration],
                          maxNewTokens: Int = 64) async throws -> String {
        let fullPrompt = FunctionGemmaPrompt.formatToolCallPrompt(tools: tools, userText: prompt)
        let promptIds = try tokenize(fullPrompt)
        return try await generateFromTokens(promptIds: promptIds, maxNewTokens: maxNewTokens)
    }

    /// Lower-level entry point: skip the chat template and tokenizer and
    /// drive the model from an already-prepared token sequence. Useful when
    /// the caller wants to construct multi-turn prompts manually.
    public func generateFromTokens(promptIds: [Int],
                                    maxNewTokens: Int) async throws -> String {
        guard promptIds.count <= cacheSeqLen else {
            throw FunctionGemmaError.promptTooLong(actual: promptIds.count, max: cacheSeqLen)
        }
        let state = model.makeState()

        var metrics = Metrics()
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let prefillOutput = try await runPrefill(promptIds: promptIds, state: state)
        metrics.prefillMs = (CFAbsoluteTimeGetCurrent() - prefillStart) * 1000

        var nextToken = argmax(prefillOutput)
        var generated: [Int] = [nextToken]

        let decodeStart = CFAbsoluteTimeGetCurrent()
        let promptLen = promptIds.count
        while generated.count < maxNewTokens {
            let position = promptLen + generated.count - 1
            if position >= cacheSeqLen { break }
            let logits = try await runDecodeStep(tokenId: nextToken, position: position, state: state)
            nextToken = argmax(logits)
            generated.append(nextToken)

            if stopIds.contains(nextToken) { break }
            let partial = tokenizer.decode(tokens: generated, skipSpecialTokens: false)
            if partial.contains("<end_function_call>") || partial.contains("<end_of_turn>") {
                break
            }
        }
        metrics.decodeMs = (CFAbsoluteTimeGetCurrent() - decodeStart) * 1000
        metrics.decodeTokens = generated.count
        self.lastMetrics = metrics

        return tokenizer.decode(tokens: generated, skipSpecialTokens: false)
    }

    /// Convenience: ``generate(...)`` plus parsing.
    public func generateCalls(prompt: String,
                               tools: [FunctionDeclaration],
                               maxNewTokens: Int = 64) async throws
        -> (text: String, calls: [FunctionCall]) {
        let text = try await generate(prompt: prompt, tools: tools, maxNewTokens: maxNewTokens)
        return (text, FunctionGemmaParser.parseFunctionCalls(text))
    }

    // MARK: - Tokenize

    private func tokenize(_ text: String) throws -> [Int] {
        let ids = tokenizer.encode(text: text)
        return ids
    }

    // MARK: - Prefill

    private func runPrefill(promptIds: [Int], state: MLState) async throws -> [Float] {
        let T = cacheSeqLen
        let inputIds = try MLMultiArray(shape: [1, NSNumber(value: T)], dataType: .int32)
        let inputPtr = inputIds.dataPointer.bindMemory(to: Int32.self, capacity: T)
        for i in 0..<T {
            inputPtr[i] = i < promptIds.count ? Int32(promptIds[i]) : 0
        }

        let attentionMask = try MLMultiArray(shape: [1, 1, NSNumber(value: T), NSNumber(value: T)],
                                              dataType: .float16)
        let attnPtr = attentionMask.dataPointer.bindMemory(to: Float16.self, capacity: T * T)
        let N = promptIds.count
        let unmasked: Float16 = 0
        for r in 0..<T {
            for c in 0..<T {
                if r < N {
                    attnPtr[r * T + c] = (c <= r) ? unmasked : Self.maskValue
                } else {
                    attnPtr[r * T + c] = (c == r) ? unmasked : Self.maskValue
                }
            }
        }

        let writeMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
        let wmPtr = writeMask.dataPointer.bindMemory(to: Float16.self, capacity: T)
        for i in 0..<T { wmPtr[i] = 1.0 }

        let logitsMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
        let lmPtr = logitsMask.dataPointer.bindMemory(to: Float16.self, capacity: T)
        for i in 0..<T { lmPtr[i] = (i == N - 1) ? 1.0 : 0.0 }

        let inputs: [String: MLFeatureValue] = [
            "input_ids":      .init(multiArray: inputIds),
            "cos_full":       .init(multiArray: cosFullAll),
            "sin_full":       .init(multiArray: sinFullAll),
            "cos_sliding":    .init(multiArray: cosSlidingAll),
            "sin_sliding":    .init(multiArray: sinSlidingAll),
            "write_mask":     .init(multiArray: writeMask),
            "attention_mask": .init(multiArray: attentionMask),
            "logits_mask":    .init(multiArray: logitsMask),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let output = try await model.prediction(from: provider, using: state)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw FunctionGemmaError.inferenceFailed("no logits output from prefill")
        }
        return Self.fp16ToFloats(logits)
    }

    // MARK: - Decode step

    private func runDecodeStep(tokenId: Int, position: Int, state: MLState) async throws -> [Float] {
        let T = cacheSeqLen
        let inputId = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputId.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(tokenId)

        let cosFull = try sliceRope(cosFullAll, at: position)
        let sinFull = try sliceRope(sinFullAll, at: position)
        let cosSliding = try sliceRope(cosSlidingAll, at: position)
        let sinSliding = try sliceRope(sinSlidingAll, at: position)

        let writeMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
        let wmPtr = writeMask.dataPointer.bindMemory(to: Float16.self, capacity: T)
        for i in 0..<T { wmPtr[i] = i == position ? 1.0 : 0.0 }

        let attnMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: T)], dataType: .float16)
        let amPtr = attnMask.dataPointer.bindMemory(to: Float16.self, capacity: T)
        for i in 0..<T { amPtr[i] = i <= position ? 0.0 : Self.maskValue }

        let logitsMask = try MLMultiArray(shape: [1, 1, 1, 1], dataType: .float16)
        logitsMask.dataPointer.bindMemory(to: Float16.self, capacity: 1)[0] = 1.0

        let inputs: [String: MLFeatureValue] = [
            "input_ids":      .init(multiArray: inputId),
            "cos_full":       .init(multiArray: cosFull),
            "sin_full":       .init(multiArray: sinFull),
            "cos_sliding":    .init(multiArray: cosSliding),
            "sin_sliding":    .init(multiArray: sinSliding),
            "write_mask":     .init(multiArray: writeMask),
            "attention_mask": .init(multiArray: attnMask),
            "logits_mask":    .init(multiArray: logitsMask),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let output = try await model.prediction(from: provider, using: state)
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw FunctionGemmaError.inferenceFailed("no logits output from decode")
        }
        return Self.fp16ToFloats(logits)
    }

    // MARK: - Tensor helpers

    private func sliceRope(_ table: MLMultiArray, at position: Int) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: [1, 1, NSNumber(value: headDim)], dataType: .float16)
        let src = table.dataPointer.bindMemory(to: Float16.self,
                                                capacity: Int(table.count))
        let dst = result.dataPointer.bindMemory(to: Float16.self, capacity: headDim)
        let srcOffset = position * headDim
        for i in 0..<headDim { dst[i] = src[srcOffset + i] }
        return result
    }

    private func argmax(_ logits: [Float]) -> Int {
        var bestIdx = 0
        var bestVal = logits.first ?? -Float.greatestFiniteMagnitude
        for i in 1..<logits.count where logits[i] > bestVal {
            bestVal = logits[i]
            bestIdx = i
        }
        return bestIdx
    }

    private static func fp16ToFloats(_ array: MLMultiArray) -> [Float] {
        let count = Int(array.count)
        let src = array.dataPointer.bindMemory(to: Float16.self, capacity: count)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = Float(src[i]) }
        return out
    }

    private static func buildRopeTable(theta: Double, seqLen: Int, headDim: Int,
                                        useCos: Bool) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1,
                                            NSNumber(value: seqLen),
                                            NSNumber(value: headDim)],
                                    dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: Float16.self,
                                             capacity: seqLen * headDim)
        let halfHeadDim = headDim / 2
        var invFreq = [Double](repeating: 0, count: halfHeadDim)
        for i in 0..<halfHeadDim {
            invFreq[i] = 1.0 / pow(theta, Double(2 * i) / Double(headDim))
        }
        for pos in 0..<seqLen {
            for i in 0..<halfHeadDim {
                let angle = Double(pos) * invFreq[i]
                let value = useCos ? cos(angle) : sin(angle)
                let f16 = Float16(value)
                dst[pos * headDim + i] = f16
                dst[pos * headDim + halfHeadDim + i] = f16  // mirrored RoPE convention
            }
        }
        return arr
    }
}

// MARK: - Helpers

private extension Array where Element == Int {
    func joinedString(tokenizer: Tokenizer) -> String {
        tokenizer.decode(tokens: self, skipSpecialTokens: false)
    }
}
