import Foundation
import CoreML
import AudioCommon
import Hub
import Tokenizers

/// Inkling generation through shared INT8 Core ML partitions and visit-local caches.
public final class Inkling2CoreMLChat: @unchecked Sendable, Qwen35ChatBackend {
    public let denseConfig: Inkling2Config
    public let tokenizer: ChatTokenizer
    public let contextLimit: Int
    private let model: Inkling2CoreMLModel
    private let nativeTokenizer: any Tokenizer
    private let generationLock = NSLock()

    private init(model: Inkling2CoreMLModel, native: any Tokenizer, tokenizer: ChatTokenizer) {
        self.model = model; denseConfig = model.config; contextLimit = model.contextLimit
        nativeTokenizer = native; self.tokenizer = tokenizer
    }

    public static func fromDirectory(_ directory: URL, contextLimit: Int = 8192,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: ((Double, String) -> Void)? = nil) throws -> Inkling2CoreMLChat {
        guard contextLimit > 0 else { throw Inkling2Error.invalid("context limit must be positive") }
        let model = try Inkling2CoreMLModel(directory: directory, contextLimit: contextLimit,
                                           computeUnits: computeUnits, progressHandler: progressHandler)
        guard var tc = try JSONSerialization.jsonObject(with: Data(contentsOf:
                directory.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any],
              let td = try JSONSerialization.jsonObject(with: Data(contentsOf:
                directory.appendingPathComponent("tokenizer.json"))) as? [NSString: Any] else {
            throw Inkling2Error.invalid("invalid tokenizer files")
        }
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        let native = try AutoTokenizer.from(tokenizerConfig: Config(tc), tokenizerData: Config(td))
        let compatibility = ChatTokenizer()
        try compatibility.load(from: directory)
        compatibility.eosTokenId = model.config.eosTokenID
        return Inkling2CoreMLChat(model: model, native: native, tokenizer: compatibility)
    }

    public func encode(_ text: String) -> [Int] { nativeTokenizer.encode(text: text, addSpecialTokens: false) }
    public func decode(_ tokens: [Int]) -> String { nativeTokenizer.decode(tokens: tokens, skipSpecialTokens: false) }
    public func resetState() {} // Requests own their caches; model weights remain reusable.

    private func validate(_ tokens: [Int], maxNewTokens: Int) throws {
        guard !tokens.isEmpty, maxNewTokens >= 0, tokens.count <= contextLimit - maxNewTokens,
              tokens.allSatisfy({ (0..<denseConfig.vocabSize).contains($0) }) else {
            throw Inkling2Error.invalid("invalid token IDs or prompt plus generation exceeds the Core ML context limit")
        }
    }

    private func prefill(_ tokens: [Int], imageFeatures: [[Float]] = [],
                         cancellation: InferenceCancellationToken? = nil) throws -> ([Float], Inkling2CoreMLModel.State) {
        var state = model.initialState(), last: [Float] = []
        let features = imageFeatures.flatMap { $0 }, width = denseConfig.hiddenSize
        if !features.isEmpty {
            guard features.count == tokens.filter({ $0 == 200054 }).count * width else {
                throw Inkling2Error.invalid("image placeholders do not match Core ML image features")
            }
        }
        var imageOffset = 0
        for start in stride(from: 0, to: tokens.count, by: model.manifest.prefillSize) {
            try cancellation?.checkCancellation()
            let end = min(start + model.manifest.prefillSize, tokens.count)
            let chunk = Array(tokens[start..<end])
            var embedded: MLMultiArray?
            if !features.isEmpty && chunk.contains(200054) {
                let base = try model.embeddings(chunk)
                var values = try InklingCoreMLTensor.floats(base)
                for (offset, token) in chunk.enumerated() where token == 200054 {
                    values.replaceSubrange((offset * width)..<((offset + 1) * width),
                                           with: features[imageOffset..<(imageOffset + width)])
                    imageOffset += width
                }
                embedded = try InklingCoreMLTensor.make([1, chunk.count, width],
                    type: model.manifest.activationType, values: values)
            }
            last = try model.forward(chunk, state: &state, embeddings: embedded,
                produceLogits: end == tokens.count, checkCancellation: { try cancellation?.checkCancellation() })
        }
        return (last, state)
    }

    public func nextTokenLogits(promptTokens: [Int]) throws -> [Float] {
        guard generationLock.try() else { throw Inkling2Error.busy }
        defer { generationLock.unlock() }
        try validate(promptTokens, maxNewTokens: 0)
        return try prefill(promptTokens).0
    }

    public func generateStream(messages: [ChatMessage], sampling: ChatSamplingConfig = .default)
        -> AsyncThrowingStream<String, Error> {
        generateStream(messages: messages, reasoningEffort: 0, sampling: sampling)
    }

    public func generateStream(messages: [ChatMessage], reasoningEffort: Float,
        sampling: ChatSamplingConfig = .default, images: [Inkling2Image] = []) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let cancellation = InferenceCancellationToken()
            let task = Task {
                do {
                    try self.generate(messages: messages, images: images, reasoningEffort: reasoningEffort,
                        sampling: sampling, cancellation: cancellation, emit: { continuation.yield($0) })
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in cancellation.cancel(); task.cancel() }
        }
    }

    private func generate(messages: [ChatMessage], images: [Inkling2Image], reasoningEffort: Float,
        sampling: ChatSamplingConfig, cancellation: InferenceCancellationToken, emit: (String) -> Void) throws {
        guard generationLock.try() else { throw Inkling2Error.busy }
        defer { generationLock.unlock() }
        try sampling.requireNoResponseFormat(backend: "Inkling2CoreMLChat")
        guard sampling.temperature.isFinite, sampling.temperature >= 0,
              sampling.topP.isFinite, sampling.topP > 0, sampling.topP <= 1, sampling.topK >= 0,
              sampling.repetitionPenalty.isFinite, sampling.repetitionPenalty > 0 else {
            throw Inkling2Error.invalid("invalid sampling settings")
        }
        let prompt = encode(try Inkling2Template.render(messages, reasoningEffort: reasoningEffort,
            imageTokenCounts: images.map { $0.rows * $0.columns }))
        try validate(prompt, maxNewTokens: sampling.maxTokens)
        guard sampling.maxTokens > 0 else { return }
        var features: [[Float]] = []
        for image in images {
            try cancellation.checkCancellation()
            features.append(try InklingCoreMLTensor.floats(model.imageEmbeddings(image)))
        }
        var (logits, state) = try prefill(prompt, imageFeatures: features, cancellation: cancellation)
        var generated: [Int] = [], content: [Int] = [], emitted: [UInt8] = [], inText = false
        for _ in 0..<sampling.maxTokens {
            try cancellation.checkCancellation()
            let token = ChatSampler.sample(logits: logits, config: sampling, previousTokens: prompt + generated)
            if token == denseConfig.eosTokenID { break }
            generated.append(token)
            if token == 200004 { inText = true }
            else if token >= 199998 { inText = false }
            else if inText {
                content.append(token)
                let decoded = nativeTokenizer.decode(tokens: content, skipSpecialTokens: true)
                let bytes = Array(decoded.utf8)
                if !decoded.hasSuffix("\u{FFFD}"), bytes.starts(with: emitted) {
                    let suffix = String(decoding: bytes.dropFirst(emitted.count), as: UTF8.self)
                    if !suffix.isEmpty { emit(suffix) }
                    emitted = bytes
                }
            }
            logits = try model.forward([token], state: &state, checkCancellation: { try cancellation.checkCancellation() })
        }
        let decoded = nativeTokenizer.decode(tokens: content, skipSpecialTokens: true)
        let bytes = Array(decoded.utf8)
        if bytes.starts(with: emitted), bytes.count > emitted.count {
            emit(String(decoding: bytes.dropFirst(emitted.count), as: UTF8.self))
        }
    }

    public var config: Qwen3ChatConfig {
        let c = denseConfig
        return Qwen3ChatConfig(hiddenSize: c.hiddenSize, numHiddenLayers: c.layerVisits.count,
            numAttentionHeads: c.numHeads, numKeyValueHeads: c.numKVHeads, headDim: c.headDim,
            intermediateSize: c.intermediateSize, vocabSize: c.vocabSize, maxSeqLen: contextLimit,
            ropeTheta: 1, rmsNormEps: Double(c.rmsEpsilon), eosTokenId: c.eosTokenID,
            padTokenId: 199999, quantization: "int8", quantizationBits: 8,
            quantizationGroupSize: model.manifest.groupSize)
    }
}
