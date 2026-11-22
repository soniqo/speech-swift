import Foundation
import MLX
import MLXCommon
import AudioCommon
import Hub
import Tokenizers

/// Inkling's native message/content framing. The model emits its own assistant
/// header; appending a Qwen-style generation prompt changes its behavior.
public enum Inkling2Template {
    public static func render(_ messages: [ChatMessage], reasoningEffort: Float = 0, imageTokenCounts: [Int] = []) throws -> String {
        guard reasoningEffort.isFinite, (0...0.99).contains(reasoningEffort) else {
            throw Inkling2Error.invalid("reasoning effort must be in 0...0.99")
        }
        let effort = reasoningEffort == 0 ? "0" : String(reasoningEffort)
        let effortTurn = "<|message_system|><|content_text|>Thinking effort level: \(effort)<|end_message|>"
        guard imageTokenCounts.allSatisfy({ $0 > 0 && $0 <= 1024 }),
              imageTokenCounts.isEmpty || messages.contains(where: { $0.role == .user }) else {
            throw Inkling2Error.invalid("images require a user turn and a positive patch count within the image budget")
        }
        let imageTurn = messages.lastIndex(where: { $0.role == .user })
        var result = "", emitted = false
        for (index, message) in messages.enumerated() {
            if !emitted && message.role != .system { result += effortTurn; emitted = true }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            result += "<|message_\(role)|><|content_text|>\(message.content)<|end_message|>"
            if index == imageTurn {
                for count in imageTokenCounts {
                    result += "<|message_user|><|content_image|>" + String(repeating: "<|unused_200054|>", count: count) + "<|end_message|>"
                }
            }
            if message.role == .assistant { result += "<|content_model_end_sampling|>" }
        }
        if !emitted { result += effortTurn }
        return result
    }
}

/// Streaming, serialized Inkling 2 Mini generation from an exported MLX bundle.
/// Compatible with `Qwen35PipelineLLM` and the package's shared sampling options.
public final class Inkling2Chat: @unchecked Sendable, Qwen35ChatBackend {
    let cacheLease = MLXCacheLimitCoordinator.shared.acquireDefault()
    public let denseConfig: Inkling2Config
    public let tokenizer: ChatTokenizer
    let nativeTokenizer: any Tokenizer
    let model: Inkling2Model
    private let generationLock = NSLock()
    public let contextLimit: Int

    private init(model: Inkling2Model, tokenizer: any Tokenizer, protocolTokenizer: ChatTokenizer, contextLimit: Int) throws {
        guard contextLimit > 0, contextLimit <= model.config.maxContext else {
            throw Inkling2Error.invalid("context limit must be within the checkpoint's maximum context")
        }
        self.model = model; denseConfig = model.config; nativeTokenizer = tokenizer
        self.tokenizer = protocolTokenizer; self.contextLimit = contextLimit
    }

    public static func fromDirectory(_ directory: URL, contextLimit: Int = 8192,
        progressHandler: ((Double, String) -> Void)? = nil) throws -> Inkling2Chat {
        guard contextLimit > 0 else { throw Inkling2Error.invalid("context limit must be positive") }
        let config = try Inkling2Config(data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard contextLimit <= config.maxContext else {
            throw Inkling2Error.invalid("context limit exceeds the checkpoint's maximum context")
        }
        let lease = MLXCacheLimitCoordinator.shared.acquireDefault()
        defer { lease?.release() }
        let model = try Inkling2Model.fromDirectory(directory, progressHandler: progressHandler)
        guard var tc = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json"))) as? [NSString: Any],
              let td = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("tokenizer.json"))) as? [NSString: Any] else {
            throw Inkling2Error.invalid("invalid tokenizer files")
        }
        // Select the generic byte-level BPE implementation; all vocabulary,
        // merge, special-token, and pre-tokenizer data remain checkpoint-owned.
        tc["tokenizer_class"] = "Qwen2Tokenizer"
        let native = try AutoTokenizer.from(tokenizerConfig: Config(tc), tokenizerData: Config(td))
        let compatibility = ChatTokenizer()
        try compatibility.load(from: directory)
        compatibility.eosTokenId = model.config.eosTokenID
        return try Inkling2Chat(model: model, tokenizer: native, protocolTokenizer: compatibility,
                                contextLimit: contextLimit)
    }

    /// Download an explicitly chosen affine export; source NVFP4 checkpoints must
    /// first pass through `scripts/export_inkling_mlx.py`.
    public static func fromPretrained(modelId: String, cacheDir: URL? = nil, contextLimit: Int = 8192,
        offlineMode: Bool = false, downloadOptions: DownloadOptions = .init(),
        progressHandler: ((Double, String) -> Void)? = nil) async throws -> Inkling2Chat {
        guard contextLimit > 0 else { throw Inkling2Error.invalid("context limit must be positive") }
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(modelId: modelId, to: directory,
            additionalFiles: ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                              "special_tokens_map.json", "processor_config.json", "model.safetensors.index.json"],
            offlineMode: offlineMode, downloadOptions: downloadOptions,
            progressHandler: { progressHandler?($0 * 0.6, "Downloading…") })
        return try fromDirectory(directory, contextLimit: contextLimit) { progressHandler?(0.6 + $0 * 0.4, $1) }
    }

    public func encode(_ text: String) -> [Int] { nativeTokenizer.encode(text: text, addSpecialTokens: false) }
    public func decode(_ tokens: [Int]) -> String { nativeTokenizer.decode(tokens: tokens, skipSpecialTokens: false) }
    public func resetState() {} // Every request owns a fresh state; none is shared between turns.

    /// Low-level next-token scores for parity checks and custom prompt formatting.
    public func nextTokenLogits(promptTokens: [Int]) throws -> [Float] {
        guard generationLock.try() else { throw Inkling2Error.busy }
        defer { generationLock.unlock() }
        try validateTokens(promptTokens, maxNewTokens: 0)
        let (logits, _) = try prefill(promptTokens, cancellation: nil)
        return logits.asArray(Float.self)
    }

    private func validateTokens(_ tokens: [Int], maxNewTokens: Int) throws {
        guard !tokens.isEmpty, maxNewTokens >= 0, tokens.count <= contextLimit - maxNewTokens,
              tokens.allSatisfy({ (0..<denseConfig.vocabSize).contains($0) }) else {
            throw Inkling2Error.invalid("empty prompt, invalid token ID, or prompt plus generation exceeds the context limit")
        }
    }

    private func prefill(_ tokens: [Int], cancellation: InferenceCancellationToken?, imageEmbeddings: MLXArray? = nil) throws -> (MLXArray, Inkling2Model.State) {
        var state = Inkling2Model.State.initial(config: denseConfig), last: MLXArray?
        var embeddings: MLXArray?
        if let imageEmbeddings {
            let indices = tokens.indices.filter { tokens[$0] == 200054 }
            guard imageEmbeddings.shape == [indices.count, denseConfig.hiddenSize] else {
                throw Inkling2Error.invalid("image placeholders do not match encoded image features")
            }
            let flat = model.embed(MLXArray(tokens.map { Int32($0) }).reshaped(1, -1)).reshaped(-1, denseConfig.hiddenSize)
            flat[MLXArray(indices.map { Int32($0) })] = imageEmbeddings.asType(flat.dtype)
            embeddings = flat.reshaped(1, tokens.count, denseConfig.hiddenSize)
        }
        for start in stride(from: 0, to: tokens.count, by: 128) {
            try cancellation?.checkCancellation()
            let ids = MLXArray(tokens[start..<min(start + 128, tokens.count)].map { Int32($0) }).reshaped(1, -1)
            let chunk = embeddings.map { $0[0..., start..<min(start + 128, tokens.count), 0...] }
            let (logits, next) = try model.forward(inputIDs: ids, state: state, inputEmbeddings: chunk)
            last = logits[0, logits.dim(1) - 1].asType(.float32)
            eval([last!] + next.arrays)
            state = next
        }
        return (last!, state)
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
                    try self.generate(messages: messages, images: images, reasoningEffort: reasoningEffort, sampling: sampling,
                                      cancellation: cancellation, emit: { continuation.yield($0) })
                    continuation.finish()
                } catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { @Sendable _ in cancellation.cancel(); task.cancel() }
        }
    }

    private func generate(messages: [ChatMessage], images: [Inkling2Image], reasoningEffort: Float, sampling: ChatSamplingConfig,
        cancellation: InferenceCancellationToken, emit: (String) -> Void) throws {
        guard generationLock.try() else { throw Inkling2Error.busy }
        defer { generationLock.unlock() }
        try sampling.requireNoResponseFormat(backend: "Inkling2Chat")
        guard sampling.temperature.isFinite, sampling.temperature >= 0,
              sampling.topP.isFinite, sampling.topP > 0, sampling.topP <= 1, sampling.topK >= 0,
              sampling.repetitionPenalty.isFinite, sampling.repetitionPenalty > 0 else {
            throw Inkling2Error.invalid("invalid sampling settings")
        }
        let counts = images.map { $0.rows * $0.columns }
        let prompt = encode(try Inkling2Template.render(messages, reasoningEffort: reasoningEffort, imageTokenCounts: counts))
        try validateTokens(prompt, maxNewTokens: sampling.maxTokens)
        guard sampling.maxTokens > 0 else { return }
        var features: [MLXArray] = []
        for image in images {
            try cancellation.checkCancellation()
            let encoded = try model.imageEmbeddings(image)
            eval(encoded); features.append(encoded)
        }
        var (logits, state) = try prefill(prompt, cancellation: cancellation,
                                         imageEmbeddings: features.isEmpty ? nil : concatenated(features, axis: 0))
        var generated: [Int] = [], content: [Int] = [], emitted: [UInt8] = [], inText = false
        for _ in 0..<sampling.maxTokens {
            try cancellation.checkCancellation()
            let token = ChatSampler.sample(logits: logits.asArray(Float.self), config: sampling,
                                           previousTokens: prompt + generated)
            if token == denseConfig.eosTokenID { break }
            generated.append(token)
            if token == 200004 { inText = true }
            else if token >= 199998 { inText = false }
            else if inText && token < 199998 {
                content.append(token)
                let decoded = nativeTokenizer.decode(tokens: content, skipSpecialTokens: true)
                // The BPE decoder may hold a partial UTF-8 sequence at a token boundary.
                let bytes = Array(decoded.utf8)
                if !decoded.hasSuffix("\u{FFFD}"), bytes.starts(with: emitted) {
                    let suffix = String(decoding: bytes.dropFirst(emitted.count), as: UTF8.self)
                    if !suffix.isEmpty { emit(suffix) }
                    emitted = bytes
                }
            }
            let (step, next) = try model.forward(inputIDs: MLXArray([Int32(token)]).reshaped(1, 1), state: state)
            logits = step[0, 0].asType(.float32)
            eval([logits] + next.arrays)
            state = next
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
            padTokenId: 199999, quantization: "int\(c.defaultQuantization.bits)",
            quantizationBits: c.defaultQuantization.bits, quantizationGroupSize: c.defaultQuantization.groupSize)
    }
}
