import Foundation
import MLX
import AudioCommon

/// Chat backend for the hand-written `Qwen3DenseModel` (standard Qwen3: 1.7B / 4B / 8B …).
///
/// Conforms to `Qwen35ChatBackend`, so `Qwen35PipelineLLM(model:)` and the rest of the stack accept
/// it with no changes — the larger chat model is a drop-in for the 0.8B. Reuses the shared
/// `ChatTokenizer` (byte-level decode) + the optimised `ChatSampler`. Non-thinking instruct models
/// (e.g. Qwen3-*-Instruct-2507), so no `<think>` handling.
public final class Qwen3DenseChat: @unchecked Sendable {
    public let dc: Qwen3DenseConfig
    public let tokenizer: ChatTokenizer
    let model: Qwen3DenseModel
    var state: Qwen3DenseModel.InferenceState

    private let imStartId: Int
    private let imEndId: Int
    private let newlineId: Int

    private init(dc: Qwen3DenseConfig, tokenizer: ChatTokenizer, model: Qwen3DenseModel) {
        self.dc = dc
        self.tokenizer = tokenizer
        self.model = model
        self.state = .initial(config: dc)
        self.imStartId = tokenizer.tokenId("<|im_start|>") ?? 151644
        self.imEndId = tokenizer.tokenId("<|im_end|>") ?? 151645
        self.newlineId = tokenizer.tokenId("\u{010A}") ?? 198   // byte-level "\n"
    }

    // MARK: - Loading

    /// Load from a local MLX model directory (config.json + tokenizer + safetensors).
    public static func fromDirectory(
        _ directory: URL, progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> Qwen3DenseChat {
        let config = try Qwen3DenseConfig.load(from: directory.appendingPathComponent("config.json"))
        let tokenizer = ChatTokenizer()
        try tokenizer.load(from: directory)
        tokenizer.eosTokenId = config.eosTokenId
        let model = Qwen3DenseModel(config: config)
        try Qwen3DenseWeightLoader.loadWeights(into: model, from: directory, progressHandler: progressHandler)
        return Qwen3DenseChat(dc: config, tokenizer: tokenizer, model: model)
    }

    /// Download an MLX model from HuggingFace, then load it.
    public static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-4B-Instruct-2507-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3DenseChat {
        let dir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId, to: dir,
            additionalFiles: [
                "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                "merges.txt", "added_tokens.json", "special_tokens_map.json",
                "model.safetensors", "model.safetensors.index.json",
            ],
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.5, "Downloading…") }
        )
        return try fromDirectory(dir, progressHandler: { progressHandler?(0.5 + $0 * 0.5, $1) })
    }

    // MARK: - Chat template (Qwen3 ChatML)

    private func encodeChat(_ messages: [ChatMessage]) -> [Int] {
        var ids: [Int] = []
        func turn(_ role: String, _ content: String) {
            ids.append(imStartId)
            ids.append(contentsOf: tokenizer.encode(role))
            ids.append(newlineId)
            ids.append(contentsOf: tokenizer.encode(content))
            ids.append(imEndId)
            ids.append(newlineId)
        }
        for m in messages { turn(m.role.rawValue, m.content) }
        ids.append(imStartId)
        ids.append(contentsOf: tokenizer.encode("assistant"))
        ids.append(newlineId)
        return ids
    }

    // MARK: - Generation

    private func lastLogits(_ logits: MLXArray) -> [Float] {
        let t = logits.dim(1)
        let last = logits[0, t - 1].asType(.float32)
        eval(last)
        return Array(last.asArray(Float.self).prefix(dc.vocabSize))
    }

    private func suppressEnd(_ logits: inout [Float]) {
        let ninf = -Float.greatestFiniteMagnitude
        for id in [dc.eosTokenId, imEndId] where id >= 0 && id < logits.count { logits[id] = ninf }
    }

    public func resetState() { state = .initial(config: dc) }

    /// Numeric-parity helper: argmax + the next-token logits for a fixed prompt (no sampling).
    public func nextTokenArgmax(promptTokens: [Int]) -> (argmax: Int, logit: Float, top5: [(Int, Float)]) {
        resetState()
        let arr = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let (logits, _) = model.forward(inputIds: arr, state: state)
        eval(logits)
        let l = lastLogits(logits)
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        let top5 = l.enumerated().sorted { $0.element > $1.element }.prefix(5).map { ($0.offset, $0.element) }
        return (best, l[best], Array(top5))
    }

    public func generateStream(
        messages: [ChatMessage], sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                self.resetState()
                let prompt = self.encodeChat(messages)
                let promptArray = MLXArray(prompt.map { Int32($0) }).expandedDimensions(axis: 0)
                let (prefill, prefillState) = self.model.forward(inputIds: promptArray, state: self.state)
                eval(prefill)
                self.state = prefillState
                var logits = self.lastLogits(prefill)

                var generated: [Int] = []
                var producedContent = false
                var pending: [UInt8] = []

                for _ in 0..<sampling.maxTokens {
                    if !producedContent { self.suppressEnd(&logits) }
                    let next = ChatSampler.sample(
                        logits: logits, config: sampling, previousTokens: prompt + generated)
                    if next == self.dc.eosTokenId || next == self.imEndId { break }
                    generated.append(next)
                    if !self.tokenizer.isSpecialToken(next) {
                        producedContent = true
                        pending.append(contentsOf: self.tokenizer.tokenBytes(next))
                        let (text, rem) = ChatTokenizer.decodeUTF8Prefix(pending)
                        pending = rem
                        if !text.isEmpty { continuation.yield(text) }
                    }
                    let arr = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
                    let (step, ns) = self.model.forward(inputIds: arr, state: self.state)
                    eval(step)
                    self.state = ns
                    logits = self.lastLogits(step)
                }
                if !pending.isEmpty { continuation.yield(String(decoding: pending, as: UTF8.self)) }
                continuation.finish()
            }
        }
    }

    /// A `Qwen3ChatConfig` view for `Qwen35ChatBackend` conformance (the pipeline doesn't read the
    /// Qwen3.5-specific fields). Internal memberwise init is visible within this module.
    public lazy var config: Qwen3ChatConfig = Qwen3ChatConfig(
        hiddenSize: dc.hiddenSize, numHiddenLayers: dc.numHiddenLayers,
        numAttentionHeads: dc.numAttentionHeads, numKeyValueHeads: dc.numKeyValueHeads,
        headDim: dc.headDim, intermediateSize: dc.intermediateSize, vocabSize: dc.vocabSize,
        maxSeqLen: 32768, ropeTheta: Double(dc.ropeTheta), rmsNormEps: Double(dc.rmsNormEps),
        eosTokenId: dc.eosTokenId, padTokenId: dc.eosTokenId, quantization: "int4",
        modelType: nil, layerTypes: nil, fullAttentionInterval: nil,
        linearNumKeyHeads: nil, linearKeyHeadDim: nil, linearNumValueHeads: nil,
        linearValueHeadDim: nil, linearConvKernelDim: nil, partialRotaryFactor: nil,
        tieWordEmbeddings: dc.tieWordEmbeddings)
}

extension Qwen3DenseChat: Qwen35ChatBackend {}
