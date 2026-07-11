import Foundation
import MLX
import AudioCommon

/// Streaming chat backend for the hand-written `Gemma4Model` (Gemma 4 text, E2B/E4B MLX int4).
///
/// Mirrors `Qwen35MLXChat`: load tokenizer + config + weights, encode the chat template, prefill,
/// then decode one token per step against the incremental KV cache. Two Gemma-4 specifics:
///   • the chat template is the `<|turn>{role}\n…<turn|>\n` form (NOT the older `<start_of_turn>`),
///     terminated by `<|turn>model\n` for the generation prompt — see `Gemma4ChatTemplate`.
///   • a reasoning *channel* `<|channel>thought\n…\n<channel|>` is emitted before the answer; for a
///     voice assistant we suppress it and stream only the post-channel answer text.
public final class Gemma4Chat: @unchecked Sendable {
    /// Gemma-4 architecture config (used by the model + parity harness).
    public let denseConfig: Gemma4DenseConfig
    let model: Gemma4Model
    public let gemmaTokenizer: Gemma4Tokenizer
    /// GPT-2-scheme tokenizer kept only to satisfy `Qwen35ChatBackend.tokenizer`; the generation
    /// path uses `gemmaTokenizer` (SentencePiece byte-fallback) for correct encode/decode.
    public let tokenizer: ChatTokenizer
    var state: Gemma4Model.InferenceState
    var _isLoaded = true

    private init(config: Gemma4DenseConfig, gemmaTokenizer: Gemma4Tokenizer,
                 tokenizer: ChatTokenizer, model: Gemma4Model) {
        self.denseConfig = config
        self.gemmaTokenizer = gemmaTokenizer
        self.tokenizer = tokenizer
        self.model = model
        self.state = .initial(config: config)
    }

    // MARK: - Loading

    /// Load from a local MLX model directory (config.json + tokenizer.json + safetensors).
    public static func fromDirectory(
        _ directory: URL, progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> Gemma4Chat {
        let config = try Gemma4DenseConfig.load(from: directory.appendingPathComponent("config.json"))
        let gemmaTok = Gemma4Tokenizer()
        try gemmaTok.load(from: directory)
        let tok = ChatTokenizer()
        try? tok.load(from: directory)   // best-effort; only the protocol surface needs it
        let model = Gemma4Model(config: config)
        try Gemma4WeightLoader.loadWeights(into: model, from: directory, progressHandler: progressHandler)
        return Gemma4Chat(config: config, gemmaTokenizer: gemmaTok, tokenizer: tok, model: model)
    }

    /// Download + load from HuggingFace (e.g. `aufklarer/gemma-4-E4B-it-MLX-4bit`).
    public static func fromPretrained(
        modelId: String = "aufklarer/gemma-4-E4B-it-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Gemma4Chat {
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "generation_config.json", "model.safetensors", "model.safetensors.index.json",
            ],
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0 * 0.6, "Downloading...") })
        return try fromDirectory(cacheDir) { p, m in progressHandler?(0.6 + p * 0.4, m) }
    }

    // MARK: - State

    public func resetState() { state = .initial(config: denseConfig) }

    // MARK: - Generation

    /// Buffered (non-streaming) generation — returns the full thinking-free reply.
    public func generate(
        messages: [ChatMessage], sampling: ChatSamplingConfig = .default
    ) throws -> String {
        var reply = ""
        let sem = DispatchSemaphore(value: 0)
        var err: Error?
        Task {
            do { for try await chunk in generateStream(messages: messages, sampling: sampling) { reply += chunk } }
            catch { err = error }
            sem.signal()
        }
        sem.wait()
        if let err { throw err }
        return reply
    }

    /// Streaming generation. Suppresses the reasoning channel and only yields answer text.
    public func generateStream(
        messages: [ChatMessage], sampling: ChatSamplingConfig = .default
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                self.resetState()

                let promptTokens = Gemma4ChatTemplate.encode(
                    messages: messages, tokenizer: self.gemmaTokenizer)

                // Prefill.
                let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
                let prefillLogits = self.model.forward(inputIds: promptArray, state: &self.state)
                eval(prefillLogits)
                var logits = self.lastLogits(prefillLogits)

                var history = promptTokens
                var produced = false
                var filter = Gemma4AnswerFilter(tokenizer: self.gemmaTokenizer)

                for _ in 0..<sampling.maxTokens {
                    // Don't let the model end the turn before emitting any visible answer.
                    if !produced { self.suppressEndTokens(&logits) }

                    let next = ChatSampler.sample(
                        logits: logits, config: sampling, previousTokens: history)

                    if self.gemmaTokenizer.eosTokenIds.contains(next) { break }
                    history.append(next)

                    let text = filter.consume(next)
                    if !text.isEmpty { produced = true; continuation.yield(text) }

                    // Decode one step.
                    let arr = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
                    let step = self.model.forward(inputIds: arr, state: &self.state)
                    eval(step)
                    logits = self.lastLogits(step)
                }

                if let tail = filter.flush(), !tail.isEmpty { continuation.yield(tail) }
                continuation.finish()
            }
        }
    }

    // MARK: - Parity harness (unchanged surface used by Gemma4ParityTests)

    /// Numeric-parity helper: argmax + next-token logits for a fixed prompt (no sampling, no cache).
    public func nextTokenArgmax(promptTokens: [Int]) -> (argmax: Int, logit: Float, top5: [(Int, Float)]) {
        let arr = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let logits = model.forward(inputIds: arr)
        eval(logits)
        let t = logits.dim(1)
        let last = logits[0, t - 1].asType(.float32)
        eval(last)
        let l = Array(last.asArray(Float.self).prefix(denseConfig.vocabSize))
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        let top5 = l.enumerated().sorted { $0.element > $1.element }.prefix(5).map { ($0.offset, $0.element) }
        return (best, l[best], Array(top5))
    }

    /// Sanity helper: argmax of the next token computed through the incremental KV-cache prefill
    /// (used by tests to confirm the cache path matches `nextTokenArgmax`'s single forward).
    public func firstTokenViaCache(promptTokens: [Int]) -> Int {
        var st = Gemma4Model.InferenceState.initial(config: denseConfig)
        let arr = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let logits = model.forward(inputIds: arr, state: &st)
        eval(logits)
        let t = logits.dim(1)
        let last = logits[0, t - 1].asType(.float32)
        eval(last)
        let l = Array(last.asArray(Float.self).prefix(denseConfig.vocabSize))
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        return best
    }

    // MARK: - Helpers

    private func suppressEndTokens(_ logits: inout [Float]) {
        let ninf = -Float.greatestFiniteMagnitude
        for id in gemmaTokenizer.eosTokenIds where id >= 0 && id < logits.count { logits[id] = ninf }
    }

    private func lastLogits(_ logits: MLXArray) -> [Float] {
        let t = logits.dim(1)
        let last = logits[0, t - 1].asType(.float32)
        eval(last)
        let all: [Float] = last.asArray(Float.self)
        return Array(all.prefix(denseConfig.vocabSize))
    }
}

// MARK: - Qwen35ChatBackend conformance

extension Gemma4Chat: Qwen35ChatBackend {
    /// Bridge the Gemma-4 config to the `Qwen3ChatConfig` shape the backend protocol exposes.
    /// Only the fields consumers read (vocab size, eos, etc.) are meaningful here.
    public var config: Qwen3ChatConfig {
        Qwen3ChatConfig(
            hiddenSize: denseConfig.hiddenSize,
            numHiddenLayers: denseConfig.numHiddenLayers,
            numAttentionHeads: denseConfig.numAttentionHeads,
            numKeyValueHeads: denseConfig.numKeyValueHeads,
            headDim: denseConfig.headDim,
            intermediateSize: denseConfig.intermediateSize,
            vocabSize: denseConfig.vocabSize,
            maxSeqLen: denseConfig.maxPositionEmbeddings,
            ropeTheta: Double(denseConfig.fullRopeTheta),
            rmsNormEps: Double(denseConfig.rmsNormEps),
            eosTokenId: denseConfig.eosTokenId,
            padTokenId: 0,
            quantization: "int\(denseConfig.quantBits)",
            modelType: nil,
            layerTypes: denseConfig.layerTypes,
            fullAttentionInterval: nil,
            linearNumKeyHeads: nil,
            linearKeyHeadDim: nil,
            linearNumValueHeads: nil,
            linearValueHeadDim: nil,
            linearConvKernelDim: nil,
            partialRotaryFactor: Double(denseConfig.fullPartialRotaryFactor),
            tieWordEmbeddings: denseConfig.tieWordEmbeddings)
    }
}

// MARK: - Reasoning-channel filter

/// Streaming filter that suppresses Gemma 4's reasoning channel and emits only the spoken answer.
///
/// Gemma 4 may emit `<|channel>thought\n …thinking… \n<channel|>` (ids 100 … 101) before the answer.
/// We drop every token from the opening `<|channel>` (100) through the matching `<channel|>` (101),
/// and never emit special/markup tokens. Everything else is byte-accumulated and decoded as UTF-8
/// (BPE can split one character across tokens). Because the channel markers are single vocab tokens,
/// id-matching is exact; the inner thought text — which *can* span many tokens — is still fully
/// skipped, so the filter is robust to multi-token reasoning blocks.
struct Gemma4AnswerFilter {
    private let tokenizer: Gemma4Tokenizer
    private let channelOpen = 100
    private let channelClose = 101
    private var inThoughtChannel = false
    private var pending: [UInt8] = []

    init(tokenizer: Gemma4Tokenizer) { self.tokenizer = tokenizer }

    /// Feed one generated token id; returns any answer text now decodable (often empty).
    mutating func consume(_ id: Int) -> String {
        if id == channelOpen { inThoughtChannel = true; return "" }
        if id == channelClose { inThoughtChannel = false; return "" }
        guard !inThoughtChannel, !tokenizer.isSpecialToken(id) else { return "" }
        pending.append(contentsOf: tokenizer.tokenBytes(id))
        let (text, rest) = ChatTokenizer.decodeUTF8Prefix(pending)
        pending = rest
        return text
    }

    /// Flush any trailing bytes (lossy) when the stream ends.
    mutating func flush() -> String? {
        guard !pending.isEmpty else { return nil }
        let s = String(decoding: pending, as: UTF8.self)
        pending = []
        return s
    }
}

// MARK: - Gemma 4 chat template

/// Renders the Gemma 4 `<|turn>` chat template (matches the model's `chat_template.jinja`):
///
/// ```
/// <bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n
/// ```
///
/// Special token ids (confirmed against the model tokenizer): `<bos>`=2, `<|turn>`=105,
/// `<turn|>`=106, `\n`=107, role words `system`/`user`/`model`. We render via the tokenizer's
/// encode so a vocab change can't desync the ids.
enum Gemma4ChatTemplate {
    static func encode(messages: [ChatMessage], tokenizer: Gemma4Tokenizer) -> [Int] {
        var tokens: [Int] = [tokenizer.bosTokenId]
        for m in messages {
            let role = (m.role == .assistant) ? "model" : m.role.rawValue
            tokens.append(contentsOf: tokenizer.encode("<|turn>" + role + "\n"))
            tokens.append(contentsOf: tokenizer.encode(m.content))
            tokens.append(contentsOf: tokenizer.encode("<turn|>\n"))
        }
        tokens.append(contentsOf: tokenizer.encode("<|turn>model\n"))
        return tokens
    }
}
