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
    private let vocabularyLock = NSLock()
    private var _constraintVocabulary: JSONTokenVocabulary?
    private var _xgrammarVocabulary: XGrammarVocabulary?

    /// Model state just past a system turn, most recently used last, for requests that repeat it.
    ///
    /// A caller running a fixed set of instructions over many inputs sends the same system turn
    /// again and again, and prefill read it from scratch every time. Measured over one Discover run
    /// on the built-in E4B (2026-09-27), system turns were 80% of prompt text and the same fifty or
    /// so recurred across 2,395 requests, so resuming from a kept state skips most of what prefill
    /// reads. Bounded by bytes: a snapshot is the K/V of every producing layer over the turn, about
    /// 57 KB a token.
    private var prefixSnapshots: [(tokens: [Int], state: Gemma4Model.InferenceState, bytes: Int)] = []
    private let prefixLock = NSLock()

    /// `SPEECH_SWIFT_GEMMA4_PREFIX_CACHE_MB`; 0 turns reuse off.
    static let prefixCacheBudget: Int = {
        let megabytes = ProcessInfo.processInfo.environment["SPEECH_SWIFT_GEMMA4_PREFIX_CACHE_MB"]
            .flatMap { Int($0.trimmingCharacters(in: .whitespaces)) } ?? 512
        return max(0, megabytes) * 1024 * 1024
    }()

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
        generateStream(
            messages: messages,
            sampling: sampling,
            shouldContinue: { true })
    }

    /// Streaming generation with cooperative token-boundary cancellation.
    ///
    /// MLX evaluation of one token and the initial prompt prefill are atomic,
    /// but the caller can stop before the next token is scheduled. Returning
    /// from `decode` also guarantees the producer is finished before a shared
    /// model is used by the next request.
    public func generateStream(
        messages: [ChatMessage],
        sampling: ChatSamplingConfig = .default,
        shouldContinue: @escaping @Sendable () -> Bool
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let constraint: (any TokenDecodeConstraint)?
                do {
                    constraint = try self.makeConstraint(sampling.responseFormat)
                } catch {
                    continuation.finish(throwing: error)
                    return
                }
                let promptTokens = Gemma4ChatTemplate.encode(
                    messages: messages, tokenizer: self.gemmaTokenizer)
                let failure = self.decode(
                    promptTokens: promptTokens,
                    reusablePrefix: Gemma4ChatTemplate.systemTurnLength(
                        messages: messages, tokenizer: self.gemmaTokenizer),
                    sampling: sampling,
                    shouldContinue: shouldContinue,
                    constraint: constraint,
                    onText: { text in continuation.yield(text) })
                if let failure { continuation.finish(throwing: failure) } else { continuation.finish() }
            }
        }
    }

    // MARK: - Constrained decoding

    /// The vocabulary index constrained decoding walks, built on first use and kept: indexing
    /// 262,144 tokens is a one-off cost of a fraction of a second.
    func constraintVocabulary() -> JSONTokenVocabulary {
        vocabularyLock.lock()
        defer { vocabularyLock.unlock() }
        if let v = _constraintVocabulary { return v }
        let v = JSONTokenVocabulary(gemma: gemmaTokenizer, size: denseConfig.vocabSize)
        _constraintVocabulary = v
        return v
    }

    /// XGrammar's compiler over this model's vocabulary, built on first use and kept.
    func xgrammarVocabulary() throws -> XGrammarVocabulary {
        vocabularyLock.lock()
        defer { vocabularyLock.unlock() }
        if let v = _xgrammarVocabulary { return v }
        let v = try XGrammarVocabulary(gemma: gemmaTokenizer, size: denseConfig.vocabSize)
        _xgrammarVocabulary = v
        return v
    }

    func makeConstraint(
        _ format: ChatResponseFormat?, engine: JSONConstraintEngine = .current
    ) throws -> (any TokenDecodeConstraint)? {
        guard let format else { return nil }
        switch format {
        case .jsonSchema(let schema):
            // The keyword gate is shared: a schema either engine would enforce only in part is
            // rejected before anything compiles.
            let grammar = try JSONSchemaGrammar(schema: schema)
            switch engine {
            case .xgrammar:
                return try xgrammarVocabulary().constraint(schema: schema)
            case .swift:
                return JSONTokenConstraint(
                    grammar: grammar, vocabulary: constraintVocabulary(),
                    endTokens: gemmaTokenizer.eosTokenIds.sorted())
            }
        }
    }

    /// One decode pass: prefill, then a token per step until an end token or the budget runs out.
    ///
    /// `onText` receives answer text as the reasoning-channel filter completes it (often nothing —
    /// one character can span several tokens); `onToken` receives every sampled id, which is what
    /// the greedy-parity test compares against the host sampler.
    ///
    /// The step is one lazy MLX graph — model forward, suppression, penalty, top-K/top-P and the
    /// draw — and reading the sampled id is the only point it is waited on. The previous shape
    /// evaluated the logits, pulled all 262k of them to the host, and sampled there: two
    /// synchronisations and a megabyte per token.
    ///
    /// With a `constraint`, every step samples only tokens that keep the output a valid prefix of
    /// the constrained document, and the loop stops the moment the document is complete. The
    /// mask for the next step is computed on the host while the device runs the forward pass for
    /// it. Returns an error only when the constraint reached a state with no admissible token.
    @discardableResult
    func decode(
        promptTokens: [Int],
        reusablePrefix: Int = 0,
        sampling: ChatSamplingConfig,
        shouldContinue: () -> Bool = { true },
        constraint: (any TokenDecodeConstraint)? = nil,
        onToken: (Int) -> Void = { _ in },
        onText: (String) -> Void
    ) -> ChatResponseFormatError? {
        // Prefill. Only the final position is sampled, so the lm_head runs on that row alone —
        // over a long prompt the discarded rows are gigabytes of 262k-wide logits.
        var logits = prefill(promptTokens, reusablePrefix: reusablePrefix)

        var history = promptTokens
        var produced = false
        var filter = Gemma4AnswerFilter(tokenizer: gemmaTokenizer)
        let endTokens = Array(gemmaTokenizer.eosTokenIds)
        let noReasoning = [Gemma4AnswerFilter.channelOpen]
        var constraint = constraint
        var mask: DeviceTokenMask?
        var failure: ChatResponseFormatError?
        if constraint != nil {
            asyncEval(logits)
            mask = constraint!.nextMask()
        }

        var remaining = sampling.maxTokens
        while remaining > 0 && shouldContinue() {
            remaining -= 1
            if let mask, mask.isEmpty { failure = .noAdmissibleToken; break }

            let next = ChatSampler.sampleOnDevice(
                logits: logits,
                config: sampling,
                // Don't let the model end the turn before emitting any visible answer, and never
                // let it open the reasoning channel: this template does not enable thinking,
                // the filter below discards whatever the channel holds, and a channel that ran
                // to the budget returned an empty reply after spending every token on it.
                suppressing: produced ? noReasoning : endTokens + noReasoning,
                previousTokens: history,
                vocabSize: denseConfig.vocabSize,
                uniform: sampling.temperature > 0 ? Float.random(in: 0 ..< 1) : 0,
                allowed: mask
            ).item(Int.self)

            if gemmaTokenizer.eosTokenIds.contains(next) { break }
            if constraint != nil, !constraint!.accept(next) { failure = .noAdmissibleToken; break }
            history.append(next)
            onToken(next)

            let text = filter.consume(next)
            if !text.isEmpty { produced = true; onText(text) }

            // A complete document ends the turn; no forward is spent on a token nothing may follow.
            if constraint?.isDone == true { break }

            // Decode one step — but not a step whose logits nothing will read. The budget's last
            // token used to be followed by a full forward that was evaluated and thrown away.
            guard remaining > 0 else { break }
            let arr = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
            logits = model.forward(inputIds: arr, state: &state)
            if constraint != nil {
                asyncEval(logits)
                mask = constraint!.nextMask()
            }
        }

        if let tail = filter.flush(), !tail.isEmpty { onText(tail) }
        return failure
    }

    /// Prefill `tokens`, resuming from a kept state for their first `reusablePrefix` tokens when
    /// one exists and keeping one when it does not. The split prefill is the whole prefill in two
    /// passes — `testSplitPrefillMatchesWholePrefill` — so the result does not depend on a hit.
    func prefill(_ tokens: [Int], reusablePrefix: Int) -> MLXArray {
        let prefixLength = reusablePrefix
        guard Self.prefixCacheBudget > 0, prefixLength > 0, prefixLength < tokens.count else {
            resetState()
            return model.lastTokenLogits(
                inputIds: MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0),
                state: &state)
        }
        let prefix = Array(tokens[0 ..< prefixLength])
        prefixLock.lock()
        let kept = prefixSnapshots.lastIndex { $0.tokens == prefix }.map { index in
            let hit = prefixSnapshots.remove(at: index)
            prefixSnapshots.append(hit)
            return hit.state
        }
        prefixLock.unlock()
        if let kept {
            state = kept.copied()
        } else {
            resetState()
            let head = model.lastTokenLogits(
                inputIds: MLXArray(prefix.map { Int32($0) }).expandedDimensions(axis: 0),
                state: &state)
            eval([head] + state.arrays)
            keepPrefix(prefix, state: state.copied())
        }
        let rest = tokens[prefixLength...].map { Int32($0) }
        return model.lastTokenLogits(
            inputIds: MLXArray(rest).expandedDimensions(axis: 0), state: &state)
    }

    private func keepPrefix(_ tokens: [Int], state kept: Gemma4Model.InferenceState) {
        let bytes = kept.byteCount
        guard bytes <= Self.prefixCacheBudget else { return }
        prefixLock.lock()
        defer { prefixLock.unlock() }
        prefixSnapshots.append((tokens, kept, bytes))
        var total = prefixSnapshots.reduce(0) { $0 + $1.bytes }
        while total > Self.prefixCacheBudget, !prefixSnapshots.isEmpty {
            total -= prefixSnapshots.removeFirst().bytes
        }
    }

    /// Drop every kept prompt state, returning its memory.
    public func clearPrefixCache() {
        prefixLock.lock()
        prefixSnapshots.removeAll()
        prefixLock.unlock()
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
            quantizationBits: denseConfig.quantBits,
            quantizationGroupSize: denseConfig.quantGroupSize,
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
    static let channelOpen = 100
    static let channelClose = 101
    private var channelOpen: Int { Self.channelOpen }
    private var channelClose: Int { Self.channelClose }
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
    /// Tokens up to the end of a leading system turn, which `encode` writes as its own segment, or
    /// 0 without one. Every request with the same system turn starts with exactly these tokens.
    static func systemTurnLength(messages: [ChatMessage], tokenizer: Gemma4Tokenizer) -> Int {
        guard let first = messages.first, first.role == .system, messages.count > 1 else { return 0 }
        return 1 + tokenizer.encode("<|turn>system\n").count + tokenizer.encode(first.content).count
            + tokenizer.encode("<turn|>\n").count
    }

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
