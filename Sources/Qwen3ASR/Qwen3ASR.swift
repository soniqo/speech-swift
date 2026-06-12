import Foundation
import MLXCommon
import MLX
import MLXNN
import MLXFast
import AudioCommon

/// Optional decoder tunables for `Qwen3ASRModel.transcribe(audio:options:)`.
///
/// Defaults match the historical greedy behaviour of `transcribe(audio:)`
/// so existing callers see zero change. Tune these when greedy decoding
/// collapses onto a single token (typical on silence or ambiguous phonemes).
///
/// The struct also carries the "long-input auto-escalation" knobs used by
/// the public `transcribe(...)` entry points to bound greedy degeneration
/// on >15 s audio without affecting short-clip behaviour. See
/// `adaptedFor(audioDurationSeconds:)`.
public struct Qwen3DecodingOptions: Sendable {
    /// Cap on decoder output per chunk.
    public var maxTokens: Int = 448

    /// Optional language hint ("en", "zh", …). `nil` = auto-detect.
    public var language: String?

    /// Context hint prepended to the decoder prompt.
    public var context: String?

    /// HuggingFace-style repetition penalty. Divides the logits of tokens
    /// already generated this chunk by this factor before `argMax`.
    /// `1.0` disables; `1.1`–`1.3` is the common tuning range.
    public var repetitionPenalty: Float = 1.0

    /// If > 0, masks any next-token whose emission would form a repeated
    /// n-gram of this size. `0` disables.
    public var noRepeatNgramSize: Int = 0

    /// `0` = greedy (argmax). `> 0` = sample with this temperature via
    /// Gumbel-max. Higher = more random.
    public var temperature: Float = 0.0

    /// Adaptive decoding threshold. When the input audio is longer than this
    /// many seconds AND the caller has left `noRepeatNgramSize` at 0 (the
    /// default greedy path), the public `transcribe(...)` entry points
    /// auto-escalate `noRepeatNgramSize` to `longInputNoRepeatNgramSize`
    /// before forwarding into `generateText`. This bounds the 0.6B
    /// greedy-decode degeneration observed on long-form audio without
    /// affecting short clips. Set to `.infinity` to disable entirely.
    public var longInputThresholdSeconds: Double = 15.0

    /// n-gram size applied by the long-input auto-escalation path. Only
    /// used when the threshold above triggers AND the caller hasn't
    /// already set a custom `noRepeatNgramSize`. 3 mirrors the slow-path
    /// default in `E2EQwen3DecodingOptionsTests`.
    public var longInputNoRepeatNgramSize: Int = 3

    public init(
        maxTokens: Int = 448,
        language: String? = nil,
        context: String? = nil,
        repetitionPenalty: Float = 1.0,
        noRepeatNgramSize: Int = 0,
        temperature: Float = 0.0,
        longInputThresholdSeconds: Double = 15.0,
        longInputNoRepeatNgramSize: Int = 3
    ) {
        self.maxTokens = maxTokens
        self.language = language
        self.context = context
        self.repetitionPenalty = repetitionPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.temperature = temperature
        self.longInputThresholdSeconds = longInputThresholdSeconds
        self.longInputNoRepeatNgramSize = longInputNoRepeatNgramSize
    }

    /// Length-gated auto-escalation. Returns a copy with
    /// `noRepeatNgramSize` bumped to `longInputNoRepeatNgramSize` IFF
    ///   1. `audioDurationSeconds > longInputThresholdSeconds`, AND
    ///   2. the caller left `noRepeatNgramSize` at 0 (default greedy), AND
    ///   3. `longInputNoRepeatNgramSize > 0` (escalation not disabled).
    /// Otherwise returns `self` unchanged.
    ///
    /// Any caller that has explicitly tuned `noRepeatNgramSize` — including
    /// setting it to a non-3 value — is honoured. This preserves the
    /// fast-path / slow-path routing semantics in `isGreedyFastPath`.
    func adaptedFor(audioDurationSeconds: Double) -> Qwen3DecodingOptions {
        guard audioDurationSeconds > longInputThresholdSeconds,
              noRepeatNgramSize == 0,
              longInputNoRepeatNgramSize > 0 else {
            return self
        }
        var copy = self
        copy.noRepeatNgramSize = longInputNoRepeatNgramSize
        return copy
    }
}

/// Special token IDs for Qwen3-ASR
public struct Qwen3ASRTokens: Sendable {
    public static let audioTokenId = 151676        // <|audio_pad|>
    public static let audioStartTokenId = 151669   // <|audio_start|>
    public static let audioEndTokenId = 151670     // <|audio_end|>
    public static let eosTokenId = 151645          // <|im_end|>
    public static let padTokenId = 151643          // <|endoftext|>
    public static let imStartTokenId = 151644      // <|im_start|>
    public static let imEndTokenId = 151645        // <|im_end|>
    public static let timestampTokenId = 151705    // <|timestamp|>
}

/// Main Qwen3-ASR model for speech recognition.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class Qwen3ASRModel {
    public let audioEncoder: Qwen3AudioEncoder
    public let featureExtractor: WhisperFeatureExtractor
    public var textDecoder: QuantizedTextModel?

    /// Tokenizer for decoding output tokens
    private var tokenizer: Qwen3Tokenizer?

    /// Text decoder config
    public let textConfig: TextDecoderConfig

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    /// MLX cache limit captured at load time for the .large variant. Stored
    /// per-instance so `unload()` can restore it — preventing the 4 GB cap
    /// from leaking into co-loaded models (PersonaPlex loads ASR + LM + TTS
    /// in the same process). `nil` when no cap was applied (small variant
    /// or already-capped global state).
    var savedMLXCacheLimit: Int?

    init(
        audioConfig: Qwen3AudioEncoderConfig = .default,
        textConfig: TextDecoderConfig = .small
    ) {
        self.audioEncoder = Qwen3AudioEncoder(config: audioConfig)
        self.featureExtractor = WhisperFeatureExtractor()
        self.textConfig = textConfig
        // Text decoder will be initialized when loading weights
        self.textDecoder = nil
    }

    /// Set tokenizer for text decoding
    func setTokenizer(_ tokenizer: Qwen3Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Initialize text decoder (called after loading)
    func initializeTextDecoder() {
        self.textDecoder = QuantizedTextModel(config: textConfig)
    }

    /// Transcribe audio to text with explicit decoder options.
    ///
    /// The legacy `transcribe(audio:sampleRate:language:maxTokens:context:)`
    /// overload below forwards into this path with default (greedy) options.
    ///
    /// Long-input adaptive decoding: before forwarding into `generateText`,
    /// `options.adaptedFor(audioDurationSeconds:)` is applied. On audio
    /// longer than `options.longInputThresholdSeconds` (default 15 s) AND
    /// when the caller hasn't customized `noRepeatNgramSize`, the options
    /// are escalated to engage the no-repeat-n-gram slow path. Short clips
    /// are unaffected; explicit caller settings are honoured.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        options: Qwen3DecodingOptions
    ) -> String {
        let durationSeconds = sampleRate > 0
            ? Double(audio.count) / Double(sampleRate)
            : 0.0
        let effective = options.adaptedFor(audioDurationSeconds: durationSeconds)

        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)
        var audioEmbeds = audioEncoder(batchedFeatures)
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: effective.language,
            maxTokens: effective.maxTokens,
            context: effective.context,
            decodingOptions: effective
        )
    }

    /// Transcribe audio to text
    ///
    /// Long-input adaptive decoding: the legacy overload constructs a
    /// default `Qwen3DecodingOptions` and routes through the same
    /// length-gated escalation as the options-based path. Callers who pin
    /// `noRepeatNgramSize` via `Qwen3DecodingOptions` directly are out of
    /// scope here (they reach the other overload).
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448,
        context: String? = nil
    ) -> String {
        let durationSeconds = sampleRate > 0
            ? Double(audio.count) / Double(sampleRate)
            : 0.0
        let baseOptions = Qwen3DecodingOptions(
            maxTokens: maxTokens, language: language, context: context)
        let effective = baseOptions.adaptedFor(audioDurationSeconds: durationSeconds)

        // Extract mel features
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // Add batch dimension: [mel, time] -> [1, mel, time]
        let batchedFeatures = melFeatures.expandedDimensions(axis: 0)

        // Encode audio - returns [time, features] without batch dim (matching Python)
        var audioEmbeds = audioEncoder(batchedFeatures)

        // Add batch dimension for consistency: [time, features] -> [1, time, features]
        audioEmbeds = audioEmbeds.expandedDimensions(axis: 0)

        // Check if text decoder is loaded
        guard let textDecoder = textDecoder else {
            let shape = audioEmbeds.shape
            return "[Audio encoded: \(shape)] - Text decoder not loaded"
        }

        // Long-form audio that triggered escalation routes through the
        // options-aware codepath (which calls `isGreedyFastPath` and falls
        // out to `generateSlow`); short clips with default greedy take the
        // legacy fast-path call shape, bit-identical to today.
        // Mirror `isGreedyFastPath` exactly so the two routes stay in sync
        // even if a fourth decoder knob is added later.
        if !Self.isGreedyFastPath(effective) {
            return generateText(
                audioEmbeds: audioEmbeds,
                textDecoder: textDecoder,
                language: effective.language,
                maxTokens: effective.maxTokens,
                context: effective.context,
                decodingOptions: effective
            )
        }
        return generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: textDecoder,
            language: effective.language,
            maxTokens: effective.maxTokens,
            context: effective.context
        )
    }

    /// Generate text from audio embeddings.
    ///
    /// When `decodingOptions` is supplied, the decoder loop applies an
    /// HF-style repetition penalty, an optional no-repeat n-gram mask, and
    /// optional temperature sampling before each token selection. With the
    /// default `Qwen3DecodingOptions()` (repetition=1.0, no-repeat=0,
    /// temperature=0) behaviour is bit-identical to plain greedy.
    func generateText(
        audioEmbeds: MLXArray,
        textDecoder: QuantizedTextModel,
        language: String?,
        maxTokens: Int,
        context: String? = nil,
        decodingOptions: Qwen3DecodingOptions = Qwen3DecodingOptions()
    ) -> String {
        // Special token IDs
        let imStartId = 151644
        let imEndId = 151645
        let audioStartId = 151669
        let audioEndId = 151670
        let audioPadId = 151676
        let asrTextId = 151704
        let newlineId = 198

        // Token IDs for "system", "user", "assistant"
        let systemId = 8948
        let userId = 872
        let assistantId = 77091

        // Number of audio tokens (from audio encoder output)
        let numAudioTokens = audioEmbeds.dim(1)

        // Build input_ids array with audio_pad placeholder tokens
        var inputIds: [Int32] = []

        // <|im_start|>system\n{context}<|im_end|>\n
        inputIds.append(contentsOf: [imStartId, systemId, newlineId].map { Int32($0) })
        if let context = context, !context.isEmpty, let tokenizer = tokenizer {
            let contextTokens = tokenizer.encode(context)
            inputIds.append(contentsOf: contextTokens.map { Int32($0) })
        }
        inputIds.append(contentsOf: [imEndId, newlineId].map { Int32($0) })

        // <|im_start|>user\n<|audio_start|>
        inputIds.append(contentsOf: [imStartId, userId, newlineId, audioStartId].map { Int32($0) })

        // <|audio_pad|> * numAudioTokens (placeholder tokens that will be replaced)
        let audioStartIndex = inputIds.count
        for _ in 0..<numAudioTokens {
            inputIds.append(Int32(audioPadId))
        }
        let audioEndIndex = inputIds.count

        // <|audio_end|><|im_end|>\n
        inputIds.append(contentsOf: [audioEndId, imEndId, newlineId].map { Int32($0) })

        // <|im_start|>assistant\n
        inputIds.append(contentsOf: [imStartId, assistantId, newlineId].map { Int32($0) })

        // Add language hint if specified, then always add <|asr_text|> marker.
        // Without <|asr_text|>, the model doesn't know it should transcribe.
        // Without language hint, the model auto-detects and prepends "language XX" to output.
        if let lang = language, let tokenizer = tokenizer {
            let langPrefix = "language \(lang)"
            let langTokens = tokenizer.encode(langPrefix)
            inputIds.append(contentsOf: langTokens.map { Int32($0) })
        }
        inputIds.append(Int32(asrTextId))

        // Get text embeddings for all tokens
        let inputIdsTensor = MLXArray(inputIds).expandedDimensions(axis: 0)
        var inputEmbeds = textDecoder.embedTokens(inputIdsTensor)

        // Replace audio_pad token positions with actual audio embeddings
        let audioEmbedsTyped = audioEmbeds.asType(inputEmbeds.dtype)
        let beforeAudio = inputEmbeds[0..., 0..<audioStartIndex, 0...]
        let afterAudio = inputEmbeds[0..., audioEndIndex..., 0...]

        inputEmbeds = concatenated([beforeAudio, audioEmbedsTyped, afterAudio], axis: 1)

        // Initialize KV cache
        var cache: [(MLXArray, MLXArray)]? = nil

        // First pass: process the full input embeddings
        let (hiddenStates, newCache) = textDecoder(inputsEmbeds: inputEmbeds, cache: cache)
        cache = newCache

        // Get logits from the last position using embedding as LM head (tied weights)
        let seqLen = hiddenStates.dim(1)
        let lastHidden = hiddenStates[0..., (seqLen-1)..<seqLen, 0...]
        let logits = textDecoder.embedTokens.asLinear(lastHidden)

        // Greedy fast path uses a double-buffered asyncEval decode loop that
        // overlaps the GPU forward pass for token N+1 with the host-side
        // bookkeeping (EOS check + Swift array append) for token N. The
        // legacy slow path stays on the per-token CPU sync because it pulls
        // the full logits tensor to CPU for repetition / n-gram / temperature
        // manipulation, which would defeat the overlap.
        let generatedTokens: [Int32]
        if Self.isGreedyFastPath(decodingOptions) {
            generatedTokens = Self.generateGreedyAsyncEval(
                textDecoder: textDecoder,
                initialLogits: logits,
                cache: cache!,
                maxTokens: maxTokens
            )
        } else {
            generatedTokens = Self.generateSlow(
                textDecoder: textDecoder,
                initialLogits: logits,
                cache: cache!,
                maxTokens: maxTokens,
                options: decodingOptions
            )
        }

        // Decode tokens to text
        if let tokenizer = tokenizer {
            let rawText = tokenizer.decode(tokens: generatedTokens.map { Int($0) })
            // Strip "language XX<asr_text>" prefix if present (auto-detection output)
            if let range = rawText.range(of: "<asr_text>") {
                return String(rawText[range.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
            return rawText
        } else {
            // Fallback: return token IDs
            return generatedTokens.map { String($0) }.joined(separator: " ")
        }
    }

    /// Greedy with default options: temperature 0, no repetition penalty,
    /// no n-gram blocking. The double-buffered asyncEval loop only kicks
    /// in for this configuration so we can guarantee bit-identical token
    /// sequences vs. the legacy `argMax(...).item()` decoder.
    static func isGreedyFastPath(_ options: Qwen3DecodingOptions) -> Bool {
        return options.repetitionPenalty == 1.0
            && options.noRepeatNgramSize == 0
            && options.temperature == 0.0
    }

    /// Double-buffered greedy decode loop. The key trick is to keep the
    /// "next token" as a lazy 0-D `MLXArray` (the result of `argMax`),
    /// build the *next* step's forward pass on top of it (still lazy),
    /// then call `MLX.asyncEval` so the GPU starts computing step N+1
    /// before we sync step N's int32 to CPU. The host-side EOS check
    /// and `generatedTokens.append` then overlap with the in-flight GPU
    /// work for step N+1 instead of stalling between every token.
    ///
    /// Greedy correctness invariant: argMax is deterministic, so this
    /// produces the exact same token sequence as the legacy loop on
    /// matching inputs.
    static func generateGreedyAsyncEval(
        textDecoder: QuantizedTextModel,
        initialLogits: MLXArray,
        cache initialCache: [(MLXArray, MLXArray)],
        maxTokens: Int
    ) -> [Int32] {
        var generatedTokens: [Int32] = []
        guard maxTokens > 0 else { return generatedTokens }

        // Stage 0: argmax of the prefill's last logits. Stays lazy until
        // the first `.item()` below.
        //
        // Cast to int32 explicitly: MLX's `argmax` returns uint32, but the
        // legacy loop fed `embedTokens` an int32 tensor (built from a Swift
        // `Int32`). Quantized embedding lookup observably dispatches
        // differently on uint32 vs. int32, producing tokens that diverge
        // from the legacy path on a small fraction of inputs. Casting here
        // restores exact dtype parity, so greedy stays token-for-token
        // identical to the pre-optimisation decoder.
        var nextTokenArr = argMax(initialLogits, axis: -1).squeezed().asType(.int32)
        var cache = initialCache
        // Kick off the GPU on the first token (and the prefill cache that
        // step 1's graph will read from).
        asyncEval(nextTokenArr, cache)

        let eosToken = Int32(Qwen3ASRTokens.eosTokenId)

        for step in 0..<maxTokens {
            // Stage N+1's graph BEFORE syncing N. embedTokens expects a
            // [batch, seq] int32 tensor; nextTokenArr is 0-D so we expand
            // twice to [1, 1].
            //
            // Skip the speculative graph build on the LAST iteration —
            // we'd never consume it, and on tiny `maxTokens` the saved
            // GPU/host work matters.
            var nextTokenArrN1: MLXArray? = nil
            var cacheN1: [(MLXArray, MLXArray)]? = nil
            if step + 1 < maxTokens {
                let nextEmbed = textDecoder.embedTokens(
                    nextTokenArr.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
                )
                let (hiddenN1, newCacheN1) = textDecoder(inputsEmbeds: nextEmbed, cache: cache)
                let lastHiddenN1 = hiddenN1[0..., (-1)..., .ellipsis]
                let logitsN1 = textDecoder.embedTokens.asLinear(lastHiddenN1)
                let argN1 = argMax(logitsN1, axis: -1).squeezed().asType(.int32)
                // Kick GPU on N+1 (chains after the asyncEval that is
                // still computing N).
                asyncEval(argN1, newCacheN1)
                nextTokenArrN1 = argN1
                cacheN1 = newCacheN1
            }

            // Now sync N. By the time we reach `.item()` the GPU has
            // very likely finished N already, and the cost of this call
            // shrinks to a host-side memcpy of one int32 — which itself
            // overlaps with the in-flight N+1 forward pass.
            let nextToken = nextTokenArr.item(Int32.self)
            // Match the legacy loop semantics exactly: append the token
            // first, *then* break on EOS. The legacy loop emitted EOS
            // into `generatedTokens` whenever EOS was the most recent
            // pick, so greedy stays bit-identical.
            generatedTokens.append(nextToken)
            if nextToken == eosToken { break }

            guard let advancedCache = cacheN1, let advancedToken = nextTokenArrN1 else {
                // Final iteration without speculative work — nothing to
                // advance to; the loop will exit naturally next.
                break
            }
            cache = advancedCache
            nextTokenArr = advancedToken
        }
        return generatedTokens
    }

    /// Legacy decode loop kept verbatim for the non-greedy slow path.
    /// `pickNextToken` here pulls logits to CPU for repetition penalty,
    /// n-gram masking, and temperature sampling, so there's no benefit
    /// from `asyncEval` overlap.
    static func generateSlow(
        textDecoder: QuantizedTextModel,
        initialLogits: MLXArray,
        cache initialCache: [(MLXArray, MLXArray)],
        maxTokens: Int,
        options: Qwen3DecodingOptions
    ) -> [Int32] {
        var generatedTokens: [Int32] = []
        guard maxTokens > 0 else { return generatedTokens }
        var cache: [(MLXArray, MLXArray)]? = initialCache

        var nextToken = Self.pickNextToken(
            logits: initialLogits,
            generatedSoFar: generatedTokens,
            options: options
        )
        generatedTokens.append(nextToken)

        for _ in 1..<maxTokens {
            if nextToken == Int32(Qwen3ASRTokens.eosTokenId) { break }

            let tokenEmbeds = textDecoder.embedTokens(
                MLXArray([nextToken]).expandedDimensions(axis: 0)
            )
            let (hiddenStates, newCache) = textDecoder(inputsEmbeds: tokenEmbeds, cache: cache)
            cache = newCache

            let lastHiddenNext = hiddenStates[0..., (-1)..., .ellipsis]
            let logits = textDecoder.embedTokens.asLinear(lastHiddenNext)
            nextToken = Self.pickNextToken(
                logits: logits,
                generatedSoFar: generatedTokens,
                options: options
            )
            generatedTokens.append(nextToken)
        }
        return generatedTokens
    }

    // MARK: - Decoder knobs

    /// Pick the next token from a logits tensor, applying repetition
    /// penalty, no-repeat n-gram masking, and optional temperature sampling.
    ///
    /// With default options (repetition=1.0, noRepeat=0, temperature=0) the
    /// result is the same `argMax` the decoder used pre-refactor.
    /// Implementation pulls logits to CPU (a 1-D Float array of vocab size)
    /// so we can manipulate entries in-place without fighting MLX indexing.
    ///
    /// Access is `internal static` (not `private`) so
    /// ``Qwen3DecodingOptionsTests`` can exercise the sampler directly via
    /// ``@testable import Qwen3ASR`` — there is no GPU or model download
    /// involved so the path is trivially unit-testable once reachable.
    static func pickNextToken(
        logits: MLXArray,
        generatedSoFar: [Int32],
        options: Qwen3DecodingOptions
    ) -> Int32 {
        // Fast path — pure greedy, no modifications.
        if options.repetitionPenalty == 1.0,
           options.noRepeatNgramSize == 0,
           options.temperature == 0 {
            return argMax(logits, axis: -1).squeezed().item(Int32.self)
        }

        // Pull logits to CPU. `logits` is [1, 1, vocabSize]; after squeeze
        // and conversion we have a plain `[Float]` of length vocabSize.
        let flat = logits.squeezed().asType(.float32)
        let vocabSize = flat.size
        var scores: [Float] = flat.asArray(Float.self)
        precondition(scores.count == vocabSize, "pickNextToken: vocab size mismatch")

        // Repetition penalty: divide logits for already-generated tokens.
        if options.repetitionPenalty > 1.0 && !generatedSoFar.isEmpty {
            let penalty = options.repetitionPenalty
            for token in Set(generatedSoFar) {
                let idx = Int(token)
                guard idx >= 0, idx < vocabSize else { continue }
                let v = scores[idx]
                // Positive logits divide; negative logits multiply — matches
                // HuggingFace's implementation so the penalty always reduces
                // the probability of the repeated token.
                scores[idx] = v > 0 ? v / penalty : v * penalty
            }
        }

        // No-repeat-ngram: any next token whose emission would form a
        // repeated n-gram of size N gets pushed to -infinity.
        let n = options.noRepeatNgramSize
        if n > 0 && generatedSoFar.count >= n - 1 {
            let lastPrefix = Array(generatedSoFar.suffix(n - 1))
            // Walk every position where `lastPrefix` already appeared —
            // the token that followed it becomes forbidden as the NEXT
            // token now.
            if generatedSoFar.count >= n {
                for i in 0...(generatedSoFar.count - n) {
                    let window = Array(generatedSoFar[i..<(i + n - 1)])
                    guard window == lastPrefix else { continue }
                    let forbidden = Int(generatedSoFar[i + n - 1])
                    if forbidden >= 0 && forbidden < vocabSize {
                        scores[forbidden] = -.infinity
                    }
                }
            }
        }

        // Temperature sampling via Gumbel-max trick:
        // argmax(logits/T + Gumbel(0,1)) ~ categorical(softmax(logits/T)).
        if options.temperature > 0 {
            let t = options.temperature
            for i in 0..<vocabSize {
                let u = Float.random(in: 1e-6...1.0)
                scores[i] = scores[i] / t - Float.log(-Float.log(u))
            }
        }

        // Argmax of the adjusted scores.
        var bestIdx = 0
        var bestScore = -Float.infinity
        for i in 0..<vocabSize where scores[i] > bestScore {
            bestScore = scores[i]
            bestIdx = i
        }
        return Int32(bestIdx)
    }
}

// MARK: - Backward Compatibility (delegates to HuggingFaceDownloader)

public extension Qwen3ASRModel {
    static func sanitizedCacheKey(for modelId: String) -> String {
        HuggingFaceDownloader.sanitizedCacheKey(for: modelId)
    }

    static func validatedRemoteFileName(_ file: String) throws -> String {
        try HuggingFaceDownloader.validatedRemoteFileName(file)
    }

    static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        try HuggingFaceDownloader.validatedLocalPath(directory: directory, fileName: fileName)
    }
}

// MARK: - Model Size Detection

/// Supported ASR model sizes
public enum ASRModelSize {
    case small  // 0.6B
    case large  // 1.7B

    /// Default model IDs on HuggingFace
    public var defaultModelId: String {
        switch self {
        case .small: return "aufklarer/Qwen3-ASR-0.6B-MLX-4bit"
        case .large: return "aufklarer/Qwen3-ASR-1.7B-MLX-8bit"
        }
    }

    /// Audio encoder config for this model size
    public var audioConfig: Qwen3AudioEncoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Text decoder config for this model size and quantization bits
    public func textConfig(bits: Int) -> TextDecoderConfig {
        switch (self, bits) {
        case (.small, 8): return .small8bit
        case (.small, _): return .small
        case (.large, 8): return .large8bit
        case (.large, _): return .large
        }
    }

    /// Text decoder config for this model size (default bits)
    public var textConfig: TextDecoderConfig {
        switch self {
        case .small: return .small
        case .large: return .large
        }
    }

    /// Detect model size from a HuggingFace model ID
    public static func detect(from modelId: String) -> ASRModelSize {
        if modelId.contains("1.7B") || modelId.contains("1.7b") {
            return .large
        }
        return .small
    }

    /// Detect quantization bits from a HuggingFace model ID.
    /// Returns 4 by default for 0.6B, 8 for 1.7B if not specified.
    public static func detectBits(from modelId: String) -> Int {
        let lower = modelId.lowercased()
        if lower.contains("8bit") || lower.contains("8-bit") {
            return 8
        }
        if lower.contains("4bit") || lower.contains("4-bit") {
            return 4
        }
        // Default: 4 for small, 8 for large (backwards-compatible)
        let size = detect(from: modelId)
        return size == .large ? 8 : 4
    }
}

// MARK: - Memory guards (Bug 4b/4f support)

internal enum Qwen3ASRMemory {
    /// Threshold below which the 1.7B variant triggers a load-time RAM
    /// warning. Total physical memory is the pragmatic signal — observed
    /// hangs cluster on 8/16 GB Macs with other apps open; 24 GB+ has
    /// consistently completed inference in our benchmarks. Exposed
    /// `internal` so unit tests can pin the threshold.
    static let largeModelRAMWarningThresholdGB: Double = 24.0

    /// MLX cache ceiling applied when loading the 1.7B variant. mlx-swift's
    /// default tracks `recommendedMaxWorkingSetSize` which on a 16 GB Mac
    /// can grow to several GB under sustained decoding — pushing residency
    /// past unified-memory headroom and triggering swap. We bound the
    /// scratch pool to `min(4 GB, 25% of physical RAM)`: well above the
    /// per-token decoder working set, but small enough to leave room for
    /// the OS and other apps.
    static func cacheLimitForLarge(physicalMemoryBytes: Int) -> Int {
        let fourGB = 4 * 1024 * 1024 * 1024
        let quarterRAM = physicalMemoryBytes / 4
        return max(0, min(fourGB, quarterRAM))
    }

    /// True when the 1.7B variant should print the soft RAM warning. Total
    /// (not available) RAM is the pragmatic signal — see threshold doc.
    static func shouldWarnForLarge(physicalMemoryBytes: UInt64) -> Bool {
        let physicalGB = Double(physicalMemoryBytes) / 1_073_741_824.0
        return physicalGB < largeModelRAMWarningThresholdGB
    }

    /// Emit a human-readable RAM-pressure warning to stderr (NDJSON-IPC safe).
    /// Naming the alternative model IDs so the user can copy-paste.
    static func emitLargeRAMWarning(physicalMemoryBytes: UInt64) {
        let physicalGB = Double(physicalMemoryBytes) / 1_073_741_824.0
        let msg = """
            [Qwen3ASR] Warning: loading 1.7B variant on \(String(format: "%.0f", physicalGB)) GB Mac.
            [Qwen3ASR]   The 1.7B model has been observed to swap and stall on <\(Int(largeModelRAMWarningThresholdGB)) GB systems
            [Qwen3ASR]   when other apps are running. If you see a hang, consider:
            [Qwen3ASR]     aufklarer/Qwen3-ASR-0.6B-MLX-8bit   (recommended for 8-16 GB)
            [Qwen3ASR]     aufklarer/Qwen3-ASR-1.7B-MLX-4bit   (smaller, similar quality)
            """
        FileHandle.standardError.write(Data((msg + "\n").utf8))
    }

    /// Format memory readings (active / cache / peak in bytes) for
    /// human-readable logging. Centralized so the formatting is consistent
    /// across load-time + transcribe-time telemetry. Sizes are reported in
    /// MB. This overload takes `Int` directly so unit tests don't depend on
    /// `MLX.Memory.Snapshot`'s sealed initializer.
    static func formatSnapshot(active: Int, cache: Int, peak: Int, label: String) -> String {
        let mb: (Int) -> String = { String(format: "%.0f MB", Double($0) / 1_048_576.0) }
        return "[Qwen3ASR][mem] \(label): "
            + "active=\(mb(active)) "
            + "cache=\(mb(cache)) "
            + "peak=\(mb(peak))"
    }

    /// Production-callsite overload that adapts a live `MLX.Memory.Snapshot`.
    static func formatSnapshot(_ s: MLX.Memory.Snapshot, label: String) -> String {
        formatSnapshot(
            active: s.activeMemory, cache: s.cacheMemory, peak: s.peakMemory,
            label: label)
    }
}

// MARK: - Model Loading

public extension Qwen3ASRModel {
    /// Load model from HuggingFace hub with automatic weight downloading
    static func fromPretrained(
        modelId: String = "aufklarer/Qwen3-ASR-0.6B-MLX-4bit",
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Qwen3ASRModel {
        progressHandler?(0.0, "Downloading model...")

        // Auto-detect model size and quantization bits from model ID
        let modelSize = ASRModelSize.detect(from: modelId)
        let detectedBits = ASRModelSize.detectBits(from: modelId)

        // Bug 4b: soft RAM warning for the 1.7B variant. Emit BEFORE the
        // download so users see it on the first byte, not after a 1.7 GB
        // transfer. Routed to stderr to keep stdout clean for NDJSON-IPC
        // consumers (speech-studio sidecar).
        if modelSize == .large {
            let physical = ProcessInfo.processInfo.physicalMemory
            if Qwen3ASRMemory.shouldWarnForLarge(physicalMemoryBytes: physical) {
                Qwen3ASRMemory.emitLargeRAMWarning(physicalMemoryBytes: physical)
            }
        }

        // Bug 4f: pre-load memory snapshot for telemetry. Cheap to call;
        // gives us a baseline to compare against the post-load snapshot.
        let memBeforeLoad = MLX.Memory.snapshot()

        // Get cache directory
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download weights and tokenizer files (skips files that already exist on disk)
        // Download is the slowest part — give it 0-80% of progress
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading weights...")
            }
        )

        progressHandler?(0.80, "Loading tokenizer...")

        // Create model with appropriate config for detected size and bits
        let model = Qwen3ASRModel(
            audioConfig: modelSize.audioConfig,
            textConfig: modelSize.textConfig(bits: detectedBits)
        )

        // Load tokenizer from vocab.json
        let vocabPath = cacheDir.appendingPathComponent("vocab.json")
        if FileManager.default.fileExists(atPath: vocabPath.path) {
            let tokenizer = Qwen3Tokenizer()
            try tokenizer.load(from: vocabPath)
            model.setTokenizer(tokenizer)
        }

        progressHandler?(0.85, "Loading audio encoder weights...")

        // Load audio encoder weights
        try WeightLoader.loadWeights(into: model.audioEncoder, from: cacheDir)

        progressHandler?(0.92, "Loading text decoder weights...")

        // Initialize and load text decoder
        model.initializeTextDecoder()
        if let textDecoder = model.textDecoder {
            try WeightLoader.loadTextDecoderWeights(into: textDecoder, from: cacheDir)
        }

        MetalBudget.pinMemory()

        // Bug 4b: cap MLX scratch pool for the 1.7B variant. Default cache
        // limit tracks `recommendedMaxWorkingSetSize` which on a 16 GB Mac
        // can grow to several GB during sustained decoding and trigger
        // swap. Bounding to `min(4 GB, 25% of physical RAM)` leaves enough
        // headroom for per-token decoder working set while keeping the
        // total residency under the OS jetsam threshold. 0.6B path is
        // unchanged.
        //
        // Process-global cap leak fix (adversarial review): we save the
        // prior limit on the model instance and restore it in `unload()`,
        // so co-loaded models in the same process (e.g. PersonaPlex
        // loading ASR + LM + TTS) inherit our cap only for the lifetime
        // of the loaded ASR. Stacks correctly across multiple ASR
        // instances: each save captures whatever was active when it
        // loaded, and each unload pops its own saved value.
        if modelSize == .large {
            let physical = Int(ProcessInfo.processInfo.physicalMemory)
            let newCap = Qwen3ASRMemory.cacheLimitForLarge(physicalMemoryBytes: physical)
            // Only apply the cap if it would lower the current limit —
            // never raise a limit a caller has already chosen for itself.
            let currentLimit = MLX.Memory.cacheLimit
            if newCap > 0 && newCap < currentLimit {
                model.savedMLXCacheLimit = currentLimit
                MLX.Memory.cacheLimit = newCap
            }
        }

        // Bug 4f: post-load memory snapshot. Difference vs `memBeforeLoad`
        // is the model's load-time footprint (weights + activations +
        // metallib JIT). Useful for tuning the cache cap and for spotting
        // load-time regressions in PRs.
        let memAfterLoad = MLX.Memory.snapshot()
        AudioLog.modelLoading.info("\(Qwen3ASRMemory.formatSnapshot(memBeforeLoad, label: "pre-load"))")
        AudioLog.modelLoading.info("\(Qwen3ASRMemory.formatSnapshot(memAfterLoad, label: "post-load"))")
        // Display max(0, delta): MLX can free cached weights between
        // snapshots, which makes "active" go down; clamp at 0 so the
        // load-delta label stays meaningful in logs.
        let loadActiveDelta = max(0, memAfterLoad.activeMemory - memBeforeLoad.activeMemory)
        AudioLog.modelLoading.info(
            "[Qwen3ASR][mem] load delta (active): \(String(format: "%.0f MB", Double(loadActiveDelta) / 1_048_576.0))")

        progressHandler?(1.0, "Ready")

        return model
    }
}
