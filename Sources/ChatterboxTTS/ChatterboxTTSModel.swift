import AudioCommon
import Foundation
import MLX
import MLXNN

// MARK: - ChatterboxTTSModel — end-to-end multilingual voice cloning
//
// Wires the full Chatterbox pipeline:
//
//   text  → MTLTokenizer → text tokens
//   ref   → VoiceEncoder (16 kHz) → speaker emb (T3 conditioning)
//         → S3TokenizerV2 (16 kHz) → T3 prompt-speech tokens
//   T3.inference(text tokens, speaker emb, prompt, emotion) → speech tokens
//         → drop-invalid / filter (< SPEECH_VOCAB_SIZE)
//   ChatterboxS3Gen.synthesize(speech tokens, ref) → 24 kHz waveform
//
// Mirrors the reference `prepare_conditionals` + `generate`.

public enum ChatterboxModelError: Error, LocalizedError {
    case missingFile(String)
    case unsupportedLanguage(String)

    public var errorDescription: String? {
        switch self {
        case let .missingFile(p): return "Chatterbox: required file not found: \(p)"
        case let .unsupportedLanguage(l):
            return "Chatterbox: language '\(l)' is not supported"
        }
    }
}

public final class ChatterboxTTSModel {
    public let tokenizer: MTLTokenizer
    public let voiceEncoder: ChatterboxVoiceEncoder
    public let t3: ChatterboxT3
    public let s3gen: ChatterboxS3Gen

    // Token constants (mirror the reference config + chatterbox.py).
    /// `[START]` text token prepended by `generate()`.
    public static let startTextToken = 255
    /// `[STOP]` text token appended by `generate()`.
    public static let stopTextToken = 0
    /// Codes >= this are invalid for S3Gen (6561 = SOS, 6562 = EOS).
    public static let speechVocabSize = 6561
    /// T3 conditioning prompt is capped to this many reference tokens.
    public static let speechCondPromptLen = 150

    // Reference conditioning lengths (chatterbox.py ENC_COND_LEN / DEC_COND_LEN).
    /// 6 s at 16 kHz — the T3-conditioning reference window.
    public static let encCondLen = 6 * 16000
    /// 10 s at 24 kHz — the S3Gen-conditioning reference window.
    public static let decCondLen = 10 * 24000

    public init(
        tokenizer: MTLTokenizer,
        voiceEncoder: ChatterboxVoiceEncoder,
        t3: ChatterboxT3,
        s3gen: ChatterboxS3Gen
    ) {
        self.tokenizer = tokenizer
        self.voiceEncoder = voiceEncoder
        self.t3 = t3
        self.s3gen = s3gen
    }

    // MARK: - Loading

    /// Default published bundle.
    public static let defaultModelId = "aufklarer/Chatterbox-Multilingual-MLX-fp16"
    /// Shared S3TokenizerV2 weights repo.
    public static let s3TokenizerModelId = "mlx-community/S3TokenizerV2"

    /// Load from a local bundle directory containing `model.safetensors` and
    /// `tokenizer.json`. The S3TokenizerV2 weights and (optionally) the conformer
    /// block weights are loaded from extra paths.
    ///
    /// - Parameters:
    ///   - bundleDir: directory with `model.safetensors` + `tokenizer.json`.
    ///   - s3TokenizerWeights: `model.safetensors` for S3TokenizerV2 (encoder /
    ///     quantizer). Required — the bundle does not carry tokenizer weights.
    ///   - conformerWeights: optional checkpoint carrying the flow-encoder
    ///     conformer blocks (`flow.encoder.encoders.* / up_encoders.*`). The
    ///     the bundle ships these as `conformer.safetensors`, which is used by
    ///     default; pass an explicit URL only to override the source.
    public static func fromPretrained(
        localDir bundleDir: URL,
        s3TokenizerWeights: URL,
        conformerWeights: URL? = nil
    ) throws -> ChatterboxTTSModel {
        let fm = FileManager.default
        let modelPath = bundleDir.appendingPathComponent("model.safetensors")
        for p in [modelPath.path, s3TokenizerWeights.path] where !fm.fileExists(atPath: p) {
            throw ChatterboxModelError.missingFile(p)
        }
        // The conformer blocks ship alongside the bundle; fall back to them when
        // the caller doesn't override, so the bundle loads self-contained.
        var conformerWeights = conformerWeights
        if conformerWeights == nil {
            let bundled = bundleDir.appendingPathComponent("conformer.safetensors")
            if fm.fileExists(atPath: bundled.path) { conformerWeights = bundled }
        }

        let tokenizer = try MTLTokenizer(modelFolder: bundleDir)

        // Load + split the combined bundle by component prefix. Cast to fp32 to
        // match the fp32 component goldens and keep the CFM diffusion loop stable.
        let raw = try MLX.loadArrays(url: modelPath)
        var ve: [String: MLXArray] = [:]
        var t3w: [String: MLXArray] = [:]
        var spk: [String: MLXArray] = [:]      // CAMPPlus (s3gen.speaker_encoder.*)
        var mel2wav: [String: MLXArray] = [:]  // S3GenVocoder (s3gen.mel2wav.*)
        var flowEnc: [String: MLXArray] = [:]  // S3GenConformer (flow encoder + proj/embed/spk)
        var cfm: [String: MLXArray] = [:]      // MatchaCFM (flow.decoder.estimator.*)

        for (k, raw32) in raw {
            let v = raw32.asType(.float32)
            if k.hasPrefix("ve.") {
                ve[String(k.dropFirst(3))] = v
            } else if k.hasPrefix("t3.") {
                t3w[String(k.dropFirst(3))] = v
            } else if k.hasPrefix("s3gen.speaker_encoder.") {
                spk[String(k.dropFirst("s3gen.speaker_encoder.".count))] = v
            } else if k.hasPrefix("s3gen.mel2wav.") {
                mel2wav[String(k.dropFirst("s3gen.mel2wav.".count))] = v
            } else if k.hasPrefix("s3gen.flow.decoder.") {
                // MatchaCFM expects keys under `estimator.*`; drop `decoder.`.
                cfm[String(k.dropFirst("s3gen.flow.decoder.".count))] = v
            } else if k.hasPrefix("s3gen.flow.") {
                flowEnc[String(k.dropFirst("s3gen.flow.".count))] = v
            }
        }

        // Instantiate components.
        let voiceEncoder = ChatterboxVoiceEncoder()
        try voiceEncoder.update(parameters: ModuleParameters.unflattened(ve), verify: .all)

        let t3 = ChatterboxT3()
        try t3.update(parameters: ModuleParameters.unflattened(t3w), verify: .all)

        let speakerEncoder = CAMPPlus()
        try speakerEncoder.loadWeights(spk)
        // CAMPPlus has BatchNorm layers — switch to inference mode so they use the
        // loaded running stats (training-mode BatchNorm on a single utterance
        // collapses to all-zeros).
        speakerEncoder.train(false)

        let vocoder = S3GenVocoder()
        try vocoder.loadWeights(mel2wav)

        // The flow-encoder param tree includes the rel-pos conformer block stacks
        // (`encoders.* / up_encoders.*`), which the published bundle omits — they
        // live only in the original checkpoint. When `conformerWeights` is given,
        // merge them in and load everything with `verify: .all`; otherwise load
        // the bundle pieces alone with a relaxed verify so the (unweighted)
        // conformer blocks stay at init, mirroring the reference's `strict=False`.
        let flow = S3GenConformer()
        var flowAll = flowEnc
        if let conformerWeights {
            for (k, v) in try Self.conformerBlockWeights(from: conformerWeights) {
                flowAll[k] = v
            }
            try flow.update(parameters: ModuleParameters.unflattened(flowAll), verify: .all)
        } else {
            // Skip `allModelKeysSet` so the missing conformer blocks are tolerated.
            try flow.update(
                parameters: ModuleParameters.unflattened(flowAll),
                verify: [.noUnusedKeys, .shapeMismatch])
        }

        let matcha = MatchaCFM()
        try matcha.update(parameters: ModuleParameters.unflattened(cfm), verify: .all)

        // S3TokenizerV2 weights (separate repo).
        let tokRaw = try MLX.loadArrays(url: s3TokenizerWeights)
        let tokW = tokRaw.mapValues { $0.asType(.float32) }
        let s3Tokenizer = S3TokenizerV2()
        try s3Tokenizer.update(parameters: ModuleParameters.unflattened(tokW), verify: .all)

        let s3gen = ChatterboxS3Gen(
            speakerEncoder: speakerEncoder, tokenizer: s3Tokenizer,
            flow: flow, cfm: matcha, vocoder: vocoder)

        // Force a lazy eval-free graph build now (parameters resolved on first use).
        MLX.eval(voiceEncoder.parameters())
        MLX.eval(t3.parameters())

        return ChatterboxTTSModel(
            tokenizer: tokenizer, voiceEncoder: voiceEncoder, t3: t3, s3gen: s3gen)
    }

    /// Download (or reuse cached) the published bundle + the S3TokenizerV2 repo,
    /// then load. The bundle ships its flow-encoder conformer blocks as
    /// `conformer.safetensors`, so cloning needs no external checkpoint.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        s3TokenizerModelId: String = s3TokenizerModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        conformerWeights: URL? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ChatterboxTTSModel {
        let bundleDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let tokenizerDir = try HuggingFaceDownloader.getCacheDirectory(
            for: s3TokenizerModelId, cacheDirName: "chatterbox-s3-tokenizer")

        let fm = FileManager.default
        if !fm.fileExists(atPath: bundleDir.appendingPathComponent("model.safetensors").path) {
            progressHandler?(0.0, "Downloading \(modelId)...")
            // No `.safetensors` in additionalFiles: that keeps downloadWeights'
            // automatic `*.safetensors` glob enabled, which fetches the bundle's
            // model + conformer (+ tokenizer) weights together. Listing a
            // `.safetensors` here would disable that glob and drop model.safetensors.
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: bundleDir,
                additionalFiles: ["config.json", "tokenizer.json"],
                offlineMode: offlineMode
            ) { progressHandler?($0 * 0.7, "Downloading model...") }
        }
        if !fm.fileExists(atPath: tokenizerDir.appendingPathComponent("model.safetensors").path) {
            progressHandler?(0.7, "Downloading S3 tokenizer...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: s3TokenizerModelId, to: tokenizerDir,
                offlineMode: offlineMode
            ) { progressHandler?(0.7 + $0 * 0.2, "Downloading tokenizer...") }
        }

        progressHandler?(0.92, "Loading weights...")
        let model = try fromPretrained(
            localDir: bundleDir,
            s3TokenizerWeights: tokenizerDir.appendingPathComponent("model.safetensors"),
            conformerWeights: conformerWeights)
        progressHandler?(1.0, "Ready")
        return model
    }

    /// Extract the flow-encoder conformer block weights from a checkpoint that
    /// carries them, keyed to merge into the `S3GenConformer` parameter tree.
    ///
    /// The original checkpoint stores them as `flow.encoder.encoders.N.*` and
    /// `flow.encoder.up_encoders.N.*`; the Swift model holds the two stacks as
    /// top-level `encoders` / `upEncoders` arrays — so the keys become
    /// `encoders.N.*` and `upEncoders.N.*`. The blocks are pure Linear /
    /// LayerNorm / pos-bias parameters (no convs), so no transpose is needed;
    /// they're cast to float32 to match the rest of the loaded graph.
    private static func conformerBlockWeights(from url: URL) throws -> [String: MLXArray] {
        let raw = try MLX.loadArrays(url: url)
        var out: [String: MLXArray] = [:]
        let encPrefix = "flow.encoder.encoders."
        let upPrefix = "flow.encoder.up_encoders."
        for (k, v) in raw {
            if k.hasPrefix(encPrefix) {
                out["encoders." + k.dropFirst(encPrefix.count)] = v.asType(.float32)
            } else if k.hasPrefix(upPrefix) {
                out["upEncoders." + k.dropFirst(upPrefix.count)] = v.asType(.float32)
            }
        }
        return out
    }

    // MARK: - Inference

    /// Drop the SOS(6561)/EOS(6562) boundary markers from a flat token sequence,
    /// matching `drop_invalid_tokens`: keep `(after first SOS, before first EOS)`.
    static func dropInvalidTokens(_ tokens: [Int]) -> [Int] {
        let sos = speechVocabSize       // 6561
        let eos = speechVocabSize + 1   // 6562
        var start = 0
        if let s = tokens.firstIndex(of: sos) { start = s + 1 }
        var end = tokens.count
        if let e = tokens.firstIndex(of: eos) { end = e }
        guard start <= end else { return [] }
        return Array(tokens[start ..< end])
    }

    /// Clone a voice: synthesize `text` in the reference speaker's voice.
    ///
    /// - Parameters:
    ///   - referenceSamples: reference clip, mono.
    ///   - sampleRate: sample rate of `referenceSamples`.
    ///   - text: text to speak.
    ///   - languageId: e.g. "en", "ar", "hi", "de", "es", "fr", "it", "pt".
    ///   - exaggeration: emotion-advance scalar (T3 `emotion_adv`).
    ///   - cfgWeight: T3 classifier-free-guidance weight.
    ///   - temperature: T3 sampling temperature (0 = greedy).
    /// - Returns: 24 kHz mono waveform.
    public func clone(
        referenceSamples: [Float],
        sampleRate: Int,
        text: String,
        languageId: String,
        exaggeration: Float = 0.5,
        maxNewTokens: Int = 1000,
        temperature: Float = 0.8,
        topP: Float = 1.0,
        minP: Float = 0.05,
        repetitionPenalty: Float = 1.2,
        cfgWeight: Float = 0.5
    ) throws -> [Float] {
        let lang = languageId.lowercased()
        guard MTLTokenizer.frontendFreeLanguages.contains(lang) else {
            throw ChatterboxModelError.unsupportedLanguage(lang)
        }

        // --- prepare_conditionals: reference at the required sample rates ---
        // 24 kHz (truncated to 10 s) for the S3Gen prompt mel.
        let ref24kFull = sampleRate == ChatterboxS3Gen.sampleRate
            ? referenceSamples
            : AudioFileLoader.resample(referenceSamples, from: sampleRate, to: ChatterboxS3Gen.sampleRate)
        let ref24k = Array(ref24kFull.prefix(Self.decCondLen))
        // 24 kHz -> 16 kHz for the S3Gen tokenizer + CAMPPlus.
        let ref16kFromRef24k = AudioFileLoader.resample(
            ref24k, from: ChatterboxS3Gen.sampleRate, to: ChatterboxS3Gen.tokenSampleRate)
        // original -> 16 kHz (untruncated) for the VoiceEncoder; 6 s slice for T3 cond tokens.
        let ref16kFull = sampleRate == ChatterboxS3Gen.tokenSampleRate
            ? referenceSamples
            : AudioFileLoader.resample(referenceSamples, from: sampleRate, to: ChatterboxS3Gen.tokenSampleRate)
        let ref16kEnc = Array(ref16kFull.prefix(Self.encCondLen))

        // S3Gen conditioning (x-vector, prompt tokens, prompt mel).
        let s3Ref = s3gen.embedRef(refWav24k: ref24k, refWav16k: ref16kFromRef24k)

        // T3 conditioning: speaker emb (full 16 kHz) + prompt-speech tokens (6 s).
        let speakerEmb = voiceEncoder.embed(samples: ref16kFull)  // [256]
        let t3PromptTokens = Array(s3gen.tokenizer.encode(ref16kEnc).prefix(Self.speechCondPromptLen))

        // --- text tokenization: [sot] + ids + [eot] ---
        let ids = tokenizer.encode(text, languageId: lang)
        let textTokens = [Self.startTextToken] + ids + [Self.stopTextToken]

        // --- T3: text -> speech tokens ---
        let rawSpeech = t3.inference(
            textTokens: textTokens,
            speakerEmb: speakerEmb,
            promptSpeechTokens: t3PromptTokens,
            emotionAdv: exaggeration,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topP: topP,
            minP: minP,
            repetitionPenalty: repetitionPenalty,
            cfgWeight: cfgWeight)

        // drop SOS/EOS, then keep only valid codes (< SPEECH_VOCAB_SIZE), order-preserving.
        let dropped = Self.dropInvalidTokens(rawSpeech)
        let speechTokens = dropped.filter { $0 < Self.speechVocabSize }

        // --- S3Gen: speech tokens -> waveform ---
        return s3gen.synthesize(speechTokens: speechTokens, ref: s3Ref)
    }
}
