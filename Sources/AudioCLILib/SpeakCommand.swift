import Foundation
import AVFoundation
import ArgumentParser
import MLX
import Qwen3TTS
import CosyVoiceTTS
import VoxCPM2TTS
import IndexTTS2TTS
import F5TTS
import HiggsTTS
import IndicMioTTS
import MagpieTTS
import MagpieTTSCoreML
import AudioCommon
import SpeechRestoration

public struct SpeakCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Text-to-speech synthesis (Qwen3-TTS, CosyVoice, VoxCPM2, IndexTTS2, F5-TTS, Higgs, Indic-Mio, or Magpie). For CoreML, use the `qwen3-tts-coreml` subcommand."
    )

    @Argument(help: "Text to synthesize (omit when using --list-speakers or --batch-file)")
    public var text: String?

    @Option(name: .long, help: "TTS engine: qwen3 (default), cosyvoice, voxcpm2, indextts2, f5, higgs, indic-mio, magpie, or magpie-coreml")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, hindi, chinese, german, japanese, spanish, french, korean, russian, italian, portuguese). Default: english, or hindi for --engine indic-mio. Omit to use speaker's native dialect when --speaker is set.")
    public var language: String?

    @Flag(name: .long, help: "Enable streaming synthesis")
    public var stream: Bool = false

    @Flag(name: .long, help: "Play audio through default output device instead of (or in addition to) saving a file")
    public var play: Bool = false

    // MARK: - Qwen3-specific options

    @Option(name: .long, help: "[qwen3] Speaker voice (requires --model customVoice)")
    public var speaker: String?

    @Option(name: .long, help: "[qwen3] Style instruction (requires CustomVoice model)")
    public var instruct: String?

    @Option(name: .long, help: "Reference audio file for voice cloning (qwen3 Base, cosyvoice, voxcpm2, indextts2, f5, higgs, or indic-mio)")
    public var voiceSample: String?

    @Flag(name: .long, help: "Restore (denoise + dereverb) the voice-cloning reference with Sidon before cloning. Opt-in; preserves speaker identity. Applies to qwen3/cosyvoice/voxcpm2/f5/higgs/indic-mio references.")
    public var cleanReference: Bool = false

    @Option(name: .long, help: "Sidon variant for --clean-reference: fp16 (default) or int8")
    public var cleanReferenceVariant: String = "fp16"

    @Option(name: .long, help: "[qwen3] Model variant: base (default), base-8bit, 1.7b (bf16), 1.7b-8bit, customVoice, customVoice-bf16, or full HF model ID. Note: --speaker requires customVoice.")
    public var model: String = "base"

    @Flag(name: .long, help: "[qwen3] List available speakers and exit")
    public var listSpeakers: Bool = false

    @Option(name: .long, help: "[qwen3/indic-mio] Sampling temperature (default: 0.3)")
    public var temperature: Float = 0.3

    @Option(name: .long, help: "[qwen3/indic-mio] Top-k sampling")
    public var topK: Int = 50

    @Option(name: .long, help: "[qwen3/indic-mio] Maximum generated tokens (500 = ~40s audio for Qwen3)")
    public var maxTokens: Int = 500

    @Option(name: .long, help: "[qwen3] File with one text per line for batch synthesis")
    public var batchFile: String?

    @Option(name: .long, help: "[qwen3] Maximum batch size for parallel generation")
    public var batchSize: Int = 4

    @Option(name: .long, help: "[qwen3] Codec frames in first streamed chunk (default 3)")
    public var firstChunkFrames: Int = 3

    @Option(name: .long, help: "Codec frames per streamed chunk (default 25)")
    public var chunkFrames: Int = 25

    // MARK: - CosyVoice-specific options

    @Option(name: .long, help: "[cosyvoice] HuggingFace model ID. Set explicitly to bypass --cosyvoice-variant resolution.")
    public var modelId: String?

    @Option(name: .long, help: "[cosyvoice] Quantization variant: bf16/16bit (default), 8bit, 8bit-full. Resolves to aufklarer/CosyVoice3-0.5B-MLX-<variant>. Ignored when --model-id is set.")
    public var cosyvoiceVariant: String = "bf16"

    @Option(name: .long, help: "[cosyvoice] Speaker mapping: s1=alice.wav,s2=bob.wav")
    public var speakers: String?

    @Option(name: .long, help: "[cosyvoice] Style instruction (overrides default)")
    public var cosyInstruct: String?

    @Option(name: .long, help: "[cosyvoice] Silence gap between turns in seconds (default 0.2)")
    public var turnGap: Float = 0.2

    @Option(name: .long, help: "[cosyvoice] Crossfade between turns in seconds (default 0)")
    public var crossfade: Float = 0.0

    @Option(name: .long, help: "[cosyvoice] MLX seed applied before each synthesis call. Fixes the flow-matching noise + Gumbel sampling + HiFiGAN init phase, so repeated calls with the same speaker embedding produce near-identical prosody and timbre across sections. Useful for long-form narration cut into chunks.")
    public var seed: UInt64?

    @Option(name: .long, help: "[cosyvoice] Path to speech_tokenizer.safetensors (S3-Tokenizer-v3). When supplied, --voice-sample is upgraded from spk-only cloning (cos~0.83 cap) to upstream zero-shot conditioning with prompt_token + prompt_feat (preserves identity through emotion changes). Auto-detected in the bundle's cache dir if omitted.")
    public var cosySpeechTokenizer: String?

    @Option(name: .long, help: "[cosyvoice] Override the model cache directory. When supplied, the bundle is loaded directly from this directory instead of HuggingFace. Useful for testing locally-converted variants (e.g. an 8-bit LLM) without an HF push.")
    public var cosyBundleDir: String?

    @Option(name: .long, help: "[cosyvoice] Reference transcript: the text content of --voice-sample. Required for proper zero-shot voice cloning — without it the LLM has acoustic context but no idea what was said in the reference, and emits content-incorrect speech in the right voice.")
    public var cosyReferenceTranscript: String?

    // MARK: - VoxCPM2-specific options

    @Option(name: .long, help: "[voxcpm2] Quantization variant: bf16 (default), int8 (int4 was decommissioned). Resolves to aufklarer/VoxCPM2-MLX-<variant>.")
    public var voxcpm2Variant: String = "bf16"

    @Option(name: .long, help: "[voxcpm2] Style instruction")
    public var voxcpm2Instruct: String?

    @Option(name: .long, help: "[voxcpm2] Reference audio file for voice cloning")
    public var voxcpm2RefAudio: String?

    @Option(name: .long, help: "[voxcpm2] Prompt text for continuation")
    public var voxcpm2PromptText: String?

    @Option(name: .long, help: "[voxcpm2] Prompt audio file for continuation")
    public var voxcpm2PromptAudio: String?

    @Option(name: .long, help: "[voxcpm2] Classifier-free guidance scale (default 2.0)")
    public var voxcpm2CfgValue: Float = 2.0

    @Option(name: .long, help: "[voxcpm2] Diffusion timesteps per patch")
    public var voxcpm2Timesteps: Int = 10

    @Option(name: .long, help: "[voxcpm2] Maximum generated patches")
    public var voxcpm2MaxTokens: Int = 2000

    @Option(name: .long, help: "[voxcpm2] Minimum generated patches before early stop")
    public var voxcpm2MinTokens: Int = 2

    @Option(name: .long, help: "[voxcpm2] Streaming prefix patches retained for continuation")
    public var voxcpm2StreamingPrefixLen: Int = 4

    @Option(name: .long, help: "[voxcpm2] Warmup patches to skip before emitting audio")
    public var voxcpm2WarmupPatches: Int = 0

    // MARK: - IndexTTS2-specific options

    @Option(name: .long, help: "[indextts2] HuggingFace model ID for the exported MLX bundle")
    public var indextts2ModelId: String = IndexTTS2TTSModel.defaultModelId

    @Option(name: .long, help: "[indextts2] Load an exported bundle from this local directory instead of HuggingFace")
    public var indextts2BundleDir: String?

    @Option(name: .long, help: "[indextts2] Optional emotion/style reference audio. Defaults to --voice-sample when omitted.")
    public var indextts2EmotionAudio: String?

    @Option(name: .long, help: "[indextts2] Emotion preset or 8-value vector. Presets: eager, happy, excited, angry, sad, afraid, disgusted, melancholic, surprised, calm.")
    public var indextts2Emotion: String?

    @Option(name: .long, help: "[indextts2] Strength for --indextts2-emotion, from 0.0 to 1.0 (default 1.0).")
    public var indextts2EmotionWeight: Float = 1.0

    @Option(name: .long, help: "[indextts2] Speaking rate multiplier, from 0.5 to 1.5 (default 1.0). Values above 1.0 shorten generated speech.")
    public var indextts2SpeakingRate: Float = 1.0

    @Option(name: .long, help: "[indextts2] Optional cap for long internal pauses, in seconds, from 0.05 to 2.0. Omit to keep raw model timing.")
    public var indextts2MaxPause: Float?

    @Option(name: .customLong("indextts2-s2mel-steps"), help: "[indextts2] S2Mel flow steps (default 15, ear-validated; 25 matches upstream exactly)")
    public var indextts2S2MelSteps: Int = 15

    // MARK: - F5-TTS-specific options

    @Option(name: .long, help: "[f5] HuggingFace model ID for the exported MLX bundle")
    public var f5ModelId: String = F5TTSModel.defaultModelId

    @Option(name: .long, help: "[f5] Load an exported bundle from this local directory instead of HuggingFace")
    public var f5BundleDir: String?

    @Option(name: .long, help: "[f5] Reference transcript: the text content of --voice-sample. Required for zero-shot cloning.")
    public var f5ReferenceText: String?

    @Option(name: .long, help: "[f5] Flow-matching steps (default 16; use 32 for maximum fidelity)")
    public var f5Steps: Int = 16

    @Option(name: .long, help: "[f5] Classifier-free guidance strength (default 2.0)")
    public var f5CfgStrength: Float = 2.0

    @Option(name: .long, help: "[f5] Sway sampling coefficient (default -1.0)")
    public var f5Sway: Float = -1.0

    @Option(name: .long, help: "[f5] Speaking rate multiplier (default 1.0). Values above 1.0 shorten generated speech.")
    public var f5Speed: Float = 1.0

    @Option(name: .long, help: "[f5] MLX seed for deterministic flow sampling (default 0)")
    public var f5Seed: UInt64 = 0

    @Option(name: .long, help: "[f5] Reference RMS normalization target (default 0.1)")
    public var f5TargetRMS: Float = 0.1

    // MARK: - Higgs-specific options

    @Option(name: .long, help: "[higgs] HuggingFace model ID for the MLX bundle")
    public var higgsModelId: String = HiggsTTSModel.defaultModelId

    @Option(name: .long, help: "[higgs] Load a bundle from this local directory instead of HuggingFace")
    public var higgsBundleDir: String?

    @Option(name: .long, help: "[higgs] Reference transcript: the text content of --voice-sample (improves cloning)")
    public var higgsRefText: String?

    @Option(name: .long, help: "[higgs] Sampling temperature (default 0.8; the upstream default 1.0 is more variable)")
    public var higgsTemperature: Float = 0.8

    @Option(name: .long, help: "[higgs] Nucleus sampling threshold (off by default)")
    public var higgsTopP: Float?

    @Option(name: .long, help: "[higgs] Top-k sampling cutoff (off by default)")
    public var higgsTopK: Int?

    @Option(name: .long, help: "[higgs] Maximum generated audio frames (default 2048, 25 frames/second)")
    public var higgsMaxNewTokens: Int = 2048

    @Option(name: .long, help: "[higgs] MLX sampling seed (default 0)")
    public var higgsSeed: UInt64 = 0

    // MARK: - Indic-Mio-specific options

    @Option(name: .long, help: "[indic-mio] HuggingFace model ID")
    public var indicMioModelId: String = IndicMioTTSModel.defaultModelId

    @Option(name: .long, help: "[indic-mio] Top-p sampling")
    public var indicMioTopP: Float = 0.9

    @Option(name: .long, help: "[indic-mio] Repetition penalty")
    public var indicMioRepetitionPenalty: Float = 1.0

    @Option(name: .long, help: "[indic-mio] 128-float MioCodec global speaker embedding as JSON, comma-separated, or whitespace-separated floats")
    public var indicMioGlobalEmbedding: String?

    // MARK: - Magpie-specific options

    @Option(name: .long, help: "[magpie] Quantization variant: int8 (int4 was decommissioned). Resolves to aufklarer/Magpie-TTS-Multilingual-357M-MLX-int8.")
    public var magpieVariant: String = "int8"

    @Option(name: .long, help: "[magpie] Baked speaker: sofia (default), aria, jason, leo, john. No voice cloning.")
    public var magpieSpeaker: String = "sofia"

    @Option(name: .long, help: "[magpie] Sampling temperature (default 0.6)")
    public var magpieTemperature: Float = 0.6

    @Option(name: .long, help: "[magpie] Top-k sampling (default 80)")
    public var magpieTopK: Int = 80

    @Option(name: .long, help: "[magpie] Maximum codec frames (500 ≈ 23s)")
    public var magpieMaxFrames: Int = 500

    @Option(name: .long, help: "[magpie] Minimum frames before EOS (default 4)")
    public var magpieMinFrames: Int = 4

    @Flag(name: .long, help: "[magpie] Treat --text input as pre-phonemised IPA (skip text normalisation)")
    public var magpiePrephonemized: Bool = false

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    public init() {}

    /// Resolved language: explicit value or default "english"
    private var effectiveLanguage: String { language ?? "english" }

    /// Indic-Mio is an Indic/Hindi-first model; keep CLI defaults aligned with
    /// that instead of inheriting Qwen3's English default.
    private var effectiveIndicMioLanguage: String { language ?? "hindi" }

    /// Whether the user explicitly passed --language
    private var languageIsExplicit: Bool { language != nil }

    public func validate() throws {
        let eng = engine.lowercased()
        guard eng == "qwen3" || eng == "cosyvoice" || eng == "voxcpm2"
                || eng == "indextts2" || eng == "f5" || eng == "higgs" || eng == "indic-mio" || eng == "magpie" || eng == "magpie-coreml" else {
            throw ValidationError("--engine must be 'qwen3', 'cosyvoice', 'voxcpm2', 'indextts2', 'f5', 'higgs', 'indic-mio', 'magpie', or 'magpie-coreml'. For Qwen3-TTS CoreML, use the `qwen3-tts-coreml` subcommand.")
        }
        if text == nil && batchFile == nil && !listSpeakers {
            throw ValidationError("Either a text argument, --batch-file, or --list-speakers must be provided")
        }
        if eng == "voxcpm2" {
            if batchFile != nil || listSpeakers {
                throw ValidationError("--engine voxcpm2 only supports a single text input")
            }
            if (voxcpm2PromptAudio == nil) != (voxcpm2PromptText == nil) {
                throw ValidationError("--voxcpm2-prompt-audio and --voxcpm2-prompt-text must be provided together")
            }
        }
        if eng == "indextts2" {
            if batchFile != nil || listSpeakers {
                throw ValidationError("--engine indextts2 only supports a single text input")
            }
            if stream {
                throw ValidationError("--engine indextts2 does not support --stream yet")
            }
            if voiceSample == nil {
                throw ValidationError("--engine indextts2 requires --voice-sample because IndexTTS2 is a zero-shot voice-cloning model")
            }
            if indextts2Emotion != nil && indextts2EmotionAudio != nil {
                throw ValidationError("--indextts2-emotion and --indextts2-emotion-audio are mutually exclusive")
            }
            _ = try parseIndexTTS2EmotionControl()
            _ = try parseIndexTTS2SynthesisOptions()
            if cleanReference {
                throw ValidationError("--clean-reference is not wired for --engine indextts2 yet")
            }
            if speaker != nil {
                throw ValidationError("--engine indextts2 does not support --speaker; use --voice-sample")
            }
            if instruct != nil {
                throw ValidationError("--engine indextts2 does not support --instruct; use --indextts2-emotion-audio for a separate style reference")
            }
        }
        if eng == "f5" {
            if batchFile != nil || listSpeakers {
                throw ValidationError("--engine f5 only supports a single text input")
            }
            if stream {
                throw ValidationError("--engine f5 does not support --stream yet")
            }
            if voiceSample == nil {
                throw ValidationError("--engine f5 requires --voice-sample because F5-TTS is a zero-shot voice-cloning model")
            }
            if f5ReferenceText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
                throw ValidationError("--engine f5 requires --f5-reference-text with the transcript of --voice-sample")
            }
            _ = try parseF5SynthesisOptions()
            if speaker != nil {
                throw ValidationError("--engine f5 does not support --speaker; use --voice-sample")
            }
            if instruct != nil {
                throw ValidationError("--engine f5 does not support --instruct")
            }
        }
        if eng == "higgs" {
            if batchFile != nil || listSpeakers {
                throw ValidationError("--engine higgs only supports a single text input")
            }
            if stream {
                throw ValidationError("--engine higgs does not support --stream yet")
            }
            if higgsRefText != nil && voiceSample == nil {
                throw ValidationError("--higgs-ref-text requires --voice-sample")
            }
            if cleanReference && voiceSample == nil {
                throw ValidationError("--clean-reference requires --voice-sample")
            }
            _ = try parseHiggsSynthesisOptions()
            if speaker != nil {
                throw ValidationError("--engine higgs does not support --speaker; use --voice-sample for cloning")
            }
            if instruct != nil {
                throw ValidationError("--engine higgs does not support --instruct; use inline control tags such as '<|emotion:elation|>'")
            }
        }
        if eng == "indic-mio" {
            if batchFile != nil || listSpeakers {
                throw ValidationError("--engine indic-mio only supports a single text input")
            }
            if stream {
                throw ValidationError("--engine indic-mio does not support --stream yet")
            }
            if voiceSample != nil && indicMioGlobalEmbedding != nil {
                throw ValidationError("--engine indic-mio accepts either --voice-sample or --indic-mio-global-embedding, not both")
            }
            if cleanReference && voiceSample == nil {
                throw ValidationError("--clean-reference requires --voice-sample")
            }
            if speaker != nil {
                throw ValidationError("--engine indic-mio does not support --speaker; pass emotion markers in text, e.g. '<happy>'")
            }
            if instruct != nil {
                throw ValidationError("--engine indic-mio does not support --instruct; use inline/suffix markers such as '<sad>' or '<angry>'")
            }
            guard indicMioTopP > 0 && indicMioTopP <= 1 else {
                throw ValidationError("--indic-mio-top-p must be in (0, 1]")
            }
            guard indicMioRepetitionPenalty >= 1 else {
                throw ValidationError("--indic-mio-repetition-penalty must be >= 1")
            }
            if let embedding = indicMioGlobalEmbedding {
                _ = try loadIndicMioGlobalEmbedding(from: embedding)
            }
        }
        if eng == "magpie" || eng == "magpie-coreml" {
            if batchFile != nil {
                throw ValidationError("--engine \(eng) does not support --batch-file (single utterance only)")
            }
            if MagpieSpeaker(named: magpieSpeaker) == nil {
                throw ValidationError("--magpie-speaker must be one of sofia, aria, jason, leo, john (got '\(magpieSpeaker)')")
            }
            guard magpieVariant.lowercased() == "int8" else {
                throw ValidationError("--magpie-variant must be int8 (int4 was decommissioned) (got '\(magpieVariant)')")
            }
            // magpie-coreml validation: nothing extra needed beyond the
            // shared --magpie-* flag checks above. --stream is supported
            // via the dedicated 8-frame streaming nanocodec model.
            // Magpie has 5 baked speakers and no zero-shot speaker
            // conditioning in the model — reject voice-cloning / speaker
            // flags borrowed from the other engines so users don't think
            // the cloning silently worked.
            if voiceSample != nil {
                throw ValidationError(
                    "--engine \(eng) does not support --voice-sample. " +
                    "Magpie has 5 baked speakers and no zero-shot cloning. " +
                    "Use --magpie-speaker {sofia|aria|jason|leo|john} instead, " +
                    "or use --engine qwen3 / cosyvoice / voxcpm2 for cloning.")
            }
            if speaker != nil {
                throw ValidationError(
                    "--engine \(eng) does not support --speaker " +
                    "(that's a qwen3 CustomVoice flag). " +
                    "Use --magpie-speaker {sofia|aria|jason|leo|john}.")
            }
            if instruct != nil {
                throw ValidationError(
                    "--engine \(eng) does not support --instruct " +
                    "(style/instruction control is not in the Magpie model).")
            }
            if listSpeakers {
                // Friendlier than a silent no-op: print the 5 baked
                // speakers and return early.
                print("Magpie has 5 baked speakers (use with --magpie-speaker):")
                for spk in MagpieSpeaker.allCases {
                    let cliName: String
                    switch spk {
                    case .sofia:       cliName = "sofia"
                    case .aria:        cliName = "aria"
                    case .jason:       cliName = "jason"
                    case .leo:         cliName = "leo"
                    case .johnVanStan: cliName = "john"
                    }
                    print("  - \(cliName)  (\(spk.displayName))")
                }
                throw ExitCode(0)
            }
        }
    }

    public func run() throws {
        switch engine.lowercased() {
        case "cosyvoice":
            try runCosyVoice()
        case "voxcpm2":
            try runVoxCPM2()
        case "indextts2":
            try runIndexTTS2()
        case "f5":
            try runF5TTS()
        case "higgs":
            try runHiggs()
        case "indic-mio":
            try runIndicMio()
        case "magpie":
            try runMagpie()
        case "magpie-coreml":
            try runMagpieCoreML()
        default:
            try runQwen3()
        }
    }

    // MARK: - Reference cleanup (opt-in, Sidon)

    /// Load a voice-cloning reference and, when `--clean-reference` is set, run
    /// it through Sidon (denoise + dereverb) before returning it at
    /// `targetSampleRate`. When the flag is off this is a plain load — same
    /// behaviour as before.
    ///
    /// Sidon's pipeline is 16 kHz in / 48 kHz out; this loads at 16 kHz, restores,
    /// then resamples the 48 kHz result to whatever the engine wants. Implemented
    /// synchronously (semaphore bridge) so it slots into both the sync Qwen3 path
    /// and the async VoxCPM2 / CosyVoice paths without signature churn.
    func loadReference(path: String, targetSampleRate: Int) throws -> [Float] {
        guard cleanReference else {
            return try AudioFileLoader.load(
                url: URL(fileURLWithPath: path), targetSampleRate: targetSampleRate)
        }
        guard let variantEnum = SidonVariant(rawValue: cleanReferenceVariant) else {
            throw ValidationError(
                "--clean-reference-variant must be one of: "
                + SidonVariant.allCases.map { $0.rawValue }.joined(separator: ", "))
        }
        let raw16k = try AudioFileLoader.load(
            url: URL(fileURLWithPath: path),
            targetSampleRate: SpeechRestorer.inputSampleRate)
        print("  Cleaning reference with Sidon (\(cleanReferenceVariant))…")

        var restored48k: [Float] = []
        var thrown: Error?
        let sema = DispatchSemaphore(value: 0)
        Task {
            do {
                let restorer = try await SpeechRestorer.fromPretrained(variant: variantEnum)
                restored48k = try restorer.restore(
                    audio: raw16k, sampleRate: SpeechRestorer.inputSampleRate)
            } catch {
                thrown = error
            }
            sema.signal()
        }
        sema.wait()
        if let thrown { throw thrown }

        if targetSampleRate == SpeechRestorer.outputSampleRate { return restored48k }
        return AudioFileLoader.resample(
            restored48k, from: SpeechRestorer.outputSampleRate, to: targetSampleRate)
    }

    // MARK: - Magpie engine

    private func runMagpie() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for Magpie")
                throw ExitCode(1)
            }
            guard let speaker = MagpieSpeaker(named: magpieSpeaker) else {
                print("Error: invalid Magpie speaker '\(magpieSpeaker)'")
                throw ExitCode(1)
            }
            let variant: MagpieTTSVariant = .int8
            let language: MagpieLanguage =
                MagpieLanguage(code: effectiveLanguage) ?? .english

            print("Loading Magpie-TTS (\(variant.huggingFaceRepoId))...")
            let model = try await MagpieTTS.fromPretrained(
                variant: variant,
                progressHandler: { reportProgress($0, "Downloading") })

            let params = MagpieTTSParams(
                temperature: magpieTemperature,
                topK: magpieTopK,
                maxSteps: magpieMaxFrames,
                minFrames: magpieMinFrames,
                seed: seed)

            print("Synthesizing with Magpie (\(language.displayName), speaker \(speaker.displayName))...")
            let t0 = CFAbsoluteTimeGetCurrent()

            if stream {
                var collected: [Float] = []
                var chunkCount = 0
                var firstPacketLatency: Double?
                let audioStream = model.synthesizeStream(
                    text: inputText, speaker: speaker, language: language,
                    prephonemized: magpiePrephonemized, params: params)
                for try await chunk in audioStream {
                    if firstPacketLatency == nil {
                        firstPacketLatency = CFAbsoluteTimeGetCurrent() - t0
                    }
                    chunkCount += 1
                    collected.append(contentsOf: chunk.samples)
                    if verbose {
                        let ms = (chunk.elapsedTime ?? 0) * 1000
                        print("  chunk \(chunkCount): \(chunk.samples.count) samples @ \(Int(ms))ms")
                    }
                    if chunk.isFinal { break }
                }
                if let l = firstPacketLatency {
                    print(String(format: "  First-packet latency: %.0f ms", l * 1000))
                }
                try writeOrPlay(samples: collected, sampleRate: MagpieTTS.sampleRate, t0: t0)
            } else {
                let audio = try model.synthesize(
                    text: inputText, speaker: speaker, language: language,
                    prephonemized: magpiePrephonemized, params: params)
                try writeOrPlay(samples: audio, sampleRate: MagpieTTS.sampleRate, t0: t0)
            }
        }
    }

    // MARK: - Magpie CoreML engine

    /// CoreML variant of `--engine magpie`. Same 5 baked speakers, no
    /// streaming, no Japanese (CoreML bundle hasn't shipped JA tokenizer
    /// assets yet). When `--language ja` is requested with this engine, we
    /// transparently route to the MLX backend and log a stderr note so the
    /// user sees the fallback.
    private func runMagpieCoreML() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for Magpie")
                throw ExitCode(1)
            }
            guard let coreSpeaker = MagpieCoreMLSpeaker(named: magpieSpeaker) else {
                print("Error: invalid Magpie speaker '\(magpieSpeaker)'")
                throw ExitCode(1)
            }
            let mlxLang: MagpieLanguage =
                MagpieLanguage(code: effectiveLanguage) ?? .english

            // Japanese: CoreML bundle has no JA tokenizer/G2P assets.
            // Fall back to the MLX backend transparently.
            if mlxLang == .japanese {
                FileHandle.standardError.write(Data(
                    "[magpie-coreml] --language ja not supported by the CoreML bundle; using MLX backend.\n".utf8))
                let variant: MagpieTTSVariant = .int8
                let model = try await MagpieTTS.fromPretrained(
                    variant: variant,
                    progressHandler: { reportProgress($0, "Downloading MLX fallback") })
                let params = MagpieTTSParams(
                    temperature: magpieTemperature,
                    topK: magpieTopK,
                    maxSteps: magpieMaxFrames,
                    minFrames: magpieMinFrames,
                    seed: seed)
                let t0 = CFAbsoluteTimeGetCurrent()
                let audio = try model.synthesize(
                    text: inputText, speaker: coreSpeaker.mlxSpeaker, language: .japanese,
                    prephonemized: magpiePrephonemized, params: params)
                try writeOrPlay(samples: audio, sampleRate: MagpieTTS.sampleRate, t0: t0)
                return
            }

            guard let coreLang = MagpieCoreMLLanguage(mlx: mlxLang) else {
                // Should be unreachable since JA is the only excluded case.
                throw ExitCode(1)
            }

            print("Loading Magpie-TTS CoreML (\(MagpieCoreMLConstants.huggingFaceRepo))...")
            let model = try await MagpieTTSCoreML.fromPretrained(
                progressHandler: { reportProgress($0, "Downloading") })

            let params = MagpieCoreMLParams(
                temperature: magpieTemperature,
                topK: magpieTopK,
                maxSteps: magpieMaxFrames,
                minFrames: magpieMinFrames,
                seed: seed)

            // Greedy sampling on ANE produces broken audio (BF16
            // precision drift flips argmax). Warn so the user knows
            // why they should let sampling stay default.
            if magpieTemperature <= 1e-3
                && ProcessInfo.processInfo.environment["MAGPIE_COREML_COMPUTE"] == nil {
                FileHandle.standardError.write(Data(
                    "[magpie-coreml] --magpie-temperature 0 (greedy) is unreliable on ANE due to BF16 precision drift. Falling back to .all compute units for this run; set MAGPIE_COREML_COMPUTE=ane to force ANE anyway, or omit --magpie-temperature for the default stochastic sampling (the recommended path — ANE-fast and quality-correct).\n".utf8))
                setenv("MAGPIE_COREML_COMPUTE", "all", 1)
            }

            print("Synthesizing with Magpie CoreML (\(coreLang.mlx.displayName), speaker \(coreSpeaker.displayName))...")
            let t0 = CFAbsoluteTimeGetCurrent()
            if stream {
                var collected: [Float] = []
                var chunkCount = 0
                var firstPacketLatency: Double?
                let audioStream = model.synthesizeStream(
                    text: inputText, speaker: coreSpeaker, language: coreLang,
                    prephonemized: magpiePrephonemized, params: params)
                for try await chunk in audioStream {
                    if firstPacketLatency == nil && !chunk.samples.isEmpty {
                        firstPacketLatency = CFAbsoluteTimeGetCurrent() - t0
                    }
                    chunkCount += 1
                    collected.append(contentsOf: chunk.samples)
                    if verbose {
                        let ms = (chunk.elapsedTime ?? 0) * 1000
                        print("  chunk \(chunkCount): \(chunk.samples.count) samples @ \(Int(ms))ms")
                    }
                    if chunk.isFinal { break }
                }
                if let l = firstPacketLatency {
                    print(String(format: "  First-packet latency: %.0f ms", l * 1000))
                }
                try writeOrPlay(samples: collected, sampleRate: MagpieTTSCoreML.sampleRate, t0: t0)
            } else {
                let audio = try model.synthesize(
                    text: inputText, speaker: coreSpeaker, language: coreLang,
                    prephonemized: magpiePrephonemized, params: params)
                try writeOrPlay(samples: audio, sampleRate: MagpieTTSCoreML.sampleRate, t0: t0)
            }
        }
    }

    /// Shared "save or play" tail used by Magpie. The other engines have
    /// bespoke logic; Magpie's output is always 22.05 kHz mono PCM.
    private func writeOrPlay(samples: [Float], sampleRate: Int, t0: CFAbsoluteTime) throws {
        guard !samples.isEmpty else {
            print("Error: no audio generated")
            throw ExitCode(1)
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        let secs = Double(samples.count) / Double(sampleRate)
        print(String(format: "  %.2fs audio in %.2fs (RTF %.2f)",
                     secs, elapsed, elapsed / secs))
        if play {
            playAudio(samples: samples, sampleRate: sampleRate)
        } else {
            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: outputURL)
            print("Saved \(samples.count) samples (\(formatDuration(samples.count, sampleRate: sampleRate))s) to \(output)")
        }
    }

    // MARK: - Qwen3 engine

    private func runQwen3() throws {
        try runAsync {
            // Resolve model ID
            let resolvedModelId: String
            switch model.lowercased() {
            case "base", "base-8bit", "base8bit":
                resolvedModelId = TTSModelVariant.base.rawValue
            case "1.7b", "large", "1.7b-bf16", "large-bf16":
                resolvedModelId = TTSModelVariant.base17Bbf16.rawValue
            case "1.7b-8bit", "large-8bit":
                resolvedModelId = TTSModelVariant.base17B8bit.rawValue
            case "customvoice", "custom_voice", "custom-voice", "customvoice-8bit", "custom_voice_8bit", "custom-voice-8bit":
                resolvedModelId = TTSModelVariant.customVoice.rawValue
            case "customvoice-bf16", "custom_voice_bf16", "custom-voice-bf16":
                resolvedModelId = TTSModelVariant.customVoiceBf16.rawValue
            default:
                resolvedModelId = model
            }

            print("Loading Qwen3-TTS model (\(resolvedModelId))...")
            let ttsModel = try await Qwen3TTSModel.fromPretrained(
                modelId: resolvedModelId, progressHandler: reportProgress)

            // --list-speakers
            if listSpeakers {
                let speakers = ttsModel.availableSpeakers
                if speakers.isEmpty {
                    print("No speakers available for this model.")
                    print("Use --model customVoice to load a model with speaker support.")
                } else {
                    print("Available speakers:")
                    for name in speakers {
                        let dialect = ttsModel.speakerConfig?.speakerDialects[name]
                        let suffix = dialect != nil ? " (\(dialect!))" : ""
                        print("  - \(name)\(suffix)")
                    }
                }
                return
            }

            let config = SamplingConfig(
                temperature: temperature,
                topK: topK,
                maxTokens: maxTokens)

            // Resolve effective instruct
            let effectiveInstruct: String?
            let instructIsDefault: Bool
            if let explicit = instruct {
                effectiveInstruct = explicit
                instructIsDefault = false
            } else if ttsModel.speakerConfig != nil {
                effectiveInstruct = Qwen3TTSModel.defaultInstruct
                instructIsDefault = true
            } else {
                effectiveInstruct = nil
                instructIsDefault = false
            }

            if stream, let inputText = text {
                try await runQwen3Streaming(
                    model: ttsModel, text: inputText,
                    instruct: effectiveInstruct, instructIsDefault: instructIsDefault,
                    config: config)
            } else if let batchFile = batchFile {
                try runQwen3Batch(model: ttsModel, batchFile: batchFile, config: config)
            } else if let inputText = text {
                try runQwen3Standard(
                    model: ttsModel, text: inputText,
                    instruct: effectiveInstruct, instructIsDefault: instructIsDefault,
                    config: config)
            }
        }
    }

    private func runQwen3Streaming(
        model: Qwen3TTSModel, text: String,
        instruct: String?, instructIsDefault: Bool,
        config: SamplingConfig
    ) async throws {
        let streamingConfig = StreamingConfig(
            firstChunkFrames: firstChunkFrames,
            chunkFrames: chunkFrames)

        var info = "Streaming synthesis: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        print(info)
        print("  First chunk: \(firstChunkFrames) frames, subsequent: \(chunkFrames) frames")

        var allSamples: [Float] = []
        var chunkCount = 0
        var firstPacketLatency: Double?

        let audioStream = model.synthesizeStream(
            text: text,
            language: effectiveLanguage,
            speaker: speaker,
            instruct: instruct,
            sampling: config,
            streaming: streamingConfig,
            languageExplicit: languageIsExplicit)

        for try await chunk in audioStream {
            chunkCount += 1
            allSamples.append(contentsOf: chunk.samples)

            if firstPacketLatency == nil {
                firstPacketLatency = chunk.elapsedTime
            }

            let chunkDuration = Double(chunk.samples.count) / 24000.0
            let marker = chunk.isFinal ? " [FINAL]" : ""
            print("  Chunk \(chunkCount): \(chunk.samples.count) samples " +
                  "(\(String(format: "%.3f", chunkDuration))s) | " +
                  "frame \(chunk.frameIndex) | " +
                  "elapsed \(String(format: "%.3f", chunk.elapsedTime ?? 0))s\(marker)")
        }

        guard !allSamples.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        print("  First-packet latency: \(String(format: "%.0f", (firstPacketLatency ?? 0) * 1000))ms")
        print("  Total: \(chunkCount) chunks, \(allSamples.count) samples (\(formatDuration(allSamples.count))s)")

        if !play {
            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
            print("Saved to \(output)")
        } else {
            playAudio(samples: allSamples, sampleRate: 24000)
        }
    }

    private func runQwen3Batch(
        model: Qwen3TTSModel, batchFile: String, config: SamplingConfig
    ) throws {
        let content = try String(contentsOfFile: batchFile, encoding: .utf8)
        let texts = content.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        guard !texts.isEmpty else {
            print("Error: No texts found in \(batchFile)")
            throw ExitCode(1)
        }

        print("Batch synthesizing \(texts.count) texts...")
        let audioList = model.synthesizeBatch(
            texts: texts,
            language: effectiveLanguage,
            instruct: instruct,
            sampling: config,
            maxBatchSize: batchSize)

        let basePath = (output as NSString).deletingPathExtension
        let ext = (output as NSString).pathExtension.isEmpty ? "wav" : (output as NSString).pathExtension

        for (i, audio) in audioList.enumerated() {
            guard !audio.isEmpty else {
                print("Warning: Item \(i) produced no audio")
                continue
            }
            let path = "\(basePath)_\(i).\(ext)"
            let url = URL(fileURLWithPath: path)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: url)
            print("Saved item \(i): \(audio.count) samples (\(formatDuration(audio.count))s) to \(path)")
        }
    }

    private func runQwen3Standard(
        model: Qwen3TTSModel, text: String,
        instruct: String?, instructIsDefault: Bool,
        config: SamplingConfig
    ) throws {
        var info = "Synthesizing: \"\(text)\""
        if let spk = speaker { info += " [speaker: \(spk)]" }
        if let inst = instruct { info += " [instruct: \(inst)\(instructIsDefault ? " (default)" : "")]" }
        if let vs = voiceSample { info += " [voice clone: \(vs)]" }
        print(info)

        let audio: [Float]
        if let voiceSamplePath = voiceSample {
            // Voice cloning mode
            let refSamples = try loadReference(path: voiceSamplePath, targetSampleRate: 24000)
            print("  Reference audio: \(refSamples.count) samples, \(String(format: "%.1f", Double(refSamples.count) / 24000.0))s")

            audio = model.synthesizeWithVoiceClone(
                text: text,
                referenceAudio: refSamples,
                referenceSampleRate: 24000,
                language: effectiveLanguage,
                sampling: config)
        } else {
            audio = model.synthesize(
                text: text,
                language: effectiveLanguage,
                speaker: speaker,
                instruct: instruct,
                sampling: config,
                languageExplicit: languageIsExplicit)
        }

        guard !audio.isEmpty else {
            print("Error: No audio generated")
            throw ExitCode(1)
        }

        if !play {
            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: audio, sampleRate: 24000, to: outputURL)
            print("Saved \(audio.count) samples (\(formatDuration(audio.count))s) to \(output)")
        } else {
            playAudio(samples: audio, sampleRate: 24000)
        }
    }

    // MARK: - VoxCPM2 engine

    private func resolvedVoxCPM2ModelId() throws -> String {
        switch voxcpm2Variant.lowercased() {
        case "bf16":
            return "aufklarer/VoxCPM2-MLX-bf16"
        case "int8":
            return "aufklarer/VoxCPM2-MLX-int8"
        default:
            throw ValidationError("--voxcpm2-variant must be bf16 or int8 (int4 was decommissioned) (got '\(voxcpm2Variant)')")
        }
    }

    private func runVoxCPM2() throws {
        try runAsync {
            let runOnCPU = ProcessInfo.processInfo.environment["VOXCPM2_FORCE_CPU"] == "1"
            let body: () async throws -> Void = {
                guard let inputText = text else {
                    print("Error: text argument is required for VoxCPM2")
                    throw ExitCode(1)
                }

                let resolvedId = try resolvedVoxCPM2ModelId()
                print("Loading VoxCPM2 model (\(resolvedId))...")
                let model = try await VoxCPM2TTSModel.fromPretrained(
                    modelId: resolvedId,
                    progressHandler: reportProgress
                )

                if let s = seed {
                    MLX.seed(s)
                    print("  Seed: \(s) (deterministic flow + LM + vocoder sampling)")
                }

                let referenceAudio: [Float]?
                if let refPath = voxcpm2RefAudio {
                    referenceAudio = try loadReference(path: refPath, targetSampleRate: 16000)
                    print("  Reference audio: \(referenceAudio?.count ?? 0) samples")
                } else if let fallbackVoiceSample = voiceSample {
                    referenceAudio = try loadReference(path: fallbackVoiceSample, targetSampleRate: 16000)
                    print("  Reference audio: \(referenceAudio?.count ?? 0) samples")
                } else {
                    referenceAudio = nil
                }

                let promptAudio: [Float]?
                if let promptPath = voxcpm2PromptAudio {
                    let promptURL = URL(fileURLWithPath: promptPath)
                    promptAudio = try AudioFileLoader.load(url: promptURL, targetSampleRate: 16000)
                    print("  Prompt audio: \(promptAudio?.count ?? 0) samples")
                } else {
                    promptAudio = nil
                }

                print("Synthesizing with VoxCPM2 (language: \(effectiveLanguage))...")
                let audio = try await model.generateVoxCPM2(
                    text: inputText,
                    language: effectiveLanguage,
                    maxTokens: voxcpm2MaxTokens,
                    minTokens: voxcpm2MinTokens,
                    refAudio: referenceAudio,
                    promptText: voxcpm2PromptText,
                    promptAudio: promptAudio,
                    inferenceTimesteps: voxcpm2Timesteps,
                    cfgValue: voxcpm2CfgValue,
                    streamingPrefixLen: voxcpm2StreamingPrefixLen,
                    warmupPatches: voxcpm2WarmupPatches,
                    instruct: voxcpm2Instruct
                )

                guard !audio.isEmpty else {
                    print("Error: No audio generated")
                    throw ExitCode(1)
                }

                let sampleRate = model.sampleRate
                let outputURL = URL(fileURLWithPath: output)
                if !play {
                    try WAVWriter.write(samples: audio, sampleRate: sampleRate, to: outputURL)
                    print("Saved \(audio.count) samples (\(formatDuration(audio.count, sampleRate: sampleRate))s) to \(output)")
                } else {
                    playAudio(samples: audio, sampleRate: sampleRate)
                }

                model.unload()
            }

            if runOnCPU {
                try await Device.withDefaultDevice(.cpu) {
                    try await Stream.withNewDefaultStream(device: .cpu) {
                        try await body()
                    }
                }
            } else {
                try await body()
            }
        }
    }

    // MARK: - IndexTTS2 engine

    private func runIndexTTS2() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for IndexTTS2")
                throw ExitCode(1)
            }
            guard let voiceSample else {
                print("Error: --voice-sample is required for IndexTTS2")
                throw ExitCode(1)
            }

            let model: IndexTTS2TTSModel
            if let bundleDir = indextts2BundleDir {
                let bundleURL = URL(fileURLWithPath: bundleDir)
                print("Loading IndexTTS2 exported bundle (\(bundleURL.path))...")
                model = try await IndexTTS2TTSModel.fromBundle(
                    bundleURL,
                    progressHandler: reportProgress)
            } else {
                print("Loading IndexTTS2 exported bundle (\(indextts2ModelId))...")
                model = try await IndexTTS2TTSModel.fromPretrained(
                    modelId: indextts2ModelId,
                    progressHandler: reportProgress)
            }

            print("  Model: \(model.manifest.displayName)")
            print("  Source: \(model.manifest.sourceRepo)")
            if let publishRepo = model.manifest.publishRepo {
                print("  Bundle: \(publishRepo)")
            }
            print("  Params: \(model.manifest.parameterCount ?? "unknown")")
            print("  Sample rate: \(model.sampleRate) Hz")
            print("  Runtime status: native Swift synthesis enabled")

            let voiceURL = URL(fileURLWithPath: voiceSample)
            let emotionURL = indextts2EmotionAudio.map { URL(fileURLWithPath: $0) }
            let emotionControl = try parseIndexTTS2EmotionControl()
            let synthesisOptions = try parseIndexTTS2SynthesisOptions()
            if let indextts2Emotion {
                print("  Emotion: \(indextts2Emotion) @ \(indextts2EmotionWeight)")
            }
            if indextts2SpeakingRate != 1.0 {
                print("  Speaking rate: \(indextts2SpeakingRate)x")
            }
            if let maxPause = indextts2MaxPause {
                print("  Max internal pause: \(maxPause)s")
            }
            let audio = try await model.generate(
                text: inputText,
                referenceAudio: voiceURL,
                emotionReferenceAudio: emotionURL,
                emotionControl: emotionControl,
                synthesisOptions: synthesisOptions,
                language: effectiveLanguage)
            let outputURL = URL(fileURLWithPath: output)
            if play {
                playAudio(samples: audio, sampleRate: model.sampleRate)
            } else {
                try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)
                print("Saved audio to \(output)")
            }
        }
    }

    private func parseIndexTTS2SynthesisOptions() throws -> IndexTTS2SynthesisOptions {
        do {
            return try IndexTTS2SynthesisOptions(
                speakingRate: indextts2SpeakingRate,
                maxInternalPauseDuration: indextts2MaxPause,
                s2MelSteps: indextts2S2MelSteps)
        } catch let error as AudioModelError {
            throw ValidationError(error.localizedDescription)
        }
    }

    private func parseIndexTTS2EmotionControl() throws -> IndexTTS2EmotionControl? {
        guard let rawEmotion = indextts2Emotion?.trimmingCharacters(in: .whitespacesAndNewlines),
              !rawEmotion.isEmpty else {
            guard indextts2EmotionWeight.isFinite,
                  indextts2EmotionWeight >= 0,
                  indextts2EmotionWeight <= 1 else {
                throw ValidationError("--indextts2-emotion-weight must be in [0, 1]")
            }
            return nil
        }

        do {
            if let preset = IndexTTS2EmotionPreset(named: rawEmotion) {
                return try IndexTTS2EmotionControl(preset: preset, weight: indextts2EmotionWeight)
            }
            let vector = try parseIndexTTS2EmotionVector(rawEmotion)
            return try IndexTTS2EmotionControl(vector: vector, weight: indextts2EmotionWeight)
        } catch let error as IndexTTS2EmotionControlError {
            throw ValidationError(error.localizedDescription)
        }
    }

    private func parseIndexTTS2EmotionVector(_ raw: String) throws -> [Float] {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("[") && text.hasSuffix("]") {
            text.removeFirst()
            text.removeLast()
        }
        let parts = text.split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard parts.count == 8 else {
            if IndexTTS2EmotionPreset(named: raw) == nil, !raw.contains(",") {
                throw ValidationError(IndexTTS2EmotionControlError.unknownPreset(raw).localizedDescription)
            }
            throw ValidationError("--indextts2-emotion vector must contain exactly 8 comma-separated values")
        }
        return try parts.map { part in
            guard let value = Float(part), value.isFinite else {
                throw ValidationError("--indextts2-emotion vector entries must be numeric")
            }
            return value
        }
    }

    // MARK: - F5-TTS engine

    private func runF5TTS() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for F5-TTS")
                throw ExitCode(1)
            }
            guard let voiceSample else {
                print("Error: --voice-sample is required for F5-TTS")
                throw ExitCode(1)
            }
            guard let referenceText = f5ReferenceText?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !referenceText.isEmpty else {
                print("Error: --f5-reference-text is required for F5-TTS")
                throw ExitCode(1)
            }

            let model: F5TTSModel
            if let bundleDir = f5BundleDir {
                let bundleURL = URL(fileURLWithPath: bundleDir)
                print("Loading F5-TTS exported bundle (\(bundleURL.path))...")
                model = try await F5TTSModel.fromBundle(
                    bundleURL,
                    progressHandler: reportProgress)
            } else {
                print("Loading F5-TTS exported bundle (\(f5ModelId))...")
                model = try await F5TTSModel.fromPretrained(
                    modelId: f5ModelId,
                    progressHandler: reportProgress)
            }

            print("  Model: \(model.config.modelName)")
            print("  Source: \(model.config.sourceRepo)")
            print("  Vocoder: \(model.config.vocoderRepo)")
            print("  Precision: \(model.config.precision)")
            print(String(format: "  Bundle weights: %.1f MB", Double(model.memoryFootprint) / 1_000_000.0))
            print("  Sample rate: \(model.sampleRate) Hz")
            print("  Runtime status: native Swift synthesis enabled")

            let options = try parseF5SynthesisOptions()
            if f5Speed != 1.0 {
                print("  Speaking rate: \(f5Speed)x")
            }
            if f5Steps != 16 || f5CfgStrength != 2.0 || f5Sway != -1.0 {
                print("  Sampling: steps \(f5Steps), cfg \(f5CfgStrength), sway \(f5Sway)")
            }

            let start = CFAbsoluteTimeGetCurrent()
            let audio: [Float]
            if cleanReference {
                let referenceAudio = try loadReference(path: voiceSample, targetSampleRate: model.sampleRate)
                audio = try model.generate(
                    text: inputText,
                    referenceAudio: referenceAudio,
                    referenceSampleRate: model.sampleRate,
                    referenceText: referenceText,
                    options: options,
                    progressHandler: reportProgress)
            } else {
                audio = try await model.generate(
                    text: inputText,
                    referenceAudio: URL(fileURLWithPath: voiceSample),
                    referenceText: referenceText,
                    options: options,
                    progressHandler: reportProgress)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            guard !audio.isEmpty else {
                print("Error: No audio generated")
                throw ExitCode(1)
            }

            let duration = Double(audio.count) / Double(model.sampleRate)
            print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                         duration, elapsed, elapsed / max(duration, 0.001)))

            let outputURL = URL(fileURLWithPath: output)
            if play {
                playAudio(samples: audio, sampleRate: model.sampleRate)
            } else {
                try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)
                print("Saved \(audio.count) samples (\(formatDuration(audio.count, sampleRate: model.sampleRate))s) to \(output)")
            }
        }
    }

    // MARK: - Higgs engine

    private func runHiggs() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for Higgs")
                throw ExitCode(1)
            }

            let model: HiggsTTSModel
            if let bundleDir = higgsBundleDir {
                let bundleURL = URL(fileURLWithPath: bundleDir)
                print("Loading Higgs TTS 3 bundle (\(bundleURL.path))...")
                model = try await HiggsTTSModel.fromBundle(
                    bundleURL,
                    progressHandler: reportProgress)
            } else {
                print("Loading Higgs TTS 3 bundle (\(higgsModelId))...")
                model = try await HiggsTTSModel.fromPretrained(
                    modelId: higgsModelId,
                    progressHandler: reportProgress)
            }

            print("  Backbone: Qwen3 \(model.config.textConfig.numHiddenLayers) layers, hidden \(model.config.textConfig.hiddenSize)")
            print("  Codebooks: \(model.config.audioNumCodebooks) x \(model.config.audioCodebookSize) at 25 fps")
            print("  Sample rate: \(model.sampleRate) Hz")

            let options = try parseHiggsSynthesisOptions()
            if higgsTemperature != 0.8 || higgsTopP != nil || higgsTopK != nil {
                print("  Sampling: temperature \(higgsTemperature)"
                      + (higgsTopP.map { ", top-p \($0)" } ?? "")
                      + (higgsTopK.map { ", top-k \($0)" } ?? ""))
            }

            var references: [HiggsTTSReference] = []
            if let voiceSample {
                print("  Encoding reference: \(voiceSample)")
                if cleanReference {
                    let cleaned = try loadReference(path: voiceSample, targetSampleRate: model.sampleRate)
                    references = [try model.encodeReference(
                        samples: cleaned, sampleRate: model.sampleRate, text: higgsRefText)]
                } else {
                    references = [try model.encodeReference(
                        audio: URL(fileURLWithPath: voiceSample), text: higgsRefText)]
                }
            }

            let start = CFAbsoluteTimeGetCurrent()
            let audio = try model.generate(
                text: inputText,
                references: references,
                options: options,
                progressHandler: reportProgress)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            guard !audio.isEmpty else {
                print("Error: No audio generated")
                throw ExitCode(1)
            }

            let duration = Double(audio.count) / Double(model.sampleRate)
            print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                         duration, elapsed, elapsed / max(duration, 0.001)))

            let outputURL = URL(fileURLWithPath: output)
            if play {
                playAudio(samples: audio, sampleRate: model.sampleRate)
            } else {
                try WAVWriter.write(samples: audio, sampleRate: model.sampleRate, to: outputURL)
                print("Saved \(audio.count) samples (\(formatDuration(audio.count, sampleRate: model.sampleRate))s) to \(output)")
            }
        }
    }

    private func parseHiggsSynthesisOptions() throws -> HiggsTTSSynthesisOptions {
        do {
            return try HiggsTTSSynthesisOptions(
                temperature: higgsTemperature,
                topP: higgsTopP,
                topK: higgsTopK,
                maxNewTokens: higgsMaxNewTokens,
                seed: higgsSeed)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    private func parseF5SynthesisOptions() throws -> F5TTSSynthesisOptions {
        do {
            return try F5TTSSynthesisOptions(
                steps: f5Steps,
                cfgStrength: f5CfgStrength,
                swaySamplingCoef: f5Sway,
                speed: f5Speed,
                seed: f5Seed,
                targetRMS: f5TargetRMS)
        } catch {
            throw ValidationError(error.localizedDescription)
        }
    }

    // MARK: - Indic-Mio engine

    private func runIndicMio() throws {
        try runAsync {
            guard let inputText = text else {
                print("Error: text argument is required for Indic-Mio")
                throw ExitCode(1)
            }

            print("Loading Indic-Mio model (\(indicMioModelId))...")
            let model = try await IndicMioTTSModel.fromPretrained(
                modelId: indicMioModelId,
                progressHandler: reportProgress
            )

            let embedding = try indicMioGlobalEmbedding.map { try loadIndicMioGlobalEmbedding(from: $0) }
            let referenceAudio: [Float]?
            if let voiceSample {
                referenceAudio = try loadReference(path: voiceSample, targetSampleRate: model.sampleRate)
                print("  Reference audio: \(referenceAudio?.count ?? 0) samples @ \(model.sampleRate) Hz")
            } else {
                referenceAudio = nil
            }
            let sampling = IndicMioSamplingConfig(
                maxNewTokens: maxTokens,
                temperature: temperature,
                topK: topK,
                topP: indicMioTopP,
                repetitionPenalty: indicMioRepetitionPenalty
            )

            let markers = IndicMioPrompt.indianLanguageEmotionMarkers.joined(separator: ", ")
            var info = "Synthesizing with Indic-Mio (language: \(effectiveIndicMioLanguage))"
            if embedding != nil { info += " [global speaker embedding]" }
            if referenceAudio != nil { info += " [voice clone: WavLM global embedding]" }
            print(info)
            print("  Emotion markers: \(markers)")

            let start = CFAbsoluteTimeGetCurrent()
            let samples: [Float]
            if let embedding {
                samples = try await model.generate(
                    text: inputText,
                    language: effectiveIndicMioLanguage,
                    globalEmbedding: embedding,
                    sampling: sampling
                )
            } else if let referenceAudio {
                samples = try await model.generate(
                    text: inputText,
                    language: effectiveIndicMioLanguage,
                    referenceAudio: referenceAudio,
                    referenceSampleRate: model.sampleRate,
                    sampling: sampling
                )
            } else {
                samples = try await model.generate(
                    text: inputText,
                    language: effectiveIndicMioLanguage,
                    sampling: sampling
                )
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            guard !samples.isEmpty else {
                print("Error: No audio generated")
                throw ExitCode(1)
            }

            let duration = Double(samples.count) / Double(model.sampleRate)
            print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                         duration, elapsed, elapsed / max(duration, 0.001)))

            if !play {
                let outputURL = URL(fileURLWithPath: output)
                try WAVWriter.write(samples: samples, sampleRate: model.sampleRate, to: outputURL)
                print("Saved \(samples.count) samples (\(formatDuration(samples.count, sampleRate: model.sampleRate))s) to \(output)")
            } else {
                playAudio(samples: samples, sampleRate: model.sampleRate)
            }
        }
    }

    private func loadIndicMioGlobalEmbedding(from value: String) throws -> [Float] {
        func parseJSONEmbedding(_ data: Data) throws -> [Float]? {
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [NSNumber] else {
                return nil
            }
            let floats = json.map { $0.floatValue }
            guard floats.count == MioCodecConfig.default.globalEmbeddingDim else {
                throw ValidationError("--indic-mio-global-embedding JSON must contain 128 floats")
            }
            return floats
        }

        let url = URL(fileURLWithPath: value)
        let text: String
        if FileManager.default.fileExists(atPath: url.path) {
            let data = try Data(contentsOf: url)
            if let floats = try parseJSONEmbedding(data) {
                return floats
            }
            text = String(decoding: data, as: UTF8.self)
        } else {
            text = value
        }

        if text.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("["),
           let floats = try parseJSONEmbedding(Data(text.utf8)) {
            return floats
        }

        let separators = CharacterSet(charactersIn: ", \n\r\t")
        let floats = text
            .split(whereSeparator: { scalar in
                scalar.unicodeScalars.allSatisfy { separators.contains($0) }
            })
            .compactMap { Float($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
        guard floats.count == MioCodecConfig.default.globalEmbeddingDim else {
            throw ValidationError("--indic-mio-global-embedding must contain 128 floats (got \(floats.count))")
        }
        return floats
    }

    // MARK: - CosyVoice engine

    private func resolvedCosyVoiceModelId() throws -> String {
        if let explicit = modelId, !explicit.isEmpty { return explicit }
        switch cosyvoiceVariant.lowercased() {
        case "8bit":      return "aufklarer/CosyVoice3-0.5B-MLX-8bit"
        case "8bit-full": return "aufklarer/CosyVoice3-0.5B-MLX-8bit-full"
        case "bf16", "bfloat16", "16", "16bit", "16-bit", "unquantized":
            return "aufklarer/CosyVoice3-0.5B-MLX-bf16"
        default:
            throw ValidationError(
                "--cosyvoice-variant must be 8bit, 8bit-full, bf16, or 16bit (got '\(cosyvoiceVariant)')")
        }
    }

    private func runCosyVoice() throws {
        try runAsync {
            print("Loading CosyVoice3 model...")
            let resolvedId = try self.resolvedCosyVoiceModelId()
            let bundleOverride = self.cosyBundleDir.map { URL(fileURLWithPath: $0) }
            let cosyModel = try await CosyVoiceTTSModel.fromPretrained(
                modelId: resolvedId,
                cacheDir: bundleOverride,
                progressHandler: reportProgress)

            guard let inputText = text else {
                print("Error: text argument is required for CosyVoice")
                throw ExitCode(1)
            }

            // Parse speaker mapping: "s1=alice.wav,s2=bob.wav"
            var speakerFiles: [String: String] = [:]
            if let speakersArg = speakers {
                for pair in speakersArg.split(separator: ",") {
                    let parts = pair.split(separator: "=", maxSplits: 1)
                    guard parts.count == 2 else {
                        print("Error: Invalid speaker mapping '\(pair)'. Expected format: name=file.wav")
                        throw ExitCode(1)
                    }
                    speakerFiles[String(parts[0]).uppercased()] = String(parts[1])
                }
            }

            // Load speaker embeddings from voice samples
            var speakerEmbeddings: [String: [Float]] = [:]
            #if canImport(CoreML)
            // Single --voice-sample (no --speakers) → used as default embedding.
            // When `speech_tokenizer.safetensors` is present in the bundle we
            // also extract the upstream zero-shot conditioning (prompt_token +
            // prompt_feat) and stash it in `defaultVoiceProfile`. The single-
            // segment synthesis path below picks the profile up automatically.
            var defaultEmbedding: [Float]?
            var defaultVoiceProfile: CosyVoiceVoiceProfile?
            if let voiceSamplePath = voiceSample, speakerFiles.isEmpty {
                let refSamples16k = try loadReference(path: voiceSamplePath, targetSampleRate: 16000)
                print("  Reference audio: \(refSamples16k.count) samples (\(String(format: "%.1f", Double(refSamples16k.count) / 16000.0))s)")

                print("Loading CAM++ speaker encoder...")
                let campp = try await CamPlusPlusSpeaker.fromPretrained { progress, status in
                    reportProgress(progress, status)
                }

                let embedding = try campp.embed(audio: refSamples16k, sampleRate: 16000)
                defaultEmbedding = embedding
                print("  Speaker embedding: \(embedding.count)-dim")

                // Look for the S3 speech tokenizer alongside the other bundle
                // files. If it's there, build a full voice profile (prompt_token
                // + prompt_feat + speaker embedding) so the flow gets per-frame
                // reference conditioning. Bundles produced before this change
                // won't have the file — we fall back to the spk-only path with
                // a warning so the operator knows why cloning quality is capped.
                let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: resolvedId)
                let tokURL = cosySpeechTokenizer.map { URL(fileURLWithPath: $0) }
                    ?? cacheDir.appendingPathComponent("speech_tokenizer.safetensors")
                if FileManager.default.fileExists(atPath: tokURL.path) {
                    print("Loading speech tokenizer (\(tokURL.lastPathComponent))...")
                    let tokenizer = try SpeechTokenizerModel.fromSafetensors(at: tokURL)
                    print("  Extracting voice profile (prompt_token + prompt_feat)...")
                    defaultVoiceProfile = try cosyModel.extractVoiceProfile(
                        audio: refSamples16k,
                        sampleRate: 16000,
                        speechTokenizer: tokenizer,
                        camppSpeaker: campp,
                        referenceTranscript: cosyReferenceTranscript
                    )
                    if let p = defaultVoiceProfile {
                        let tokLen = p.promptToken?.dim(1) ?? 0
                        let mel50Hz = p.promptFeat?.dim(2) ?? 0
                        print("  Voice profile: \(tokLen) prompt tokens (25 Hz), \(mel50Hz) mel frames (50 Hz)")
                    }
                } else {
                    print("  No speech_tokenizer.safetensors in bundle — falling back to spk-only cloning (cap ≈ cos 0.83).")
                    print("    Re-export the bundle with `convert_speech_tokenizer` (speech-models) to enable.")
                }
            }

            // Multi-speaker: load CAM++ once, extract embedding per speaker file
            if !speakerFiles.isEmpty {
                print("Loading CAM++ speaker encoder...")
                let campp = try await CamPlusPlusSpeaker.fromPretrained { progress, status in
                    reportProgress(progress, status)
                }

                for (name, path) in speakerFiles {
                    let refURL = URL(fileURLWithPath: path)
                    let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
                    let embedding = try campp.embed(audio: refSamples, sampleRate: 16000)
                    speakerEmbeddings[name] = embedding
                    print("  Speaker \(name): \(embedding.count)-dim embedding from \(path)")
                }
            }
            #else
            let defaultEmbedding: [Float]? = nil
            let defaultVoiceProfile: CosyVoiceVoiceProfile? = nil
            #endif

            // Parse dialogue segments
            let segments = DialogueParser.parse(inputText)
            let isDialogue = segments.count > 1
                || segments.first?.speaker != nil
                || segments.first?.emotion != nil

            let defaultInstruction = cosyInstruct ?? "You are a helpful assistant."

            print("  Language: \(effectiveLanguage)")

            // Seed every stochastic source in the pipeline (LLM Gumbel sampling,
            // flow-matching initial noise, HiFiGAN init-phase + noise injections)
            // BEFORE the first synthesis call. With a fixed seed, repeated CLI
            // invocations on different scripts but the same speaker embedding
            // produce near-identical prosody and timbre — necessary for long-form
            // narration cut into per-section chunks where per-call diffusion
            // variance otherwise drifts the voice between sections.
            if let s = seed {
                MLX.seed(s)
                print("  Seed: \(s) (deterministic flow + LLM + vocoder sampling)")
            }

            let startTime = CFAbsoluteTimeGetCurrent()

            if isDialogue {
                // Multi-segment dialogue synthesis
                if verbose {
                    print("  Dialogue: \(segments.count) segments")
                    for (i, seg) in segments.enumerated() {
                        var desc = "    [\(i + 1)] \"\(seg.text)\""
                        if let spk = seg.speaker { desc += " speaker=\(spk)" }
                        if let emo = seg.emotion { desc += " emotion=\(emo)" }
                        print(desc)
                    }
                }

                // Merge default embedding into per-speaker map for segments without speaker tags
                var allEmbeddings = speakerEmbeddings
                if let defEmb = defaultEmbedding {
                    // Assign default embedding to any speaker tag not in the map
                    for seg in segments {
                        if let spk = seg.speaker?.uppercased(), allEmbeddings[spk] == nil {
                            allEmbeddings[spk] = defEmb
                        }
                    }
                }

                let config = DialogueSynthesisConfig(
                    turnGapSeconds: turnGap,
                    crossfadeSeconds: self.crossfade,
                    defaultInstruction: defaultInstruction
                )

                let samples = DialogueSynthesizer.synthesize(
                    segments: segments,
                    speakerEmbeddings: allEmbeddings,
                    model: cosyModel,
                    language: effectiveLanguage,
                    config: config,
                    verbose: verbose
                )

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(samples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: samples, sampleRate: 24000)
                }
            } else if stream {
                // Streaming (single segment, no dialogue)
                var allSamples: [Float] = []
                var chunkCount = 0
                for try await chunk in cosyModel.synthesizeStream(text: inputText, language: effectiveLanguage) {
                    allSamples.append(contentsOf: chunk.samples)
                    chunkCount += 1
                    let chunkDuration = Double(chunk.samples.count) / Double(chunk.sampleRate)
                    print("  Chunk \(chunkCount): \(String(format: "%.2f", chunkDuration))s (\(chunk.samples.count) samples)")
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(allSamples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: allSamples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: allSamples, sampleRate: 24000)
                }
            } else {
                // Single segment synthesis
                let instruction = segments.first?.emotion.map {
                    DialogueParser.emotionToInstruction($0)
                } ?? defaultInstruction

                var info = "Synthesizing: \"\(inputText)\""
                if defaultVoiceProfile != nil {
                    info += " [voice clone: prompt_token + prompt_feat]"
                } else if defaultEmbedding != nil || !speakerEmbeddings.isEmpty {
                    info += " [voice clone: spk-only]"
                }
                if instruction != "You are a helpful assistant." { info += " [instruction: \(instruction)]" }
                print(info)

                let samples: [Float]
                if let profile = defaultVoiceProfile {
                    samples = cosyModel.synthesize(
                        text: inputText,
                        voiceProfile: profile,
                        language: effectiveLanguage,
                        instruction: instruction,
                        seed: self.seed,
                        verbose: verbose
                    )
                } else {
                    samples = cosyModel.synthesize(
                        text: inputText, language: effectiveLanguage,
                        instruction: instruction,
                        speakerEmbedding: defaultEmbedding,
                        verbose: verbose
                    )
                }

                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let duration = Double(samples.count) / 24000.0
                print(String(format: "  Duration: %.2fs, Time: %.2fs, RTF: %.2f",
                             duration, elapsed, elapsed / max(duration, 0.001)))

                if !self.play {
                    let outputURL = URL(fileURLWithPath: self.output)
                    try WAVWriter.write(samples: samples, sampleRate: 24000, to: outputURL)
                    print("Saved to \(self.output)")
                } else {
                    self.playAudio(samples: samples, sampleRate: 24000)
                }
            }
        }
    }

    // MARK: - Audio Playback

    private func playAudio(samples: [Float], sampleRate: Int) {
        let engine = AVAudioEngine()
        let playerNode = AVAudioPlayerNode()
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: format)

        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            buffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }

        do {
            try engine.start()
        } catch {
            print("Error: Failed to start audio engine: \(error)")
            return
        }

        let semaphore = DispatchSemaphore(value: 0)
        playerNode.play()
        playerNode.scheduleBuffer(buffer) {
            semaphore.signal()
        }

        print("Playing \(formatDuration(samples.count))s audio...")
        semaphore.wait()
        // Small delay for audio to finish draining
        usleep(100_000)
        engine.stop()
    }
}
