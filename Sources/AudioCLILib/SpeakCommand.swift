import Foundation
import AVFoundation
import ArgumentParser
import MLX
import Qwen3TTS
import CosyVoiceTTS
import VoxCPM2TTS
import MagpieTTS
import MagpieTTSCoreML
import AudioCommon

public struct SpeakCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speak",
        abstract: "Text-to-speech synthesis (Qwen3-TTS, CosyVoice, VoxCPM2, or Magpie). For CoreML, use the `qwen3-tts-coreml` subcommand."
    )

    @Argument(help: "Text to synthesize (omit when using --list-speakers or --batch-file)")
    public var text: String?

    @Option(name: .long, help: "TTS engine: qwen3 (default), cosyvoice, voxcpm2, magpie, or magpie-coreml")
    public var engine: String = "qwen3"

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Language (english, chinese, german, japanese, spanish, french, korean, russian, italian, portuguese). Default: english. Omit to use speaker's native dialect when --speaker is set.")
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

    @Option(name: .long, help: "Reference audio file for voice cloning (qwen3 Base, cosyvoice, or voxcpm2)")
    public var voiceSample: String?

    @Option(name: .long, help: "[qwen3] Model variant: base (default), base-8bit, 1.7b, 1.7b-8bit, customVoice, or full HF model ID. Note: --speaker requires customVoice.")
    public var model: String = "base"

    @Flag(name: .long, help: "[qwen3] List available speakers and exit")
    public var listSpeakers: Bool = false

    @Option(name: .long, help: "[qwen3] Sampling temperature (default: 0.3)")
    public var temperature: Float = 0.3

    @Option(name: .long, help: "[qwen3] Top-k sampling")
    public var topK: Int = 50

    @Option(name: .long, help: "[qwen3] Maximum tokens (500 = ~40s audio)")
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

    @Option(name: .long, help: "[cosyvoice] HuggingFace model ID")
    public var modelId: String = "aufklarer/CosyVoice3-0.5B-MLX-4bit"

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

    @Option(name: .long, help: "[voxcpm2] Quantization variant: bf16 (default), int8, int4. Resolves to aufklarer/VoxCPM2-MLX-<variant>.")
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

    // MARK: - Magpie-specific options

    @Option(name: .long, help: "[magpie] Quantization variant: int4 (default) or int8. Resolves to aufklarer/Magpie-TTS-Multilingual-357M-MLX-<variant>.")
    public var magpieVariant: String = "int4"

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

    /// Whether the user explicitly passed --language
    private var languageIsExplicit: Bool { language != nil }

    public func validate() throws {
        let eng = engine.lowercased()
        guard eng == "qwen3" || eng == "cosyvoice" || eng == "voxcpm2"
                || eng == "magpie" || eng == "magpie-coreml" else {
            throw ValidationError("--engine must be 'qwen3', 'cosyvoice', 'voxcpm2', 'magpie', or 'magpie-coreml'. For Qwen3-TTS CoreML, use the `qwen3-tts-coreml` subcommand.")
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
        if eng == "magpie" || eng == "magpie-coreml" {
            if batchFile != nil {
                throw ValidationError("--engine \(eng) does not support --batch-file (single utterance only)")
            }
            if MagpieSpeaker(named: magpieSpeaker) == nil {
                throw ValidationError("--magpie-speaker must be one of sofia, aria, jason, leo, john (got '\(magpieSpeaker)')")
            }
            guard magpieVariant.lowercased() == "int4" || magpieVariant.lowercased() == "int8" else {
                throw ValidationError("--magpie-variant must be int4 or int8 (got '\(magpieVariant)')")
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
        case "magpie":
            try runMagpie()
        case "magpie-coreml":
            try runMagpieCoreML()
        default:
            try runQwen3()
        }
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
            let variant: MagpieTTSVariant =
                (magpieVariant.lowercased() == "int8") ? .int8 : .int4
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
                let variant: MagpieTTSVariant =
                    (magpieVariant.lowercased() == "int8") ? .int8 : .int4
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
            case "base":
                resolvedModelId = TTSModelVariant.base.rawValue
            case "base-8bit", "base8bit":
                resolvedModelId = TTSModelVariant.base8bit.rawValue
            case "1.7b", "large":
                resolvedModelId = TTSModelVariant.base17B.rawValue
            case "1.7b-8bit", "large-8bit":
                resolvedModelId = TTSModelVariant.base17B8bit.rawValue
            case "customvoice", "custom_voice", "custom-voice":
                resolvedModelId = TTSModelVariant.customVoice.rawValue
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
            let refURL = URL(fileURLWithPath: voiceSamplePath)
            let refSamples = try AudioFileLoader.load(url: refURL, targetSampleRate: 24000)
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
        case "int4":
            return "aufklarer/VoxCPM2-MLX-int4"
        default:
            throw ValidationError("--voxcpm2-variant must be bf16, int8, or int4 (got '\(voxcpm2Variant)')")
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
                    let refURL = URL(fileURLWithPath: refPath)
                    referenceAudio = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
                    print("  Reference audio: \(referenceAudio?.count ?? 0) samples")
                } else if let fallbackVoiceSample = voiceSample {
                    let refURL = URL(fileURLWithPath: fallbackVoiceSample)
                    referenceAudio = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
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

    // MARK: - CosyVoice engine

    private func runCosyVoice() throws {
        try runAsync {
            print("Loading CosyVoice3 model...")
            let bundleOverride = cosyBundleDir.map { URL(fileURLWithPath: $0) }
            let cosyModel = try await CosyVoiceTTSModel.fromPretrained(
                modelId: modelId,
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
                let refURL = URL(fileURLWithPath: voiceSamplePath)
                let refSamples16k = try AudioFileLoader.load(url: refURL, targetSampleRate: 16000)
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
                let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
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
