import Foundation
import MLXCommon
import MLX
import ArgumentParser
import CohereTranscribeASR
import KyutaiSTT
import Qwen3ASR
import Qwen3ASRRuntime
import ParakeetASR
import NemotronStreamingASR
import OmnilingualASR
import WhisperASR
import VoxtralASR
import MossTranscribe
import SpeechVAD
import AudioCommon

public struct TranscribeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe speech to text (Qwen3-ASR, Cohere, Voxtral, MOSS, and other ASR backends)"
    )

    @Argument(help: "Audio file to transcribe (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .long, help: "ASR engine: qwen3 (default), cohere, kyutai, voxtral, moss, parakeet, nemotron, omnilingual, whisper, qwen3-coreml, or qwen3-coreml-full")
    public var engine: String = "qwen3"

    @Option(name: .long, help: "[omnilingual] Window size in seconds: 5 or 10 (default 10) — CoreML backend only")
    public var window: Int = 10

    @Option(name: .long, help: "[omnilingual/moss] Backend: coreml (default) or mlx")
    public var backend: String = "coreml"

    @Option(name: .long, help: "[omnilingual mlx] Variant: 300M (default), 1B, 3B, or 7B")
    public var variant: String = "300M"

    @Option(name: .long, help: "[omnilingual mlx] Quantization bits: 4 (default) or 8")
    public var bits: Int = 4

    @Option(name: .long, help: "[kyutai] MLX's process-wide free-buffer cache limit in MiB for this command (default: 256; 0 disables caching)")
    public var mlxCacheLimitMB: Int?

    var effectiveKyutaiCacheLimitMB: Int { mlxCacheLimitMB ?? 256 }

    @Option(name: .shortAndLong, help: "[qwen3] 0.6B (default), 0.6B-8bit, 1.7B, or 1.7B-4bit; [parakeet] v3 (default) or ultra; [cohere/voxtral] int5 (default), int8, or fp16; [kyutai] int8 (default), int4, fp16, upstream, or 2.6b; [moss coreml] int8 (default) or fp16; [moss mlx] int5 (default) or int8; [whisper] default or turbo; alternatively a model ID or local directory")
    public var model: String = "0.6B"

    @Option(name: .long, help: "[moss] Maximum generated transcript tokens (default: Core ML 512, MLX 5120)")
    public var maxTokens: Int?

    @Option(name: .long, help: "[moss mlx] KV-cache precision: fp16 (default) or int8")
    public var kvCache: String = "fp16"

    @Option(name: .long, help: "Language hint (optional)")
    public var language: String?

    @Option(name: .long, help: "Context string to bias recognition (e.g., 'Project: Foo, participants: Alice, Bob'). Improves proper-noun accuracy.")
    public var context: String?

    @Flag(name: .long, help: "Enable streaming transcription (native stream for Kyutai; VAD for Qwen3)")
    public var stream: Bool = false

    @Option(name: .long, help: "Maximum segment duration in seconds (default 10)")
    public var maxSegment: Float = 10.0

    @Flag(name: .long, help: "Emit partial results during speech")
    public var partial: Bool = false

    public init() {}

    public func validate() throws {
        let eng = engine.lowercased()
        let supportedEngines = [
            "qwen3", "cohere", "kyutai", "voxtral", "moss", "parakeet", "nemotron",
            "omnilingual", "whisper", "qwen3-coreml", "qwen3-coreml-full",
        ]
        guard supportedEngines.contains(eng) else {
            throw ValidationError(
                "--engine must be one of: \(supportedEngines.joined(separator: ", "))")
        }
        if let limit = mlxCacheLimitMB {
            guard eng == "kyutai", (0...(Int.max / (1024 * 1024))).contains(limit) else {
                throw ValidationError("--mlx-cache-limit-mb requires --engine kyutai and a nonnegative value that fits in bytes")
            }
        }
        if ["cohere", "voxtral"].contains(eng), stream {
            throw ValidationError("--stream is not supported by the \(eng) engine")
        }
        if eng == "parakeet" {
            _ = try resolveParakeetModel(model)
        }
        if eng == "kyutai" {
            _ = try resolveKyutaiModelId(model)
            guard language == nil && context == nil else {
                throw ValidationError("Kyutai detects language automatically and does not accept --language or --context")
            }
            guard !partial || stream else {
                throw ValidationError("Use --stream with --partial for Kyutai")
            }
        }
        if eng == "moss" {
            guard !stream else {
                throw ValidationError("--stream is not supported by the MOSS batch runtime")
            }
            if let maxTokens, maxTokens <= 0 {
                throw ValidationError("--max-tokens must be positive")
            }
            let backendNorm = backend.lowercased()
            guard backendNorm == "coreml" || backendNorm == "mlx" else {
                throw ValidationError("--backend must be 'coreml' or 'mlx' for moss")
            }
            _ = try resolveMossModelId(model, backend: backendNorm)
            if backendNorm == "mlx" {
                guard MossMLXKVCachePrecision(rawValue: kvCache.lowercased()) != nil else {
                    throw ValidationError(
                        "--kv-cache must be 'fp16' or 'int8' for moss mlx"
                    )
                }
            }
        }
        if eng == "omnilingual" {
            if window != 5 && window != 10 {
                throw ValidationError("--window must be 5 or 10 for omnilingual")
            }
            let backendNorm = backend.lowercased()
            guard backendNorm == "coreml" || backendNorm == "mlx" else {
                throw ValidationError("--backend must be 'coreml' or 'mlx' for omnilingual")
            }
            if backendNorm == "mlx" {
                guard ["300M", "1B", "3B", "7B"].contains(variant.uppercased()) else {
                    throw ValidationError("--variant must be 300M, 1B, 3B, or 7B")
                }
                guard bits == 4 || bits == 8 else {
                    throw ValidationError("--bits must be 4 or 8")
                }
            }
        }
    }

    public func run() throws {
        switch engine.lowercased() {
        case "cohere":
            try runCohereTranscription()
        case "kyutai":
            try runKyutaiTranscription()
        case "voxtral":
            try runVoxtralTranscription()
        case "parakeet":
            try runParakeetTranscription()
        case "nemotron":
            try runNemotronTranscription()
        case "omnilingual":
            try runOmnilingualTranscription()
        case "whisper":
            try runWhisperTranscription()
        case "moss":
            try runMossTranscription()
        case "qwen3-coreml":
            try runCoreMLTranscription()
        case "qwen3-coreml-full":
            try runFullCoreMLTranscription()
        default:
            if stream {
                try runStreamingTranscription()
            } else {
                try runBatchTranscription()
            }
        }
    }

    private func runKyutaiTranscription() throws {
        try runAsync {
            let cacheLimitLease = MLXCacheLimitCoordinator.shared.acquire(
                ceiling: effectiveKyutaiCacheLimitMB * 1024 * 1024)
            defer { cacheLimitLease?.release() }
            let modelId = try resolveKyutaiModelId(model)
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: KyutaiSTTConfig.sampleRate)
            let duration = Double(audio.count) / Double(KyutaiSTTConfig.sampleRate)
            print("Loading Kyutai STT model: \(modelId)")
            let asr = try await KyutaiSTTModel.load(
                modelId, progressHandler: { reportProgress($0, "Downloading") })
            let start = CFAbsoluteTimeGetCurrent()
            let text: String
            if stream {
                let session = try asr.makeSession()
                defer { session.cancel() }
                func emit(_ tokens: [KyutaiSTTToken]) {
                    for token in tokens { print(token.text, terminator: "") }
                    fflush(stdout)
                }
                for offset in stride(from: 0, to: audio.count, by: KyutaiSTTConfig.frameSize) {
                    emit(try session.append(Array(audio[offset..<min(offset + KyutaiSTTConfig.frameSize, audio.count)])))
                }
                emit(try session.finish())
                print()
                text = session.text
            } else {
                text = try asr.transcribe(audio: audio, sampleRate: KyutaiSTTConfig.sampleRate)
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            print("Result: \(text)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, duration > 0 ? elapsed / duration : 0))
        }
    }

    private func runCohereTranscription() throws {
        try runAsync {
            let modelId = try resolveCohereModelId(model)
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Cohere Transcribe model: \(modelId)")
            let asrModel = try await CohereTranscribeModel.load(
                modelId,
                progressHandler: { reportProgress($0, "Downloading") })

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = asrModel.transcribe(
                audio: audio, sampleRate: 16000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
    }

    private func runVoxtralTranscription() throws {
        try runAsync {
            let modelId = try resolveVoxtralModelId(model)
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Voxtral model: \(modelId)")
            let asrModel = try await VoxtralModel.load(
                modelId,
                progressHandler: { reportProgress($0, "Downloading") })

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = asrModel.transcribe(
                audio: audio, sampleRate: 16000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
    }

    private func runMossTranscription() throws {
        try runAsync {
            let backend = backend.lowercased()
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile),
                targetSampleRate: MossMLXModel.inputSampleRate
            )
            let duration =
                Double(audio.count)
                / Double(MossMLXModel.inputSampleRate)
            print(
                "  Loaded \(audio.count) samples "
                    + "(\(String(format: "%.2f", duration))s)"
            )

            let modelId = try resolveMossModelId(model, backend: backend)
            let result: MossTranscription
            if backend == "mlx" {
                print("Loading MOSS MLX model: \(modelId)")
                let moss: MossMLXModel
                if FileManager.default.fileExists(atPath: modelId) {
                    moss = try await MossMLXModel.fromDirectory(
                        URL(fileURLWithPath: modelId),
                        progressHandler: reportProgress
                    )
                } else {
                    moss = try await MossMLXModel.fromPretrained(
                        modelId: modelId,
                        progressHandler: reportProgress
                    )
                }
                print("Warming up MLX...")
                try moss.warmUp()
                let cachePrecision = MossMLXKVCachePrecision(
                    rawValue: kvCache.lowercased()
                ) ?? .float16
                let options = MossMLXDecodingOptions(
                    maxTokens: maxTokens ?? 5_120,
                    kvCachePrecision: cachePrecision
                )
                print(
                    "Transcribing offline with global context "
                        + "(\(cachePrecision.rawValue) KV cache)..."
                )
                result = try moss.transcribeDetailed(
                    audio: audio,
                    sampleRate: MossMLXModel.inputSampleRate,
                    options: options
                )
            } else {
                print("Loading MOSS Core ML model: \(modelId)")
                let moss: MossTranscribeModel
                if FileManager.default.fileExists(atPath: modelId) {
                    moss = try await MossTranscribeModel.fromDirectory(
                        URL(fileURLWithPath: modelId),
                        progressHandler: reportProgress
                    )
                } else {
                    moss = try await MossTranscribeModel.fromPretrained(
                        modelId: modelId,
                        progressHandler: reportProgress
                    )
                }
                print("Warming up Core ML...")
                try moss.warmUp()
                print("Transcribing with timestamps and speakers...")
                result = try moss.transcribeDetailed(
                    audio: audio,
                    sampleRate: MossTranscribeModel.inputSampleRate,
                    maxNewTokens: maxTokens ?? 512
                )
            }
            print("Result: \(result.text)")
            if !result.segments.isEmpty {
                print("Segments:")
                for segment in result.segments {
                    print(
                        String(
                            format: "  [%.2fs-%.2fs][%@] %@",
                            segment.startTime,
                            segment.endTime,
                            segment.speaker,
                            segment.text
                        )
                    )
                }
            }
            let metrics = result.metrics
            print(
                String(
                    format:
                        "  Time: %.2fs, RTF: %.3f, throughput: %.1fx realtime",
                    metrics.totalSeconds,
                    metrics.realTimeFactor,
                    metrics.realtimeThroughput
                )
            )
            if metrics.stopReason != .endOfSequence {
                print(
                    "  Warning: generation stopped at "
                        + metrics.stopReason.rawValue
                )
            }
        }
    }

    private func runBatchTranscription() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)
            let detectedSize = ASRModelSize.detect(from: modelId)
            let sizeLabel = detectedSize == .large ? "1.7B" : "0.6B"

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 24000)
            print("  Loaded \(audio.count) samples (\(formatDuration(audio.count))s)")

            print("Loading model (\(sizeLabel)): \(modelId)")
            let runtime = Qwen3ASRRuntime(
                descriptor: Qwen3ASRDescriptors.mlx(modelID: modelId), progressHandler: reportProgress)
            try await runtime.prepare()

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await runtime.transcribe(audio: audio, sampleRate: 24000, language: language, context: context)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let duration = Float(audio.count) / 24000.0
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
    }

    private func runStreamingTranscription() throws {
        try runAsync {
            let modelId = resolveASRModelId(model)

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading models...")
            let streaming = try await StreamingASR.fromPretrained(
                asrModelId: modelId, progressHandler: reportProgress)

            let config = StreamingASRConfig(
                maxSegmentDuration: maxSegment,
                language: language,
                emitPartialResults: partial,
                context: context
            )

            print("Streaming transcription (VAD + ASR)...")
            let stream = streaming.transcribeStream(
                audio: audio, sampleRate: 16000, config: config)

            for try await segment in stream {
                let tag = segment.isFinal ? "FINAL" : "partial"
                let start = String(format: "%.2f", segment.startTime)
                let end = String(format: "%.2f", segment.endTime)
                print("[\(start)s-\(end)s] [\(tag)] \(segment.text)")
            }
        }
    }

    private func runCoreMLTranscription() throws {
        #if canImport(CoreML)
        try runAsync {
            let modelId = resolveASRModelId(model)

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            // Load CoreML encoder
            print("Loading CoreML encoder...")
            let coremlEncoder = try await CoreMLASREncoder.fromPretrained(
                progressHandler: reportProgress)

            print("Warming up CoreML...")
            try coremlEncoder.warmUp()

            // Load MLX text decoder
            print("Loading text decoder: \(modelId)")
            let asrModel = try await Qwen3ASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Transcribing (CoreML encoder + MLX decoder)...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try asrModel.transcribe(
                audio: audio, sampleRate: 16000, language: language,
                coremlEncoder: coremlEncoder)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
        #else
        print("CoreML is not available on this platform.")
        #endif
    }

    private func runFullCoreMLTranscription() throws {
        #if canImport(CoreML)
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            if #available(macOS 15, iOS 18, *) {
                print("Loading full CoreML ASR pipeline...")
                let runtime = Qwen3ASRRuntime(descriptor: Qwen3ASRDescriptors.coreML,
                                               progressHandler: reportProgress)
                try await runtime.prepare()

                print("Warming up CoreML...")
                let warmupStart = CFAbsoluteTimeGetCurrent()
                try await runtime.prepare(warmUp: true)
                let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

                print("Transcribing (full CoreML: encoder + decoder)...")
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try await runtime.transcribe(
                    audio: audio, sampleRate: 16000, language: language)
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let rtf = elapsed / Double(duration)

                print("Result: \(result)")
                print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
            } else {
                print("Full CoreML ASR requires macOS 15+ / iOS 18+ for MLState KV cache.")
                print("Use --engine qwen3-coreml for hybrid CoreML encoder + MLX decoder.")
            }
        }
        #else
        print("CoreML is not available on this platform.")
        #endif
    }

    private func runParakeetTranscription() throws {
        let (modelId, directory) = try resolveParakeetModel(model)
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Parakeet-TDT model: \(directory?.path ?? modelId ?? ParakeetASRModel.defaultModelId)")
            let parakeetModel = try await ParakeetASRModel.fromPretrained(
                modelId: modelId, cacheDir: directory, offlineMode: directory != nil,
                progressHandler: reportProgress)

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try parakeetModel.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try parakeetModel.transcribeAudio(audio, sampleRate: 16000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
        }
    }

    private func runNemotronTranscription() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Nemotron Streaming model: \(NemotronStreamingASRModel.defaultModelId)")
            let model = try await NemotronStreamingASRModel.fromPretrained(
                progressHandler: reportProgress)

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try model.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            if stream {
                print("Streaming transcription...")
                let startTime = CFAbsoluteTimeGetCurrent()
                for await partial in model.transcribeStream(audio: audio, sampleRate: 16000) {
                    let tag = partial.isFinal ? "FINAL" : "partial"
                    if partial.isFinal || self.partial {
                        print("[\(tag)] \(partial.text)")
                    }
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let rtf = elapsed / Double(duration)
                print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
            } else {
                print("Transcribing...")
                let startTime = CFAbsoluteTimeGetCurrent()
                let result = try model.transcribeAudio(audio, sampleRate: 16000)
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                let rtf = elapsed / Double(duration)
                print("Result: \(result)")
                print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
            }
        }
    }

    private func runWhisperTranscription() throws {
        try runAsync {
            let modelId = try resolveWhisperModelId(model)

            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            print("Loading Whisper model: \(modelId)")
            let asrModel = try await WhisperASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try await asrModel.transcribeAudio(audio, sampleRate: 16000, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f", elapsed, rtf))
        }
    }

    private func runOmnilingualTranscription() throws {
        if backend.lowercased() == "mlx" {
            try runOmnilingualMLXTranscription()
        } else {
            try runOmnilingualCoreMLTranscription()
        }
    }

    private func runOmnilingualCoreMLTranscription() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            let modelId = window == 5
                ? OmnilingualASRModel.shortWindowModelId
                : OmnilingualASRModel.defaultModelId
            print("Loading Omnilingual ASR model (\(window)s window): \(modelId)")
            let model = try await OmnilingualASRModel.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Warming up CoreML...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try model.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try model.transcribeAudio(audio, sampleRate: 16000)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
        }
    }

    private func runOmnilingualMLXTranscription() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = Float(audio.count) / 16000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s)")

            let resolved = OmnilingualMLXConfig.Variant(rawValue: variant.uppercased()) ?? .m300
            let modelId = OmnilingualMLXConfig.defaultModelId(variant: resolved, bits: bits)
            print("Loading Omnilingual MLX model (\(resolved.rawValue), \(bits)-bit): \(modelId)")
            let model = try await OmnilingualASRMLXModel.fromPretrained(
                variant: resolved, bits: bits, progressHandler: reportProgress)

            print("Warming up MLX...")
            let warmupStart = CFAbsoluteTimeGetCurrent()
            try model.warmUp()
            let warmupTime = CFAbsoluteTimeGetCurrent() - warmupStart

            print("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let result = try model.transcribeAudio(audio, sampleRate: 16000)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / Double(duration)

            print("Result: \(result)")
            print(String(format: "  Time: %.2fs, RTF: %.3f (warmup: %.2fs)", elapsed, rtf, warmupTime))
        }
    }
}

/// Resolve shorthand model specifiers to HuggingFace model IDs.
public func resolveASRModelId(_ specifier: String) -> String {
    Qwen3ASRDescriptors.resolveModelID(specifier)
}

/// Resolve Whisper model aliases to the CoreML repo layout used by WhisperASR.
public func resolveWhisperModelId(_ specifier: String) throws -> String {
    switch specifier.lowercased() {
    case "0.6b", "default", "turbo", "whisper", "whisper-turbo", "large-v3-turbo",
         "large-v3-v20240930-turbo", "large-v3-v20240930_turbo":
        return WhisperASRModel.defaultModelId
    default:
        if specifier.contains("/") {
            return specifier
        }
        throw ValidationError("[whisper] --model must be 'default', 'turbo', or a full CoreML HuggingFace repo ID")
    }
}

/// Resolve Cohere Transcribe aliases to published MLX bundles.
public func resolveCohereModelId(_ specifier: String) throws -> String {
    try resolveNewASRModelId(
        specifier,
        engine: "cohere",
        fp16: CohereTranscribeVariant.fp16.modelId,
        int5: CohereTranscribeVariant.int5.modelId,
        int8: CohereTranscribeVariant.int8.modelId)
}

/// Resolve Voxtral aliases to published MLX bundles.
public func resolveVoxtralModelId(_ specifier: String) throws -> String {
    try resolveNewASRModelId(
        specifier,
        engine: "voxtral",
        fp16: VoxtralVariant.fp16.modelId,
        int5: VoxtralVariant.int5.modelId,
        int8: VoxtralVariant.int8.modelId)
}

private func resolveNewASRModelId(
    _ specifier: String,
    engine: String,
    fp16: String,
    int5: String,
    int8: String
) throws -> String {
    switch specifier.lowercased() {
    case "0.6b", "default", "int5", "5bit":
        return int5
    case "int8", "8bit":
        return int8
    case "fp16", "16bit":
        return fp16
    case "int7", "7bit":
        throw ValidationError(
            "[\(engine)] INT7 is not supported by MLX affine quantization; use int8")
    default:
        if specifier.contains("/") {
            return specifier
        }
        throw ValidationError(
            "[\(engine)] --model must be int5, int8, fp16, a Hugging Face model ID, or a local directory")
    }
}

/// Resolve MOSS model aliases for either the Core ML or MLX runtime.
public func resolveMossModelId(
    _ specifier: String,
    backend: String = "coreml"
) throws -> String {
    switch backend.lowercased() {
    case "coreml":
        switch specifier.lowercased() {
        case "0.6b", "0.9b", "default", "int8", "8bit":
            return MossModelVariant.int8.modelId
        case "fp16", "16bit":
            return MossModelVariant.fp16.modelId
        default:
            if specifier.contains("/") {
                return specifier
            }
            throw ValidationError(
                "[moss coreml] --model must be 'int8', 'fp16', "
                    + "a Hugging Face model ID, or a local directory"
            )
        }
    case "mlx":
        switch specifier.lowercased() {
        case "0.6b", "0.9b", "default", "int5", "5bit":
            return MossMLXVariant.int5.modelId
        case "int8", "8bit":
            return MossMLXVariant.int8.modelId
        default:
            if specifier.contains("/") {
                return specifier
            }
            throw ValidationError(
                "[moss mlx] --model must be 'int5', 'int8', "
                    + "a Hugging Face model ID, or a local directory"
            )
        }
    default:
        throw ValidationError("[moss] --backend must be 'coreml' or 'mlx'")
    }
}

/// Resolve Parakeet aliases, Hub model IDs and local Core ML bundles.
func resolveParakeetModel(_ value: String) throws -> (modelId: String?, directory: URL?) {
    switch value.lowercased() {
    case "0.6b", "default", "v3": return (nil, nil)
    case "ultra": return (ParakeetASRModel.ultraModelId, nil)
    default:
        let path = (value as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue {
            return (nil, URL(fileURLWithPath: path, isDirectory: true))
        }
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        if parts.count == 2, parts.allSatisfy({ !$0.isEmpty && !$0.hasPrefix(".") }), !value.hasPrefix("~") {
            return (value, nil)
        }
        throw ValidationError("Parakeet --model must be v3, ultra, a Hugging Face model ID, or an existing local bundle directory")
    }
}

/// Resolve published precision aliases, original checkpoints, and local bundles.
func resolveKyutaiModelId(_ value: String) throws -> String {
    switch value.lowercased() {
    case "0.6b", "default", "1b", "1b-en-fr": return KyutaiSTTModel.defaultModelId
    case "fp16": return KyutaiSTTVariant.fp16.modelId
    case "int4", "4bit": return KyutaiSTTVariant.int4.modelId
    case "int8", "8bit": return KyutaiSTTVariant.int8.modelId
    case "upstream": return KyutaiSTTModel.upstreamModelId
    case "2.6b", "2.6b-en": return KyutaiSTTModel.englishModelId
    default:
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: value, isDirectory: &isDirectory), isDirectory.boolValue {
            return value
        }
        if value.split(separator: "/").count == 2 && !value.hasPrefix(".") && !value.hasPrefix("/") {
            return value
        }
        throw ValidationError("Kyutai --model must be 1b, fp16, int4, int8, upstream, 2.6b, a HuggingFace model ID, or an existing local export directory")
    }
}
