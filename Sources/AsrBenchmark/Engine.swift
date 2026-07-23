import Foundation

/// Wall-clock timings collected for a single transcription call.
public struct Timings: Sendable {
    public var elapsed: Double       // synthesis time in seconds
    public var audioDuration: Double // input audio duration in seconds

    public var rtf: Double { elapsed / max(audioDuration, 1e-6) }
}

/// A single benchmark engine — model load up front, repeated transcribe calls.
///
/// Implementations are intentionally minimal: load the model, hold it, expose
/// a transcribe call that returns text + wall-clock timing. The harness owns
/// dataset, normalization, WER, and reporting.
public protocol BenchEngine: AnyObject, Sendable {
    /// Stable short name for the report row.
    var name: String { get }

    /// Time spent loading the model from disk + warming up, in seconds.
    /// Populated by `load()`. `0` until then.
    var loadElapsed: Double { get }

    /// Loads weights and runs a single warmup transcription if the engine
    /// benefits from one (most CoreML/MLX backends do). The warmup pass is
    /// included in `loadElapsed`.
    func load(warmupAudio: [Float], sampleRate: Int) async throws

    /// Transcribes a single mono Float32 utterance. Returns hypothesis text +
    /// the wall-clock elapsed time. The hypothesis is returned raw — caller
    /// runs the normalizer before WER.
    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings)
}

public enum EngineID: String, CaseIterable, Sendable {
    case cohereMLXFP16 = "cohere-transcribe-mlx-fp16"
    case cohereMLXInt5 = "cohere-transcribe-mlx-int5"
    case cohereMLXInt8 = "cohere-transcribe-mlx-int8"
    case qwen3CoreML = "qwen3-coreml"
    case qwen3MLX06b4bit = "qwen3-mlx-0.6b-4bit"
    case qwen3MLX06b5bit = "qwen3-mlx-0.6b-5bit"
    case qwen3MLX06b8bit = "qwen3-mlx-0.6b-8bit"
    case qwen3MLX17b4bit = "qwen3-mlx-1.7b-4bit"
    case qwen3MLX17b5bit = "qwen3-mlx-1.7b-5bit"
    case qwen3MLX17b8bit = "qwen3-mlx-1.7b-8bit"
    case parakeet = "parakeet"
    case nemotron = "nemotron"
    case omnilingual = "omnilingual"
    case omnilingualMLX300m4bit = "omnilingual-mlx-300m-4bit"
    case omnilingualMLX1b4bit = "omnilingual-mlx-1b-4bit"
    case omnilingualMLX3b4bit = "omnilingual-mlx-3b-4bit"
    case omnilingualMLX7b4bit = "omnilingual-mlx-7b-4bit"
    case whisperASRTurbo = "whisper-asr-turbo"
    case whisperKitLargeV3Turbo = "whisperkit-large-v3-turbo"
    case whisperKitLargeV3 = "whisperkit-large-v3"
    case whisperKitDistilLargeV3 = "whisperkit-distil-large-v3"
    case voxtralMLXFP16 = "voxtral-mini-mlx-fp16"
    case voxtralMLXInt5 = "voxtral-mini-mlx-int5"
    case voxtralMLXInt8 = "voxtral-mini-mlx-int8"

    public func make() -> BenchEngine {
        switch self {
        case .cohereMLXFP16: return CohereTranscribeMLXEngine(variant: .fp16)
        case .cohereMLXInt5: return CohereTranscribeMLXEngine(variant: .int5)
        case .cohereMLXInt8: return CohereTranscribeMLXEngine(variant: .int8)
        case .qwen3CoreML: return Qwen3CoreMLEngine()
        case .qwen3MLX06b4bit: return Qwen3MLXEngine(size: "0.6B", bits: 4)
        case .qwen3MLX06b5bit: return Qwen3MLXEngine(size: "0.6B", bits: 5)
        case .qwen3MLX06b8bit: return Qwen3MLXEngine(size: "0.6B", bits: 8)
        case .qwen3MLX17b4bit: return Qwen3MLXEngine(size: "1.7B", bits: 4)
        case .qwen3MLX17b5bit: return Qwen3MLXEngine(size: "1.7B", bits: 5)
        case .qwen3MLX17b8bit: return Qwen3MLXEngine(size: "1.7B", bits: 8)
        case .parakeet: return ParakeetEngine()
        case .nemotron: return NemotronEngine()
        case .omnilingual: return OmnilingualEngine()
        case .omnilingualMLX300m4bit: return OmnilingualMLXEngine(variant: .m300, bits: 4)
        case .omnilingualMLX1b4bit: return OmnilingualMLXEngine(variant: .b1, bits: 4)
        case .omnilingualMLX3b4bit: return OmnilingualMLXEngine(variant: .b3, bits: 4)
        case .omnilingualMLX7b4bit: return OmnilingualMLXEngine(variant: .b7, bits: 4)
        case .whisperASRTurbo: return WhisperASREngine()
        case .whisperKitLargeV3Turbo: return WhisperKitEngine(model: "openai_whisper-large-v3-v20240930_turbo")
        case .whisperKitLargeV3: return WhisperKitEngine(model: "openai_whisper-large-v3")
        case .whisperKitDistilLargeV3: return WhisperKitEngine(model: "distil-whisper_distil-large-v3")
        case .voxtralMLXFP16: return VoxtralMLXEngine(variant: .fp16)
        case .voxtralMLXInt5: return VoxtralMLXEngine(variant: .int5)
        case .voxtralMLXInt8: return VoxtralMLXEngine(variant: .int8)
        }
    }
}
