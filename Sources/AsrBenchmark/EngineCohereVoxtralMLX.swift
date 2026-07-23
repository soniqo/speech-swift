import CohereTranscribeASR
import Foundation
import VoxtralASR

private func configuredModelPath(environmentKey: String, fallback: String) -> String {
    let value = ProcessInfo.processInfo.environment[environmentKey]
    return value.flatMap { $0.isEmpty ? nil : $0 } ?? fallback
}

final class CohereTranscribeMLXEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let modelPath: String
    private(set) var loadElapsed: Double = 0
    private var model: CohereTranscribeModel?

    init(variant: CohereTranscribeVariant) {
        let suffix: String
        let environmentKey: String
        switch variant {
        case .fp16:
            suffix = "fp16"
            environmentKey = "COHERE_MLX_FP16_MODEL_PATH"
        case .int5:
            suffix = "int5"
            environmentKey = "COHERE_MLX_INT5_MODEL_PATH"
        case .int8:
            suffix = "int8"
            environmentKey = "COHERE_MLX_INT8_MODEL_PATH"
        }
        name = "cohere-transcribe-mlx-\(suffix)"
        modelPath = configuredModelPath(environmentKey: environmentKey, fallback: variant.modelId)
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let start = Date()
        let loaded = try await CohereTranscribeModel.load(modelPath)
        _ = loaded.transcribe(
            audio: warmupAudio,
            sampleRate: sampleRate,
            language: "en")
        model = loaded
        loadElapsed = Date().timeIntervalSince(start)
    }

    func transcribe(
        audio: [Float],
        sampleRate: Int,
        language: String?
    ) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        let start = Date()
        let text = model.transcribe(
            audio: audio,
            sampleRate: sampleRate,
            language: language ?? "en")
        return (
            text,
            Timings(
                elapsed: Date().timeIntervalSince(start),
                audioDuration: Double(audio.count) / Double(sampleRate)))
    }
}

final class VoxtralMLXEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let modelPath: String
    private(set) var loadElapsed: Double = 0
    private var model: VoxtralModel?

    init(variant: VoxtralVariant) {
        let suffix: String
        let environmentKey: String
        switch variant {
        case .fp16:
            suffix = "fp16"
            environmentKey = "VOXTRAL_MLX_FP16_MODEL_PATH"
        case .int5:
            suffix = "int5"
            environmentKey = "VOXTRAL_MLX_INT5_MODEL_PATH"
        case .int8:
            suffix = "int8"
            environmentKey = "VOXTRAL_MLX_INT8_MODEL_PATH"
        }
        name = "voxtral-mini-mlx-\(suffix)"
        modelPath = configuredModelPath(environmentKey: environmentKey, fallback: variant.modelId)
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let start = Date()
        let loaded = try await VoxtralModel.load(modelPath)
        _ = loaded.transcribe(
            audio: warmupAudio,
            sampleRate: sampleRate,
            language: "en")
        model = loaded
        loadElapsed = Date().timeIntervalSince(start)
    }

    func transcribe(
        audio: [Float],
        sampleRate: Int,
        language: String?
    ) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        let start = Date()
        let text = model.transcribe(
            audio: audio,
            sampleRate: sampleRate,
            language: language)
        return (
            text,
            Timings(
                elapsed: Date().timeIntervalSince(start),
                audioDuration: Double(audio.count) / Double(sampleRate)))
    }
}
