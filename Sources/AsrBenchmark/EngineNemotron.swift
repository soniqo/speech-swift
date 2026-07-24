import Foundation
import NemotronStreamingASR

final class NemotronEngine: BenchEngine, @unchecked Sendable {
    let name = "nemotron-streaming-coreml-int8"
    private(set) var loadElapsed: Double = 0
    private var model: NemotronStreamingASRModel?

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let m = try await NemotronStreamingASRModel.fromPretrained()
        _ = try m.transcribeAudio(warmupAudio, sampleRate: sampleRate, language: nil)
        self.model = m
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let m = model else { throw BenchError.notLoaded(name) }
        var result: (text: String, elapsed: Double)!
        try autoreleasepool {
            let t0 = Date()
            let text = try m.transcribeAudio(audio, sampleRate: sampleRate, language: language)
            result = (text, Date().timeIntervalSince(t0))
        }
        let dur = Double(audio.count) / Double(sampleRate)
        return (result.text, Timings(elapsed: result.elapsed, audioDuration: dur))
    }
}

final class NemotronMLXEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private(set) var loadElapsed: Double = 0
    private let variant: NemotronMLXVariant
    private var model: NemotronStreamingASRMLXModel?

    init(variant: NemotronMLXVariant) {
        self.variant = variant
        name = "nemotron-streaming-mlx-\(variant.rawValue)"
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let started = Date()
        let environmentKey = variant == .int5
            ? "NEMOTRON_MLX_LOCAL_BUNDLE"
            : "NEMOTRON_MLX_INT8_LOCAL_BUNDLE"
        let loaded: NemotronStreamingASRMLXModel
        if let path = ProcessInfo.processInfo.environment[environmentKey],
           !path.isEmpty
        {
            loaded = try await NemotronStreamingASRMLXModel.fromDirectory(
                URL(fileURLWithPath: path, isDirectory: true))
        } else {
            loaded = try await NemotronStreamingASRMLXModel.fromPretrained(
                variant: variant)
        }
        _ = try loaded.transcribeAudio(
            warmupAudio,
            sampleRate: sampleRate,
            language: "en-US")
        model = loaded
        loadElapsed = Date().timeIntervalSince(started)
    }

    func transcribe(
        audio: [Float],
        sampleRate: Int,
        language: String?
    ) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        var result: (text: String, elapsed: Double)!
        try autoreleasepool {
            let started = Date()
            let text = try model.transcribeAudio(
                audio,
                sampleRate: sampleRate,
                language: language)
            result = (text, Date().timeIntervalSince(started))
        }
        return (
            result.text,
            Timings(
                elapsed: result.elapsed,
                audioDuration: Double(audio.count) / Double(sampleRate)))
    }
}
