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
