import Foundation
import WhisperASR

final class WhisperASREngine: BenchEngine, @unchecked Sendable {
    let name = "whisper-asr-large-v3-turbo"
    private(set) var loadElapsed: Double = 0
    private var model: WhisperASRModel?

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let model = try await WhisperASRModel.fromPretrained()
        _ = try await model.transcribeAudio(warmupAudio, sampleRate: sampleRate)
        self.model = model
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        let t0 = Date()
        let text = try await model.transcribeAudio(audio, sampleRate: sampleRate, language: language)
        let elapsed = Date().timeIntervalSince(t0)
        let dur = Double(audio.count) / Double(sampleRate)
        return (text, Timings(elapsed: elapsed, audioDuration: dur))
    }
}
