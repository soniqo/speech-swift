import Foundation
import WhisperKit

final class WhisperKitEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let modelVariant: String
    private(set) var loadElapsed: Double = 0
    private var pipe: WhisperKit?

    init(model: String) {
        self.modelVariant = model
        self.name = "whisperkit-\(model.replacingOccurrences(of: "openai_whisper-", with: ""))"
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let config = WhisperKitConfig(model: modelVariant)
        let pipe = try await WhisperKit(config)
        _ = try await pipe.transcribe(audioArray: warmupAudio)
        self.pipe = pipe
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let pipe = pipe else { throw BenchError.notLoaded(name) }
        // WhisperKit expects 16 kHz mono Float32 — caller is responsible.
        let t0 = Date()
        let options = DecodingOptions(task: .transcribe, language: language, temperatureFallbackCount: 0)
        let results = try await pipe.transcribe(audioArray: audio, decodeOptions: options)
        let elapsed = Date().timeIntervalSince(t0)
        let text = results.map { $0.text }.joined(separator: " ")
        let dur = Double(audio.count) / Double(sampleRate)
        return (text, Timings(elapsed: elapsed, audioDuration: dur))
    }
}
