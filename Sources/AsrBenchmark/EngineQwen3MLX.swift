import Foundation
import Qwen3ASR

final class Qwen3MLXEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let modelId: String
    private(set) var loadElapsed: Double = 0
    private var model: Qwen3ASRModel?

    init(size: String, bits: Int) {
        let sizeUpper = size.uppercased()
        self.modelId = "aufklarer/Qwen3-ASR-\(sizeUpper)-MLX-\(bits)bit"
        self.name = "qwen3-asr-mlx-\(size.lowercased())-\(bits)bit"
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let m = try await Qwen3ASRModel.fromPretrained(modelId: modelId)
        _ = m.transcribe(audio: warmupAudio, sampleRate: sampleRate, language: nil)
        self.model = m
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let m = model else { throw BenchError.notLoaded(name) }
        let t0 = Date()
        let text = m.transcribe(audio: audio, sampleRate: sampleRate, language: language)
        let elapsed = Date().timeIntervalSince(t0)
        let dur = Double(audio.count) / Double(sampleRate)
        return (text, Timings(elapsed: elapsed, audioDuration: dur))
    }
}
