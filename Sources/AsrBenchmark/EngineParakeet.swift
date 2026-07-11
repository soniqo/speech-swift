import Foundation
import ParakeetASR

final class ParakeetEngine: BenchEngine, @unchecked Sendable {
    let name = "parakeet-tdt-coreml-int8"
    private(set) var loadElapsed: Double = 0
    private var model: ParakeetASRModel?

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let m = try await ParakeetASRModel.fromPretrained()
        _ = try m.transcribeAudio(warmupAudio, sampleRate: sampleRate, language: nil)
        self.model = m
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let m = model else { throw BenchError.notLoaded(name) }
        // Wrap in an autoreleasepool so CoreML's per-prediction IOSurface
        // backing buffers get released after each call instead of piling up.
        // Without this the bench dies at ~10 utterances with
        // 'Failed to allocate memory IOSurface object'.
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
