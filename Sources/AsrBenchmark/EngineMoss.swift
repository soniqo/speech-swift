import Foundation
import MossTranscribe

final class MossCoreMLEngine: BenchEngine, @unchecked Sendable {
    let variant: MossModelVariant
    let name: String
    private(set) var loadElapsed: Double = 0
    private var model: MossTranscribeModel?

    init(variant: MossModelVariant) {
        self.variant = variant
        name = "moss-coreml-\(variant.rawValue)"
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let started = Date()
        let model = try await MossTranscribeModel.fromPretrained(
            variant: variant
        )
        _ = try model.transcribeDetailed(
            audio: warmupAudio,
            sampleRate: sampleRate
        )
        self.model = model
        loadElapsed = Date().timeIntervalSince(started)
    }

    func transcribe(
        audio: [Float],
        sampleRate: Int,
        language: String?
    ) async throws -> (text: String, timings: Timings) {
        guard let model else { throw BenchError.notLoaded(name) }
        var text = ""
        var elapsed = 0.0
        try autoreleasepool {
            let started = Date()
            text = try model.transcribeDetailed(
                audio: audio,
                sampleRate: sampleRate
            ).text
            elapsed = Date().timeIntervalSince(started)
        }
        let duration = Double(audio.count) / Double(sampleRate)
        return (
            text,
            Timings(elapsed: elapsed, audioDuration: duration)
        )
    }
}
