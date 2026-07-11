import Foundation
import Qwen3ASR

final class Qwen3CoreMLEngine: BenchEngine, @unchecked Sendable {
    let name = "qwen3-asr-coreml"
    private(set) var loadElapsed: Double = 0
    private var model: CoreMLASRModel?

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let m = try await CoreMLASRModel.fromPretrained()
        _ = m.transcribe(audio: warmupAudio, sampleRate: sampleRate, language: nil)
        self.model = m
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let m = model else { throw BenchError.notLoaded(name) }
        // Wrap in an autoreleasepool so CoreML's per-prediction IOSurface
        // buffers (MLMultiArray scratch, MLState backing pages) are released
        // between calls instead of piling up. Without this the bench grows
        // ~30 MB / utt and peaks above 9 GB by utt 200 on M5 Pro.
        var text = ""
        var elapsed: Double = 0
        autoreleasepool {
            let t0 = Date()
            text = m.transcribe(audio: audio, sampleRate: sampleRate, language: language)
            elapsed = Date().timeIntervalSince(t0)
        }
        let dur = Double(audio.count) / Double(sampleRate)
        return (text, Timings(elapsed: elapsed, audioDuration: dur))
    }
}

enum BenchError: Error, CustomStringConvertible {
    case notLoaded(String)
    case datasetEmpty(String)
    case datasetMalformed(String)
    case engineFailed(String, underlying: Error)

    var description: String {
        switch self {
        case .notLoaded(let n): return "engine '\(n)' was not loaded before transcribe()"
        case .datasetEmpty(let p): return "no utterances found under \(p)"
        case .datasetMalformed(let m): return "dataset malformed: \(m)"
        case .engineFailed(let n, let e): return "engine '\(n)' failed: \(e)"
        }
    }
}
