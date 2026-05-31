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
        let t0 = Date()
        let text = m.transcribe(audio: audio, sampleRate: sampleRate, language: language)
        let elapsed = Date().timeIntervalSince(t0)
        if ProcessInfo.processInfo.environment["BENCH_DUMP_HYP"] == "1" {
            FileHandle.standardError.write(Data(("[\(name)] elapsed=\(elapsed)s hyp=\"\(text)\"\n").utf8))
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
