import Foundation
import OmnilingualASR

final class OmnilingualMLXEngine: BenchEngine, @unchecked Sendable {
    let name: String
    private let variant: OmnilingualMLXConfig.Variant
    private let bits: Int
    private(set) var loadElapsed: Double = 0
    private var model: OmnilingualASRMLXModel?

    init(variant: OmnilingualMLXConfig.Variant, bits: Int) {
        self.variant = variant
        self.bits = bits
        self.name = "omnilingual-mlx-\(variant.rawValue.lowercased())-\(bits)bit"
    }

    func load(warmupAudio: [Float], sampleRate: Int) async throws {
        let t0 = Date()
        let m = try await OmnilingualASRMLXModel.fromPretrained(variant: variant, bits: bits)
        _ = try m.transcribeAudio(warmupAudio, sampleRate: sampleRate, language: nil)
        self.model = m
        self.loadElapsed = Date().timeIntervalSince(t0)
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) async throws -> (text: String, timings: Timings) {
        guard let m = model else { throw BenchError.notLoaded(name) }
        let t0 = Date()
        let text = try m.transcribeAudio(audio, sampleRate: sampleRate, language: language)
        let elapsed = Date().timeIntervalSince(t0)
        let dur = Double(audio.count) / Double(sampleRate)
        return (text, Timings(elapsed: elapsed, audioDuration: dur))
    }
}
