#if canImport(CoreML)
import Foundation
import AudioCommon

/// Conform to the shared `SpeechGenerationModel` so SupertonicTTS drops into the same pipelines /
/// CLI / server as Kokoro & the other TTS models. `generateStream` uses the protocol's default
/// (one final `AudioChunk`) — Supertonic synthesis is non-streaming.
extension SupertonicTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { 44100 }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try synthesize(text: text, voice: nil, language: language ?? "en")
    }
}
#endif
