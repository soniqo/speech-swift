import AudioCommon

extension KokoroTTSModel: SpeechGenerationModel {
    public var sampleRate: Int { config.sampleRate }

    public func generate(text: String, language: String?) async throws -> [Float] {
        try synthesize(text: text, voice: "af_heart", language: language ?? "en")
    }

    public func generateStream(text: String, language: String?) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Kokoro is non-autoregressive — single shot, no streaming.
                    // Yield the full result as one chunk.
                    let audio = try self.synthesize(
                        text: text, voice: "af_heart", language: language ?? "en")
                    let chunk = AudioChunk(
                        samples: audio,
                        sampleRate: self.config.sampleRate,
                        frameIndex: 0,
                        isFinal: true
                    )
                    continuation.yield(chunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
