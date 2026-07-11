import AudioCommon

/// Conform `FlashSR` to the existing audio→audio enhancement protocol.
/// Semantically a stretch — `enhance` was minted for noise suppression
/// (DeepFilterNet3) — but the signature is identical (audio, sampleRate → audio)
/// and FlashSR's output is the same sample rate as its input (48 kHz), so a
/// VoicePipeline-style consumer can swap implementations.
extension FlashSR: SpeechEnhancementModel {
    public var inputSampleRate: Int { FlashSR.sampleRate }

    public func enhance(audio: [Float], sampleRate: Int) throws -> [Float] {
        return try upsample(audio: audio, sampleRate: sampleRate)
    }
}
