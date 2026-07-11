import AudioCommon

/// Sidon conforms to the shared `SpeechEnhancementModel` protocol so it can be
/// used anywhere a noise-suppressor can. `enhance` is an alias for `restore`;
/// the input sample rate the protocol advertises is the front-end's 16 kHz.
///
/// Note the **output** is 48 kHz (the DAC vocoder), unlike DeepFilterNet3 which
/// returns audio at its input rate — callers that care should use `restore`
/// directly and read `SpeechRestorer.outputSampleRate`.
extension SpeechRestorer: SpeechEnhancementModel {
    public var inputSampleRate: Int { Self.inputSampleRate }

    public func enhance(audio: [Float], sampleRate: Int) throws -> [Float] {
        try restore(audio: audio, sampleRate: sampleRate)
    }
}
