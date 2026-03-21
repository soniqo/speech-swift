import AVFoundation
import AudioCommon

/// Wraps Apple's AVSpeechSynthesizer as a SpeechGenerationModel.
/// Zero CoreML memory — uses the built-in system speech engine.
/// Quality is lower than Kokoro but fits in minimal memory tier.
final class SystemTTS: SpeechGenerationModel {
    let sampleRate: Int = 24000

    func generate(text: String, language: String?) async throws -> [Float] {
        // AVSpeechSynthesizer plays audio directly — return empty samples.
        // The pipeline will hear silence and move on; actual audio plays via system.
        let synthesizer = AVSpeechSynthesizer()
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate

        return await withCheckedContinuation { cont in
            let delegate = SystemTTSDelegate {
                cont.resume(returning: [])
            }
            // Keep delegate alive
            objc_setAssociatedObject(synthesizer, "delegate", delegate, .OBJC_ASSOCIATION_RETAIN)
            synthesizer.delegate = delegate
            synthesizer.speak(utterance)
        }
    }
}

private final class SystemTTSDelegate: NSObject, AVSpeechSynthesizerDelegate {
    let onFinished: () -> Void
    init(onFinished: @escaping () -> Void) { self.onFinished = onFinished }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        onFinished()
    }
}
