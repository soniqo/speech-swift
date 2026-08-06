import AudioCommon
import Foundation

extension CanaryASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        transcribeWithLanguage(audio: audio, sampleRate: sampleRate, language: language).text
    }

    public func transcribeWithLanguage(
        audio: [Float], sampleRate: Int, language: String?
    ) -> TranscriptionResult {
        do {
            // Canary decodes from a language prompt rather than detecting, so
            // the language it was asked for is the language it produced — no
            // detector pass afterwards.
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Canary transcription failed: \(error)")
            return TranscriptionResult(text: "")
        }
    }
}
