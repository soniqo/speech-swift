import Foundation
import AudioCommon

extension NemotronStreamingASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Nemotron streaming transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        let text = transcribe(audio: audio, sampleRate: sampleRate, language: language)
        // Multilingual: report the requested language verbatim if recognized,
        // otherwise nil. The model itself does not emit a confidence score.
        let reportedLang: String?
        if text.isEmpty {
            reportedLang = nil
        } else if let lang = language, languages.promptDictionary[lang] != nil {
            reportedLang = lang
        } else {
            reportedLang = nil
        }
        return TranscriptionResult(text: text, language: reportedLang, confidence: 0)
    }
}
