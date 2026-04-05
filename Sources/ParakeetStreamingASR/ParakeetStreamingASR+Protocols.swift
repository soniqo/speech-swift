import Foundation
import AudioCommon
import NaturalLanguage

extension ParakeetStreamingASRModel: SpeechRecognitionModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        do {
            return try transcribeAudio(audio, sampleRate: sampleRate, language: language)
        } catch {
            AudioLog.inference.error("Parakeet EOU transcription failed: \(error)")
            return ""
        }
    }

    public func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        let text = transcribe(audio: audio, sampleRate: sampleRate, language: language)
        guard !text.isEmpty else { return TranscriptionResult(text: text) }
        let detectedLang = detectLanguage(text)
        return TranscriptionResult(text: text, language: detectedLang, confidence: 0)
    }

    private func detectLanguage(_ text: String) -> String? {
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let lang = recognizer.dominantLanguage else { return nil }
        switch lang {
        case .english: return "english"
        case .russian: return "russian"
        case .german: return "german"
        case .french: return "french"
        case .spanish: return "spanish"
        case .italian: return "italian"
        case .portuguese: return "portuguese"
        case .dutch: return "dutch"
        case .polish: return "polish"
        case .swedish: return "swedish"
        default: return lang.rawValue
        }
    }
}
