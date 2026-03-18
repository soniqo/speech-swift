import AVFoundation
import AudioCommon
import Speech

/// Wraps Apple's SFSpeechRecognizer as a SpeechRecognitionModel.
/// Zero memory footprint — uses the built-in on-device speech engine.
final class AppleSpeechASR: SpeechRecognitionModel {
    let inputSampleRate: Int = 16000

    private let recognizer: SFSpeechRecognizer
    private let onDevice: Bool

    init(locale: Locale = Locale(identifier: "en-US")) {
        self.recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer()!
        self.onDevice = recognizer.supportsOnDeviceRecognition
        if onDevice {
            recognizer.defaultTaskHint = .dictation
        }
    }

    /// Request speech recognition permission (call once at app startup).
    static func requestAuthorization() async -> Bool {
        await withCheckedContinuation { cont in
            SFSpeechRecognizer.requestAuthorization { status in
                cont.resume(returning: status == .authorized)
            }
        }
    }

    func transcribe(audio: [Float], sampleRate: Int, language: String?) -> String {
        transcribeWithLanguage(audio: audio, sampleRate: sampleRate, language: language).text
    }

    func transcribeWithLanguage(audio: [Float], sampleRate: Int, language: String?) -> TranscriptionResult {
        guard !audio.isEmpty else { return TranscriptionResult(text: "") }

        // Convert Float32 PCM to AVAudioPCMBuffer
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sampleRate),
            channels: 1,
            interleaved: false
        ) else {
            return TranscriptionResult(text: "")
        }

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audio.count)) else {
            return TranscriptionResult(text: "")
        }
        buffer.frameLength = AVAudioFrameCount(audio.count)
        memcpy(buffer.floatChannelData![0], audio, audio.count * MemoryLayout<Float>.size)

        // Synchronous recognition using semaphore (called from C++ worker thread)
        let sem = DispatchSemaphore(value: 0)
        var resultText = ""
        var resultConfidence: Float = 0
        var detectedLocale: String?

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = false
        if onDevice {
            request.requiresOnDeviceRecognition = true
        }

        request.append(buffer)
        request.endAudio()

        recognizer.recognitionTask(with: request) { result, error in
            defer { sem.signal() }
            guard let result, result.isFinal else {
                if error != nil { sem.signal() }
                return
            }
            resultText = result.bestTranscription.formattedString
            // Confidence from segments
            let segments = result.bestTranscription.segments
            if !segments.isEmpty {
                resultConfidence = segments.map(\.confidence).reduce(0, +) / Float(segments.count)
            }
            detectedLocale = result.bestTranscription.formattedString.isEmpty ? nil : "english"
        }

        // Wait up to 10 seconds for recognition
        let timeout = sem.wait(timeout: .now() + 10)
        if timeout == .timedOut {
            return TranscriptionResult(text: "")
        }

        return TranscriptionResult(
            text: resultText,
            language: detectedLocale,
            confidence: resultConfidence
        )
    }
}
