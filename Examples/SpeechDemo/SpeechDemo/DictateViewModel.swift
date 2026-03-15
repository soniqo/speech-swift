#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif
import Foundation
import Observation
import ParakeetASR
#if os(macOS)
import Qwen3ASR
#endif

enum ASREngine: String, CaseIterable, Identifiable {
    case parakeet = "Parakeet TDT"
    #if os(macOS)
    case qwen3 = "Qwen3-ASR"
    #endif
    var id: String { rawValue }
}

@Observable
@MainActor
final class DictateViewModel {
    var transcription = ""
    var isLoading = false
    var isRecording = false
    var isTranscribing = false
    var loadingStatus = ""
    var errorMessage: String?
    var selectedEngine: ASREngine = .parakeet
    var selectedLanguage: String = "auto"

    private var parakeetModel: ParakeetASRModel?
    #if os(macOS)
    private var qwen3Model: Qwen3ASRModel?
    #endif
    let recorder = AudioRecorder()

    var modelLoaded: Bool {
        switch selectedEngine {
        case .parakeet: return parakeetModel != nil
        #if os(macOS)
        case .qwen3: return qwen3Model != nil
        #endif
        }
    }

    func loadModel() async {
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            switch selectedEngine {
            case .parakeet:
                if parakeetModel == nil {
                    let model = try await Task.detached {
                        try await ParakeetASRModel.fromPretrained { [weak self] progress, status in
                            DispatchQueue.main.async {
                                self?.loadingStatus = status.isEmpty
                                    ? "Downloading... \(Int(progress * 100))%"
                                    : "\(status) (\(Int(progress * 100))%)"
                            }
                        }
                    }.value
                    loadingStatus = "Compiling CoreML model..."
                    try model.warmUp()
                    parakeetModel = model
                }
            #if os(macOS)
            case .qwen3:
                if qwen3Model == nil {
                    let model = try await Task.detached {
                        try await Qwen3ASRModel.fromPretrained { [weak self] progress, status in
                            DispatchQueue.main.async {
                                self?.loadingStatus = status.isEmpty
                                    ? "Downloading... \(Int(progress * 100))%"
                                    : "\(status) (\(Int(progress * 100))%)"
                            }
                        }
                    }.value
                    qwen3Model = model
                }
            #endif
            }
            loadingStatus = ""
        } catch {
            errorMessage = "Failed to load model: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    func startRecording() {
        transcription = ""
        errorMessage = nil
        recorder.startRecording()
        isRecording = true
    }

    func stopAndTranscribe() async {
        let audio = recorder.stopRecording()
        isRecording = false

        guard !audio.isEmpty else {
            errorMessage = "No audio captured."
            return
        }

        isTranscribing = true
        errorMessage = nil

        do {
            let text: String
            switch selectedEngine {
            case .parakeet:
                guard let model = parakeetModel else {
                    errorMessage = "Model not loaded."
                    isTranscribing = false
                    return
                }
                text = try model.transcribeAudio(audio, sampleRate: 16000)
            #if os(macOS)
            case .qwen3:
                guard let model = qwen3Model else {
                    errorMessage = "Model not loaded."
                    isTranscribing = false
                    return
                }
                let lang: String? = selectedLanguage == "auto" ? nil : selectedLanguage
                text = model.transcribe(audio: audio, sampleRate: 16000, language: lang)
            #endif
            }
            transcription = text
        } catch {
            errorMessage = "Transcription failed: \(error.localizedDescription)"
        }

        isTranscribing = false
    }

    func copyToClipboard() {
        #if os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(transcription, forType: .string)
        #elseif os(iOS)
        UIPasteboard.general.string = transcription
        #endif
    }
}
