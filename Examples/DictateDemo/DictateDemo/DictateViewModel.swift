import AppKit
import Foundation
import Observation
import ParakeetStreamingASR

@Observable
@MainActor
final class DictateViewModel {
    var partialText = ""
    var committedText = ""
    var isRecording = false
    var isLoading = false
    var loadingStatus = ""
    var errorMessage: String?

    private var model: ParakeetStreamingASRModel?
    private var session: StreamingSession?
    private let recorder = StreamingRecorder()
    private var processingTask: Task<Void, Never>?

    var modelLoaded: Bool { model != nil }
    var audioLevel: Float { recorder.audioLevel }

    /// Full transcript: committed segments + current partial
    var fullText: String {
        let committed = committedText.isEmpty ? "" : committedText
        let partial = partialText.isEmpty ? "" : partialText
        if committed.isEmpty { return partial }
        if partial.isEmpty { return committed }
        return committed + " " + partial
    }

    // MARK: - Model Loading

    func loadModel() async {
        guard model == nil else { return }
        isLoading = true
        errorMessage = nil
        loadingStatus = "Downloading model..."

        do {
            let loaded = try await Task.detached {
                try await ParakeetStreamingASRModel.fromPretrained { [weak self] progress, status in
                    DispatchQueue.main.async {
                        self?.loadingStatus = status.isEmpty
                            ? "Downloading... \(Int(progress * 100))%"
                            : "\(status) (\(Int(progress * 100))%)"
                    }
                }
            }.value

            loadingStatus = "Warming up..."
            try loaded.warmUp()
            model = loaded
            loadingStatus = ""
        } catch {
            errorMessage = "Failed to load: \(error.localizedDescription)"
            loadingStatus = ""
        }

        isLoading = false
    }

    // MARK: - Recording

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            startRecording()
        }
    }

    func startRecording() {
        guard let model else { return }
        errorMessage = nil
        partialText = ""

        do {
            let newSession = try model.createSession()
            session = newSession

            recorder.start { [weak self] chunk in
                self?.processAudioChunk(chunk)
            }
            isRecording = true
        } catch {
            errorMessage = "Failed to start: \(error.localizedDescription)"
        }
    }

    func stopRecording() {
        recorder.stop()
        isRecording = false

        // Finalize remaining audio
        guard let session else { return }
        do {
            let finals = try session.finalize()
            for partial in finals {
                if partial.isFinal && !partial.text.isEmpty {
                    commitText(partial.text)
                }
            }
        } catch {
            errorMessage = "Finalize failed: \(error.localizedDescription)"
        }
        self.session = nil
        partialText = ""
    }

    // MARK: - Audio Processing

    private func processAudioChunk(_ samples: [Float]) {
        guard let session else { return }
        do {
            let partials = try session.pushAudio(samples)
            for partial in partials {
                DispatchQueue.main.async { [weak self] in
                    if partial.isFinal {
                        self?.commitText(partial.text)
                        self?.partialText = ""
                    } else {
                        self?.partialText = partial.text
                    }
                }
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.errorMessage = "Processing error: \(error.localizedDescription)"
            }
        }
    }

    private func commitText(_ text: String) {
        if committedText.isEmpty {
            committedText = text
        } else {
            committedText += " " + text
        }
    }

    // MARK: - Actions

    func pasteToFrontApp() {
        let text = fullText
        guard !text.isEmpty else { return }

        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)

        // Simulate Cmd+V in the frontmost app
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            let src = CGEventSource(stateID: .hidSystemState)
            let keyDown = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: true)  // V
            keyDown?.flags = .maskCommand
            let keyUp = CGEvent(keyboardEventSource: src, virtualKey: 0x09, keyDown: false)
            keyUp?.flags = .maskCommand
            keyDown?.post(tap: .cghidEventTap)
            keyUp?.post(tap: .cghidEventTap)
        }
    }

    func copyToClipboard() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(fullText, forType: .string)
    }

    func clearText() {
        committedText = ""
        partialText = ""
    }
}
