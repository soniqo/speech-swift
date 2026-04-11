import Foundation

/// SpeechUI provides reusable SwiftUI views for building speech-driven apps:
///
/// - ``WaveformView`` — render `[Float]` PCM samples as a waveform
/// - ``MicLevelView`` — horizontal level meter for live mic monitoring
/// - ``TranscriptionView`` — scrolling transcript with finals + in-progress partial
///
/// The views are deliberately decoupled from any specific ASR backend. Feed
/// them plain Swift values you derive from your model's output (Parakeet,
/// Qwen3-ASR, Whisper, anything).
public enum SpeechUI {
    public static let version = "0.1.0"
}
