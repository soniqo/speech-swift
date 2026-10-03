import Foundation

/// Keeps speaker playback out of the VAD stream while preserving audio timing.
struct EchoMicrophoneGate {
    static func samplesToPush(_ samples: [Float], muted: Bool) -> [Float] {
        guard muted else { return samples }
        return [Float](repeating: 0, count: samples.count)
    }
}
