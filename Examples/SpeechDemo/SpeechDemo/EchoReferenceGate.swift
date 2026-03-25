import Foundation

/// Energy-gated echo suppression using TTS output as reference.
///
/// Instead of full adaptive echo cancellation, this uses a simpler approach:
/// we know exactly what audio we're sending to the speaker (TTS chunks).
/// If the mic energy is close to the expected echo energy, it's likely
/// speaker output leaking into the mic — not real user speech.
///
/// This works because:
/// - We know the reference signal (TTS output)
/// - Real user speech is significantly louder than the attenuated echo
/// - Apple's VP already partially cancels the echo, so what leaks through
///   is typically 20-40dB below the original TTS level
///
/// The gate runs on the mic tap thread — must be fast and lock-free.
final class EchoReferenceGate: @unchecked Sendable {

    /// Rolling RMS of recent TTS output (at 16kHz, matching mic rate)
    private var referenceRMS: Float = 0

    /// True while TTS is actively generating (between responseCreated and responseDone)
    private var isPlaying: Bool = false

    /// Decay factor for reference RMS (exponential moving average)
    /// At 16kHz with ~100ms mic buffers (~1600 samples), this decays
    /// the reference over ~300ms after TTS stops.
    private let decayFactor: Float = 0.7

    /// Gate threshold: mic is considered echo if micRMS < referenceRMS * multiplier.
    /// VP already attenuates echo by ~20dB, so the leaked echo is roughly
    /// 10x quieter than the original. A multiplier of 1.5 means we gate
    /// only when mic energy is clearly below the echo floor.
    /// Conservative to avoid blocking real user speech.
    private let echoMultiplier: Float = 1.5

    /// Minimum reference RMS to activate gating. Below this, the TTS
    /// output is too quiet to cause meaningful echo.
    private let minReferenceRMS: Float = 0.005

    /// Feed TTS audio samples as echo reference.
    /// Called from main thread on responseAudioDelta events.
    func feedReference(_ samples: [Float], sampleRate: Int) {
        guard !samples.isEmpty else { return }

        // Compute RMS of the TTS chunk
        var sumSq: Float = 0
        for s in samples { sumSq += s * s }
        let chunkRMS = sqrt(sumSq / Float(samples.count))

        // Scale by expected attenuation through speaker → air → mic path
        // VP + physical attenuation typically reduces signal by 20-30dB (~10-30x)
        let estimatedEchoRMS = chunkRMS * 0.05  // ~26dB attenuation estimate

        // Update rolling reference (EMA)
        referenceRMS = max(referenceRMS * decayFactor, estimatedEchoRMS)
        isPlaying = true
    }

    /// Current estimated echo floor RMS. Used by mic buffer to detect
    /// user speech above echo during TTS playback.
    var expectedEchoRMS: Float { referenceRMS }

    /// Check if mic signal is likely echo from our own TTS playback.
    /// Called from mic tap thread — must be fast.
    func isLikelyEcho(micRMS: Float) -> Bool {
        guard isPlaying, referenceRMS > minReferenceRMS else { return false }

        // Mic energy is below the echo threshold — likely our own playback
        return micRMS < referenceRMS * echoMultiplier
    }

    /// Signal that TTS generation is complete. Clear gate immediately —
    /// user naturally speaks right after hearing the echo response.
    func markPlaybackDone() {
        isPlaying = false
        referenceRMS = 0
    }

    /// Reset for a new generation cycle.
    func reset() {
        referenceRMS = 0
        isPlaying = false
    }
}
