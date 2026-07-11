import Foundation

/// Static geometry of the SupertonicTTS-3 CoreML graphs (44.1 kHz, non-autoregressive
/// flow-matching). Mirrors `speech-models/stmodels/infer.py` and the C++ `LiteRTSupertonicTts`.
public struct SupertonicConfig: Codable, Sendable {
    /// Vocoder output rate.
    public let sampleRate: Int
    /// Fixed text length of the duration/text-encoder graphs (relpos pad is T-dependent).
    public let textLength: Int
    /// `144 = latent_dim(24) * chunk_compress(6)`.
    public let latentChannels: Int
    /// Samples per latent frame (`base_chunk_size 512 * chunk_compress 6`).
    public let chunkSamples: Int
    /// Exported vector_estimator RangeDim floor — the latent length is clamped up to this.
    public let latentMin: Int

    public init(sampleRate: Int = 44100,
                textLength: Int = 128,
                latentChannels: Int = 144,
                chunkSamples: Int = 512 * 6,
                latentMin: Int = 17) {
        self.sampleRate = sampleRate
        self.textLength = textLength
        self.latentChannels = latentChannels
        self.chunkSamples = chunkSamples
        self.latentMin = latentMin
    }

    public static let `default` = SupertonicConfig()

    var styleTtlCount: Int { 50 * 256 }
    var styleDpCount: Int { 8 * 16 }
}

/// Per-call quality/rate knobs.
public struct SupertonicOptions: Sendable {
    /// Flow-matching ODE steps: 5 (fast) · 8 (default) · 12 (quality).
    public var totalStep: Int
    /// Speech rate; divides the predicted duration.
    public var speed: Float
    /// Latent-noise seed; 0 ⇒ a fresh seed per synthesis.
    public var seed: UInt64

    public init(totalStep: Int = 8, speed: Float = 1.05, seed: UInt64 = 0) {
        self.totalStep = totalStep
        self.speed = speed
        self.seed = seed
    }

    public static let `default` = SupertonicOptions()
}
