import Foundation

/// Model configuration for pyannote PyanNet segmentation.
public struct SegmentationConfig: Sendable {
    /// Audio sample rate in Hz
    public let sampleRate: Int

    /// SincNet filter counts per layer
    public let sincnetFilters: [Int]
    /// SincNet kernel sizes per layer
    public let sincnetKernelSizes: [Int]
    /// SincNet strides per layer
    public let sincnetStrides: [Int]
    /// SincNet max-pool kernel sizes per layer
    public let sincnetPoolSizes: [Int]

    /// LSTM hidden size (per direction; output is 2x for bidirectional)
    public let lstmHiddenSize: Int
    /// Number of LSTM layers
    public let lstmNumLayers: Int

    /// Linear layer hidden size
    public let linearHiddenSize: Int
    /// Number of linear layers (before classifier)
    public let linearNumLayers: Int

    /// Number of output classes (powerset)
    public let numClasses: Int

    /// Default configuration for pyannote/segmentation-3.0
    public static let `default` = SegmentationConfig(
        sampleRate: 16000,
        sincnetFilters: [80, 60, 60],
        sincnetKernelSizes: [251, 5, 5],
        sincnetStrides: [10, 1, 1],
        sincnetPoolSizes: [3, 3, 3],
        lstmHiddenSize: 128,
        lstmNumLayers: 4,
        linearHiddenSize: 128,
        linearNumLayers: 2,
        numClasses: 7
    )
}

/// VAD pipeline configuration with default hysteresis thresholds.
public struct VADConfig: Sendable {
    /// Onset threshold (speech starts when probability exceeds this)
    public var onset: Float
    /// Offset threshold (speech ends when probability drops below this)
    public var offset: Float
    /// Minimum speech duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence duration in seconds
    public var minSilenceDuration: Float
    /// Analysis window duration in seconds
    public var windowDuration: Float
    /// Step ratio for sliding window (fraction of window)
    public var stepRatio: Float

    public init(
        onset: Float, offset: Float,
        minSpeechDuration: Float, minSilenceDuration: Float,
        windowDuration: Float, stepRatio: Float
    ) {
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.windowDuration = windowDuration
        self.stepRatio = stepRatio
    }

    /// Default pyannote VAD thresholds
    public static let `default` = VADConfig(
        onset: 0.767,
        offset: 0.377,
        minSpeechDuration: 0.136,
        minSilenceDuration: 0.067,
        windowDuration: 10.0,
        stepRatio: 0.1
    )

    /// Default Silero VAD thresholds (streaming-optimized)
    public static let sileroDefault = VADConfig(
        onset: 0.5,
        offset: 0.35,
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.1,
        windowDuration: 0.032,
        stepRatio: 1.0
    )
}

/// Configuration for the DSP energy pre-filter (Stage 1 before Silero).
public struct EnergyPreFilterConfig: Sendable {
    /// EMA decay factor for adaptive noise floor estimation.
    public var noiseAlpha: Float

    /// Number of initial chunks that always invoke Silero (~320ms at 10 chunks).
    public var warmupChunks: Int

    /// Energy must exceed noise floor by this many dB to invoke Silero.
    public var marginDB: Float

    /// After this many consecutive skips (~1.6s at 50), reset LSTM state.
    public var maxConsecutiveSkips: Int

    /// Number of look-back chunks to replay on skip→invoke transition.
    public var lookBackChunks: Int

    public init(
        noiseAlpha: Float = 0.01,
        warmupChunks: Int = 10,
        marginDB: Float = 10.0,
        maxConsecutiveSkips: Int = 50,
        lookBackChunks: Int = 2
    ) {
        self.noiseAlpha = noiseAlpha
        self.warmupChunks = warmupChunks
        self.marginDB = marginDB
        self.maxConsecutiveSkips = maxConsecutiveSkips
        self.lookBackChunks = lookBackChunks
    }

    /// Default configuration.
    public static let `default` = EnergyPreFilterConfig()
}
