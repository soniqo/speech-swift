import Foundation

/// Configuration for the Sortformer diarization model.
///
/// Sortformer is NVIDIA's end-to-end neural diarization model that directly
/// predicts speaker activity without requiring separate embedding extraction
/// or clustering stages.
public struct SortformerConfig: Sendable {

    // MARK: - Mel Feature Extraction

    /// Number of mel frequency bins
    public let nMels: Int
    /// FFT window size in samples
    public let nFFT: Int
    /// Hop length in samples
    public let hopLength: Int
    /// Expected input sample rate in Hz
    public let sampleRate: Int

    // MARK: - Streaming Chunking

    /// Chunk length in seconds for streaming inference
    public let chunkLenSeconds: Float
    /// Left context in seconds (prepended from previous chunk)
    public let leftContextSeconds: Float
    /// Right context in seconds (lookahead)
    public let rightContextSeconds: Float
    /// Subsampling factor of the encoder (frames → mel frames)
    public let subsamplingFactor: Int

    // MARK: - State Dimensions

    /// Speaker cache length (number of frames)
    public let spkcacheLen: Int
    /// FIFO buffer length (number of frames)
    public let fifoLen: Int
    /// Feature/hidden dimension of the model
    public let fcDModel: Int

    // MARK: - Model I/O Shapes

    /// Maximum number of speakers the model can predict
    public let maxSpeakers: Int

    // MARK: - Post-processing

    /// Onset threshold for speaker activity binarization
    public var onset: Float
    /// Offset threshold for speaker activity binarization
    public var offset: Float
    /// Minimum speech segment duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence gap to split segments, in seconds
    public var minSilenceDuration: Float

    // MARK: - Variant identity

    /// Identifies which exported `.mlmodelc` to load from the HF repo. Empty
    /// is the legacy filename (`Sortformer.mlmodelc`); other variants get a
    /// suffixed name (`Sortformer_high.mlmodelc`). Lets the same HF repo host
    /// multiple shape-specific exports of the same NeMo checkpoint.
    public var variantName: String

    // MARK: - Streaming State Updater

    /// How often (in encoder frames) the spkcache rotates / gets compressed.
    public var spkcacheUpdatePeriod: Int

    /// Reserved slots per speaker in spkcache filled with the running silence
    /// profile after compression. Models "what does no-one-speaking look like
    /// for this acoustic environment" so the compression keeps a baseline.
    public var spkcacheSilFramesPerSpk: Int

    /// Frames whose summed speaker probability is below this are treated as
    /// silence and folded into `meanSilenceEmbedding`.
    public var silenceThreshold: Float

    /// Lower clamp on probabilities going into the log-score computation —
    /// prevents log(0) blow-up while keeping the relative ordering intact.
    public var predScoreThreshold: Float

    /// Boost added to scores from the most recent (post-spkcacheCapacity)
    /// frames so that newer evidence outweighs equally-confident older
    /// evidence during compression.
    public var scoresBoostLatest: Float

    /// Fraction of per-speaker spkcache slots that get the strong boost
    /// (factor 2.0). Highest-scoring frames per speaker.
    public var strongBoostRate: Float

    /// Fraction of per-speaker spkcache slots that get the weak boost
    /// (factor 1.0). Picks up moderately-confident frames once the
    /// strong-boost set is full.
    public var weakBoostRate: Float

    /// Fraction of per-speaker spkcache slots that must have positive
    /// scores before non-positive ones are disabled. Stops compression
    /// from prematurely throwing away "weak but still positive" frames
    /// when the speaker has very few confident frames.
    public var minPosScoresRate: Float

    /// Sentinel used as a top-k placeholder for disabled / silent slots.
    /// Must be larger than any real frame index in a chunk.
    public var maxIndex: Int

    // MARK: - Derived dimensions

    /// Total mel frames per CoreML call. The model has a fixed input shape
    /// `[1, coreMLInputFrames, nMels]`, so this is the encoder window
    /// expressed in mel frames: `(left + core + right) × subsamplingFactor`.
    /// The fields are still named `*Seconds` for legacy reasons, but their
    /// values are encoder-frame counts.
    public var coreMLInputFrames: Int {
        (Int(leftContextSeconds) + Int(chunkLenSeconds) + Int(rightContextSeconds))
            * subsamplingFactor
    }

    /// Filename of the `.mlmodelc` directory inside the HF repo for this
    /// variant. Default variant keeps the legacy `Sortformer.mlmodelc` so
    /// existing caches keep working without re-download.
    public var coreMLModelFileName: String {
        variantName.isEmpty ? "Sortformer.mlmodelc" : "Sortformer_\(variantName).mlmodelc"
    }

    // MARK: - Presets

    /// Default configuration — high-throughput offline diarization.
    /// `chunk_len=340` encoder frames → ~30 s of audio per CoreML call,
    /// measured ~125–750× RTF on M-series ANE depending on input length.
    /// First-load BNNS codegen on a cold device cache can take minutes
    /// because the compiled ANE graph is large; use `.balanced` if that
    /// matters more than peak RTF.
    public static let `default` = SortformerConfig(
        nMels: 128,
        nFFT: 400,
        hopLength: 160,
        sampleRate: 16000,
        chunkLenSeconds: 340.0,
        leftContextSeconds: 1.0,
        rightContextSeconds: 40.0,
        subsamplingFactor: 8,
        spkcacheLen: 188,
        fifoLen: 40,
        fcDModel: 512,
        maxSpeakers: 4,
        onset: 0.5,
        offset: 0.3,
        minSpeechDuration: 0.3,
        minSilenceDuration: 0.15,
        spkcacheUpdatePeriod: 188,
        variantName: ""
    )

    /// Balanced configuration — same checkpoint, ~3× smaller attention
    /// cost in the compiled graph (188+40+121 vs 188+40+381 tokens).
    /// First-load BNNS codegen is correspondingly faster, while RTF stays
    /// in the hundreds of × on M-series ANE. Use this when first-launch
    /// UX matters more than peak throughput (iOS apps amortizing the
    /// compile cost during onboarding).
    public static let balanced = SortformerConfig(
        nMels: 128,
        nFFT: 400,
        hopLength: 160,
        sampleRate: 16000,
        chunkLenSeconds: 100.0,
        leftContextSeconds: 1.0,
        rightContextSeconds: 20.0,
        subsamplingFactor: 8,
        spkcacheLen: 188,
        fifoLen: 40,
        fcDModel: 512,
        maxSpeakers: 4,
        onset: 0.5,
        offset: 0.3,
        minSpeechDuration: 0.3,
        minSilenceDuration: 0.15,
        spkcacheUpdatePeriod: 100,
        variantName: "balanced"
    )

    /// Streaming / low-latency configuration. `chunk_len=6` encoder frames
    /// (~480 ms of audio per CoreML call) — useful when first output is
    /// needed before 30 s of audio is buffered. RTF is roughly an order of
    /// magnitude lower than `.default`; only pick this if you actually need
    /// the latency.
    public static let streaming = SortformerConfig(
        nMels: 128,
        nFFT: 400,
        hopLength: 160,
        sampleRate: 16000,
        chunkLenSeconds: 6.0,
        leftContextSeconds: 1.0,
        rightContextSeconds: 7.0,
        subsamplingFactor: 8,
        spkcacheLen: 188,
        fifoLen: 40,
        fcDModel: 512,
        maxSpeakers: 4,
        onset: 0.5,
        offset: 0.3,
        minSpeechDuration: 0.3,
        minSilenceDuration: 0.15,
        spkcacheUpdatePeriod: 32,
        variantName: "streaming"
    )

    public init(
        nMels: Int = 128,
        nFFT: Int = 400,
        hopLength: Int = 160,
        sampleRate: Int = 16000,
        chunkLenSeconds: Float = 6.0,
        leftContextSeconds: Float = 1.0,
        rightContextSeconds: Float = 7.0,
        subsamplingFactor: Int = 8,
        spkcacheLen: Int = 188,
        fifoLen: Int = 40,
        fcDModel: Int = 512,
        maxSpeakers: Int = 4,
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        spkcacheUpdatePeriod: Int = 32,
        spkcacheSilFramesPerSpk: Int = 3,
        silenceThreshold: Float = 0.2,
        predScoreThreshold: Float = 0.25,
        scoresBoostLatest: Float = 0.05,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        maxIndex: Int = 99_999,
        variantName: String = ""
    ) {
        self.nMels = nMels
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.sampleRate = sampleRate
        self.chunkLenSeconds = chunkLenSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.subsamplingFactor = subsamplingFactor
        self.spkcacheLen = spkcacheLen
        self.fifoLen = fifoLen
        self.fcDModel = fcDModel
        self.maxSpeakers = maxSpeakers
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        self.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        self.silenceThreshold = silenceThreshold
        self.predScoreThreshold = predScoreThreshold
        self.scoresBoostLatest = scoresBoostLatest
        self.strongBoostRate = strongBoostRate
        self.weakBoostRate = weakBoostRate
        self.minPosScoresRate = minPosScoresRate
        self.maxIndex = maxIndex
        self.variantName = variantName
    }
}
