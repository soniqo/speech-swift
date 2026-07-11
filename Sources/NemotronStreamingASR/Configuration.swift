import Foundation

/// Configuration for Nemotron-3.5 ASR Streaming 0.6B (Multilingual, 76 languages).
///
/// Cache-aware FastConformer encoder + prompt-conditioned RNN-T decoder.
/// Language is conditioned per-session via a 128-slot one-hot prompt mask;
/// the encoder concatenates this mask with each output frame before projecting
/// to the joint network. Native punctuation + capitalization are emitted as
/// regular BPE tokens. End of stream is signaled via `finalize()`.
public struct NemotronStreamingConfig: Codable, Sendable {
    public let numMelBins: Int
    public let sampleRate: Int
    public let nFFT: Int
    public let hopLength: Int
    public let winLength: Int
    public let preEmphasis: Float
    public let encoderHidden: Int
    public let encoderLayers: Int
    public let subsamplingFactor: Int
    public let attentionContext: Int
    public let convCacheSize: Int
    public let decoderHidden: Int
    public let decoderLayers: Int
    public let vocabSize: Int
    public let blankTokenId: Int
    /// Number of language prompt slots in the language_mask input. 128 for
    /// nemotron-3.5-asr-streaming-0.6b (84 used + 'auto' + reserved/aliases).
    public let numPrompts: Int
    public let streaming: StreamingConfig

    public struct StreamingConfig: Codable, Sendable {
        public let chunkMs: Int
        public let chunkSize: Int
        /// Number of right-context (future) frames the encoder was trained
        /// with. **This is model-topology metadata, not a runtime knob.**
        ///
        /// The CoreML encoder graph already incorporates right-context via
        /// the streaming cache mechanism: at conversion time the export
        /// script (`speech-models/.../convert.py:172`, `keep_all_outputs=False`)
        /// trimmed the right-context outputs, and the encoder pulls future
        /// context from `cache_last_channel`/`cache_last_time` filled by
        /// the *previous* chunk. The Swift streaming session correctly
        /// feeds exactly `chunk_size` mel frames per call with no audio
        /// overlap — that's what the trained graph expects.
        ///
        /// DO NOT add audio overlap at the Swift layer (e.g. shrinking
        /// `shiftSamples` below `samplesPerChunk` to mirror Parakeet). The
        /// RNN-T predictor's LSTM state advances on every non-blank
        /// emission and would be permanently desynced by re-feeding
        /// overlapped frames. The streaming-vs-batch recall measured on
        /// `E2ENemotronHarshAudioTests.testStreamingMatchesBatchOnCleanLongUtterance`
        /// is the chunker's correctness signal — a regression there means
        /// the cache I/O wiring broke, not that the chunker needs overlap.
        public let rightContext: Int
        public let melFrames: Int
        public let preCacheSize: Int
        public let outputFrames: Int
    }

    enum CodingKeys: String, CodingKey {
        case numMelBins, sampleRate, nFFT, hopLength, winLength, preEmphasis
        case encoderHidden, encoderLayers, subsamplingFactor, attentionContext
        case convCacheSize, decoderHidden, decoderLayers, vocabSize, blankTokenId
        case numPrompts, streaming
    }

    /// Aliases used by other bundle generators (e.g. the multilingual export
    /// pipeline writes `attentionLeftContext`). Read on decode only — does
    /// not participate in encode.
    private struct AnyKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
        init(_ stringValue: String) { self.stringValue = stringValue }
    }

    public init(
        numMelBins: Int, sampleRate: Int, nFFT: Int, hopLength: Int, winLength: Int,
        preEmphasis: Float, encoderHidden: Int, encoderLayers: Int, subsamplingFactor: Int,
        attentionContext: Int, convCacheSize: Int, decoderHidden: Int, decoderLayers: Int,
        vocabSize: Int, blankTokenId: Int, numPrompts: Int = 128, streaming: StreamingConfig
    ) {
        self.numMelBins = numMelBins
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.preEmphasis = preEmphasis
        self.encoderHidden = encoderHidden
        self.encoderLayers = encoderLayers
        self.subsamplingFactor = subsamplingFactor
        self.attentionContext = attentionContext
        self.convCacheSize = convCacheSize
        self.decoderHidden = decoderHidden
        self.decoderLayers = decoderLayers
        self.vocabSize = vocabSize
        self.blankTokenId = blankTokenId
        self.numPrompts = numPrompts
        self.streaming = streaming
    }

    // Older bundle configs may omit `numPrompts`; default to 128.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try c.decode(Int.self, forKey: .numMelBins)
        sampleRate = try c.decode(Int.self, forKey: .sampleRate)
        nFFT = try c.decode(Int.self, forKey: .nFFT)
        hopLength = try c.decode(Int.self, forKey: .hopLength)
        winLength = try c.decode(Int.self, forKey: .winLength)
        preEmphasis = try c.decode(Float.self, forKey: .preEmphasis)
        encoderHidden = try c.decode(Int.self, forKey: .encoderHidden)
        encoderLayers = try c.decode(Int.self, forKey: .encoderLayers)
        subsamplingFactor = try c.decode(Int.self, forKey: .subsamplingFactor)
        // The multilingual bundle ships as `attentionLeftContext` (more
        // accurate name); older English bundles use `attentionContext`.
        if let v = try c.decodeIfPresent(Int.self, forKey: .attentionContext) {
            attentionContext = v
        } else {
            let alt = try decoder.container(keyedBy: AnyKey.self)
            attentionContext = try alt.decode(Int.self, forKey: AnyKey("attentionLeftContext"))
        }
        convCacheSize = try c.decode(Int.self, forKey: .convCacheSize)
        decoderHidden = try c.decode(Int.self, forKey: .decoderHidden)
        decoderLayers = try c.decode(Int.self, forKey: .decoderLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        blankTokenId = try c.decode(Int.self, forKey: .blankTokenId)
        numPrompts = try c.decodeIfPresent(Int.self, forKey: .numPrompts) ?? 128
        streaming = try c.decode(StreamingConfig.self, forKey: .streaming)
    }

    /// Default config matching the 320 ms multilingual bundle at
    /// `aufklarer/Nemotron-3.5-ASR-Streaming-0.6B-CoreML-INT8`.
    public static let `default` = NemotronStreamingConfig(
        numMelBins: 128,
        sampleRate: 16000,
        nFFT: 512,
        hopLength: 160,
        winLength: 400,
        preEmphasis: 0.97,
        encoderHidden: 1024,
        encoderLayers: 24,
        subsamplingFactor: 8,
        attentionContext: 56,           // multilingual att_context_size[0]
        convCacheSize: 8,               // conv_kernel_size - 1
        decoderHidden: 640,
        decoderLayers: 2,
        vocabSize: 13087,               // multilingual BPE + lang tags
        blankTokenId: 13087,            // = vocabSize (RNN-T blank)
        numPrompts: 128,
        streaming: StreamingConfig(
            chunkMs: 320,
            chunkSize: 4,
            rightContext: 3,
            melFrames: 32,              // chunk_size × subsampling = 4 × 8
            preCacheSize: 9,            // ≥320 ms uses pre_encode_cache_size = 9
            outputFrames: 4
        )
    )
}
