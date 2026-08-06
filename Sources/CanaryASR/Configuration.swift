import Foundation

/// Configuration for NVIDIA Canary (NeMo Conformer-AED) ASR.
///
/// Decoded from the `config.json` that ships in the model bundle. The decode
/// contract — prompt token ids, decoder cache dimensions, end-of-text — is read
/// from the bundle rather than resolved by token string: this vocabulary has no
/// bare-space token (word boundaries are U+2581) and the aggregate tokenizer
/// repeats ordinary pieces once per sub-tokenizer, so a string lookup can
/// silently return the wrong id or none at all.
public struct CanaryConfig: Codable, Sendable {
    /// Number of mel-spectrogram frequency bins.
    public let numMelBins: Int
    /// Expected input audio sample rate in Hz.
    public let sampleRate: Int
    /// FFT window size.
    public let nFFT: Int
    /// Hop length between successive STFT frames.
    public let hopLength: Int
    /// Window length for STFT.
    public let winLength: Int
    /// Pre-emphasis coefficient applied to the raw waveform.
    public let preEmphasis: Float
    /// Width of `encoder_embeddings` as the graph emits it, after the
    /// projection into decoder width.
    public let encoderHidden: Int
    /// Hidden width of the Transformer decoder.
    public let decoderHidden: Int
    /// Leading dimension of `decoder_mems` — one entry per cached layer.
    public let decoderMemLayers: Int
    /// Vocabulary size.
    public let vocabSize: Int
    /// Whether the head emits log probabilities rather than raw logits.
    public let logitsAreLogProbs: Bool
    /// Decode prompts by language pair, e.g. `"en-en"`.
    public let promptIds: [String: [Int]]
    /// Named special tokens: `bos`, `eos`, `pad`, `nospeech`, …
    public let specialTokenIds: [String: Int]
    /// Language code to prompt token id, e.g. `"de"` → 76.
    public let languageTokenIds: [String: Int]
    /// Languages the checkpoint was trained on.
    public let languages: [String]
    /// Core ML-specific shapes; absent in the ONNX bundle.
    public let coreml: CoreMLShapes?

    public struct CoreMLShapes: Codable, Sendable {
        /// Fixed mel frames the encoder accepts. Shorter audio is zero-padded
        /// and the true frame count passed as `length`, which is what drives
        /// masking, so padding does not change the result.
        public let encoderMelFrames: Int
        /// Encoder output frames, after the conformer's 8× subsampling.
        public let encodedFrames: Int
        /// Number of tokens in the decode prompt.
        public let promptTokens: Int
    }

    /// End-of-text id. Decoding stops when the argmax reaches it.
    public var endOfTextId: Int { specialTokenIds["eos"] ?? 3 }

    /// The decode prompt for a language pair, with the source and target
    /// language tokens patched in. Returns nil when the bundle has no token
    /// for one of the codes.
    public func prompt(source: String, target: String? = nil) -> [Int]? {
        guard var ids = promptIds["en-en"] ?? promptIds.values.first else { return nil }
        guard let sourceId = languageTokenIds[source] else { return nil }
        let targetId = languageTokenIds[target ?? source]
        guard let targetId else { return nil }

        // The prompt carries exactly one source/target pair — find it by token
        // rather than by position, so a bundle whose template differs still
        // switches language correctly.
        let languageIds = Set(languageTokenIds.values)
        guard let first = ids.indices.dropLast().first(where: {
            languageIds.contains(ids[$0]) && languageIds.contains(ids[$0 + 1])
        }) else { return nil }

        ids[first] = sourceId
        ids[first + 1] = targetId
        return ids
    }

    public static let `default` = CanaryConfig(
        numMelBins: 128,
        sampleRate: 16000,
        nFFT: 512,
        hopLength: 160,
        winLength: 400,
        preEmphasis: 0.97,
        encoderHidden: 1024,
        decoderHidden: 1024,
        decoderMemLayers: 6,
        vocabSize: 5248,
        logitsAreLogProbs: true,
        promptIds: ["en-en": [7, 4, 16, 62, 62, 5, 9, 11, 13]],
        specialTokenIds: ["bos": 4, "eos": 3, "pad": 2, "nospeech": 1],
        languageTokenIds: ["en": 62, "de": 76, "es": 169, "fr": 69],
        languages: ["en", "de", "es", "fr"],
        coreml: CoreMLShapes(encoderMelFrames: 3000, encodedFrames: 375, promptTokens: 9)
    )

    public init(
        numMelBins: Int = 128,
        sampleRate: Int = 16000,
        nFFT: Int = 512,
        hopLength: Int = 160,
        winLength: Int = 400,
        preEmphasis: Float = 0.97,
        encoderHidden: Int = 1024,
        decoderHidden: Int = 1024,
        decoderMemLayers: Int = 6,
        vocabSize: Int = 5248,
        logitsAreLogProbs: Bool = true,
        promptIds: [String: [Int]] = [:],
        specialTokenIds: [String: Int] = [:],
        languageTokenIds: [String: Int] = [:],
        languages: [String] = [],
        coreml: CoreMLShapes? = nil
    ) {
        self.numMelBins = numMelBins
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        self.preEmphasis = preEmphasis
        self.encoderHidden = encoderHidden
        self.decoderHidden = decoderHidden
        self.decoderMemLayers = decoderMemLayers
        self.vocabSize = vocabSize
        self.logitsAreLogProbs = logitsAreLogProbs
        self.promptIds = promptIds
        self.specialTokenIds = specialTokenIds
        self.languageTokenIds = languageTokenIds
        self.languages = languages
        self.coreml = coreml
    }
}
