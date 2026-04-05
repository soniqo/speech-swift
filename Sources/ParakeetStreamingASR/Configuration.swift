import Foundation

/// Configuration for Parakeet EOU 120M streaming ASR model.
public struct ParakeetEOUConfig: Codable, Sendable {
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
    public let eouTokenId: Int
    public let eobTokenId: Int
    public let streaming: StreamingConfig

    public struct StreamingConfig: Codable, Sendable {
        public let chunkMs: Int
        public let melFrames: Int
        public let preCacheSize: Int
        public let outputFrames: Int
    }

    public static let `default` = ParakeetEOUConfig(
        numMelBins: 128,
        sampleRate: 16000,
        nFFT: 512,
        hopLength: 160,
        winLength: 400,
        preEmphasis: 0.97,
        encoderHidden: 512,
        encoderLayers: 17,
        subsamplingFactor: 8,
        attentionContext: 70,
        convCacheSize: 8,
        decoderHidden: 640,
        decoderLayers: 1,
        vocabSize: 1026,
        blankTokenId: 1026,
        eouTokenId: 1024,
        eobTokenId: 1025,
        streaming: StreamingConfig(
            chunkMs: 320,
            melFrames: 33,
            preCacheSize: 9,
            outputFrames: 4
        )
    )
}
