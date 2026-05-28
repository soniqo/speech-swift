import Foundation

/// Configuration for Hybrid Transformer Demucs (Demucs v4), decoded from the
/// `<model>_config.json` produced by the speech-models exporter.
///
/// `htdemucs_ft` is a bag of 4 sub-models (one fine-tuned per source). The bag
/// `weights` are diagonal — model `i` contributes only source `i` — so
/// inference runs each sub-model on the mixture and keeps its own stem.
public struct HTDemucsConfig: Codable, Sendable {
    public let modelName: String
    public let dtype: String          // "fp16" | "fp32"
    public let sources: [String]      // [drums, bass, other, vocals]
    public let numModels: Int
    public let weights: [[Float]]?    // [numModels][numSources] bag combine-weights
    public let samplerate: Int
    public let segment: Double         // seconds per inference window (7.8)
    public let audioChannels: Int
    public let arch: Arch

    public var hopLength: Int { arch.nfft / 4 }
    public var trainingLength: Int { Int(segment * Double(samplerate)) }

    public struct Arch: Codable, Sendable {
        public let channels: Int
        public let growth: Int
        public let nfft: Int
        public let cac: Bool
        public let depth: Int
        public let rewrite: Bool
        public let freqEmb: Float
        public let embScale: Int
        public let embSmooth: Bool
        public let kernelSize: Int
        public let stride: Int
        public let timeStride: Int
        public let context: Int
        public let contextEnc: Int
        public let normStarts: Int
        public let normGroups: Int
        public let dconvMode: Int
        public let dconvDepth: Int
        public let dconvComp: Int
        public let bottomChannels: Int
        // Transformer
        public let tLayers: Int
        public let tHeads: Int
        public let tHiddenScale: Float
        public let tEmb: String
        public let tMaxPeriod: Float
        public let tLayerScale: Bool
        public let tGelu: Bool
        public let tNormIn: Bool
        public let tNormFirst: Bool
        public let tNormOut: Bool
        public let tWeightPosEmbed: Float
        public let tCrossFirst: Bool
    }

    public static func load(from url: URL) throws -> HTDemucsConfig {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(HTDemucsConfig.self, from: Data(contentsOf: url))
    }
}
