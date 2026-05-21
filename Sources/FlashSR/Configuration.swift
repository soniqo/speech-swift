import Foundation

/// FlashSR model variant. Both bundles use the same architecture; only the
/// on-disk storage quantisation differs (int4 vs int8). At runtime the
/// weights are dequantised to FP — quantisation here is a download-size
/// optimisation only, not a runtime kernel.
public enum FlashSRVariant: String, Sendable, CaseIterable {
    case int4
    case int8

    public var huggingFaceRepoId: String {
        switch self {
        case .int4: return "aufklarer/FlashSR-MLX-4bit"
        case .int8: return "aufklarer/FlashSR-MLX-8bit"
        }
    }

    public var bits: Int {
        switch self {
        case .int4: return 4
        case .int8: return 8
        }
    }
}

// MARK: - Bundle config.json

public struct FlashSRVAEConfig: Codable, Sendable {
    public let inChannels: Int
    public let outCh: Int
    public let ch: Int
    public let chMult: [Int]
    public let numResBlocks: Int
    public let attnResolutions: [Int]
    public let doubleZ: Bool
    public let zChannels: Int
    public let embedDim: Int
    public let dropout: Float
    public let resolution: Int
    public let melBins: Int
    public let scaleFactorZ: Float

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case outCh = "out_ch"
        case ch
        case chMult = "ch_mult"
        case numResBlocks = "num_res_blocks"
        case attnResolutions = "attn_resolutions"
        case doubleZ = "double_z"
        case zChannels = "z_channels"
        case embedDim = "embed_dim"
        case dropout, resolution
        case melBins = "mel_bins"
        case scaleFactorZ = "scale_factor_z"
    }
}

public struct FlashSRUNetConfig: Codable, Sendable {
    public let inChannels: Int
    public let modelChannels: Int
    public let outChannels: Int
    public let numResBlocks: Int

    enum CodingKeys: String, CodingKey {
        case inChannels = "in_channels"
        case modelChannels = "model_channels"
        case outChannels = "out_channels"
        case numResBlocks = "num_res_blocks"
    }

    // Fixed UNet hyperparams (not stored in bundle config; come from the
    // upstream FlashSR config and match the trained checkpoint).
    public var attentionResolutions: [Int] { [8, 4, 2] }
    public var channelMult: [Int] { [1, 2, 3, 5] }
    public var numHeadChannels: Int { 32 }
    public var transformerDepth: Int { 1 }
    public var extraSALayer: Bool { true }
}

public struct FlashSRMelConfig: Codable, Sendable {
    public let nFft: Int
    public let hop: Int
    public let sr: Int
    public let nMels: Int
    public let fmin: Float
    public let fmax: Float

    enum CodingKeys: String, CodingKey {
        case nFft = "n_fft"
        case hop, sr
        case nMels = "n_mels"
        case fmin, fmax
    }
}

public struct FlashSRAudioConfig: Codable, Sendable {
    public let sampleRate: Int
    public let frameSamples: Int
    public let frameSec: Float

    enum CodingKeys: String, CodingKey {
        case sampleRate = "sample_rate"
        case frameSamples = "frame_samples"
        case frameSec = "frame_sec"
    }
}

public struct FlashSRQuantizationConfig: Codable, Sendable {
    public let mode: String
    public let bits: Int
    public let groupSize: Int
    public let rule: String

    enum CodingKeys: String, CodingKey {
        case mode, bits
        case groupSize = "group_size"
        case rule
    }
}

public struct FlashSRConfig: Codable, Sendable {
    public let vae: FlashSRVAEConfig
    public let ldm: FlashSRUNetConfig
    public let mel: FlashSRMelConfig
    public let audio: FlashSRAudioConfig
    public let format: String
    public let quantization: FlashSRQuantizationConfig?
    /// Map of `quantized_key → original-shape` for every flat-quantised weight
    /// in the bundle. Used by the loader to call `mx.dequantize` and then
    /// reshape back to the original conv/linear tensor shape.
    public let quantizedShapes: [String: [Int]]?

    enum CodingKeys: String, CodingKey {
        case vae, ldm, mel, audio, format, quantization
        case quantizedShapes = "quantized_shapes"
    }

    public static func load(from url: URL) throws -> FlashSRConfig {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(FlashSRConfig.self, from: data)
    }
}
