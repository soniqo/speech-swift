import Foundation

/// Stable Audio 3 model variant. Pairs each DiT family (Medium 1.4 B,
/// Small-Music 50 M, Small-SFX 50 M) with its INT4 or INT8 quantization.
/// All bundles ship the matching SAME codec (SAME-L for Medium, SAME-S for
/// the Small variants) and T5Gemma text encoder alongside the DiT.
public enum StableAudio3Variant: String, Sendable, CaseIterable {
    case mediumInt8 = "medium-int8"
    case mediumInt4 = "medium-int4"
    case smallMusicInt8 = "small-music-int8"
    case smallMusicInt4 = "small-music-int4"
    case smallSFXInt8 = "small-sfx-int8"
    case smallSFXInt4 = "small-sfx-int4"

    public var huggingFaceRepoId: String {
        switch self {
        case .mediumInt8:      return "aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit"
        case .mediumInt4:      return "aufklarer/Stable-Audio-3-DiT-Medium-MLX-4bit"
        case .smallMusicInt8:  return "aufklarer/Stable-Audio-3-DiT-Small-Music-MLX-8bit"
        case .smallMusicInt4:  return "aufklarer/Stable-Audio-3-DiT-Small-Music-MLX-4bit"
        case .smallSFXInt8:    return "aufklarer/Stable-Audio-3-DiT-Small-SFX-MLX-8bit"
        case .smallSFXInt4:    return "aufklarer/Stable-Audio-3-DiT-Small-SFX-MLX-4bit"
        }
    }

    /// Bits for the DiT quantization. SAME codec stays FP32 in every bundle
    /// (its differential attention cancels in FP16); T5Gemma stays FP16.
    public var bits: Int {
        switch self {
        case .mediumInt8, .smallMusicInt8, .smallSFXInt8: return 8
        case .mediumInt4, .smallMusicInt4, .smallSFXInt4: return 4
        }
    }

    /// Which DiT family this variant belongs to. Drives module choice
    /// (medium ⇒ DiT-Medium + SAME-L, small-* ⇒ DiT-Small + SAME-S).
    public var family: StableAudio3Family {
        switch self {
        case .mediumInt8, .mediumInt4:           return .medium
        case .smallMusicInt8, .smallMusicInt4:   return .smallMusic
        case .smallSFXInt8, .smallSFXInt4:       return .smallSFX
        }
    }
}

public enum StableAudio3Family: Sendable {
    case medium       // DiT-Medium (24 layers, 1536 dim, differential attn) + SAME-L
    case smallMusic   // DiT-Small  (20 layers, 1024 dim, standard MHA)      + SAME-S
    case smallSFX     // same shape as smallMusic but trained on SFX
}

/// SA3 DiT-Medium architecture constants (mirror dit_mlx_medium.py).
public enum DiTMediumDims {
    public static let ioChannels = 256
    public static let embedDim = 1536
    public static let depth = 24
    public static let numHeads = 24
    public static let headDim = 64
    public static let ropeDims = 32
    public static let condTokenDim = 768
    public static let globalCondDim = 768
    public static let localAddCondDim = 257
    public static let numMemoryTokens = 64
    public static let ffInner = 6144
    public static let timestepFeatDim = 256
    public static let normEps: Float = 1e-5
    public static let qkNormEps: Float = 1e-6
    public static let ropeBase: Float = 10000.0
}

/// SA3 DiT-Small architecture constants (mirror dit_mlx.py).
public enum DiTSmallDims {
    public static let ioChannels = 256
    public static let embedDim = 1024
    public static let depth = 20
    public static let numHeads = 16
    public static let headDim = 64
    public static let ropeDims = 32
    public static let condTokenDim = 768
    public static let globalCondDim = 768
    public static let localAddCondDim = 257
    public static let numMemoryTokens = 64
    public static let ffInner = 4096
    public static let timestepFeatDim = 256
    public static let normEps: Float = 1e-5
    public static let qkNormEps: Float = 1e-6
    public static let ropeBase: Float = 10000.0
}

/// SAME-L decoder constants (mirror same_l_decoder.py).
public enum SAMELDims {
    public static let latentDim = 256
    public static let dim = 1536
    public static let numHeads = 24
    public static let headDim = 64
    public static let ropeDims = 32
    public static let numBlocks = 12
    public static let ffInner = 4608
    public static let sinStartBlock = 5
    public static let outChannels = 512
    public static let stride = 16
    public static let subChunkSize = stride + 1     // 17
    public static let sinPerPos = subChunkSize - 1  // 16
    public static let ropeBase: Float = 10000.0
}

/// SAME-S decoder constants (mirror same_s_decoder.py).
public enum SAMESDims {
    public static let latentDim = 256
    public static let dim = 768
    public static let numHeads = 12
    public static let headDim = 64
    public static let ropeDims = 32
    public static let numBlocks = 6
    public static let ffInner = 2304
    public static let sinStartBlock = 5
    public static let outChannels = 512
    public static let stride = 16
    public static let subChunkSize = stride + 1
    public static let sinPerPos = subChunkSize - 1
    public static let ropeBase: Float = 10000.0
}

/// T5Gemma encoder dims (small Gemma2-style — 12 layers / 768 / 12 heads).
public enum T5GemmaDims {
    public static let hiddenSize = 768
    public static let numLayers = 12
    public static let numAttentionHeads = 12
    public static let numKeyValueHeads = 12
    public static let headDim = 64
    public static let intermediateSize = 2048
    public static let vocabSize = 256000
    public static let maxPositionEmbeddings = 8192
    public static let ropeTheta: Float = 10000.0
    public static let rmsNormEps: Float = 1e-6
    public static let attnLogitSoftcapping: Float = 50.0
    public static let queryPreAttnScalar: Int = 64
    public static let padTokenId: Int32 = 0
}

/// Audio pipeline constants.
public enum SA3Audio {
    public static let sampleRate = 44100
    public static let samplesPerLatent = 4096
    public static let channels = 2          // stereo
    public static let patchSize = 256       // PatchedPretransform patch
}

/// Per-bundle component sub-directory names (must match the publisher layout
/// in `speech-models/models/stable-audio-3/export/scripts/quantize.py`).
public enum SA3Components {
    public static let ditMedium = "dit_medium"
    public static let ditSmallMusic = "dit_sm_music"
    public static let ditSmallSFX = "dit_sm_sfx"
    public static let sameLEncoder = "same_l_encoder"
    public static let sameLDecoder = "same_l_decoder"
    public static let sameSEncoder = "same_s_encoder"
    public static let sameSDecoder = "same_s_decoder"
    public static let t5gemma = "t5gemma"

    /// Component sub-directory holding the DiT for a given variant.
    public static func dit(for variant: StableAudio3Variant) -> String {
        switch variant.family {
        case .medium:     return ditMedium
        case .smallMusic: return ditSmallMusic
        case .smallSFX:   return ditSmallSFX
        }
    }

    public static func sameEncoder(for variant: StableAudio3Variant) -> String {
        variant.family == .medium ? sameLEncoder : sameSEncoder
    }

    public static func sameDecoder(for variant: StableAudio3Variant) -> String {
        variant.family == .medium ? sameLDecoder : sameSDecoder
    }
}
