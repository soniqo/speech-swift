import Foundation
import MLX
import MLXCommon
import MLXNN

public enum MioCodecFSQ {
    public static let levels: [Int32] = [8, 8, 8, 5, 5]
    public static let basis: [Int32] = [1, 8, 64, 512, 2_560]
    public static let codebookSize = 12_800

    /// MioCodec FSQ inverse: `[B, T]` content-token ids -> `[B, T, 5]`.
    public static func decode(_ indices: MLXArray) -> MLXArray {
        let basisArray = MLXArray(basis)
        let levelsArray = MLXArray(levels)
        let expanded = MLX.floorDivide(indices.expandedDimensions(axis: -1), basisArray)
        let nonCentered = expanded % levelsArray
        let halfWidth = MLX.floorDivide(levelsArray, MLXArray(Int32(2))).asType(.float32)
        return (nonCentered.asType(.float32) - halfWidth) / halfWidth
    }
}

public final class MioCodecContentDecoder: Module {
    public let config: MioCodecConfig

    @ModuleInfo(key: "proj_out") var projection: Linear

    public init(config: MioCodecConfig = .default) {
        self.config = config
        self._projection.wrappedValue = Linear(
            MioCodecFSQ.levels.count,
            config.contentEmbeddingDim,
            bias: true)
        super.init()
    }

    /// Decode one content-code stream into MioCodec content embeddings.
    ///
    /// - Parameter tokenIds: `[B, T]` Int32 content codes in `0..<12800`.
    /// - Returns: `[B, T, 768]` content embeddings after the FSQ output projection.
    public func decodeContentEmbeddings(_ tokenIds: MLXArray) -> MLXArray {
        projection(MioCodecFSQ.decode(tokenIds))
    }

    public func loadWeights(from codecDirectory: URL) throws {
        let weights = try CommonWeightLoader.loadAllSafetensors(from: codecDirectory)
        loadWeights(from: weights)
        eval(self)
    }

    public func loadWeights(from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(
            to: projection,
            prefix: "local_quantizer.proj_out",
            from: weights)
    }
}

public struct MioCodecDecodePlan: Sendable, Equatable {
    public let tokenCount: Int
    public let sampleRate: Int
    public let estimatedSamples: Int
    public let stftFrames: Int

    public init(
        tokenCount: Int,
        targetAudioLength: Int? = nil,
        config: MioCodecConfig = .default
    ) {
        self.tokenCount = tokenCount
        self.sampleRate = config.sampleRate
        self.estimatedSamples = targetAudioLength ?? tokenCount * config.samplesPerToken
        self.stftFrames = max(1, estimatedSamples / max(1, config.hopLength))
    }
}
