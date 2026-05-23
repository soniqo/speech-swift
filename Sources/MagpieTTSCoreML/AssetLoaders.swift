import Foundation

/// Loads the per-bundle constant tensors from `constants/*.npy`:
/// - 5 speaker contexts (110 × 768 each)
/// - 8 audio-codebook embedding tables (2024 × 768 each)
///
/// All tensors are returned as flat Float32 buffers in row-major layout. The
/// caller is expected to know the shapes.
public enum MagpieCoreMLAssets {

    /// Returns the 5 speaker context tensors, indexed by ``MagpieCoreMLSpeaker``
    /// raw value. Each is a flat `[T * D]` Float32 buffer (110 × 768 = 84480).
    public static func loadSpeakerBank(constantsDir: URL) throws -> [[Float]] {
        var contexts: [[Float]] = []
        contexts.reserveCapacity(MagpieCoreMLConstants.numSpeakers)
        for i in 0..<MagpieCoreMLConstants.numSpeakers {
            let url = constantsDir.appendingPathComponent("speaker_\(i).npy")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieCoreMLError.missingFile("constants/speaker_\(i).npy")
            }
            let arr = try NpyReader.read(from: url)
            try arr.assertShape(
                [MagpieCoreMLConstants.speakerContextLength,
                 MagpieCoreMLConstants.dModel],
                label: "speaker_\(i).npy")
            contexts.append(arr.data)
        }
        return contexts
    }

    /// Returns the 8 audio embedding tables, one per codebook. Each is a flat
    /// `[V * D]` Float32 buffer (2024 × 768 = 1554432). Used by the
    /// LocalTransformer sampler to embed the previous-codebook token before
    /// projecting to the next codebook's logits.
    public static func loadAudioEmbeddings(constantsDir: URL) throws -> [[Float]] {
        var embeds: [[Float]] = []
        embeds.reserveCapacity(MagpieCoreMLConstants.numCodebooks)
        for k in 0..<MagpieCoreMLConstants.numCodebooks {
            let url = constantsDir.appendingPathComponent("audio_embedding_\(k).npy")
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieCoreMLError.missingFile("constants/audio_embedding_\(k).npy")
            }
            let arr = try NpyReader.read(from: url)
            try arr.assertShape(
                [MagpieCoreMLConstants.numCodesPerCodebook,
                 MagpieCoreMLConstants.dModel],
                label: "audio_embedding_\(k).npy")
            embeds.append(arr.data)
        }
        return embeds
    }

    /// Decode `constants/constants.json`. Caller should sanity-check key fields
    /// against ``MagpieCoreMLConstants`` (we treat the hard-coded values as
    /// authoritative — the JSON is for diagnostic / build-stamp purposes).
    public static func loadBundleConstants(
        constantsDir: URL
    ) throws -> MagpieCoreMLBundleConstants {
        let url = constantsDir.appendingPathComponent("constants.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieCoreMLError.missingFile("constants/constants.json")
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MagpieCoreMLBundleConstants.self, from: data)
    }
}

/// Weights for the 1-layer Local Transformer (d=256, FFN=1024, 8 codebook
/// heads). Shapes match the FluidInference NPY layout. All Float32 row-major.
public struct MagpieCoreMLLocalTransformerWeights: Sendable {
    public let inProjWeight: [Float]    // (localDim, dModel) = (256, 768)
    public let inProjBias: [Float]      // (localDim,) = (256,)
    public let posEmbedding: [Float]    // (maxPositions, localDim) = (10, 256)
    public let norm1Weight: [Float]     // (localDim,)
    public let norm2Weight: [Float]     // (localDim,)
    public let saQkvWeight: [Float]     // (3*localDim, localDim) = (768, 256)
    public let saOWeight: [Float]       // (localDim, localDim) = (256, 256)
    public let ffnConv1Weight: [Float]  // (ffnDim, localDim) = (1024, 256)
    public let ffnConv2Weight: [Float]  // (localDim, ffnDim) = (256, 1024)
    public let outProjWeights: [[Float]]  // 8 × (numCodes, localDim) = 8 × (2024, 256)
    public let outProjBiases: [[Float]]   // 8 × (numCodes,)

    public let localDim: Int
    public let dModel: Int
    public let ffnDim: Int
    public let maxPositions: Int
    public let numCodebooks: Int
    public let numCodesPerCodebook: Int
}

public enum MagpieCoreMLLocalTransformerLoader {

    public static func load(
        from ltDir: URL
    ) throws -> MagpieCoreMLLocalTransformerWeights {
        let localDim = MagpieCoreMLConstants.localTransformerDim
        let dModel = MagpieCoreMLConstants.dModel
        let ffnDim = MagpieCoreMLConstants.localTransformerFfnDim
        let maxPositions = MagpieCoreMLConstants.localTransformerMaxPositions
        let numCodebooks = MagpieCoreMLConstants.numCodebooks
        let numCodes = MagpieCoreMLConstants.numCodesPerCodebook

        guard FileManager.default.fileExists(atPath: ltDir.path) else {
            throw MagpieCoreMLError.missingFile("constants/local_transformer/")
        }

        func loadNpy(_ name: String, expecting shape: [Int]) throws -> [Float] {
            let url = ltDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieCoreMLError.missingFile("constants/local_transformer/\(name)")
            }
            let arr = try NpyReader.read(from: url)
            try arr.assertShape(shape, label: "local_transformer/\(name)")
            return arr.data
        }

        // Conv1d kernel=1 may ship as (out, in) plain matmul OR (out, in, 1)
        // with the trailing kernel dim preserved by the exporter.
        func loadFlex(_ name: String, primary: [Int], alt: [Int]) throws -> [Float] {
            let url = ltDir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw MagpieCoreMLError.missingFile("constants/local_transformer/\(name)")
            }
            let arr = try NpyReader.read(from: url)
            if arr.shape == primary || arr.shape == alt {
                return arr.data
            }
            throw MagpieCoreMLError.invalidNpyFile(
                path: name,
                reason: "expected \(primary) or \(alt), got \(arr.shape)")
        }

        let inW = try loadNpy("in_proj_weight.npy", expecting: [localDim, dModel])
        let inB = try loadNpy("in_proj_bias.npy",   expecting: [localDim])
        let posE = try loadNpy("pos_emb.npy",       expecting: [maxPositions, localDim])
        let n1 = try loadNpy("norm1_weight.npy",    expecting: [localDim])
        let n2 = try loadNpy("norm2_weight.npy",    expecting: [localDim])
        let qkv = try loadNpy("sa_qkv_weight.npy",  expecting: [3 * localDim, localDim])
        let saO = try loadNpy("sa_o_weight.npy",    expecting: [localDim, localDim])
        let f1 = try loadFlex("ffn_conv1_weight.npy",
                              primary: [ffnDim, localDim],
                              alt:     [ffnDim, localDim, 1])
        let f2 = try loadFlex("ffn_conv2_weight.npy",
                              primary: [localDim, ffnDim],
                              alt:     [localDim, ffnDim, 1])

        var outW: [[Float]] = []
        var outB: [[Float]] = []
        outW.reserveCapacity(numCodebooks)
        outB.reserveCapacity(numCodebooks)
        for cb in 0..<numCodebooks {
            outW.append(try loadNpy("out_proj_\(cb)_weight.npy",
                                     expecting: [numCodes, localDim]))
            outB.append(try loadNpy("out_proj_\(cb)_bias.npy",
                                     expecting: [numCodes]))
        }

        return MagpieCoreMLLocalTransformerWeights(
            inProjWeight: inW, inProjBias: inB,
            posEmbedding: posE,
            norm1Weight: n1, norm2Weight: n2,
            saQkvWeight: qkv, saOWeight: saO,
            ffnConv1Weight: f1, ffnConv2Weight: f2,
            outProjWeights: outW, outProjBiases: outB,
            localDim: localDim, dModel: dModel, ffnDim: ffnDim,
            maxPositions: maxPositions,
            numCodebooks: numCodebooks,
            numCodesPerCodebook: numCodes)
    }
}
