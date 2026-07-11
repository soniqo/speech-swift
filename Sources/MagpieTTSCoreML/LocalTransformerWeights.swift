import Foundation
import MLX
import MagpieTTS

/// Flat Float32 weights for the 1-layer LocalTransformer (d=256, FFN=1024,
/// 8 per-codebook output heads). Layouts match the Accelerate forward pass
/// in ``MagpieCoreMLLocalTransformer``.
public struct MagpieCoreMLLocalTransformerWeights: Sendable {
    public let inProjWeight: [Float]      // (256, 768)
    public let inProjBias: [Float]        // (256,)
    public let posEmbedding: [Float]      // (10, 256)
    public let norm1Weight: [Float]       // (256,)
    public let norm2Weight: [Float]       // (256,)
    public let saQkvWeight: [Float]       // (768, 256)
    public let saOWeight: [Float]         // (256, 256)
    public let ffnConv1Weight: [Float]    // (1024, 256)
    public let ffnConv2Weight: [Float]    // (256, 1024)
    public let outProjWeights: [[Float]]  // 8 × (2024, 256)
    public let outProjBiases: [[Float]]   // 8 × (2024,)

    public let localDim: Int
    public let dModel: Int
    public let ffnDim: Int
    public let maxPositions: Int
    public let numCodebooks: Int
    public let numCodesPerCodebook: Int
}

/// Extracts the LocalTransformer weights + the 8 audio embedding tables
/// from the MLX ``MagpieTTS`` instance. We load the full MLX model once
/// at init, snapshot every tensor we need as a flat Float32 buffer, then
/// allow the MLX module to go out of scope. After this initialisation the
/// AR loop runs pure CoreML + Swift — no MLX dispatch overhead per frame.
enum MagpieCoreMLWeightExtractor {

    static func extract(from model: MagpieTTS) throws -> (
        lt: MagpieCoreMLLocalTransformerWeights,
        audioEmbeds: [[Float]]
    ) {
        let lt = try extractLocalTransformer(from: model)
        let embeds = try extractAudioEmbeddings(from: model)
        return (lt, embeds)
    }

    private static func extractLocalTransformer(from model: MagpieTTS) throws
        -> MagpieCoreMLLocalTransformerWeights
    {
        let mlxLT = model.localTransformer
        let localDim = MagpieCoreMLConstants.localTransformerDim
        let dModel = MagpieCoreMLConstants.dModel
        let ffnDim = MagpieCoreMLConstants.localTransformerFfnDim
        let maxPos = MagpieCoreMLConstants.localTransformerMaxPositions
        let K = MagpieCoreMLConstants.numCodebooks
        let V = MagpieCoreMLConstants.numCodesPerCodebook

        let inProjW = try snapshot(mlxLT.inProjection.weight,
                                    expected: [localDim, dModel],
                                    label: "lt.inProj.weight")
        let inProjB = try snapshot(mlxLT.inProjection.bias!,
                                    expected: [localDim],
                                    label: "lt.inProj.bias")
        let posEmb = try snapshot(mlxLT.positionEmbeddings.weight,
                                   expected: [maxPos, localDim],
                                   label: "lt.position_embeddings.weight")

        // Single layer. NeMo's TransformerLayer with kernel_size=1 has:
        //   norm_self / norm_pos_ff (LayerNorm without bias),
        //   self_attention.qkv_net (fused QKV Linear, no bias),
        //   self_attention.o_net (output Linear, no bias),
        //   pos_ff.conv1 / conv2 (1×1 Conv1d represented as Linear in MLX).
        // Names below match Sources/MagpieTTS/Transformer.swift.
        let layer = mlxLT.layers[0]
        guard let normSelfW = layer.normSelf.weight,
              let normPosFfW = layer.normPosFf.weight else {
            throw MagpieCoreMLError.invalidConstants(
                "lt.layer0: missing norm_self or norm_pos_ff weight")
        }
        let n1 = try snapshot(normSelfW,
                              expected: [localDim],
                              label: "lt.layer0.norm_self")
        let n2 = try snapshot(normPosFfW,
                              expected: [localDim],
                              label: "lt.layer0.norm_pos_ff")
        let qkv = try snapshot(layer.selfAttention.qkvNet.weight,
                               expected: [3 * localDim, localDim],
                               label: "lt.layer0.self_attn.qkv")
        let saO = try snapshot(layer.selfAttention.oNet.weight,
                               expected: [localDim, localDim],
                               label: "lt.layer0.self_attn.o")
        // MagpieCausalConv1d stores weight as (out, kernel, in). For LT
        // (kernel=1) that's (out, 1, in); the flat row-major buffer is
        // identical to a 2D (out, in) matrix so the Accelerate GEMM consumes
        // it directly.
        let f1 = try snapshotConv1d(layer.posFf.proj.weight,
                                     expected2D: [ffnDim, localDim],
                                     label: "lt.layer0.pos_ff.proj")
        let f2 = try snapshotConv1d(layer.posFf.oNet.weight,
                                     expected2D: [localDim, ffnDim],
                                     label: "lt.layer0.pos_ff.oNet")

        var outW: [[Float]] = []
        var outB: [[Float]] = []
        outW.reserveCapacity(K)
        outB.reserveCapacity(K)
        for cb in 0..<K {
            let head = mlxLT.outProjections[cb]
            outW.append(try snapshot(head.weight,
                                      expected: [V, localDim],
                                      label: "lt.out_proj.\(cb).weight"))
            outB.append(try snapshot(head.bias!,
                                      expected: [V],
                                      label: "lt.out_proj.\(cb).bias"))
        }

        return MagpieCoreMLLocalTransformerWeights(
            inProjWeight: inProjW, inProjBias: inProjB,
            posEmbedding: posEmb,
            norm1Weight: n1, norm2Weight: n2,
            saQkvWeight: qkv, saOWeight: saO,
            ffnConv1Weight: f1, ffnConv2Weight: f2,
            outProjWeights: outW, outProjBiases: outB,
            localDim: localDim, dModel: dModel, ffnDim: ffnDim,
            maxPositions: maxPos, numCodebooks: K, numCodesPerCodebook: V)
    }

    private static func extractAudioEmbeddings(from model: MagpieTTS) throws
        -> [[Float]]
    {
        let K = MagpieCoreMLConstants.numCodebooks
        let V = MagpieCoreMLConstants.numCodesPerCodebook
        let D = MagpieCoreMLConstants.dModel
        precondition(model.decoder.audioEmbeddings.count == K)
        var out: [[Float]] = []
        out.reserveCapacity(K)
        for k in 0..<K {
            let arr = try snapshot(model.decoder.audioEmbeddings[k].weight,
                                    expected: [V, D],
                                    label: "audio_embeddings.\(k).weight")
            out.append(arr)
        }
        return out
    }

    private static func snapshot(_ a: MLXArray, expected: [Int], label: String) throws
        -> [Float]
    {
        if a.shape != expected {
            throw MagpieCoreMLError.invalidConstants(
                "\(label): expected shape \(expected), got \(a.shape)")
        }
        let out: [Float] = a.asArray(Float.self)
        let n = expected.reduce(1, *)
        precondition(out.count == n, "\(label): expected \(n) elems, got \(out.count)")
        return out
    }

    /// MagpieCausalConv1d stores weight as `(out, kernel, in)` — for
    /// kernel=1 that's `(out, 1, in)`. The flat row-major byte order is
    /// identical to a 2D `(out, in)` matrix because the kernel axis has
    /// length 1, so the Accelerate GEMM consumes the buffer unchanged.
    /// We also accept `(out, in)` (already squeezed) and `(out, in, 1)`
    /// (trailing kernel axis) defensively in case the layout changes.
    private static func snapshotConv1d(_ a: MLXArray, expected2D: [Int], label: String) throws
        -> [Float]
    {
        let outF = expected2D[0]
        let inF = expected2D[1]
        let acceptable: [[Int]] = [
            expected2D,
            [outF, inF, 1],
            [outF, 1, inF],  // MLX MagpieCausalConv1d native layout
        ]
        if acceptable.contains(a.shape) {
            let out: [Float] = a.asArray(Float.self)
            let want = outF * inF
            precondition(out.count == want,
                         "\(label): expected \(want) elems, got \(out.count)")
            return out
        }
        throw MagpieCoreMLError.invalidConstants(
            "\(label): expected one of \(acceptable), got \(a.shape)")
    }
}
