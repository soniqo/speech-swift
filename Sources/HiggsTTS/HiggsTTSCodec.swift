import Foundation
import MLX

/// Higgs codec decode path (tokens → 24 kHz waveform): 8-codebook RVQ decode,
/// a 1024→256 projection, and a DAC-style transposed-conv decoder with Snake
/// activations. Functional graph over the checkpoint tensors (F5 house
/// style); weights are cast to float32 at load to match the reference, whose
/// PyTorch loader materializes the bf16 codec in float32.
///
/// Encode path (reference cloning) lands separately: it additionally needs
/// the acoustic encoder, the embedded HuBERT semantic model, and the
/// semantic/acoustic fusion projection.
final class HiggsTTSCodec {
    /// Samples per codec frame at 24 kHz (25 fps).
    static let samplesPerFrame = 960
    static let numCodebooks = 8
    static let codebookSize = 1024

    let weights: [String: MLXArray]

    /// - Parameter rawWeights: codec tensors keyed without the checkpoint
    ///   prefix, in the original PyTorch layouts.
    init(rawWeights: [String: MLXArray]) throws {
        var prepared: [String: MLXArray] = [:]
        for (key, value) in rawWeights {
            var key = key
            var value = value.asType(.float32)
            if key.hasSuffix(".codebook.embed") {
                key = String(key.dropLast("embed".count)) + "weight"
            } else if key.hasSuffix(".embed_avg") || key.hasSuffix(".cluster_size")
                || key.hasSuffix(".inited") || key == "semantic_model.masked_spec_embed" {
                continue
            }
            // Weight-normed convolutions store parametrized weights.
            if key.hasSuffix(".parametrizations.weight.original0") {
                key = String(key.dropLast(".parametrizations.weight.original0".count)) + ".weight_g"
            } else if key.hasSuffix(".parametrizations.weight.original1") {
                key = String(key.dropLast(".parametrizations.weight.original1".count)) + ".weight_v"
            }
            if value.ndim == 3, key.hasSuffix(".alpha") {
                value = value.transposed(0, 2, 1)
            } else if value.ndim == 3,
                      key.hasSuffix(".weight") || key.hasSuffix(".weight_g") || key.hasSuffix(".weight_v") {
                // PyTorch conv layouts → MLX: Conv1d [Cout, Cin, K] → [Cout, K, Cin];
                // ConvTranspose1d [Cin, Cout, K] → [Cout, K, Cin].
                value = key.contains("conv_t")
                    ? value.transposed(1, 2, 0)
                    : value.transposed(0, 2, 1)
            }
            prepared[key] = value
        }
        self.weights = prepared

        for key in Self.requiredDecodeKeys {
            guard prepared[key] != nil else {
                throw HiggsTTSError.missingRequiredFile("codec tensor \(key)")
            }
        }
    }

    var memoryFootprint: Int {
        weights.values.reduce(0) { $0 + $1.nbytes }
    }

    /// Decodes raw (de-delayed) codec codes `[T][8]` into `T * 960` samples.
    func decode(_ codes: [[Int32]]) throws -> MLXArray {
        guard !codes.isEmpty else {
            return MLXArray.zeros([0])
        }
        guard codes.allSatisfy({ $0.count == Self.numCodebooks }) else {
            throw HiggsTTSError.invalidCodes("expected \(Self.numCodebooks) codebooks per frame")
        }
        let frames = codes.count
        let ids = MLXArray(codes.flatMap { $0 }).reshaped(1, frames, Self.numCodebooks)

        var z: MLXArray?
        for codebook in 0..<Self.numCodebooks {
            let table = weights["quantizer.quantizers.\(codebook).codebook.weight"]!
            let embedded = take(table, ids[0..., 0..., codebook], axis: 0)
            let projected = linear(embedded, prefix: "quantizer.quantizers.\(codebook).project_out")
            z = z.map { $0 + projected } ?? projected
        }
        var x = linear(z!, prefix: "fc2")

        x = conv(x, prefix: "acoustic_decoder.conv1", padding: 3)
        for (index, stride) in Self.decoderStrides.enumerated() {
            x = decoderBlock(x, prefix: "acoustic_decoder.block.\(index)", stride: stride)
        }
        x = snake(x, alphaKey: "acoustic_decoder.snake1.alpha")
        x = conv(x, prefix: "acoustic_decoder.conv2", padding: 3)
        return x[0, 0..., 0]
    }

    // MARK: - Graph pieces

    private static let decoderStrides = [8, 5, 4, 2, 3]

    private static let requiredDecodeKeys: [String] = {
        var keys = ["fc2.weight", "fc2.bias",
                    "acoustic_decoder.conv1.weight", "acoustic_decoder.conv2.weight",
                    "acoustic_decoder.snake1.alpha"]
        for codebook in 0..<numCodebooks {
            keys.append("quantizer.quantizers.\(codebook).codebook.weight")
            keys.append("quantizer.quantizers.\(codebook).project_out.weight")
        }
        for block in 0..<decoderStrides.count {
            keys.append("acoustic_decoder.block.\(block).conv_t1.weight")
        }
        return keys
    }()

    func decoderBlock(_ input: MLXArray, prefix: String, stride: Int) -> MLXArray {
        let frames = input.dim(1)
        var x = snake(input, alphaKey: "\(prefix).snake1.alpha")
        x = convTransposed(x, prefix: "\(prefix).conv_t1", stride: stride, padding: stride / 2)
        let expected = frames * stride
        if x.dim(1) > expected {
            x = x[0..., 0..<expected, 0...]
        }
        x = residualUnit(x, prefix: "\(prefix).res_unit1", dilation: 1)
        x = residualUnit(x, prefix: "\(prefix).res_unit2", dilation: 3)
        x = residualUnit(x, prefix: "\(prefix).res_unit3", dilation: 9)
        return x
    }

    func residualUnit(_ input: MLXArray, prefix: String, dilation: Int) -> MLXArray {
        var y = snake(input, alphaKey: "\(prefix).snake1.alpha")
        y = conv(y, prefix: "\(prefix).conv1", padding: 3 * dilation, dilation: dilation)
        y = snake(y, alphaKey: "\(prefix).snake2.alpha")
        y = conv(y, prefix: "\(prefix).conv2", padding: 0)
        var x = input
        let trim = (x.dim(1) - y.dim(1)) / 2
        if trim > 0 {
            x = x[0..., trim..<(x.dim(1) - trim), 0...]
        }
        return x + y
    }

    /// Snake activation, computed in float32 like the reference:
    /// `x + sin²(αx) / (α + 1e-9)`.
    func snake(_ x: MLXArray, alphaKey: String) -> MLXArray {
        let alpha = weights[alphaKey]!
        let scaled = MLX.sin(alpha * x)
        return x + (1.0 / (alpha + 1e-9)) * scaled * scaled
    }

    func conv(
        _ x: MLXArray, prefix: String, padding: Int, stride: Int = 1, dilation: Int = 1
    ) -> MLXArray {
        var y = conv1d(x, weights["\(prefix).weight"]!,
                       stride: stride, padding: padding, dilation: dilation)
        if let bias = weights["\(prefix).bias"] {
            y = y + bias
        }
        return y
    }

    func convTransposed(
        _ x: MLXArray, prefix: String, stride: Int, padding: Int
    ) -> MLXArray {
        var y = convTransposed1d(x, weights["\(prefix).weight"]!,
                                 stride: stride, padding: padding)
        if let bias = weights["\(prefix).bias"] {
            y = y + bias
        }
        return y
    }

    func linear(_ x: MLXArray, prefix: String) -> MLXArray {
        var y = matmul(x, weights["\(prefix).weight"]!.transposed())
        if let bias = weights["\(prefix).bias"] {
            y = y + bias
        }
        return y
    }
}
