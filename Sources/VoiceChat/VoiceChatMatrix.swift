import MLX

/// Dense or pre-quantized affine matrix from an exported VoiceChat bundle.
///
/// Quantized tensors cannot be treated as ordinary transposable arrays: their
/// UInt32 payload is packed and each tensor may have a different bit width.
/// Keeping the quantization metadata beside the weight also lets the very large
/// EAR-TTS mixture head dequantize only the rows selected for the current frame.
public struct VoiceChatMatrix {
    public let weight: MLXArray
    public let scales: MLXArray?
    public let biases: MLXArray?
    public let bias: MLXArray?
    public let quantization: QuantizationConfig?

    init(
        weights: [String: MLXArray],
        name: String,
        quantization: QuantizationConfig?,
        biasName: String? = nil
    ) throws {
        guard let raw = weights[name] else {
            throw VoiceChatLoadError.unexpectedKeys([name])
        }
        self.quantization = quantization
        if quantization != nil {
            let prefix = String(name.dropLast(".weight".count))
            guard let scales = weights["\(prefix).scales"] else {
                throw VoiceChatLoadError.unexpectedKeys(["\(prefix).scales"])
            }
            self.weight = raw
            self.scales = scales
            self.biases = weights["\(prefix).biases"]
        } else {
            self.weight = raw.asType(.float32)
            self.scales = nil
            self.biases = nil
        }
        if let biasName {
            guard let bias = weights[biasName] else {
                throw VoiceChatLoadError.unexpectedKeys([biasName])
            }
            self.bias = bias.asType(.float32)
        } else {
            self.bias = nil
        }
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        let output: MLXArray
        if let spec = quantization, let scales {
            output = MLX.quantizedMM(
                input, weight, scales: scales, biases: biases,
                transpose: true, groupSize: spec.groupSize, bits: spec.bits,
                mode: .affine)
        } else {
            output = MLX.matmul(input, weight.transposed())
        }
        return bias.map { output + $0 } ?? output
    }

    /// Dequantize only the requested output rows.
    ///
    /// This supports cheap exact preconditions on very large vocabulary
    /// projections. A function-call start token cannot be the full-head argmax
    /// unless its logit first beats the function-channel PAD token, so those two
    /// rows can screen ordinary duplex frames without touching all 131k rows.
    func outputRows(_ indices: MLXArray) -> MLXArray {
        if let spec = quantization, let scales {
            return MLX.dequantized(
                weight[indices],
                scales: scales[indices],
                biases: biases.map { $0[indices] },
                groupSize: spec.groupSize,
                bits: spec.bits,
                mode: .affine)
        }
        return weight[indices]
    }

    /// Gather one row group per index, dequantizing only those selected rows.
    func selectedRows(_ indices: MLXArray, groups: Int, rowsPerGroup: Int) -> MLXArray {
        if let spec = quantization, let scales {
            let packed = weight.reshaped([groups, rowsPerGroup, -1])[indices]
            let selectedScales = scales.reshaped([groups, rowsPerGroup, -1])[indices]
            let selectedBiases = biases.map {
                $0.reshaped([groups, rowsPerGroup, -1])[indices]
            }
            return MLX.dequantized(
                packed, scales: selectedScales, biases: selectedBiases,
                groupSize: spec.groupSize, bits: spec.bits, mode: .affine)
        }
        return weight.reshaped([groups, rowsPerGroup, -1])[indices]
    }
}
