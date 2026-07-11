import Foundation
import MLX
import MLXNN

/// Pre-quantized embedding that can be loaded directly from safetensors
public class PreQuantizedEmbedding: Module {
    public let groupSize: Int
    public let bits: Int
    public let embeddingCount: Int
    public let dimensions: Int

    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var scales: MLXArray
    @ParameterInfo public var biases: MLXArray

    public init(embeddingCount: Int, dimensions: Int, groupSize: Int = 64, bits: Int = 4) {
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        self.groupSize = groupSize
        self.bits = bits

        // Packed last dim = dimensions * bits / 32 (MLX convention). For 4/8-bit this equals
        // dimensions/(32/bits); written this way it's also correct for 3/5/6-bit (32 not divisible by bits).
        let packedDim = dimensions * bits / 32
        let numGroups = dimensions / groupSize

        // Initialize with zeros - will be loaded from weights
        self._weight.wrappedValue = MLXArray.zeros([embeddingCount, packedDim], dtype: .uint32)
        self._scales.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)
        self._biases.wrappedValue = MLXArray.zeros([embeddingCount, numGroups], dtype: .bfloat16)

        super.init()
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        let x = x.flattened()
        let out = dequantized(
            weight[x], scales: scales[x], biases: biases[x],
            groupSize: groupSize, bits: bits)
        return out.reshaped(s + [-1])
    }

    /// For use as LM head (matmul with transposed weight)
    public func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMatmul(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
    }
}
