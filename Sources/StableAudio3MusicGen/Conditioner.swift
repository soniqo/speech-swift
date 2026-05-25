import Foundation
import MLX

/// Replicates ExpoFourierFeatures from sa3_pipeline.py — log-spaced frequencies
/// in `[min_freq, max_freq]` projected onto cos+sin. Used by SecondsTotalEmbedder
/// and the DiT's timestep features (same code, different `dim` and freq range).
@inline(__always)
func expoFourierFeatures(_ t: MLXArray, dim: Int,
                          minFreq: Float, maxFreq: Float) -> MLXArray {
    let half = dim / 2
    let logMin = MLX.log(MLXArray(minFreq))
    let logMax = MLX.log(MLXArray(maxFreq))
    let ramp = MLXArray(0..<half).asType(.float32) / Float(max(half - 1, 1))
    let freqs = MLX.exp(ramp * (logMax - logMin) + logMin) * MLXArray(2 * Float.pi)
    let t32 = t.asType(.float32).reshaped([t.size, 1])      // [B, 1]
    let args = t32 * freqs.reshaped([1, half])              // [B, half]
    return MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)  // [B, dim]
}

/// `NumberConditioner({min_val:0, max_val:384, fourier_features_type:"expo"})`
/// → 768-d embedding: ExpoFourierFeatures(256) → Linear(256, 768) + bias.
public final class SecondsTotalEmbedder {
    public let weight: MLXArray    // (768, 256)
    public let bias: MLXArray      // (768,)
    public let minVal: Float
    public let maxVal: Float
    public let fourierDim: Int

    public init(weight: MLXArray, bias: MLXArray,
                minVal: Float = 0.0, maxVal: Float = 384.0, fourierDim: Int = 256) {
        self.weight = weight
        self.bias = bias
        self.minVal = minVal
        self.maxVal = maxVal
        self.fourierDim = fourierDim
    }

    public func callAsFunction(_ seconds: Float) -> MLXArray {
        let clipped = max(minVal, min(maxVal, seconds))
        let norm = (clipped - minVal) / (maxVal - minVal)
        let t = MLXArray([norm])
        let ff = expoFourierFeatures(t, dim: fourierDim, minFreq: 0.5, maxFreq: 10_000)
        let out = MLX.matmul(ff, weight.transposed()) + bias                   // [1, 768]
        return out.expandedDimensions(axis: 1)                                  // [1, 1, 768]
    }
}

/// Replace padded T5Gemma positions with the learned padding embedding.
/// `embeds` [B,S,768] · `mask` [B,S] int32 · `paddingEmbedding` [768].
public func applyPromptPadding(embeds: MLXArray, mask: MLXArray, paddingEmbedding: MLXArray) -> MLXArray {
    let m = mask.asType(embeds.dtype).expandedDimensions(axis: -1)              // [B,S,1]
    let pe = paddingEmbedding.asType(embeds.dtype).reshaped([1, 1, -1])
    return embeds * m + pe * (MLXArray(Float(1.0)).asType(embeds.dtype) - m)
}
