import Foundation
import MLX
import MLXNN

// Demucs tensors are channels-first (NCL / NCHW); MLX convolutions are
// channels-last (NLC / NHWC). These helpers transpose around a directly-held
// MLXNN conv so the module tree stays flat and safetensors keys line up
// (e.g. `encoder.0.conv.weight`, not `...conv.conv.weight`).

@inline(__always)
func applyConv1dNCL(_ x: MLXArray, _ conv: Conv1d) -> MLXArray {
    // (N, C, L) -> (N, L, C) -> conv -> (N, C, L)
    conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
}

@inline(__always)
func applyConvT1dNCL(_ x: MLXArray, _ conv: ConvTransposed1d) -> MLXArray {
    conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
}

@inline(__always)
func applyConv2dNCHW(_ x: MLXArray, _ conv: Conv2d) -> MLXArray {
    // (N, C, H, W) -> (N, H, W, C) -> conv -> (N, C, H, W)
    conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
}

@inline(__always)
func applyConvT2dNCHW(_ x: MLXArray, _ conv: ConvTransposed2d) -> MLXArray {
    conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
}

/// Gated Linear Unit on a channels-first axis: split in half, `a * sigmoid(b)`.
/// Matches PyTorch `F.glu` (and demucs' GLU after the 1x1 "rewrite" conv).
@inline(__always)
func glu(_ x: MLXArray, axis: Int = 1) -> MLXArray {
    let halves = split(x, parts: 2, axis: axis)
    return halves[0] * sigmoid(halves[1])
}
