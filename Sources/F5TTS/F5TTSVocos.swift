import Foundation
import MLX
import MLXNN

final class F5TTSVocos {
    private let weights: [String: MLXArray]
    private let istft = F5ISTFT(nFFT: 1024, hopLength: 256, winLength: 1024)

    init(weights: [String: MLXArray]) {
        self.weights = weights
    }

    static func validate(_ weights: [String: MLXArray]) throws {
        try require(weights, component: "Vocos", key: "backbone.embed.weight", shape: [512, 100, 7])
        try require(weights, component: "Vocos", key: "backbone.convnext.0.dwconv.weight", shape: [512, 1, 7])
        try require(weights, component: "Vocos", key: "backbone.final_layer_norm.weight", shape: [512])
        try require(weights, component: "Vocos", key: "head.out.weight", shape: [1026, 512])
        try require(weights, component: "Vocos", key: "head.istft.window", shape: [1024])
    }

    func decode(mel: MLXArray) -> MLXArray {
        var h = mel
        if h.shape.count == 2 {
            h = h.expandedDimensions(axis: 0)
        }
        if h.dim(1) != 100 {
            h = h.transposed(0, 2, 1)
        }

        h = convNCL(h, prefix: "backbone.embed", padding: 3, groups: 1)
        h = f5LayerNorm(h.transposed(0, 2, 1), weights: weights, prefix: "backbone.norm", eps: 1e-6)
            .transposed(0, 2, 1)

        for layer in 0..<8 {
            h = convNext(h, prefix: "backbone.convnext.\(layer)")
        }

        h = f5LayerNorm(h.transposed(0, 2, 1), weights: weights, prefix: "backbone.final_layer_norm", eps: 1e-6)
        var out = f5Linear(h, weights: weights, prefix: "head.out")
        let parts = split(out, parts: 2, axis: -1)
        var mag = exp(parts[0])
        mag = MLX.minimum(mag, MLXArray(Float(1e2)).asType(mag.dtype))
        let phase = parts[1]
        let real = mag * cos(phase)
        let imag = mag * sin(phase)
        out = (real + imag.asImaginary()).transposed(0, 2, 1)
        return istft(out)
    }

    private func convNext(_ x: MLXArray, prefix: String) -> MLXArray {
        let residual = x
        var h = convNCL(x, prefix: "\(prefix).dwconv", padding: 3, groups: 512)
        h = h.transposed(0, 2, 1)
        h = f5LayerNorm(h, weights: weights, prefix: "\(prefix).norm", eps: 1e-6)
        h = f5Linear(h, weights: weights, prefix: "\(prefix).pwconv1")
        h = gelu(h)
        h = f5Linear(h, weights: weights, prefix: "\(prefix).pwconv2")
        h = weights["\(prefix).gamma"]!.asType(h.dtype) * h
        return residual + h.transposed(0, 2, 1)
    }

    private func convNCL(_ x: MLXArray, prefix: String, padding: Int, groups: Int) -> MLXArray {
        f5Conv1dNCL(
            x,
            weight: weights["\(prefix).weight"]!,
            bias: weights["\(prefix).bias"],
            padding: padding,
            groups: groups)
    }
}

func require(_ weights: [String: MLXArray], component: String, key: String, shape: [Int]) throws {
    guard let tensor = weights[key] else {
        throw F5TTSError.missingTensor(component: component, key: key)
    }
    guard tensor.shape == shape else {
        throw F5TTSError.invalidTensorShape(component: component, key: key, expected: shape, actual: tensor.shape)
    }
}
