// Copyright 2026 the HuggingFace Team. All rights reserved.
// Adapted to Swift from the Apache-2.0 Inkling 2 vision reference.
// See ThirdPartyNotices/Inkling2Mini.txt.
import Foundation
import CoreGraphics
import ImageIO
import MLX
import MLXNN

/// Still-image RGB input. Images are padded to 40-pixel patch boundaries;
/// dimensions and the patch budget are checked before allocating model tensors.
public struct Inkling2Image: Sendable {
    public let width, height: Int
    public let rgb: [UInt8]
    public init(width: Int, height: Int, rgb: [UInt8]) throws {
        guard (1...4096).contains(width), (1...4096).contains(height), rgb.count == width * height * 3 else {
            throw Inkling2Error.invalid("expected RGB image data with dimensions in 1...4096")
        }
        self.width = width; self.height = height; self.rgb = rgb
    }
    public static func load(_ url: URL) throws -> Inkling2Image {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
              image.width <= 4096, image.height <= 4096 else {
            throw Inkling2Error.invalid("unable to decode image, or image exceeds 4096 pixels per dimension")
        }
        let w = image.width, h = image.height
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: w, height: h,
                bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            for y in stride(from: 0, to: h, by: 8) {
                for x in stride(from: 0, to: w, by: 8) {
                    let shade: CGFloat = (y / 8 + x / 8) % 2 == 0 ? 204 / 255 : 153 / 255
                    context.setFillColor(red: shade, green: shade, blue: shade, alpha: 1)
                    context.fill(CGRect(x: x, y: y, width: 8, height: 8))
                }
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw Inkling2Error.invalid("unable to allocate image conversion context") }
        var rgb = [UInt8](); rgb.reserveCapacity(w * h * 3)
        for i in stride(from: 0, to: rgba.count, by: 4) { rgb.append(contentsOf: rgba[i..<(i + 3)]) }
        return try Inkling2Image(width: w, height: h, rgb: rgb)
    }
    public var rows: Int { (height + 39) / 40 }
    public var columns: Int { (width + 39) / 40 }

    public func patchPixels(maxTokens: Int = 1024) throws -> [Float] {
        guard maxTokens > 0, rows * columns <= maxTokens else {
            throw Inkling2Error.invalid("image exceeds the \(maxTokens)-patch budget; resize it before inference")
        }
        var result = [Float](repeating: -1 / 127.5 - 1, count: rows * columns * 40 * 40 * 3)
        for y in 0..<height {
            for x in 0..<width {
                let target = (((y / 40) * columns + x / 40) * 1600 + (y % 40) * 40 + x % 40) * 3
                let source = (y * width + x) * 3
                for c in 0..<3 { result[target + c] = Float(rgb[source + c]) / 127.5 - 1 }
            }
        }
        return result
    }
}

extension Inkling2Model {
    func validateVisionWeights() throws {
        let p = "vision_tower."
        try checkMatrix(p + "embeddings.patch_embedding", rows: 1024, columns: 1200)
        try check(p + "embeddings.position_embedding", shape: [1024, 1024])
        for name in ["class_embedding", "cls_position_embedding"] { try check(p + "embeddings." + name, shape: [1024]) }
        for suffix in ["weight", "bias"] { try check(p + "pre_layernorm." + suffix, shape: [1024]) }
        try checkMatrix(p + "proj_in", rows: 4096, columns: 4096)
        try checkMatrix(p + "proj_out", rows: config.hiddenSize, columns: 4096)
        for i in 0..<23 {
            let layer = p + "layers.\(i)."
            for name in ["layernorm_before", "layernorm_after"] {
                for suffix in ["weight", "bias"] { try check(layer + name + "." + suffix, shape: [1024]) }
            }
            for name in ["lambda_1", "lambda_2"] { try check(layer + name, shape: [1024]) }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                try checkMatrix(layer + "self_attn." + name, rows: 1024, columns: 1024)
                try check(layer + "self_attn." + name + ".bias", shape: [1024])
            }
            try checkMatrix(layer + "mlp.fc1", rows: 4096, columns: 1024)
            try check(layer + "mlp.fc1.bias", shape: [4096])
            try checkMatrix(layer + "mlp.fc2", rows: 1024, columns: 4096)
            try check(layer + "mlp.fc2.bias", shape: [1024])
        }
    }
    /// Encode normalized [patches,40,40,3] still-image patches into text-space
    /// embeddings. The supported Mini vision architecture uses 20-pixel native
    /// patches and merges each 2×2 group into one language token.
    public func imageEmbeddings(_ image: Inkling2Image, maxTokens: Int = 1024) throws -> MLXArray {
        try validateVisionWeights()
        let prefix = "vision_tower.", width = 1024, heads = 16, dim = 64
        guard weights[prefix + "embeddings.patch_embedding.weight"]?.shape == [width, 1200],
              weights[prefix + "embeddings.position_embedding"]?.shape == [1024, width],
              weights[prefix + "proj_out.weight"]?.shape == [config.hiddenSize, 4096] else {
            throw Inkling2Error.invalid("checkpoint does not contain the supported Mini vision encoder")
        }
        let count = image.rows * image.columns
        let pixels = try image.patchPixels(maxTokens: maxTokens)
        let dtype = weight(prefix + "pre_layernorm.weight").dtype
        let input = MLXArray(pixels).reshaped(count, 2, 20, 2, 20, 3).transposed(0, 1, 3, 2, 4, 5)
            .reshaped(count * 4, 1200).asType(dtype)
        var x = linear(input, prefix + "embeddings.patch_embedding")
        // Half-pixel bilinear interpolation of the learned 32×32 position grid.
        let rows = image.rows * 2, columns = image.columns * 2
        let pos = weight(prefix + "embeddings.position_embedding").reshaped(32, 32, width).asType(.float32)
        var interpolated: [MLXArray] = []
        for y in 0..<rows {
            let yf = max(0, (Float(y) + 0.5) * 32 / Float(rows) - 0.5)
            let y0 = min(Int(yf), 31), y1 = min(y0 + 1, 31), wy = yf - Float(y0)
            let xf = maximum((MLXArray(0..<columns).asType(.float32) + 0.5) * 32 / Float(columns) - 0.5, 0)
            let x0 = minimum(floor(xf), 31).asType(.int32), x1 = minimum(x0 + 1, 31)
            let wx = (xf - x0.asType(.float32)).expandedDimensions(axis: -1)
            let first = pos[y0][x0] * (1 - wx) + pos[y0][x1] * wx
            let second = pos[y1][x0] * (1 - wx) + pos[y1][x1] * wx
            interpolated.append(first * (1 - wy) + second * wy)
        }
        let position = stacked(interpolated).reshaped(image.rows, 2, image.columns, 2, width)
            .transposed(0, 2, 1, 3, 4).reshaped(count * 4, width).asType(dtype)
        x = x + position
        let cls = (weight(prefix + "embeddings.class_embedding") + weight(prefix + "embeddings.cls_position_embedding")).asType(dtype)
        x = MLXFast.layerNorm(concatenated([cls.expandedDimensions(axis: 0), x], axis: 0),
                              weight: weight(prefix + "pre_layernorm.weight"), bias: weight(prefix + "pre_layernorm.bias"), eps: 1e-5)
        var coordinates: [Float] = [0, 0]
        for row in 0..<image.rows {
            for col in 0..<image.columns {
                for dy in 0..<2 { for dx in 0..<2 { coordinates += [Float(col * 2 + dx + 1), Float(row * 2 + dy + 1)] } }
            }
        }
        let positions = MLXArray(coordinates).reshaped(-1, 2, 1)
        let inv = pow(MLXArray(Float(10000)), -MLXArray(stride(from: 0, to: dim / 2, by: 2)).asType(.float32) / Float(dim / 2))
        let angles = repeated((positions * inv).reshaped(count * 4 + 1, -1), count: 2, axis: -1)
        let cosine = cos(angles).reshaped(1, count * 4 + 1, 1, dim)
        let sine = sin(angles).reshaped(1, count * 4 + 1, 1, dim)
        func rotate(_ value: MLXArray) -> MLXArray {
            let pairs = value.reshaped(Array(value.shape.dropLast()) + [dim / 2, 2])
            return stacked([-pairs[0..., 0..., 0..., 0..., 1], pairs[0..., 0..., 0..., 0..., 0]], axis: -1).reshaped(value.shape)
        }
        for i in 0..<23 {
            let p = prefix + "layers.\(i)."
            for name in ["layernorm_before", "layernorm_after"] {
                guard weights[p + name + ".weight"]?.shape == [width], weights[p + name + ".bias"]?.shape == [width] else {
                    throw Inkling2Error.invalid("incomplete vision layer \(i)")
                }
            }
            let input = MLXFast.layerNorm(x, weight: weight(p + "layernorm_before.weight"),
                                          bias: weight(p + "layernorm_before.bias"), eps: 1e-5)
            var q = linear(input, p + "self_attn.q_proj").reshaped(1, -1, heads, dim)
            var k = linear(input, p + "self_attn.k_proj").reshaped(1, -1, heads, dim)
            q = (q.asType(.float32) * cosine + rotate(q.asType(.float32)) * sine).asType(dtype)
            k = (k.asType(.float32) * cosine + rotate(k.asType(.float32)) * sine).asType(dtype)
            let v = linear(input, p + "self_attn.v_proj").reshaped(1, -1, heads, dim)
            let attention = MLXFast.scaledDotProductAttention(queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3), scale: 1 / sqrt(Float(dim)), mask: .none)
                .transposed(0, 2, 1, 3).reshaped(-1, width)
            x = x + weight(p + "lambda_1") * linear(attention, p + "self_attn.out_proj")
            let normalized = MLXFast.layerNorm(x, weight: weight(p + "layernorm_after.weight"),
                                               bias: weight(p + "layernorm_after.bias"), eps: 1e-5)
            x = x + weight(p + "lambda_2") * linear(gelu(linear(normalized, p + "mlp.fc1")), p + "mlp.fc2")
            eval(x)
        }
        let patches = x[1...].reshaped(count, 4 * width)
        return linear(gelu(linear(patches, prefix + "proj_in")), prefix + "proj_out")
    }
}
