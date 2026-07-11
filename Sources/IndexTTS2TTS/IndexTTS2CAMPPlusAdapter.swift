import ChatterboxTTS
import Foundation
import MLX

enum IndexTTS2CAMPPlusAdapter {
    static func load(from weights: [String: MLXArray]) throws -> CAMPPlus {
        let encoder = CAMPPlus()
        let remapped = remap(weights)
        try encoder.loadWeights(remapped)
        encoder.train(false)
        return encoder
    }

    static func remap(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            guard !key.hasSuffix(".num_batches_tracked") else { continue }
            guard let mapped = remapKey(key) else { continue }
            out[mapped] = convert(value, key: key)
        }

        return out
    }

    private static func remapKey(_ key: String) -> String? {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard !parts.isEmpty else { return nil }

        if parts[0] == "head" {
            return normalizeNonLinearSegments(key)
        }

        guard parts[0] == "xvector", parts.count >= 2 else {
            return normalizeNonLinearSegments(key)
        }

        switch parts[1] {
        case "tdnn":
            return normalizeNonLinearSegments("tdnn.\(parts.dropFirst(2).joined(separator: "."))")

        case let block where block.hasPrefix("block"):
            guard let blockIndex = Int(block.dropFirst("block".count)),
                  parts.count >= 3,
                  parts[2].hasPrefix("tdnnd"),
                  let layerIndex = Int(parts[2].dropFirst("tdnnd".count))
            else {
                return nil
            }
            let suffix = parts.dropFirst(3).joined(separator: ".")
            return normalizeNonLinearSegments("blocks.\(blockIndex - 1).layers.\(layerIndex - 1).\(suffix)")

        case let transit where transit.hasPrefix("transit"):
            guard let transitIndex = Int(transit.dropFirst("transit".count)) else {
                return nil
            }
            let suffix = parts.dropFirst(2).joined(separator: ".")
            return normalizeNonLinearSegments("transits.\(transitIndex - 1).\(suffix)")

        case "out_nonlinear":
            let suffix = parts.dropFirst(2).joined(separator: ".")
            return normalizeNonLinearSegments("out_nonlinear.\(suffix)")

        case "dense":
            let suffix = parts.dropFirst(2).joined(separator: ".")
            return normalizeNonLinearSegments("dense.\(suffix)")

        default:
            return nil
        }
    }

    private static func normalizeNonLinearSegments(_ key: String) -> String {
        key.replacingOccurrences(of: ".batchnorm.", with: ".0.")
    }

    private static func convert(_ value: MLXArray, key: String) -> MLXArray {
        guard key.hasSuffix(".weight") else {
            return value.asType(.float32)
        }

        switch value.shape.count {
        case 3:
            return value.asType(.float32).transposed(0, 2, 1)
        case 4:
            return value.asType(.float32).transposed(0, 2, 3, 1)
        default:
            return value.asType(.float32)
        }
    }
}
