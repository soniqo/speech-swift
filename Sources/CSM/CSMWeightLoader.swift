import Foundation
import MLX
import MLXNN

// MARK: - Weight loading
//
// Loads OUR exported CSM weights (a single `model.safetensors` with `model.`
// HF-style keys) into `CSMModel`. Strips the `model.` prefix so keys line up
// with the Swift module tree, then verifies a clean bijection — a missing or
// unused key throws, which is exactly what catches an export/format mismatch.

public enum CSMWeightLoader {
    public static func load(model: CSMModel, from directory: URL) throws {
        let file = directory.appendingPathComponent("model.safetensors")
        let raw = try MLX.loadArrays(url: file)

        // Our export uses HF snake_case keys (model.backbone…input_layernorm,
        // mlp.gate_proj, text_embeddings). The Swift module tree keys parameters
        // by camelCase property name, so strip the `model.` prefix and convert
        // each dot-segment snake_case → camelCase.
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(raw.count)
        for (k, v) in raw {
            let stripped = k.hasPrefix("model.") ? String(k.dropFirst(6)) : k
            let key = stripped.split(separator: ".", omittingEmptySubsequences: false)
                .map { snakeToCamel(String($0)) }.joined(separator: ".")
            weights[key] = v
        }

        let params = ModuleParameters.unflattened(weights)
        try model.update(parameters: params, verify: .all)
        eval(model)
    }
}

/// "input_layernorm" → "inputLayernorm", "gate_proj" → "gateProj",
/// "codebook0_head" → "codebook0Head". Segments without underscores pass through.
func snakeToCamel(_ s: String) -> String {
    let parts = s.split(separator: "_", omittingEmptySubsequences: false)
    guard let first = parts.first else { return s }
    return String(first) + parts.dropFirst().map {
        $0.isEmpty ? "" : $0.prefix(1).uppercased() + $0.dropFirst()
    }.joined()
}
