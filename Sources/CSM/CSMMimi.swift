import Foundation
import MLX
import MLXNN
import PersonaPlex

// MARK: - Mimi codec loading (32 codebooks for CSM)
//
// Reuses PersonaPlex's Mimi. CSM uses the same Kyutai Mimi checkpoint as Moshi
// but with all 32 RVQ codebooks (Moshi uses 16). Loads from our export's
// bundled `mimi.safetensors`.

public enum CSMMimi {
    public static func load(from file: URL, numCodebooks: Int = 32) throws -> Mimi {
        let cfg = MimiConfig.moshiko(numCodebooks: numCodebooks)
        let model = Mimi(cfg: cfg)
        var weights = try MLX.loadArrays(url: file)
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        finalizeCodebooks(model)
        eval(model)
        return model
    }

    /// EuclideanCodebook stores embed_sum / cluster_usage; materialize the embedding.
    static func finalizeCodebooks(_ m: Module) {
        if let cb = m as? EuclideanCodebook { cb.updateInPlace() }
        for (_, child) in m.children().flattened() { finalizeCodebooks(child) }
    }
}
