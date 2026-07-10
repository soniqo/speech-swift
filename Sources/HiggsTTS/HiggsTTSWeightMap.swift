import Foundation

/// Destination component for a checkpoint tensor.
public enum HiggsTTSWeightComponent: Equatable, Sendable {
    case backbone
    case fusedEmbedding
    case codec
}

/// Maps upstream checkpoint keys onto runtime components, matching the
/// reference implementations:
///   body.layers.* / body.norm.*                     -> Qwen3 backbone
///   tied.embedding.text_embedding.*                 -> backbone embed_tokens
///   tied.embedding.modality_embeddings.0.embedding.* -> fused codebook table
///   tied.embedding.modality_embeddings.0.model.*    -> codec
///   tied.head.*                                     -> dropped (tied weights)
public enum HiggsTTSWeightMap {
    public static func remap(_ key: String) -> (component: HiggsTTSWeightComponent, key: String)? {
        if key.hasPrefix("body.layers.") {
            return (.backbone, "layers." + String(key.dropFirst("body.layers.".count)))
        }
        if key.hasPrefix("body.norm.") {
            return (.backbone, "norm." + String(key.dropFirst("body.norm.".count)))
        }
        if key.hasPrefix("tied.embedding.text_embedding.") {
            return (.backbone, "embed_tokens."
                + String(key.dropFirst("tied.embedding.text_embedding.".count)))
        }
        if key.hasPrefix("tied.embedding.modality_embeddings.0.embedding.") {
            return (.fusedEmbedding,
                String(key.dropFirst("tied.embedding.modality_embeddings.0.embedding.".count)))
        }
        if key.hasPrefix("tied.embedding.modality_embeddings.0.model.") {
            return (.codec,
                String(key.dropFirst("tied.embedding.modality_embeddings.0.model.".count)))
        }
        if key.hasPrefix("tied.head.") {
            return nil
        }
        return nil
    }
}
