import Foundation
import MLX
import MLXNN

/// The fused multi-codebook audio interface: one `[N * V, hidden]` embedding
/// table shared between input and output. Input frames embed each codebook's
/// code at `code + codebook * V` and sum the `N` vectors; output logits reuse
/// the same table as a tied linear head, reshaped to `[N, V]` per position.
final class HiggsTTSFusedCodebook: Module {
    let numCodebooks: Int
    let codebookSize: Int

    @ModuleInfo var embedding: Embedding

    init(numCodebooks: Int, codebookSize: Int, hiddenSize: Int) {
        self.numCodebooks = numCodebooks
        self.codebookSize = codebookSize
        _embedding = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: numCodebooks * codebookSize, dimensions: hiddenSize))
        super.init()
    }

    /// Embeds audio frames `[T, N]` (or `[B, T, N]`) into `[T, hidden]`
    /// (or `[B, T, hidden]`) by summing the per-codebook embeddings.
    func embed(_ codes: MLXArray) -> MLXArray {
        let offsets = MLXArray(0..<Int32(numCodebooks)) * Int32(codebookSize)
        let fused = codes.asType(.int32) + offsets
        return embedding(fused).sum(axis: -2)
    }

    /// Projects hidden states `[..., hidden]` to logits `[..., N, V]` with the
    /// tied embedding table.
    func logits(_ hidden: MLXArray) -> MLXArray {
        let flat = embedding.asLinear(hidden)
        return flat.reshaped(hidden.shape.dropLast() + [numCodebooks, codebookSize])
    }
}
