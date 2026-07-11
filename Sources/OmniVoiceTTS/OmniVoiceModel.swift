import Foundation
import MLX
import MLXNN

/// Non-Module holder so the precomputed offsets aren't reflected as a loadable
/// parameter (a bare `MLXArray` property would be).
final class OVOffsets {
    let value: MLXArray
    init(_ v: MLXArray) { value = v }
}

/// OmniVoice top model: the Qwen3 backbone (`llm.*`) plus the audio token I/O.
///
/// The sequence interleaves text and audio positions. At each position the input
/// carries a text token (row 0) and `numAudioCodebook` audio tokens (rows 1…);
/// `audioMask` selects whether that position embeds as text or as the summed
/// audio codebooks. The backbone (bidirectional) predicts per-codebook logits.
public final class OmniVoiceModel: Module {
    let cfg: OmniVoiceConfig

    @ModuleInfo(key: "llm") var llm: OVBackbone
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "audio_heads") var audioHeads: Linear

    /// `arange(numCodebook) * audioVocabSize` — shifts each codebook into its
    /// slice of the shared 8200-row audio embedding / head matrices. A buffer in
    /// the checkpoint; recomputed here (not a learnable parameter).
    let offsets: OVOffsets

    public init(_ cfg: OmniVoiceConfig = OmniVoiceConfig()) {
        self.cfg = cfg
        _llm.wrappedValue = OVBackbone(cfg)
        _audioEmbeddings.wrappedValue = Embedding(
            embeddingCount: cfg.audioMatrixRows, dimensions: cfg.hiddenSize)
        _audioHeads.wrappedValue = Linear(cfg.hiddenSize, cfg.audioMatrixRows, bias: false)
        offsets = OVOffsets(MLXArray(
            (0 ..< cfg.numAudioCodebook).map { Int32($0 * cfg.audioVocabSize) }))
        super.init()
    }

    /// Fuse text + audio tokens into backbone input embeddings.
    /// - inputIds: `[B, numCodebook, L]` — at text positions the text token id is
    ///   repeated across all `numCodebook` rows; at audio positions each row holds
    ///   that codebook's token. Text uses **row 0**; audio zeroes non-audio
    ///   positions (`× audio_mask`), shifts each row into its codebook slice, and
    ///   sums the 8 codebook embeddings.
    /// - audioMask: `[B, L]` (0/1) — 1 at audio positions.
    func prepareEmbedInputs(inputIds: MLXArray, audioMask: MLXArray) -> MLXArray {
        let textEmbeds = llm.embedTokens(inputIds[0..., 0, 0...])             // row 0 → [B, L, H]
        let masked = inputIds * audioMask.expandedDimensions(axis: 1)         // [B, C, L]
        let shifted = masked + offsets.value.reshaped([1, cfg.numAudioCodebook, 1])
        let audioEmbeds = audioEmbeddings(shifted).sum(axis: 1)              // [B, L, H]
        let mask = (audioMask .!= MLXArray(Int32(0))).expandedDimensions(axis: -1)  // [B, L, 1]
        return MLX.where(mask, audioEmbeds, textEmbeds)
    }

    /// Forward pass → per-codebook logits `[B, numCodebook, L, audioVocabSize]`
    /// (matches the reference `audio_logits` layout).
    public func callAsFunction(inputIds: MLXArray, audioMask: MLXArray) -> MLXArray {
        let embeds = prepareEmbedInputs(inputIds: inputIds, audioMask: audioMask)
        let hidden = llm(embeds)                                              // [B, L, H]
        let (b, l) = (hidden.dim(0), hidden.dim(1))
        let flat = audioHeads(hidden)                                        // [B, L, C*V]
        return flat
            .reshaped([b, l, cfg.numAudioCodebook, cfg.audioVocabSize])      // [B, L, C, V]
            .transposed(0, 2, 1, 3)                                          // [B, C, L, V]
    }

    /// Load the published bundle's `model.safetensors`. Splits by prefix:
    /// `llm.*` → backbone, `audio_embeddings`/`audio_heads` → here. The
    /// `codebook_layer_offsets` buffer is recomputed, so it's dropped.
    ///
    /// - Parameter quantization: when set (e.g. group 64 / 8-bit for the
    ///   `OmniVoice-MLX-int8` bundle), the Linear/Embedding layers are swapped to
    ///   their quantized form *before* loading so the packed weight + scales +
    ///   biases line up. The float-weight bundles (fp16/fp32) pass `nil`.
    public func loadWeights(
        from modelSafetensors: URL, quantization: (groupSize: Int, bits: Int)? = nil
    ) throws {
        if let q = quantization {
            MLXNN.quantize(model: self, groupSize: q.groupSize, bits: q.bits)
        }
        let raw = try MLX.loadArrays(url: modelSafetensors)
        var weights: [String: MLXArray] = [:]
        for (k, v) in raw where k != "codebook_layer_offsets" {
            // Quantized bundles carry packed-uint32 weights + float scales/biases;
            // keep their dtypes. Float bundles are upcast to fp32 for compute.
            weights[k] = quantization == nil ? v.asType(.float32) : v
        }
        try update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(parameters())
    }
}
