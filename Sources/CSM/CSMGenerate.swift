import Foundation
import MLX
import MLXNN
import PersonaPlex

// MARK: - Per-frame generation
//
// Ports SesameModel.generate_frame: sum the (text + 32 audio codebook) embeddings
// at each position, run the backbone, sample codebook 0 from codebook0_head, then
// autoregressively decode codebooks 1..31 with the small decoder + audio_head.
// Returns the 32 Mimi codebook tokens for one 80 ms frame.

extension CSMModel {

    func applyEmbedding(_ m: Module, _ x: MLXArray) -> MLXArray {
        if let e = m as? Embedding { return e(x) }
        if let q = m as? QuantizedEmbedding { return q(x) }
        fatalError("CSM: unsupported embedding module")
    }

    /// audio_embeddings(tokens + codebook * audioVocabSize)
    func embedAudio(_ codebook: Int, _ tokens: MLXArray) -> MLXArray {
        applyEmbedding(audioEmbeddings, tokens + MLXArray(Int32(codebook * cfg.audioVocabSize)))
    }

    /// tokens: [B, T, C+1] (last column text, 0..<C audio codebooks) -> [B, T, C+1, dim]
    func embedTokens(_ tokens: MLXArray) -> MLXArray {
        let c = cfg.audioNumCodebooks
        let b = tokens.shape[0], t = tokens.shape[1]
        let textTok = tokens[0..., 0..., -1]                                   // [B,T]
        let textEmb = applyEmbedding(textEmbeddings, textTok).expandedDimensions(axis: -2) // [B,T,1,dim]
        let offsets = (MLXArray(Int32(0)..<Int32(c)) * MLXArray(Int32(cfg.audioVocabSize)))
            .reshaped([1, 1, c])                                               // [1,1,C]
        let audioTok = tokens[0..., 0..., 0..<c] + offsets                     // [B,T,C]
        let audioEmbFlat = applyEmbedding(audioEmbeddings, audioTok.flattened())
        let audioEmb = audioEmbFlat.reshaped([b, t, c, -1])                    // [B,T,C,dim]
        return concatenated([audioEmb, textEmb], axis: -2)                     // [B,T,C+1,dim]
    }

    /// Generate one frame's 32 codebook tokens.
    /// - tokens: [B, T, C+1], tokensMask: [B, T, C+1]
    /// - backboneOffset: RoPE position for this segment in the backbone cache.
    /// Returns [B, C] Int32 codebook tokens.
    public func generateFrame(
        tokens: MLXArray, tokensMask: MLXArray, backboneOffset: Int,
        sampler: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let embeds = embedTokens(tokens)                                       // [B,T,C+1,dim]
        let masked = embeds * tokensMask.expandedDimensions(axis: -1)
        var h = masked.sum(axis: 2)                                            // [B,T,dim]
        h = backbone(h, offset: backboneOffset)                                // [B,T,dim]

        let lastH = h[0..., -1]                                                // [B,dim]
        let c0Logits = applyLinear(codebook0Head, lastH)                       // [B,audioVocab]
        var c0 = sampler(c0Logits).expandedDimensions(axis: -1)                // [B,1]
        var curr = embedAudio(0, c0)                                           // [B,1,dim]
        curr = concatenated([lastH.expandedDimensions(axis: 1), curr], axis: 1) // [B,2,dim]

        var samples = c0                                                       // [B,1]
        decoder.resetCache()
        var decOffset = 0
        for i in 1..<cfg.audioNumCodebooks {
            let decH = decoder(applyLinear(projection, curr), offset: decOffset) // [B,*,decDim]
            decOffset += curr.shape[1]
            let ciLogits = matmul(decH[0..., -1], audioHead[i - 1])            // [B,audioVocab]
            let ci = sampler(ciLogits).expandedDimensions(axis: -1)            // [B,1]
            curr = embedAudio(i, ci)                                           // [B,1,dim]
            samples = concatenated([samples, ci], axis: 1)
        }
        return samples                                                        // [B, C]
    }
}

extension CSMModel {
    /// Autoregressive frame loop. Primes the backbone with the prompt, then
    /// generates up to `maxFrames` frames, stopping on the all-zero EOS frame.
    /// - promptTokens: [1, T, C+1] Int32, promptMask: [1, T, C+1] Float
    /// - Returns codebook tokens shaped [1, C, numFrames] ready for Mimi.decode.
    public func generate(
        promptTokens: MLXArray, promptMask: MLXArray, maxFrames: Int,
        sampler: (MLXArray) -> MLXArray = argmaxSampler
    ) -> MLXArray {
        let c = cfg.audioNumCodebooks
        backbone.resetCache()
        var curr = promptTokens
        var currMask = promptMask
        var offset = 0
        var frames: [MLXArray] = []

        for _ in 0..<maxFrames {
            let sample = generateFrame(
                tokens: curr, tokensMask: currMask, backboneOffset: offset, sampler: sampler) // [1,C]
            offset += curr.shape[1]
            eval(sample)
            if all(sample .== MLXArray(Int32(0))).item(Bool.self) { break }   // EOS
            frames.append(sample)
            let zeroTok = MLXArray.zeros([1, 1], type: Int32.self)
            curr = concatenated([sample, zeroTok], axis: 1).expandedDimensions(axis: 1) // [1,1,C+1]
            let onesM = MLXArray.ones([1, c], type: Float.self)
            let zeroM = MLXArray.zeros([1, 1], type: Float.self)
            currMask = concatenated([onesM, zeroM], axis: 1).expandedDimensions(axis: 1)
        }
        if frames.isEmpty { return MLXArray.zeros([1, c, 0], type: Int32.self) }
        return stacked(frames, axis: 0).transposed(1, 2, 0)   // [numFrames,1,C] -> [1,C,numFrames]
    }
}

/// Greedy argmax sampler (deterministic — for smoke tests / reproducible runs).
public func argmaxSampler(_ logits: MLXArray) -> MLXArray {
    logits.argMax(axis: -1).asType(.int32)
}
