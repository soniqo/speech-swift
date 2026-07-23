import Foundation
import MLX
import MLXRandom
import MLXNN
import Tokenizers
import Hub
import PersonaPlex

// MARK: - CSM text→audio pipeline
//
// The full runtime: Llama BPE tokenizer + Mimi codec (32 codebooks) + CSM model.
// Given a text and a reference clip (voice), tokenizes the prompt, runs the
// autoregressive frame loop, and Mimi-decodes to a 24 kHz waveform — entirely
// in speech-swift on our exported weights.

public final class CSMPipeline {
    public let cfg: CSMConfig
    let model: CSMModel
    let mimi: Mimi
    let tokenizer: Tokenizer

    public init(directory: URL) async throws {
        cfg = try CSMConfig.load(from: directory)
        model = CSMModel(cfg)
        try CSMWeightLoader.load(model: model, from: directory)
        mimi = try CSMMimi.load(from: directory.appendingPathComponent("mimi.safetensors"),
                                numCodebooks: cfg.audioNumCodebooks)
        tokenizer = try await AutoTokenizer.from(modelFolder: directory)
    }

    // Text → [len, C+1] frame where the last column carries the text tokens.
    func tokenizeText(_ text: String, speaker: Int) -> (MLXArray, MLXArray) {
        let ids = tokenizer.encode(text: "[\(speaker)] \(text)").map { Int32($0) }
        let len = ids.count
        let c = cfg.audioNumCodebooks
        let textCol = MLXArray(ids).reshaped([len, 1])
        let frame = concatenated([MLXArray.zeros([len, c], type: Int32.self), textCol], axis: 1)
        let mask = concatenated([MLXArray.zeros([len, c], type: Float.self),
                                 MLXArray.ones([len, 1], type: Float.self)], axis: 1)
        return (frame, mask)
    }

    // Reference audio → [T(+1), C+1] frames where the first C columns carry Mimi codes.
    func tokenizeAudio(_ audio: MLXArray, addEos: Bool) -> (MLXArray, MLXArray) {
        var codes = mimi.encode(audio.reshaped([1, 1, -1]))[0].asType(.int32)   // [C, T]
        let c = cfg.audioNumCodebooks
        if addEos {
            codes = concatenated([codes, MLXArray.zeros([c, 1], type: Int32.self)], axis: 1)
        }
        let tp = codes.transposed(1, 0).asType(.int32)             // [T, C]
        let t = tp.shape[0]
        let frame = concatenated([tp, MLXArray.zeros([t, 1], type: Int32.self)], axis: 1)
        let mask = concatenated([MLXArray.ones([t, c], type: Float.self),
                                 MLXArray.zeros([t, 1], type: Float.self)], axis: 1)
        return (frame, mask)
    }

    /// Generate audio tokens for `text` in the reference voice (voice_match: the
    /// reference text and target text share one text segment over the ref audio).
    public func generateFrames(
        text: String, refAudio: MLXArray, refText: String, speaker: Int = 0,
        maxFrames: Int = 1024, temperature: Float = 0.9, topK: Int = 50
    ) -> MLXArray {
        let (tt, tm) = tokenizeText(refText + " " + text, speaker: speaker)
        let (at, am) = tokenizeAudio(refAudio, addEos: false)
        let tokens = concatenated([tt, at], axis: 0).expandedDimensions(axis: 0)
        let mask = concatenated([tm, am], axis: 0).expandedDimensions(axis: 0)
        return model.generate(promptTokens: tokens, promptMask: mask, maxFrames: maxFrames,
                              sampler: makeSampler(temperature: temperature, topK: topK))
    }

    /// Full text→audio: returns a 24 kHz mono waveform [samples].
    public func synthesize(
        text: String, refAudio: MLXArray, refText: String, speaker: Int = 0,
        maxFrames: Int = 1024, temperature: Float = 0.9, topK: Int = 50
    ) -> MLXArray {
        let frames = generateFrames(text: text, refAudio: refAudio, refText: refText,
                                    speaker: speaker, maxFrames: maxFrames,
                                    temperature: temperature, topK: topK)
        if frames.shape[2] == 0 { return MLXArray.zeros([0]) }
        return MimiStreamingDecoder(mimi).decodeFrames(frames).squeezed()
    }
}

/// Temperature + top-k categorical sampler (temperature 0 → greedy argmax).
/// Mirrors mlx_lm.sample_utils: mask all but the top-k logits to -inf, then
/// mx.random.categorical(logits * (1/temp)).
func makeSampler(temperature: Float, topK: Int) -> (MLXArray) -> MLXArray {
    { logits in
        if temperature <= 0 { return logits.argMax(axis: -1).asType(.int32) }
        var l = logits
        let vocab = l.shape[l.ndim - 1]
        if topK > 0 && topK < vocab {
            let sortedAsc = MLX.sorted(l, axis: -1)                             // ascending
            let kth = sortedAsc[0..., (vocab - topK)].expandedDimensions(axis: -1)   // k-th largest
            l = MLX.where(l .< kth, MLXArray(Float(-Double.infinity)), l)
        }
        return MLXRandom.categorical(l * (1.0 / temperature)).asType(.int32)
    }
}
