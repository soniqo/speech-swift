import Foundation
import MLX
import MLXNN

/// Wraps the Llama backbone under `tfmr.model.*` to match the checkpoint keys.
final class T3Tfmr: Module {
    @ModuleInfo(key: "model") var model: T3Backbone
    init(_ cfg: T3Config) { _model.wrappedValue = T3Backbone(cfg) }
}

/// T3: text → speech tokens. A Llama backbone conditioned on a speaker embedding
/// (+ optional prompt-speech + emotion), generated autoregressively with CFG.
public final class ChatterboxT3: Module {
    @ModuleInfo(key: "tfmr") var tfmr: T3Tfmr
    @ModuleInfo(key: "cond_enc") var condEnc: T3CondEnc
    @ModuleInfo(key: "text_emb") var textEmb: Embedding
    @ModuleInfo(key: "speech_emb") var speechEmb: Embedding
    @ModuleInfo(key: "text_pos_emb") var textPosEmb: LearnedPositionEmbeddings
    @ModuleInfo(key: "speech_pos_emb") var speechPosEmb: LearnedPositionEmbeddings
    @ModuleInfo(key: "text_head") var textHead: Linear
    @ModuleInfo(key: "speech_head") var speechHead: Linear

    let cfg: T3Config

    public init(_ cfg: T3Config = T3Config()) {
        self.cfg = cfg
        _tfmr.wrappedValue = T3Tfmr(cfg)
        _condEnc.wrappedValue = T3CondEnc(cfg)
        _textEmb.wrappedValue = Embedding(embeddingCount: cfg.textTokensDictSize, dimensions: cfg.hiddenSize)
        _speechEmb.wrappedValue = Embedding(embeddingCount: cfg.speechTokensDictSize, dimensions: cfg.hiddenSize)
        _textPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: cfg.maxTextTokens + 2, dim: cfg.hiddenSize)
        _speechPosEmb.wrappedValue = LearnedPositionEmbeddings(seqLen: cfg.maxSpeechTokens + 4, dim: cfg.hiddenSize)
        _textHead.wrappedValue = Linear(cfg.hiddenSize, cfg.textTokensDictSize, bias: false)
        _speechHead.wrappedValue = Linear(cfg.hiddenSize, cfg.speechTokensDictSize, bias: false)
        super.init()
    }

    /// Build the prefill embeddings `[cond | text | bos]` (batch doubled for CFG).
    private func buildPrefix(
        textTokens: [Int], speakerEmb: MLXArray, promptSpeechTokens: [Int]?,
        emotionAdv: Float, useCFG: Bool
    ) -> MLXArray {
        let spk = speakerEmb.reshaped([1, cfg.speakerEmbedSize])
        var promptEmb: MLXArray?
        if let pst = promptSpeechTokens, !pst.isEmpty {
            let t = MLXArray(pst.map { Int32($0) }).reshaped([1, pst.count])
            promptEmb = speechEmb(t) + speechPosEmb(t)
        }
        let condEmb = condEnc(speakerEmb: spk, promptSpeechEmb: promptEmb, emotionAdv: emotionAdv)

        let textArr = MLXArray(textTokens.map { Int32($0) }).reshaped([1, textTokens.count])
        var textE = textEmb(textArr) + textPosEmb(textArr)
        var bosE = speechEmb(MLXArray([Int32(cfg.startSpeechToken)]).reshaped([1, 1])) + speechPosEmb.fixed(0)
        var condB = condEmb

        if useCFG {
            textE = concatenated([textE, MLXArray.zeros(textE.shape)], axis: 0)  // uncond text zeroed
            condB = broadcast(condEmb, to: [2] + Array(condEmb.shape[1...]))
            bosE = concatenated([bosE, bosE], axis: 0)
        }
        return concatenated([condB, textE, bosE], axis: 1)
    }

    /// Generate speech tokens for `textTokens`, conditioned on the speaker
    /// embedding (and optional prompt-speech tokens + emotion).
    public func inference(
        textTokens: [Int],
        speakerEmb: MLXArray,
        promptSpeechTokens: [Int]? = nil,
        emotionAdv: Float = 0.5,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.8,
        topP: Float = 0.95,
        minP: Float = 0.05,
        repetitionPenalty: Float = 1.2,
        cfgWeight: Float = 0.5
    ) -> [Int] {
        let useCFG = cfgWeight > 0.0
        let input = buildPrefix(
            textTokens: textTokens, speakerEmb: speakerEmb,
            promptSpeechTokens: promptSpeechTokens, emotionAdv: emotionAdv, useCFG: useCFG)
        let caches = (0 ..< cfg.numLayers).map { _ in T3KVCache() }

        var hidden = tfmr.model(input, caches: caches)
        eval(hidden)
        var generated: [Int] = []

        for step in 0 ..< maxNewTokens {
            let last = hidden.dim(1) - 1
            let logitsArr = speechHead(hidden[0..., last ..< (last + 1), 0...])  // [B,1,vocab]
            var logits = nextStepLogits(logitsArr, useCFG: useCFG, cfgWeight: cfgWeight)  // host [vocab]
            applyRepetitionPenalty(&logits, tokens: generated, penalty: repetitionPenalty)
            let next = sampleToken(logits, temperature: temperature, topP: topP, minP: minP)

            if next == cfg.stopSpeechToken { break }
            generated.append(next)

            var nextE = speechEmb(MLXArray([Int32(next)]).reshaped([1, 1])) + speechPosEmb.fixed(step + 1)
            if useCFG { nextE = concatenated([nextE, nextE], axis: 0) }
            hidden = tfmr.model(nextE, caches: caches)
            eval(hidden)
        }
        return generated
    }

    /// CFG-combine the last-position logits and return them on the host.
    private func nextStepLogits(_ logitsBV: MLXArray, useCFG: Bool, cfgWeight: Float) -> [Float] {
        let v = cfg.speechTokensDictSize
        let l = logitsBV.reshaped([logitsBV.dim(0), v])
        let combined: MLXArray
        if useCFG && l.dim(0) > 1 {
            let cond = l[0 ..< 1]
            let uncond = l[1 ..< 2]
            combined = cond + cfgWeight * (cond - uncond)
        } else {
            combined = l[0 ..< 1]
        }
        eval(combined)
        return combined.reshaped([v]).asArray(Float.self)
    }

    // Repetition penalty: divide positive logits, multiply negative.
    private func applyRepetitionPenalty(_ logits: inout [Float], tokens: [Int], penalty: Float) {
        guard penalty != 1.0 else { return }
        for t in Set(tokens) where t < logits.count {
            logits[t] = logits[t] > 0 ? logits[t] / penalty : logits[t] * penalty
        }
    }

    /// Temperature + min-p + top-p filtering, then categorical sample (host).
    private func sampleToken(_ logits: [Float], temperature: Float, topP: Float, minP: Float) -> Int {
        if temperature <= 0 {
            return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
        }
        // softmax(logits / temperature)
        let scaled = logits.map { $0 / temperature }
        let m = scaled.max() ?? 0
        var probs = scaled.map { Foundation.exp($0 - m) }
        let sum = probs.reduce(0, +)
        if sum > 0 { for i in probs.indices { probs[i] /= sum } }

        let maxP = probs.max() ?? 0
        // min-p: drop tokens below minP * maxP.
        if minP > 0 { for i in probs.indices where probs[i] < minP * maxP { probs[i] = 0 } }
        // top-p (nucleus): keep the smallest set whose cumulative prob ≥ topP.
        if topP < 1.0 {
            let order = probs.indices.sorted { probs[$0] > probs[$1] }
            var cum: Float = 0
            var keep = Set<Int>()
            for i in order { cum += probs[i]; keep.insert(i); if cum >= topP { break } }
            for i in probs.indices where !keep.contains(i) { probs[i] = 0 }
        }
        let total = probs.reduce(0, +)
        guard total > 0 else { return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0 }
        var r = Float.random(in: 0 ..< total)
        for (i, p) in probs.enumerated() {
            r -= p
            if r <= 0 { return i }
        }
        return probs.indices.max(by: { probs[$0] < probs[$1] }) ?? 0
    }
}
