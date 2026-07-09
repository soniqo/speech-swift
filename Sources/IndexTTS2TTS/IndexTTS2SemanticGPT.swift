import AudioCommon
import Foundation
import MLX
import MLXFast
import MLXNN

public struct IndexTTS2SemanticGeneration {
    public let codes: [Int32]
    public let codeTensor: MLXArray
    public let conditioningLatent: MLXArray

    public var codeCount: Int { codes.count }
    public var codeShape: [Int] { codeTensor.shape }
    public var conditioningLatentShape: [Int] { conditioningLatent.shape }
}

public struct IndexTTS2SemanticGenerationOptions: Sendable {
    public var maxSemanticTokens: Int
    public var greedy: Bool
    public var temperature: Float
    public var topK: Int
    public var topP: Float
    public var repetitionPenalty: Float
    public var seed: UInt64
    public var beamWidth: Int
    public var lengthPenalty: Float

    public init(
        maxSemanticTokens: Int = 1_500,
        greedy: Bool = false,
        temperature: Float = 0.8,
        topK: Int = 30,
        topP: Float = 0.8,
        repetitionPenalty: Float = 10.0,
        seed: UInt64 = 11,
        beamWidth: Int = 3,
        lengthPenalty: Float = 0.0
    ) {
        self.maxSemanticTokens = maxSemanticTokens
        self.greedy = greedy
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.beamWidth = beamWidth
        self.lengthPenalty = lengthPenalty
    }
}

final class IndexTTS2SemanticGPT {
    private let weights: [String: MLXArray]
    private let config: IndexTTS2RuntimeConfig
    private let modelDim: Int
    private let heads: Int
    private let headDim: Int

    init(weights: [String: MLXArray], config: IndexTTS2RuntimeConfig) {
        self.weights = weights
        self.config = config
        self.modelDim = config.gpt.modelDim
        self.heads = config.gpt.heads
        self.headDim = config.gpt.modelDim / config.gpt.heads
    }

    func generateSemanticCodes(
        textTokens: [Int],
        conditioning: IndexTTS2ReferenceConditioning,
        options: IndexTTS2SemanticGenerationOptions = IndexTTS2SemanticGenerationOptions()
    ) throws -> IndexTTS2SemanticGeneration {
        guard !textTokens.isEmpty else {
            throw AudioModelError.inferenceFailed(
                operation: "IndexTTS2 semantic generation",
                reason: "Text tokenization produced no tokens.")
        }

        let speakerLatent = speakerConditioning(
            conditioning.speakerSemanticHidden,
            validFrames: conditioning.speakerSemanticHidden.dim(1))
        let baseEmotionVector = emotionVector(
            conditioning.emotionSemanticHidden,
            validFrames: conditioning.emotionSemanticHidden.dim(1))
        let emotionVector = resolvedEmotionVector(
            base: baseEmotionVector,
            conditioning: conditioning)
        let prefix = prefixConditioning(
            speakerLatent: speakerLatent,
            emotionVector: emotionVector)
        let prefixText = prefixTextEmbeddings(prefix: prefix, textTokens: textTokens)

        let generated: [Int32]
        if options.beamWidth > 1 {
            if options.greedy {
                generated = beamSearchSemanticCodes(prefixText: prefixText, options: options)
            } else {
                generated = beamSampleSemanticCodes(prefixText: prefixText, options: options)
            }
        } else {
            generated = sampleSemanticCodes(prefixText: prefixText, options: options)
        }

        let codes = MLXArray(generated, [1, generated.count])
        eval(codes, speakerLatent)
        return IndexTTS2SemanticGeneration(
            codes: generated,
            codeTensor: codes,
            conditioningLatent: speakerLatent)
    }

    private struct SemanticBeam {
        var tokens: [Int32]
        var score: Float
        var ended: Bool

        func rankingScore(lengthPenalty: Float) -> Float {
            guard lengthPenalty != 0 else { return score }
            let length = Float(max(tokens.count, 1))
            let divisor = Float(Foundation.pow(Double(length), Double(lengthPenalty)))
            return score / max(divisor, Float.ulpOfOne)
        }
    }

    private struct SemanticCandidate {
        var beamIndex: Int
        var token: Int32
        var score: Float
    }

    private func sampleSemanticCodes(
        prefixText: MLXArray,
        options: IndexTTS2SemanticGenerationOptions
    ) -> [Int32] {
        var generated: [Int32] = []
        generated.reserveCapacity(options.maxSemanticTokens)
        var rng = IndexTTS2SeededRNG(seed: options.seed)

        for _ in 0..<options.maxSemanticTokens {
            let melInput = [Int32(config.gpt.startMelToken)] + generated
            let logits = nextMelLogits(prefixText: prefixText, melInput: melInput)
            let next = sampleToken(logits, previousTokens: generated, options: options, rng: &rng)
            if next == Int32(config.gpt.stopMelToken) {
                break
            }
            guard next >= 0, next < Int32(config.semanticCodec.codebookSize) else {
                break
            }
            generated.append(next)
            if generated.count % 16 == 0 {
                eval(logits)
            }
        }
        return generated
    }

    private func beamSearchSemanticCodes(
        prefixText: MLXArray,
        options: IndexTTS2SemanticGenerationOptions
    ) -> [Int32] {
        let width = max(1, options.beamWidth)
        let stop = Int32(config.gpt.stopMelToken)
        var beams = [SemanticBeam(tokens: [], score: 0, ended: false)]

        for _ in 0..<options.maxSemanticTokens {
            var candidates: [SemanticBeam] = []
            candidates.reserveCapacity(width * width * 2)

            for beam in beams {
                if beam.ended {
                    candidates.append(beam)
                    continue
                }

                let melInput = [Int32(config.gpt.startMelToken)] + beam.tokens
                let logits = nextMelLogits(prefixText: prefixText, melInput: melInput)
                var values = logits.asType(.float32).asArray(Float.self)
                applyRepetitionPenalty(&values, previousTokens: beam.tokens, penalty: options.repetitionPenalty)
                let logProbs = Self.logSoftmax(values)
                for (token, logProb) in Self.topLogProbs(logProbs, count: width * 2) {
                    if token == stop {
                        candidates.append(SemanticBeam(
                            tokens: beam.tokens,
                            score: beam.score + logProb,
                            ended: true))
                    } else if token >= 0, token < Int32(config.semanticCodec.codebookSize) {
                        candidates.append(SemanticBeam(
                            tokens: beam.tokens + [token],
                            score: beam.score + logProb,
                            ended: false))
                    }
                }
                if beam.tokens.count % 16 == 0 {
                    eval(logits)
                }
            }

            beams = candidates
                .sorted { lhs, rhs in
                    let lhsScore = lhs.rankingScore(lengthPenalty: options.lengthPenalty)
                    let rhsScore = rhs.rankingScore(lengthPenalty: options.lengthPenalty)
                    if lhsScore != rhsScore { return lhsScore > rhsScore }
                    return lhs.tokens.count > rhs.tokens.count
                }
                .prefix(width)
                .map { $0 }
            if beams.allSatisfy(\.ended) {
                break
            }
        }

        return beams
            .sorted {
                $0.rankingScore(lengthPenalty: options.lengthPenalty) >
                    $1.rankingScore(lengthPenalty: options.lengthPenalty)
            }
            .first?
            .tokens ?? []
    }

    private func beamSampleSemanticCodes(
        prefixText: MLXArray,
        options: IndexTTS2SemanticGenerationOptions
    ) -> [Int32] {
        let width = max(1, options.beamWidth)
        let stop = Int32(config.gpt.stopMelToken)
        let samplesPerStep = max(2, width * 2)
        var rng = IndexTTS2SeededRNG(seed: options.seed)
        var active = [SemanticBeam(tokens: [], score: 0, ended: false)]
        var completed: [SemanticBeam] = []

        for _ in 0..<options.maxSemanticTokens {
            var candidates: [SemanticCandidate] = []

            for (beamIndex, beam) in active.enumerated() {
                let melInput = [Int32(config.gpt.startMelToken)] + beam.tokens
                let logits = nextMelLogits(prefixText: prefixText, melInput: melInput)
                var scores = Self.logSoftmax(logits.asType(.float32).asArray(Float.self))
                applyRepetitionPenalty(&scores, previousTokens: beam.tokens, penalty: options.repetitionPenalty)
                applyTemperature(&scores, temperature: options.temperature)
                applyTopK(&scores, k: options.topK, minTokensToKeep: 2)
                applyTopP(&scores, p: options.topP, minTokensToKeep: 2)

                for token in scores.indices where scores[token].isFinite {
                    candidates.append(SemanticCandidate(
                        beamIndex: beamIndex,
                        token: Int32(token),
                        score: beam.score + scores[token]))
                }
                if beam.tokens.count % 16 == 0 {
                    eval(logits)
                }
            }

            guard !candidates.isEmpty else { break }

            let sampledIndices = Self.sampleIndicesWithoutReplacement(
                scores: candidates.map { $0.score },
                count: samplesPerStep,
                rng: &rng)
            var sampled = sampledIndices.map { candidates[$0] }
            if sampled.count < samplesPerStep {
                let sampledSet = Set(sampledIndices)
                sampled += candidates.indices
                    .filter { !sampledSet.contains($0) }
                    .sorted { candidates[$0].score > candidates[$1].score }
                    .prefix(samplesPerStep - sampled.count)
                    .map { candidates[$0] }
            }
            sampled.sort { $0.score > $1.score }

            var nextActive: [SemanticBeam] = []
            nextActive.reserveCapacity(width)
            for (rank, candidate) in sampled.enumerated() {
                let source = active[candidate.beamIndex]
                if candidate.token == stop {
                    if rank < width {
                        completed.append(SemanticBeam(
                            tokens: source.tokens,
                            score: candidate.score,
                            ended: true))
                    }
                    continue
                }
                guard candidate.token >= 0, candidate.token < Int32(config.semanticCodec.codebookSize) else {
                    continue
                }
                nextActive.append(SemanticBeam(
                    tokens: source.tokens + [candidate.token],
                    score: candidate.score,
                    ended: false))
                if nextActive.count == width {
                    break
                }
            }

            active = nextActive
            completed = bestBeams(completed, count: width, lengthPenalty: options.lengthPenalty)
            if active.isEmpty || isBeamSampleDone(
                active: active,
                completed: completed,
                width: width,
                lengthPenalty: options.lengthPenalty
            ) {
                break
            }
        }

        let final = bestBeams(
            completed + active,
            count: 1,
            lengthPenalty: options.lengthPenalty)
        return final.first?.tokens ?? []
    }

    private func bestBeams(
        _ beams: [SemanticBeam],
        count: Int,
        lengthPenalty: Float
    ) -> [SemanticBeam] {
        beams
            .sorted {
                let lhs = $0.rankingScore(lengthPenalty: lengthPenalty)
                let rhs = $1.rankingScore(lengthPenalty: lengthPenalty)
                if lhs != rhs { return lhs > rhs }
                return $0.tokens.count > $1.tokens.count
            }
            .prefix(count)
            .map { $0 }
    }

    private func isBeamSampleDone(
        active: [SemanticBeam],
        completed: [SemanticBeam],
        width: Int,
        lengthPenalty: Float
    ) -> Bool {
        guard completed.count >= width else { return false }
        guard let worstCompleted = bestBeams(
            completed,
            count: width,
            lengthPenalty: lengthPenalty
        ).last else {
            return false
        }
        let bestActive = active
            .map { $0.rankingScore(lengthPenalty: lengthPenalty) }
            .max() ?? -.infinity
        return worstCompleted.rankingScore(lengthPenalty: lengthPenalty) >= bestActive
    }

    func semanticGeneration(
        codes: [Int32],
        conditioning: IndexTTS2ReferenceConditioning
    ) -> IndexTTS2SemanticGeneration {
        let speakerLatent = speakerConditioning(
            conditioning.speakerSemanticHidden,
            validFrames: conditioning.speakerSemanticHidden.dim(1))
        let codeTensor = MLXArray(codes, [1, codes.count])
        eval(codeTensor, speakerLatent)
        return IndexTTS2SemanticGeneration(
            codes: codes,
            codeTensor: codeTensor,
            conditioningLatent: speakerLatent)
    }

    func latentForS2Mel(
        textTokens: [Int],
        generatedCodes: MLXArray,
        conditioningLatent: MLXArray,
        emotionHidden: MLXArray,
        emotionVectorOverride: MLXArray? = nil,
        emotionVectorOverrideWeightSum: Float = 0
    ) -> MLXArray {
        let baseEmotion = emotionVector(emotionHidden, validFrames: emotionHidden.dim(1))
        let emotion = resolvedEmotionVector(
            base: baseEmotion,
            conditioningOverride: emotionVectorOverride,
            overrideWeightSum: emotionVectorOverrideWeightSum)
        let prefix = prefixConditioning(
            speakerLatent: conditioningLatent,
            emotionVector: emotion)

        let textInput = [Int32(config.gpt.startTextToken)]
            + textTokens.map(Int32.init)
            + [Int32(config.gpt.stopTextToken)]
        let melInput = [Int32(config.gpt.startMelToken)]
            + generatedCodes.asArray(Int32.self)
            + [Int32(config.gpt.stopMelToken)]

        let textEmb = textEmbeddings(textInput)
        let melEmb = melEmbeddings(melInput)
        let hidden = gpt(concatenated([prefix, textEmb, melEmb], axis: 1))
        let afterPrefix = hidden[0..., prefix.dim(1)..<hidden.dim(1), 0...]
        let melHidden = afterPrefix[0..., textEmb.dim(1)..<afterPrefix.dim(1), 0...]
        let normalized = layerNorm(melHidden, prefix: "final_norm")
        return normalized[0..., 0..<max(generatedCodes.dim(1), 0), 0...]
    }

    // MARK: - Conditioning

    private func speakerConditioning(_ hidden: MLXArray, validFrames: Int) -> MLXArray {
        let (encoded, mask) = conformerEncode(
            hidden,
            prefix: "conditioning_encoder",
            blocks: config.gpt.conditionBlocks,
            heads: 8,
            feedForwardHidden: 2_048,
            validFrames: validFrames)
        return perceiver(
            encoded,
            mask: mask,
            prefix: "perceiver_encoder",
            dim: modelDim,
            contextDim: 512,
            latents: 32,
            heads: 8,
            ffInner: 1_706)
    }

    private func emotionVector(_ hidden: MLXArray, validFrames: Int) -> MLXArray {
        let (encoded, mask) = conformerEncode(
            hidden,
            prefix: "emo_conditioning_encoder",
            blocks: config.gpt.emotionConditionBlocks,
            heads: 4,
            feedForwardHidden: 1_024,
            validFrames: validFrames)
        let latent = perceiver(
            encoded,
            mask: mask,
            prefix: "emo_perceiver_encoder",
            dim: 1_024,
            contextDim: 512,
            latents: 1,
            heads: 4,
            ffInner: 1_365)
        let squeezed = latent.squeezed(axis: 1)
        return linear(linear(squeezed, prefix: "emovec_layer"), prefix: "emo_layer")
    }

    private func resolvedEmotionVector(
        base: MLXArray,
        conditioning: IndexTTS2ReferenceConditioning
    ) -> MLXArray {
        resolvedEmotionVector(
            base: base,
            conditioningOverride: conditioning.emotionVectorOverride,
            overrideWeightSum: conditioning.emotionVectorOverrideWeightSum)
    }

    private func resolvedEmotionVector(
        base: MLXArray,
        conditioningOverride: MLXArray?,
        overrideWeightSum: Float
    ) -> MLXArray {
        guard let conditioningOverride else { return base }
        let residual = max(0, 1 - overrideWeightSum)
        let blended = conditioningOverride.asType(base.dtype) +
            MLXArray(residual).asType(base.dtype) * base
        eval(blended)
        return blended
    }

    private func prefixConditioning(speakerLatent: MLXArray, emotionVector: MLXArray) -> MLXArray {
        let speed = embed([0], key: "speed_emb.weight").squeezed(axis: 0)
        let speedHalf = embed([1], key: "speed_emb.weight").squeezed(axis: 0)
        let conditioned = speakerLatent + emotionVector.expandedDimensions(axis: 1)
        return concatenated([
            conditioned,
            speedHalf.expandedDimensions(axis: 1),
            speed.expandedDimensions(axis: 1),
        ], axis: 1)
    }

    private func prefixTextEmbeddings(prefix: MLXArray, textTokens: [Int]) -> MLXArray {
        let textInput = [Int32(config.gpt.startTextToken)]
            + textTokens.map(Int32.init)
            + [Int32(config.gpt.stopTextToken)]
        return concatenated([prefix, textEmbeddings(textInput)], axis: 1)
    }

    private func conformerEncode(
        _ hidden: MLXArray,
        prefix: String,
        blocks: Int,
        heads: Int,
        feedForwardHidden: Int,
        validFrames: Int
    ) -> (MLXArray, MLXArray) {
        var h = conv2dSubsample(hidden, prefix: "\(prefix).embed")
        let t = h.dim(1)
        let scale = MLXArray(Foundation.sqrt(Float(512))).asType(h.dtype)
        h = h * scale
        let pos = weights["\(prefix).embed.pos_enc.pe"]![0..., 0..<t, 0...].asType(h.dtype)
        let mask = MLXArray([Int32](repeating: 1, count: t), [1, 1, t])

        for i in 0..<blocks {
            h = conformerLayer(
                h,
                pos: pos,
                mask: mask,
                prefix: "\(prefix).encoders.\(i)",
                heads: heads,
                feedForwardHidden: feedForwardHidden)
        }

        h = layerNorm(h, prefix: "\(prefix).after_norm")
        eval(h, mask)
        return (h, mask)
    }

    private func conv2dSubsample(_ x: MLXArray, prefix: String) -> MLXArray {
        let weight = weights["\(prefix).conv.0.weight"]!
        let bias = weights["\(prefix).conv.0.bias"]!
        var h = x.expandedDimensions(axis: 1) // [B, 1, T, C]
        h = conv2dNCHW(
            h,
            weight: weight,
            bias: bias,
            stride: (2, 2),
            padding: (0, 0))
        h = MLXNN.relu(h)
        let b = h.dim(0)
        let c = h.dim(1)
        let t = h.dim(2)
        let f = h.dim(3)
        h = h.transposed(0, 2, 1, 3).reshaped([b, t, c * f])
        return linear(h, prefix: "\(prefix).out.0")
    }

    private func conformerLayer(
        _ x: MLXArray,
        pos: MLXArray,
        mask: MLXArray,
        prefix: String,
        heads: Int,
        feedForwardHidden: Int
    ) -> MLXArray {
        var h = x + relPosAttention(
            layerNorm(x, prefix: "\(prefix).norm_mha"),
            pos: pos,
            mask: mask,
            prefix: "\(prefix).self_attn",
            heads: heads)

        h = h + convolutionModule(
            layerNorm(h, prefix: "\(prefix).norm_conv"),
            mask: mask,
            prefix: "\(prefix).conv_module")

        h = h + feedForward(
            layerNorm(h, prefix: "\(prefix).norm_ff"),
            prefix: "\(prefix).feed_forward")
        return layerNorm(h, prefix: "\(prefix).norm_final")
    }

    private func relPosAttention(
        _ x: MLXArray,
        pos: MLXArray,
        mask: MLXArray,
        prefix: String,
        heads: Int
    ) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let d = x.dim(2)
        let dk = d / heads

        let q = linear(x, prefix: "\(prefix).linear_q")
            .reshaped([b, t, heads, dk])
        let k = linear(x, prefix: "\(prefix).linear_k")
            .reshaped([b, t, heads, dk])
            .transposed(0, 2, 1, 3)
        let v = linear(x, prefix: "\(prefix).linear_v")
            .reshaped([b, t, heads, dk])
            .transposed(0, 2, 1, 3)
        let p = linear(pos, prefix: "\(prefix).linear_pos", bias: false)
            .reshaped([pos.dim(0), pos.dim(1), heads, dk])
            .transposed(0, 2, 1, 3)

        let qU = (q + weights["\(prefix).pos_bias_u"]!.asType(x.dtype))
            .transposed(0, 2, 1, 3)
        let qV = (q + weights["\(prefix).pos_bias_v"]!.asType(x.dtype))
            .transposed(0, 2, 1, 3)
        let scale = MLXArray(1.0 / Foundation.sqrt(Float(dk))).asType(x.dtype)
        var scores = (matmul(qU, k.transposed(0, 1, 3, 2)) +
            matmul(qV, p.transposed(0, 1, 3, 2))) * scale

        let m = mask.expandedDimensions(axis: 1)
        let neg = MLXArray(-Float.greatestFiniteMagnitude).asType(scores.dtype)
        scores = MLX.where(m .== MLXArray(Int32(0)), neg, scores)
        var attn = softmax(scores.asType(.float32), axis: -1).asType(x.dtype)
        attn = MLX.where(m .== MLXArray(Int32(0)), MLXArray(Float(0)).asType(attn.dtype), attn)

        let out = matmul(attn, v)
            .transposed(0, 2, 1, 3)
            .reshaped([b, t, d])
        return linear(out, prefix: "\(prefix).linear_out")
    }

    private func convolutionModule(_ x: MLXArray, mask: MLXArray, prefix: String) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        let nclMask = mask.asType(h.dtype)
        h = h * nclMask
        h = conv1dNCL(
            h,
            weight: weights["\(prefix).pointwise_conv1.weight"]!,
            bias: weights["\(prefix).pointwise_conv1.bias"]!)
        let parts = split(h, parts: 2, axis: 1)
        h = parts[0] * sigmoid(parts[1])
        h = conv1dNCL(
            h,
            weight: weights["\(prefix).depthwise_conv.weight"]!,
            bias: weights["\(prefix).depthwise_conv.bias"]!,
            padding: 7,
            groups: 512)
        h = layerNorm(
            h.transposed(0, 2, 1),
            prefix: "\(prefix).norm")
            .transposed(0, 2, 1)
        h = silu(h)
        h = conv1dNCL(
            h,
            weight: weights["\(prefix).pointwise_conv2.weight"]!,
            bias: weights["\(prefix).pointwise_conv2.bias"]!)
        h = h * nclMask
        return h.transposed(0, 2, 1)
    }

    private func feedForward(_ x: MLXArray, prefix: String) -> MLXArray {
        linear(silu(linear(x, prefix: "\(prefix).w_1")), prefix: "\(prefix).w_2")
    }

    private func perceiver(
        _ x: MLXArray,
        mask: MLXArray,
        prefix: String,
        dim: Int,
        contextDim: Int,
        latents: Int,
        heads: Int,
        ffInner: Int
    ) -> MLXArray {
        let batch = x.dim(0)
        let context = linear(x, prefix: "\(prefix).proj_context")
        var h = broadcast(
            weights["\(prefix).latents"]!.expandedDimensions(axis: 0).asType(context.dtype),
            to: [batch, latents, dim])

        for i in 0..<2 {
            h = h + perceiverAttention(
                h,
                context: context,
                prefix: "\(prefix).layers.\(i).0",
                heads: heads)
            h = h + perceiverFeedForward(
                h,
                prefix: "\(prefix).layers.\(i).1",
                ffInner: ffInner)
        }

        return perceiverRMSNorm(h, prefix: "\(prefix).norm")
    }

    private func perceiverAttention(
        _ x: MLXArray,
        context: MLXArray,
        prefix: String,
        heads: Int
    ) -> MLXArray {
        let b = x.dim(0)
        let qLen = x.dim(1)
        let dimHead = 64
        let fullContext = concatenated([x, context], axis: 1)
        let q = linear(x, prefix: "\(prefix).to_q", bias: false)
            .reshaped([b, qLen, heads, dimHead])
            .transposed(0, 2, 1, 3)
        let kv = linear(fullContext, prefix: "\(prefix).to_kv", bias: false)
        let parts = split(kv, parts: 2, axis: -1)
        let contextLen = fullContext.dim(1)
        let k = parts[0].reshaped([b, contextLen, heads, dimHead]).transposed(0, 2, 1, 3)
        let v = parts[1].reshaped([b, contextLen, heads, dimHead]).transposed(0, 2, 1, 3)
        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / Foundation.sqrt(Float(dimHead)),
            mask: nil)
        let out = attn.transposed(0, 2, 1, 3).reshaped([b, qLen, heads * dimHead])
        return linear(out, prefix: "\(prefix).to_out", bias: false)
    }

    private func perceiverFeedForward(_ x: MLXArray, prefix: String, ffInner: Int) -> MLXArray {
        let projected = linear(x, prefix: "\(prefix).0")
        let parts = split(projected, parts: 2, axis: -1)
        let activated = geluApproximate(parts[1]) * parts[0]
        return linear(activated, prefix: "\(prefix).2")
    }

    private func perceiverRMSNorm(_ x: MLXArray, prefix: String) -> MLXArray {
        let norm = sqrt((x * x).sum(axis: -1, keepDims: true) + MLXArray(Float(1e-12)))
        let scale = MLXArray(Foundation.sqrt(Float(x.dim(-1)))).asType(x.dtype)
        let gamma = weights["\(prefix).gamma"]!.asType(x.dtype)
        return x / norm * scale * gamma
    }

    // MARK: - GPT

    private func nextMelLogits(prefixText: MLXArray, melInput: [Int32]) -> MLXArray {
        let mel = melGenerationEmbeddings(melInput)
        let hidden = gpt(concatenated([prefixText, mel], axis: 1))
        let last = hidden[0..., (hidden.dim(1) - 1)..<hidden.dim(1), 0...]
        let normalized = layerNorm(last, prefix: "final_norm")
        return linear(normalized, prefix: "mel_head").squeezed(axis: 1)
    }

    private func gpt(_ embeddings: MLXArray) -> MLXArray {
        var h = embeddings
        for i in 0..<config.gpt.layers {
            h = gptBlock(h, prefix: "gpt.h.\(i)")
        }
        return layerNorm(h, prefix: "gpt.ln_f")
    }

    private func gptBlock(_ x: MLXArray, prefix: String) -> MLXArray {
        var h = x + gptAttention(layerNorm(x, prefix: "\(prefix).ln_1"), prefix: "\(prefix).attn")
        h = h + gptMLP(layerNorm(h, prefix: "\(prefix).ln_2"), prefix: "\(prefix).mlp")
        return h
    }

    private func gptAttention(_ x: MLXArray, prefix: String) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let qkv = conv1DLinear(x, prefix: "\(prefix).c_attn")
        let parts = split(qkv, parts: 3, axis: -1)
        let q = parts[0].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let k = parts[1].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let causal = causalMask(length: t, dtype: x.dtype)
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / Foundation.sqrt(Float(headDim)),
            mask: causal)
        let merged = out.transposed(0, 2, 1, 3).reshaped([b, t, modelDim])
        return conv1DLinear(merged, prefix: "\(prefix).c_proj")
    }

    private func gptMLP(_ x: MLXArray, prefix: String) -> MLXArray {
        conv1DLinear(geluApproximate(conv1DLinear(x, prefix: "\(prefix).c_fc")), prefix: "\(prefix).c_proj")
    }

    private func textEmbeddings(_ ids: [Int32]) -> MLXArray {
        let token = embed(ids, key: "text_embedding.weight")
        let pos = positionEmbeddings(count: ids.count, key: "text_pos_embedding.emb.weight", dtype: token.dtype)
        return token + pos
    }

    private func melEmbeddings(_ ids: [Int32]) -> MLXArray {
        let token = embed(ids, key: "mel_embedding.weight")
        let pos = positionEmbeddings(count: ids.count, key: "mel_pos_embedding.emb.weight", dtype: token.dtype)
        return token + pos
    }

    private func melGenerationEmbeddings(_ ids: [Int32]) -> MLXArray {
        let token = embed(ids, key: "mel_embedding.weight")
        guard ids.count > 1 else {
            let pos = weights["mel_pos_embedding.emb.weight"]!
                .take(MLXArray([Int32(0)], [1]), axis: 0)
                .expandedDimensions(axis: 0)
                .asType(token.dtype)
            return token + pos
        }

        var positions = [Int32(0)]
        positions.reserveCapacity(ids.count)
        for i in 1..<ids.count {
            positions.append(Int32(i + 1))
        }
        let pos = weights["mel_pos_embedding.emb.weight"]!
            .take(MLXArray(positions, [positions.count]), axis: 0)
            .expandedDimensions(axis: 0)
            .asType(token.dtype)
        return token + pos
    }

    private func positionEmbeddings(count: Int, key: String, dtype: DType) -> MLXArray {
        let ids = MLXArray((0..<count).map(Int32.init), [count])
        return weights[key]!.take(ids, axis: 0).expandedDimensions(axis: 0).asType(dtype)
    }

    private func greedyToken(_ logits: MLXArray) -> Int32 {
        let token = argMax(logits.asType(.float32), axis: -1).asType(.int32)
        eval(token)
        return token.asArray(Int32.self)[0]
    }

    private func sampleToken(
        _ logits: MLXArray,
        previousTokens: [Int32],
        options: IndexTTS2SemanticGenerationOptions,
        rng: inout IndexTTS2SeededRNG
    ) -> Int32 {
        var values = logits.asType(.float32).asArray(Float.self)
        applyRepetitionPenalty(&values, previousTokens: previousTokens, penalty: options.repetitionPenalty)

        if options.greedy || options.temperature <= 1e-4 {
            var bestValue: Float = -.infinity
            var bestIndex = 0
            for i in values.indices where values[i] > bestValue {
                bestValue = values[i]
                bestIndex = i
            }
            return Int32(bestIndex)
        }

        applyTopK(&values, k: options.topK)
        applyTopP(&values, p: options.topP, temperature: options.temperature)

        let temperature = max(options.temperature, 1e-4)
        var bestScore: Float = -.infinity
        var bestIndex = 0
        for i in values.indices where values[i].isFinite {
            let u = max(Float.ulpOfOne, Float(Double.random(in: Double(Float.ulpOfOne)..<1.0, using: &rng)))
            let gumbel = -log(-log(u))
            let score = values[i] / temperature + gumbel
            if score > bestScore {
                bestScore = score
                bestIndex = i
            }
        }
        return Int32(bestIndex)
    }

    private func applyRepetitionPenalty(_ logits: inout [Float], previousTokens: [Int32], penalty: Float) {
        guard penalty > 1.0 else { return }
        for token in Set(previousTokens) {
            let index = Int(token)
            guard index >= 0, index < logits.count else { continue }
            if logits[index] < 0 {
                logits[index] *= penalty
            } else {
                logits[index] /= penalty
            }
        }
    }

    private func applyTemperature(_ logits: inout [Float], temperature: Float) {
        guard temperature > 1e-4, temperature != 1.0 else { return }
        for i in logits.indices where logits[i].isFinite {
            logits[i] /= temperature
        }
    }

    private func applyTopK(_ logits: inout [Float], k: Int, minTokensToKeep: Int = 1) {
        let keep = max(k, minTokensToKeep)
        guard keep > 0, keep < logits.count else { return }
        let threshold = logits.sorted(by: >)[keep - 1]
        for i in logits.indices where logits[i] < threshold {
            logits[i] = -.infinity
        }
    }

    private func applyTopP(_ logits: inout [Float], p: Float, temperature: Float) {
        var scaled = logits
        applyTemperature(&scaled, temperature: temperature)
        applyTopP(&scaled, p: p, minTokensToKeep: 1)
        for i in logits.indices where !scaled[i].isFinite {
            logits[i] = -.infinity
        }
    }

    private func applyTopP(_ logits: inout [Float], p: Float, minTokensToKeep: Int) {
        guard p > 0, p < 1 else { return }
        let finite = logits.indices.filter { logits[$0].isFinite }
        guard finite.count > 1 else { return }

        let maxValue = finite.map { logits[$0] }.max() ?? 0
        let scored = finite
            .map { index -> (index: Int, prob: Float) in
                (index, exp(logits[index] - maxValue))
            }
            .sorted { $0.prob > $1.prob }
        let total = scored.reduce(Float(0)) { $0 + $1.prob }
        guard total > 0 else { return }

        var allowed = Set<Int>()
        var cumulative: Float = 0
        var kept = 0
        for item in scored {
            allowed.insert(item.index)
            kept += 1
            cumulative += item.prob / total
            if cumulative >= p && kept >= minTokensToKeep {
                break
            }
        }
        for i in logits.indices where logits[i].isFinite && !allowed.contains(i) {
            logits[i] = -.infinity
        }
    }

    private static func logSoftmax(_ values: [Float]) -> [Float] {
        let finite = values.filter(\.isFinite)
        guard let maxValue = finite.max() else {
            return values.map { _ in -.infinity }
        }
        let total = finite.reduce(Float(0)) { $0 + exp($1 - maxValue) }
        guard total > 0 else {
            return values.map { _ in -.infinity }
        }
        let logTotal = log(total)
        return values.map { value in
            value.isFinite ? value - maxValue - logTotal : -.infinity
        }
    }

    private static func topLogProbs(_ values: [Float], count: Int) -> [(Int32, Float)] {
        values.indices
            .filter { values[$0].isFinite }
            .sorted { values[$0] > values[$1] }
            .prefix(count)
            .map { (Int32($0), values[$0]) }
    }

    private static func sampleIndicesWithoutReplacement(
        scores: [Float],
        count: Int,
        rng: inout IndexTTS2SeededRNG
    ) -> [Int] {
        guard count > 0 else { return [] }
        let finite = scores.indices.filter { scores[$0].isFinite }
        guard let maxScore = finite.map({ scores[$0] }).max() else { return [] }

        var weights = Array(repeating: Float(0), count: scores.count)
        var total = Float(0)
        for index in finite {
            let weight = exp(scores[index] - maxScore)
            weights[index] = weight
            total += weight
        }

        var selected: [Int] = []
        selected.reserveCapacity(min(count, finite.count))
        for _ in 0..<min(count, finite.count) {
            guard total > 0, total.isFinite else { break }
            let draw = Float(Double.random(in: 0..<Double(total), using: &rng))
            var cumulative = Float(0)
            var chosen = finite.last
            for index in finite where weights[index] > 0 {
                cumulative += weights[index]
                if cumulative >= draw {
                    chosen = index
                    break
                }
            }
            guard let chosen else { break }
            selected.append(chosen)
            total -= weights[chosen]
            weights[chosen] = 0
        }
        return selected
    }

    private func causalMask(length: Int, dtype: DType) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let keep = MLXArray.tri(length, type: Float.self)
        let neg = MLXArray(-Float.greatestFiniteMagnitude)
        let additive = MLX.where(keep .== MLXArray(Float(0)), neg, MLXArray(Float(0)))
        return .array(additive.reshaped([1, 1, length, length]).asType(dtype))
    }

    // MARK: - Tensor primitives

    private func linear(_ x: MLXArray, prefix: String, bias: Bool = true) -> MLXArray {
        let weight = weights["\(prefix).weight"]!.asType(x.dtype)
        var y = matmul(x, weight.transposed(1, 0))
        if bias, let b = weights["\(prefix).bias"] {
            y = y + b.asType(y.dtype)
        }
        return y
    }

    private func conv1DLinear(_ x: MLXArray, prefix: String) -> MLXArray {
        let weight = weights["\(prefix).weight"]!.asType(x.dtype)
        var y = matmul(x, weight)
        if let b = weights["\(prefix).bias"] {
            y = y + b.asType(y.dtype)
        }
        return y
    }

    private func layerNorm(_ x: MLXArray, prefix: String, eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        var y = centered / sqrt(variance + MLXArray(eps).asType(x.dtype))
        if let weight = weights["\(prefix).weight"] {
            y = y * weight.asType(y.dtype)
        }
        if let bias = weights["\(prefix).bias"] {
            y = y + bias.asType(y.dtype)
        }
        return y
    }

    private func embed(_ ids: [Int32], key: String) -> MLXArray {
        weights[key]!.take(MLXArray(ids, [ids.count]), axis: 0).expandedDimensions(axis: 0)
    }

    private func conv1dNCL(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray? = nil,
        padding: Int = 0,
        groups: Int = 1
    ) -> MLXArray {
        let weightNLC = weight.asType(x.dtype).transposed(0, 2, 1)
        var y = MLX.conv1d(
            x.transposed(0, 2, 1),
            weightNLC,
            stride: 1,
            padding: padding,
            groups: groups)
            .transposed(0, 2, 1)
        if let bias {
            y = y + bias.asType(y.dtype).reshaped([1, bias.dim(0), 1])
        }
        return y
    }

    private func conv2dNCHW(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        stride: (Int, Int),
        padding: (Int, Int)
    ) -> MLXArray {
        let weightNHWC = weight.asType(x.dtype).transposed(0, 2, 3, 1)
        var y = MLX.conv2d(
            x.transposed(0, 2, 3, 1),
            weightNHWC,
            stride: .init(stride),
            padding: .init(padding))
            .transposed(0, 3, 1, 2)
        y = y + bias.asType(y.dtype).reshaped([1, bias.dim(0), 1, 1])
        return y
    }
}

private struct IndexTTS2SeededRNG: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
