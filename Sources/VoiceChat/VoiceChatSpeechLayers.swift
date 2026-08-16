import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

let voiceChatSpeechComponents = 1_024
let voiceChatSpeechLowRank = 64
let voiceChatSpeechMinimumLogStandardDeviation: Float = -4

struct VoiceChatSpeechWeightStore {
    let values: [String: MLXArray]
    let quantization: [String: QuantizationConfig]
    let defaultQuantization: QuantizationConfig?

    func dense(_ name: String) throws -> MLXArray {
        guard let value = values[name] else {
            throw VoiceChatLoadError.unexpectedKeys([name])
        }
        return value.asType(.float32)
    }

    func matrix(_ name: String, bias: String? = nil) throws -> VoiceChatMatrix {
        let prefix = String(name.dropLast(".weight".count))
        let spec = quantization[name]
            ?? (values["\(prefix).scales"] != nil ? defaultQuantization : nil)
        return try VoiceChatMatrix(
            weights: values, name: name, quantization: spec, biasName: bias)
    }
}

@inline(__always)
func voiceChatGemmaRMSNorm(
    _ input: MLXArray,
    weight: MLXArray,
    epsilon: Float = 1e-6
) -> MLXArray {
    let input32 = input.asType(.float32)
    let normalized = input32
        * MLX.rsqrt(MLX.mean(input32.square(), axis: -1, keepDims: true) + MLXArray(epsilon))
    return (normalized * (MLXArray(Float(1)) + weight.asType(.float32)))
        .asType(input.dtype)
}

func voiceChatSpeechRoPE(
    _ input: MLXArray,
    offset: Int,
    theta: Float,
    headDimension: Int
) -> MLXArray {
    let length = input.dim(-2)
    let half = headDimension / 2
    let positions = MLXArray(Int32(offset) ..< Int32(offset + length))
        .asType(.float32).expandedDimensions(axis: 1)
    let dimensions = MLXArray(0 ..< Int32(half)).asType(.float32)
    let inverse = MLX.exp(
        -MLX.log(MLXArray(theta)) * dimensions / MLXArray(Float(half)))
    let angles = positions * inverse.expandedDimensions(axis: 0)
    let cosine = MLX.concatenated([MLX.cos(angles), MLX.cos(angles)], axis: -1)
        .expandedDimensions(axes: [0, 1])
    let sine = MLX.concatenated([MLX.sin(angles), MLX.sin(angles)], axis: -1)
        .expandedDimensions(axes: [0, 1])
    let rotated = MLX.concatenated([
        -input[.ellipsis, half..<headDimension],
        input[.ellipsis, 0..<half],
    ], axis: -1)
    return input * cosine + rotated * sine
}

func voiceChatTopPFilter(_ logits: MLXArray, topP: Float) -> MLXArray {
    guard topP < 1 else { return logits }
    let order = MLX.argSort(-logits, axis: -1)
    let ordered = MLX.takeAlong(logits, order, axis: -1)
    let probabilities = MLX.softmax(ordered, axis: -1)
    let drop = MLX.cumsum(probabilities, axis: -1) - probabilities .> MLXArray(topP)
    let filtered = MLX.where(drop, MLXArray(Float(-1e9)), ordered)
    return MLX.takeAlong(filtered, MLX.argSort(order, axis: -1), axis: -1)
}

func voiceChatGumbelLike(_ input: MLXArray, epsilon: Float = 1e-8) -> MLXArray {
    let uniform = MLXRandom.uniform(
        low: Float(0), high: Float(1), input.shape)
    return -MLX.log(-MLX.log(uniform + MLXArray(epsilon)) + MLXArray(epsilon))
}

final class VoiceChatSpeechAttentionCache {
    var keys: MLXArray?
    var values: MLXArray?
    private(set) var offset = 0
    let retainedPrefixFrames: Int
    let recentContextFrames: Int?

    init(
        retainedPrefixFrames: Int = 0,
        recentContextFrames: Int? = nil
    ) {
        self.retainedPrefixFrames = retainedPrefixFrames
        self.recentContextFrames = recentContextFrames
    }

    /// Append projected keys/values and return the context visible to the
    /// current query. Bounded live caches keep the immutable speaker prompt
    /// plus a recent rolling window; `offset` remains absolute so RoPE does
    /// not reset when middle history is removed.
    func update(
        keys newKeys: MLXArray,
        values newValues: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let attentionKeys: MLXArray
        let attentionValues: MLXArray
        if let keys, let values {
            attentionKeys = MLX.concatenated([keys, newKeys], axis: 2)
            attentionValues = MLX.concatenated([values, newValues], axis: 2)
        } else {
            attentionKeys = newKeys
            attentionValues = newValues
        }
        offset += newKeys.dim(2)

        guard let recentContextFrames else {
            keys = attentionKeys
            values = attentionValues
            return (attentionKeys, attentionValues)
        }

        let length = attentionKeys.dim(2)
        let prefix = min(retainedPrefixFrames, length)
        let recentStart = max(prefix, length - recentContextFrames)
        if recentStart > prefix {
            let recentKeys = attentionKeys[.ellipsis, recentStart..., 0...]
            let recentValues = attentionValues[.ellipsis, recentStart..., 0...]
            if prefix > 0 {
                keys = MLX.concatenated([
                    attentionKeys[.ellipsis, 0 ..< prefix, 0...],
                    recentKeys,
                ], axis: 2)
                values = MLX.concatenated([
                    attentionValues[.ellipsis, 0 ..< prefix, 0...],
                    recentValues,
                ], axis: 2)
            } else {
                keys = recentKeys
                values = recentValues
            }
        } else {
            keys = attentionKeys
            values = attentionValues
        }
        return (attentionKeys, attentionValues)
    }
}

final class VoiceChatSpeechAttention {
    private let query: VoiceChatMatrix
    private let key: VoiceChatMatrix
    private let value: VoiceChatMatrix
    private let output: VoiceChatMatrix
    private let queryNorm: MLXArray?
    private let keyNorm: MLXArray?
    private let ropeTheta: Float
    private let softcap: Float?
    private let causal: Bool
    private let window: Int?
    private let heads: Int
    private let headDimension: Int

    init(
        _ store: VoiceChatSpeechWeightStore,
        prefix: String,
        heads: Int,
        headDimension: Int,
        ropeTheta: Float,
        softcap: Float? = nil,
        causal: Bool = true,
        window: Int? = nil,
        queryNormName: String = "q_norm"
    ) throws {
        self.query = try store.matrix("\(prefix).q_proj.weight")
        self.key = try store.matrix("\(prefix).k_proj.weight")
        self.value = try store.matrix("\(prefix).v_proj.weight")
        self.output = try store.matrix("\(prefix).o_proj.weight")
        let qName = "\(prefix).\(queryNormName).weight"
        let kName = "\(prefix).k_norm.weight"
        self.queryNorm = store.values[qName]?.asType(.float32)
        self.keyNorm = store.values[kName]?.asType(.float32)
        self.ropeTheta = ropeTheta
        self.softcap = softcap
        self.causal = causal
        self.window = window
        self.heads = heads
        self.headDimension = headDimension
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: VoiceChatSpeechAttentionCache? = nil,
        additiveMask: MLXArray? = nil
    ) -> MLXArray {
        let batch = input.dim(0)
        let length = input.dim(1)
        let cachedLength = cache?.keys?.dim(2) ?? 0
        var q = query(input).reshaped([batch, length, heads, headDimension])
        var k = key(input).reshaped([batch, length, heads, headDimension])
        var v = value(input).reshaped([batch, length, heads, headDimension])

        if let queryNorm, let keyNorm {
            q = voiceChatGemmaRMSNorm(q, weight: queryNorm)
            k = voiceChatGemmaRMSNorm(k, weight: keyNorm)
        }

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        q = voiceChatSpeechRoPE(q, offset: offset, theta: ropeTheta, headDimension: headDimension)
        k = voiceChatSpeechRoPE(k, offset: offset, theta: ropeTheta, headDimension: headDimension)

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // Generation advances one cached frame at a time. With no softcap or
        // explicit mask, fused SDPA is the same attention computation with far
        // fewer Metal dispatches across the 28-layer speech backbone.
        if length == 1, softcap == nil, additiveMask == nil {
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: 1 / Float(16), mask: nil)
                .transposed(0, 2, 1, 3)
                .reshaped([batch, length, heads * headDimension])
            return output(attended)
        }

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) / MLXArray(Float(16))
        if let softcap {
            scores = MLX.tanh(scores / MLXArray(softcap)) * MLXArray(softcap)
        }

        if let additiveMask {
            scores = scores + additiveMask
        } else if causal && length > 1 {
            let total = k.dim(2)
            let queryPositions = MLXArray(
                Int32(cachedLength) ..< Int32(cachedLength + length))
                .expandedDimensions(axis: 1)
            let keyPositions = MLXArray(0 ..< Int32(total)).expandedDimensions(axis: 0)
            var blocked = keyPositions .> queryPositions
            if let window {
                blocked = MLX.logicalOr(
                    blocked, queryPositions - keyPositions .>= MLXArray(Int32(window)))
            }
            scores = scores + MLX.where(
                blocked, MLXArray(Float(-1e9)), MLXArray(Float(0)))
                .expandedDimensions(axes: [0, 1])
        }

        let probabilities = MLX.softmax(scores.asType(.float32), axis: -1)
            .asType(scores.dtype)
        let attended = MLX.matmul(probabilities, v)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, length, heads * headDimension])
        return output(attended)
    }
}

struct VoiceChatSpeechMLP {
    let gate: VoiceChatMatrix
    let up: VoiceChatMatrix
    let down: VoiceChatMatrix

    init(_ store: VoiceChatSpeechWeightStore, prefix: String) throws {
        gate = try store.matrix("\(prefix).gate_proj.weight")
        up = try store.matrix("\(prefix).up_proj.weight")
        down = try store.matrix("\(prefix).down_proj.weight")
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(geluApproximate(gate(input)) * up(input))
    }
}

struct VoiceChatSpeechBackboneLayer {
    let attention: VoiceChatSpeechAttention
    let mlp: VoiceChatSpeechMLP
    let inputNorm: MLXArray
    let postAttentionNorm: MLXArray
    let preFeedForwardNorm: MLXArray
    let postFeedForwardNorm: MLXArray
}

final class VoiceChatSpeechBackbone {
    private let layers: [VoiceChatSpeechBackboneLayer]
    private let finalNorm: MLXArray

    init(_ store: VoiceChatSpeechWeightStore, configuration: VoiceChatSpeechConfiguration) throws {
        let prefix = "tts_model.tts_model.backbone"
        let fullAttention = Set([5, 11, 17, 23])
        layers = try (0 ..< configuration.numHiddenLayers).map { index in
            let layer = "\(prefix).layers.\(index)"
            let full = fullAttention.contains(index)
            return VoiceChatSpeechBackboneLayer(
                attention: try VoiceChatSpeechAttention(
                    store, prefix: "\(layer).self_attn",
                    heads: configuration.numAttentionHeads,
                    headDimension: configuration.headDim,
                    ropeTheta: full ? 1_000_000 : 10_000,
                    window: full ? nil : 7_500),
                mlp: try VoiceChatSpeechMLP(store, prefix: "\(layer).mlp"),
                inputNorm: try store.dense("\(layer).input_layernorm.weight"),
                postAttentionNorm: try store.dense("\(layer).post_attention_layernorm.weight"),
                preFeedForwardNorm: try store.dense("\(layer).pre_feedforward_layernorm.weight"),
                postFeedForwardNorm: try store.dense("\(layer).post_feedforward_layernorm.weight"))
        }
        finalNorm = try store.dense("\(prefix).norm.weight")
    }

    func makeCache(
        retainedPrefixFrames: Int = 0,
        recentContextFrames: Int? = nil
    ) -> [VoiceChatSpeechAttentionCache] {
        layers.map { _ in
            VoiceChatSpeechAttentionCache(
                retainedPrefixFrames: retainedPrefixFrames,
                recentContextFrames: recentContextFrames)
        }
    }

    func callAsFunction(
        _ input: MLXArray,
        cache: [VoiceChatSpeechAttentionCache]? = nil
    ) -> MLXArray {
        var hidden = input
        for (index, layer) in layers.enumerated() {
            var residual = voiceChatGemmaRMSNorm(hidden, weight: layer.inputNorm)
            residual = layer.attention(residual, cache: cache?[index])
            hidden = hidden + voiceChatGemmaRMSNorm(
                residual, weight: layer.postAttentionNorm)
            residual = voiceChatGemmaRMSNorm(hidden, weight: layer.preFeedForwardNorm)
            hidden = hidden + voiceChatGemmaRMSNorm(
                layer.mlp(residual), weight: layer.postFeedForwardNorm)
        }
        return voiceChatGemmaRMSNorm(hidden, weight: finalNorm)
    }
}

final class VoiceChatCharacterEncoder {
    private let mapping: [Int: [Int32]]
    private let embedding: MLXArray
    private let projection: VoiceChatMatrix
    private let continuationEmbedding: MLXArray
    private let isContinuation: MLXArray
    private let specialEmbedding: MLXArray
    private let specialFlags: MLXArray
    private let padCharacter: Int32
    private let attention: VoiceChatSpeechAttention
    private let mlp: VoiceChatSpeechMLP
    private let preAttentionNorm: MLXArray
    private let postAttentionNorm: MLXArray
    private let preFeedForwardNorm: MLXArray
    private let postFeedForwardNorm: MLXArray
    private let finalNorm: MLXArray
    private let hiddenSize: Int

    init(
        _ store: VoiceChatSpeechWeightStore,
        tokenizer: VoiceChatTokenizer,
        configuration: VoiceChatSpeechConfiguration
    ) throws {
        let prefix = "tts_model.tts_model.embed_subword"
        mapping = tokenizer.subwordToCharacters
        embedding = try store.dense("\(prefix).embed_tokens.weight")
        projection = try store.matrix("\(prefix).proj_embedding.weight")
        continuationEmbedding = try store.dense(
            "\(prefix).subword_flag_emb.cont_emb.weight")
        guard let continuation = store.values[
            "\(prefix).subword_flag_emb.is_continuation"
        ], let flags = store.values["\(prefix).bos_eos_emb.special_flags"] else {
            throw VoiceChatLoadError.unexpectedKeys([
                "\(prefix).subword_flag_emb.is_continuation",
                "\(prefix).bos_eos_emb.special_flags",
            ])
        }
        isContinuation = continuation
        specialEmbedding = try store.dense("\(prefix).bos_eos_emb.special_emb.weight")
        specialFlags = flags
        padCharacter = Int32(embedding.dim(0) - 1)
        hiddenSize = configuration.hiddenSize

        let layer = "\(prefix).backbone.encoder.layers.0"
        attention = try VoiceChatSpeechAttention(
            store, prefix: "\(layer).self_attn",
            heads: configuration.numAttentionHeads,
            headDimension: configuration.headDim,
            ropeTheta: 10_000, softcap: 50, causal: false)
        mlp = try VoiceChatSpeechMLP(store, prefix: "\(layer).mlp")
        preAttentionNorm = try store.dense("\(layer).pre_self_attn_layernorm.weight")
        postAttentionNorm = try store.dense("\(layer).post_self_attn_layernorm.weight")
        preFeedForwardNorm = try store.dense("\(layer).pre_feedforward_layernorm.weight")
        postFeedForwardNorm = try store.dense("\(layer).post_feedforward_layernorm.weight")
        finalNorm = try store.dense("\(prefix).backbone.encoder.norm.weight")
    }

    func callAsFunction(_ tokenID: Int) -> MLXArray {
        let characters = mapping[tokenID] ?? [padCharacter]
        let ids = MLXArray(characters).expandedDimensions(axis: 0)
        var hidden = embedding[ids] * MLXArray(Float(Foundation.sqrt(Double(hiddenSize))))
        var residual = voiceChatGemmaRMSNorm(hidden, weight: preAttentionNorm)
        residual = attention(residual)
        hidden = hidden + voiceChatGemmaRMSNorm(residual, weight: postAttentionNorm)
        residual = voiceChatGemmaRMSNorm(hidden, weight: preFeedForwardNorm)
        hidden = hidden + voiceChatGemmaRMSNorm(
            mlp(residual), weight: postFeedForwardNorm)
        hidden = voiceChatGemmaRMSNorm(hidden, weight: finalNorm)

        let pooled = MLX.sum(hidden, axis: 1) / MLXArray(Float(characters.count))
        var output = projection(pooled)
        let continuation = isContinuation[tokenID].item(Int.self)
        let special = specialFlags[tokenID].item(Int.self)
        output = output + continuationEmbedding[continuation] + specialEmbedding[special]
        return output.expandedDimensions(axis: 0)
    }
}

struct VoiceChatGatedFusion {
    let audio: VoiceChatMatrix
    let text: VoiceChatMatrix
    let gate: MLXArray
    let residualScale: MLXArray
    let finalNorm: MLXArray
    let quantizers: Int

    init(
        _ store: VoiceChatSpeechWeightStore,
        configuration: VoiceChatSpeechConfiguration
    ) throws {
        let prefix = "tts_model.tts_model.gated_fusion_audio_text"
        audio = try store.matrix(
            "\(prefix).audio_proj.weight", bias: "\(prefix).audio_proj.bias")
        text = try store.matrix(
            "\(prefix).text_proj.weight", bias: "\(prefix).text_proj.bias")
        gate = try store.dense("\(prefix).gate")
        residualScale = try store.dense("\(prefix).residual_scale")
        finalNorm = try store.dense("\(prefix).final_norm.weight")
        quantizers = configuration.numQuantizers
    }

    func callAsFunction(audio audioInput: MLXArray, text textInput: MLXArray) -> MLXArray {
        let projectedAudio = audio(audioInput / MLXArray(Float(quantizers)))
        let projectedText = text(textInput)
        let mix = MLX.sigmoid(gate)
        let residual = MLX.sigmoid(residualScale)
        let hidden = mix * projectedAudio + (MLXArray(Float(1)) - mix) * projectedText
        return voiceChatGemmaRMSNorm(hidden * residual, weight: finalNorm)
    }
}

struct VoiceChatMoGBlock {
    let preNorm: MLXArray
    let postNorm: MLXArray
    let mlp: VoiceChatSpeechMLP
}

final class VoiceChatMoGHead {
    private let blocks: [VoiceChatMoGBlock]
    private let finalNorm: MLXArray
    private let lowMatrix: MLXArray
    private let logitsProjection: VoiceChatMatrix
    private let meansProjection: VoiceChatMatrix
    private let residualProjection: VoiceChatMatrix
    private let logsProjection: VoiceChatMatrix

    init(_ store: VoiceChatSpeechWeightStore) throws {
        let prefix = "tts_model.tts_model.mog_head"
        blocks = try (0 ..< 3).map { index in
            let block = "\(prefix).mlp_stack.\(index)"
            return VoiceChatMoGBlock(
                preNorm: try store.dense("\(block).pre_norm.weight"),
                postNorm: try store.dense("\(block).post_norm.weight"),
                mlp: try VoiceChatSpeechMLP(store, prefix: "\(block).mlp"))
        }
        finalNorm = try store.dense("\(prefix).mlp_stack.3.weight")
        lowMatrix = try store.dense("\(prefix).low_mat")
        logitsProjection = try store.matrix("\(prefix).proj_logits.weight")
        meansProjection = try store.matrix("\(prefix).proj_mus.weight")
        residualProjection = try store.matrix("\(prefix).proj_else.weight")
        logsProjection = try store.matrix("\(prefix).proj_logs.weight")
    }

    func infer(
        _ input: MLXArray,
        guidance: Float,
        topP: Float
    ) -> (sample: MLXArray, logs: MLXArray) {
        var hidden = input
        for block in blocks {
            let residual = voiceChatGemmaRMSNorm(hidden, weight: block.preNorm)
            hidden = hidden + voiceChatGemmaRMSNorm(
                block.mlp(residual), weight: block.postNorm)
        }
        hidden = voiceChatGemmaRMSNorm(hidden, weight: finalNorm)

        if guidance > 0 {
            let half = hidden.dim(0) / 2
            let conditioned = hidden[0..<half, 0..., 0...]
            let unconditioned = hidden[half..., 0..., 0...]
            hidden = conditioned + MLXArray(guidance) * (conditioned - unconditioned)
        }

        var logits = logitsProjection(hidden)
        logits = voiceChatTopPFilter(logits, topP: topP)
        let components = MLX.argMax(
            logits + voiceChatGumbelLike(logits), axis: -1)

        let selected = meansProjection.selectedRows(
            components, groups: voiceChatSpeechComponents,
            rowsPerGroup: voiceChatSpeechLowRank)
        let coefficients = MLX.sum(
            selected * hidden.expandedDimensions(axis: -2), axis: -1)
        let mean = MLX.sum(
            lowMatrix[components] * coefficients.expandedDimensions(axis: -2),
            axis: -1)
        let logs = MLX.maximum(
            logsProjection(hidden),
            MLXArray(voiceChatSpeechMinimumLogStandardDeviation))
        return (mean * MLX.exp(logs) + residualProjection(hidden), logs)
    }
}
