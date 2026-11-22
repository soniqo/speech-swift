// Copyright 2026 the HuggingFace Team. All rights reserved.
// The Inkling 2 tensor layout and equations follow the Apache-2.0 reference
// implementations by Hugging Face and MLX-VLM. See ThirdPartyNotices/Inkling2Mini.txt.
import Foundation
import MLX
import MLXFast

/// Native MLX Inkling 2 decoder, with one cache per logical layer visit.
public final class Inkling2Model {
    public let config: Inkling2Config
    let weights: [String: MLXArray]
    private let routed: @Sendable ([MLXArray]) -> [MLXArray]
    private let mixedResidual: @Sendable ([MLXArray]) -> [MLXArray]
    static let base = "language_model.model."

    public struct LayerState {
        public var keys: MLXArray?
        public var values: MLXArray?
        var convolution: [MLXArray?] = Array(repeating: nil, count: 4)
    }
    public struct State {
        public var layers: [LayerState]
        public var position: Int
        public var arrays: [MLXArray] {
            layers.flatMap { [$0.keys, $0.values] + $0.convolution }.compactMap { $0 }
        }
        public static func initial(config: Inkling2Config) -> State {
            State(layers: config.layerVisits.map { _ in LayerState() }, position: 0)
        }
    }

    public init(config: Inkling2Config, weights: [String: MLXArray]) throws {
        self.config = config; self.weights = weights
        routed = compile { (a: [MLXArray]) -> [MLXArray] in
            let scores = sigmoid(matmul(a[0].asType(.float32), a[1].asType(.float32).T))
            let partition = argPartition(-(scores + a[2]), kth: config.expertsPerToken - 1, axis: -1)
            let indices = partition[0..., 0..., ..<config.expertsPerToken].asType(.uint32)
            let probabilities = takeAlong(scores, indices, axis: -1)
            return [indices, probabilities / (sum(probabilities, axis: -1, keepDims: true) + 1e-20) * config.routeScale]
        }
        mixedResidual = compile { (a: [MLXArray]) -> [MLXArray] in
            let states = a[0].asType(.float32), keys = Self.unweightedNorm(states)
            let scores = matmul(keys, a[1].reshaped([-1, config.hiddenSize]).T) * (1 / sqrt(Float(config.hiddenSize)))
            let probabilities = softmax(scores, axis: -2).swappedAxes(-1, -2)
            return [matmul(probabilities, config.normalizeResidualValues ? keys : states)]
        }
        try validateWeights()
    }

    func weight(_ path: String) -> MLXArray { weights[path]! }

    func linear(_ x: MLXArray, _ path: String) -> MLXArray {
        let result: MLXArray
        if let scales = weights[path + ".scales"] {
            let q = config.quantization(for: path)
            result = quantizedMM(x, weight(path + ".weight"), scales: scales,
                                 biases: weights[path + ".biases"], groupSize: q.groupSize, bits: q.bits)
        } else {
            result = matmul(x, weight(path + ".weight").swappedAxes(-1, -2))
        }
        return weights[path + ".bias"].map { result + $0 } ?? result
    }

    func norm(_ x: MLXArray, _ path: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight(path + ".weight"), eps: config.rmsEpsilon)
    }
    static func unweightedNorm(_ x: MLXArray, epsilon: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: MLXArray.ones([x.dim(-1)]), eps: epsilon)
    }
    private static let clipped: @Sendable ([MLXArray]) -> [MLXArray] = compile { (a: [MLXArray]) -> [MLXArray] in
        let f = a[0].asType(.float32) * a[2].asType(.float32)
        return [(f * minimum(a[1] / (sqrt(sum(f * f, axis: -1, keepDims: true)) + 1e-6), 1)).asType(a[0].dtype)]
    }
    static func clip(_ x: MLXArray, threshold: Float, scale: MLXArray? = nil) -> MLXArray {
        clipped([x, MLXArray(threshold), scale ?? MLXArray(Float(1))])[0]
    }
    private static let activated: @Sendable ([MLXArray]) -> [MLXArray] = compile { (a: [MLXArray]) -> [MLXArray] in
        let gate = a[0].asType(.float32), up = a[1].asType(.float32)
        return [(tanh(gate / a[2]) * a[2] * sigmoid(gate) * tanh(up / a[3]) * a[3]).asType(a[0].dtype)]
    }
    private static let combined: @Sendable ([MLXArray]) -> [MLXArray] = compile { (a: [MLXArray]) -> [MLXArray] in
        [sum(a[0].asType(.float32) * a[1].expandedDimensions(axis: -1), axis: -2).asType(a[0].dtype)]
    }

    func embed(_ ids: MLXArray) -> MLXArray {
        let p = Self.base + "embed_tokens"
        let flat = ids.flattened()
        let x: MLXArray
        if let scales = weights[p + ".scales"] {
            let q = config.quantization(for: p)
            x = dequantized(weight(p + ".weight")[flat], scales: scales[flat],
                            biases: weight(p + ".biases")[flat], groupSize: q.groupSize, bits: q.bits)
        } else { x = weight(p + ".weight")[flat] }
        return norm(x.reshaped(ids.shape + [config.hiddenSize]), Self.base + "embed_norm")
    }

    func residual(_ blocks: [MLXArray], _ partial: MLXArray, query: MLXArray) -> MLXArray {
        mixedResidual([stacked(blocks + [partial], axis: -2), query])[0]
    }

    func controller(_ x: MLXArray, message: MLXArray, path: String) -> MLXArray {
        let pair = concatenated([Self.unweightedNorm(x, epsilon: config.rmsEpsilon),
                                 Self.unweightedNorm(message, epsilon: config.rmsEpsilon)], axis: -1)
        let down = weight(path + ".down_weight")
        let h = matmul(pair.asType(down.dtype), down.T).asType(.float32)
        let forget = tanh(matmul(h, weight(path + ".forget_weight").T))
        let gain = tanh(matmul(h, weight(path + ".gain_weight").T))
        return (1 - forget) * x + gain * message
    }

    func convolution(_ x: MLXArray, path: String, slot: Int,
                     state: inout LayerState, dynamicInput: MLXArray? = nil, heads: Int = 0) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), channels = x.dim(2), k = config.convolutionSize
        let f = x.asType(.float32)
        let previous = state.convolution[slot] ?? MLXArray.zeros([b, k - 1, channels])
        let context = concatenated([previous, f], axis: 1)
        state.convolution[slot] = context[0..., (context.dim(1) - k + 1)..., 0...]
        let out: MLXArray
        if let dynamicInput {
            let dynamic = linear(dynamicInput, path + ".weight_proj").asType(.float32).reshaped(b, t, heads, 1, k)
            let taps = weight(path + ".static_weight").T.reshaped(heads, channels / heads, k) + dynamic
            var result = MLXArray.zeros([b, t, heads, channels / heads])
            for lag in 0..<k {
                let values = context[0..., (k - 1 - lag)..<(k - 1 - lag + t), 0...].reshaped(result.shape)
                result = result + values * taps[0..., 0..., 0..., 0..., lag]
            }
            out = (result.reshaped(x.shape) + f) / sqrt(Float(2 * k + 1))
        } else {
            out = (conv1d(context, weight(path + ".conv.weight").asType(.float32), groups: channels) + f)
                / sqrt(Float(k + 1))
        }
        return out.asType(x.dtype)
    }

    func attention(_ x: MLXArray, keyInput: MLXArray, valueInput: MLXArray,
                   layer: Int, position: Int, state: inout LayerState) -> MLXArray {
        let c = config, local = c.localLayers.contains(layer)
        let heads = local ? c.slidingHeads : c.numHeads
        let kvHeads = local ? c.slidingKVHeads : c.numKVHeads
        let dim = local ? c.slidingHeadDim : c.headDim
        let extent = local ? c.slidingWindow : c.relativeExtent
        let p = Self.base + "layers.\(layer).self_attn"
        let b = x.dim(0), t = x.dim(1)
        var k = linear(keyInput, p + ".k_proj"), v = linear(valueInput, p + ".v_proj")
        k = convolution(k, path: p + ".k_sconv", slot: 0, state: &state,
                        dynamicInput: local ? nil : keyInput, heads: kvHeads)
        v = convolution(v, path: p + ".v_sconv", slot: 1, state: &state,
                        dynamicInput: local ? nil : valueInput, heads: kvHeads)
        var q = Self.clip(linear(x, p + ".q_proj").reshaped(b, t, heads, dim),
                          threshold: c.queryClip, scale: weight(p + ".q_scale"))
        k = Self.clip(k.reshaped(b, t, kvHeads, dim), threshold: c.queryClip)
        v = v.reshaped(b, t, kvHeads, dim)
        if c.normalizeValues { v = Self.unweightedNorm(v, epsilon: c.rmsEpsilon).asType(x.dtype) }
        var r = Self.clip(linear(x, p + ".r_proj").reshaped(b, t, heads, c.relativeDim),
                          threshold: c.relativeClip, scale: weight(p + ".rel_scale"))
        if !local && c.logScalingFloor > 0 {
            let positions = MLXArray((position + 1)...(position + t)).asType(.float32)
            let tau = (1 + c.logScalingAlpha * log(maximum(positions / Float(c.logScalingFloor), 1)))
                .reshaped(1, t, 1, 1)
            q = (q.asType(.float32) * tau).asType(x.dtype)
            r = (r.asType(.float32) * tau).asType(x.dtype)
        }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3); v = v.transposed(0, 2, 1, 3)
        if let previous = state.keys { k = concatenated([previous, k], axis: 2) }
        if let previous = state.values { v = concatenated([previous, v], axis: 2) }
        // Keep the complete current chunk for attention, then retain only the
        // chronological sliding tail for the next chunk.
        let size = k.dim(2), keep = local ? min(size, c.slidingWindow) : size
        state.keys = k[0..., 0..., (size - keep)..., 0...]
        state.values = v[0..., 0..., (size - keep)..., 0...]
        let distances = (MLXArray(0..<t) + size - t).expandedDimensions(axis: -1)
            - MLXArray(0..<size).expandedDimensions(axis: 0)
        let relative = matmul(r, weight(p + ".rel_proj").asType(x.dtype)).transposed(0, 2, 1, 3)
        let indices = broadcast(MLX.clip(distances, min: 0, max: extent - 1).asType(.int32), to: [b, heads, t, size])
        var mask = takeAlong(relative, indices, axis: -1)
        mask = which(distances .>= extent, MLXArray(0).asType(x.dtype), mask)
        let allowed = local ? (distances .>= 0) .&& (distances .< c.slidingWindow) : distances .>= 0
        mask = which(allowed, mask, MLXArray(-Float.infinity).asType(x.dtype))
        let sink = linear(x, p + ".sink_proj").transposed(0, 2, 1).expandedDimensions(axis: -1)
        mask = concatenated([mask, sink], axis: -1)
        k = concatenated([k, MLXArray.zeros([b, kvHeads, 1, dim], dtype: k.dtype)], axis: 2)
        v = concatenated([v, MLXArray.zeros([b, kvHeads, 1, dim], dtype: v.dtype)], axis: 2)
        var result = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(dim)), mask: .array(mask))
        result = result.transposed(0, 2, 1, 3).reshaped(b, t, heads * dim)
        result = (result.asType(.float32) * sigmoid(linear(x, p + ".gate_proj").asType(.float32))).asType(x.dtype)
        return linear(result, p + ".o_proj")
    }

    func expertLinear(_ x: MLXArray, indices: MLXArray, path: String, sorted: Bool) -> MLXArray {
        if let scales = weights[path + ".scales"] {
            let q = config.quantization(for: path)
            return gatherQuantizedMM(x, weight(path + ".weight"), scales: scales,
                                     biases: weights[path + ".biases"], rhsIndices: indices,
                                     groupSize: q.groupSize, bits: q.bits, sortedIndices: sorted)
        }
        return gatherMM(x, weight(path + ".weight").swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }

    func mixture(_ x: MLXArray, layer: Int) -> MLXArray {
        let c = config, p = Self.base + "layers.\(layer).mlp"
        let routing = routed([x, weight(p + ".gate_weight"), weight(p + ".e_score_correction_bias")])
        let indices = routing[0], probabilities = routing[1]
        // Match the reference's grouped prefill dispatch. Grouping changes the
        // quantized reduction kernel as well as avoiding repeated expert reads.
        let sorted = indices.size >= 64
        var input = x.expandedDimensions(axes: [-2, -3]), selected = indices
        var inverse: MLXArray?
        if sorted {
            let flat = indices.flattened(), order = argSort(flat)
            inverse = argSort(order)
            input = input.reshaped(-1, 1, c.hiddenSize)[floorDivide(order, c.expertsPerToken)]
            selected = flat[order]
        }
        let gate = expertLinear(input, indices: selected, path: p + ".switch_mlp.gate_proj", sorted: sorted)
        let up = expertLinear(input, indices: selected, path: p + ".switch_mlp.up_proj", sorted: sorted)
        let hidden = Self.activated([gate, up, MLXArray(c.gateCap), MLXArray(c.upCap)])[0]
        var out = expertLinear(hidden, indices: selected, path: p + ".switch_mlp.down_proj", sorted: sorted)
        if let inverse { out = out[inverse].reshaped(indices.shape + [1, c.hiddenSize]) }
        out = out.squeezed(axis: -2)
        return Self.combined([out, probabilities])[0]
    }

    /// Input embeddings may replace token embeddings for image-prefill callers.
    public func forward(inputIDs: MLXArray, state: State, inputEmbeddings: MLXArray? = nil) throws -> (MLXArray, State) {
        guard inputIDs.ndim == 2 else { throw Inkling2Error.invalid("input IDs must have shape [1, tokens]") }
        let c = config, t = inputIDs.dim(1)
        guard inputIDs.dim(0) == 1, t > 0, t <= c.maxContext, state.position >= 0,
              state.layers.count == c.layerVisits.count, state.position <= c.maxContext - t,
              [.int32, .uint32, .int64, .uint64].contains(inputIDs.dtype) else {
            throw Inkling2Error.invalid("expected a nonempty single sequence within the context limit and a cache for every visit")
        }
        if let inputEmbeddings, inputEmbeddings.shape != [1, t, c.hiddenSize] {
            throw Inkling2Error.invalid("input embeddings must match input IDs and hidden width")
        }
        var x = inputEmbeddings ?? embed(inputIDs)
        var next = state, blocks: [MLXArray] = [], preludeBlocks: [MLXArray] = []
        var prelude: MLXArray?, bank: MLXArray?
        for (visit, layer) in c.layerVisits.enumerated() {
            if layer == c.coreStart {
                if visit == layer { prelude = x; preludeBlocks = blocks }
                else {
                    bank = Self.unweightedNorm(stacked(Array(blocks.dropFirst(preludeBlocks.count + 1)) + [x], axis: -2))
                    x = prelude!; blocks = preludeBlocks
                }
            }
            let p = Self.base + "layers.\(layer)", local = c.localLayers.contains(layer)
            var messages: MLXArray?
            if let bank, layer < c.coreEnd {
                let query = weight(Self.base + "message_queries")[layer - c.coreStart]
                let scores = matmul(query, bank.swappedAxes(-1, -2)) * (1 / sqrt(Float(c.hiddenSize)))
                messages = matmul(softmax(scores, axis: -1), bank)
            }
            let mixed = layer == 0 ? x.expandedDimensions(axis: -2)
                : residual(blocks, x, query: weight(p + ".attention_input.query"))
            var parts = (0..<3).map { mixed[0..., 0..., min($0, mixed.dim(-2) - 1), 0...] }
            for i in 0..<3 {
                if let messages {
                    parts[i] = controller(parts[i], message: messages[0..., 0..., i, 0...],
                                          path: p + ".controllers.\(["attn", "k", "v"][i])")
                }
                parts[i] = norm(parts[i], p + (local ? ".input_layernorm" : ".\(["q", "k", "v"][i])_input_layernorm")).asType(x.dtype)
            }
            var cache = next.layers[visit]
            let attn = attention(parts[0], keyInput: parts[1], valueInput: parts[2], layer: layer,
                                 position: state.position, state: &cache)
            var partial = convolution(attn, path: p + ".attn_sconv", slot: 2, state: &cache).asType(.float32)
            if layer % c.blockSize == 0 { blocks.append(x) } else { partial = partial + x.asType(.float32) }
            var hidden = residual(blocks, partial, query: weight(p + ".mlp_input.query"))[0..., 0..., 0, 0...]
            if let messages { hidden = controller(hidden, message: messages[0..., 0..., 3, 0...], path: p + ".controllers.mlp") }
            hidden = mixture(norm(hidden, p + ".post_attention_layernorm").asType(x.dtype), layer: layer)
            x = (partial + convolution(hidden, path: p + ".mlp_sconv", slot: 3, state: &cache).asType(.float32)).asType(x.dtype)
            next.layers[visit] = cache
        }
        let mixed = residual(blocks, x, query: weight(Self.base + "final_input.query"))[0..., 0..., 0, 0...]
        let hidden = norm(mixed, Self.base + "norm").asType(weight(Self.base + "embed_norm.weight").dtype).asType(.float32)
        var logits = linear(hidden, "language_model.lm_head")
        if c.logitCap > 0 { logits = tanh(logits / c.logitCap) * c.logitCap }
        next.position += t
        return (logits[0..., 0..., ..<c.unpaddedVocabSize], next)
    }
}
