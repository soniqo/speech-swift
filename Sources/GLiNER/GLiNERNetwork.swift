import Foundation
import MLX
import MLXNN

// DeBERTa and span-head equations adapted from gliner2-mlx (MIT).
// Attribution and license are distributed in LICENSE-reference.
final class GLiNERNetwork {
    let config: GLiNERConfig
    let weights: [String: MLXArray]
    let evaluateLayers: Bool
    let quantization: GLiNERQuantization?
    let quantizedKeys: Set<String>
    /// Per-layer query/key projections of the relative-position embeddings,
    /// shaped [heads, 2·buckets, headDim]. They depend only on weights, so they
    /// are computed once at load instead of on every request, where they would
    /// cost 2 × layers matmuls of [2·buckets, hidden] × [hidden, hidden].
    private(set) var positionQueries = [MLXArray]()
    private(set) var positionKeys = [MLXArray]()
    init(config: GLiNERConfig, weights: [String: MLXArray], evaluateLayers: Bool = false, quantization: GLiNERQuantization? = nil) throws {
        self.config = config; self.weights = weights; self.evaluateLayers = evaluateLayers
        try quantization?.validate()
        self.quantization = quantization; self.quantizedKeys = Set(quantization?.quantizedKeys ?? [])
        let h = config.hiddenSize
        var required: [String: [Int]] = [
            "encoder.embeddings.word_embeddings.weight": [config.vocabSize, h],
            "encoder.encoder.rel_embeddings.weight": [2 * config.positionBuckets, h],
        ]
        func norm(_ key: String) { required[key + ".weight"] = [h]; required[key + ".bias"] = [h] }
        func linear(_ key: String, _ input: Int, _ output: Int) {
            required[key + ".weight"] = [output, input]; required[key + ".bias"] = [output]
        }
        norm("encoder.embeddings.LayerNorm"); norm("encoder.encoder.LayerNorm")
        for i in 0..<config.numHiddenLayers {
            let p = "encoder.encoder.layers.\(i)"
            for name in ["query_proj", "key_proj", "value_proj"] { linear(p + ".attention.self_attn." + name, h, h) }
            linear(p + ".attention.output.dense", h, h); norm(p + ".attention.output.LayerNorm")
            linear(p + ".intermediate.dense", h, config.intermediateSize)
            linear(p + ".output.dense", config.intermediateSize, h); norm(p + ".output.LayerNorm")
        }
        linear("classifier.layers.0", h, h * 2); linear("classifier.layers.2", h * 2, 1)
        linear("count_pred.layers.0", h, h * 2); linear("count_pred.layers.2", h * 2, 20)
        required["count_embed.pos_embedding.weight"] = [20, h]
        for k in ["ih", "hh"] {
            required["count_embed.gru.weight_\(k)_l0"] = [3 * h, h]
            required["count_embed.gru.bias_\(k)_l0"] = [3 * h]
        }
        linear("count_embed.projector.layers.0", 2 * h, 4 * h)
        linear("count_embed.projector.layers.2", 4 * h, h)
        for p in ["project_start", "project_end", "out_project"] {
            let key = "span_rep.span_rep_layer.\(p).layers"
            linear(key + ".0", p == "out_project" ? 2 * h : h, 4 * h)
            linear(key + ".3", 4 * h, h)
        }
        guard quantizedKeys.isSubset(of:Set(required.keys)) else {
            throw GLiNERError.invalidConfiguration("Quantized tensor does not belong to the supported architecture.")
        }
        for (key, shape) in required {
            if quantizedKeys.contains(key) {
                guard let quantization, shape.count == 2, shape[1] % quantization.groupSize == 0,
                      let value = weights[key], value.dtype == .uint32, value.shape == [shape[0],shape[1]/4] else {
                    throw GLiNERError.missingWeight("Malformed packed INT8 tensor: \(key)")
                }
                let base = String(key.dropLast(".weight".count))
                for suffix in [".scales",".biases"] {
                    guard let value = weights[base+suffix], value.dtype == .float16,
                          value.shape == [shape[0],shape[1]/quantization.groupSize] else {
                        throw GLiNERError.missingWeight("Missing or malformed INT8 scales/offsets: \(base+suffix)")
                    }
                }
                continue
            }
            guard let value = weights[key], value.shape == shape,
                  quantization == nil ? [.float16,.float32].contains(value.dtype) : value.dtype == .float16 else {
                throw GLiNERError.missingWeight("Missing or malformed weight: \(key); expected \(shape)")
            }
        }
        let relative = self.norm(weights["encoder.encoder.rel_embeddings.weight"]!, "encoder.encoder.LayerNorm")
        for i in 0..<config.numHiddenLayers {
            let a = "encoder.encoder.layers.\(i).attention.self_attn"
            positionQueries.append(self.heads(self.linear(relative, a + ".query_proj")))
            positionKeys.append(self.heads(self.linear(relative, a + ".key_proj")))
        }
        eval(positionQueries + positionKeys)
    }
    /// [tokens, hidden] -> [heads, tokens, headDim]
    func heads(_ x: MLXArray) -> MLXArray {
        x.reshaped(-1, config.numAttentionHeads, config.hiddenSize / config.numAttentionHeads).transposed(1, 0, 2)
    }
    func linear(_ x: MLXArray, _ key: String) -> MLXArray {
        if let quantization, quantizedKeys.contains(key + ".weight") {
            return quantizedMM(x,weights[key + ".weight"]!,scales:weights[key + ".scales"]!,biases:weights[key + ".biases"]!,groupSize:quantization.groupSize,bits:quantization.bits) + weights[key + ".bias"]!
        }
        return matmul(x, weights[key + ".weight"]!.T) + weights[key + ".bias"]!
    }
    func embedding(_ ids: [Int]) -> MLXArray {
        let key = "encoder.embeddings.word_embeddings"
        let indices = MLXArray(ids.map(Int32.init))
        if let quantization, quantizedKeys.contains(key + ".weight") {
            // Only materialize the requested token rows, never the full table.
            return dequantized(weights[key + ".weight"]![indices],scales:weights[key + ".scales"]![indices],biases:weights[key + ".biases"]![indices],groupSize:quantization.groupSize,bits:quantization.bits)
        }
        return weights[key + ".weight"]![indices]
    }
    func norm(_ x: MLXArray, _ key: String) -> MLXArray {
        MLXFast.layerNorm(x,weight:weights[key + ".weight"]!,bias:weights[key + ".bias"]!,eps:config.layerNormEps)
    }
    /// DeBERTa log-bucketed relative position for q - k.
    static func bucket(_ relative: Int, buckets: Int, maxPosition: Int) -> Int {
        let mid = buckets / 2
        let magnitude = abs(relative) < mid ? mid - 1 : abs(relative)
        if magnitude <= mid { return relative }
        let logged = Int(ceil(log(Double(magnitude) / Double(mid)) / log(Double(maxPosition - 1) / Double(mid)) * Double(mid - 1))) + mid
        return relative < 0 ? -logged : logged
    }
    static func relativeIndices(length: Int, buckets: Int, maxPosition: Int, reverse: Bool = false) -> [Int32] {
        // The bucket depends only on q - k: compute 2n-1 values, then look up.
        let table = (-(length - 1)..<length).map { bucket($0, buckets: buckets, maxPosition: maxPosition) }
        var result = [Int32](); result.reserveCapacity(length * length)
        for q in 0..<length { for k in 0..<length {
            let b = table[q - k + length - 1]
            result.append(Int32(min(max((reverse ? -b : b) + buckets, 0), 2 * buckets - 1)))
        }}
        return result
    }
    func encode(_ ids: [Int]) -> MLXArray {
        let n = ids.count, h = config.hiddenSize, heads = config.numAttentionHeads, d = h / heads
        var x = norm(embedding(ids), "encoder.embeddings.LayerNorm")
        let cRel = Self.relativeIndices(length: n, buckets: config.positionBuckets, maxPosition: config.maxPositionEmbeddings)
        let pRel = Self.relativeIndices(length: n, buckets: config.positionBuckets, maxPosition: config.maxPositionEmbeddings, reverse: true)
        // Short inputs reach only a narrow band of position buckets. Score
        // against that band instead of all 2·buckets rows; the gathered values
        // are identical because indices are shifted by the same offset.
        let lo = min(cRel.min() ?? 0, pRel.min() ?? 0), hi = max(cRel.max() ?? 0, pRel.max() ?? 0)
        let cIndices = broadcast(MLXArray(cRel.map { $0 - lo }).reshaped(1,n,n), to: [heads,n,n])
        let pIndices = broadcast(MLXArray(pRel.map { $0 - lo }).reshaped(1,n,n), to: [heads,n,n])
        let band = Int(lo)...Int(hi)
        let scale = Float(1 / sqrt(Double(d * 3)))
        for i in 0..<config.numHiddenLayers {
            let p = "encoder.encoder.layers.\(i)", a = p + ".attention.self_attn"
            let q = self.heads(linear(x,a + ".query_proj")), k = self.heads(linear(x,a + ".key_proj")), v = self.heads(linear(x,a + ".value_proj"))
            let pq = positionQueries[i][0..., band], pk = positionKeys[i][0..., band]
            let c2p = takeAlong(matmul(q,pk.transposed(0,2,1)), cIndices, axis: -1)
            let p2c = takeAlong(matmul(k,pq.transposed(0,2,1)), pIndices, axis: -1).transposed(0,2,1)
            let scores = (matmul(q,k.transposed(0,2,1)) + c2p + p2c) * scale
            let context = matmul(softmax(scores,axis: -1),v).transposed(1,0,2).reshaped(n,h)
            x = norm(x + linear(context,p + ".attention.output.dense"),p + ".attention.output.LayerNorm")
            let intermediate = gelu(linear(x,p + ".intermediate.dense"))
            x = norm(x + linear(intermediate,p + ".output.dense"),p + ".output.LayerNorm")
            // Materialize each layer before building the next graph so its
            // temporary attention and feed-forward arrays can be released.
            if evaluateLayers { eval(x) }
        }
        return x
    }
    func classify(_ embeddings: MLXArray) -> MLXArray {
        softmax(linear(relu(linear(embeddings,"classifier.layers.0")),"classifier.layers.2").squeezed(axis: -1),axis: -1)
    }
    func spans(words: MLXArray, parent: MLXArray, fields: MLXArray, maxWidth: Int) -> MLXArray? {
        let count = linear(relu(linear(parent,"count_pred.layers.0")),"count_pred.layers.2").argMax().item(Int.self)
        guard count > 0 else { return nil }
        // Entity extraction consumes the first recurrent slot only.
        let h = config.hiddenSize
        let pos = weights["count_embed.pos_embedding.weight"]![0]
        let gi = matmul(pos,weights["count_embed.gru.weight_ih_l0"]!.T) + weights["count_embed.gru.bias_ih_l0"]!
        let gh = matmul(fields,weights["count_embed.gru.weight_hh_l0"]!.T) + weights["count_embed.gru.bias_hh_l0"]!
        let r = sigmoid(gi[0..<h] + gh[0...,0..<h])
        let z = sigmoid(gi[h..<(2*h)] + gh[0...,h..<(2*h)])
        let candidate = tanh(gi[(2*h)..<(3*h)] + r * gh[0...,(2*h)..<(3*h)])
        let state = (1-z) * candidate + z * fields
        let projected = linear(relu(linear(concatenated([state,fields],axis: -1),"count_embed.projector.layers.0")),"count_embed.projector.layers.2")
        func project(_ x: MLXArray, _ name: String) -> MLXArray {
            let p = "span_rep.span_rep_layer.\(name).layers"
            return linear(relu(linear(x,p + ".0")),p + ".3")
        }
        let start = project(words,"project_start"), end = project(words,"project_end")
        let n = words.dim(0)
        var starts = [Int32](), ends = [Int32]()
        for i in 0..<n { for w in 0..<maxWidth { starts.append(Int32(i+w<n ? i : 0)); ends.append(Int32(i+w<n ? i+w : 0)) } }
        let combined = relu(concatenated([start[MLXArray(starts)],end[MLXArray(ends)]],axis: -1))
        let span = project(combined,"out_project")
        return sigmoid(matmul(projected,span.T)).reshaped(fields.dim(0),n,maxWidth)
    }
}
