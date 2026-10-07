import Foundation
import MLX
import MLXNN
import MLXFast

struct ClefHeadConfig: Decodable {
    let hidden_size: Int
    let width: Int
    let routing_layers: Int
    let layers: Int
    let heads: Int
    let feedforward: Int
}

/// Joint schema head: evidence routing, field interaction, and lexical prior.
/// Architecture and tensor names follow Cloudflare/clef-flash (Apache-2.0).
final class ClefHead {
    let config: ClefHeadConfig
    let weights: [String: MLXArray]

    init(config: ClefHeadConfig, weights: [String: MLXArray]) throws {
        guard config.hidden_size > 0, config.width > 0, config.heads > 0,
              config.width % config.heads == 0, config.layers >= 0,
              config.routing_layers >= 0, config.feedforward > 0 else {
            throw ClefError.invalidModel("Invalid joint head dimensions.")
        }
        self.config = config
        self.weights = weights.mapValues { $0.asType(.float32) }
        var required: [String: [Int]] = [:]
        let w = config.width, h = config.hidden_size, f = config.feedforward
        func norm(_ p: String, _ d: Int) { required[p + ".weight"] = [d]; required[p + ".bias"] = [d] }
        func linear(_ p: String, _ input: Int, _ output: Int, bias: Bool = true) {
            required[p + ".weight"] = [output, input]
            if bias { required[p + ".bias"] = [output] }
        }
        func attention(_ p: String) {
            required[p + ".in_proj_weight"] = [3 * w, w]
            required[p + ".in_proj_bias"] = [3 * w]
            linear(p + ".out_proj", w, w)
        }
        norm("hidden_norm", h)
        for p in ["memory_projection", "question_projection", "option_question_projection", "global_projection", "option_context_projection", "option_lexical_projection"] {
            linear(p, h, w, bias: false)
        }
        required["type_embedding.weight"] = [3, w]
        for i in 0..<config.routing_layers {
            let p = "evidence_layers.\(i)"
            for n in ["query_norm", "memory_norm", "feedforward_norm"] { norm(p + "." + n, w) }
            attention(p + ".attention")
            linear(p + ".feedforward.0", w, f); linear(p + ".feedforward.3", f, w)
        }
        for i in 0..<config.layers {
            let p = "layers.\(i)"
            for n in ["norm1", "norm2", "norm3"] { norm(p + "." + n, w) }
            attention(p + ".self_attn"); attention(p + ".multihead_attn")
            linear(p + ".linear1", w, f); linear(p + ".linear2", f, w)
        }
        for p in ["option_summary_norm", "field_norm", "option_norm"] { norm(p, w) }
        linear("residual_scorer.0", 4 * w, w); linear("residual_scorer.3", w, 1)
        for p in ["prior_logit_scale", "joint_logit_scale", "residual_gate"] { required[p] = [] }
        for (key, shape) in required {
            guard weights[key]?.shape == shape else { throw ClefError.invalidModel("Missing or incorrectly shaped joint head tensor: \(key)") }
        }
    }

    private func linear(_ p: String, _ x: MLXArray, bias: Bool = true) -> MLXArray {
        let y = matmul(x, weights[p + ".weight"]!.T)
        return bias ? y + weights[p + ".bias"]! : y
    }
    private func norm(_ p: String, _ x: MLXArray) -> MLXArray {
        let centered = x - x.mean(axis: -1, keepDims: true)
        return centered * rsqrt((centered * centered).mean(axis: -1, keepDims: true) + 1e-5)
            * weights[p + ".weight"]! + weights[p + ".bias"]!
    }
    private func unit(_ x: MLXArray, epsilon: Float = 1e-12) -> MLXArray {
        x / maximum(sqrt((x * x).sum(axis: -1, keepDims: true)), epsilon)
    }
    private func attention(_ p: String, _ query: MLXArray, _ memory: MLXArray) -> MLXArray {
        let w = config.width, heads = config.heads, d = w / heads
        let matrix = weights[p + ".in_proj_weight"]!, bias = weights[p + ".in_proj_bias"]!
        func project(_ x: MLXArray, _ i: Int) -> MLXArray {
            (matmul(x, matrix[(i * w)..<((i + 1) * w)].T) + bias[(i * w)..<((i + 1) * w)])
                .reshaped(-1, heads, d).transposed(1, 0, 2).expandedDimensions(axis: 0)
        }
        let out = MLXFast.scaledDotProductAttention(queries: project(query, 0), keys: project(memory, 1),
            values: project(memory, 2), scale: 1 / sqrt(Float(d)), mask: .none)
        return linear(p + ".out_proj", out[0].transposed(1, 0, 2).reshaped(-1, w))
    }

    func logits(hidden: MLXArray, encoding: ClefEncoding, lexical: ([Int]) -> MLXArray) -> [MLXArray] {
        let h = norm("hidden_norm", hidden.asType(.float32))
        let memory = linear("memory_projection", h, bias: false)
        let global = h[h.dim(0) - 1]
        let questions = encoding.questions
        let qv = stacked(questions.map { h[$0.span].mean(axis: 0) })
        let contexts = questions.map { q in stacked(q.optionSpans.map { h[$0].mean(axis: 0) }) }
        let lex = questions.map { q in stacked(q.optionSpans.map { lexical(Array(encoding.tokens[$0])).asType(.float32).mean(axis: 0) }) }
        var queries = concatenated(questions.indices.map { i in
            linear("option_context_projection", contexts[i], bias: false)
                + linear("option_lexical_projection", lex[i], bias: false)
                + linear("option_question_projection", qv[i], bias: false)
        })
        for i in 0..<config.routing_layers {
            let p = "evidence_layers.\(i)"
            queries = queries + attention(p + ".attention", norm(p + ".query_norm", queries), norm(p + ".memory_norm", memory))
            queries = queries + linear(p + ".feedforward.3", gelu(linear(p + ".feedforward.0", norm(p + ".feedforward_norm", queries))))
        }
        var cursor = 0
        let options = questions.map { q -> MLXArray in
            let count = q.optionSpans.count
            defer { cursor += count }
            return queries[cursor..<(cursor + count)]
        }
        let base = linear("question_projection", qv, bias: false)
        let summary = stacked(options.enumerated().map { i, o in
            (softmax(matmul(o, base[i]) / sqrt(Float(config.width)), axis: 0).expandedDimensions(axis: -1) * o).sum(axis: 0)
        })
        var fields = base + norm("option_summary_norm", summary) + linear("global_projection", global, bias: false)
            + weights["type_embedding.weight"]![MLXArray(questions.map { Int32($0.question.typeID) })]
        for i in 0..<config.layers {
            let p = "layers.\(i)"
            let x = norm(p + ".norm1", fields)
            fields = fields + attention(p + ".self_attn", x, x)
            fields = fields + attention(p + ".multihead_attn", norm(p + ".norm2", fields), memory)
            fields = fields + linear(p + ".linear2", gelu(linear(p + ".linear1", norm(p + ".norm3", fields))))
        }
        fields = norm("field_norm", fields)
        let priorScale = exp(minimum(weights["prior_logit_scale"]!, log(Float(100))))
        let jointScale = exp(minimum(weights["joint_logit_scale"]!, log(Float(100))))
        let gate = sigmoid(weights["residual_gate"]!)
        return options.enumerated().map { i, raw in
            let prior = priorScale * matmul(unit(lex[i]), unit(qv[i] + global))
            let o = norm("option_norm", raw)
            let f = broadcast(fields[i], to: o.shape)
            let cosine = (unit(f, epsilon: 1e-8) * unit(o, epsilon: 1e-8)).sum(axis: -1)
            let features = concatenated([f, o, f * o, abs(f - o)], axis: -1)
            let residual = linear("residual_scorer.3", gelu(linear("residual_scorer.0", features))).squeezed(axis: -1)
            return prior + gate * (jointScale * cosine + residual)
        }
    }
}
