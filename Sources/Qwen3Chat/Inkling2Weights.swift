import Foundation
import MLX

extension Inkling2Model {
    /// Load indexed shards only; optional draft weights are not part of this decoder.
    public static func fromDirectory(_ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil) throws -> Inkling2Model {
        let config = try Inkling2Config(data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
        guard let map = index?["weight_map"] as? [String: String], !map.isEmpty else {
            throw Inkling2Error.invalid("missing or empty safetensors weight index")
        }
        let shards = Set(map.values).sorted()
        var weights: [String: MLXArray] = [:]
        for (i, shard) in shards.enumerated() {
            guard URL(fileURLWithPath: shard).lastPathComponent == shard,
                  !shard.contains("\\"), shard.hasSuffix(".safetensors") else {
                throw Inkling2Error.invalid("invalid shard filename \(shard)")
            }
            let arrays = try MLX.loadArrays(url: directory.appendingPathComponent(shard))
            for (name, value) in arrays {
                guard weights[name] == nil, map[name] == shard else {
                    throw Inkling2Error.invalid("duplicate or incorrectly indexed tensor \(name)")
                }
                weights[name] = value
            }
            eval(Array(arrays.values))
            progressHandler?(Double(i + 1) / Double(shards.count), "Loaded shard \(i + 1)/\(shards.count)")
        }
        guard Set(map.keys) == Set(weights.keys) else { throw Inkling2Error.invalid("indexed tensors are missing") }
        return try Inkling2Model(config: config, weights: weights)
    }

    func check(_ name: String, shape: [Int], packed: Bool = false) throws {
        guard let value = weights[name], value.shape == shape,
              packed ? value.dtype == .uint32 : [.float16, .bfloat16, .float32].contains(value.dtype) else {
            throw Inkling2Error.invalid("missing or invalid \(name); expected \(shape)\(packed ? " UINT32" : " floating point")")
        }
    }

    func checkMatrix(_ name: String, rows: Int, columns: Int, experts: Int? = nil) throws {
        let leading = experts.map { [$0, rows] } ?? [rows]
        if weights[name + ".scales"] != nil {
            let q = config.quantization(for: name)
            guard columns % q.groupSize == 0, columns * q.bits % 32 == 0 else {
                throw Inkling2Error.invalid("invalid quantized dimension for \(name)")
            }
            try check(name + ".weight", shape: leading + [columns * q.bits / 32], packed: true)
            try check(name + ".scales", shape: leading + [columns / q.groupSize])
            try check(name + ".biases", shape: leading + [columns / q.groupSize])
        } else {
            guard weights[name + ".biases"] == nil else { throw Inkling2Error.invalid("missing scales for \(name)") }
            try check(name + ".weight", shape: leading + [columns])
        }
    }

    func validateWeights() throws {
        let c = config, root = Self.base, h = c.hiddenSize
        try checkMatrix(root + "embed_tokens", rows: c.vocabSize, columns: h)
        try checkMatrix("language_model.lm_head", rows: c.vocabSize, columns: h)
        for name in ["embed_norm.weight", "norm.weight", "final_input.query"] { try check(root + name, shape: [h]) }
        try check(root + "message_queries", shape: [c.coreEnd - c.coreStart, 4, h])
        for i in 0..<c.numLayers {
            let p = root + "layers.\(i)", local = c.localLayers.contains(i)
            let heads = local ? c.slidingHeads : c.numHeads
            let kv = local ? c.slidingKVHeads : c.numKVHeads
            let dim = local ? c.slidingHeadDim : c.headDim
            if i > 0 { try check(p + ".attention_input.query", shape: local ? [h] : [3, h]) }
            try check(p + ".mlp_input.query", shape: [h])
            for name in local ? ["input_layernorm"] : ["q_input_layernorm", "k_input_layernorm", "v_input_layernorm"] {
                try check(p + ".\(name).weight", shape: [h])
            }
            try check(p + ".post_attention_layernorm.weight", shape: [h])
            let a = p + ".self_attn"
            for (name, rows, columns) in [("q_proj", heads * dim, h), ("k_proj", kv * dim, h),
                                         ("v_proj", kv * dim, h), ("r_proj", heads * c.relativeDim, h),
                                         ("o_proj", h, heads * dim), ("gate_proj", heads * dim, h), ("sink_proj", heads, h)] {
                try checkMatrix(a + "." + name, rows: rows, columns: columns)
            }
            try check(a + ".q_scale", shape: [heads, dim])
            try check(a + ".rel_scale", shape: [heads, c.relativeDim])
            try check(a + ".rel_proj", shape: [c.relativeDim, local ? c.slidingWindow : c.relativeExtent])
            for name in ["k_sconv", "v_sconv"] {
                if local { try check(a + ".\(name).conv.weight", shape: [kv * dim, c.convolutionSize, 1]) }
                else {
                    try check(a + ".\(name).static_weight", shape: [c.convolutionSize, kv * dim])
                    try checkMatrix(a + ".\(name).weight_proj", rows: kv * c.convolutionSize, columns: h)
                }
            }
            for name in ["attn_sconv", "mlp_sconv"] {
                try check(p + ".\(name).conv.weight", shape: [h, c.convolutionSize, 1])
            }
            try check(p + ".mlp.gate_weight", shape: [c.numExperts, h])
            try check(p + ".mlp.e_score_correction_bias", shape: [c.numExperts])
            for (name, rows, columns) in [("gate_proj", c.intermediateSize, h), ("up_proj", c.intermediateSize, h),
                                         ("down_proj", h, c.intermediateSize)] {
                try checkMatrix(p + ".mlp.switch_mlp." + name, rows: rows, columns: columns, experts: c.numExperts)
            }
            if (c.coreStart..<c.coreEnd).contains(i) {
                for name in ["attn", "k", "v", "mlp"] {
                    let controller = p + ".controllers." + name
                    try check(controller + ".down_weight", shape: [c.gateRank, 2 * h])
                    for projection in ["forget_weight", "gain_weight"] {
                        try check(controller + "." + projection, shape: [h, c.gateRank])
                    }
                }
            }
        }
    }
}
