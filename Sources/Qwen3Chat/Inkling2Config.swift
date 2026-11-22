import Foundation

public enum Inkling2Error: LocalizedError {
    case invalid(String)
    case busy
    public var errorDescription: String? {
        switch self {
        case .invalid(let reason): return "Inkling 2: \(reason)"
        case .busy: return "Inkling 2 is already generating. Use one request per instance."
        }
    }
}

/// Inkling 2 Mini's looped decoder configuration. Repeated visits share weights,
/// but keep independent attention and convolution histories.
public struct Inkling2Config: Sendable {
    public struct Quantization: Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
    }
    public let hiddenSize, numLayers, vocabSize, unpaddedVocabSize: Int
    public let numHeads, numKVHeads, headDim, relativeDim, relativeExtent: Int
    public let slidingHeads, slidingKVHeads, slidingHeadDim, slidingWindow: Int
    public let localLayers: Set<Int>
    public let intermediateSize, numExperts, expertsPerToken, convolutionSize, blockSize: Int
    public let coreStart, coreEnd, numPasses, gateRank: Int
    public let rmsEpsilon, queryClip, relativeClip, routeScale, gateCap, upCap, logitCap: Float
    public let normalizeResidualValues, normalizeValues: Bool
    public let logScalingFloor: Int
    public let logScalingAlpha: Float
    public let maxContext, eosTokenID: Int
    public let defaultQuantization: Quantization
    public let quantizationOverrides: [String: Quantization]

    public var layerVisits: [Int] {
        Array(0..<coreStart) + Array(repeating: Array(coreStart..<coreEnd), count: numPasses).flatMap { $0 }
            + Array(coreEnd..<numLayers)
    }

    public func quantization(for path: String) -> Quantization {
        quantizationOverrides[path] ?? defaultQuantization
    }

    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              root["model_type"] as? String == "inkling2_loop",
              let text = root["text_config"] as? [String: Any],
              let loop = text["loop_config"] as? [String: Any] else {
            throw Inkling2Error.invalid("expected an inkling2_loop config with text_config and loop_config")
        }
        func integer(_ object: [String: Any], _ key: String, _ fallback: Int = 0) -> Int {
            (object[key] as? NSNumber)?.intValue ?? fallback
        }
        func number(_ key: String, _ fallback: Float) -> Float { (text[key] as? NSNumber)?.floatValue ?? fallback }
        hiddenSize = integer(text, "hidden_size"); numLayers = integer(text, "num_hidden_layers")
        vocabSize = integer(text, "vocab_size"); unpaddedVocabSize = integer(text, "unpadded_vocab_size", vocabSize)
        guard (1...16384).contains(hiddenSize), (1...1024).contains(numLayers),
              (1...2000000).contains(vocabSize) else {
            throw Inkling2Error.invalid("unsupported hidden width, layer count, or vocabulary size")
        }
        numHeads = integer(text, "num_attention_heads"); numKVHeads = integer(text, "num_key_value_heads")
        headDim = integer(text, "head_dim"); relativeDim = integer(text, "d_rel", 16)
        relativeExtent = integer(text, "rel_extent", 1024)
        slidingHeads = integer(text, "swa_num_attention_heads", numHeads)
        slidingKVHeads = integer(text, "swa_num_key_value_heads", numKVHeads)
        slidingHeadDim = integer(text, "swa_head_dim", headDim)
        slidingWindow = integer(text, "sliding_window_size", 1024)
        localLayers = Set(text["local_layer_ids"] as? [Int] ?? (0..<max(0, numLayers)).filter { ($0 + 1) % 6 != 0 })
        intermediateSize = integer(text, "intermediate_size"); numExperts = integer(text, "n_routed_experts")
        expertsPerToken = integer(text, "num_experts_per_tok"); convolutionSize = integer(text, "sconv_kernel_size", 4)
        blockSize = integer(text, "n_layers_per_attn_res_block", 6)
        coreStart = integer(loop, "core_start"); coreEnd = integer(loop, "core_end")
        numPasses = integer(loop, "num_passes"); gateRank = integer(loop, "gate_rank")
        rmsEpsilon = number("rms_norm_eps", 1e-6); queryClip = number("qk_l2_clip_threshold", 22.627417)
        relativeClip = number("rel_l2_clip_threshold", 8); routeScale = number("route_scale", 8)
        gateCap = number("mlp_gate_softcap", 4); upCap = number("mlp_up_softcap", 25)
        logitCap = number("final_logit_softcapping", 30)
        normalizeResidualValues = text["attn_res_normalize_values"] as? Bool ?? true
        normalizeValues = text["use_value_norm_no_gamma"] as? Bool ?? true
        logScalingFloor = integer(text, "log_scaling_n_floor", 8192)
        logScalingAlpha = number("log_scaling_alpha", 0.1)
        maxContext = integer(text, "model_max_length", integer(text, "max_position_embeddings", 1048576))
        eosTokenID = integer(root, "eos_token_id", 200006)
        func quant(_ object: [String: Any]) throws -> Quantization {
            let mode = object["mode"] as? String ?? "affine"
            let bits = integer(object, "bits", 4), group = integer(object, "group_size", 64)
            guard mode == "affine", [4, 5, 8].contains(bits), [32, 64, 128].contains(group) else {
                throw Inkling2Error.invalid("requires affine INT4/INT5/INT8 weights from export_inkling_mlx.py")
            }
            return Quantization(bits: bits, groupSize: group)
        }
        let q = root["quantization"] as? [String: Any] ?? root["quantization_config"] as? [String: Any] ?? [:]
        guard q["quant_method"] == nil else { throw Inkling2Error.invalid("export the upstream checkpoint to MLX first") }
        defaultQuantization = try quant(q)
        var overrides: [String: Quantization] = [:]
        for (path, value) in q where value is [String: Any] { overrides[path] = try quant(value as! [String: Any]) }
        quantizationOverrides = overrides
        guard hiddenSize > 0, numLayers > 0, vocabSize > 0, unpaddedVocabSize > 0, unpaddedVocabSize <= vocabSize,
              numHeads > 0, numKVHeads > 0, numHeads % numKVHeads == 0,
              slidingHeads > 0, slidingKVHeads > 0, slidingHeads % slidingKVHeads == 0,
              headDim > 0, slidingHeadDim > 0, relativeDim > 0, relativeExtent > 0, slidingWindow > 0,
              numExperts > 0, expertsPerToken > 0, expertsPerToken <= numExperts, intermediateSize > 0,
              convolutionSize > 1, blockSize > 0, coreStart > 0, coreStart < coreEnd, coreEnd <= numLayers,
              coreStart % blockSize == 0, coreEnd % blockSize == 0, numPasses > 0, numPasses <= 16,
              gateRank > 0, loop["is_varloop"] as? Bool != true,
              integer(text, "n_shared_experts") == 0, maxContext > 0,
              logScalingAlpha.isFinite, logScalingAlpha >= 0, logScalingFloor >= 0,
              logitCap.isFinite, logitCap >= 0,
              [rmsEpsilon, queryClip, relativeClip, routeScale, gateCap, upCap].allSatisfy({ $0.isFinite && $0 > 0 }),
              localLayers.allSatisfy({ (0..<numLayers).contains($0) }) else {
            throw Inkling2Error.invalid("unsupported decoder dimensions, routing, or loop configuration")
        }
    }
}
