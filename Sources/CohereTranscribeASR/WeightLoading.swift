// Portions adapted from mlx-audio-swift (MIT License).
// Copyright (c) 2025 Blaizzy and contributors.

import MLX

private let cohereWeightPrefixAliases: [(source: String, target: String)] = [
    ("encoder.pre_encode.", "encoder.subsampling."),
    ("encoder_decoder_proj.", "bridge_proj."),
    ("log_softmax.mlp.layer0.", "lm_head."),
    ("transf_decoder.embedding.", "decoder.embedding."),
    ("transf_decoder._embedding.", "decoder.embedding."),
    ("transf_decoder.decoder.", "decoder.core."),
    ("transf_decoder._decoder.", "decoder.core."),
]

private func qkvPart(for key: String) -> (prefix: String, part: String, suffix: String)? {
    for (needle, part) in [
        (".linear_q.", "q"), (".linear_k.", "k"), (".linear_v.", "v"),
        (".query_net.", "q"), (".key_net.", "k"), (".value_net.", "v"),
    ] {
        guard let range = key.range(of: needle) else { continue }
        let suffix = String(key[range.upperBound...])
        guard ["weight", "bias", "scales", "biases"].contains(suffix) else { continue }
        return (String(key[..<range.lowerBound]), part, suffix)
    }
    return nil
}

private func mapCohereWeightName(_ key: String) -> String {
    let mapped = cohereWeightPrefixAliases.first(where: { key.hasPrefix($0.source) }).map {
        key.replacingOccurrences(of: $0.source, with: $0.target)
    } ?? key
    return mapped
        .replacingOccurrences(of: "self_attn.linear_out.", with: "self_attn.out_proj.")
        .replacingOccurrences(of: "self_attn.linear_pos.", with: "self_attn.pos_proj.")
        .replacingOccurrences(of: "first_sub_layer.out_projection.", with: "first_sub_layer.out_proj.")
        .replacingOccurrences(of: "second_sub_layer.out_projection.", with: "second_sub_layer.out_proj.")
}

func normalizeCohereWeightKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let substitutions = [
        "encoder.subsampling.conv.0.": "encoder.subsampling.conv0.",
        "encoder.subsampling.conv.2.": "encoder.subsampling.conv2.",
        "encoder.subsampling.conv.3.": "encoder.subsampling.conv3.",
        "encoder.subsampling.conv.5.": "encoder.subsampling.conv5.",
        "encoder.subsampling.conv.6.": "encoder.subsampling.conv6.",
    ]
    let subsamplingKernels = [
        "encoder.subsampling.conv0.weight": [3, 3],
        "encoder.subsampling.conv2.weight": [3, 3],
        "encoder.subsampling.conv3.weight": [1, 1],
        "encoder.subsampling.conv5.weight": [3, 3],
        "encoder.subsampling.conv6.weight": [1, 1],
    ]
    var normalized: [String: MLXArray] = [:]
    var pendingQKV: [String: [String: MLXArray]] = [:]

    for (key, value) in weights {
        if key.hasSuffix(".num_batches_tracked") || key.hasPrefix("preprocessor.") { continue }
        if let part = qkvPart(for: key) {
            let target = "\(mapCohereWeightName(part.prefix)).qkv_proj.\(part.suffix)"
            pendingQKV[target, default: [:]][part.part] = value
            continue
        }

        let initiallyMapped = mapCohereWeightName(key)
        let mapped = substitutions.first(where: { initiallyMapped.hasPrefix($0.key) }).map {
            initiallyMapped.replacingOccurrences(of: $0.key, with: $0.value)
        } ?? initiallyMapped
        if mapped.hasPrefix("decoder.embedding.position_embedding") { continue }

        if mapped.hasSuffix(".weight"), value.ndim == 4,
           let kernel = subsamplingKernels[mapped] {
            if value.shape[1] == kernel[0], value.shape[2] == kernel[1] {
                normalized[mapped] = value
            } else if value.shape[2] == kernel[0], value.shape[3] == kernel[1] {
                normalized[mapped] = value.transposed(0, 2, 3, 1)
            } else {
                normalized[mapped] = value
            }
        } else if mapped.hasSuffix(".weight"), value.ndim == 3, mapped.contains(".conv.") {
            let pytorchLayout = mapped.contains("depthwise_conv")
                ? value.shape[1] == 1 && value.shape[2] > 1
                : value.shape[2] == 1 && value.shape[1] > 1
            normalized[mapped] = pytorchLayout ? value.transposed(0, 2, 1) : value
        } else {
            normalized[mapped] = value
        }
    }

    for (key, parts) in pendingQKV {
        guard let query = parts["q"], let keyPart = parts["k"], let value = parts["v"] else {
            continue
        }
        normalized[key] = MLX.concatenated([query, keyPart, value], axis: 0)
    }
    return normalized
}
