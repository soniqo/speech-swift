import Foundation
import MLX
import MLXNN

/// Build a Linear layer that is quantized when bits > 0, plain when bits == 0.
/// QuantizedLinear inherits from Linear, so the return type is always Linear and
/// caller code can store it in a single `@ModuleInfo var x: Linear` field.
public func makeMaybeQuantizedLinear(
    _ inputDimensions: Int,
    _ outputDimensions: Int,
    bias: Bool,
    groupSize: Int,
    bits: Int
) -> Linear {
    if bits > 0 {
        return QuantizedLinear(inputDimensions, outputDimensions, bias: bias,
                               groupSize: groupSize, bits: bits)
    } else {
        return Linear(inputDimensions, outputDimensions, bias: bias)
    }
}

/// Generic weight loading utilities shared between ASR and TTS
public enum CommonWeightLoader {

    /// Load weights from safetensors file
    public static func loadSafetensors(url: URL) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: url)
    }

    /// Load all safetensors from a directory, optionally filtering by prefix
    public static func loadAllSafetensors(
        from directory: URL,
        prefix: String? = nil,
        stripPrefix: Bool = true
    ) throws -> [String: MLXArray] {
        let fileManager = FileManager.default
        let contents = try fileManager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw WeightLoadingError.noWeightsFound(directory)
        }

        var allWeights: [String: MLXArray] = [:]
        for file in safetensorFiles {
            let weights = try loadSafetensors(url: file)
            allWeights.merge(weights) { _, new in new }
        }

        // Filter and strip prefix if specified
        guard let prefix = prefix else { return allWeights }

        var filtered: [String: MLXArray] = [:]
        for (key, value) in allWeights {
            if key.hasPrefix(prefix) {
                let strippedKey = stripPrefix ? String(key.dropFirst(prefix.count)) : key
                filtered[strippedKey] = value
            }
        }
        return filtered
    }

    // MARK: - Quantized Weight Application Helpers

    public static func applyQuantizedEmbeddingWeights(
        to embedding: PreQuantizedEmbedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let scales = weights["\(prefix).scales"] {
            params["scales"] = .value(scales)
        }
        if let biases = weights["\(prefix).biases"] {
            params["biases"] = .value(biases)
        }

        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    // MARK: - Checked quantized weight application

    /// Apply quantized embedding weights, failing closed when the checkpoint's packed
    /// layout disagrees with the bit width / group size the layer was built with.
    ///
    /// See ``QuantizedWeightMismatch`` for why this check has to be done here.
    public static func applyCheckedQuantizedEmbeddingWeights(
        to embedding: PreQuantizedEmbedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        try verifyQuantizedLayout(
            tensor: prefix,
            rows: embedding.embeddingCount,
            columns: embedding.dimensions,
            bits: embedding.bits,
            groupSize: embedding.groupSize,
            weight: weights["\(prefix).weight"],
            scales: weights["\(prefix).scales"])
        applyQuantizedEmbeddingWeights(to: embedding, prefix: prefix, from: weights)
    }

    /// Apply quantized linear weights, failing closed when the checkpoint's packed
    /// layout disagrees with the bit width / group size the layer was built with.
    ///
    /// See ``QuantizedWeightMismatch`` for why this check has to be done here.
    public static func applyCheckedQuantizedLinearWeights(
        to linear: QuantizedLinear,
        prefix: String,
        from weights: [String: MLXArray]
    ) throws {
        // `weight` is packed: [outputDimensions, inputDimensions * bits / 32].
        let packed = linear.weight.shape2
        try verifyQuantizedLayout(
            tensor: prefix,
            rows: packed.0,
            columns: packed.1 * 32 / linear.bits,
            bits: linear.bits,
            groupSize: linear.groupSize,
            weight: weights["\(prefix).weight"],
            scales: weights["\(prefix).scales"])
        applyQuantizedLinearWeights(to: linear, prefix: prefix, from: weights)
    }

    /// Compare a checkpoint's `weight`/`scales` pair against the layout implied by
    /// `rows`/`columns`/`bits`/`groupSize` — i.e. a packed weight of
    /// `[rows, columns * bits / 32]` and scales of `[rows, columns / groupSize]`.
    static func verifyQuantizedLayout(
        tensor: String,
        rows: Int,
        columns: Int,
        bits: Int,
        groupSize: Int,
        weight: MLXArray?,
        scales: MLXArray?
    ) throws {
        guard let weight else {
            throw QuantizedWeightMismatch.missingWeight(tensor: tensor)
        }
        guard let scales else {
            // A packed weight cannot be dequantized without its scales; loading it
            // alone leaves the layer multiplying by zeros.
            throw QuantizedWeightMismatch.missingScales(tensor: tensor)
        }

        let expectedWeight = [rows, columns * bits / 32]
        if weight.shape != expectedWeight {
            // A row of the wrong width almost always means the checkpoint was packed
            // at a different bit width than the config declared.
            if weight.ndim == 2, weight.dim(0) == rows, columns > 0 {
                let packedBitsTotal = weight.dim(1) * 32
                if packedBitsTotal % columns == 0 {
                    throw QuantizedWeightMismatch.bits(
                        tensor: tensor, declared: bits, implied: packedBitsTotal / columns)
                }
            }
            throw QuantizedWeightMismatch.shape(
                tensor: "\(tensor).weight", expected: expectedWeight, actual: weight.shape)
        }

        let expectedScales = [rows, columns / groupSize]
        if scales.shape != expectedScales {
            if scales.ndim == 2, scales.dim(0) == rows, scales.dim(1) > 0,
               columns % scales.dim(1) == 0 {
                throw QuantizedWeightMismatch.groupSize(
                    tensor: tensor, declared: groupSize, implied: columns / scales.dim(1))
            }
            throw QuantizedWeightMismatch.shape(
                tensor: "\(tensor).scales", expected: expectedScales, actual: scales.shape)
        }
    }

    /// Apply weights to a Linear (or QuantizedLinear, since the latter inherits from
    /// the former). When the layer is a QuantizedLinear, `.scales`/`.biases` are wired
    /// in addition to `.weight`. For plain Linear those keys are absent in the
    /// safetensors (bf16/fp32 model) and only `.weight` (+ optional `.bias`) apply.
    public static func applyQuantizedLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if linear is QuantizedLinear {
            if let scales = weights["\(prefix).scales"] {
                params["scales"] = .value(scales)
            }
            if let biases = weights["\(prefix).biases"] {
                params["biases"] = .value(biases)
            }
        }
        // Regular linear bias (separate from quantization biases)
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyRMSNormWeights(
        to norm: RMSNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            norm.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            linear.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyLayerNormWeights(
        to layerNorm: LayerNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            layerNorm.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyEmbeddingWeights(
        to embedding: Embedding,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight)
        }

        if !params.isEmpty {
            embedding.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConv1dWeights(
        to conv: Conv1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch Conv1d: [out, in, kernel] -> MLX Conv1d: [out, kernel, in]
            let w = transpose ? weight.transposed(0, 2, 1) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    public static func applyConvTransposed1dWeights(
        to conv: ConvTransposed1d,
        prefix: String,
        from weights: [String: MLXArray],
        transpose: Bool = false
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]

        if let weight = weights["\(prefix).weight"] {
            // PyTorch ConvTranspose1d: [in, out, kernel] -> MLX ConvTransposed1d: [out, kernel, in]
            let w = transpose ? weight.transposed(1, 2, 0) : weight
            params["weight"] = .value(w)
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias)
        }

        if !params.isEmpty {
            conv.update(parameters: ModuleParameters(values: params))
        }
    }

    /// Apply QuantizedMLP weights (SwiGLU)
    public static func applyQuantizedMLPWeights(
        to mlp: QuantizedMLP,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(prefix).gate_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.upProj, prefix: "\(prefix).up_proj", from: weights)
        applyQuantizedLinearWeights(to: mlp.downProj, prefix: "\(prefix).down_proj", from: weights)
    }

    /// Apply MLP weights (SwiGLU) — dispatches to quantized or plain
    /// projections per-leaf based on `.scales` presence. Use when the
    /// surrounding module was declared with `Linear` and may have been
    /// swapped to `QuantizedLinear` by `quantize(model:filter:)`.
    public static func applyMLPWeights(
        to mlp: MLP,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        applyMaybeQuantizedLinearWeights(to: mlp.gateProj, prefix: "\(prefix).gate_proj", from: weights)
        applyMaybeQuantizedLinearWeights(to: mlp.upProj, prefix: "\(prefix).up_proj", from: weights)
        applyMaybeQuantizedLinearWeights(to: mlp.downProj, prefix: "\(prefix).down_proj", from: weights)
    }

    /// Apply weights to a `Linear` that may have been swapped to
    /// `QuantizedLinear`. Picks the right keys (`weight`, optional `bias`
    /// for plain Linear; plus `scales`, `biases` when quantized) based on
    /// what is present in the safetensors.
    public static func applyMaybeQuantizedLinearWeights(
        to linear: Linear,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        if weights["\(prefix).scales"] != nil, let q = linear as? QuantizedLinear {
            applyQuantizedLinearWeights(to: q, prefix: prefix, from: weights)
        } else {
            applyLinearWeights(to: linear, prefix: prefix, from: weights)
        }
    }
}

/// A checkpoint's quantized tensors disagreeing with the layer receiving them.
///
/// Nothing below this raises on its own: `Module.update(parameters:)` calls
/// `update(parameters:verify:)` with `verify: .none`, so mlx-swift installs a packed
/// row of the wrong width without complaint and the layer then dequantizes garbage —
/// a wrong answer instead of an error. An INT8 checkpoint read as INT4 is exactly
/// that case: same tensor names, same row count, twice the packed width.
public enum QuantizedWeightMismatch: Error, LocalizedError {
    case missingWeight(tensor: String)
    case missingScales(tensor: String)
    case bits(tensor: String, declared: Int, implied: Int)
    case groupSize(tensor: String, declared: Int, implied: Int)
    case shape(tensor: String, expected: [Int], actual: [Int])

    public var errorDescription: String? {
        switch self {
        case .missingWeight(let tensor):
            return "Quantized weight missing from checkpoint: \(tensor).weight"
        case .missingScales(let tensor):
            return "Quantized weight \(tensor).weight has no matching \(tensor).scales"
        case .bits(let tensor, let declared, let implied):
            return """
                Quantization mismatch at \(tensor): config declares \(declared)-bit weights \
                but the checkpoint is packed \(implied)-bit. Fix the model config's \
                quantization_bits rather than loading it as \(declared)-bit.
                """
        case .groupSize(let tensor, let declared, let implied):
            return """
                Quantization mismatch at \(tensor): config declares group size \(declared) \
                but the checkpoint's scales imply \(implied). Fix the model config's \
                quantization_group_size.
                """
        case .shape(let tensor, let expected, let actual):
            return "Weight shape mismatch at \(tensor): expected \(expected), got \(actual)"
        }
    }
}

/// Weight loading errors
public enum WeightLoadingError: Error, LocalizedError {
    case noWeightsFound(URL)
    case incompatibleWeights(String)
    case missingRequiredWeight(String)

    public var errorDescription: String? {
        switch self {
        case .noWeightsFound(let url):
            return "No safetensors files found in: \(url.path)"
        case .incompatibleWeights(let reason):
            return "Incompatible weights: \(reason)"
        case .missingRequiredWeight(let key):
            return "Missing required weight: \(key)"
        }
    }
}
