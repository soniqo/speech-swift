import Foundation
import MLX
import MLXCommon
import MLXNN

final class IndexTTS2Conv1dNCL: Module {
    @ModuleInfo var conv: Conv1d

    init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self._conv = ModuleInfo(wrappedValue: Conv1d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            padding: padding,
            groups: groups,
            bias: bias))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.swappedAxes(1, 2)).swappedAxes(1, 2)
    }
}

final class IndexTTS2SemanticVocosConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwiseConv: IndexTTS2Conv1dNCL
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    @ParameterInfo var gamma: MLXArray

    init(dim: Int = 384, intermediateDim: Int = 2_048) {
        self._depthwiseConv.wrappedValue = IndexTTS2Conv1dNCL(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 7,
            padding: 3,
            groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: true)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim, bias: true)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim, bias: true)
        self._gamma.wrappedValue = MLXArray.ones([dim])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = depthwiseConv(x)
        h = norm(h.transposed(0, 2, 1))
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma.asType(h.dtype) * h
        return residual + h.transposed(0, 2, 1)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyConv1dWeights(
            to: depthwiseConv.conv,
            prefix: "\(prefix).dwconv",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyLayerNormWeights(to: norm, prefix: "\(prefix).norm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: pwconv1, prefix: "\(prefix).pwconv1", from: weights)
        CommonWeightLoader.applyLinearWeights(to: pwconv2, prefix: "\(prefix).pwconv2", from: weights)
        if let gamma = weights["\(prefix).gamma"] {
            update(parameters: ModuleParameters(values: ["gamma": .value(gamma.asType(.float32))]))
        }
    }
}

final class IndexTTS2SemanticVocosBackbone: Module {
    @ModuleInfo var embed: IndexTTS2Conv1dNCL
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var convnext: [IndexTTS2SemanticVocosConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(inputChannels: Int = 1024, dim: Int = 384, intermediateDim: Int = 2_048, numLayers: Int = 12) {
        self._embed.wrappedValue = IndexTTS2Conv1dNCL(
            inputChannels: inputChannels,
            outputChannels: dim,
            kernelSize: 7,
            padding: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: true)
        self._convnext.wrappedValue = (0..<numLayers).map { _ in
            IndexTTS2SemanticVocosConvNeXtBlock(dim: dim, intermediateDim: intermediateDim)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: true)
        super.init()
    }

    /// Input `[B, 1024, T]`, output `[B, T, 384]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embed(x)
        h = norm(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        for block in convnext {
            h = block(h)
        }
        return finalLayerNorm(h.transposed(0, 2, 1))
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyConv1dWeights(
            to: embed.conv,
            prefix: "\(prefix).embed",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyLayerNormWeights(to: norm, prefix: "\(prefix).norm", from: weights)
        for i in convnext.indices {
            convnext[i].loadWeights(prefix: "\(prefix).convnext.\(i)", from: weights)
        }
        CommonWeightLoader.applyLayerNormWeights(
            to: finalLayerNorm,
            prefix: "\(prefix).final_layer_norm",
            from: weights)
    }
}

final class IndexTTS2SemanticEncoder: Module {
    @ModuleInfo var backbone: IndexTTS2SemanticVocosBackbone
    @ModuleInfo var projection: Linear

    override init() {
        self._backbone.wrappedValue = IndexTTS2SemanticVocosBackbone()
        self._projection.wrappedValue = Linear(384, 1024, bias: true)
        super.init()
    }

    static func load(from weights: [String: MLXArray]) throws -> IndexTTS2SemanticEncoder {
        let encoder = IndexTTS2SemanticEncoder()
        encoder.loadWeights(weights)
        encoder.train(false)
        return encoder
    }

    /// Input `[B, T, 1024]`, output `[B, T, 1024]`.
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        projection(backbone(hiddenStates.transposed(0, 2, 1)))
    }

    func loadWeights(_ weights: [String: MLXArray]) {
        backbone.loadWeights(prefix: "encoder.0", from: weights)
        CommonWeightLoader.applyLinearWeights(to: projection, prefix: "encoder.1", from: weights)
    }
}

final class IndexTTS2SemanticQuantizer: Module {
    @ModuleInfo(key: "in_project") var inProject: IndexTTS2Conv1dNCL
    @ModuleInfo(key: "out_project") var outProject: IndexTTS2Conv1dNCL
    @ModuleInfo var codebook: Embedding

    init(inputDim: Int = 1024, codebookSize: Int = 8192, codebookDim: Int = 8) {
        self._inProject = ModuleInfo(
            wrappedValue: IndexTTS2Conv1dNCL(
                inputChannels: inputDim,
                outputChannels: codebookDim,
                kernelSize: 1),
            key: "in_project")
        self._outProject = ModuleInfo(
            wrappedValue: IndexTTS2Conv1dNCL(
                inputChannels: codebookDim,
                outputChannels: inputDim,
                kernelSize: 1),
            key: "out_project")
        self._codebook = ModuleInfo(wrappedValue: Embedding(
            embeddingCount: codebookSize,
            dimensions: codebookDim))
        super.init()
    }

    static func load(from weights: [String: MLXArray]) throws -> IndexTTS2SemanticQuantizer {
        let quantizer = IndexTTS2SemanticQuantizer()
        try quantizer.loadWeights(weights)
        return quantizer
    }

    func loadWeights(_ weights: [String: MLXArray]) throws {
        let prefix = "quantizer.quantizers.0"
        var mapped: [String: MLXArray] = [:]
        if let codebook = weights["\(prefix).codebook.weight"] {
            mapped["codebook.weight"] = codebook.asType(.float32)
        }

        if let weight = Self.fusedWeightNormConv1d(prefix: "\(prefix).in_project", from: weights) {
            mapped["in_project.conv.weight"] = weight
        }
        if let bias = weights["\(prefix).in_project.bias"] {
            mapped["in_project.conv.bias"] = bias.asType(.float32)
        }
        if let weight = Self.fusedWeightNormConv1d(prefix: "\(prefix).out_project", from: weights) {
            mapped["out_project.conv.weight"] = weight
        }
        if let bias = weights["\(prefix).out_project.bias"] {
            mapped["out_project.conv.bias"] = bias.asType(.float32)
        }

        try update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
    }

    /// Codes `[B, T]` -> embeddings `[B, 1024, T]`.
    func vq2Emb(codes: MLXArray) -> MLXArray {
        let emb = codebook(codes).transposed(0, 2, 1)
        return outProject(emb)
    }

    /// Encoded hidden `[B, T, 1024]` -> semantic codes `[B, T]` and prompt embedding `[B, T, 1024]`.
    func quantize(_ encodedHidden: MLXArray) -> (codes: MLXArray, embeddings: MLXArray) {
        let b = encodedHidden.dim(0)
        let t = encodedHidden.dim(1)
        let projected = inProject(encodedHidden.transposed(0, 2, 1))       // [B, 8, T]
        let flat = projected.transposed(0, 2, 1).reshaped([-1, projected.dim(1)])

        let encodings = Self.l2Normalize(flat, axis: 1)
        let table = Self.l2Normalize(codebook.weight.asType(encodedHidden.dtype), axis: 1)
        let tableT = table.transposed(1, 0)
        let scaled = (encodings * encodings).sum(axis: 1, keepDims: true)
        let cross = matmul(encodings, tableT)
        let tableSq = (tableT * tableT).sum(axis: 0, keepDims: true)
        let dist = -(scaled - 2 * cross + tableSq)
        let codes = argMax(dist, axis: -1).asType(.int32).reshaped([b, t])

        let quantized = outProject(codebook(codes).transposed(0, 2, 1)).transposed(0, 2, 1)
        eval(codes, quantized)
        return (codes, quantized)
    }

    private static func fusedWeightNormConv1d(prefix: String, from weights: [String: MLXArray]) -> MLXArray? {
        guard let g = weights["\(prefix).weight_g"],
              let v = weights["\(prefix).weight_v"]
        else {
            return nil
        }
        let v32 = v.asType(.float32)
        let g32 = g.asType(.float32)
        let flat = v32.reshaped([v32.shape[0], -1])
        let norm = sqrt((flat * flat).sum(axis: 1)).reshaped(g32.shape)
        let fused = g32 * (v32 / (norm + MLXArray(Float(1e-9))))
        return fused.transposed(0, 2, 1)
    }

    private static func l2Normalize(_ x: MLXArray, axis: Int) -> MLXArray {
        x / sqrt((x * x).sum(axis: axis, keepDims: true) + MLXArray(Float(1e-12)))
    }
}
