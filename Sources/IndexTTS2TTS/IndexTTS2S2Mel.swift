import Foundation
import MLX
import MLXCommon
import MLXNN

private func indexTTS2Mish(_ x: MLXArray) -> MLXArray {
    x * tanh(log1p(exp(x)))
}

final class IndexTTS2S2MelLengthRegulator: Module {
    @ModuleInfo(key: "content_in_proj") var contentInProj: Linear
    @ModuleInfo var convs: [IndexTTS2Conv1dNCL]
    @ModuleInfo var norms: [GroupNorm]
    @ModuleInfo var outputConv: IndexTTS2Conv1dNCL

    override init() {
        _contentInProj.wrappedValue = Linear(1024, 512, bias: true)
        _convs.wrappedValue = (0..<4).map { _ in
            IndexTTS2Conv1dNCL(
                inputChannels: 512,
                outputChannels: 512,
                kernelSize: 3,
                padding: 1)
        }
        _norms.wrappedValue = (0..<4).map { _ in
            GroupNorm(
                groupCount: 1,
                dimensions: 512,
                eps: 1e-5,
                affine: true,
                pytorchCompatible: true)
        }
        _outputConv.wrappedValue = IndexTTS2Conv1dNCL(
            inputChannels: 512,
            outputChannels: 512,
            kernelSize: 1)
        super.init()
    }

    static func load(from weights: [String: MLXArray]) throws -> IndexTTS2S2MelLengthRegulator {
        let regulator = IndexTTS2S2MelLengthRegulator()
        regulator.loadWeights(weights)
        regulator.train(false)
        return regulator
    }

    /// Semantic prompt `[B, T, 1024]` -> S2Mel prompt condition `[B, targetLength, 512]`.
    func callAsFunction(_ semanticPrompt: MLXArray, targetLength: Int) -> MLXArray {
        var h = contentInProj(semanticPrompt).transposed(0, 2, 1)    // [B, 512, T]
        h = Self.interpolateNearestNCL(h, targetLength: targetLength)
        for i in convs.indices {
            h = convs[i](h)
            h = norms[i](h.transposed(0, 2, 1)).transposed(0, 2, 1)
            h = indexTTS2Mish(h)
        }
        let out = outputConv(h).transposed(0, 2, 1)
        eval(out)
        return out
    }

    private func loadWeights(_ weights: [String: MLXArray]) {
        let prefix = "length_regulator"
        CommonWeightLoader.applyLinearWeights(to: contentInProj, prefix: "\(prefix).content_in_proj", from: weights)

        let convIndices = [0, 3, 6, 9]
        let normIndices = [1, 4, 7, 10]
        for i in convs.indices {
            CommonWeightLoader.applyConv1dWeights(
                to: convs[i].conv,
                prefix: "\(prefix).model.\(convIndices[i])",
                from: weights,
                transpose: true)
            Self.applyGroupNormWeights(to: norms[i], prefix: "\(prefix).model.\(normIndices[i])", from: weights)
        }
        CommonWeightLoader.applyConv1dWeights(
            to: outputConv.conv,
            prefix: "\(prefix).model.12",
            from: weights,
            transpose: true)
    }

    private static func interpolateNearestNCL(_ x: MLXArray, targetLength: Int) -> MLXArray {
        let inputLength = x.dim(2)
        guard targetLength != inputLength else { return x }
        guard targetLength > 0, inputLength > 0 else { return x }
        let scale = Float(inputLength) / Float(targetLength)
        let indices = (0..<targetLength).map { i in
            Int32(min(Int(floorf(Float(i) * scale)), inputLength - 1))
        }
        return x.take(MLXArray(indices), axis: 2)
    }

    private static func applyGroupNormWeights(
        to groupNorm: GroupNorm,
        prefix: String,
        from weights: [String: MLXArray]
    ) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let weight = weights["\(prefix).weight"] {
            params["weight"] = .value(weight.asType(.float32))
        }
        if let bias = weights["\(prefix).bias"] {
            params["bias"] = .value(bias.asType(.float32))
        }
        if !params.isEmpty {
            groupNorm.update(parameters: ModuleParameters(values: params))
        }
    }
}
