import Foundation
import MLX
import MLXCommon
import MLXNN

final class MioGlobalConvNeXtBlock: Module {
    @ModuleInfo(key: "dwconv") var depthwiseConv: MioConv1dNCL
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var pwconv1: Linear
    @ModuleInfo var pwconv2: Linear
    @ParameterInfo var gamma: MLXArray

    init(dim: Int = 384, intermediateDim: Int = 1_152) {
        self._depthwiseConv.wrappedValue = MioConv1dNCL(
            inputChannels: dim,
            outputChannels: dim,
            kernelSize: 7,
            padding: 3,
            groups: dim)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._pwconv1.wrappedValue = Linear(dim, intermediateDim, bias: true)
        self._pwconv2.wrappedValue = Linear(intermediateDim, dim, bias: true)
        self._gamma = ParameterInfo(wrappedValue: MLXArray.ones([dim]))
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = depthwiseConv(x)
        h = h.transposed(0, 2, 1)
        h = norm(h)
        h = pwconv1(h)
        h = gelu(h)
        h = pwconv2(h)
        h = gamma * h
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
            update(parameters: ModuleParameters(values: ["gamma": .value(gamma)]))
        }
    }
}

final class MioGlobalConvNeXtBackbone: Module {
    @ModuleInfo var embed: MioConv1dNCL
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var convnext: [MioGlobalConvNeXtBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(inputChannels: Int = 768, dim: Int = 384, intermediateDim: Int = 1_152, numLayers: Int = 4) {
        self._embed.wrappedValue = MioConv1dNCL(
            inputChannels: inputChannels,
            outputChannels: dim,
            kernelSize: 7,
            padding: 3)
        self._norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._convnext.wrappedValue = (0..<numLayers).map { _ in
            MioGlobalConvNeXtBlock(dim: dim, intermediateDim: intermediateDim)
        }
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        super.init()
    }

    /// Input `[B, T, C]`, output `[B, T, 384]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = embed(x.transposed(0, 2, 1))
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

final class MioAttentiveStatsPool: Module {
    @ModuleInfo(key: "attn_0") var attn0: MioConv1dNCL
    @ModuleInfo(key: "attn_2") var attn2: MioConv1dNCL
    @ModuleInfo var proj: Linear
    @ModuleInfo var norm: LayerNorm

    init(inputChannels: Int = 384, outputChannels: Int = 128, attentionChannels: Int = 128) {
        self._attn0.wrappedValue = MioConv1dNCL(
            inputChannels: inputChannels,
            outputChannels: attentionChannels,
            kernelSize: 1)
        self._attn2.wrappedValue = MioConv1dNCL(
            inputChannels: attentionChannels,
            outputChannels: inputChannels,
            kernelSize: 1)
        self._proj.wrappedValue = Linear(inputChannels * 2, outputChannels, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: outputChannels, eps: 1e-5)
        super.init()
    }

    /// Input `[B, C, T]`, output `[B, 128]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var alpha = attn0(x)
        alpha = MLX.tanh(alpha)
        alpha = attn2(alpha)
        alpha = MLX.softmax(alpha, axis: 2)

        let mean = MLX.sum(alpha * x, axis: 2)
        let secondMoment = MLX.sum(alpha * x * x, axis: 2)
        let residuals = MLX.clip(secondMoment - mean * mean, min: MLXArray(Float(1e-4)), max: MLXArray(Float(1e4)))
        let std = MLX.sqrt(residuals)
        return norm(proj(concatenated([mean, std], axis: 1)))
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyConv1dWeights(
            to: attn0.conv,
            prefix: "\(prefix).attn.0",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyConv1dWeights(
            to: attn2.conv,
            prefix: "\(prefix).attn.2",
            from: weights,
            transpose: true)
        CommonWeightLoader.applyLinearWeights(to: proj, prefix: "\(prefix).proj", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: norm, prefix: "\(prefix).norm", from: weights)
    }
}

public final class MioCodecGlobalEncoder: Module {
    @ModuleInfo var backbone: MioGlobalConvNeXtBackbone
    @ModuleInfo var pooling: MioAttentiveStatsPool

    public override init() {
        self._backbone.wrappedValue = MioGlobalConvNeXtBackbone()
        self._pooling.wrappedValue = MioAttentiveStatsPool()
        super.init()
    }

    /// Input `[B, T, 768]` averaged WavLM features, output `[B, 128]`.
    public func callAsFunction(_ sslFeatures: MLXArray) -> MLXArray {
        let features = backbone(sslFeatures)
        return pooling(features.transposed(0, 2, 1))
    }

    public func loadWeights(from weights: [String: MLXArray]) {
        backbone.loadWeights(prefix: "global_encoder.backbone", from: weights)
        pooling.loadWeights(prefix: "global_encoder.pooling", from: weights)
        eval(self)
    }
}
