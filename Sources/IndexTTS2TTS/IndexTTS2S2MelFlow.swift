import Foundation
import MLX
import MLXFast
import MLXNN

final class IndexTTS2S2MelFlow {
    private let weights: [String: MLXArray]
    private let config: IndexTTS2RuntimeConfig
    private let hidden = 512
    private let heads = 8
    private let headDim = 64
    private let depth = 13

    init(weights: [String: MLXArray], config: IndexTTS2RuntimeConfig) {
        self.weights = weights
        self.config = config
    }

    func gptLatent(_ latent: MLXArray) -> MLXArray {
        var h = linear(latent, prefix: "gpt_layer.0")
        h = linear(h, prefix: "gpt_layer.1")
        return linear(h, prefix: "gpt_layer.2")
    }

    func inference(
        condition: MLXArray,
        promptMel: MLXArray,
        style: MLXArray,
        steps: Int = 25,
        cfgRate: Float = 0.7,
        temperature: Float = 1.0
    ) -> MLXArray {
        let batch = condition.dim(0)
        let totalFrames = condition.dim(1)
        let promptFrames = promptMel.dim(2)
        var x = MLXRandom.normal(
            [batch, config.s2Mel.nMels, totalFrames],
            key: MLXRandom.key(0))
            .asType(.float32)
        if temperature != 1.0 {
            x = x * MLXArray(temperature)
        }

        let promptX = MLXArray.zeros(x.shape, dtype: x.dtype)
        promptX[0..., 0..., 0..<promptFrames] = promptMel.asType(x.dtype)
        x[0..., 0..., 0..<promptFrames] = MLXArray(Float(0)).asType(x.dtype)

        let lengths = MLXArray([Int32(totalFrames)], [batch])
        var t: Float = 0
        let schedule = (0...steps).map { Float($0) / Float(steps) }

        for step in 1..<schedule.count {
            let nextT = schedule[step]
            let dt = nextT - t
            let velocity: MLXArray
            if cfgRate > 0 {
                let stackedX = concatenated([x, x], axis: 0)
                let zerosPrompt = MLXArray.zeros(promptX.shape, dtype: promptX.dtype)
                let zerosStyle = MLXArray.zeros(style.shape, dtype: style.dtype)
                let zerosCondition = MLXArray.zeros(condition.shape, dtype: condition.dtype)
                let stackedPrompt = concatenated([promptX, zerosPrompt], axis: 0)
                let stackedStyle = concatenated([style, zerosStyle], axis: 0)
                let stackedCondition = concatenated([condition, zerosCondition], axis: 0)
                let stackedLengths = MLXArray([Int32(totalFrames), Int32(totalFrames)], [2])
                let stackedT = MLXArray([t, t], [2])
                let both = estimator(
                    x: stackedX,
                    promptX: stackedPrompt,
                    lengths: stackedLengths,
                    t: stackedT,
                    style: stackedStyle,
                    condition: stackedCondition)
                let cond = both[0..<batch]
                let uncond = both[batch...]
                velocity = (1 + cfgRate) * cond - cfgRate * uncond
            } else {
                velocity = estimator(
                    x: x,
                    promptX: promptX,
                    lengths: lengths,
                    t: MLXArray([t], [1]),
                    style: style,
                    condition: condition)
            }

            x = x + MLXArray(dt).asType(x.dtype) * velocity.asType(x.dtype)
            x[0..., 0..., 0..<promptFrames] = MLXArray(Float(0)).asType(x.dtype)
            t = nextT
            eval(x)
        }

        return x
    }

    private func estimator(
        x: MLXArray,
        promptX: MLXArray,
        lengths: MLXArray,
        t: MLXArray,
        style: MLXArray,
        condition: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let frames = x.dim(2)
        let t1 = timestepEmbedding(t, prefix: "cfm.estimator.t_embedder")
        let cond = linear(condition, prefix: "cfm.estimator.cond_projection")

        let xT = x.transposed(0, 2, 1)
        let promptT = promptX.transposed(0, 2, 1)
        let styleT = broadcast(
            style.expandedDimensions(axis: 1),
            to: [batch, frames, style.dim(1)])
        var h = concatenated([xT, promptT, cond, styleT], axis: -1)
        h = linear(h, prefix: "cfm.estimator.cond_x_merge_linear")

        var xRes = transformer(h, condition: t1.expandedDimensions(axis: 1))
        xRes = linear(concatenated([xRes, xT], axis: -1), prefix: "cfm.estimator.skip_linear")

        var y = linear(xRes, prefix: "cfm.estimator.conv1").transposed(0, 2, 1)
        let t2 = timestepEmbedding(t, prefix: "cfm.estimator.t_embedder2")
        y = wavenet(y, tCondition: t2.expandedDimensions(axis: 2))
            .transposed(0, 2, 1) + linear(xRes, prefix: "cfm.estimator.res_projection")
        y = finalLayer(y, condition: t1).transposed(0, 2, 1)
        return conv1dNCL(
            y,
            weight: weights["cfm.estimator.conv2.weight"]!,
            bias: weights["cfm.estimator.conv2.bias"]!)
    }

    // MARK: - DiT transformer

    private func transformer(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        var h = x
        var skip: [MLXArray] = []
        for i in 0..<depth {
            let skipIn: MLXArray?
            if i > depth / 2 {
                skipIn = skip.popLast()
            } else {
                skipIn = nil
            }
            h = transformerBlock(h, condition: condition, layer: i, skipIn: skipIn)
            if i < depth / 2 {
                skip.append(h)
            }
        }
        return adaptiveRMSNorm(h, condition: condition, prefix: "cfm.estimator.transformer.norm")
    }

    private func transformerBlock(
        _ x: MLXArray,
        condition: MLXArray,
        layer: Int,
        skipIn: MLXArray?
    ) -> MLXArray {
        let prefix = "cfm.estimator.transformer.layers.\(layer)"
        var h = x
        if let skipIn {
            h = linear(concatenated([h, skipIn], axis: -1), prefix: "\(prefix).skip_in_linear")
        }
        h = h + attention(
            adaptiveRMSNorm(h, condition: condition, prefix: "\(prefix).attention_norm"),
            prefix: "\(prefix).attention")
        h = h + feedForward(
            adaptiveRMSNorm(h, condition: condition, prefix: "\(prefix).ffn_norm"),
            prefix: "\(prefix).feed_forward")
        return h
    }

    private func attention(_ x: MLXArray, prefix: String) -> MLXArray {
        let b = x.dim(0)
        let t = x.dim(1)
        let qkv = linear(x, prefix: "\(prefix).wqkv", bias: false)
        let parts = split(qkv, parts: 3, axis: -1)
        var q = parts[0].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        var k = parts[1].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let v = parts[2].reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(
            q,
            dimensions: headDim,
            traditional: true,
            base: 10_000,
            scale: 1.0,
            offset: 0)
        k = MLXFast.RoPE(
            k,
            dimensions: headDim,
            traditional: true,
            base: 10_000,
            scale: 1.0,
            offset: 0)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / Foundation.sqrt(Float(headDim)),
            mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([b, t, hidden])
        return linear(merged, prefix: "\(prefix).wo", bias: false)
    }

    private func feedForward(_ x: MLXArray, prefix: String) -> MLXArray {
        linear(silu(linear(x, prefix: "\(prefix).w1", bias: false)) *
            linear(x, prefix: "\(prefix).w3", bias: false),
            prefix: "\(prefix).w2",
            bias: false)
    }

    private func adaptiveRMSNorm(_ x: MLXArray, condition: MLXArray, prefix: String) -> MLXArray {
        let projected = linear(condition, prefix: "\(prefix).project_layer")
        let parts = split(projected, parts: 2, axis: -1)
        return parts[0] * rmsNorm(x, prefix: "\(prefix).norm") + parts[1]
    }

    private func rmsNorm(_ x: MLXArray, prefix: String, eps: Float = 1e-5) -> MLXArray {
        var y = x / sqrt((x * x).mean(axis: -1, keepDims: true) + MLXArray(eps).asType(x.dtype))
        y = y * weights["\(prefix).weight"]!.asType(y.dtype)
        return y
    }

    // MARK: - WaveNet final layer

    private func wavenet(_ x: MLXArray, tCondition: MLXArray) -> MLXArray {
        let channels = hidden
        var h = x
        var out = MLXArray.zeros(x.shape, dtype: x.dtype)
        let g = weightNormConv1dNCL(
            tCondition,
            prefix: "cfm.estimator.wavenet.cond_layer.conv.conv")

        for i in 0..<8 {
            let xIn = weightNormConv1dNCL(
                h,
                prefix: "cfm.estimator.wavenet.in_layers.\(i).conv.conv",
                padding: 2)
            let gSlice = g[0..., (i * 2 * channels)..<((i + 1) * 2 * channels), 0...]
            let actsIn = xIn + gSlice
            let acts = tanh(actsIn[0..., 0..<channels, 0...]) *
                sigmoid(actsIn[0..., channels..<(2 * channels), 0...])
            let resSkip = weightNormConv1dNCL(
                acts,
                prefix: "cfm.estimator.wavenet.res_skip_layers.\(i).conv.conv")
            if i < 7 {
                let residual = resSkip[0..., 0..<channels, 0...]
                let skip = resSkip[0..., channels..<(2 * channels), 0...]
                h = h + residual
                out = out + skip
            } else {
                out = out + resSkip
            }
        }
        return out
    }

    private func finalLayer(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        let modulation = linear(
            silu(condition),
            prefix: "cfm.estimator.final_layer.adaLN_modulation.1")
        let parts = split(modulation, parts: 2, axis: -1)
        let normed = layerNormNoAffine(x, eps: 1e-6)
        let modulated = normed * (1 + parts[1].expandedDimensions(axis: 1)) +
            parts[0].expandedDimensions(axis: 1)
        return weightNormLinear(modulated, prefix: "cfm.estimator.final_layer.linear")
    }

    private func timestepEmbedding(_ t: MLXArray, prefix: String) -> MLXArray {
        let freqs = weights["\(prefix).freqs"]!.asType(.float32)
        let args = (t.asType(.float32).reshaped([t.dim(0), 1]) * 1000) *
            freqs.reshaped([1, freqs.dim(0)])
        let embedding = concatenated([cos(args), sin(args)], axis: -1)
        return linear(silu(linear(embedding, prefix: "\(prefix).mlp.0")), prefix: "\(prefix).mlp.2")
    }

    // MARK: - primitives

    private func linear(_ x: MLXArray, prefix: String, bias: Bool = true) -> MLXArray {
        let weight = weights["\(prefix).weight"]!.asType(x.dtype)
        var y = matmul(x, weight.transposed(1, 0))
        if bias, let b = weights["\(prefix).bias"] {
            y = y + b.asType(y.dtype)
        }
        return y
    }

    private func weightNormLinear(_ x: MLXArray, prefix: String) -> MLXArray {
        let weight = fusedWeightNormLinear(prefix: prefix).asType(x.dtype)
        var y = matmul(x, weight.transposed(1, 0))
        if let b = weights["\(prefix).bias"] {
            y = y + b.asType(y.dtype)
        }
        return y
    }

    private func fusedWeightNormLinear(prefix: String) -> MLXArray {
        let g = weights["\(prefix).weight_g"]!.asType(.float32)
        let v = weights["\(prefix).weight_v"]!.asType(.float32)
        let norm = sqrt((v * v).sum(axis: 1, keepDims: true) + MLXArray(Float(1e-9)))
        return g * v / norm
    }

    private func layerNormNoAffine(_ x: MLXArray, eps: Float) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let variance = (centered * centered).mean(axis: -1, keepDims: true)
        return centered / sqrt(variance + MLXArray(eps).asType(x.dtype))
    }

    private func conv1dNCL(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray? = nil,
        padding: Int = 0,
        groups: Int = 1
    ) -> MLXArray {
        let weightNLC = weight.asType(x.dtype).transposed(0, 2, 1)
        var y = MLX.conv1d(
            x.transposed(0, 2, 1),
            weightNLC,
            stride: 1,
            padding: padding,
            groups: groups)
            .transposed(0, 2, 1)
        if let bias {
            y = y + bias.asType(y.dtype).reshaped([1, bias.dim(0), 1])
        }
        return y
    }

    private func weightNormConv1dNCL(
        _ x: MLXArray,
        prefix: String,
        padding: Int = 0
    ) -> MLXArray {
        conv1dNCL(
            x,
            weight: fusedWeightNormConv1d(prefix: prefix),
            bias: weights["\(prefix).bias"],
            padding: padding)
    }

    private func fusedWeightNormConv1d(prefix: String) -> MLXArray {
        let g = weights["\(prefix).weight_g"]!.asType(.float32)
        let v = weights["\(prefix).weight_v"]!.asType(.float32)
        let flat = v.reshaped([v.dim(0), -1])
        let norm = sqrt((flat * flat).sum(axis: 1)).reshaped(g.shape) + MLXArray(Float(1e-9))
        return g * v / norm
    }
}
