import Foundation
import MLX
import MLXCommon
import MLXFast
import MLXNN

/// Higgs codec encode path (waveform → codes) for reference cloning:
/// a HuBERT semantic model over 16 kHz audio (mean of all 13 hidden states,
/// stride-sliced 50→25 fps, refined by a small CNN), a DAC-style acoustic
/// encoder over the 24 kHz waveform, concat + fusion projection, and greedy
/// 8-layer residual vector quantization.
extension HiggsTTSCodec {
    static let semanticSampleRate = 16_000
    /// HuBERT conv downsample factor is 320; the reference pads half of it
    /// on each side before the semantic model.
    static let hubertPad = 160

    /// Encodes a 24 kHz mono waveform into raw codec codes `[T][8]`.
    func encode(waveform24k: [Float]) throws -> [[Int32]] {
        for key in Self.requiredEncodeKeys {
            guard weights[key] != nil else {
                throw HiggsTTSError.missingRequiredFile("codec tensor \(key)")
            }
        }
        guard !waveform24k.isEmpty else {
            throw HiggsTTSError.invalidCodes("empty reference waveform")
        }

        let audio16 = Self.sincResample(waveform24k, from: 24_000, to: Self.semanticSampleRate)
        let padded = [Float](repeating: 0, count: Self.hubertPad) + audio16
            + [Float](repeating: 0, count: Self.hubertPad)

        var semantic = hubertMeanHiddenStates(MLXArray(padded).reshaped(1, padded.count))
        let strided = MLXArray(
            Swift.stride(from: 0, to: semantic.dim(1), by: 2).map { Int32($0) })
        semantic = take(semantic, strided, axis: 1)
        semantic = semanticEncoder(semantic)

        let acoustic = acousticEncoder(
            MLXArray(waveform24k).reshaped(1, waveform24k.count, 1))

        let frames = min(semantic.dim(1), acoustic.dim(1))
        let fusion = concatenated(
            [acoustic[0..., 0..<frames, 0...], semantic[0..., 0..<frames, 0...]],
            axis: -1)
        let z = linear(fusion, prefix: "fc")
        return rvqEncode(z)
    }

    // MARK: - HuBERT semantic model

    private func hubertMeanHiddenStates(_ audio: MLXArray) -> MLXArray {
        // Feature extractor: 7 convs, strides [5,2,2,2,2,2,2], kernels [10,3,3,3,3,2,2].
        var x = audio.reshaped(audio.dim(0), audio.dim(1), 1)
        let strides = [5, 2, 2, 2, 2, 2, 2]
        for (index, stride) in strides.enumerated() {
            x = conv(x, prefix: "semantic_model.feature_extractor.conv_layers.\(index).conv",
                     padding: 0, stride: stride)
            if index == 0 {
                x = perChannelGroupNorm(
                    x, prefix: "semantic_model.feature_extractor.conv_layers.0.layer_norm")
            }
            x = geluErf(x)
        }

        x = layerNorm(x, prefix: "semantic_model.feature_projection.layer_norm")
        x = linear(x, prefix: "semantic_model.feature_projection.projection")

        // Positional conv embedding (weight-normed grouped conv, k=128, pad 64,
        // 16 groups; even kernel drops the trailing frame), then pre-encoder norm.
        let g = weights["semantic_model.encoder.pos_conv_embed.conv.weight_g"]!
        let v = weights["semantic_model.encoder.pos_conv_embed.conv.weight_v"]!
        let norm = sqrt(sum(v * v, axes: [0, 2], keepDims: true))
        let posWeight = g * v / norm
        var pos = conv1d(x, posWeight, stride: 1, padding: 64, groups: 16)
        pos = pos + weights["semantic_model.encoder.pos_conv_embed.conv.bias"]!
        pos = pos[0..., 0..<(pos.dim(1) - 1), 0...]
        x = x + geluErf(pos)
        x = layerNorm(x, prefix: "semantic_model.encoder.layer_norm")

        var sum = x
        for layer in 0..<12 {
            x = hubertLayer(x, prefix: "semantic_model.encoder.layers.\(layer)")
            sum = sum + x
        }
        return sum / 13.0
    }

    private func hubertLayer(_ input: MLXArray, prefix: String) -> MLXArray {
        let batch = input.dim(0)
        let frames = input.dim(1)
        let heads = 12
        let headDim = 64

        func project(_ name: String) -> MLXArray {
            linear(input, prefix: "\(prefix).attention.\(name)")
                .reshaped(batch, frames, heads, headDim)
                .transposed(0, 2, 1, 3)
        }
        let attnOut = SDPA.attendAndMerge(
            qHeads: project("q_proj"),
            kHeads: project("k_proj"),
            vHeads: project("v_proj"),
            scale: 1.0 / Float(headDim).squareRoot(),
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        var x = input + linear(attnOut, prefix: "\(prefix).attention.out_proj")
        x = layerNorm(x, prefix: "\(prefix).layer_norm")

        var ff = linear(x, prefix: "\(prefix).feed_forward.intermediate_dense")
        ff = geluErf(ff)
        ff = linear(ff, prefix: "\(prefix).feed_forward.output_dense")
        x = x + ff
        return layerNorm(x, prefix: "\(prefix).final_layer_norm")
    }

    // MARK: - Semantic refinement CNN

    private func semanticEncoder(_ input: MLXArray) -> MLXArray {
        var x = conv(input, prefix: "encoder_semantic.conv", padding: 1)
        for block in 0..<2 {
            let prefix = "encoder_semantic.conv_blocks.\(block)"
            for unit in 0..<2 {
                let unitPrefix = "\(prefix).res_units.\(unit)"
                var y = elu(x)
                y = conv(y, prefix: "\(unitPrefix).conv1", padding: 1)
                y = elu(y)
                y = conv(y, prefix: "\(unitPrefix).conv2", padding: 0)
                x = x + y
            }
            x = conv(x, prefix: "\(prefix).conv", padding: 1)
        }
        return x
    }

    // MARK: - Acoustic encoder

    private func acousticEncoder(_ input: MLXArray) -> MLXArray {
        var x = conv(input, prefix: "acoustic_encoder.conv1", padding: 3)
        for (index, stride) in [8, 5, 4, 2, 3].enumerated() {
            let prefix = "acoustic_encoder.block.\(index)"
            x = residualUnit(x, prefix: "\(prefix).res_unit1", dilation: 1)
            x = residualUnit(x, prefix: "\(prefix).res_unit2", dilation: 3)
            x = residualUnit(x, prefix: "\(prefix).res_unit3", dilation: 9)
            x = snake(x, alphaKey: "\(prefix).snake1.alpha")
            x = conv(x, prefix: "\(prefix).conv1", padding: (stride + 1) / 2, stride: stride)
        }
        x = snake(x, alphaKey: "acoustic_encoder.snake1.alpha")
        x = conv(x, prefix: "acoustic_encoder.conv2", padding: 1)
        return x
    }

    // MARK: - Residual vector quantization (encode)

    private func rvqEncode(_ z: MLXArray) -> [[Int32]] {
        let frames = z.dim(1)
        var residual = z
        var perCodebook: [[Int32]] = []
        for quantizer in 0..<Self.numCodebooks {
            let prefix = "quantizer.quantizers.\(quantizer)"
            let projected = linear(residual, prefix: "\(prefix).project_in")
            let codebook = weights["\(prefix).codebook.weight"]!
            let distances = sum(projected * projected, axis: -1, keepDims: true)
                + sum(codebook * codebook, axis: -1)
                - 2 * matmul(projected, codebook.transposed())
            let indices = argMin(distances, axis: -1).asType(.int32)
            eval(indices)
            perCodebook.append(indices.asArray(Int32.self))

            let embedded = take(codebook, indices.reshaped(-1), axis: 0)
                .reshaped(1, frames, codebook.dim(1))
            let reconstruction = linear(embedded, prefix: "\(prefix).project_out")
            residual = residual - reconstruction
        }
        return (0..<frames).map { frame in
            (0..<Self.numCodebooks).map { perCodebook[$0][frame] }
        }
    }

    // MARK: - Small ops

    private func layerNorm(_ x: MLXArray, prefix: String, eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: -1, keepDims: true)
        let centered = x - mean
        let varr = (centered * centered).mean(axis: -1, keepDims: true)
        let normed = centered * rsqrt(varr + eps)
        return normed * weights["\(prefix).weight"]! + weights["\(prefix).bias"]!
    }

    /// GroupNorm with one group per channel (PyTorch-compatible): normalize
    /// each channel over time.
    private func perChannelGroupNorm(_ x: MLXArray, prefix: String, eps: Float = 1e-5) -> MLXArray {
        let mean = x.mean(axis: 1, keepDims: true)
        let centered = x - mean
        let varr = (centered * centered).mean(axis: 1, keepDims: true)
        let normed = centered * rsqrt(varr + eps)
        return normed * weights["\(prefix).weight"]! + weights["\(prefix).bias"]!
    }

    private func geluErf(_ x: MLXArray) -> MLXArray {
        x * (1 + erf(x / Float(2.0).squareRoot())) / 2
    }

    private func elu(_ x: MLXArray) -> MLXArray {
        MLX.which(x .> 0, x, MLX.exp(x) - 1)
    }

    private static let requiredEncodeKeys = [
        "fc.weight", "fc.bias",
        "acoustic_encoder.conv1.weight", "acoustic_encoder.conv2.weight",
        "encoder_semantic.conv.weight",
        "semantic_model.feature_projection.projection.weight",
        "semantic_model.encoder.pos_conv_embed.conv.weight_g",
        "semantic_model.encoder.pos_conv_embed.conv.weight_v",
        "semantic_model.encoder.layers.11.final_layer_norm.weight",
        "quantizer.quantizers.0.project_in.weight",
    ]

    // MARK: - Resampling

    /// Hann-windowed sinc resampling matching
    /// `torchaudio.functional.resample(method="sinc_interp_hann")`, as used by
    /// the reference encode path (lowpass width 6, rolloff 0.99).
    static func sincResample(_ input: [Float], from origFreq: Int, to newFreq: Int) -> [Float] {
        if origFreq == newFreq { return input }
        func gcd(_ a: Int, _ b: Int) -> Int { b == 0 ? a : gcd(b, a % b) }
        let divisor = gcd(origFreq, newFreq)
        let origR = origFreq / divisor
        let newR = newFreq / divisor

        let lowpassWidth = 6.0
        let baseFreq = Double(min(origR, newR)) * 0.99
        let width = Int((lowpassWidth * Double(origR) / baseFreq).rounded(.up))
        let kernelLength = 2 * width + origR

        var kernel = [[Float]](repeating: [Float](repeating: 0, count: kernelLength), count: newR)
        for phase in 0..<newR {
            for tap in 0..<kernelLength {
                let idx = Double(tap - width) / Double(origR)
                var t = (-Double(phase) / Double(newR) + idx) * baseFreq
                t = min(max(t, -lowpassWidth), lowpassWidth)
                let window = pow(cos(t * .pi / lowpassWidth / 2), 2)
                let tPi = t * .pi
                let sinc = tPi == 0 ? 1.0 : sin(tPi) / tPi
                kernel[phase][tap] = Float(sinc * window * (baseFreq / Double(origR)))
            }
        }

        let padded = [Float](repeating: 0, count: width) + input
            + [Float](repeating: 0, count: width + origR)
        let outLength = Int((Double(input.count) * Double(newR) / Double(origR)).rounded(.up))
        var output = [Float](repeating: 0, count: outLength)
        for phase in 0..<newR {
            var sample = 0
            while true {
                let position = phase + sample * newR
                let start = sample * origR
                if position >= outLength || start + kernelLength > padded.count { break }
                var accumulator: Float = 0
                for tap in 0..<kernelLength {
                    accumulator += padded[start + tap] * kernel[phase][tap]
                }
                output[position] = accumulator
                sample += 1
            }
        }
        return output
    }
}
