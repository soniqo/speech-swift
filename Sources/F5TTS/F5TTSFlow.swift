import Foundation
import MLX
import MLXFast
import MLXNN

public struct F5TTSSynthesisOptions: Equatable, Sendable {
    public let steps: Int
    public let cfgStrength: Float
    public let swaySamplingCoef: Float?
    public let speed: Float
    public let seed: UInt64
    public let targetRMS: Float

    public init(
        steps: Int = 16,
        cfgStrength: Float = 2.0,
        swaySamplingCoef: Float? = -1.0,
        speed: Float = 1.0,
        seed: UInt64 = 0,
        targetRMS: Float = 0.1
    ) throws {
        guard steps > 0 else {
            throw F5TTSError.unsupportedText("steps must be positive")
        }
        guard cfgStrength.isFinite, cfgStrength >= 0 else {
            throw F5TTSError.unsupportedText("cfgStrength must be finite and non-negative")
        }
        guard speed.isFinite, speed > 0 else {
            throw F5TTSError.unsupportedText("speed must be finite and positive")
        }
        self.steps = steps
        self.cfgStrength = cfgStrength
        self.swaySamplingCoef = swaySamplingCoef
        self.speed = speed
        self.seed = seed
        self.targetRMS = targetRMS
    }

    public static let `default` = try! F5TTSSynthesisOptions()
}

struct F5TTSPreparedReference {
    let samples: [Float]
    let rms: Float
    let mel: MLXArray
    let rawFrameCount: Int
}

final class F5TTSFlow {
    private let weights: [String: MLXArray]
    private let config: F5TTSConfig
    private let dim: Int
    private let textDim: Int
    private let heads: Int
    private let headDim: Int
    private let melDim: Int
    private var textCond: MLXArray?
    private var textUncond: MLXArray?

    init(weights: [String: MLXArray], config: F5TTSConfig) {
        self.weights = weights
        self.config = config
        self.dim = config.architecture.dim
        self.textDim = config.architecture.textDim
        self.heads = config.architecture.heads
        self.headDim = config.architecture.dim / config.architecture.heads
        self.melDim = config.melSpec.nMelChannels
    }

    static func validate(_ weights: [String: MLXArray], config: F5TTSConfig) throws {
        try require(weights, component: "DiT", key: "transformer.input_embed.proj.weight", shape: [config.architecture.dim, config.melSpec.nMelChannels * 2 + config.architecture.textDim])
        try require(weights, component: "DiT", key: "transformer.text_embed.text_embed.weight", shape: [2546, config.architecture.textDim])
        try require(weights, component: "DiT", key: "transformer.time_embed.time_mlp.0.weight", shape: [config.architecture.dim, 256])
        try require(weights, component: "DiT", key: "transformer.transformer_blocks.0.attn.to_q.weight", shape: [config.architecture.dim, config.architecture.dim])
        try require(weights, component: "DiT", key: "transformer.proj_out.weight", shape: [config.melSpec.nMelChannels, config.architecture.dim])
    }

    func prepareReference(samples: [Float], options: F5TTSSynthesisOptions) -> F5TTSPreparedReference {
        var audio = samples
        let rms = sqrt(max(audio.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(audio.count, 1)), 0))
        if rms > 0, rms < options.targetRMS {
            let gain = options.targetRMS / rms
            audio = audio.map { $0 * gain }
        }
        let mel = F5MelFrontend.melSpec(samples: audio, config: config.melSpec)
        return F5TTSPreparedReference(
            samples: audio,
            rms: rms,
            mel: mel.expandedDimensions(axis: 0),
            rawFrameCount: audio.count / config.melSpec.hopLength)
    }

    func sampleMel(
        reference: F5TTSPreparedReference,
        referenceText: String,
        targetText: String,
        tokenizer: F5TTSTokenizer,
        options: F5TTSSynthesisOptions,
        initialNoise: MLXArray? = nil,
        stepDumpDirectory: URL? = nil
    ) throws -> MLXArray {
        clearCache()
        let refText = Self.normalizedReferenceText(referenceText)
        let fullText = refText + targetText
        let tokenIds = try tokenizer.encode(fullText)
        let refTextBytes = max(refText.lengthOfBytes(using: .utf8), 1)
        let targetTextBytes = targetText.lengthOfBytes(using: .utf8)
        let localSpeed: Float = targetTextBytes < 10 ? 0.3 : options.speed
        var duration = reference.rawFrameCount + Int(Float(reference.rawFrameCount) / Float(refTextBytes) * Float(targetTextBytes) / localSpeed)
        let condFrames = reference.mel.dim(1)
        duration = max(duration, condFrames + 1, tokenIds.count + 1)

        let cond = reference.mel.asType(.float32)
        let maxDuration = duration
        var condPadded = MLX.padded(
            cond,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, maxDuration - condFrames)),
                IntOrPair((0, 0)),
            ])
        condPadded = condPadded.asType(.float32)

        var x: MLXArray
        if let initialNoise {
            x = initialNoise.asType(condPadded.dtype)
        } else {
            let key = MLXRandom.key(options.seed)
            x = MLXRandom.normal([1, maxDuration, melDim], dtype: condPadded.dtype, key: key)
        }
        let ts = timesteps(steps: options.steps, sway: options.swaySamplingCoef)
        if let stepDumpDirectory {
            try FileManager.default.createDirectory(at: stepDumpDirectory, withIntermediateDirectories: true)
            try writeFloat32Array(x, to: stepDumpDirectory.appendingPathComponent("step-00.f32"))
            try writeFloat32Array(condPadded, to: stepDumpDirectory.appendingPathComponent("cond-padded.f32"))
            try writeInt32Array(tokenIds, to: stepDumpDirectory.appendingPathComponent("tokens.i32"))
            let firstT = ts.first ?? 0
            let firstDt = (ts.count > 1) ? ts[1] - ts[0] : 0
            let sway = options.swaySamplingCoef.map { String($0) } ?? "nil"
            let metadata = [
                "seqLen=\(maxDuration)",
                "condFrames=\(condFrames)",
                "rawFrameCount=\(reference.rawFrameCount)",
                "tokenCount=\(tokenIds.count)",
                "steps=\(options.steps)",
                "cfgStrength=\(options.cfgStrength)",
                "swaySamplingCoef=\(sway)",
                "speed=\(options.speed)",
                "firstT=\(firstT)",
                "firstDt=\(firstDt)",
            ].joined(separator: "\n")
            try metadata.write(
                to: stepDumpDirectory.appendingPathComponent("metadata.txt"),
                atomically: true,
                encoding: String.Encoding.utf8)
        }
        for i in 0..<(ts.count - 1) {
            let t = ts[i]
            let dt = ts[i + 1] - ts[i]
            let velocity = transformerCFG(
                x: x,
                cond: condPadded,
                tokenIds: tokenIds,
                time: t,
                seqLen: maxDuration,
                cfgStrength: options.cfgStrength)
            if let stepDumpDirectory {
                try writeFloat32Array(
                    velocity,
                    to: stepDumpDirectory.appendingPathComponent(String(format: "velocity-step-%02d.f32", i)))
            }
            x = x + MLXArray(dt).asType(x.dtype) * velocity.asType(x.dtype)
            eval(x)
            if let stepDumpDirectory {
                try writeFloat32Array(x, to: stepDumpDirectory.appendingPathComponent(String(format: "step-%02d.f32", i + 1)))
            }
        }

        let out = x
        out[0..., 0..<condFrames, 0...] = condPadded[0..., 0..<condFrames, 0...]
        clearCache()
        return out[0..., reference.rawFrameCount..<maxDuration, 0...].transposed(0, 2, 1)
    }

    func predictVelocityForTesting(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float,
        cfgStrength: Float
    ) -> MLXArray {
        clearCache()
        return transformerCFG(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            time: time,
            seqLen: x.dim(1),
            cfgStrength: cfgStrength)
    }

    func traceConditionedBranchForTesting(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float
    ) -> [(String, MLXArray)] {
        traceBranchForTesting(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            time: time,
            dropAudioCond: false,
            dropText: false)
    }

    func traceUnconditionedBranchForTesting(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float
    ) -> [(String, MLXArray)] {
        traceBranchForTesting(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            time: time,
            dropAudioCond: true,
            dropText: true)
    }

    private func traceBranchForTesting(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float,
        dropAudioCond: Bool,
        dropText: Bool
    ) -> [(String, MLXArray)] {
        clearCache()
        let timeInput = MLXArray([time], [1])
        let t = timeEmbedding(timeInput)
        var traces: [(String, MLXArray)] = [("time", t)]
        var h = inputEmbedding(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            seqLen: x.dim(1),
            dropAudioCond: dropAudioCond,
            dropText: dropText)
        traces.append(("input", h))
        for layer in 0..<config.architecture.depth {
            if layer == 0 {
                let prefix = "transformer.transformer_blocks.0"
                let modulation = f5Linear(silu(t), weights: weights, prefix: "\(prefix).attn_norm.linear")
                let parts = split(modulation, parts: 6, axis: -1)
                let shiftMSA = parts[0]
                let scaleMSA = parts[1]
                let gateMSA = parts[2]
                let shiftMLP = parts[3]
                let scaleMLP = parts[4]
                let gateMLP = parts[5]
                var norm = f5LayerNormNoAffine(h, eps: 1e-6)
                norm = norm * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
                traces.append(("block0AttnNorm", norm))
                let batch = norm.dim(0)
                let seqLen = norm.dim(1)
                var q = f5Linear(norm, weights: weights, prefix: "\(prefix).attn.to_q")
                var k = f5Linear(norm, weights: weights, prefix: "\(prefix).attn.to_k")
                let v = f5Linear(norm, weights: weights, prefix: "\(prefix).attn.to_v")
                q = q.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
                k = k.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
                let vHeads = v.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
                traces.append(("block0Q", q))
                traces.append(("block0K", k))
                traces.append(("block0V", vHeads))
                traces.append(("block0QRope", f5ApplyXTransformersRoPE(q, dimensions: headDim)))
                traces.append(("block0KRope", f5ApplyXTransformersRoPE(k, dimensions: headDim)))
                let attnOut = attention(norm, prefix: "\(prefix).attn")
                traces.append(("block0AttnOut", attnOut))
                h = h + gateMSA.expandedDimensions(axis: 1) * attnOut
                traces.append(("block0AfterAttn", h))
                norm = f5LayerNormNoAffine(h, eps: 1e-6)
                norm = norm * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
                traces.append(("block0FFNorm", norm))
                var ff = f5Linear(norm, weights: weights, prefix: "\(prefix).ff.ff.0.0")
                ff = f5GELUTanh(ff)
                ff = f5Linear(ff, weights: weights, prefix: "\(prefix).ff.ff.2")
                traces.append(("block0FFOut", ff))
                h = h + gateMLP.expandedDimensions(axis: 1) * ff
                traces.append(("block0", h))
            } else {
                h = ditBlock(h, t: t, layer: layer)
            }
        }
        traces.append(("blockLast", h))
        h = finalAdaLayerNorm(h, t: t)
        traces.append(("finalNorm", h))
        traces.append(("projOut", f5Linear(h, weights: weights, prefix: "transformer.proj_out")))
        return traces
    }

    private func transformerCFG(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float,
        seqLen: Int,
        cfgStrength: Float
    ) -> MLXArray {
        if cfgStrength < 1e-5 {
            return transformer(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                time: time,
                seqLen: seqLen,
                dropAudioCond: false,
                dropText: false,
                cfgInfer: false)
        }

        let predBoth = transformer(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            time: time,
            seqLen: seqLen,
            dropAudioCond: false,
            dropText: false,
            cfgInfer: true)
        let pred = predBoth[0..<1, 0..., 0...]
        let nullPred = predBoth[1..<2, 0..., 0...]
        return pred + (pred - nullPred) * MLXArray(cfgStrength).asType(pred.dtype)
    }

    private func transformer(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        time: Float,
        seqLen: Int,
        dropAudioCond: Bool,
        dropText: Bool,
        cfgInfer: Bool
    ) -> MLXArray {
        let timeInput = MLXArray([time], [1])
        var t = timeEmbedding(timeInput)

        var h: MLXArray
        if cfgInfer {
            let xCond = inputEmbedding(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                seqLen: seqLen,
                dropAudioCond: false,
                dropText: false)
            let xUncond = inputEmbedding(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                seqLen: seqLen,
                dropAudioCond: true,
                dropText: true)
            h = concatenated([xCond, xUncond], axis: 0)
            t = concatenated([t, t], axis: 0)
        } else {
            h = inputEmbedding(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                seqLen: seqLen,
                dropAudioCond: dropAudioCond,
                dropText: dropText)
        }

        for layer in 0..<config.architecture.depth {
            h = ditBlock(h, t: t, layer: layer)
        }
        h = finalAdaLayerNorm(h, t: t)
        return f5Linear(h, weights: weights, prefix: "transformer.proj_out")
    }

    private func inputEmbedding(
        x: MLXArray,
        cond: MLXArray,
        tokenIds: [Int32],
        seqLen: Int,
        dropAudioCond: Bool,
        dropText: Bool
    ) -> MLXArray {
        let textEmbed: MLXArray
        if dropText {
            if let textUncond {
                textEmbed = textUncond
            } else {
                let computed = textEmbedding(tokenIds: tokenIds, seqLen: seqLen, dropText: true)
                textUncond = computed
                textEmbed = computed
            }
        } else {
            if let textCond {
                textEmbed = textCond
            } else {
                let computed = textEmbedding(tokenIds: tokenIds, seqLen: seqLen, dropText: false)
                textCond = computed
                textEmbed = computed
            }
        }

        let audioCond = dropAudioCond ? MLXArray.zeros(cond.shape, dtype: cond.dtype) : cond
        var h = concatenated([x, audioCond, textEmbed.asType(x.dtype)], axis: -1)
        h = f5Linear(h, weights: weights, prefix: "transformer.input_embed.proj")
        let pos = convPositionEmbedding(h)
        return pos + h
    }

    private func textEmbedding(tokenIds: [Int32], seqLen: Int, dropText: Bool) -> MLXArray {
        let shifted = tokenIds.prefix(seqLen).map { $0 + 1 }
        let paddedIds = shifted + [Int32](repeating: 0, count: max(0, seqLen - shifted.count))
        let ids = dropText ? [Int32](repeating: 0, count: seqLen) : paddedIds
        let idsArray = MLXArray(ids, [1, seqLen])
        var h = take(weights["transformer.text_embed.text_embed.weight"]!, idsArray, axis: 0)
        h = h.asType(.float32)

        let freqs = f5PrecomputeFreqsCis(dim: textDim, end: seqLen).asType(h.dtype).expandedDimensions(axis: 0)
        h = h + freqs
        let maskValues = paddedIds.map { $0 == 0 ? Float(0) : Float(1) }
        let mask = MLXArray(maskValues, [1, seqLen, 1]).asType(h.dtype)
        h = h * mask
        for layer in 0..<config.architecture.convLayers {
            h = textConvNeXtBlock(h, layer: layer) * mask
        }
        return h
    }

    private func textConvNeXtBlock(_ x: MLXArray, layer: Int) -> MLXArray {
        let prefix = "transformer.text_embed.text_blocks.\(layer)"
        let residual = x
        var h = f5Conv1dNLC(x, weights: weights, prefix: "\(prefix).dwconv", padding: 3, groups: textDim)
        h = f5LayerNorm(h, weights: weights, prefix: "\(prefix).norm", eps: 1e-6)
        h = f5Linear(h, weights: weights, prefix: "\(prefix).pwconv1")
        h = gelu(h)
        let gx = MLX.sqrt((h * h).sum(axis: 1, keepDims: true))
        let nx = gx / (gx.mean(axis: -1, keepDims: true) + MLXArray(Float(1e-6)).asType(gx.dtype))
        h = weights["\(prefix).grn.gamma"]!.asType(h.dtype) * (h * nx) +
            weights["\(prefix).grn.beta"]!.asType(h.dtype) + h
        h = f5Linear(h, weights: weights, prefix: "\(prefix).pwconv2")
        return residual + h
    }

    private func convPositionEmbedding(_ x: MLXArray) -> MLXArray {
        var h = x.transposed(0, 2, 1)
        h = f5Conv1dNCL(
            h,
            weight: weights["transformer.input_embed.conv_pos_embed.conv1d.0.weight"]!,
            bias: weights["transformer.input_embed.conv_pos_embed.conv1d.0.bias"],
            padding: 15,
            groups: 16)
        h = f5Mish(h)
        h = f5Conv1dNCL(
            h,
            weight: weights["transformer.input_embed.conv_pos_embed.conv1d.2.weight"]!,
            bias: weights["transformer.input_embed.conv_pos_embed.conv1d.2.bias"],
            padding: 15,
            groups: 16)
        h = f5Mish(h)
        return h.transposed(0, 2, 1)
    }

    private func timeEmbedding(_ time: MLXArray) -> MLXArray {
        var h = f5SinusoidalPositionEmbedding(time, dim: 256)
        h = f5Linear(h, weights: weights, prefix: "transformer.time_embed.time_mlp.0")
        h = silu(h)
        return f5Linear(h, weights: weights, prefix: "transformer.time_embed.time_mlp.2")
    }

    private func ditBlock(_ x: MLXArray, t: MLXArray, layer: Int) -> MLXArray {
        let prefix = "transformer.transformer_blocks.\(layer)"
        let modulation = f5Linear(silu(t), weights: weights, prefix: "\(prefix).attn_norm.linear")
        let parts = split(modulation, parts: 6, axis: -1)
        let shiftMSA = parts[0]
        let scaleMSA = parts[1]
        let gateMSA = parts[2]
        let shiftMLP = parts[3]
        let scaleMLP = parts[4]
        let gateMLP = parts[5]

        var norm = f5LayerNormNoAffine(x, eps: 1e-6)
        norm = norm * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        var h = x + gateMSA.expandedDimensions(axis: 1) * attention(norm, prefix: "\(prefix).attn")

        norm = f5LayerNormNoAffine(h, eps: 1e-6)
        norm = norm * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
        var ff = f5Linear(norm, weights: weights, prefix: "\(prefix).ff.ff.0.0")
        ff = f5GELUTanh(ff)
        ff = f5Linear(ff, weights: weights, prefix: "\(prefix).ff.ff.2")
        h = h + gateMLP.expandedDimensions(axis: 1) * ff
        return h
    }

    private func attention(_ x: MLXArray, prefix: String) -> MLXArray {
        let batch = x.dim(0)
        let seqLen = x.dim(1)
        var q = f5Linear(x, weights: weights, prefix: "\(prefix).to_q")
        var k = f5Linear(x, weights: weights, prefix: "\(prefix).to_k")
        let v = f5Linear(x, weights: weights, prefix: "\(prefix).to_v")
        q = q.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
        k = k.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
        let vh = v.reshaped([batch, seqLen, heads, headDim]).transposed(0, 2, 1, 3)
        q = f5ApplyXTransformersRoPE(q, dimensions: headDim)
        k = f5ApplyXTransformersRoPE(k, dimensions: headDim)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: vh,
            scale: 1.0 / Foundation.sqrt(Float(headDim)),
            mask: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped([batch, seqLen, dim])
        return f5Linear(merged, weights: weights, prefix: "\(prefix).to_out.0")
    }

    private func finalAdaLayerNorm(_ x: MLXArray, t: MLXArray) -> MLXArray {
        let modulation = f5Linear(silu(t), weights: weights, prefix: "transformer.norm_out.linear")
        let parts = split(modulation, parts: 2, axis: -1)
        return f5LayerNormNoAffine(x, eps: 1e-6) *
            (1 + parts[0].expandedDimensions(axis: 1)) +
            parts[1].expandedDimensions(axis: 1)
    }

    private func clearCache() {
        textCond = nil
        textUncond = nil
    }

    private func writeFloat32Array(_ array: MLXArray, to url: URL) throws {
        let values = array.asType(.float32).asArray(Float.self)
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url)
    }

    private func writeInt32Array(_ values: [Int32], to url: URL) throws {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        try data.write(to: url)
    }

    private func timesteps(steps: Int, sway: Float?) -> [Float] {
        let predefined: [Int: [Float]] = [
            5: [0, 2, 4, 8, 16, 32],
            6: [0, 2, 4, 6, 8, 16, 32],
            7: [0, 2, 4, 6, 8, 16, 24, 32],
            10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
            12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
            16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        ]
        var values: [Float]
        if let selected = predefined[steps] {
            values = selected.map { $0 / 32.0 }
        } else {
            values = (0...steps).map { Float($0) / Float(steps) }
        }
        if let sway {
            values = values.map { t in
                t + sway * (cos(Float.pi / 2.0 * t) - 1.0 + t)
            }
        }
        return values
    }

    static func normalizedReferenceText(_ text: String) -> String {
        var result = text
        if !result.hasSuffix(". "), !result.hasSuffix("。") {
            if result.hasSuffix(".") {
                result += " "
            } else {
                result += ". "
            }
        }
        if result.last?.utf8.count == 1 {
            return result + " "
        }
        return result
    }
}
