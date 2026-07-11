import Foundation
import MLX
import MLXFFT
import MLXFast
import MLXNN

@inline(__always)
func f5Linear(_ x: MLXArray, weights: [String: MLXArray], prefix: String, bias: Bool = true) -> MLXArray {
    let originalType = x.dtype
    let xCompute = x.asType(.float32)
    let w = weights["\(prefix).weight"]!.asType(.float32)
    var y = matmul(xCompute, w.transposed(1, 0))
    if bias, let b = weights["\(prefix).bias"] {
        y = y + b.asType(.float32)
    }
    return y.asType(originalType)
}

@inline(__always)
func f5LayerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    let originalType = x.dtype
    let xf = x.asType(.float32)
    let mean = xf.mean(axis: -1, keepDims: true)
    let centered = xf - mean
    let variance = (centered * centered).mean(axis: -1, keepDims: true)
    return (centered / MLX.sqrt(variance + MLXArray(eps))).asType(originalType)
}

@inline(__always)
func f5LayerNorm(_ x: MLXArray, weights: [String: MLXArray], prefix: String, eps: Float = 1e-6) -> MLXArray {
    var y = f5LayerNormNoAffine(x, eps: eps)
    y = y * weights["\(prefix).weight"]!.asType(y.dtype)
    y = y + weights["\(prefix).bias"]!.asType(y.dtype)
    return y
}

@inline(__always)
func f5Mish(_ x: MLXArray) -> MLXArray {
    x * tanh(log1p(exp(x)))
}

func f5Conv1dNCL(
    _ x: MLXArray,
    weight: MLXArray,
    bias: MLXArray? = nil,
    padding: Int = 0,
    groups: Int = 1
) -> MLXArray {
    let w = weight.asType(x.dtype).transposed(0, 2, 1)
    var y = MLX.conv1d(
        x.transposed(0, 2, 1),
        w,
        stride: 1,
        padding: padding,
        groups: groups)
        .transposed(0, 2, 1)
    if let bias {
        y = y + bias.asType(y.dtype).reshaped([1, bias.dim(0), 1])
    }
    return y
}

func f5Conv1dNLC(
    _ x: MLXArray,
    weights: [String: MLXArray],
    prefix: String,
    padding: Int,
    groups: Int = 1
) -> MLXArray {
    f5Conv1dNCL(
        x.transposed(0, 2, 1),
        weight: weights["\(prefix).weight"]!,
        bias: weights["\(prefix).bias"],
        padding: padding,
        groups: groups)
        .transposed(0, 2, 1)
}

func f5SinusoidalPositionEmbedding(_ t: MLXArray, dim: Int, scale: Float = 1000) -> MLXArray {
    var time = t
    if time.ndim == 0 {
        time = time.expandedDimensions(axis: 0)
    }
    let half = dim / 2
    let factor = Float(log(10000.0)) / Float(half - 1)
    let freqs = exp(MLXArray(0..<Int32(half)).asType(.float32) * (-factor))
    let args = MLXArray(scale) * time.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return concatenated([sin(args), cos(args)], axis: -1)
}

func f5PrecomputeFreqsCis(dim: Int, end: Int, theta: Float = 10000) -> MLXArray {
    let half = dim / 2
    let idx = MLXArray(0..<Int32(half)).asType(.float32)
    let freqs = 1.0 / MLX.pow(MLXArray(theta), idx * 2.0 / MLXArray(Float(dim)))
    let t = MLXArray(0..<Int32(end)).asType(.float32)
    let outer = t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    return concatenated([cos(outer), sin(outer)], axis: -1)
}

func f5ApplyXTransformersRoPE(_ x: MLXArray, dimensions: Int, theta: Float = 10000) -> MLXArray {
    let originalType = x.dtype
    let rotaryType: DType = originalType == .float16 ? .float16 : .float32
    let batch = x.dim(0)
    let heads = x.dim(1)
    let seqLen = x.dim(2)
    let half = dimensions / 2
    let idx = MLXArray(0..<Int32(half)).asType(.float32)
    let invFreq = (1.0 / MLX.pow(MLXArray(theta), idx * 2.0 / MLXArray(Float(dimensions))))
        .asType(rotaryType)
    let positions = MLXArray(0..<Int32(seqLen)).asType(rotaryType)
    let angles = (positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0))
        .asType(rotaryType)
    let cosValues = cos(angles).asType(rotaryType).reshaped([1, 1, seqLen, half])
    let sinValues = sin(angles).asType(rotaryType).reshaped([1, 1, seqLen, half])

    let pairs = x.asType(rotaryType).reshaped([batch, heads, seqLen, half, 2])
    let x1 = pairs[0..., 0..., 0..., 0..., 0]
    let x2 = pairs[0..., 0..., 0..., 0..., 1]
    let out1 = x1 * cosValues - x2 * sinValues
    let out2 = x2 * cosValues + x1 * sinValues
    return concatenated(
        [out1.expandedDimensions(axis: -1), out2.expandedDimensions(axis: -1)],
        axis: -1
    ).reshaped(x.shape).asType(originalType)
}

func f5GELUTanh(_ x: MLXArray) -> MLXArray {
    geluApproximate(x)
}

enum F5MelFrontend {
    static func melSpec(samples: [Float], config c: F5TTSMelSpecConfig) -> MLXArray {
        let nBins = c.nFFT / 2 + 1
        let pad = c.nFFT / 2
        let padded = reflectPad(samples, pad: pad)
        let frames = max(0, (padded.count - c.nFFT) / c.hopLength + 1)
        let window = hannWindow(length: c.winLength, nFFT: c.nFFT)
        var framed = [Float](repeating: 0, count: frames * c.nFFT)
        for frame in 0..<frames {
            let start = frame * c.hopLength
            for i in 0..<c.nFFT {
                framed[frame * c.nFFT + i] = padded[start + i] * window[i]
            }
        }

        let spec = MLXFFT.rfft(MLXArray(framed, [frames, c.nFFT]), axis: -1)
        let mag = abs(spec)
        let fb = MLXArray(
            melFilterbankHTK(
                sampleRate: c.targetSampleRate,
                nFFT: c.nFFT,
                nMels: c.nMelChannels),
            [nBins, c.nMelChannels])
        let mel = matmul(mag, fb)
        return log(MLX.maximum(mel, MLXArray(Float(1e-5))))
    }

    private static func hannWindow(length: Int, nFFT: Int) -> [Float] {
        var out = [Float](repeating: 0, count: nFFT)
        for i in 0..<length {
            out[i] = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(i) / Float(length))
        }
        return out
    }

    private static func reflectPad(_ row: [Float], pad: Int) -> [Float] {
        guard !row.isEmpty else { return row }
        var out = [Float](repeating: 0, count: pad + row.count + pad)
        for i in 0..<row.count {
            out[pad + i] = row[i]
        }
        for i in 0..<pad {
            let source = min(max(pad - i, 0), row.count - 1)
            out[i] = row[source]
        }
        let last = row.count - 1
        for i in 0..<pad {
            let source = min(max(last - 1 - i, 0), row.count - 1)
            out[pad + row.count + i] = row[source]
        }
        return out
    }

    private static func hzToMelHTK(_ hz: Float) -> Float {
        2595.0 * log10(1.0 + hz / 700.0)
    }

    private static func melToHzHTK(_ mel: Float) -> Float {
        700.0 * (pow(10.0, mel / 2595.0) - 1.0)
    }

    private static func melFilterbankHTK(sampleRate: Int, nFFT: Int, nMels: Int) -> [Float] {
        let nBins = nFFT / 2 + 1
        let fMax = Float(sampleRate) / 2.0
        let melMin = hzToMelHTK(0)
        let melMax = hzToMelHTK(fMax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
        }
        let hzPoints = melPoints.map(melToHzHTK)
        var fb = [Float](repeating: 0, count: nBins * nMels)
        for bin in 0..<nBins {
            let freq = Float(bin) * Float(sampleRate) / Float(nFFT)
            for mel in 0..<nMels {
                let left = hzPoints[mel]
                let center = hzPoints[mel + 1]
                let right = hzPoints[mel + 2]
                let value: Float
                if freq < left || freq > right {
                    value = 0
                } else if freq <= center {
                    value = (freq - left) / max(center - left, 1e-12)
                } else {
                    value = (right - freq) / max(right - center, 1e-12)
                }
                fb[bin * nMels + mel] = value
            }
        }
        return fb
    }
}

struct F5ISTFT {
    let nFFT: Int
    let hopLength: Int
    let winLength: Int
    let window: MLXArray

    init(nFFT: Int = 1024, hopLength: Int = 256, winLength: Int = 1024) {
        self.nFFT = nFFT
        self.hopLength = hopLength
        self.winLength = winLength
        let values = (0..<winLength).map {
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(winLength)))
        }
        self.window = MLXArray(values)
    }

    func callAsFunction(_ spec: MLXArray) -> MLXArray {
        precondition(nFFT % hopLength == 0)
        let k = nFFT / hopLength
        let pad = (winLength - hopLength) / 2
        let zt = spec.transposed(0, 2, 1)
        var frames = MLXFFT.irfft(zt, n: nFFT, axis: -1)
        let batch = frames.dim(0)
        let frameCount = frames.dim(1)
        frames = frames * window.reshaped([1, 1, nFFT])

        let sub = frames.reshaped([batch, frameCount, k, hopLength])
        var accum: MLXArray?
        for j in 0..<k {
            let slice = sub[0..., 0..., j, 0...]
            let padded = MLX.padded(
                slice,
                widths: [
                    IntOrPair((0, 0)),
                    IntOrPair((j, (k - 1) - j)),
                    IntOrPair((0, 0)),
                ])
            accum = accum.map { $0 + padded } ?? padded
        }
        let combined = accum!.reshaped([batch, (frameCount + k - 1) * hopLength])

        let w2 = (window * window).reshaped([k, hopLength])
        var w2Accum: MLXArray?
        for j in 0..<k {
            let row = MLX.broadcast(w2[j, 0...].reshaped([1, hopLength]), to: [frameCount, hopLength])
            let padded = MLX.padded(
                row,
                widths: [
                    IntOrPair((j, (k - 1) - j)),
                    IntOrPair((0, 0)),
                ])
            w2Accum = w2Accum.map { $0 + padded } ?? padded
        }
        let envelope = w2Accum!.reshaped([(frameCount + k - 1) * hopLength])
        let normalized = combined / MLX.maximum(envelope, MLXArray(Float(1e-8)))
        return normalized[0..., pad..<(normalized.dim(1) - pad)]
    }
}
