import Foundation
import MLX
import MLXFFT
import MLXNN

// CAM++ (TDNN x-vector) speaker encoder for the S3Gen conditioning path.
//
// Ported to MLX-Swift to match the reference implementation. The module/parameter
// keys below map 1:1 to the converted bundle's `s3gen.speaker_encoder.*` keys
// (after stripping that prefix), so `module.update(parameters:verify:.all)` loads
// cleanly. Conv weights in the converted bundle are already in MLX layout
// (Conv2d `[O,H,W,I]`, Conv1d `[O,K,I]`); no transpose happens at load time.
//
// Layout convention inside the network mirrors the reference: tensors flow in
// PyTorch-style `(B, C, T)` between layers, and each layer swaps to channels-last
// `(B, T, C)` only for the MLX `Conv1d`/`BatchNorm` calls, then swaps back. The
// 2-D FCM head works in NHWC `(B, H, W, C)` for `Conv2d`/`BatchNorm`.

// MARK: - Kaldi-style fbank front-end

/// 80-bin log-mel filterbank matching the reference `kaldi_fbank`:
/// 16 kHz, 25 ms frame / 10 ms shift, Povey window, per-frame DC removal +
/// pre-emphasis 0.97, n_fft rounded up to 512, HTK mel scale, f_min = 20 Hz,
/// no filterbank normalization, log floor = float32 epsilon. Returns `(frames, 80)`.
private enum KaldiFbank {
    static let sampleRate = 16000
    static let numMelBins = 80
    static let winLength = 400   // 25 ms @ 16 kHz
    static let hopLength = 160   // 10 ms @ 16 kHz
    static let nFft = 512        // next power of 2 >= winLength
    static let preemph: Float = 0.97
    static let logFloor: Float = 1.1920929e-07  // std::numeric_limits<float>::epsilon()

    /// Povey window: `(0.5 - 0.5*cos(2*pi*n/(N-1)))^0.85`.
    private static func poveyWindow() -> [Float] {
        var w = [Float](repeating: 0, count: winLength)
        let denom = Float(winLength - 1)
        for n in 0 ..< winLength {
            let hann = 0.5 - 0.5 * cos(2.0 * Float.pi * Float(n) / denom)
            w[n] = pow(hann, 0.85)
        }
        return w
    }

    /// HTK mel filterbank, row-major `(nMels, nBins)`, no normalization, f_min = 20 Hz.
    /// FFT bin frequencies use `linspace(0, sampleRate/2, nBins)` to match the reference.
    private static func melFilterbank() -> [Float] {
        let nBins = nFft / 2 + 1
        let fMin: Float = 20.0
        let fMax: Float = Float(sampleRate) / 2.0

        func hzToMel(_ hz: Float) -> Float { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Float) -> Float { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        var allFreqs = [Float](repeating: 0, count: nBins)
        let top = Float(sampleRate / 2)
        for i in 0 ..< nBins {
            allFreqs[i] = top * Float(i) / Float(nBins - 1)
        }

        let mMin = hzToMel(fMin)
        let mMax = hzToMel(fMax)
        let nPts = numMelBins + 2
        var fPts = [Float](repeating: 0, count: nPts)
        for i in 0 ..< nPts {
            let mel = mMin + (mMax - mMin) * Float(i) / Float(nPts - 1)
            fPts[i] = melToHz(mel)
        }

        var fb = [Float](repeating: 0, count: numMelBins * nBins)
        for m in 0 ..< numMelBins {
            let lower = fPts[m]
            let center = fPts[m + 1]
            let upper = fPts[m + 2]
            let dLow = center - lower
            let dHigh = upper - center
            for k in 0 ..< nBins {
                let freq = allFreqs[k]
                let down = (freq - lower) / dLow
                let up = (upper - freq) / dHigh
                fb[m * nBins + k] = max(0.0, min(down, up))
            }
        }
        return fb
    }

    /// Returns the log-mel features `(frames, 80)` as an MLX float32 array.
    static func features(_ audio: [Float]) -> MLXArray {
        let nBins = nFft / 2 + 1
        let signalLen = audio.count
        var numFrames = (signalLen - winLength) / hopLength + 1
        if numFrames < 1 { numFrames = 1 }

        let window = poveyWindow()

        // Build windowed frames on the host: per-frame DC removal, pre-emphasis,
        // window, then zero-pad to nFft. Layout: [numFrames, nFft].
        var framed = [Float](repeating: 0, count: numFrames * nFft)
        for t in 0 ..< numFrames {
            let start = t * hopLength
            // Copy + DC removal.
            var frame = [Float](repeating: 0, count: winLength)
            var mean: Float = 0
            for i in 0 ..< winLength {
                let idx = start + i
                frame[i] = idx < signalLen ? audio[idx] : 0
                mean += frame[i]
            }
            mean /= Float(winLength)
            for i in 0 ..< winLength { frame[i] -= mean }
            // Pre-emphasis: frame[1:] - 0.97 * frame[:-1]; frame[0] unchanged.
            var prev = frame[0]
            for i in 1 ..< winLength {
                let cur = frame[i]
                frame[i] = cur - preemph * prev
                prev = cur
            }
            // Window + write into padded row.
            let base = t * nFft
            for i in 0 ..< winLength { framed[base + i] = frame[i] * window[i] }
        }

        let framedMx = MLXArray(framed, [numFrames, nFft])
        let spec = rfft(framedMx, axis: -1)              // [numFrames, nBins] complex
        let power = abs(spec) * abs(spec)                // |.|^2

        let fbFloats = melFilterbank()
        let fb = MLXArray(fbFloats, [numMelBins, nBins])
        let mel = matmul(power, fb.transposed())         // [numFrames, numMelBins]
        return MLX.log(maximum(mel, MLXArray(logFloor)))
    }
}

// MARK: - Building blocks

/// `batchnorm-relu` non-linearity: a `BatchNorm` (key `.0`) followed by ReLU.
/// The optional `affine` toggle covers the `batchnorm_` variant (no weight/bias),
/// and `relu` toggles whether the ReLU is applied.
final class NonLinear: Module, UnaryLayer {
    @ModuleInfo(key: "0") var bn: BatchNorm
    let relu: Bool

    init(_ channels: Int, affine: Bool = true, relu: Bool = true) {
        self._bn.wrappedValue = BatchNorm(featureCount: channels, affine: affine)
        self.relu = relu
        super.init()
    }

    /// Input/output in channels-last `(.., C)` so `BatchNorm` normalizes channels.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = bn(x)
        return relu ? MLXNN.relu(y) : y
    }
}

/// Basic 2-D residual block used by the FCM head. Stride applies to the
/// frequency (H) dimension only, matching the reference `stride=(stride, 1)`.
final class BasicResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm
    @ModuleInfo(key: "shortcut") var shortcut: [Module]

    init(inPlanes: Int, planes: Int, stride: Int = 1) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inPlanes, outputChannels: planes,
            kernelSize: .init((3, 3)), stride: .init((stride, 1)),
            padding: .init((1, 1)), bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: planes)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: planes, outputChannels: planes,
            kernelSize: .init((3, 3)), stride: .init((1, 1)),
            padding: .init((1, 1)), bias: false)
        self._bn2.wrappedValue = BatchNorm(featureCount: planes)

        if stride != 1 || inPlanes != planes {
            self._shortcut.wrappedValue = [
                Conv2d(
                    inputChannels: inPlanes, outputChannels: planes,
                    kernelSize: .init((1, 1)), stride: .init((stride, 1)), bias: false),
                BatchNorm(featureCount: planes),
            ]
        } else {
            self._shortcut.wrappedValue = []
        }
        super.init()
    }

    /// x: NHWC `(B, H, W, C)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = MLXNN.relu(bn1(conv1(x)))
        out = bn2(conv2(out))
        var sc = x
        for layer in shortcut {
            // shortcut is [Conv2d, BatchNorm]; both are UnaryLayer.
            sc = (layer as! UnaryLayer)(sc)
        }
        return MLXNN.relu(out + sc)
    }
}

/// FCM front-end: a small 2-D ResNet over the mel feature map, flattened back to
/// `(B, C', T)` for the TDNN trunk. Output channels = `mChannels * (featDim / 8)`.
final class FCM: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "bn1") var bn1: BatchNorm
    @ModuleInfo(key: "layer1") var layer1: [BasicResBlock]
    @ModuleInfo(key: "layer2") var layer2: [BasicResBlock]
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "bn2") var bn2: BatchNorm

    let outChannels: Int

    init(mChannels: Int = 32, featDim: Int = 80) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: mChannels,
            kernelSize: .init((3, 3)), stride: .init((1, 1)),
            padding: .init((1, 1)), bias: false)
        self._bn1.wrappedValue = BatchNorm(featureCount: mChannels)

        // Two stages of 2 blocks each; first block of each stage strides H by 2.
        var inPlanes = mChannels
        func makeLayer() -> [BasicResBlock] {
            let strides = [2, 1]
            var blocks: [BasicResBlock] = []
            for s in strides {
                blocks.append(BasicResBlock(inPlanes: inPlanes, planes: mChannels, stride: s))
                inPlanes = mChannels
            }
            return blocks
        }
        self._layer1.wrappedValue = makeLayer()
        self._layer2.wrappedValue = makeLayer()

        self._conv2.wrappedValue = Conv2d(
            inputChannels: mChannels, outputChannels: mChannels,
            kernelSize: .init((3, 3)), stride: .init((2, 1)),
            padding: .init((1, 1)), bias: false)
        self._bn2.wrappedValue = BatchNorm(featureCount: mChannels)
        self.outChannels = mChannels * (featDim / 8)
        super.init()
    }

    /// x: PyTorch-style `(B, F, T)`. Returns `(B, C * H, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // (B, F, T) -> NHWC (B, H=F, W=T, C=1).
        var out = expandedDimensions(x, axis: -1)
        out = MLXNN.relu(bn1(conv1(out)))
        for blk in layer1 { out = blk(out) }
        for blk in layer2 { out = blk(out) }
        out = MLXNN.relu(bn2(conv2(out)))

        // NHWC (B, H, W, C) -> (B, C, H, W) -> (B, C*H, W).
        let (b, h, w, c) = (out.dim(0), out.dim(1), out.dim(2), out.dim(3))
        out = out.transposed(0, 3, 1, 2)
        return out.reshaped([b, c * h, w])
    }
}

/// Initial TDNN layer (Conv1d + batchnorm-relu).
final class TDNNLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NonLinear

    init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, dilation: Int = 1
    ) {
        // padding = -1 in the reference -> symmetric "same" padding for odd kernels.
        let padding = (kernelSize - 1) / 2 * dilation
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, bias: false)
        self._nonlinear.wrappedValue = NonLinear(outChannels)
        super.init()
    }

    /// x: `(B, C, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.swappedAxes(1, 2)   // (B, T, C)
        y = linear(y)                  // (B, T', C')
        y = nonlinear(y)               // batchnorm + relu in channels-last
        return y.swappedAxes(1, 2)     // (B, C', T')
    }
}

/// Context-aware masking layer inside a dense TDNN layer.
final class CAMLayer: Module {
    @ModuleInfo(key: "linear_local") var linearLocal: Conv1d
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "linear2") var linear2: Conv1d

    let segLen = 100

    init(
        bnChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int, padding: Int, dilation: Int, reduction: Int = 2
    ) {
        self._linearLocal.wrappedValue = Conv1d(
            inputChannels: bnChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, bias: false)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: bnChannels, outputChannels: bnChannels / reduction,
            kernelSize: 1)
        self._linear2.wrappedValue = Conv1d(
            inputChannels: bnChannels / reduction, outputChannels: outChannels,
            kernelSize: 1)
        super.init()
    }

    /// Conv1d wrapper for `(B, C, T)` tensors.
    private func conv(_ x: MLXArray, _ layer: Conv1d) -> MLXArray {
        layer(x.swappedAxes(1, 2)).swappedAxes(1, 2)
    }

    /// x: `(B, C, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv(x, linearLocal)
        var context = MLX.mean(x, axis: -1, keepDims: true) + segPooling(x)
        context = MLXNN.relu(conv(context, linear1))
        let m = MLXNN.sigmoid(conv(context, linear2))
        return y * m
    }

    /// Segment average pooling with broadcast back to T (ceil-mode segments).
    private func segPooling(_ x: MLXArray) -> MLXArray {
        let (b, c, t) = (x.dim(0), x.dim(1), x.dim(2))
        let nSegs = (t + segLen - 1) / segLen
        let padLen = nSegs * segLen - t
        var xp = x
        if padLen > 0 {
            xp = concatenated([x, MLXArray.zeros([b, c, padLen])], axis: -1)
        }
        var seg = xp.reshaped([b, c, nSegs, segLen])
        seg = MLX.mean(seg, axis: -1)                         // (B, C, nSegs)
        seg = expandedDimensions(seg, axis: -1)               // (B, C, nSegs, 1)
        seg = broadcast(seg, to: [b, c, nSegs, segLen])
        seg = seg.reshaped([b, c, nSegs * segLen])
        return seg[0..., 0..., 0 ..< t]
    }
}

/// One layer of a dense TDNN block: two batchnorm-relu + 1x1 bottleneck, then CAM.
final class CAMDenseTDNNLayer: Module {
    @ModuleInfo(key: "nonlinear1") var nonlinear1: NonLinear
    @ModuleInfo(key: "linear1") var linear1: Conv1d
    @ModuleInfo(key: "nonlinear2") var nonlinear2: NonLinear
    @ModuleInfo(key: "cam_layer") var camLayer: CAMLayer

    init(
        inChannels: Int, outChannels: Int, bnChannels: Int,
        kernelSize: Int, stride: Int = 1, dilation: Int = 1
    ) {
        let padding = (kernelSize - 1) / 2 * dilation
        self._nonlinear1.wrappedValue = NonLinear(inChannels)
        self._linear1.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: bnChannels,
            kernelSize: 1, bias: false)
        self._nonlinear2.wrappedValue = NonLinear(bnChannels)
        self._camLayer.wrappedValue = CAMLayer(
            bnChannels: bnChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation)
        super.init()
    }

    /// x: `(B, C, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.swappedAxes(1, 2)    // (B, T, C)
        y = nonlinear1(y)
        y = linear1(y)
        y = nonlinear2(y)
        y = y.swappedAxes(1, 2)        // (B, C, T)
        return camLayer(y)
    }
}

/// Dense TDNN block: each layer's output is concatenated onto the channel axis.
final class CAMDenseTDNNBlock: Module {
    @ModuleInfo(key: "layers") var layers: [CAMDenseTDNNLayer]

    init(
        numLayers: Int, inChannels: Int, outChannels: Int, bnChannels: Int,
        kernelSize: Int, stride: Int = 1, dilation: Int = 1
    ) {
        self._layers.wrappedValue = (0 ..< numLayers).map { i in
            CAMDenseTDNNLayer(
                inChannels: inChannels + i * outChannels,
                outChannels: outChannels, bnChannels: bnChannels,
                kernelSize: kernelSize, stride: stride, dilation: dilation)
        }
        super.init()
    }

    /// x: `(B, C, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers {
            out = concatenated([out, layer(out)], axis: 1)
        }
        return out
    }
}

/// Transition layer between dense blocks (batchnorm-relu + 1x1 conv).
final class TransitLayer: Module {
    @ModuleInfo(key: "nonlinear") var nonlinear: NonLinear
    @ModuleInfo(key: "linear") var linear: Conv1d

    init(inChannels: Int, outChannels: Int, bias: Bool = false) {
        self._nonlinear.wrappedValue = NonLinear(inChannels)
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 1, bias: bias)
        super.init()
    }

    /// x: `(B, C, T)`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x.swappedAxes(1, 2)    // (B, T, C)
        y = nonlinear(y)
        y = linear(y)
        return y.swappedAxes(1, 2)     // (B, C, T)
    }
}

/// Final projection to the embedding (1x1 conv + non-affine batchnorm, no relu).
final class DenseLayer: Module {
    @ModuleInfo(key: "linear") var linear: Conv1d
    @ModuleInfo(key: "nonlinear") var nonlinear: NonLinear

    init(inChannels: Int, outChannels: Int) {
        self._linear.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 1, bias: false)
        // config_str="batchnorm_" -> affine=false, and no relu.
        self._nonlinear.wrappedValue = NonLinear(outChannels, affine: false, relu: false)
        super.init()
    }

    /// x: `(B, C)` (statistics-pooled vector). Returns `(B, C')`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = expandedDimensions(x, axis: 1)   // (B, 1, C)
        y = linear(y)                             // (B, 1, C')
        y = nonlinear(y)                          // batchnorm (no relu), channels-last
        return squeezed(y, axis: 1)               // (B, C')
    }
}

// MARK: - CAMPPlus

/// CAM++ speaker encoder producing a 192-d x-vector from 16 kHz audio.
public final class CAMPPlus: Module {
    @ModuleInfo(key: "head") var head: FCM
    @ModuleInfo(key: "tdnn") var tdnn: TDNNLayer
    @ModuleInfo(key: "blocks") var blocks: [CAMDenseTDNNBlock]
    @ModuleInfo(key: "transits") var transits: [TransitLayer]
    @ModuleInfo(key: "out_nonlinear") var outNonlinear: NonLinear
    @ModuleInfo(key: "dense") var dense: DenseLayer

    public static let embeddingSize = 192
    public static let featDim = 80
    public static let sampleRate = 16000

    public init(
        featDim: Int = 80, embeddingSize: Int = 192,
        growthRate: Int = 32, bnSize: Int = 4, initChannels: Int = 128
    ) {
        let head = FCM(featDim: featDim)
        self._head.wrappedValue = head
        var channels = head.outChannels

        self._tdnn.wrappedValue = TDNNLayer(
            inChannels: channels, outChannels: initChannels,
            kernelSize: 5, stride: 2, dilation: 1)
        channels = initChannels

        // Three dense blocks with matching transitions (halving channels).
        let blockSpecs: [(numLayers: Int, kernelSize: Int, dilation: Int)] =
            [(12, 3, 1), (24, 3, 2), (16, 3, 2)]
        var blocks: [CAMDenseTDNNBlock] = []
        var transits: [TransitLayer] = []
        for spec in blockSpecs {
            blocks.append(CAMDenseTDNNBlock(
                numLayers: spec.numLayers, inChannels: channels,
                outChannels: growthRate, bnChannels: bnSize * growthRate,
                kernelSize: spec.kernelSize, dilation: spec.dilation))
            channels = channels + spec.numLayers * growthRate
            transits.append(TransitLayer(inChannels: channels, outChannels: channels / 2))
            channels /= 2
        }
        self._blocks.wrappedValue = blocks
        self._transits.wrappedValue = transits

        self._outNonlinear.wrappedValue = NonLinear(channels)
        // Statistics pooling doubles the channel count before the dense projection.
        self._dense.wrappedValue = DenseLayer(inChannels: channels * 2, outChannels: embeddingSize)
        super.init()
    }

    /// Forward over precomputed features `(B, T, F)` -> embeddings `(B, embeddingSize)`.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features.swappedAxes(1, 2)   // (B, T, F) -> (B, F, T)
        x = head(x)
        x = tdnn(x)
        for (block, transit) in zip(blocks, transits) {
            x = block(x)
            x = transit(x)
        }
        // out_nonlinear (batchnorm+relu) in channels-last.
        x = x.swappedAxes(1, 2)              // (B, C, T) -> (B, T, C)
        x = outNonlinear(x)
        x = x.swappedAxes(1, 2)              // (B, T, C) -> (B, C, T)

        x = statisticsPooling(x)             // (B, 2C)
        return dense(x)                      // (B, embeddingSize)
    }

    /// Speaker x-vector from a 16 kHz mono waveform `(T,)` or `(B, T)`.
    /// Mirrors the reference `inference`: per-utterance kaldi fbank, then subtract
    /// the per-utterance temporal mean, pad to a common length, run the network.
    public func inference(_ refWav16k: [Float]) -> MLXArray {
        var feat = KaldiFbank.features(refWav16k)                  // (T, F)
        feat = feat - MLX.mean(feat, axis: 0, keepDims: true)      // CMN over time
        let batch = expandedDimensions(feat, axis: 0)              // (1, T, F)
        return self(batch)
    }

    /// Statistics pooling over the time axis: concat(mean, std) on the channel axis.
    /// x: `(B, C, T)` -> `(B, 2C)`.
    private func statisticsPooling(_ x: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1)                           // (B, C)
        let variance = MLX.variance(x, axis: -1)                   // (B, C)
        let std = MLX.sqrt(variance + MLXArray(Float(1e-5)))
        return concatenated([mean, std], axis: -1)                 // (B, 2C)
    }
}
