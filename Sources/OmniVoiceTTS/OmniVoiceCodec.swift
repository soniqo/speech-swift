import Foundation
import MLX
import MLXNN

// MARK: - Snake activation

/// Snake activation as used by DAC: `x + (1/(alpha+1e-9)) * sin(alpha*x)^2`,
/// with a learned per-channel `alpha` of shape `[1, C, 1]` (NCL broadcast).
///
/// The checkpoint stores `alpha` exactly as `[1, C, 1]`, so we keep that shape
/// as the parameter (matches the key `*.alpha`) and broadcast directly.
final class OVSnake1d: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray  // [1, C, 1]

    init(channels: Int) {
        self._alpha.wrappedValue = MLXArray.ones([1, channels, 1])
        super.init()
    }

    /// Input/output: `[B, C, T]` (NCL).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = alpha  // [1, C, 1]
        let s = sin(a * x)
        return x + (1.0 / (a + 1e-9)) * (s * s)
    }
}

// MARK: - NCL Conv1d / ConvTranspose1d wrappers

/// Conv1d operating in NCL (channels-first) layout, wrapping MLX's NLC `Conv1d`.
/// The underlying `conv.weight` key is `<name>.conv.weight`; the loader remaps the
/// PyTorch `[out, in, k]` weight into MLX's `[out, k, in]` (transpose axes 1↔2).
final class OVConv1dNCL: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d

    init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, dilation: Int = 1, bias: Bool = true,
        groups: Int = 1
    ) {
        self._conv.wrappedValue = Conv1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            dilation: dilation, groups: groups, bias: bias)
        super.init()
    }

    /// Input: `[B, C, T]` -> Output: `[B, C_out, T_out]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

/// ConvTranspose1d operating in NCL layout, wrapping MLX's NLC `ConvTransposed1d`.
/// PyTorch transposed-conv weight is `[in, out, k]`; MLX wants `[out, k, in]`. The
/// loader remaps `[in, out, k]` -> `[out, k, in]` (transpose axes to (1, 2, 0)).
final class OVConvTranspose1dNCL: Module {
    @ModuleInfo(key: "conv") var conv: ConvTransposed1d

    init(
        inChannels: Int, outChannels: Int, kernelSize: Int,
        stride: Int = 1, padding: Int = 0, outputPadding: Int = 0, bias: Bool = true
    ) {
        self._conv.wrappedValue = ConvTransposed1d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: kernelSize, stride: stride, padding: padding,
            outputPadding: outputPadding, bias: bias)
        super.init()
    }

    /// Input: `[B, C, T]` -> Output: `[B, C_out, T_out]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
    }
}

// MARK: - DAC residual unit

/// DAC residual unit: `out = conv2(snake2(conv1(snake1(x))))`, then center-crop
/// the skip `x` to the (smaller) `out` length and add. conv1 is a dilated k=7
/// conv with `padding = ((7-1)*dilation)//2`; conv2 is a k=1 conv. Both carry
/// bias in this checkpoint.
final class DacResidualUnit: Module {
    @ModuleInfo(key: "snake1") var snake1: OVSnake1d
    @ModuleInfo(key: "conv1") var conv1: OVConv1dNCL
    @ModuleInfo(key: "snake2") var snake2: OVSnake1d
    @ModuleInfo(key: "conv2") var conv2: OVConv1dNCL

    init(dimension: Int, dilation: Int) {
        let pad = ((7 - 1) * dilation) / 2
        self._snake1.wrappedValue = OVSnake1d(channels: dimension)
        self._conv1.wrappedValue = OVConv1dNCL(
            inChannels: dimension, outChannels: dimension, kernelSize: 7,
            padding: pad, dilation: dilation)
        self._snake2.wrappedValue = OVSnake1d(channels: dimension)
        self._conv2.wrappedValue = OVConv1dNCL(
            inChannels: dimension, outChannels: dimension, kernelSize: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1(snake1(x))
        out = conv2(snake2(out))
        let pad = (x.dim(2) - out.dim(2)) / 2
        var skip = x
        if pad > 0 {
            skip = x[0..., 0..., pad ..< (x.dim(2) - pad)]
        }
        return skip + out
    }
}

// MARK: - DAC decoder block

/// DAC decoder block: `snake1 -> conv_t1 (upsample) -> res_unit1/2/3 (dil 1,3,9)`.
/// `conv_t1` is a transposed conv with `kernel=2*stride`, `padding=ceil(stride/2)`,
/// and (per Higgs `_adjust_dac_decoder`) `output_padding = stride % 2`.
final class DacDecoderBlock: Module {
    @ModuleInfo(key: "snake1") var snake1: OVSnake1d
    @ModuleInfo(key: "conv_t1") var convT1: OVConvTranspose1dNCL
    @ModuleInfo(key: "res_unit1") var resUnit1: DacResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: DacResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: DacResidualUnit

    init(inputDim: Int, outputDim: Int, stride: Int) {
        self._snake1.wrappedValue = OVSnake1d(channels: inputDim)
        let padding = Int((Double(stride) / 2.0).rounded(.up))  // ceil(stride/2)
        self._convT1.wrappedValue = OVConvTranspose1dNCL(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride,
            padding: padding, outputPadding: stride % 2)
        self._resUnit1.wrappedValue = DacResidualUnit(dimension: outputDim, dilation: 1)
        self._resUnit2.wrappedValue = DacResidualUnit(dimension: outputDim, dilation: 3)
        self._resUnit3.wrappedValue = DacResidualUnit(dimension: outputDim, dilation: 9)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = snake1(x)
        h = convT1(h)
        h = resUnit1(h)
        h = resUnit2(h)
        h = resUnit3(h)
        return h
    }
}

// MARK: - DAC acoustic decoder

/// DAC decoder: `conv1 (k=7) -> block.0..4 -> snake1 -> conv2 (k=7)`. Higgs removes
/// the final tanh (`_adjust_dac_decoder` -> Identity), so there is no tanh here.
/// Channel ladder: 1024 -> 512 -> 256 -> 128 -> 64 -> 32 over upsampling ratios
/// `[8,5,4,2,3]`; conv2 maps 32 -> 1.
final class DacAcousticDecoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OVConv1dNCL
    @ModuleInfo(key: "block") var block: [DacDecoderBlock]
    @ModuleInfo(key: "snake1") var snake1: OVSnake1d
    @ModuleInfo(key: "conv2") var conv2: OVConv1dNCL

    init(inputChannels: Int, decoderHidden: Int, upsamplingRatios: [Int]) {
        self._conv1.wrappedValue = OVConv1dNCL(
            inChannels: inputChannels, outChannels: decoderHidden,
            kernelSize: 7, padding: 3)

        var blocks: [DacDecoderBlock] = []
        for (i, stride) in upsamplingRatios.enumerated() {
            let inDim = decoderHidden / (1 << i)
            let outDim = decoderHidden / (1 << (i + 1))
            blocks.append(DacDecoderBlock(inputDim: inDim, outputDim: outDim, stride: stride))
        }
        self._block.wrappedValue = blocks

        let outDim = decoderHidden / (1 << upsamplingRatios.count)
        self._snake1.wrappedValue = OVSnake1d(channels: outDim)
        self._conv2.wrappedValue = OVConv1dNCL(
            inChannels: outDim, outChannels: 1, kernelSize: 7, padding: 3)
        super.init()
    }

    /// Input: `[B, hidden, T]` -> Output: `[B, 1, T*960]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block {
            h = b(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }
}

// MARK: - RVQ quantizer (decode path)

/// A single RVQ codebook decode path: `embedding(indices) @ project_out`.
/// `codebook.embed` is `[1024, 64]`; `project_out` is a Linear 64 -> 1024.
/// (Encoder-only buffers — `cluster_size`, `embed_avg`, `inited`, `project_in` —
/// are dropped at load.)
final class OVVectorQuantizer: Module {
    @ModuleInfo(key: "codebook") var codebook: OVCodebook
    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(hiddenSize: Int, codebookSize: Int, codebookDim: Int) {
        self._codebook.wrappedValue = OVCodebook(
            codebookSize: codebookSize, codebookDim: codebookDim)
        self._projectOut.wrappedValue = Linear(codebookDim, hiddenSize)
        super.init()
    }

    /// indices: `[B, T]` Int32 -> `[B, T, hidden]`.
    func decode(_ indices: MLXArray) -> MLXArray {
        projectOut(codebook(indices))
    }
}

/// Euclidean codebook decode: `F.embedding(indices, embed)`. Holds `embed`
/// `[codebookSize, codebookDim]` under key `codebook.embed`.
final class OVCodebook: Module {
    @ParameterInfo(key: "embed") var embed: MLXArray  // [N, D]

    init(codebookSize: Int, codebookDim: Int) {
        self._embed.wrappedValue = MLXArray.zeros([codebookSize, codebookDim])
        super.init()
    }

    /// indices: `[B, T]` -> `[B, T, D]`.
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        embed[indices]
    }
}

/// Residual vector quantizer decode: for each of the 8 codebooks, embed + project,
/// then sum. Input codes `[B, 8, T]` -> `[B, hidden, T]` (NCL, after a transpose).
final class OVResidualVQ: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [OVVectorQuantizer]

    init(numQuantizers: Int, hiddenSize: Int, codebookSize: Int, codebookDim: Int) {
        var qs: [OVVectorQuantizer] = []
        for _ in 0 ..< numQuantizers {
            qs.append(OVVectorQuantizer(
                hiddenSize: hiddenSize, codebookSize: codebookSize, codebookDim: codebookDim))
        }
        self._quantizers.wrappedValue = qs
        super.init()
    }

    /// codes: `[B, numQuantizers, T]` Int32 -> quantized `[B, hidden, T]` (NCL).
    func decode(_ codes: MLXArray) -> MLXArray {
        // Reference sums each quantizer's `[B, hidden, T]` (it permutes to NCL per
        // quantizer). We accumulate in NLC `[B, T, hidden]` then transpose once.
        var acc: MLXArray? = nil
        for (i, q) in quantizers.enumerated() {
            let indices = codes[0..., i, 0...]            // [B, T]
            let quant = q.decode(indices)                 // [B, T, hidden]
            acc = (acc == nil) ? quant : acc! + quant
        }
        return acc!.transposed(0, 2, 1)                   // [B, hidden, T]
    }
}

// MARK: - Codec

/// Higgs-audio v2 codec **decoder** (audio tokens -> 24 kHz waveform).
///
/// Mirrors `HiggsAudioV2TokenizerModel.decode`:
/// `codes.transpose(0,1)` (handled by indexing) -> RVQ decode (sum of 8 codebooks)
/// -> `fc2` (Linear 1024 -> 256) -> DAC `acoustic_decoder` (960x upsample) -> wav.
///
/// Weights: the audio-tokenizer `model.safetensors` (e.g. under the OmniVoice
/// bundle's `audio_tokenizer/`). Encoder/semantic/EMA keys are dropped at load;
/// only the decode-path keys are kept and conv weights are transposed into MLX
/// layout.
public final class OmniVoiceCodec: Module {
    @ModuleInfo(key: "quantizer") var quantizer: OVResidualVQ
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "acoustic_decoder") var acousticDecoder: DacAcousticDecoder

    let numQuantizers: Int

    public init(
        numQuantizers: Int = 8,
        hiddenSize: Int = 1024,
        codebookSize: Int = 1024,
        codebookDim: Int = 64,
        acousticHidden: Int = 256,
        decoderHidden: Int = 1024,
        upsamplingRatios: [Int] = [8, 5, 4, 2, 3]
    ) {
        self.numQuantizers = numQuantizers
        self._quantizer.wrappedValue = OVResidualVQ(
            numQuantizers: numQuantizers, hiddenSize: hiddenSize,
            codebookSize: codebookSize, codebookDim: codebookDim)
        self._fc2.wrappedValue = Linear(hiddenSize, acousticHidden)
        self._acousticDecoder.wrappedValue = DacAcousticDecoder(
            inputChannels: acousticHidden, decoderHidden: decoderHidden,
            upsamplingRatios: upsamplingRatios)
        super.init()
    }

    /// Decode audio codes to a mono 24 kHz waveform.
    /// - Parameter audioCodes: `[1, 8, T]` Int32 codebook indices.
    /// - Returns: `[T*960]` Float32 waveform samples.
    public func decode(_ audioCodes: MLXArray) -> MLXArray {
        // RVQ decode -> [B, hidden=1024, T]
        let quantized = quantizer.decode(audioCodes)
        // fc2: Linear over the channel axis. NCL [B,1024,T] -> NLC -> Linear -> NCL.
        let q = fc2(quantized.transposed(0, 2, 1)).transposed(0, 2, 1)  // [B, 256, T]
        let wav = acousticDecoder(q)                                    // [B, 1, T*960]
        return wav.reshaped([-1])                                       // [T*960]
    }

    /// Load the audio-tokenizer `model.safetensors`, keeping only decode-path keys
    /// and remapping conv weights into MLX layout. Uses `verify: .all` so the kept
    /// key set must match this module's parameters exactly.
    public func loadWeights(from modelSafetensors: URL) throws {
        let raw = try MLX.loadArrays(url: modelSafetensors)
        var weights: [String: MLXArray] = [:]

        for (k, v) in raw {
            // Keep only decode-path keys.
            let keep =
                k.hasPrefix("fc2.")
                || k.hasPrefix("acoustic_decoder.")
                || (k.hasPrefix("quantizer.")
                    && (k.contains(".codebook.embed") && !k.contains("embed_avg")
                        || k.contains(".project_out.")))
            guard keep else { continue }

            var arr = v.asType(.float32)

            if k.hasPrefix("acoustic_decoder.") && k.hasSuffix(".weight") {
                if k.contains(".conv_t1.") {
                    // PyTorch ConvTranspose1d weight [in, out, k] -> MLX [out, k, in].
                    arr = arr.transposed(1, 2, 0)
                } else if arr.ndim == 3 {
                    // PyTorch Conv1d weight [out, in, k] -> MLX [out, k, in].
                    arr = arr.transposed(0, 2, 1)
                }
                // snake alphas ([1, C, 1]) and biases ([C]) pass through unchanged.
            }

            // Map checkpoint conv keys (`*.weight`/`*.bias`) onto our wrapper's
            // nested `conv.*` parameter for acoustic_decoder convs.
            let mapped = remapKey(k)
            weights[mapped] = arr
        }

        try update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(parameters())
    }

    /// Insert the `.conv` level for the NCL conv wrappers so the flat checkpoint
    /// keys line up with `OVConv1dNCL` / `OVConvTranspose1dNCL` (`<name>.conv.*`).
    private func remapKey(_ k: String) -> String {
        guard k.hasPrefix("acoustic_decoder.") else { return k }
        guard k.hasSuffix(".weight") || k.hasSuffix(".bias") else { return k }

        // Split off the trailing param name (weight/bias).
        let parts = k.split(separator: ".").map(String.init)
        guard let param = parts.last else { return k }
        let prefix = parts.dropLast().joined(separator: ".")

        // Conv-bearing module names whose params live under a `.conv` child.
        let convOwners = ["conv1", "conv2", "conv_t1"]
        if let owner = prefix.split(separator: ".").last.map(String.init),
            convOwners.contains(owner)
        {
            return "\(prefix).conv.\(param)"
        }
        return k
    }
}
