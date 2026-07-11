import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon

// MARK: - Resampler (torchaudio sinc_interp_hann)

/// Windowed-sinc resampler matching `torchaudio.functional.resample`'s default
/// (`sinc_interp_hann`, lowpass_filter_width=6, rolloff=0.99). Builds the
/// polyphase FIR kernel exactly as torchaudio's `_get_sinc_resample_kernel`
/// and applies it with a strided conv (`_apply_sinc_resample_kernel`).
enum Resampler {
    static func resample(
        _ waveform: MLXArray, from origFreq: Int, to newFreq: Int,
        lowpassFilterWidth: Int = 6, rolloff: Double = 0.99
    ) -> MLXArray {
        if origFreq == newFreq { return waveform }
        let g = gcd(origFreq, newFreq)
        let orig = origFreq / g
        let neu = newFreq / g

        // --- Build the kernel: shape [neu, 1, 2*width + orig]. ---
        let baseFreq = Double(min(orig, neu)) * rolloff
        let width = Int((Double(lowpassFilterWidth) * Double(orig) / baseFreq).rounded(.up))
        let lfw = Double(lowpassFilterWidth)

        // idx[i] = (-width + i) / orig, for i in 0 ..< (2*width + orig).
        let klen = 2 * width + orig
        var kernel = [Float](repeating: 0, count: neu * klen)
        for j in 0 ..< neu {
            // t = (-j / neu) + idx ; t *= baseFreq ; clamp ; window ; t *= pi ; sinc
            let tj = -Double(j) / Double(neu)
            for i in 0 ..< klen {
                let idx = Double(-width + i) / Double(orig)
                var t = (tj + idx) * baseFreq
                if t > lfw { t = lfw }
                if t < -lfw { t = -lfw }
                let window = pow(cos(t * Double.pi / lfw / 2.0), 2.0)
                let tp = t * Double.pi
                let scale = baseFreq / Double(orig)
                let sinc = tp == 0 ? 1.0 : sin(tp) / tp
                kernel[j * klen + i] = Float(sinc * window * scale)
            }
        }

        // --- Apply: pad (width, width + orig), strided conv, interleave. ---
        var wav = waveform
        let packedToFlat = wav.ndim == 1
        if packedToFlat { wav = wav.reshaped([1, wav.dim(0)]) }
        let numWavs = wav.dim(0)
        let length = wav.dim(1)

        let padded = MLX.padded(
            wav, widths: [.init((0, 0)), .init((width, width + orig))])  // [num, L+pad]
        // conv1d: input [num, 1, Lp] (NCL), weight [neu, 1, klen], stride orig.
        // Use MLX conv via OVConv1dNCL-style: transpose to NLC.
        let inNLC = padded.reshaped([numWavs, 1, padded.dim(1)]).transposed(0, 2, 1)  // [num, Lp, 1]
        // MLX weight layout for conv1d: [outC, K, inC] = [neu, klen, 1].
        let w = MLXArray(kernel).reshaped([neu, 1, klen]).transposed(0, 2, 1)  // [neu, klen, 1]
        let convOut = MLX.conv1d(inNLC, w, stride: orig, padding: 0)  // [num, Tout, neu]
        // transpose(1,2) -> [num, neu, Tout] then reshape to [num, neu*Tout] interleaved.
        // torch does resampled.transpose(1,2).reshape(num, -1): conv output is
        // [num, neu, Tout]; transpose(1,2)->[num,Tout,neu]; reshape interleaves
        // neu within each time step. Our convOut is already [num, Tout, neu], so
        // reshape directly interleaves.
        var resampled = convOut.reshaped([numWavs, -1])  // [num, Tout*neu]

        let targetLength = Int((Double(neu) * Double(length) / Double(orig)).rounded(.up))
        if resampled.dim(1) > targetLength {
            resampled = resampled[0..., 0 ..< targetLength]
        }
        if packedToFlat { resampled = resampled.reshaped([resampled.dim(1)]) }
        return resampled
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = a, y = b
        while y != 0 { (x, y) = (y, x % y) }
        return x
    }
}

// MARK: - DAC acoustic encoder (mirror of the decoder)

/// DAC encoder residual unit: `out = conv2(snake2(conv1(snake1(x))))`, then add
/// the (center-cropped) skip. Identical structure to `DacResidualUnit` in
/// `OmniVoiceCodec.swift` — reused directly there. The encoder uses the same
/// unit, so we don't redefine it.

/// DAC encoder block: `res_unit1/2/3 (dil 1,3,9) -> snake1 -> conv1 (strided
/// downsample, kernel = 2*stride)`. Note the order is residual-units THEN the
/// strided downsample — the mirror of the decoder block (upsample then units).
final class DacEncoderBlock: Module {
    @ModuleInfo(key: "res_unit1") var resUnit1: DacResidualUnit
    @ModuleInfo(key: "res_unit2") var resUnit2: DacResidualUnit
    @ModuleInfo(key: "res_unit3") var resUnit3: DacResidualUnit
    @ModuleInfo(key: "snake1") var snake1: OVSnake1d
    @ModuleInfo(key: "conv1") var conv1: OVConv1dNCL

    init(inputDim: Int, outputDim: Int, stride: Int) {
        self._resUnit1.wrappedValue = DacResidualUnit(dimension: inputDim, dilation: 1)
        self._resUnit2.wrappedValue = DacResidualUnit(dimension: inputDim, dilation: 3)
        self._resUnit3.wrappedValue = DacResidualUnit(dimension: inputDim, dilation: 9)
        self._snake1.wrappedValue = OVSnake1d(channels: inputDim)
        // HF DAC encoder downsample: kernel = 2*stride, stride = stride,
        // padding = ceil(stride / 2).
        let padding = Int((Double(stride) / 2.0).rounded(.up))
        self._conv1.wrappedValue = OVConv1dNCL(
            inChannels: inputDim, outChannels: outputDim,
            kernelSize: 2 * stride, stride: stride, padding: padding)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resUnit1(x)
        h = resUnit2(h)
        h = resUnit3(h)
        h = snake1(h)
        h = conv1(h)
        return h
    }
}

/// DAC acoustic encoder: `conv1 (k=7) -> block.0..4 (downsample) -> snake1 ->
/// conv2 (k=3)`. Channel ladder mirrors the decoder reversed:
/// 1 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 over downsample ratios
/// `[8,5,4,2,3]`; conv2 maps 2048 -> 256 (the acoustic embedding dim).
final class DacAcousticEncoder: Module {
    @ModuleInfo(key: "conv1") var conv1: OVConv1dNCL
    @ModuleInfo(key: "block") var block: [DacEncoderBlock]
    @ModuleInfo(key: "snake1") var snake1: OVSnake1d
    @ModuleInfo(key: "conv2") var conv2: OVConv1dNCL

    init(encoderHidden: Int, outChannels: Int, downsamplingRatios: [Int]) {
        // conv1: 1 -> encoderHidden, k=7, pad=3.
        self._conv1.wrappedValue = OVConv1dNCL(
            inChannels: 1, outChannels: encoderHidden, kernelSize: 7, padding: 3)

        var blocks: [DacEncoderBlock] = []
        var dim = encoderHidden
        for stride in downsamplingRatios {
            let outDim = dim * 2
            blocks.append(DacEncoderBlock(inputDim: dim, outputDim: outDim, stride: stride))
            dim = outDim
        }
        self._block.wrappedValue = blocks

        // dim is now encoderHidden * 2^len(ratios) (= 2048 for 64 and 5 blocks).
        self._snake1.wrappedValue = OVSnake1d(channels: dim)
        self._conv2.wrappedValue = OVConv1dNCL(
            inChannels: dim, outChannels: outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    /// Input: `[B, 1, N]` -> Output: `[B, outChannels, T]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block {
            h = b(h)
        }
        h = snake1(h)
        h = conv2(h)
        return h
    }

    /// Conv1d output length helper (floor((L + 2*pad - dilation*(k-1) - 1)/stride)+1)
    /// applied across every conv in the encoder, matching torch's
    /// `_get_conv1d_output_lengths`. Mirrors the layer geometry above.
    static func outputLength(for inputLength: Int, downsamplingRatios: [Int]) -> Int {
        func conv(_ L: Int, _ k: Int, _ s: Int, _ p: Int, _ d: Int = 1) -> Int {
            (L + 2 * p - d * (k - 1) - 1) / s + 1
        }
        var L = inputLength
        // conv1: k7 s1 p3
        L = conv(L, 7, 1, 3)
        for stride in downsamplingRatios {
            // res units (k7 dilated p, k1) keep length; the strided conv1 downsamples.
            // res_unit conv1: k7, pad=((7-1)*dil)//2, stride 1 -> length preserved.
            // res_unit conv2: k1 -> length preserved.
            // strided conv1: k=2*stride, stride=stride, pad=ceil(stride/2).
            let pad = Int((Double(stride) / 2.0).rounded(.up))
            L = conv(L, 2 * stride, stride, pad)
        }
        // conv2: k3 s1 p1 -> length preserved.
        L = conv(L, 3, 1, 1)
        return L
    }
}

// MARK: - HuBERT semantic model

/// HuBERT feature-extractor conv layer. Layer 0 (`feat_extract_norm="group"`)
/// has a GroupNorm with `num_groups == num_channels` (per-channel norm over
/// time); all other layers are conv-only. Conv has no bias (`conv_bias=false`).
final class HubertConvLayer: Module {
    @ModuleInfo(key: "conv") var conv: OVConv1dNCL
    @ModuleInfo(key: "layer_norm") var layerNorm: GroupNorm?
    let useNorm: Bool

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, groupNorm: Bool) {
        self._conv.wrappedValue = OVConv1dNCL(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernelSize, stride: stride, bias: false)
        self.useNorm = groupNorm
        if groupNorm {
            self._layerNorm.wrappedValue = GroupNorm(
                groupCount: outChannels, dimensions: outChannels,
                eps: 1e-5, affine: true, pytorchCompatible: true)
        } else {
            self._layerNorm.wrappedValue = nil
        }
        super.init()
    }

    /// Input/output NCL `[B, C, T]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)  // [B, C, T]
        if useNorm, let ln = layerNorm {
            // GroupNorm expects features last; conv output is NCL, so run norm in
            // NLC and transpose back.
            h = ln(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        }
        h = gelu(h)
        return h
    }
}

/// HuBERT feature extractor: 7 strided conv layers over the raw 16 kHz waveform.
/// `conv_dim` all 512, strides `[5,2,2,2,2,2,2]`, kernels `[10,3,3,3,3,2,2]`.
/// Only layer 0 carries the GroupNorm. Total downsample 320x.
final class HubertFeatureEncoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [HubertConvLayer]

    override init() {
        let dims = [512, 512, 512, 512, 512, 512, 512]
        let kernels = [10, 3, 3, 3, 3, 2, 2]
        let strides = [5, 2, 2, 2, 2, 2, 2]
        var layers: [HubertConvLayer] = []
        var inC = 1
        for i in 0 ..< 7 {
            layers.append(HubertConvLayer(
                inChannels: inC, outChannels: dims[i],
                kernelSize: kernels[i], stride: strides[i], groupNorm: i == 0))
            inC = dims[i]
        }
        self._convLayers.wrappedValue = layers
        super.init()
    }

    /// Input: `[B, 1, N]` (NCL) -> Output: `[B, 512, T]` (NCL).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for l in convLayers {
            h = l(h)
        }
        return h
    }
}

/// HuBERT feature projection: LayerNorm(512) -> Linear(512 -> 768).
final class HubertFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "projection") var projection: Linear

    override init() {
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1e-5)
        self._projection.wrappedValue = Linear(512, 768)
        super.init()
    }

    /// Input NLC `[B, T, 512]` -> `[B, T, 768]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }
}

/// HuBERT positional conv embedding: a grouped (groups=16) Conv1d k=128 with
/// weight-norm, followed by "same-pad" trim (remove the last frame for even
/// kernel) and GELU. The loader folds weight-norm `original0`/`original1`
/// into a plain conv weight.
final class HubertPositionalConvEmbedding: Module {
    @ModuleInfo(key: "conv") var conv: OVConv1dNCL

    init(hidden: Int = 768, kernel: Int = 128, groups: Int = 16) {
        // padding = kernel // 2 = 64.
        self._conv.wrappedValue = OVConv1dNCL(
            inChannels: hidden, outChannels: hidden, kernelSize: kernel,
            stride: 1, padding: kernel / 2, dilation: 1, bias: true, groups: groups)
        super.init()
    }

    /// Input NCL `[B, 768, T]` -> NCL `[B, 768, T]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)            // [B, 768, T+1] (even kernel -> one extra frame)
        // same-pad: drop the last time step.
        h = h[0..., 0..., 0 ..< (h.dim(2) - 1)]
        return gelu(h)
    }
}

/// One post-norm HuBERT transformer encoder layer:
/// `h = layer_norm(h + attn(h))`, then `h = final_layer_norm(h + ffn(h))`.
final class HubertEncoderLayer: Module {
    @ModuleInfo(key: "attention") var attention: HubertAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: HubertFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hidden: Int = 768, heads: Int = 12, intermediate: Int = 3072) {
        self._attention.wrappedValue = HubertAttention(hidden: hidden, heads: heads)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-5)
        self._feedForward.wrappedValue = HubertFeedForward(hidden: hidden, intermediate: intermediate)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-5)
        super.init()
    }

    /// Input/output NLC `[B, T, 768]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(x)
        h = layerNorm(h)
        h = h + feedForward(h)
        h = finalLayerNorm(h)
        return h
    }
}

/// HuBERT self-attention (full bidirectional, no mask). q/k/v/out projections
/// are plain biased Linears. q is scaled by `1/sqrt(head_dim)`.
final class HubertAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let heads: Int
    let headDim: Int
    let scale: Float

    init(hidden: Int, heads: Int) {
        self.heads = heads
        self.headDim = hidden / heads
        self.scale = 1.0 / sqrt(Float(hidden / heads))
        self._qProj.wrappedValue = Linear(hidden, hidden, bias: true)
        self._kProj.wrappedValue = Linear(hidden, hidden, bias: true)
        self._vProj.wrappedValue = Linear(hidden, hidden, bias: true)
        self._outProj.wrappedValue = Linear(hidden, hidden, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let q = qProj(x)
        let k = kProj(x)
        let v = vProj(x)
        let merged = SDPA.multiHead(
            q: q, k: k, v: v,
            numHeads: heads, headDim: headDim, scale: scale, mask: nil)
        return outProj(merged)
    }
}

/// HuBERT FFN: `output_dense(gelu(intermediate_dense(x)))`.
final class HubertFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    init(hidden: Int, intermediate: Int) {
        self._intermediateDense.wrappedValue = Linear(hidden, intermediate, bias: true)
        self._outputDense.wrappedValue = Linear(intermediate, hidden, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }
}

/// HuBERT encoder: pos_conv + add, layer_norm, then 12 post-norm transformer
/// layers. Collects all hidden states (input to layer 0 + output of each layer
/// = 13) for the mean-over-all-layers used by the semantic feature extractor.
final class HubertEncoder: Module {
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: HubertPositionalConvEmbedding
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [HubertEncoderLayer]

    init(numLayers: Int = 12, hidden: Int = 768) {
        self._posConvEmbed.wrappedValue = HubertPositionalConvEmbedding()
        self._layerNorm.wrappedValue = LayerNorm(dimensions: hidden, eps: 1e-5)
        var ls: [HubertEncoderLayer] = []
        for _ in 0 ..< numLayers {
            ls.append(HubertEncoderLayer())
        }
        self._layers.wrappedValue = ls
        super.init()
    }

    /// Input NLC `[B, T, 768]` (the feature-projection output). Returns all 13
    /// hidden states, each NLC `[B, T, 768]`.
    func hiddenStates(_ x: MLXArray) -> [MLXArray] {
        // pos_conv runs in NCL.
        let pos = posConvEmbed(x.transposed(0, 2, 1)).transposed(0, 2, 1)  // [B, T, 768]
        var h = x + pos
        h = layerNorm(h)
        var states: [MLXArray] = [h]  // input to layer 0
        for l in layers {
            h = l(h)
            states.append(h)
        }
        return states
    }
}

/// HuBERT semantic model: feature_extractor -> feature_projection -> encoder.
/// Exposes `meanHiddenStates` (mean over the 13 encoder hidden states).
final class HubertSemanticModel: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: HubertFeatureEncoder
    @ModuleInfo(key: "feature_projection") var featureProjection: HubertFeatureProjection
    @ModuleInfo(key: "encoder") var encoder: HubertEncoder

    override init() {
        self._featureExtractor.wrappedValue = HubertFeatureEncoder()
        self._featureProjection.wrappedValue = HubertFeatureProjection()
        self._encoder.wrappedValue = HubertEncoder()
        super.init()
    }

    /// Input: `[B, 1, N]` 16 kHz waveform (already resampled + padded).
    /// Output: `[B, T, 768]` mean over all 13 hidden states.
    func meanHiddenStates(_ x: MLXArray) -> MLXArray {
        var feats = featureExtractor(x)             // [B, 512, T] NCL
        feats = feats.transposed(0, 2, 1)           // [B, T, 512] NLC
        let projected = featureProjection(feats)    // [B, T, 768]
        let states = encoder.hiddenStates(projected)  // 13 x [B, T, 768]
        var acc = states[0]
        for i in 1 ..< states.count {
            acc = acc + states[i]
        }
        return acc / Float(states.count)
    }
}

// MARK: - Semantic encoder (encoder_semantic)

/// Semantic-encoder residual unit (ELU-based, no bias):
/// `out = conv2(elu(conv1(elu(x))))`, then `x + out`. conv1 is a dilated k=3
/// conv (pad = ((k-1)//2)*dil); conv2 is k=1.
final class SemanticResidualUnit: Module {
    @ModuleInfo(key: "conv1") var conv1: OVConv1dNCL
    @ModuleInfo(key: "conv2") var conv2: OVConv1dNCL

    init(channels: Int, dilation: Int, unitKernel: Int = 3) {
        let pad = ((unitKernel - 1) / 2) * dilation
        self._conv1.wrappedValue = OVConv1dNCL(
            inChannels: channels, outChannels: channels, kernelSize: unitKernel,
            stride: 1, padding: pad, dilation: dilation, bias: false)
        self._conv2.wrappedValue = OVConv1dNCL(
            inChannels: channels, outChannels: channels, kernelSize: 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = conv1(elu(x))
        out = conv2(elu(out))
        return x + out
    }
}

/// Semantic-encoder block: `res_units (block_dilations) -> conv (stride)`.
/// For our config strides are 1 (kernel 3, pad 1) so length is preserved.
final class SemanticEncoderBlock: Module {
    @ModuleInfo(key: "res_units") var resUnits: [SemanticResidualUnit]
    @ModuleInfo(key: "conv") var conv: OVConv1dNCL

    init(inChannels: Int, outChannels: Int, stride: Int, blockDilations: [Int]) {
        var units: [SemanticResidualUnit] = []
        for d in blockDilations {
            units.append(SemanticResidualUnit(channels: inChannels, dilation: d))
        }
        self._resUnits.wrappedValue = units
        // stride==1 -> kernel 3; else kernel 2*stride. padding = (kernel-1)//2.
        let kernel = stride == 1 ? 3 : (2 * stride)
        let padding = (kernel - 1) / 2
        self._conv.wrappedValue = OVConv1dNCL(
            inChannels: inChannels, outChannels: outChannels,
            kernelSize: kernel, stride: stride, padding: padding, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for u in resUnits {
            h = u(h)
        }
        return conv(h)
    }
}

/// SemanticEncoder (`encoder_semantic`): `conv (k3 no bias) -> conv_blocks`.
/// Channels stay 768 and length stays T for our config (strides [1,1]).
final class SemanticEncoder: Module {
    @ModuleInfo(key: "conv") var conv: OVConv1dNCL
    @ModuleInfo(key: "conv_blocks") var convBlocks: [SemanticEncoderBlock]

    init(
        hidden: Int = 768, kernelSize: Int = 3,
        channelRatios: [Double] = [1, 1], strides: [Int] = [1, 1],
        blockDilations: [Int] = [1, 1]
    ) {
        self._conv.wrappedValue = OVConv1dNCL(
            inChannels: hidden, outChannels: hidden, kernelSize: kernelSize,
            stride: 1, padding: kernelSize / 2, bias: false)
        var blocks: [SemanticEncoderBlock] = []
        var inC = hidden
        for i in 0 ..< strides.count {
            let outC = Int(Double(hidden) * channelRatios[i])
            blocks.append(SemanticEncoderBlock(
                inChannels: inC, outChannels: outC, stride: strides[i],
                blockDilations: blockDilations))
            inC = outC
        }
        self._convBlocks.wrappedValue = blocks
        super.init()
    }

    /// Input/output NCL `[B, 768, T]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        for b in convBlocks {
            h = b(h)
        }
        return h
    }
}

// MARK: - RVQ encode path

/// A single RVQ codebook encode path: `project_in` (Linear 1024 -> 64) then
/// nearest-neighbour search against `codebook.embed` (`[1024, 64]`).
/// Reuses `OVCodebook` (which holds `embed`) and adds `project_in` / `project_out`.
final class OVVectorQuantizerEncode: Module {
    @ModuleInfo(key: "codebook") var codebook: OVCodebook
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear

    init(hiddenSize: Int, codebookSize: Int, codebookDim: Int) {
        self._codebook.wrappedValue = OVCodebook(
            codebookSize: codebookSize, codebookDim: codebookDim)
        self._projectIn.wrappedValue = Linear(hiddenSize, codebookDim)
        self._projectOut.wrappedValue = Linear(codebookDim, hiddenSize)
        super.init()
    }

    /// hidden: `[B, hidden, T]` NCL -> indices `[B, T]` Int32.
    func encode(_ hidden: MLXArray) -> MLXArray {
        // reference: permute to [B, T, hidden], project_in -> [B, T, 64].
        let x = projectIn(hidden.transposed(0, 2, 1))  // [B, T, 64]
        let shape = x.shape  // [B, T, 64]
        let flat = x.reshaped([-1, shape[shape.count - 1]])  // [B*T, 64]
        let idx = quantize(flat)                              // [B*T]
        return idx.reshaped([shape[0], shape[1]])             // [B, T]
    }

    /// Nearest codebook entry by Euclidean distance.
    /// `embed = codebook.embed.T` ([64, 1024]);
    /// `dist = -(x^2.sum(1) - 2 x@embed + embed^2.sum(0))`; argmax over the dist.
    private func quantize(_ x: MLXArray) -> MLXArray {
        let embed = codebook.embed.transposed(1, 0)               // [64, 1024]
        let scaled = (x * x).sum(axis: 1, keepDims: true)         // [N, 1]
        let cross = matmul(x, embed)                              // [N, 1024]
        let embSq = (embed * embed).sum(axis: 0, keepDims: true)  // [1, 1024]
        let dist = -(scaled - 2 * cross + embSq)                  // [N, 1024]
        return argMax(dist, axis: -1).asType(.int32)             // [N]
    }

    /// indices `[B, T]` -> quantized `[B, hidden, T]` NCL (for residual update).
    func decode(_ indices: MLXArray) -> MLXArray {
        // embed(indices) [B,T,64] -> project_out [B,T,hidden] -> NCL.
        projectOut(codebook(indices)).transposed(0, 2, 1)
    }
}

/// Residual vector quantizer encode: 8 codebooks, residual loop.
/// Input `[B, hidden, T]` NCL -> codes `[B, 8, T]` Int32.
final class OVResidualVQEncode: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [OVVectorQuantizerEncode]

    init(numQuantizers: Int, hiddenSize: Int, codebookSize: Int, codebookDim: Int) {
        var qs: [OVVectorQuantizerEncode] = []
        for _ in 0 ..< numQuantizers {
            qs.append(OVVectorQuantizerEncode(
                hiddenSize: hiddenSize, codebookSize: codebookSize, codebookDim: codebookDim))
        }
        self._quantizers.wrappedValue = qs
        super.init()
    }

    /// embeddings: `[B, hidden, T]` NCL -> codes `[B, numQuantizers, T]` Int32.
    func encode(_ embeddings: MLXArray) -> MLXArray {
        var residual = embeddings
        var all: [MLXArray] = []
        for q in quantizers {
            let idx = q.encode(residual)          // [B, T]
            let quant = q.decode(idx)             // [B, hidden, T]
            residual = residual - quant
            all.append(idx)                       // [B, T]
        }
        // stack along a new axis-1 -> [B, numQuantizers, T].
        let stacked = stacked(all, axis: 1)       // [B, Q, T]
        return stacked
    }
}

// MARK: - Encoder

/// Higgs-audio v2 codec **encoder** (24 kHz reference waveform -> 8-codebook
/// audio tokens). Mirrors `HiggsAudioV2TokenizerModel.encode`:
///
/// 1. `_extract_semantic_features`: resample 24k->16k, pad (160,160), run HuBERT
///    with all-hidden-state mean, subsample x2 -> `[B, T, 768]`.
/// 2. `encoder_semantic(.transpose(1,2))` -> `[B, 768, T]`.
/// 3. `acoustic_encoder(F.pad(wav, (480,480)))` (DAC) -> `[B, 256, T]`.
/// 4. `cat([acoustic, semantic], dim=1)` -> `[B, 1024, T]`; `fc` (Linear 1024->1024).
/// 5. RVQ encode -> `[B, 8, T]` Int32.
///
/// Weights come from the same audio-tokenizer `model.safetensors` as the
/// decoder; only the encode-path keys are kept and conv weights are transposed
/// into MLX layout (with HuBERT pos-conv weight-norm folded at load).
public final class OmniVoiceCodecEncoder: Module {
    @ModuleInfo(key: "semantic_model") var semanticModel: HubertSemanticModel
    @ModuleInfo(key: "encoder_semantic") var encoderSemantic: SemanticEncoder
    @ModuleInfo(key: "acoustic_encoder") var acousticEncoder: DacAcousticEncoder
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "quantizer") var quantizer: OVResidualVQEncode

    let sampleRate: Int
    let semanticSampleRate: Int
    let semanticDownsampleFactor: Int
    let semanticPad: Int          // 160
    let acousticPad: Int          // hop_length // 2 = 480
    let downsamplingRatios: [Int]

    public init(
        numQuantizers: Int = 8,
        hiddenSize: Int = 1024,
        codebookSize: Int = 1024,
        codebookDim: Int = 64,
        acousticHidden: Int = 256,
        encoderHidden: Int = 64,
        downsamplingRatios: [Int] = [8, 5, 4, 2, 3],
        sampleRate: Int = 24000,
        semanticSampleRate: Int = 16000,
        semanticDownsampleFactor: Int = 2
    ) {
        self.sampleRate = sampleRate
        self.semanticSampleRate = semanticSampleRate
        self.semanticDownsampleFactor = semanticDownsampleFactor
        self.semanticPad = 160
        self.acousticPad = 480
        self.downsamplingRatios = downsamplingRatios

        self._semanticModel.wrappedValue = HubertSemanticModel()
        self._encoderSemantic.wrappedValue = SemanticEncoder()
        self._acousticEncoder.wrappedValue = DacAcousticEncoder(
            encoderHidden: encoderHidden, outChannels: acousticHidden,
            downsamplingRatios: downsamplingRatios)
        self._fc.wrappedValue = Linear(hiddenSize, hiddenSize)
        self._quantizer.wrappedValue = OVResidualVQEncode(
            numQuantizers: numQuantizers, hiddenSize: hiddenSize,
            codebookSize: codebookSize, codebookDim: codebookDim)
        super.init()
    }

    // MARK: Stage entry points (for golden gating)

    /// HuBERT mean-hidden-state features from a 16 kHz, (160,160)-padded input.
    /// Input: `[1, N16]` or `[1, 1, N16]`. Output: `[1, T, 768]` (subsampled x2).
    public func semanticFeatures16k(_ padded16k: MLXArray) -> MLXArray {
        var x = padded16k
        if x.ndim == 2 { x = x.reshaped([x.dim(0), 1, x.dim(1)]) }  // [B,1,N]
        var feats = semanticModel.meanHiddenStates(x)              // [B, Tfull, 768]
        if semanticDownsampleFactor > 1 {
            feats = feats[0..., .stride(from: 0, by: semanticDownsampleFactor), 0...]
        }
        return feats
    }

    /// encoder_semantic over semantic features `[1, T, 768]` -> `[1, 768, T]`.
    public func encodeSemantic(_ feats: MLXArray) -> MLXArray {
        encoderSemantic(feats.transposed(0, 2, 1))
    }

    /// DAC acoustic encoder over the 24 kHz waveform `[1, 1, N]` (or `[N]`),
    /// padding (480,480) when the conv length differs from the semantic length.
    /// Output: `[1, 256, T]`.
    public func encodeAcoustic(_ wav: MLXArray, semanticT: Int) -> MLXArray {
        var x = wav
        if x.ndim == 1 { x = x.reshaped([1, 1, x.dim(0)]) }
        if x.ndim == 2 { x = x.reshaped([x.dim(0), 1, x.dim(1)]) }
        let n = x.dim(2)
        let convLen = DacAcousticEncoder.outputLength(
            for: n, downsamplingRatios: downsamplingRatios)
        if convLen != semanticT {
            x = padded(x, widths: [.init((0, 0)), .init((0, 0)), .init((acousticPad, acousticPad))])
        }
        return acousticEncoder(x)
    }

    /// cat([acoustic, semantic]) + fc -> `[1, 1024, T]`.
    public func fuseEmbeddings(acoustic: MLXArray, semantic: MLXArray) -> MLXArray {
        let cat = concatenated([acoustic, semantic], axis: 1)  // [B, 1024, T]
        // fc over channel axis: NCL -> NLC -> Linear -> NCL.
        return fc(cat.transposed(0, 2, 1)).transposed(0, 2, 1)
    }

    /// RVQ encode of fused embeddings `[1, 1024, T]` -> codes `[1, 8, T]` Int32.
    public func quantize(_ embeddings: MLXArray) -> MLXArray {
        quantizer.encode(embeddings)
    }

    /// Full encode from a 24 kHz waveform. The resample 24k->16k is done with a
    /// windowed-sinc kernel matching torchaudio's default.
    /// - Parameter wav: `[1, 1, N]`, `[1, N]`, or `[N]` Float32 @ 24 kHz.
    /// - Returns: `[1, 8, T]` Int32 audio codes.
    public func encode(_ wav: MLXArray) -> MLXArray {
        var x = wav
        if x.ndim == 1 { x = x.reshaped([1, 1, x.dim(0)]) }
        if x.ndim == 2 { x = x.reshaped([x.dim(0), 1, x.dim(1)]) }

        // 1. semantic: resample to 16k, pad (160,160), HuBERT mean-hidden, x2 sub.
        let mono = x[0..., 0, 0...]                                  // [B, N]
        let resampled = Resampler.resample(
            mono, from: sampleRate, to: semanticSampleRate)         // [B, N16]
        let padded16 = padded(
            resampled, widths: [.init((0, 0)), .init((semanticPad, semanticPad))])
        let feats = semanticFeatures16k(padded16)                   // [B, T, 768]
        let eSemantic = encodeSemantic(feats)                       // [B, 768, T]

        // 2. acoustic from the 24k wav.
        let eAcoustic = encodeAcoustic(x, semanticT: eSemantic.dim(2))  // [B, 256, T]

        // 3. fuse + quantize.
        let embeddings = fuseEmbeddings(acoustic: eAcoustic, semantic: eSemantic)
        return quantize(embeddings)
    }

    // MARK: Loading

    /// Load the audio-tokenizer `model.safetensors`, keeping only encode-path
    /// keys, folding HuBERT pos-conv weight-norm, and remapping conv weights into
    /// MLX layout. Uses `verify: .all` so the kept key set must match exactly.
    public func loadWeights(from modelSafetensors: URL) throws {
        let raw = try MLX.loadArrays(url: modelSafetensors)
        var weights: [String: MLXArray] = [:]

        // --- Fold HuBERT pos_conv weight-norm first (needs both g and v). ---
        let posG = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0"
        let posV = "semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"
        if let g = raw[posG], let v = raw[posV] {
            // PyTorch weight_norm with dim=2: norm over dims (0,1) per kernel index.
            // weight = g * v / ||v||_{over dims 0,1}. v: [O, I/groups, K], g: [1,1,K].
            let vf = v.asType(.float32)
            let gf = g.asType(.float32)
            let norm = sqrt((vf * vf).sum(axes: [0, 1], keepDims: true))  // [1,1,K]
            let w = gf * vf / norm                                        // [O, I/groups, K]
            // PyTorch Conv1d [O, I/groups, K] -> MLX [O, K, I/groups].
            let key = "semantic_model.encoder.pos_conv_embed.conv.conv.weight"
            weights[key] = w.transposed(0, 2, 1)
        }

        for (k, v) in raw {
            guard shouldKeep(k) else { continue }

            var arr = v.asType(.float32)
            let mapped = remapKey(k)

            // Transpose conv weights into MLX layout where needed.
            if mapped.hasSuffix(".conv.weight"), arr.ndim == 3 {
                // PyTorch Conv1d [O, I, K] -> MLX [O, K, I].
                arr = arr.transposed(0, 2, 1)
            }

            weights[mapped] = arr
        }

        try update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(parameters())
    }

    /// Whether a checkpoint key belongs to the encode path.
    private func shouldKeep(_ k: String) -> Bool {
        if k.hasPrefix("acoustic_encoder.") { return true }
        if k.hasPrefix("encoder_semantic.") { return true }
        if k.hasPrefix("fc.") { return true }
        if k.hasPrefix("semantic_model.") {
            // Drop the weight-norm parametrization (handled separately) and the
            // masked_spec_embed buffer.
            if k.contains(".parametrizations.") { return false }
            if k.contains("masked_spec_embed") { return false }
            return true
        }
        if k.hasPrefix("quantizer.") {
            // Keep project_in / project_out / codebook.embed; drop EMA buffers.
            if k.contains(".project_in.") { return true }
            if k.contains(".project_out.") { return true }
            if k.contains(".codebook.embed") && !k.contains("embed_avg") { return true }
            return false
        }
        return false
    }

    /// Insert the `.conv` level for the NCL conv wrappers so the flat checkpoint
    /// keys line up with `OVConv1dNCL` (`<name>.conv.*`), for both DAC convs and
    /// the HuBERT / semantic convs.
    private func remapKey(_ k: String) -> String {
        guard k.hasSuffix(".weight") || k.hasSuffix(".bias") else { return k }
        let parts = k.split(separator: ".").map(String.init)
        guard let param = parts.last else { return k }
        let prefix = parts.dropLast().joined(separator: ".")
        guard let owner = prefix.split(separator: ".").last.map(String.init) else { return k }

        // Conv-bearing module names whose params live under a `.conv` child in
        // our Swift wrappers. Linear / LayerNorm / GroupNorm keep their flat keys.
        let convOwners: Set<String> = ["conv1", "conv2", "conv"]

        // Exclude Linear "conv"? There are no Linears named conv1/conv2 here.
        // acoustic_encoder.conv1/conv2/block.*.conv1, res_unit*.conv1/conv2,
        // encoder_semantic.conv / conv_blocks.*.conv / res_units.*.conv1/conv2 —
        // all are OVConv1dNCL. feature_extractor.conv_layers.*.conv is also NCL.
        if convOwners.contains(owner) {
            return "\(prefix).conv.\(param)"
        }
        return k
    }
}
