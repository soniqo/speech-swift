import Foundation
import MLX
import MLXNN
import MLXFast
import MLXCommon
import AudioCommon

// MARK: - Encoder Residual Unit

/// Mirror of DecoderResidualUnit.
/// Dilated residual: SnakeBeta -> CausalConv1d(dilated) -> SnakeBeta -> CausalConv1d(1x1) -> residual
public class EncoderResidualUnit: Module {
    @ModuleInfo var snake1: SnakeBeta
    @ModuleInfo var conv1: CausalConv1d
    @ModuleInfo var snake2: SnakeBeta
    @ModuleInfo var conv2: CausalConv1d

    public init(dim: Int, dilation: Int) {
        self._snake1.wrappedValue = SnakeBeta(channels: dim)
        self._conv1.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 7, dilation: dilation)
        self._snake2.wrappedValue = SnakeBeta(channels: dim)
        self._conv2.wrappedValue = CausalConv1d(
            inputChannels: dim, outputChannels: dim,
            kernelSize: 1)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = snake1(x)
        h = conv1(h)
        h = snake2(h)
        h = conv2(h)
        return h + residual
    }
}

// MARK: - Encoder Block (Downsample)

/// Mirror of DecoderBlock, reversed.
/// 3x EncoderResidualUnit -> SnakeBeta -> CausalConv1d(strided, downsample)
public class EncoderBlock: Module {
    @ModuleInfo var residualUnits: [EncoderResidualUnit]
    @ModuleInfo var snake: SnakeBeta
    @ModuleInfo var downsample: CausalConv1d

    public init(inputDim: Int, outputDim: Int, stride: Int) {
        // 3 residual units with dilations [1, 3, 9]
        self._residualUnits.wrappedValue = [1, 3, 9].map { dilation in
            EncoderResidualUnit(dim: inputDim, dilation: dilation)
        }
        self._snake.wrappedValue = SnakeBeta(channels: inputDim)
        self._downsample.wrappedValue = CausalConv1d(
            inputChannels: inputDim, outputChannels: outputDim,
            kernelSize: stride * 2, stride: stride)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for unit in residualUnits {
            h = unit(h)
        }
        h = snake(h)
        return downsample(h)
    }
}

// MARK: - Encoder Transformer

/// Mirror of DecoderTransformer — 8-layer transformer with 1024→512 bottleneck.
/// Reuses DecoderTransformerLayer (same attention + SwiGLU MLP + LayerScale).
public class EncoderTransformer: Module {
    @ModuleInfo var inputProj: Linear
    @ModuleInfo var layers: [DecoderTransformerLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var outputProj: Linear

    public init(config: SpeechTokenizerDecoderConfig) {
        self._inputProj.wrappedValue = Linear(config.latentDim, config.hiddenSize)
        self._layers.wrappedValue = (0..<config.numLayers).map { _ in
            DecoderTransformerLayer(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize)
        self._outputProj.wrappedValue = Linear(config.hiddenSize, config.latentDim)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputProj(x)  // [B, T, 1024] → [B, T, 512]
        for layer in layers {
            (h, _) = layer(h)  // Discard cache (encoder doesn't need it)
        }
        h = norm(h)
        // Skip outputProj — RVQ encodes in hiddenSize=512 space
        return h  // [B, T, 512]
    }
}

// MARK: - Encoder RVQ (quantize direction)

/// Quantizes a latent [B, T, hiddenSize] tensor into 16 codebook indices.
/// Shares codebook weights with SplitResidualVectorQuantizer (decoder-side).
public class EncoderRVQ: Module {
    @ModuleInfo var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo var rvqRest: ResidualVectorQuantizer

    public init(config: SpeechTokenizerDecoderConfig) {
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: 1,
            codebookSize: config.semanticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            numQuantizers: config.numQuantizers - 1,
            codebookSize: config.acousticCodebookSize,
            codebookDim: config.codebookDim,
            outputDim: config.hiddenSize)

        super.init()
    }

    /// - Parameter h: [B, T, hiddenSize=512]
    /// - Returns: [B, 16, T] — all codebook indices
    public func encode(_ h: MLXArray) -> MLXArray {
        let firstCodes = rvqFirst.encode(h)    // [B, 1, T]
        let restCodes  = rvqRest.encode(h)     // [B, 15, T]
        return concatenated([firstCodes, restCodes], axis: 1)
    }
}

// MARK: - Full Speech Tokenizer Encoder

/// Mimi-based speech tokenizer encoder.
/// Converts 24 kHz audio waveform → 16-codebook indices at 12.5 Hz.
///
/// Architecture (exact mirror of SpeechTokenizerDecoder, reversed):
///   1. inputConv:        1 → encoderDim (96)
///   2. encoderBlocks:    [3x, 4x, 5x, 8x] downsample (channels double each time)
///   3. postConvNeXt1/2:  latentDim ConvNeXt + 2x downsample (each)
///   4. postConv:         latentDim (1024) → hiddenSize (512)
///   5. transformer:      8-layer EncoderTransformer
///   6. rvq:              EncoderRVQ → [B, 16, T_frames]
public class SpeechTokenizerEncoder: Module {
    public let config: SpeechTokenizerDecoderConfig

    @ModuleInfo var inputConv: CausalConv1d
    @ModuleInfo var encoderBlocks: [EncoderBlock]
    @ModuleInfo var channelProj: CausalConv1d  // decoderDim → latentDim
    @ModuleInfo var postConvNeXt1: ConvNeXtBlock
    @ModuleInfo var postDownsample1: CausalConv1d
    @ModuleInfo var postConvNeXt2: ConvNeXtBlock
    @ModuleInfo var postDownsample2: CausalConv1d
    @ModuleInfo var postConv: CausalConv1d
    @ModuleInfo var transformer: EncoderTransformer
    @ModuleInfo var rvq: EncoderRVQ

    public init(config: SpeechTokenizerDecoderConfig) {
        self.config = config

        // Mirror decoder channel schedule (reversed).
        // Decoder dims: [1536, 768, 384, 192, 96]
        // Encoder dims: [96, 192, 384, 768, 1536]
        var decoderDims: [Int] = [config.decoderDim]
        for _ in config.upsampleRates {
            decoderDims.append(decoderDims.last! / 2)
        }
        let encoderDims = decoderDims.reversed() as [Int]

        // 1. Input conv: 1 -> encoderDims[0] (=96), kernel=7 (mirrors decoder finalConv)
        self._inputConv.wrappedValue = CausalConv1d(
            inputChannels: 1, outputChannels: encoderDims[0],
            kernelSize: 7)

        // 2. Encoder blocks: downsample rates = decoder upsampleRates reversed [8,5,4,3] → [3,4,5,8]
        let downsampleRates = config.upsampleRates.reversed() as [Int]
        self._encoderBlocks.wrappedValue = downsampleRates.enumerated().map { i, rate in
            EncoderBlock(inputDim: encoderDims[i], outputDim: encoderDims[i + 1], stride: rate)
        }

        // 3. Channel projection: decoderDim (1536) → latentDim (1024)
        // Mirrors the decoder's transition from latentDim to decoderDim
        self._channelProj.wrappedValue = CausalConv1d(
            inputChannels: config.decoderDim, outputChannels: config.latentDim,
            kernelSize: 7)

        // 4. Post-downsample (mirror of preUpsample in decoder, reversed)
        let latent = config.latentDim
        self._postConvNeXt1.wrappedValue = ConvNeXtBlock(dim: latent)
        self._postDownsample1.wrappedValue = CausalConv1d(
            inputChannels: latent, outputChannels: latent,
            kernelSize: config.upsamplingRatios[1] * 2,
            stride: config.upsamplingRatios[1])
        self._postConvNeXt2.wrappedValue = ConvNeXtBlock(dim: latent)
        self._postDownsample2.wrappedValue = CausalConv1d(
            inputChannels: latent, outputChannels: latent,
            kernelSize: config.upsamplingRatios[0] * 2,
            stride: config.upsamplingRatios[0])

        // 4. Post-conv: latentDim (1024) -> latentDim (1024), kernel=3
        // Transformer's inputProj handles latentDim→hiddenSize projection
        self._postConv.wrappedValue = CausalConv1d(
            inputChannels: config.latentDim, outputChannels: config.latentDim,
            kernelSize: 3)

        // 5. Transformer
        self._transformer.wrappedValue = EncoderTransformer(config: config)

        // 6. RVQ
        self._rvq.wrappedValue = EncoderRVQ(config: config)

        super.init()
    }

    /// Encode audio waveform to codec token indices.
    /// - Parameter audio: [B, T_samples, 1] at 24 kHz
    /// - Returns: [B, 16, T_frames] at 12.5 Hz
    public func callAsFunction(_ audio: MLXArray) -> MLXArray {
        var h = inputConv(audio)

        for block in encoderBlocks {
            h = block(h)
        }

        // Channel projection: 1536 → 1024
        h = channelProj(h)

        h = postConvNeXt1(h)
        h = postDownsample1(h)
        h = postConvNeXt2(h)
        h = postDownsample2(h)
        h = postConv(h)
        h = transformer(h)

        return rvq.encode(h)
    }

    /// Encode a [Float] mono PCM buffer (24 kHz) to codec indices.
    /// - Parameter samples: Raw 24 kHz mono samples
    /// - Returns: [1, 16, T_frames]
    public func encode(samples: [Float]) -> MLXArray {
        let audio = MLXArray(samples).reshaped([1, samples.count, 1])
        return callAsFunction(audio)
    }
}
