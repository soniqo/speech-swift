import Foundation
import MLX
import MLXNN

// MARK: - FSQ inverse

public let MagpieFSQGroups = 8
public let MagpieFSQLevels: [Int32] = [8, 7, 6, 6]
public let MagpieFSQDimPerGroup = 4
public let MagpieFSQTotalDim = MagpieFSQGroups * MagpieFSQDimPerGroup  // 32

/// `(B, T, 8) int32` → `(B, T, 32) float32`.
///
/// For each codebook value `i ∈ [0, 2015]`:
///   d_j      = (i // base[j]) % level[j]    for j ∈ 0..3
///   dequant  = (d_j - level[j]//2) / (level[j]//2)
/// where ``base = cumprod([1, 8, 7, 6]) = [1, 8, 56, 336]``.
public func magpieFSQDecode(_ indices: MLXArray) -> MLXArray {
    let base   = MLXArray([Int32(1), 8, 56, 336])
    let levels = MLXArray(MagpieFSQLevels)
    // PyTorch / NeMo reference uses integer floor division and Python modulo
    // (`(i // base) % level`). MLX-swift's `/` is true division (returns
    // float), so we use `floorDivide` to keep integer semantics — otherwise
    // every FSQ slot decodes to fractional offsets and the codec produces
    // smeared audio.
    let expanded  = MLX.floorDivide(indices.expandedDimensions(axis: -1), base)
    let nonneg    = expanded % levels
    let halfLevel = MLX.floorDivide(levels, MLXArray(Int32(2))).asType(.float32)
    let dequant   = (nonneg.asType(.float32) - halfLevel) / halfLevel
    let B = dequant.dim(0)
    let T = dequant.dim(1)
    return dequant.reshaped([B, T, MagpieFSQTotalDim])
}

// MARK: - Snake / HalfSnake activations

public final class MagpieSnake: Module {
    @ParameterInfo public var alpha: MLXArray
    public init(channels: Int) {
        // NeMo stores α as (1, C, 1) for NCL; for our NLC layout it's (1, 1, C).
        self._alpha = ParameterInfo(wrappedValue: MLXArray.ones([1, 1, channels]))
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let eps = MLXArray(Float(1e-9))
        let s = MLX.sin(alpha * x)
        return x + (MLXArray(Float(1.0)) / (alpha + eps)) * s * s
    }
}

public final class MagpieHalfSnake: Module {
    @ModuleInfo(key: "snake_act") public var snakeAct: MagpieSnake
    public let snakeChannels: Int
    public init(channels: Int) {
        self.snakeChannels = channels / 2
        self._snakeAct = ModuleInfo(
            wrappedValue: MagpieSnake(channels: snakeChannels),
            key: "snake_act")
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let C = x.dim(-1)
        let half = snakeChannels
        let sPart = snakeAct(x[.ellipsis, 0..<half])
        let lRaw  = x[.ellipsis, half..<C]
        // LeakyReLU(0.01)
        let lPart = MLX.where(lRaw .>= MLXArray(Float(0)),
                              lRaw, MLXArray(Float(0.01)) * lRaw)
        return MLX.concatenated([sPart, lPart], axis: -1)
    }
}

// MARK: - Weight-norm-merged conv layers (NLC)

/// 1-D causal convolution. Weight stored as `(out_channels, kernel_size, in_channels)`.
public final class MagpieCausalConv1dWN: Module {
    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var bias: MLXArray
    public let leftPad: Int
    public let dilation: Int
    public let groups: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                dilation: Int = 1, groups: Int = 1) {
        let effK = (kernelSize - 1) * dilation + 1
        self.leftPad = effK - 1
        self.dilation = dilation
        self.groups = groups
        self._weight = ParameterInfo(wrappedValue:
            MLXArray.zeros([outChannels, kernelSize, inChannels / groups]))
        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([outChannels]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if leftPad > 0 {
            h = MLX.padded(h, widths: [.init((0, 0)), .init((leftPad, 0)), .init((0, 0))],
                            value: MLXArray(Float(0)))
        }
        let y = MLX.conv1d(h, weight, stride: 1, padding: 0,
                            dilation: dilation, groups: groups)
        return y + bias
    }
}

/// 1-D causal transposed convolution (groups = out_channels). Right-trimmed
/// to match PyTorch's causal padding convention.
public final class MagpieCausalConvTranspose1dWN: Module {
    @ParameterInfo public var weight: MLXArray
    @ParameterInfo public var bias: MLXArray
    public let stride: Int
    public let groups: Int
    public let trimRight: Int

    public init(inChannels: Int, outChannels: Int, kernelSize: Int,
                stride: Int, groups: Int) {
        self.stride = stride
        self.groups = groups
        self.trimRight = Int((Double(kernelSize - stride)).rounded(.up))
        // MLX convTransposed1d weight layout: (out_channels, kernel, in/groups).
        // For depthwise codec layers groups == out_channels, in/groups == 2.
        self._weight = ParameterInfo(wrappedValue:
            MLXArray.zeros([outChannels, kernelSize, inChannels / groups]))
        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([outChannels]))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.convTransposed1d(
            x, weight, stride: stride, padding: 0,
            dilation: 1, outputPadding: 0, groups: groups)
        if trimRight > 0 {
            y = y[0..., 0..<(y.dim(1) - trimRight), 0...]
        }
        return y + bias
    }
}

// MARK: - Residual + HiFi-GAN stacks

public final class MagpieResidualBlock: Module {
    @ModuleInfo(key: "input_activation") public var inputActivation: MagpieHalfSnake
    @ModuleInfo(key: "input_conv")       public var inputConv: MagpieCausalConv1dWN
    @ModuleInfo(key: "skip_activation")  public var skipActivation: MagpieHalfSnake
    @ModuleInfo(key: "skip_conv")        public var skipConv: MagpieCausalConv1dWN

    public init(channels: Int, kernelSize: Int, dilation: Int) {
        self._inputActivation = ModuleInfo(
            wrappedValue: MagpieHalfSnake(channels: channels),
            key: "input_activation")
        self._inputConv = ModuleInfo(
            wrappedValue: MagpieCausalConv1dWN(inChannels: channels, outChannels: channels,
                                                kernelSize: kernelSize, dilation: dilation),
            key: "input_conv")
        self._skipActivation = ModuleInfo(
            wrappedValue: MagpieHalfSnake(channels: channels),
            key: "skip_activation")
        self._skipConv = ModuleInfo(
            wrappedValue: MagpieCausalConv1dWN(inChannels: channels, outChannels: channels,
                                                kernelSize: kernelSize, dilation: 1),
            key: "skip_conv")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = inputActivation(x)
        h = inputConv(h)
        h = skipActivation(h)
        h = skipConv(h)
        return x + h
    }
}

public final class MagpieHiFiGANResBlock: Module {
    @ModuleInfo(key: "res_blocks") public var resBlocks: [MagpieResidualBlock]
    public init(channels: Int, kernelSize: Int, dilations: [Int]) {
        self._resBlocks = ModuleInfo(
            wrappedValue: dilations.map {
                MagpieResidualBlock(channels: channels, kernelSize: kernelSize, dilation: $0)
            },
            key: "res_blocks")
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for rb in resBlocks { h = rb(h) }
        return h
    }
}

public final class MagpieHiFiGANResLayer: Module {
    @ModuleInfo(key: "res_blocks") public var resBlocks: [MagpieHiFiGANResBlock]
    public init(channels: Int, kernelSizes: [Int], dilations: [Int]) {
        self._resBlocks = ModuleInfo(
            wrappedValue: kernelSizes.map {
                MagpieHiFiGANResBlock(channels: channels, kernelSize: $0, dilations: dilations)
            },
            key: "res_blocks")
        super.init()
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var s = resBlocks[0](x)
        for i in 1..<resBlocks.count {
            s = s + resBlocks[i](x)
        }
        return s / MLXArray(Float(resBlocks.count))
    }
}

// MARK: - Causal HiFi-GAN decoder

public final class MagpieCausalHiFiGANDecoder: Module {
    @ModuleInfo(key: "pre_conv")            public var preConv: MagpieCausalConv1dWN
    @ModuleInfo(key: "activations")          public var activations: [MagpieHalfSnake]
    @ModuleInfo(key: "up_sample_conv_layers") public var upSampleConvLayers: [MagpieCausalConvTranspose1dWN]
    @ModuleInfo(key: "res_layers")           public var resLayers: [MagpieHiFiGANResLayer]
    @ModuleInfo(key: "post_activation")      public var postActivation: MagpieHalfSnake
    @ModuleInfo(key: "post_conv")            public var postConv: MagpieCausalConv1dWN

    public let upSampleRates: [Int]

    public init(inputDim: Int = MagpieFSQTotalDim,
                baseChannels: Int = 864,
                inKernelSize: Int = 7,
                outKernelSize: Int = 3,
                upSampleRates: [Int] = [8, 8, 4, 2, 2],
                resblockKernelSizes: [Int] = [3, 7, 11],
                resblockDilationSizes: [Int] = [1, 3, 5]) {
        self.upSampleRates = upSampleRates
        self._preConv = ModuleInfo(
            wrappedValue: MagpieCausalConv1dWN(inChannels: inputDim, outChannels: baseChannels,
                                                kernelSize: inKernelSize),
            key: "pre_conv")

        var inCh = baseChannels
        var actsList: [MagpieHalfSnake] = []
        var upList: [MagpieCausalConvTranspose1dWN] = []
        var resList: [MagpieHiFiGANResLayer] = []
        for up in upSampleRates {
            let outCh = inCh / 2
            let k = 2 * up
            actsList.append(MagpieHalfSnake(channels: inCh))
            upList.append(MagpieCausalConvTranspose1dWN(
                inChannels: inCh, outChannels: outCh,
                kernelSize: k, stride: up, groups: outCh))
            resList.append(MagpieHiFiGANResLayer(
                channels: outCh,
                kernelSizes: resblockKernelSizes,
                dilations: resblockDilationSizes))
            inCh = outCh
        }
        self._activations = ModuleInfo(wrappedValue: actsList, key: "activations")
        self._upSampleConvLayers = ModuleInfo(wrappedValue: upList, key: "up_sample_conv_layers")
        self._resLayers = ModuleInfo(wrappedValue: resList, key: "res_layers")

        self._postActivation = ModuleInfo(
            wrappedValue: MagpieHalfSnake(channels: inCh),
            key: "post_activation")
        self._postConv = ModuleInfo(
            wrappedValue: MagpieCausalConv1dWN(inChannels: inCh, outChannels: 1,
                                                kernelSize: outKernelSize),
            key: "post_conv")
        super.init()
    }

    /// `latents: (B, T, 32)` → `(B, T_audio)`.
    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var x = preConv(latents)
        for i in 0..<activations.count {
            x = activations[i](x)
            x = upSampleConvLayers[i](x)
            x = resLayers[i](x)
        }
        x = postActivation(x)
        x = postConv(x)
        x = MLX.clip(x, min: MLXArray(Float(-1.0)), max: MLXArray(Float(1.0)))
        return x[.ellipsis, 0]
    }
}

/// Bundle 4: indices `(B, T, 8)` → 22.05 kHz audio `(B, T_audio)`.
public final class MagpieNanoCodec: Module {
    @ModuleInfo public var decoder: MagpieCausalHiFiGANDecoder

    public override init() {
        self._decoder = ModuleInfo(wrappedValue: MagpieCausalHiFiGANDecoder())
        super.init()
    }

    public func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let latents = magpieFSQDecode(indices)
        return decoder(latents)
    }
}
