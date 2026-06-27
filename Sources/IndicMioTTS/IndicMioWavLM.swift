import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXNN

public enum IndicMioReferenceConfig {
    public static let defaultWavLMModelId = "aufklarer/WavLM-Base-Plus-MLX-fp16"
    public static let microsoftWavLMModelId = "microsoft/wavlm-base-plus"
    public static let codecSampleRate = 24_000
    public static let wavLMSampleRate = 16_000
    public static let wavLMHopSize = 320
}

final class IndicMioWavLMConvLayer: Module {
    @ModuleInfo var conv: MioConv1dNCL
    @ModuleInfo(key: "layer_norm") var layerNorm: GroupNorm?
    let useGroupNorm: Bool

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int, useGroupNorm: Bool) {
        self._conv.wrappedValue = MioConv1dNCL(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: false)
        self.useGroupNorm = useGroupNorm
        self._layerNorm.wrappedValue = useGroupNorm
            ? GroupNorm(
                groupCount: outputChannels,
                dimensions: outputChannels,
                eps: 1e-5,
                affine: true,
                pytorchCompatible: true)
            : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        if useGroupNorm, let layerNorm {
            h = layerNorm(h.transposed(0, 2, 1)).transposed(0, 2, 1)
        }
        return gelu(h)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyConv1dWeights(
            to: conv.conv,
            prefix: "\(prefix).conv",
            from: weights,
            transpose: true)
        if let layerNorm {
            applyMioGroupNormWeights(to: layerNorm, prefix: "\(prefix).layer_norm", from: weights)
        }
    }
}

final class IndicMioWavLMFeatureEncoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [IndicMioWavLMConvLayer]

    let kernels = [10, 3, 3, 3, 3, 2, 2]
    let strides = [5, 2, 2, 2, 2, 2, 2]

    override init() {
        var layers: [IndicMioWavLMConvLayer] = []
        var inputChannels = 1
        for i in 0..<kernels.count {
            layers.append(IndicMioWavLMConvLayer(
                inputChannels: inputChannels,
                outputChannels: 512,
                kernelSize: kernels[i],
                stride: strides[i],
                useGroupNorm: i == 0))
            inputChannels = 512
        }
        self._convLayers.wrappedValue = layers
        super.init()
    }

    /// Input `[B, samples]`, output `[B, 512, T]`.
    func callAsFunction(_ inputValues: MLXArray) -> MLXArray {
        var h = inputValues.expandedDimensions(axis: 1)
        for layer in convLayers {
            h = layer(h)
        }
        return h
    }

    func getMinimumInputLength(desiredOutputLength: Int) -> Int {
        var length = desiredOutputLength
        for (kernel, stride) in zip(kernels, strides).reversed() {
            length = (length - 1) * stride + kernel
        }
        return length
    }

    func loadWeights(from weights: [String: MLXArray]) {
        for i in convLayers.indices {
            convLayers[i].loadWeights(prefix: "feature_extractor.conv_layers.\(i)", from: weights)
        }
    }
}

final class IndicMioWavLMFeatureProjection: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var projection: Linear

    override init() {
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1e-5)
        self._projection.wrappedValue = Linear(512, 768, bias: true)
        super.init()
    }

    /// Input `[B, T, 512]`, output `[B, T, 768]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        projection(layerNorm(x))
    }

    func loadWeights(from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLayerNormWeights(to: layerNorm, prefix: "feature_projection.layer_norm", from: weights)
        CommonWeightLoader.applyLinearWeights(to: projection, prefix: "feature_projection.projection", from: weights)
    }
}

final class IndicMioWavLMPositionConv: Module {
    @ModuleInfo var conv: MioConv1dNCL

    override init() {
        self._conv.wrappedValue = MioConv1dNCL(
            inputChannels: 768,
            outputChannels: 768,
            kernelSize: 128,
            padding: 64,
            groups: 16)
        super.init()
    }

    /// Input `[B, T, 768]`, output `[B, T, 768]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let length = x.dim(1)
        var h = conv(x.transposed(0, 2, 1)).transposed(0, 2, 1)
        h = h[0..., 0..<length, 0...]
        return gelu(h)
    }

    func loadWeights(from weights: [String: MLXArray]) {
        var params: [String: NestedItem<String, MLXArray>] = [:]
        if let g = weights["encoder.pos_conv_embed.conv.weight_g"],
           let v = weights["encoder.pos_conv_embed.conv.weight_v"] {
            let vf = v.asType(.float32)
            let gf = g.asType(.float32)
            let norm = MLX.sqrt((vf * vf).sum(axes: [0, 1], keepDims: true))
            params["weight"] = .value((gf * vf / norm).transposed(0, 2, 1))
        } else if let weight = weights["encoder.pos_conv_embed.conv.weight"] {
            params["weight"] = .value(weight.transposed(0, 2, 1))
        }
        if let bias = weights["encoder.pos_conv_embed.conv.bias"] {
            params["bias"] = .value(bias)
        }
        if !params.isEmpty {
            conv.conv.update(parameters: ModuleParameters(values: params))
        }
    }
}

final class IndicMioWavLMAttention: Module {
    let numHeads = 12
    let headDim = 64
    let scale: Float = 1.0 / Foundation.sqrt(Float(64))
    let numBuckets = 320
    let maxDistance = 800

    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "gru_rel_pos_linear") var gruRelPosLinear: Linear
    @ModuleInfo(key: "rel_attn_embed") var relAttnEmbed: Embedding?
    @ParameterInfo(key: "gru_rel_pos_const") var gruRelPosConst: MLXArray

    init(hasRelativePositionBias: Bool) {
        self._kProj.wrappedValue = Linear(768, 768, bias: true)
        self._vProj.wrappedValue = Linear(768, 768, bias: true)
        self._qProj.wrappedValue = Linear(768, 768, bias: true)
        self._outProj.wrappedValue = Linear(768, 768, bias: true)
        self._gruRelPosLinear.wrappedValue = Linear(headDim, 8, bias: true)
        self._relAttnEmbed.wrappedValue = hasRelativePositionBias
            ? Embedding(embeddingCount: numBuckets, dimensions: numHeads)
            : nil
        self._gruRelPosConst = ParameterInfo(wrappedValue: MLXArray.ones([1, numHeads, 1, 1]))
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, positionBias: MLXArray?) -> (MLXArray, MLXArray) {
        let batch = hiddenStates.dim(0)
        let targetLength = hiddenStates.dim(1)

        let basePositionBias = positionBias ?? computeBias(queryLength: targetLength, keyLength: targetLength)

        let gatedHidden = hiddenStates
            .reshaped(batch, targetLength, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let relativePositionProj = gruRelPosLinear(gatedHidden)
            .reshaped(batch, numHeads, targetLength, 2, 4)
            .sum(axis: -1)
        let gateA = sigmoid(relativePositionProj[0..., 0..., 0..., 0]).expandedDimensions(axis: -1)
        let gateB = sigmoid(relativePositionProj[0..., 0..., 0..., 1]).expandedDimensions(axis: -1)
        let gateOutput = gateA * (gateB * gruRelPosConst - 1.0) + 2.0
        let gatedPositionBias = gateOutput * basePositionBias

        let q = qProj(hiddenStates)
            .reshaped(batch, targetLength, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let k = kProj(hiddenStates)
            .reshaped(batch, targetLength, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let v = vProj(hiddenStates)
            .reshaped(batch, targetLength, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let attended = SDPA.attendAndMerge(
            qHeads: q,
            kHeads: k,
            vHeads: v,
            scale: scale,
            mask: gatedPositionBias.asType(q.dtype))
        return (outProj(attended), basePositionBias)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(to: kProj, prefix: "\(prefix).k_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: vProj, prefix: "\(prefix).v_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: qProj, prefix: "\(prefix).q_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(to: outProj, prefix: "\(prefix).out_proj", from: weights)
        CommonWeightLoader.applyLinearWeights(
            to: gruRelPosLinear,
            prefix: "\(prefix).gru_rel_pos_linear",
            from: weights)
        if let relAttnEmbed {
            CommonWeightLoader.applyEmbeddingWeights(
                to: relAttnEmbed,
                prefix: "\(prefix).rel_attn_embed",
                from: weights)
        }
        if let value = weights["\(prefix).gru_rel_pos_const"] {
            update(parameters: ModuleParameters(values: ["gru_rel_pos_const": .value(value)]))
        }
    }

    private func computeBias(queryLength: Int, keyLength: Int) -> MLXArray {
        guard let relAttnEmbed else {
            fatalError("First WavLM layer must own rel_attn_embed")
        }
        var buckets = [Int32]()
        buckets.reserveCapacity(queryLength * keyLength)
        for query in 0..<queryLength {
            for key in 0..<keyLength {
                buckets.append(Int32(relativePositionBucket(key - query)))
            }
        }
        let bucketIds = MLXArray(buckets).reshaped(queryLength, keyLength)
        let values = relAttnEmbed(bucketIds)
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)
    }

    private func relativePositionBucket(_ relativePosition: Int) -> Int {
        let halfBuckets = numBuckets / 2
        var bucket = relativePosition > 0 ? halfBuckets : 0
        let n = abs(relativePosition)
        let maxExact = halfBuckets / 2
        if n < maxExact {
            bucket += n
        } else {
            let logRatio = Foundation.log(Double(max(n, 1)) / Double(maxExact))
                / Foundation.log(Double(maxDistance) / Double(maxExact))
            let large = min(halfBuckets - 1, maxExact + Int(logRatio * Double(halfBuckets - maxExact)))
            bucket += large
        }
        return bucket
    }
}

final class IndicMioWavLMFeedForward: Module {
    @ModuleInfo(key: "intermediate_dense") var intermediateDense: Linear
    @ModuleInfo(key: "output_dense") var outputDense: Linear

    override init() {
        self._intermediateDense.wrappedValue = Linear(768, 3_072, bias: true)
        self._outputDense.wrappedValue = Linear(3_072, 768, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputDense(gelu(intermediateDense(x)))
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        CommonWeightLoader.applyLinearWeights(
            to: intermediateDense,
            prefix: "\(prefix).intermediate_dense",
            from: weights)
        CommonWeightLoader.applyLinearWeights(to: outputDense, prefix: "\(prefix).output_dense", from: weights)
    }
}

final class IndicMioWavLMEncoderLayer: Module {
    @ModuleInfo var attention: IndicMioWavLMAttention
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: IndicMioWavLMFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(hasRelativePositionBias: Bool) {
        self._attention.wrappedValue = IndicMioWavLMAttention(
            hasRelativePositionBias: hasRelativePositionBias)
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        self._feedForward.wrappedValue = IndicMioWavLMFeedForward()
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray?) -> (MLXArray, MLXArray) {
        let (attended, nextBias) = attention(x, positionBias: positionBias)
        var h = layerNorm(x + attended)
        h = finalLayerNorm(h + feedForward(h))
        return (h, nextBias)
    }

    func loadWeights(prefix: String, from weights: [String: MLXArray]) {
        attention.loadWeights(prefix: "\(prefix).attention", from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: layerNorm, prefix: "\(prefix).layer_norm", from: weights)
        feedForward.loadWeights(prefix: "\(prefix).feed_forward", from: weights)
        CommonWeightLoader.applyLayerNormWeights(
            to: finalLayerNorm,
            prefix: "\(prefix).final_layer_norm",
            from: weights)
    }
}

public final class IndicMioWavLMFeatureModel: Module {
    public static let defaultModelId = IndicMioReferenceConfig.defaultWavLMModelId

    @ModuleInfo(key: "feature_extractor") var featureExtractor: IndicMioWavLMFeatureEncoder
    @ModuleInfo(key: "feature_projection") var featureProjection: IndicMioWavLMFeatureProjection
    @ModuleInfo(key: "pos_conv_embed") var posConvEmbed: IndicMioWavLMPositionConv
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo var layers: [IndicMioWavLMEncoderLayer]

    public override init() {
        self._featureExtractor.wrappedValue = IndicMioWavLMFeatureEncoder()
        self._featureProjection.wrappedValue = IndicMioWavLMFeatureProjection()
        self._posConvEmbed.wrappedValue = IndicMioWavLMPositionConv()
        self._layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        self._layers.wrappedValue = [
            IndicMioWavLMEncoderLayer(hasRelativePositionBias: true),
            IndicMioWavLMEncoderLayer(hasRelativePositionBias: false),
        ]
        super.init()
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> IndicMioWavLMFeatureModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: ["config.json", "model.safetensors"],
            offlineMode: offlineMode,
            progressHandler: { progressHandler?($0, "Downloading WavLM reference encoder") })
        return try fromBundle(directory)
    }

    public static func fromBundle(_ directory: URL) throws -> IndicMioWavLMFeatureModel {
        let weights = try CommonWeightLoader.loadAllSafetensors(from: directory)
        let model = IndicMioWavLMFeatureModel()
        model.loadWeights(from: weights)
        return model
    }

    public func averagedGlobalFeatures(audio16k: [Float]) -> MLXArray {
        let waveform = MLXArray(audio16k).expandedDimensions(axis: 0)
        var features = featureExtractor(waveform)
        features = features.transposed(0, 2, 1)
        var hidden = featureProjection(features)
        let pos = posConvEmbed(hidden)
        hidden = layerNorm(hidden + pos)

        var positionBias: MLXArray?
        var selected: [MLXArray] = []
        selected.reserveCapacity(layers.count)
        for layer in layers {
            let (next, bias) = layer(hidden, positionBias: positionBias)
            hidden = next
            positionBias = bias
            selected.append(hidden)
        }
        return (selected[0] + selected[1]) / 2.0
    }

    public func minimumWavLMInputLength(forFeatureFrames frames: Int) -> Int {
        featureExtractor.getMinimumInputLength(desiredOutputLength: frames)
    }

    func loadWeights(from weights: [String: MLXArray]) {
        featureExtractor.loadWeights(from: weights)
        featureProjection.loadWeights(from: weights)
        posConvEmbed.loadWeights(from: weights)
        CommonWeightLoader.applyLayerNormWeights(to: layerNorm, prefix: "encoder.layer_norm", from: weights)
        for i in layers.indices {
            layers[i].loadWeights(prefix: "encoder.layers.\(i)", from: weights)
        }
        eval(self)
    }
}

private func applyMioGroupNormWeights(
    to groupNorm: GroupNorm,
    prefix: String,
    from weights: [String: MLXArray]
) {
    var params: [String: NestedItem<String, MLXArray>] = [:]
    if let weight = weights["\(prefix).weight"] {
        params["weight"] = .value(weight)
    }
    if let bias = weights["\(prefix).bias"] {
        params["bias"] = .value(bias)
    }
    if !params.isEmpty {
        groupNorm.update(parameters: ModuleParameters(values: params))
    }
}
