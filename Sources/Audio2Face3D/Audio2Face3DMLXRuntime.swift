import Foundation
import MLX

final class Audio2Face3DMLXRuntime {
    private let configuration: Audio2Face3DConfiguration
    private let weights: [String: MLXArray]
    private let defaultEmotion: [Float]
    private let graphTensors: GraphTensors
    private let transformerCombineWeights: [Float]

    private static let sqrt2Scalar = Float(1.4142135623730951)
    private static let halfScalar = Float(0.5)
    private static let oneScalar = Float(1.0)
    private static let epsScalar = Float(1e-5)
    private static let autocorrScaleScalar = Float(1.0 / 640.0)

    private static let autocorrWindow: [Float] = {
        let count = 640
        return (0..<count).map { index in
            Float(0.5 - 0.5 * Foundation.cos(2.0 * Double.pi * Double(index) / Double(count)))
        }
    }()

    init(directory: URL, configuration: Audio2Face3DConfiguration) throws {
        let weightsURL = directory.appendingPathComponent("audio2face3d.safetensors")
        let loadedWeights = try MLX.loadArrays(url: weightsURL)
        let resolvedGraphTensors = try Self.resolveGraphTensors(weights: loadedWeights)
        self.configuration = configuration
        self.weights = loadedWeights
        self.graphTensors = resolvedGraphTensors
        self.transformerCombineWeights = loadedWeights["combine_weights"]?.asArray(Float.self) ?? []
        self.defaultEmotion = Self.readDefaultEmotion(
            from: directory,
            expectedCount: configuration.emotionVectorLength
        ) ?? [Float](repeating: 0, count: configuration.emotionVectorLength)
        try validateRequiredTensors()
    }

    func frames(
        for audio: [Float],
        sampleRate: Int,
        hopLength: Int? = nil,
        emotion: [Float]? = nil
    ) throws -> [Audio2Face3DFrame] {
        guard sampleRate == configuration.inputSampleRate else {
            throw Audio2Face3DError.invalidSampleRate(sampleRate)
        }

        let hop = hopLength ?? configuration.frameSampleCount
        guard hop > 0 else {
            throw Audio2Face3DError.invalidAudioWindow(expected: 1, got: hop)
        }
        var frames: [Audio2Face3DFrame] = []
        var target = 0
        repeat {
            let start = target - configuration.hopLength
            var window = [Float](repeating: 0, count: configuration.bufferLength)
            for index in 0..<configuration.bufferLength {
                let sampleIndex = start + index
                if sampleIndex >= 0 && sampleIndex < audio.count {
                    window[index] = audio[sampleIndex]
                }
            }
            let coefficients = try coefficients(forWindow: window, emotion: emotion)
            frames.append(Audio2Face3DFrame(
                timeSeconds: Double(target) / Double(configuration.inputSampleRate),
                coefficients: coefficients,
                layout: configuration.coefficientLayout))
            target += hop
        } while target < max(audio.count, 1)

        return frames
    }

    func emotionVector(explicit: [Float]) throws -> [Float] {
        guard explicit.count == configuration.explicitEmotionCount else {
            throw Audio2Face3DError.invalidEmotionVector(
                expected: configuration.explicitEmotionCount,
                got: explicit.count)
        }
        var vector = defaultEmotion
        let start = configuration.implicitEmotionCount
        for index in 0..<configuration.explicitEmotionCount {
            vector[start + index] = explicit[index]
        }
        return vector
    }

    func coefficients(forWindow audioWindow: [Float], emotion: [Float]? = nil) throws -> [Float] {
        guard audioWindow.count == configuration.bufferLength else {
            throw Audio2Face3DError.invalidAudioWindow(
                expected: configuration.bufferLength,
                got: audioWindow.count)
        }

        let emotionVector = emotion ?? defaultEmotion
        guard emotionVector.count == configuration.emotionVectorLength else {
            throw Audio2Face3DError.invalidEmotionVector(
                expected: configuration.emotionVectorLength,
                got: emotionVector.count)
        }

        let scaledWindow = audioWindow.map { $0 * configuration.inputStrength }
        let input = MLXArray(scaledWindow, [1, 1, configuration.bufferLength])
        let emotionInput = MLXArray(emotionVector, [1, 1, configuration.emotionVectorLength])
        let output = forward(input: input, emotion: emotionInput)
        let flat = output.asType(.float32).reshaped([output.size]).asArray(Float.self)
        guard flat.count == configuration.outputCoefficientCount else {
            throw Audio2Face3DError.invalidModelOutput(
                expected: configuration.outputCoefficientCount,
                got: flat.count)
        }
        return flat
    }

    private func forward(input: MLXArray, emotion: MLXArray) -> MLXArray {
        let implicitEmotion = emotion[0..., 0..., 0..<configuration.implicitEmotionCount]
        let explicitEmotion = emotion[
            0...,
            0...,
            configuration.implicitEmotionCount..<configuration.emotionVectorLength
        ]
        let projectedEmotion = linearMatMul(
            explicitEmotion,
            weight: tensor(graphTensors.emotionProjection),
            bias: tensor("emo_linear1.bias"))

        let autocorrFeatures = frequencyBranch(input)
        let audioFeatures = audioBranch(input)

        let audioHead = relu(linearMatMul(
            audioFeatures.reshaped([audioFeatures.dim(0), audioFeatures.dim(1), audioFeatures.dim(2)]),
            weight: tensor(graphTensors.audioFeatureMap),
            bias: tensor("audio_feature_map.bias")))
        let audioHead5D = audioHead
            .reshaped([audioHead.dim(0), 1, audioHead.dim(1), audioHead.dim(2), 1])
            .transposed(0, 1, 3, 2, 4)

        let freq5D = autocorrFeatures.reshaped([
            autocorrFeatures.dim(0),
            1,
            autocorrFeatures.dim(1),
            autocorrFeatures.dim(2),
            autocorrFeatures.dim(3)
        ])
        let joined = concatenated([audioHead5D, freq5D], axis: 2)
        var x = joined.reshaped([joined.dim(0) * joined.dim(1), joined.dim(2), joined.dim(3), joined.dim(4)])

        x = relu(conv2dNCHW(x, weight: tensor("time1.weight"), bias: tensor("time1.bias"), stride: (2, 1), padding: (1, 0))) * scalar(Self.sqrt2Scalar)
        x = appendEmotion(to: x, implicitEmotion: implicitEmotion, projectedEmotion: projectedEmotion)
        x = relu(conv2dNCHW(x, weight: tensor("time2.weight"), bias: tensor("time2.bias"), stride: (2, 1), padding: (1, 0))) * scalar(Self.sqrt2Scalar)
        x = appendEmotion(to: x, implicitEmotion: implicitEmotion, projectedEmotion: projectedEmotion)
        x = relu(conv2dNCHW(x, weight: tensor("time3.weight"), bias: tensor("time3.bias"), stride: (2, 1), padding: (1, 0))) * scalar(Self.sqrt2Scalar)
        x = appendEmotion(to: x, implicitEmotion: implicitEmotion, projectedEmotion: projectedEmotion)
        x = relu(conv2dNCHW(x, weight: tensor("time4.weight"), bias: tensor("time4.bias"), stride: (2, 1), padding: (1, 0))) * scalar(Self.sqrt2Scalar)
        x = appendEmotion(to: x, implicitEmotion: implicitEmotion, projectedEmotion: projectedEmotion)

        let flattened = x.reshaped([x.dim(0), x.dim(1) * x.dim(2) * x.dim(3)])
        let time5 = relu(gemm(flattened, weight: tensor("time5.weight"), bias: tensor("time5.bias"))
            .reshaped([x.dim(0), 1, 256])) * scalar(Self.sqrt2Scalar)
        let headInput = concatenated([time5, implicitEmotion, projectedEmotion], axis: 2)
        let headFlat = headInput.reshaped([headInput.dim(0) * headInput.dim(1), headInput.dim(2)])

        let faceCount = configuration.coefficientLayout.skinCount
            + configuration.coefficientLayout.jawCount
            + configuration.coefficientLayout.eyeCount
        let face0 = gemm(headFlat, weight: tensor("out_mapping.weight"), bias: tensor("out_mapping.bias"))
            .reshaped([headInput.dim(0), headInput.dim(1), faceCount])
        let face = gemm(
            face0.reshaped([face0.dim(0) * face0.dim(1), face0.dim(2)]),
            weight: tensor("out_mapping1.weight"),
            bias: tensor("out_mapping1.bias"))
            .reshaped([face0.dim(0), face0.dim(1), faceCount])

        let tongueCount = configuration.coefficientLayout.tongueCount
        let tongue0 = gemm(headFlat, weight: tensor("out_mapping_tongue.weight"), bias: tensor("out_mapping_tongue.bias"))
            .reshaped([headInput.dim(0), headInput.dim(1), tongueCount])
        let tongue = gemm(
            tongue0.reshaped([tongue0.dim(0) * tongue0.dim(1), tongue0.dim(2)]),
            weight: tensor("out_mapping1_tongue.weight"),
            bias: tensor("out_mapping1_tongue.bias"))
            .reshaped([tongue0.dim(0), tongue0.dim(1), tongueCount])

        let skin = face[0..., 0..., 0..<configuration.coefficientLayout.skinCount]
        let jawAndEye = face[0..., 0..., configuration.coefficientLayout.skinCount..<faceCount]
        return concatenated([skin, tongue, jawAndEye], axis: 2)
    }

    private func audioBranch(_ input: MLXArray) -> MLXArray {
        var x = input.reshaped([input.dim(0) * input.dim(1), input.dim(2)])
            .expandedDimensions(axis: 1)

        x = conv1dNCL(
            x,
            weight: tensor("wave2vec_model.feature_extractor.conv_layers.0.conv.weight"),
            stride: 5)
        x = instanceNorm1d(
            x,
            weight: tensor(graphTensors.featureExtractorScale),
            bias: tensor(graphTensors.featureExtractorBias))
        x = gelu(x)

        for layer in 1...6 {
            let stride = 2
            x = conv1dNCL(
                x,
                weight: tensor("wave2vec_model.feature_extractor.conv_layers.\(layer).conv.weight"),
                stride: stride)
            x = gelu(x)
        }

        x = x.transposed(0, 2, 1)
        x = layerNorm(
            x,
            weight: tensor("wave2vec_model.encoder.feature_projection.layer_norm.weight"),
            bias: tensor("wave2vec_model.encoder.feature_projection.layer_norm.bias"))
        x = linearMatMul(
            x,
            weight: tensor(graphTensors.featureProjection),
            bias: tensor("wave2vec_model.encoder.feature_projection.projection.bias"))

        var pos = conv1dNCL(
            x.transposed(0, 2, 1),
            weight: tensor(graphTensors.positionConv),
            bias: tensor("wave2vec_model.encoder.transformer.pos_conv_embed.conv.bias"),
            padding: 64,
            groups: 16)
        pos = pos[0..., 0..., 0..<x.dim(1)]
        x = x + gelu(pos).transposed(0, 2, 1)

        x = layerNorm(
            x,
            weight: tensor("wave2vec_model.encoder.transformer.layer_norm.weight"),
            bias: tensor("wave2vec_model.encoder.transformer.layer_norm.bias"))

        var layerOutputs: [MLXArray] = []
        layerOutputs.reserveCapacity(graphTensors.transformerLayers.count)
        for layer in graphTensors.transformerLayers {
            x = transformerLayer(x, tensors: layer)
            layerOutputs.append(x)
        }

        return combineTransformerOutputs(layerOutputs)
    }

    private func transformerLayer(_ input: MLXArray, tensors: TransformerLayerTensors) -> MLXArray {
        var x = input
        let batch = x.dim(0)
        let time = x.dim(1)
        let hidden = x.dim(2)
        let heads = 12
        let headDim = hidden / heads

        let q = linearMatMul(
            x,
            weight: tensor(tensors.queryProjection),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).attention.q_proj.bias"))
            .reshaped([batch, time, heads, headDim])
            .transposed(0, 2, 1, 3)
        let k = linearMatMul(
            x,
            weight: tensor(tensors.keyProjection),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).attention.k_proj.bias"))
            .reshaped([batch, time, heads, headDim])
            .transposed(0, 2, 3, 1)
        let v = linearMatMul(
            x,
            weight: tensor(tensors.valueProjection),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).attention.v_proj.bias"))
            .reshaped([batch, time, heads, headDim])
            .transposed(0, 2, 1, 3)

        var scores = matmul(q * MLXArray(Float(1.0 / Foundation.sqrt(Double(headDim)))), k)
        scores = scores - MLX.max(scores, axis: -1, keepDims: true)
        let attention = softmax(scores, axis: -1)
        let context = matmul(attention, v)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, time, hidden])
        let attentionOut = linearMatMul(
            context,
            weight: tensor(tensors.outputProjection),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).attention.out_proj.bias"))

        x = x + attentionOut
        x = layerNorm(
            x,
            weight: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).layer_norm.weight"),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).layer_norm.bias"))

        let ff = gelu(linearMatMul(
            x,
            weight: tensor(tensors.feedForwardIntermediate),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).feed_forward.intermediate_dense.bias")))
        let ffOut = linearMatMul(
            ff,
            weight: tensor(tensors.feedForwardOutput),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).feed_forward.output_dense.bias"))
        x = x + ffOut

        return layerNorm(
            x,
            weight: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).final_layer_norm.weight"),
            bias: tensor("wave2vec_model.encoder.transformer.layers.\(tensors.index).final_layer_norm.bias"))
    }

    private func combineTransformerOutputs(_ outputs: [MLXArray]) -> MLXArray {
        precondition(!outputs.isEmpty, "Audio2Face3D graph must contain at least one transformer layer")
        guard outputs.count > 1 else { return outputs[0] }
        precondition(transformerCombineWeights.count == outputs.count, "Audio2Face3D combine_weights does not match transformer layer count")
        let total = transformerCombineWeights.reduce(Float(0), +)
        precondition(total != 0, "Audio2Face3D combine_weights sum must be non-zero")
        var combined = outputs[0] * scalar(transformerCombineWeights[0] / total)
        for index in 1..<outputs.count {
            combined = combined + outputs[index] * scalar(transformerCombineWeights[index] / total)
        }
        return combined
    }

    private func frequencyBranch(_ input: MLXArray) -> MLXArray {
        let scaled = input * tensor("in_scale_autocorr")
        let hann = MLXArray(Self.autocorrWindow, [1, 1, 1, 640])

        var windows: [MLXArray] = []
        windows.reserveCapacity(25)
        for index in 0..<25 {
            let start = index * 320
            let window = scaled[0..., 0..., start..<(start + 640)]
                .expandedDimensions(axis: 2)
            windows.append(window)
        }
        let framed = concatenated(windows, axis: 2)
        let centered = (framed - MLX.mean(framed, axis: -1, keepDims: true)) * hann

        var correlations: [MLXArray] = []
        correlations.reserveCapacity(32)
        for lag in 0..<32 {
            let lhs: MLXArray
            let rhs: MLXArray
            if lag == 0 {
                lhs = centered
                rhs = centered
            } else {
                lhs = centered[0..., 0..., 0..., 0..<(640 - lag)]
                rhs = centered[0..., 0..., 0..., lag..<640]
            }
            correlations.append(MLX.sum(lhs * rhs, axis: -1).expandedDimensions(axis: -1))
        }

        var x = concatenated(correlations, axis: -1) * scalar(Self.autocorrScaleScalar)
        x = relu(conv2dNCHW(x, weight: tensor("freq1.weight"), bias: tensor("freq1.bias"), stride: (1, 2), padding: (0, 1))) * scalar(Self.sqrt2Scalar)
        x = relu(conv2dNCHW(x, weight: tensor("freq2.weight"), bias: tensor("freq2.bias"), stride: (1, 2), padding: (0, 1))) * scalar(Self.sqrt2Scalar)
        x = relu(conv2dNCHW(x, weight: tensor("freq3.weight"), bias: tensor("freq3.bias"), stride: (1, 2), padding: (0, 1))) * scalar(Self.sqrt2Scalar)
        x = relu(conv2dNCHW(x, weight: tensor("freq4.weight"), bias: tensor("freq4.bias"), stride: (1, 2), padding: (0, 1))) * scalar(Self.sqrt2Scalar)
        x = relu(conv2dNCHW(x, weight: tensor("freq5.weight"), bias: tensor("freq5.bias"), stride: (1, 1), padding: (0, 0))) * scalar(Self.sqrt2Scalar)
        return x
    }

    private func appendEmotion(
        to x: MLXArray,
        implicitEmotion: MLXArray,
        projectedEmotion: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(2)
        let width = x.dim(3)
        let implicit = MLX.broadcast(
            implicitEmotion.reshaped([batch, configuration.implicitEmotionCount, 1, 1]),
            to: [batch, configuration.implicitEmotionCount, height, width])
        let projected = MLX.broadcast(
            projectedEmotion.reshaped([batch, 8, 1, 1]),
            to: [batch, 8, height, width])
        return concatenated([x, implicit, projected], axis: 1)
    }

    private func conv1dNCL(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray? = nil,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1
    ) -> MLXArray {
        let weightNLC = weight.transposed(0, 2, 1)
        var y = MLX.conv1d(
            x.transposed(0, 2, 1),
            weightNLC,
            stride: stride,
            padding: padding,
            groups: groups)
            .transposed(0, 2, 1)
        if let bias {
            y = y + bias.reshaped([1, bias.dim(0), 1])
        }
        return y
    }

    private func conv2dNCHW(
        _ x: MLXArray,
        weight: MLXArray,
        bias: MLXArray,
        stride: (Int, Int),
        padding: (Int, Int)
    ) -> MLXArray {
        let weightNHWC = weight.transposed(0, 2, 3, 1)
        var y = MLX.conv2d(
            x.transposed(0, 2, 3, 1),
            weightNHWC,
            stride: .init(stride),
            padding: .init(padding))
            .transposed(0, 3, 1, 2)
        y = y + bias.reshaped([1, bias.dim(0), 1, 1])
        return y
    }

    private func instanceNorm1d(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let centered = x - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        return centered / MLX.sqrt(variance + scalar(Self.epsScalar)) * weight.reshaped([1, weight.dim(0), 1]) + bias.reshaped([1, bias.dim(0), 1])
    }

    private func layerNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let mean = MLX.mean(x, axis: -1, keepDims: true)
        let centered = x - mean
        let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
        return centered / MLX.sqrt(variance + scalar(Self.epsScalar)) * weight + bias
    }

    private func gelu(_ x: MLXArray) -> MLXArray {
        x * (scalar(Self.oneScalar) + MLX.erf(x / scalar(Self.sqrt2Scalar))) * scalar(Self.halfScalar)
    }

    private func relu(_ x: MLXArray) -> MLXArray {
        MLX.maximum(x, MLXArray(Float(0)))
    }

    private func scalar(_ value: Float) -> MLXArray {
        MLXArray(value)
    }

    private func linearMatMul(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        matmul(x, weight) + bias
    }

    private func gemm(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        matmul(x, weight.transposed(1, 0)) + bias
    }

    private func tensor(_ key: String) -> MLXArray {
        guard let value = weights[key] else {
            preconditionFailure("Missing validated Audio2Face3D tensor \(key)")
        }
        return value
    }

    private static func readDefaultEmotion(from directory: URL, expectedCount: Int) -> [Float]? {
        let url = directory.appendingPathComponent("default_emotion.f32")
        guard
            expectedCount > 0,
            let data = try? Data(contentsOf: url),
            data.count == expectedCount * MemoryLayout<Float>.size
        else {
            return nil
        }

        return data.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            return (0..<expectedCount).map { index in
                let offset = index * MemoryLayout<Float>.size
                let bits =
                    UInt32(bytes[offset]) |
                    (UInt32(bytes[offset + 1]) << 8) |
                    (UInt32(bytes[offset + 2]) << 16) |
                    (UInt32(bytes[offset + 3]) << 24)
                return Float(bitPattern: bits)
            }
        }
    }

    private func validateRequiredTensors() throws {
        var required = [
            "audio_feature_map.bias",
            "combine_weights",
            "emo_linear1.bias",
            "freq1.bias",
            "freq1.weight",
            "freq2.bias",
            "freq2.weight",
            "freq3.bias",
            "freq3.weight",
            "freq4.bias",
            "freq4.weight",
            "freq5.bias",
            "freq5.weight",
            "in_scale_autocorr",
            "out_mapping.bias",
            "out_mapping.weight",
            "out_mapping1.bias",
            "out_mapping1.weight",
            "out_mapping_tongue.bias",
            "out_mapping_tongue.weight",
            "out_mapping1_tongue.bias",
            "out_mapping1_tongue.weight",
            "time1.bias",
            "time1.weight",
            "time2.bias",
            "time2.weight",
            "time3.bias",
            "time3.weight",
            "time4.bias",
            "time4.weight",
            "time5.bias",
            "time5.weight",
            "wave2vec_model.encoder.feature_projection.layer_norm.bias",
            "wave2vec_model.encoder.feature_projection.layer_norm.weight",
            "wave2vec_model.encoder.feature_projection.projection.bias",
            "wave2vec_model.encoder.transformer.layer_norm.bias",
            "wave2vec_model.encoder.transformer.layer_norm.weight",
            "wave2vec_model.encoder.transformer.pos_conv_embed.conv.bias",
            "wave2vec_model.feature_extractor.conv_layers.0.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.1.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.2.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.3.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.4.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.5.conv.weight",
            "wave2vec_model.feature_extractor.conv_layers.6.conv.weight"
        ]
        required.append(contentsOf: [
            graphTensors.audioFeatureMap,
            graphTensors.emotionProjection,
            graphTensors.featureExtractorBias,
            graphTensors.featureExtractorScale,
            graphTensors.featureProjection,
            graphTensors.positionConv
        ])
        for layer in graphTensors.transformerLayers {
            required.append(contentsOf: [
                layer.queryProjection,
                layer.keyProjection,
                layer.valueProjection,
                layer.outputProjection,
                layer.feedForwardIntermediate,
                layer.feedForwardOutput,
                "wave2vec_model.encoder.transformer.layers.\(layer.index).attention.k_proj.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).attention.out_proj.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).attention.q_proj.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).attention.v_proj.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).feed_forward.intermediate_dense.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).feed_forward.output_dense.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).final_layer_norm.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).final_layer_norm.weight",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).layer_norm.bias",
                "wave2vec_model.encoder.transformer.layers.\(layer.index).layer_norm.weight"
            ])
        }

        for key in required where weights[key] == nil {
            throw Audio2Face3DError.missingTensor(key)
        }
        if transformerCombineWeights.count != graphTensors.transformerLayers.count {
            throw Audio2Face3DError.missingTensor("combine_weights[\(graphTensors.transformerLayers.count)]")
        }
    }

    private static func resolveGraphTensors(weights: [String: MLXArray]) throws -> GraphTensors {
        if weights["onnx::MatMul_1738"] != nil {
            return GraphTensors(
                emotionProjection: "onnx::MatMul_1738",
                featureExtractorScale: "onnx::Mul_1742",
                featureExtractorBias: "onnx::Add_1743",
                featureProjection: "onnx::MatMul_1744",
                positionConv: "onnx::Conv_1747",
                audioFeatureMap: "onnx::MatMul_1760",
                transformerLayers: [
                    TransformerLayerTensors(
                        index: 0,
                        queryProjection: "onnx::MatMul_1748",
                        keyProjection: "onnx::MatMul_1755",
                        valueProjection: "onnx::MatMul_1756",
                        outputProjection: "onnx::MatMul_1757",
                        feedForwardIntermediate: "onnx::MatMul_1758",
                        feedForwardOutput: "onnx::MatMul_1759")
                ])
        }

        if weights["2230"] != nil {
            return GraphTensors(
                emotionProjection: "2230",
                featureExtractorScale: "2235",
                featureExtractorBias: "2236",
                featureProjection: "2237",
                positionConv: "2240",
                audioFeatureMap: "2289",
                transformerLayers: [
                    TransformerLayerTensors(
                        index: 0,
                        queryProjection: "2241",
                        keyProjection: "2244",
                        valueProjection: "2247",
                        outputProjection: "2250",
                        feedForwardIntermediate: "2251",
                        feedForwardOutput: "2252"),
                    TransformerLayerTensors(
                        index: 1,
                        queryProjection: "2253",
                        keyProjection: "2256",
                        valueProjection: "2259",
                        outputProjection: "2262",
                        feedForwardIntermediate: "2263",
                        feedForwardOutput: "2264"),
                    TransformerLayerTensors(
                        index: 2,
                        queryProjection: "2265",
                        keyProjection: "2268",
                        valueProjection: "2271",
                        outputProjection: "2274",
                        feedForwardIntermediate: "2275",
                        feedForwardOutput: "2276"),
                    TransformerLayerTensors(
                        index: 3,
                        queryProjection: "2277",
                        keyProjection: "2280",
                        valueProjection: "2283",
                        outputProjection: "2286",
                        feedForwardIntermediate: "2287",
                        feedForwardOutput: "2288")
                ])
        }

        if weights["2232"] != nil {
            return GraphTensors(
                emotionProjection: "2232",
                featureExtractorScale: "2237",
                featureExtractorBias: "2238",
                featureProjection: "2239",
                positionConv: "2242",
                audioFeatureMap: "2291",
                transformerLayers: [
                    TransformerLayerTensors(
                        index: 0,
                        queryProjection: "2243",
                        keyProjection: "2246",
                        valueProjection: "2249",
                        outputProjection: "2252",
                        feedForwardIntermediate: "2253",
                        feedForwardOutput: "2254"),
                    TransformerLayerTensors(
                        index: 1,
                        queryProjection: "2255",
                        keyProjection: "2258",
                        valueProjection: "2261",
                        outputProjection: "2264",
                        feedForwardIntermediate: "2265",
                        feedForwardOutput: "2266"),
                    TransformerLayerTensors(
                        index: 2,
                        queryProjection: "2267",
                        keyProjection: "2270",
                        valueProjection: "2273",
                        outputProjection: "2276",
                        feedForwardIntermediate: "2277",
                        feedForwardOutput: "2278"),
                    TransformerLayerTensors(
                        index: 3,
                        queryProjection: "2279",
                        keyProjection: "2282",
                        valueProjection: "2285",
                        outputProjection: "2288",
                        feedForwardIntermediate: "2289",
                        feedForwardOutput: "2290")
                ])
        }

        throw Audio2Face3DError.missingTensor("known Audio2Face3D graph tensor set")
    }
}

private struct GraphTensors {
    let emotionProjection: String
    let featureExtractorScale: String
    let featureExtractorBias: String
    let featureProjection: String
    let positionConv: String
    let audioFeatureMap: String
    let transformerLayers: [TransformerLayerTensors]
}

private struct TransformerLayerTensors {
    let index: Int
    let queryProjection: String
    let keyProjection: String
    let valueProjection: String
    let outputProjection: String
    let feedForwardIntermediate: String
    let feedForwardOutput: String
}
