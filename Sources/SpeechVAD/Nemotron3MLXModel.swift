import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXNN

private final class Nemotron3RoPE {
    let cosine: MLXArray
    let sine: MLXArray

    init(headDimension: Int = 64, maximumLength: Int = 684) {
        let half = headDimension / 2
        let exponents = MLXArray(
            (0..<half).map { Float(2 * $0) / Float(headDimension) })
        let inverse = MLXArray(Float(1))
            / MLX.pow(MLXArray(Float(10_000)), exponents)
        let positions = MLXArray((0..<maximumLength).map(Float.init))
        let frequencies = positions.reshaped(maximumLength, 1)
            * inverse.reshaped(1, half)
        let halfCosine = cos(frequencies)
        let halfSine = sin(frequencies)
        cosine = concatenated([halfCosine, halfCosine], axis: -1)
        sine = concatenated([halfSine, halfSine], axis: -1)
    }

    func apply(_ value: MLXArray) -> MLXArray {
        // value: [B, H, T, D], NeMo/GPT-NeoX split-half rotation.
        let time = value.dim(2)
        let dimensions = value.dim(3)
        let half = dimensions / 2
        let cosine = cosine[0..<time, 0...]
            .reshaped(1, 1, time, dimensions).asType(value.dtype)
        let sine = sine[0..<time, 0...]
            .reshaped(1, 1, time, dimensions).asType(value.dtype)
        let first = value[0..., 0..., 0..., 0..<half]
        let second = value[0..., 0..., 0..., half..<dimensions]
        return value * cosine + concatenated([-second, first], axis: -1) * sine
    }
}

private final class Nemotron3MLXAttention: Module {
    private let heads = 8
    private let headDimension = 64
    private let scale: Float = 1 / 8
    private let rope: Nemotron3RoPE

    @ModuleInfo(key: "w_qkv") var qkvProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(rope: Nemotron3RoPE) {
        self.rope = rope
        _qkvProjection.wrappedValue = Linear(512, 1_536, bias: false)
        _outputProjection.wrappedValue = Linear(512, 512, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = input.dim(0)
        let time = input.dim(1)
        let qkv = qkvProjection(input)
            .reshaped(batch, time, 3, heads, headDimension)
            .transposed(2, 0, 3, 1, 4)
        var query = qkv[0]
        var key = qkv[1]
        let value = qkv[2]
        query = rope.apply(query)
        key = rope.apply(key)
        let attended = SDPA.attendAndMerge(
            qHeads: query,
            kHeads: key,
            vHeads: value,
            scale: scale,
            mask: mask)
        return outputProjection(attended)
    }
}

private final class Nemotron3MLXFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    override init() {
        _linear1.wrappedValue = Linear(512, 2_048, bias: true)
        _linear2.wrappedValue = Linear(2_048, 512, bias: true)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear2(gelu(linear1(input)))
    }
}

private final class Nemotron3MLXLayer: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: Nemotron3MLXAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var feedForward: Nemotron3MLXFeedForward

    init(rope: Nemotron3RoPE) {
        _norm1.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _attention.wrappedValue = Nemotron3MLXAttention(rope: rope)
        _norm2.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _feedForward.wrappedValue = Nemotron3MLXFeedForward()
        super.init()
    }

    func callAsFunction(_ input: MLXArray, mask: MLXArray) -> MLXArray {
        let attended = input + attention(norm1(input), mask: mask)
        return attended + feedForward(norm2(attended))
    }
}

private final class Nemotron3MLXPreEncoder: Module {
    @ModuleInfo var proj: Linear

    override init() {
        _proj.wrappedValue = Linear(1_024, 512, bias: false)
        super.init()
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        let time = features.dim(1)
        precondition(time % 8 == 0)
        return proj(features.reshaped(-1, time / 8, 1_024))
    }
}

private final class Nemotron3MLXEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: Nemotron3MLXPreEncoder
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm
    @ModuleInfo var layers: [Nemotron3MLXLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    override init() {
        let rope = Nemotron3RoPE()
        _preEncode.wrappedValue = Nemotron3MLXPreEncoder()
        _embedNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        _layers.wrappedValue = (0..<31).map { _ in
            Nemotron3MLXLayer(rope: rope)
        }
        _finalNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1.0e-5)
        super.init()
    }

    func encode(_ embeddings: MLXArray, mask: MLXArray) -> MLXArray {
        var hidden = embedNorm(embeddings)
        for layer in layers {
            hidden = layer(hidden, mask: mask)
        }
        return finalNorm(hidden)
    }
}

private final class Nemotron3MLXHead: Module {
    @ModuleInfo(key: "encoder_proj") var encoderProjection: Linear
    @ModuleInfo(key: "subpixel_upsample") var subpixelUpsample: Conv1d
    @ModuleInfo(key: "first_hidden_to_hidden") var hiddenProjection: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var speakerProjection: Linear

    override init() {
        _encoderProjection.wrappedValue = Linear(512, 192, bias: true)
        _subpixelUpsample.wrappedValue = Conv1d(
            inputChannels: 192,
            outputChannels: 1_536,
            kernelSize: 3,
            padding: 1,
            bias: true)
        _hiddenProjection.wrappedValue = Linear(192, 192, bias: true)
        _speakerProjection.wrappedValue = Linear(192, 8, bias: true)
        super.init()
    }

    func callAsFunction(
        _ encoded: MLXArray, validLength: Int
    ) -> (high: MLXArray, low: MLXArray) {
        let batch = encoded.dim(0)
        let frames = encoded.dim(1)
        var hidden = subpixelUpsample(encoderProjection(encoded))
        hidden = hidden.reshaped(batch, frames, 8, 192)
            .reshaped(batch, frames * 8, 192)
        hidden = maximum(hidden, MLXArray(Float(0)))
        hidden = maximum(hiddenProjection(hidden), MLXArray(Float(0)))
        var high = sigmoid(speakerProjection(hidden))
        let valid = MLXArray(0..<Int32(frames * 8))
            .< MLXArray(Int32(validLength * 8))
        high = high * valid.reshaped(1, frames * 8, 1).asType(high.dtype)
        let low = high.reshaped(batch, frames, 8, 8).mean(axis: 2)
        return (high, low)
    }
}

private final class Nemotron3MLXNetwork: Module {
    @ModuleInfo var encoder: Nemotron3MLXEncoder
    @ModuleInfo(key: "sortformer_modules") var head: Nemotron3MLXHead

    override init() {
        _encoder.wrappedValue = Nemotron3MLXEncoder()
        _head.wrappedValue = Nemotron3MLXHead()
        super.init()
    }

    func preencode(_ features: MLXArray) -> MLXArray {
        encoder.preEncode(features)
    }

    func inferHead(
        _ embeddings: MLXArray, validLength: Int
    ) -> (high: MLXArray, low: MLXArray) {
        let frames = embeddings.dim(1)
        let valid = MLXArray(0..<Int32(frames))
            .< MLXArray(Int32(validLength))
        let zero = MLXArray(Float(0))
        let negative = MLXArray(Float(-10_000))
        let mask = MLX.where(valid, zero, negative)
            .reshaped(1, 1, 1, frames)
        return head(encoder.encode(embeddings, mask: mask), validLength: validLength)
    }
}

final class Nemotron3MLXBackend: Nemotron3InferenceBackend {
    private let network: Nemotron3MLXNetwork
    let learnedSilenceEmbedding: [Float]

    init(directory: URL) throws {
        let configuration = try Nemotron3ArtifactConfiguration.load(from: directory)
        guard configuration.quantization.groupSize == 64,
              configuration.quantization.bits == 8,
              configuration.quantization.mode == "affine" else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "MLX runtime requires affine group-64 INT8 weights")
        }
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Nemotron3DiarizationError.missingArtifact(weightsURL.path)
        }
        var weights = try MLX.loadArrays(url: weightsURL)
        guard let silence = weights.removeValue(
            forKey: "sortformer_modules.learnable_sil_emb") else {
            throw Nemotron3DiarizationError.incompatibleWeights(
                "learnable silence embedding is missing")
        }
        let silenceFloat = silence.asType(.float32)
        eval(silenceFloat)
        learnedSilenceEmbedding = silenceFloat.asArray(Float.self)
        guard learnedSilenceEmbedding.count == 512 else {
            throw Nemotron3DiarizationError.incompatibleWeights(
                "learnable silence embedding must contain 512 values")
        }

        weights = weights.filter {
            $0.key.hasPrefix("encoder.")
                || $0.key.hasPrefix("sortformer_modules.")
        }
        var remapped: [String: MLXArray] = [:]
        remapped.reserveCapacity(weights.count)
        for (key, value) in weights {
            let localKey = key
                .replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.linear1.")
                .replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.linear2.")
            remapped[localKey] = value
        }

        let network = Nemotron3MLXNetwork()
        MLXNN.quantize(model: network) { path, _ in
            remapped["\(path).scales"] == nil ? nil : (64, 8, .affine)
        }
        do {
            try network.update(
                parameters: ModuleParameters.unflattened(remapped),
                verify: .all)
        } catch {
            throw Nemotron3DiarizationError.incompatibleWeights(
                error.localizedDescription)
        }
        network.train(false)
        eval(network)
        self.network = network
        Memory.clearCache()
    }

    func preencode(chunk: [Float]) throws -> [Float] {
        guard chunk.count == 3_040 * 128 else {
            throw Nemotron3DiarizationError.runtime(
                "MLX pre-encoder expects 3040 × 128 values")
        }
        let output = network.preencode(
            MLXArray(chunk).reshaped(1, 3_040, 128))
        eval(output)
        let values = output.asType(.float32).asArray(Float.self)
        Memory.clearCache()
        return values
    }

    func predictHead(
        packedEmbeddings: [Float], validLength: Int
    ) throws -> Nemotron3HeadOutput {
        guard packedEmbeddings.count == 684 * 512,
              (0...684).contains(validLength) else {
            throw Nemotron3DiarizationError.runtime(
                "MLX head received invalid packed embeddings")
        }
        let embeddings = MLXArray(packedEmbeddings).reshaped(1, 684, 512)
        let output = network.inferHead(embeddings, validLength: validLength)
        eval(output.high, output.low)
        let result = Nemotron3HeadOutput(
            probabilities10ms: output.high.asType(.float32).asArray(Float.self),
            probabilities80ms: output.low.asType(.float32).asArray(Float.self))
        Memory.clearCache()
        return result
    }
}

public extension Nemotron3Diarizer {
    static let defaultMLXModelId =
        "aufklarer/Nemotron-3-Diarization-100M-MLX-INT8"

    /// Download and load the final MLX INT8 bundle.
    static func fromMLXPretrained(
        modelId: String = defaultMLXModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Nemotron3Diarizer {
        let directory = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        progressHandler?(0, "Downloading Nemotron 3 MLX bundle...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: ["model.safetensors", "config.json"],
            offlineMode: offlineMode,
            progressHandler: { fraction in
                progressHandler?(fraction * 0.8, "Downloading Nemotron 3 MLX bundle...")
            }
        )
        progressHandler?(0.8, "Loading Nemotron 3 MLX weights...")
        let model = try fromMLXDirectory(directory)
        progressHandler?(1, "Ready")
        return model
    }

    /// Load a previously downloaded MLX INT8 bundle.
    static func fromMLXDirectory(_ directory: URL) throws -> Nemotron3Diarizer {
        Nemotron3Diarizer(backend: try Nemotron3MLXBackend(directory: directory))
    }
}
