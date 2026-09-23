#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

final class Nemotron3CoreMLBackend: Nemotron3InferenceBackend {
    private let preencoder: MLModel
    private let head: MLModel
    let learnedSilenceEmbedding: [Float]

    init(directory: URL, computeUnits: MLComputeUnits) throws {
        _ = try Nemotron3ArtifactConfiguration.load(from: directory)
        let preencoderURL = directory.appendingPathComponent(
            "Nemotron3PreEncoder.mlmodelc", isDirectory: true)
        let headURL = directory.appendingPathComponent(
            "Nemotron3Head.mlmodelc", isDirectory: true)
        for url in [preencoderURL, headURL] where
            !FileManager.default.fileExists(atPath: url.path)
        {
            throw Nemotron3DiarizationError.missingArtifact(url.path)
        }

        let modelConfiguration = MLModelConfiguration()
        modelConfiguration.computeUnits = CoreMLComputeUnitsResolver.resolved(
            default: computeUnits)
        modelConfiguration.allowLowPrecisionAccumulationOnGPU = true
        do {
            preencoder = try MLModel(
                contentsOf: preencoderURL, configuration: modelConfiguration)
            head = try MLModel(
                contentsOf: headURL, configuration: modelConfiguration)
        } catch {
            throw Nemotron3DiarizationError.runtime(
                "could not load compiled Core ML graph: \(error.localizedDescription)")
        }

        let silenceURL = directory.appendingPathComponent("learnable_silence.f32")
        guard FileManager.default.fileExists(atPath: silenceURL.path) else {
            throw Nemotron3DiarizationError.missingArtifact(silenceURL.path)
        }
        let data = try Data(contentsOf: silenceURL)
        guard data.count == 512 * MemoryLayout<Float>.size else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "learnable_silence.f32 must contain 512 Float32 values")
        }
        learnedSilenceEmbedding = data.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
    }

    func preencode(chunk: [Float]) throws -> [Float] {
        guard chunk.count == 3_040 * 128 else {
            throw Nemotron3DiarizationError.runtime(
                "Core ML pre-encoder expects 3040 × 128 values")
        }
        let input = try Self.multiArray(
            shape: [1, 3_040, 128], values: chunk)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "chunk": MLFeatureValue(multiArray: input)
        ])
        let result = try preencoder.prediction(from: provider)
        guard let output = result.featureValue(for: "chunk_embeddings")?.multiArrayValue else {
            throw Nemotron3DiarizationError.runtime(
                "Core ML pre-encoder did not return chunk_embeddings")
        }
        return SortformerCoreMLModel.denseFloats(from: output)
    }

    func predictHead(
        packedEmbeddings: [Float], validLength: Int
    ) throws -> Nemotron3HeadOutput {
        guard packedEmbeddings.count == 684 * 512,
              (0...684).contains(validLength) else {
            throw Nemotron3DiarizationError.runtime(
                "Core ML head received invalid packed embeddings")
        }
        let embeddings = try Self.multiArray(
            shape: [1, 684, 512], values: packedEmbeddings)
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(validLength)
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "packed_embeddings": MLFeatureValue(multiArray: embeddings),
            "packed_length": MLFeatureValue(multiArray: length),
        ])
        let result = try head.prediction(from: provider)
        guard let high = result.featureValue(
            for: "speaker_probabilities_10ms")?.multiArrayValue,
              let low = result.featureValue(
                for: "speaker_probabilities_80ms")?.multiArrayValue else {
            throw Nemotron3DiarizationError.runtime(
                "Core ML head did not return both probability tensors")
        }
        return Nemotron3HeadOutput(
            probabilities10ms: SortformerCoreMLModel.denseFloats(from: high),
            probabilities80ms: SortformerCoreMLModel.denseFloats(from: low))
    }

    private static func multiArray(
        shape: [NSNumber], values: [Float]
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        guard array.count == values.count else {
            throw Nemotron3DiarizationError.runtime(
                "Core ML input shape does not match the supplied values")
        }
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        values.withUnsafeBufferPointer { source in
            pointer.update(from: source.baseAddress!, count: values.count)
        }
        return array
    }
}

public extension Nemotron3Diarizer {
    static let defaultCoreMLModelId =
        "aufklarer/Nemotron-3-Diarization-100M-CoreML-INT8"

    /// Download and load the final compiled Core ML INT8 bundle.
    static func fromCoreMLPretrained(
        modelId: String = defaultCoreMLModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Nemotron3Diarizer {
        let directory = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        progressHandler?(0, "Downloading Nemotron 3 Core ML bundle...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: [
                "Nemotron3PreEncoder.mlmodelc/**",
                "Nemotron3Head.mlmodelc/**",
                "learnable_silence.f32",
                "config.json",
            ],
            offlineMode: offlineMode,
            progressHandler: { fraction in
                progressHandler?(fraction * 0.8, "Downloading Nemotron 3 Core ML bundle...")
            }
        )
        progressHandler?(0.8, "Loading Nemotron 3 Core ML graphs...")
        let model = try fromCoreMLDirectory(directory, computeUnits: computeUnits)
        progressHandler?(1, "Ready")
        return model
    }

    /// Load a previously downloaded compiled Core ML INT8 bundle.
    static func fromCoreMLDirectory(
        _ directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) throws -> Nemotron3Diarizer {
        let backend = try Nemotron3CoreMLBackend(
            directory: directory, computeUnits: computeUnits)
        return Nemotron3Diarizer(backend: backend)
    }
}
#endif
