#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

struct ReDimNet2ModelConfiguration: Decodable, Equatable {
    let modelType: String
    let sampleRate: Int
    let inputSamples: Int
    let embeddingDimension: Int
    let inputName: String
    let outputName: String
    let compiledModel: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case inputSamples = "input_samples"
        case embeddingDimension = "embedding_dimension"
        case inputName = "input_name"
        case outputName = "output_name"
        case compiledModel = "compiled_model"
    }
}

/// Core ML speaker-identity encoder based on ReDimNet2-B6.
///
/// This model is separate from the WeSpeaker embedding stage used inside
/// diarization. It produces 192-dimensional embeddings for comparing clean
/// voice samples across recordings and for persistent named voice profiles.
///
/// Input audio is resampled to 16 kHz and prepared as one fixed six-second
/// window. Two-to-six-second clips are repeated to fill the window; longer
/// clips are center-cropped. The default API rejects clips shorter than two
/// seconds because they did not meet the quality threshold used for identity
/// matching. `embedShortUtterance` is an explicit, lower-confidence path for
/// matching 0.6-to-2-second speech against an identity established from longer
/// audio; it must not be used to enroll or create an identity.
public final class ReDimNet2SpeakerModel {
    public static let defaultModelId = "aufklarer/ReDimNet2-B6-CoreML"
    public static let inputSampleRate = 16_000
    public static let inputSampleCount = 96_000
    public static let minimumSampleCount = 32_000
    public static let minimumShortUtteranceSampleCount = 9_600
    public static let embeddingDimension = 192

    private let model: MLModel

    init(model: MLModel) {
        self.model = model
    }

    /// Download and load the compiled ReDimNet2-B6 Core ML model.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ReDimNet2SpeakerModel {
        let cacheDir = try cacheDir
            ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading ReDimNet2 speaker identity model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["ReDimNet2B6.mlmodelc/**", "config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(
                    progress * 0.8,
                    "Downloading ReDimNet2 speaker identity model...")
            }
        )

        progressHandler?(0.8, "Loading ReDimNet2 speaker identity model...")
        let configuration = try loadConfiguration(
            at: cacheDir.appendingPathComponent("config.json"),
            modelId: modelId)
        let modelURL = cacheDir.appendingPathComponent(
            configuration.compiledModel, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Core ML model not found at \(modelURL.path)")
        }

        let loaded: MLModel
        do {
            loaded = try CoreMLLoader.load(
                url: modelURL,
                computeUnits: .all,
                name: "redimnet2-b6-speaker")
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to load compiled Core ML model",
                underlying: error)
        }

        progressHandler?(1.0, "Ready")
        return ReDimNet2SpeakerModel(model: loaded)
    }

    static func decodeConfiguration(_ data: Data) throws -> ReDimNet2ModelConfiguration {
        let configuration = try JSONDecoder().decode(
            ReDimNet2ModelConfiguration.self, from: data)
        guard configuration.modelType == "redimnet2-b6-speaker-coreml",
            configuration.sampleRate == inputSampleRate,
            configuration.inputSamples == inputSampleCount,
            configuration.embeddingDimension == embeddingDimension,
            configuration.inputName == "audio",
            configuration.outputName == "embedding",
            configuration.compiledModel == "ReDimNet2B6.mlmodelc"
        else {
            throw AudioModelError.invalidConfiguration(
                model: "ReDimNet2-B6",
                reason: "config.json does not match the compiled identity runtime")
        }
        return configuration
    }

    private static func loadConfiguration(
        at url: URL,
        modelId: String
    ) throws -> ReDimNet2ModelConfiguration {
        do {
            return try decodeConfiguration(Data(contentsOf: url))
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Missing or incompatible config.json",
                underlying: error)
        }
    }

    /// Extract a normalized 192-dimensional identity embedding.
    ///
    /// - Throws: An explicit input or inference error. No zero-vector fallback
    ///   is returned when identity extraction fails.
    public func embed(audio: [Float], sampleRate: Int) throws -> [Float] {
        let resampled = try Self.resampled(audio, sampleRate: sampleRate)
        return try predict(Self.preparedAudio(resampled))
    }

    /// Extract a normalized embedding from 0.6-to-2 seconds of clean speech.
    ///
    /// Short inputs contain less identity evidence than the default two-second
    /// minimum. Use this only for conservative retrieval against identities
    /// established from `embed`; never use it for enrollment, cluster creation,
    /// or centroid updates. Match thresholds require duration-specific
    /// calibration on the target domain.
    public func embedShortUtterance(
        audio: [Float], sampleRate: Int
    ) throws -> [Float] {
        let resampled = try Self.resampled(audio, sampleRate: sampleRate)
        return try predict(Self.preparedShortUtteranceAudio(resampled))
    }

    /// Compile and execute the graph once before latency-sensitive use.
    public func prewarm() throws {
        var waveform = [Float](repeating: 0, count: Self.inputSampleCount)
        for index in waveform.indices {
            waveform[index] = 0.01 * sin(
                2 * Float.pi * 173 * Float(index) / Float(Self.inputSampleRate))
        }
        _ = try predict(waveform)
    }

    static func preparedAudio(_ samples: [Float]) throws -> [Float] {
        try preparedAudio(samples, minimumSampleCount: minimumSampleCount)
    }

    static func preparedShortUtteranceAudio(_ samples: [Float]) throws -> [Float] {
        try preparedAudio(
            samples,
            minimumSampleCount: minimumShortUtteranceSampleCount)
    }

    private static func resampled(
        _ audio: [Float], sampleRate: Int
    ) throws -> [Float] {
        if sampleRate == inputSampleRate { return audio }
        guard sampleRate > 0 else {
            throw AudioModelError.invalidConfiguration(
                model: "ReDimNet2-B6",
                reason: "sample rate must be greater than zero")
        }
        return AudioFileLoader.resample(
            audio, from: sampleRate, to: inputSampleRate)
    }

    private static func preparedAudio(
        _ samples: [Float], minimumSampleCount: Int
    ) throws -> [Float] {
        guard samples.count >= minimumSampleCount else {
            let seconds = Double(samples.count) / Double(inputSampleRate)
            let minimumSeconds = Double(minimumSampleCount) / Double(inputSampleRate)
            throw AudioModelError.invalidConfiguration(
                model: "ReDimNet2-B6",
                reason: String(
                    format: "speaker identity requires at least %.1f seconds of clean audio (received %.2f seconds)",
                    minimumSeconds, seconds))
        }
        guard samples.allSatisfy(\.isFinite) else {
            throw AudioModelError.invalidConfiguration(
                model: "ReDimNet2-B6",
                reason: "audio contains a non-finite sample")
        }

        if samples.count == inputSampleCount {
            return samples
        }
        if samples.count > inputSampleCount {
            let start = (samples.count - inputSampleCount) / 2
            return Array(samples[start..<(start + inputSampleCount)])
        }

        var prepared = [Float](repeating: 0, count: inputSampleCount)
        for index in prepared.indices {
            prepared[index] = samples[index % samples.count]
        }
        return prepared
    }

    private func predict(_ preparedAudio: [Float]) throws -> [Float] {
        let input = try MLMultiArray(
            shape: [1, Self.inputSampleCount as NSNumber],
            dataType: .float32)
        let inputPointer = input.dataPointer.assumingMemoryBound(to: Float.self)
        preparedAudio.withUnsafeBufferPointer { source in
            inputPointer.update(from: source.baseAddress!, count: preparedAudio.count)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: input),
        ])
        let output: MLFeatureProvider
        do {
            output = try model.prediction(from: provider)
        } catch {
            throw AudioModelError.inferenceFailed(
                operation: "ReDimNet2 speaker identity",
                reason: error.localizedDescription)
        }

        guard let values = output.featureValue(
            for: "embedding")?.multiArrayValue,
            values.count == Self.embeddingDimension
        else {
            throw AudioModelError.inferenceFailed(
                operation: "ReDimNet2 speaker identity",
                reason: "missing 192-dimensional 'embedding' output")
        }

        var embedding = [Float](repeating: 0, count: Self.embeddingDimension)
        for index in embedding.indices {
            embedding[index] = values[index].floatValue
        }
        guard embedding.allSatisfy(\.isFinite) else {
            throw AudioModelError.inferenceFailed(
                operation: "ReDimNet2 speaker identity",
                reason: "model returned a non-finite embedding")
        }

        let norm = sqrt(embedding.reduce(Float(0)) { $0 + $1 * $1 })
        guard norm > 1e-8 else {
            throw AudioModelError.inferenceFailed(
                operation: "ReDimNet2 speaker identity",
                reason: "model returned a zero embedding")
        }
        for index in embedding.indices {
            embedding[index] /= norm
        }
        return embedding
    }

    public static func cosineSimilarity(_ left: [Float], _ right: [Float]) -> Float {
        guard left.count == right.count, !left.isEmpty else { return 0 }
        var dot: Float = 0
        var leftNorm: Float = 0
        var rightNorm: Float = 0
        for index in left.indices {
            dot += left[index] * right[index]
            leftNorm += left[index] * left[index]
            rightNorm += right[index] * right[index]
        }
        let denominator = sqrt(leftNorm) * sqrt(rightNorm)
        return denominator > 0 ? dot / denominator : 0
    }
}
#endif
