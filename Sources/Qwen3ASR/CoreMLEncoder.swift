#if canImport(CoreML)
import CoreML
import Foundation
import MLX
import AudioCommon

/// CoreML audio encoder for Qwen3-ASR.
///
/// Runs the audio encoder on Neural Engine via CoreML instead of GPU via MLX.
/// Produces audio embeddings that feed into the MLX text decoder. This enables
/// lower power consumption on macOS and is a step toward full iOS deployment.
///
/// The encoder uses a single fixed 30 s mel shape ``[1, 128, 3000]`` and
/// applies upstream's chunked block-attention (100-frame chunks → 13 tokens
/// each, 8-chunk attention windows). Mel input is zero-padded to 3000 frames
/// and the real length is signaled via a separate ``mel_length`` input so
/// the in-graph block-attention bias can mask out the padded frames; the
/// model returns the matching real audio-token count via ``output_length``.
public class CoreMLASREncoder {
    private let model: MLModel
    /// Fixed mel length the chunked-attention encoder is exported with.
    /// 3000 mel frames = 30 s @ 100 Hz hop, matching upstream training.
    public static let paddedMelLength: Int = 3000
    /// Max audio tokens out of the padded encoder (3000 mel / 8 conv stride ≈
    /// 30 chunks × 13 tokens). The model writes the real count to ``output_length``.
    public static let paddedAudioTokens: Int = 390

    public static let defaultModelId = "aufklarer/Qwen3-ASR-CoreML"

    /// Embeddings + the real, un-padded audio-token count (from the model's
    /// ``output_length`` output). Callers should iterate only the first
    /// ``outputLength`` tokens of ``embeddings``.
    public struct EncodedAudio {
        public let embeddings: MLMultiArray
        public let outputLength: Int
    }

    public init(model: MLModel) {
        self.model = model
    }

    /// Load encoder from a directory containing `encoder.mlmodelc`.
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .all)
    ) throws -> CoreMLASREncoder {
        let modelURL = directory.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: "encoder",
                reason: "CoreML encoder not found at \(modelURL.path)")
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        return CoreMLASREncoder(model: model)
    }

    /// Load encoder from HuggingFace.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        computeUnits: MLComputeUnits = CoreMLComputeUnitsResolver.resolved(default: .all),
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CoreMLASREncoder {
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        progressHandler?(0.0, "Downloading CoreML encoder...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["encoder.mlmodelc/**", "config.json"],
            offlineMode: offlineMode
        ) { fraction in
            progressHandler?(fraction * 0.8, "Downloading CoreML encoder...")
        }

        progressHandler?(0.9, "Loading CoreML encoder...")
        let encoder = try load(from: cacheDir, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return encoder
    }

    /// Warm up the encoder with a short dummy input to trigger CoreML compilation.
    public func warmUp() throws {
        // Use a small fake length so warmup doesn't depend on a real clip.
        _ = try encodeRaw(melData: [Float](repeating: 0, count: 128 * Self.paddedMelLength),
                          melBins: 128, realFrames: 100)
    }

    /// Encode a mel spectrogram and return embeddings as an MLXArray.
    ///
    /// - Parameter melFeatures: Mel spectrogram as MLXArray `[128, T]`
    /// - Returns: `(embeddings: [1, paddedAudioTokens, 1024], outputLength)`
    public func encode(_ melFeatures: MLXArray) throws -> (embeddings: MLXArray, outputLength: Int) {
        let melBins = melFeatures.dim(0)
        let melTime = melFeatures.dim(1)
        let melData: [Float] = melFeatures.asArray(Float.self)
        let raw = try encodeRaw(melData: melData, melBins: melBins, realFrames: melTime)
        return (multiArrayToMLXArray(raw.embeddings), raw.outputLength)
    }

    // MARK: - MLX-free encoding (for iOS background / pure CoreML path)

    /// Encode mel spectrogram to audio embeddings without any MLXArray dependency.
    ///
    /// Accepts raw `[Float]` mel data in `[melBins, timeFrames]` layout (the same
    /// layout produced by `WhisperFeatureExtractor.extractFeaturesRaw`).
    /// Returns the encoder output as `MLMultiArray` directly, avoiding the
    /// Metal GPU eval that `MLXArray` would trigger.
    ///
    /// - Parameters:
    ///   - melData: Flat float array in row-major `[melBins, timeFrames]` order
    ///   - melBins: Number of mel frequency bins (typically 128)
    ///   - timeFrames: Number of time frames
    /// - Returns: Audio embeddings as `MLMultiArray` with shape `[1, T/8, 1024]`
    public func encode(melData: [Float], melBins: Int, timeFrames: Int) throws -> EncodedAudio {
        return try encodeRaw(melData: melData, melBins: melBins, realFrames: timeFrames)
    }

    /// Convenience: encode a `MelFeatures` struct directly.
    public func encode(melFeatures: MelFeatures) throws -> EncodedAudio {
        return try encodeRaw(melData: melFeatures.data,
                             melBins: melFeatures.melBins,
                             realFrames: melFeatures.timeFrames)
    }

    /// Shared core: zero-pads ``melData`` to the fixed ``paddedMelLength``,
    /// runs the two-input/two-output graph, and returns the model's reported
    /// ``output_length`` alongside the full padded embeddings.
    private func encodeRaw(
        melData: [Float], melBins: Int, realFrames: Int
    ) throws -> EncodedAudio {
        let padded = Self.paddedMelLength
        guard realFrames <= padded else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder",
                reason: "Audio too long: \(realFrames) mel frames exceeds fixed shape \(padded). Segment with SpeechVAD or process in 30s windows.")
        }
        // Mel input: [1, melBins, paddedMelLength], zero-padded past realFrames.
        let melArray = try MLMultiArray(
            shape: [1, melBins as NSNumber, padded as NSNumber], dataType: .float32)
        let mptr = melArray.dataPointer.assumingMemoryBound(to: Float.self)
        for bin in 0..<melBins {
            let src = bin * realFrames
            let dst = bin * padded
            for t in 0..<realFrames { mptr[dst + t] = melData[src + t] }
            for t in realFrames..<padded { mptr[dst + t] = 0 }
        }
        // mel_length input: [1] int32 with the real (un-padded) frame count.
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: Int32(realFrames))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArray),
            "mel_length": MLFeatureValue(multiArray: lengthArray),
        ])
        let output = try model.prediction(from: input)

        guard let embeddings = output.featureValue(for: "audio_embeddings")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder", reason: "Missing audio_embeddings output")
        }
        guard let lengthOut = output.featureValue(for: "output_length")?.multiArrayValue else {
            throw AudioModelError.inferenceFailed(
                operation: "CoreML encoder", reason: "Missing output_length output (encoder may be an older export without the chunked-attention mask)")
        }
        let outLen = max(0, Int(lengthOut[0].int32Value))
        return EncodedAudio(embeddings: embeddings, outputLength: outLen)
    }

    private func multiArrayToMLXArray(_ array: MLMultiArray) -> MLXArray {
        let shape = array.shape.map { $0.intValue }
        let count = array.count

        switch array.dataType {
        case .float16:
            let src = array.dataPointer.assumingMemoryBound(to: Float16.self)
            var floats = [Float](repeating: 0, count: count)
            for i in 0..<count { floats[i] = Float(src[i]) }
            return MLXArray(floats, shape)
        case .float32:
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            return MLXArray(Array(UnsafeBufferPointer(start: src, count: count)), shape)
        default:
            let src = array.dataPointer.assumingMemoryBound(to: Float.self)
            return MLXArray(Array(UnsafeBufferPointer(start: src, count: count)), shape)
        }
    }
}

// MARK: - Qwen3ASRModel Integration

extension Qwen3ASRModel {
    /// Transcribe audio using CoreML encoder + MLX text decoder.
    ///
    /// This hybrid approach runs the encoder on Neural Engine (CoreML) and the
    /// text decoder on GPU (MLX), combining the power efficiency of ANE with
    /// the flexibility of MLX for autoregressive decoding.
    public func transcribe(
        audio: [Float],
        sampleRate: Int = 16000,
        language: String? = nil,
        maxTokens: Int = 448,
        coremlEncoder: CoreMLASREncoder
    ) throws -> String {
        let melFeatures = featureExtractor.process(audio, sampleRate: sampleRate)

        // CoreML encoder returns the padded `[1, paddedAudioTokens, 1024]`
        // embeddings + the real audio-token count via ``outputLength``.
        let (audioEmbeds, audioLength) = try coremlEncoder.encode(melFeatures)

        guard let textDecoder = textDecoder else {
            return "[CoreML encoded: \(audioEmbeds.shape) length=\(audioLength)] - Text decoder not loaded"
        }

        // Slice the embeddings to the real (un-padded) length before passing
        // to the MLX text decoder, otherwise trailing zero-derived tokens
        // would pollute decoder cross-attention.
        let realEmbeds = audioEmbeds[0..., 0..<audioLength, 0...]
        return generateText(
            audioEmbeds: realEmbeds,
            textDecoder: textDecoder,
            language: language,
            maxTokens: maxTokens
        )
    }
}
#endif
