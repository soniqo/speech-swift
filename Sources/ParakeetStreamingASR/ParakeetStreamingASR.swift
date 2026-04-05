import CoreML
import Foundation
import AudioCommon

/// Parakeet EOU 120M — streaming ASR with end-of-utterance detection on CoreML.
///
/// Uses a cache-aware FastConformer encoder (RNNT) that processes audio in chunks,
/// maintaining encoder state between calls. The RNNT joint network emits an `<EOU>`
/// token when it detects the speaker has finished an utterance.
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public class ParakeetStreamingASRModel {
    public let config: ParakeetEOUConfig

    public static let defaultModelId = "aufklarer/Parakeet-EOU-120M-CoreML-INT8"

    var _isLoaded = true
    private let melPreprocessor: StreamingMelPreprocessor
    var encoder: MLModel?
    var decoder: MLModel?
    var joint: MLModel?
    private let vocabulary: ParakeetEOUVocabulary

    private init(
        config: ParakeetEOUConfig,
        encoder: MLModel?,
        decoder: MLModel?,
        joint: MLModel?,
        vocabulary: ParakeetEOUVocabulary
    ) {
        self.config = config
        self.melPreprocessor = StreamingMelPreprocessor(config: config)
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.vocabulary = vocabulary
    }

    // MARK: - Streaming Session

    /// A partial transcript from streaming recognition.
    public struct PartialTranscript: Sendable {
        /// Current best transcript text
        public let text: String
        /// Whether this is a final, committed transcript (EOU detected or stream ended)
        public let isFinal: Bool
        /// Confidence score (0.0–1.0)
        public let confidence: Float
        /// Whether end-of-utterance was detected
        public let eouDetected: Bool
        /// Segment index (increments on each final transcript)
        public let segmentIndex: Int
    }

    /// Create a new streaming session.
    ///
    /// Push audio chunks via `pushAudio(_:)` and receive partial transcripts.
    /// Call `finalize()` when done to get any remaining text.
    public func createSession() throws -> StreamingSession {
        guard _isLoaded, let encoder, let decoder, let joint else {
            throw AudioModelError.inferenceFailed(operation: "createSession", reason: "Model not loaded")
        }
        return try StreamingSession(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary,
            melPreprocessor: melPreprocessor
        )
    }

    /// Convenience: stream transcription from an audio buffer, yielding partial results.
    public func transcribeStream(
        audio: [Float],
        sampleRate: Int,
        chunkDuration: Float? = nil
    ) -> AsyncStream<PartialTranscript> {
        let chunkMs = chunkDuration.map { Int($0 * 1000) } ?? config.streaming.chunkMs

        return AsyncStream { continuation in
            Task {
                do {
                    // Resample if needed
                    let samples: [Float]
                    if sampleRate != self.config.sampleRate {
                        samples = AudioFileLoader.resample(audio, from: sampleRate, to: self.config.sampleRate)
                    } else {
                        samples = audio
                    }
                    let actualSamplesPerChunk = chunkMs * self.config.sampleRate / 1000

                    let session = try self.createSession()
                    var offset = 0

                    while offset < samples.count {
                        let end = min(offset + actualSamplesPerChunk, samples.count)
                        let chunk = Array(samples[offset..<end])
                        let partials = try session.pushAudio(chunk)
                        for partial in partials {
                            continuation.yield(partial)
                        }
                        offset = end
                    }

                    // Finalize
                    let finals = try session.finalize()
                    for partial in finals {
                        continuation.yield(partial)
                    }
                    continuation.finish()
                } catch {
                    AudioLog.inference.error("Streaming transcription failed: \(error)")
                    continuation.finish()
                }
            }
        }
    }

    // MARK: - Batch Transcription (fallback)

    /// Transcribe full audio buffer (non-streaming).
    public func transcribeAudio(_ audio: [Float], sampleRate: Int, language: String? = nil) throws -> String {
        let samples: [Float]
        if sampleRate != config.sampleRate {
            samples = AudioFileLoader.resample(audio, from: sampleRate, to: config.sampleRate)
        } else {
            samples = audio
        }

        let session = try createSession()
        // Collect all partials — EOU may fire during pushAudio, clearing allTokens
        var allPartials = try session.pushAudio(samples)
        allPartials.append(contentsOf: try session.finalize())

        // Return the last final transcript, or the last partial if no final
        if let lastFinal = allPartials.last(where: { $0.isFinal }) {
            return lastFinal.text
        }
        return allPartials.last?.text ?? ""
    }

    // MARK: - Warmup

    public func warmUp() throws {
        let dummyAudio = [Float](repeating: 0, count: config.sampleRate)
        _ = try transcribeAudio(dummyAudio, sampleRate: config.sampleRate)
    }

    // MARK: - Model Loading

    public static func fromPretrained(
        modelId: String? = nil,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> ParakeetStreamingASRModel {
        let effectiveModelId = modelId ?? defaultModelId
        AudioLog.modelLoading.info("Loading Parakeet EOU model: \(effectiveModelId)")

        let cacheDir: URL
        do {
            cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: effectiveModelId)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Failed to resolve cache directory", underlying: error)
        }

        progressHandler?(0.0, "Downloading model...")
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: effectiveModelId,
                to: cacheDir,
                additionalFiles: [
                    "encoder.mlmodelc/**",
                    "decoder.mlmodelc/**",
                    "joint.mlmodelc/**",
                    "vocab.json",
                    "config.json",
                ]
            ) { fraction in
                progressHandler?(fraction * 0.7, "Downloading model...")
            }
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: effectiveModelId, reason: "Download failed", underlying: error)
        }

        progressHandler?(0.70, "Loading configuration...")
        let config: ParakeetEOUConfig
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            let data = try Data(contentsOf: configURL)
            config = try JSONDecoder().decode(ParakeetEOUConfig.self, from: data)
        } else {
            config = .default
        }

        progressHandler?(0.75, "Loading vocabulary...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        let vocabulary = try ParakeetEOUVocabulary.load(from: vocabURL)

        progressHandler?(0.80, "Loading CoreML models...")
        let encoder = try loadCoreMLModel(name: "encoder", from: cacheDir, computeUnits: .cpuAndGPU)
        progressHandler?(0.90, "Loading decoder...")
        let decoder = try loadCoreMLModel(name: "decoder", from: cacheDir, computeUnits: .cpuAndGPU)
        progressHandler?(0.95, "Loading joint network...")
        let joint = try loadCoreMLModel(name: "joint", from: cacheDir, computeUnits: .cpuAndGPU)

        progressHandler?(1.0, "Model loaded")
        AudioLog.modelLoading.info("Parakeet EOU model loaded (\(vocabulary.count) tokens)")

        return ParakeetStreamingASRModel(
            config: config,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            vocabulary: vocabulary
        )
    }

    private static func loadCoreMLModel(
        name: String,
        from directory: URL,
        computeUnits: MLComputeUnits
    ) throws -> MLModel {
        let modelURL = directory.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: name, reason: "CoreML model not found at \(modelURL.path)")
        }
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        return try MLModel(contentsOf: modelURL, configuration: mlConfig)
    }
}
