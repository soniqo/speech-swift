import AudioCommon
import Foundation

enum Nemotron3DiarizationError: LocalizedError {
    case missingArtifact(String)
    case invalidConfiguration(String)
    case incompatibleWeights(String)
    case runtime(String)

    var errorDescription: String? {
        switch self {
        case .missingArtifact(let path):
            return "Nemotron 3 local artifact is missing: \(path)"
        case .invalidConfiguration(let reason):
            return "Invalid Nemotron 3 artifact configuration: \(reason)"
        case .incompatibleWeights(let reason):
            return "Incompatible Nemotron 3 weights: \(reason)"
        case .runtime(let reason):
            return "Nemotron 3 inference failed: \(reason)"
        }
    }
}

struct Nemotron3ArtifactConfiguration: Decodable, Sendable {
    struct Quantization: Decodable, Sendable {
        let groupSize: Int?
        let bits: Int?
        let mode: String?
    }

    let modelType: String
    let sourceModel: String
    let sourceRevision: String
    let dtype: String
    let sampleRate: Int
    let nMels: Int
    let dModel: Int
    let tfModel: Int
    let numLayers: Int
    let numHeads: Int
    let numSpeakers: Int
    let subsamplingFactor: Int
    let upsampleFactor: Int
    let spkcacheLen: Int
    let fifoLen: Int
    let chunkLen: Int
    let rightContext: Int
    let spkcacheUpdatePeriod: Int
    let quantization: Quantization

    static func load(from directory: URL) throws -> Self {
        let url = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Nemotron3DiarizationError.missingArtifact(url.path)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let configuration = try decoder.decode(Self.self, from: Data(contentsOf: url))
        try configuration.validate()
        return configuration
    }

    private func validate() throws {
        guard modelType == "nemotron3_diarization",
              sourceModel == "nvidia/Nemotron-3-Diarization",
              sourceRevision == "a435e9867d79e789e90053f9b6d6834053af564a" else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "bundle must use the pinned final Nemotron 3 checkpoint")
        }
        guard dtype == "int8" else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "only the INT8 export is supported")
        }
        guard sampleRate == 16_000, nMels == 128, dModel == 512,
              tfModel == 192, numLayers == 31, numHeads == 8,
              numSpeakers == 8, subsamplingFactor == 8,
              upsampleFactor == 8, spkcacheLen == 264, fifoLen == 40,
              chunkLen == 340, rightContext == 40,
              spkcacheUpdatePeriod == 300 else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "artifact geometry does not match the supported offline graph")
        }
    }
}

struct Nemotron3HeadOutput {
    let probabilities10ms: [Float]
    let probabilities80ms: [Float]
}

protocol Nemotron3InferenceBackend: AnyObject {
    var learnedSilenceEmbedding: [Float] { get }
    func preencode(chunk: [Float]) throws -> [Float]
    func predictHead(packedEmbeddings: [Float], validLength: Int) throws
        -> Nemotron3HeadOutput
}

/// Eight-speaker diarization using the final Nemotron 3 Core ML or MLX export.
/// The model predicts 10 ms speaker activity while keeping cache state at 80 ms.
public final class Nemotron3Diarizer {
    private let backend: Nemotron3InferenceBackend
    private let melExtractor: SortformerMelExtractor
    let config: SortformerConfig
    private var state: SortformerStreamingState
    private let updater: SortformerStateUpdater

    init(backend: Nemotron3InferenceBackend) {
        self.backend = backend
        self.config = .nemotron3Offline
        self.melExtractor = SortformerMelExtractor(config: config)
        self.state = SortformerStreamingState(config: config)
        self.updater = SortformerStateUpdater(
            config: config,
            learnedSilenceEmbedding: backend.learnedSilenceEmbedding
        )
    }

    public func resetState() {
        state.reset()
    }

    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config thresholds: DiarizationConfig = .sortformer
    ) throws -> DiarizationResult {
        let samples = DiarizationHelpers.resample(
            audio, from: sampleRate, to: config.sampleRate)
        guard !samples.isEmpty else { return Self.emptyResult }

        resetState()
        let (melSpec, totalMelFrames) = melExtractor.extract(samples)
        guard totalMelFrames > 0 else { return Self.emptyResult }

        let encoderStride = config.subsamplingFactor
        let coreEncoderFrames = Int(config.chunkLenSeconds)
        let rightEncoderFrames = Int(config.rightContextSeconds)
        let coreMelFrames = coreEncoderFrames * encoderStride
        let fixedChunkMelFrames = (coreEncoderFrames + rightEncoderFrames)
            * encoderStride
        let fixedChunkEmbeddingFrames = coreEncoderFrames + rightEncoderFrames
        let packedCapacity = config.spkcacheLen + config.fifoLen
            + fixedChunkEmbeddingFrames
        let dim = config.fcDModel
        let speakers = config.maxSpeakers

        var allHighResolutionProbabilities: [Float] = []
        allHighResolutionProbabilities.reserveCapacity(totalMelFrames * speakers)
        var startFrame = 0

        while startFrame < totalMelFrames {
            let endFrame = min(startFrame + coreMelFrames, totalMelFrames)
            let rightOffset = min(
                rightEncoderFrames * encoderStride,
                totalMelFrames - endFrame)
            let validChunkMelFrames = endFrame + rightOffset - startFrame

            var chunk = [Float](
                repeating: 0,
                count: fixedChunkMelFrames * config.nMels)
            Self.copy(
                source: melSpec,
                sourceOffset: startFrame * config.nMels,
                destination: &chunk,
                destinationOffset: 0,
                count: validChunkMelFrames * config.nMels)

            let allChunkEmbeddings = try backend.preencode(chunk: chunk)
            let validChunkEmbeddingFrames = min(
                fixedChunkEmbeddingFrames,
                (validChunkMelFrames + encoderStride - 1) / encoderStride)
            guard allChunkEmbeddings.count == fixedChunkEmbeddingFrames * dim else {
                throw Nemotron3DiarizationError.runtime(
                    "pre-encoder returned \(allChunkEmbeddings.count) values")
            }

            let previousSpkcacheLength = state.spkcacheLength
            let previousFifoLength = state.fifoLength
            var packed = [Float](repeating: 0, count: packedCapacity * dim)
            var packedRows = 0
            Self.copy(
                source: state.spkcache,
                sourceOffset: 0,
                destination: &packed,
                destinationOffset: packedRows * dim,
                count: state.spkcacheLength * dim)
            packedRows += state.spkcacheLength
            Self.copy(
                source: state.fifo,
                sourceOffset: 0,
                destination: &packed,
                destinationOffset: packedRows * dim,
                count: state.fifoLength * dim)
            packedRows += state.fifoLength
            Self.copy(
                source: allChunkEmbeddings,
                sourceOffset: 0,
                destination: &packed,
                destinationOffset: packedRows * dim,
                count: validChunkEmbeddingFrames * dim)
            packedRows += validChunkEmbeddingFrames

            let output = try backend.predictHead(
                packedEmbeddings: packed, validLength: packedRows)
            guard output.probabilities80ms.count == packedCapacity * speakers,
                  output.probabilities10ms.count
                    == packedCapacity * encoderStride * speakers else {
                throw Nemotron3DiarizationError.runtime(
                    "head returned an unexpected output shape")
            }

            let rightContext = (rightOffset + encoderStride - 1) / encoderStride
            let validChunkEmbeddings = Array(
                allChunkEmbeddings.prefix(validChunkEmbeddingFrames * dim))
            _ = updater.update(
                state: &state,
                chunkEmbs: validChunkEmbeddings,
                preds: output.probabilities80ms,
                leftContext: 0,
                rightContext: rightContext)

            // High-resolution predictions are laid out over
            // [spkcache | fifo | chunk] at eight 10 ms rows per cache row.
            let highStartFrame = (previousSpkcacheLength + previousFifoLength)
                * encoderStride
            let emittedFrames = endFrame - startFrame
            let highStart = highStartFrame * speakers
            let highEnd = highStart + emittedFrames * speakers
            guard highEnd <= output.probabilities10ms.count else {
                throw Nemotron3DiarizationError.runtime(
                    "10 ms core slice exceeds the head output")
            }
            allHighResolutionProbabilities.append(
                contentsOf: output.probabilities10ms[highStart..<highEnd])
            startFrame = endFrame
        }

        let audioDuration = Float(samples.count) / Float(config.sampleRate)
        let segments = SortformerDiarizer.binarize(
            probs: allHighResolutionProbabilities,
            frameCount: allHighResolutionProbabilities.count / config.maxSpeakers,
            audioDuration: audioDuration,
            config: config,
            thresholds: thresholds)
        return DiarizationResult(
            segments: segments,
            numSpeakers: Set(segments.map(\.speakerId)).count,
            speakerEmbeddings: [])
    }

    private static let emptyResult = DiarizationResult(
        segments: [], numSpeakers: 0, speakerEmbeddings: [])

    private static func copy(
        source: [Float],
        sourceOffset: Int,
        destination: inout [Float],
        destinationOffset: Int,
        count: Int
    ) {
        guard count > 0 else { return }
        precondition(sourceOffset >= 0 && sourceOffset + count <= source.count)
        precondition(destinationOffset >= 0 && destinationOffset + count <= destination.count)
        destination.withUnsafeMutableBufferPointer { target in
            source.withUnsafeBufferPointer { input in
                target.baseAddress!.advanced(by: destinationOffset).update(
                    from: input.baseAddress!.advanced(by: sourceOffset),
                    count: count)
            }
        }
    }
}
