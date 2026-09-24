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

    /// Open an independent recording-wide stream. The fixed Core ML/MLX graph
    /// is padded, but only completed core frames advance the speaker cache.
    public func makeStreamingSession(
        coreEncoderFrames: Int = 6,
        rightContextEncoderFrames: Int = 7
    ) throws -> Nemotron3StreamingSession {
        try Nemotron3StreamingSession(
            backend: backend,
            coreEncoderFrames: coreEncoderFrames,
            rightContextEncoderFrames: rightContextEncoderFrames)
    }

    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config thresholds: DiarizationConfig = .sortformer,
        coreEncoderFrames: Int = 340,
        rightContextEncoderFrames: Int = 40
    ) throws -> DiarizationResult {
        guard (1...340).contains(coreEncoderFrames),
              (0...40).contains(rightContextEncoderFrames),
              coreEncoderFrames + rightContextEncoderFrames <= 380 else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "chunk geometry exceeds the fixed Core ML/MLX pre-encoder input")
        }
        let samples = DiarizationHelpers.resample(
            audio, from: sampleRate, to: config.sampleRate)
        guard !samples.isEmpty else { return Self.emptyResult }

        resetState()
        let (melSpec, totalMelFrames) = melExtractor.extract(samples)
        guard totalMelFrames > 0 else { return Self.emptyResult }

        let encoderStride = config.subsamplingFactor
        let rightEncoderFrames = rightContextEncoderFrames
        let coreMelFrames = coreEncoderFrames * encoderStride
        // The compiled graph is fixed at 340 core + 40 right-context frames;
        // shorter experimental cadences still pad its input to that shape.
        let fixedChunkMelFrames = 380 * encoderStride
        let fixedChunkEmbeddingFrames = 380
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

            let allChunkEmbeddings = try autoreleasepool {
                try backend.preencode(chunk: chunk)
            }
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

            let output = try autoreleasepool {
                try backend.predictHead(
                    packedEmbeddings: packed, validLength: packedRows)
            }
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

    fileprivate static func copy(
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

/// Incremental Nemotron 3 diarization with recording-local, stable cache slots.
/// Input is 16 kHz mono PCM. A complete snapshot is returned whenever a core
/// window and its lookahead have arrived; shorter pushes reuse the last result.
public final class Nemotron3StreamingSession {
    private let backend: Nemotron3InferenceBackend
    private let config: SortformerConfig = .nemotron3Offline
    private let melExtractor: SortformerMelExtractor
    private let updater: SortformerStateUpdater
    private var state: SortformerStreamingState
    private let coreEncoderFrames: Int
    private let rightContextEncoderFrames: Int
    private var pcm: [Float] = []
    private var pcmBaseSample = 0
    private var totalSamples = 0
    private var nextMelFrame = 0
    private var confirmedProbabilities: [Float] = []
    private var processedProbabilityFrames = 0
    private var closedSegments = [[DiarizedSegment]](
        repeating: [], count: 8)
    private var openSpeechStarts = [Int?](repeating: nil, count: 8)
    private var cachedResult = DiarizationResult(
        segments: [], numSpeakers: 0, speakerEmbeddings: [])
    private var resultDirty = false
    private var thresholdsDirty = false
    private var finished = false

    public var binarization: DiarizationConfig = .sortformer {
        didSet {
            thresholdsDirty = true
            resultDirty = true
        }
    }
    public private(set) var confirmedThroughSample = 0

    private static let marginMelFrames = 2

    fileprivate init(
        backend: Nemotron3InferenceBackend,
        coreEncoderFrames: Int,
        rightContextEncoderFrames: Int
    ) throws {
        guard (1...340).contains(coreEncoderFrames),
              (0...40).contains(rightContextEncoderFrames),
              coreEncoderFrames + rightContextEncoderFrames <= 380 else {
            throw Nemotron3DiarizationError.invalidConfiguration(
                "stream chunk geometry exceeds the fixed pre-encoder input")
        }
        self.backend = backend
        self.coreEncoderFrames = coreEncoderFrames
        self.rightContextEncoderFrames = rightContextEncoderFrames
        melExtractor = SortformerMelExtractor(config: config)
        updater = SortformerStateUpdater(
            config: config,
            learnedSilenceEmbedding: backend.learnedSilenceEmbedding)
        state = SortformerStreamingState(config: config)
    }

    public func reset() {
        state.reset()
        pcm.removeAll(keepingCapacity: true)
        pcmBaseSample = 0
        totalSamples = 0
        nextMelFrame = 0
        confirmedProbabilities.removeAll(keepingCapacity: true)
        processedProbabilityFrames = 0
        closedSegments = [[DiarizedSegment]](
            repeating: [], count: config.maxSpeakers)
        openSpeechStarts = [Int?](
            repeating: nil, count: config.maxSpeakers)
        cachedResult = DiarizationResult(
            segments: [], numSpeakers: 0, speakerEmbeddings: [])
        confirmedThroughSample = 0
        resultDirty = false
        thresholdsDirty = false
        finished = false
    }

    @discardableResult
    public func push(audio: [Float]) throws -> DiarizationResult {
        precondition(!finished, "push(audio:) after finish(); call reset() first")
        pcm.append(contentsOf: audio)
        totalSamples += audio.count
        try drain(flushing: false)
        return currentResult()
    }

    public func finish() throws -> DiarizationResult {
        guard !finished else { return currentResult() }
        if totalSamples > 0 {
            try drain(flushing: true)
        }
        finished = true
        return currentResult()
    }

    public func currentResult() -> DiarizationResult {
        guard resultDirty else { return cachedResult }
        if thresholdsDirty {
            processedProbabilityFrames = 0
            closedSegments = [[DiarizedSegment]](
                repeating: [], count: config.maxSpeakers)
            openSpeechStarts = [Int?](
                repeating: nil, count: config.maxSpeakers)
            thresholdsDirty = false
        }
        consumeNewProbabilities()
        let frameDuration = Float(config.hopLength)
            / Float(config.sampleRate)
        let frames = confirmedProbabilities.count / config.maxSpeakers
        var segments = closedSegments.flatMap { $0 }
        for speaker in 0..<config.maxSpeakers {
            guard let start = openSpeechStarts[speaker] else { continue }
            let startTime = Float(start) * frameDuration
            let endTime = Float(frames) * frameDuration
            guard endTime - startTime >= binarization.minSpeechDuration else {
                continue
            }
            let provisional = DiarizedSegment(
                startTime: startTime,
                endTime: min(
                    endTime,
                    Float(confirmedThroughSample) / Float(config.sampleRate)),
                speakerId: speaker)
            if let previous = closedSegments[speaker].last,
               provisional.startTime - previous.endTime
                    < binarization.minSilenceDuration {
                segments.removeAll { segment in
                    segment.speakerId == speaker
                        && segment.startTime == previous.startTime
                        && segment.endTime == previous.endTime
                }
                segments.append(DiarizedSegment(
                    startTime: previous.startTime,
                    endTime: provisional.endTime,
                    speakerId: speaker))
            } else {
                segments.append(provisional)
            }
        }
        segments.sort { $0.startTime < $1.startTime }
        segments = DiarizationHelpers.compactSpeakerIds(segments)
        cachedResult = DiarizationResult(
            segments: segments,
            numSpeakers: Set(segments.map(\.speakerId)).count,
            speakerEmbeddings: [])
        resultDirty = false
        return cachedResult
    }

    private func consumeNewProbabilities() {
        let speakers = config.maxSpeakers
        let frames = confirmedProbabilities.count / speakers
        let frameDuration = Float(config.hopLength)
            / Float(config.sampleRate)
        guard processedProbabilityFrames < frames else { return }
        for frame in processedProbabilityFrames..<frames {
            let time = Float(frame) * frameDuration
            for speaker in 0..<speakers {
                var probability = confirmedProbabilities[
                    frame * speakers + speaker]
                if probability > 1 || probability < 0 {
                    probability = 1 / (1 + exp(-probability))
                }
                if openSpeechStarts[speaker] == nil {
                    if probability >= binarization.onset {
                        openSpeechStarts[speaker] = frame
                    }
                } else if probability < binarization.offset {
                    let startTime = Float(openSpeechStarts[speaker]!)
                        * frameDuration
                    openSpeechStarts[speaker] = nil
                    guard time - startTime >= binarization.minSpeechDuration else {
                        continue
                    }
                    if let previousIndex = closedSegments[speaker].indices.last,
                       startTime - closedSegments[speaker][previousIndex].endTime
                            < binarization.minSilenceDuration {
                        let previous = closedSegments[speaker][previousIndex]
                        closedSegments[speaker][previousIndex] = DiarizedSegment(
                            startTime: previous.startTime,
                            endTime: time,
                            speakerId: speaker)
                    } else {
                        closedSegments[speaker].append(DiarizedSegment(
                            startTime: startTime,
                            endTime: time,
                            speakerId: speaker))
                    }
                }
            }
        }
        processedProbabilityFrames = frames
    }

    public static func firstConfirmationSampleCount(
        coreEncoderFrames: Int = 6,
        rightContextEncoderFrames: Int = 7
    ) -> Int {
        (coreEncoderFrames + rightContextEncoderFrames) * 8 * 160
            + marginMelFrames * 160
    }

    private func drain(flushing: Bool) throws {
        let hop = config.hopLength
        let stride = config.subsamplingFactor
        let coreMelFrames = coreEncoderFrames * stride
        let rightMelFrames = rightContextEncoderFrames * stride
        let totalMelFrames = flushing
            ? totalSamples / hop + 1
            : totalSamples / hop
        while nextMelFrame < totalMelFrames {
            let endMelFrame = min(nextMelFrame + coreMelFrames, totalMelFrames)
            let rightAvailable = totalMelFrames - endMelFrame
            if !flushing {
                guard endMelFrame - nextMelFrame == coreMelFrames,
                      rightAvailable >= rightMelFrames + Self.marginMelFrames else {
                    break
                }
            }
            let rightOffset = min(rightMelFrames, rightAvailable)
            try runChunk(
                startMelFrame: nextMelFrame,
                endMelFrame: endMelFrame,
                rightOffset: rightOffset)
            nextMelFrame = endMelFrame
            confirmedThroughSample = min(totalSamples, nextMelFrame * hop)
            resultDirty = true
        }
        let neededFromSample = max(
            0, nextMelFrame - Self.marginMelFrames) * hop
        if neededFromSample > pcmBaseSample {
            let count = min(pcm.count, neededFromSample - pcmBaseSample)
            pcm.removeFirst(count)
            pcmBaseSample += count
        }
    }

    private func runChunk(
        startMelFrame: Int,
        endMelFrame: Int,
        rightOffset: Int
    ) throws {
        let hop = config.hopLength
        let stride = config.subsamplingFactor
        let melCount = config.nMels
        let speakers = config.maxSpeakers
        let margin = Self.marginMelFrames
        let extractStartMel = max(0, startMelFrame - margin)
        let extractStartSample = extractStartMel * hop
        let extractEndSample = min(
            totalSamples,
            (endMelFrame + rightOffset + margin) * hop + config.nFFT)
        let sliceStart = extractStartSample - pcmBaseSample
        let sliceEnd = extractEndSample - pcmBaseSample
        guard sliceStart >= 0, sliceEnd <= pcm.count, sliceEnd > sliceStart else {
            throw Nemotron3DiarizationError.runtime(
                "stream mel span is no longer present in the PCM buffer")
        }
        let (mel, extractedFrames) = melExtractor.extract(
            Array(pcm[sliceStart..<sliceEnd]))
        let localStart = startMelFrame - extractStartMel
        let localEnd = min(
            extractedFrames,
            endMelFrame + rightOffset - extractStartMel)
        guard localEnd > localStart else {
            throw Nemotron3DiarizationError.runtime(
                "stream mel extraction returned no frames")
        }
        let validMelFrames = localEnd - localStart
        var chunk = [Float](repeating: 0, count: 3_040 * melCount)
        Nemotron3Diarizer.copy(
            source: mel,
            sourceOffset: localStart * melCount,
            destination: &chunk,
            destinationOffset: 0,
            count: validMelFrames * melCount)
        let allEmbeddings = try autoreleasepool {
            try backend.preencode(chunk: chunk)
        }
        let validEmbeddingFrames = min(380, (validMelFrames + stride - 1) / stride)
        let dim = config.fcDModel
        guard allEmbeddings.count == 380 * dim else {
            throw Nemotron3DiarizationError.runtime(
                "stream pre-encoder returned an unexpected shape")
        }
        let previousCache = state.spkcacheLength
        let previousFifo = state.fifoLength
        var packed = [Float](repeating: 0, count: 684 * dim)
        var packedRows = 0
        for (source, rows) in [
            (state.spkcache, previousCache),
            (state.fifo, previousFifo),
            (allEmbeddings, validEmbeddingFrames),
        ] {
            Nemotron3Diarizer.copy(
                source: source,
                sourceOffset: 0,
                destination: &packed,
                destinationOffset: packedRows * dim,
                count: rows * dim)
            packedRows += rows
        }
        let output = try autoreleasepool {
            try backend.predictHead(
                packedEmbeddings: packed, validLength: packedRows)
        }
        guard output.probabilities80ms.count == 684 * speakers,
              output.probabilities10ms.count == 684 * stride * speakers else {
            throw Nemotron3DiarizationError.runtime(
                "stream head returned an unexpected shape")
        }
        let highStart = (previousCache + previousFifo) * stride * speakers
        let coreHighCount = (endMelFrame - startMelFrame) * speakers
        guard highStart + coreHighCount <= output.probabilities10ms.count else {
            throw Nemotron3DiarizationError.runtime(
                "stream core slice exceeds head output")
        }
        confirmedProbabilities.append(contentsOf:
            output.probabilities10ms[
                highStart..<(highStart + coreHighCount)])
        let rightEncoderFrames = (rightOffset + stride - 1) / stride
        _ = updater.update(
            state: &state,
            chunkEmbs: Array(allEmbeddings.prefix(validEmbeddingFrames * dim)),
            preds: output.probabilities80ms,
            leftContext: 0,
            rightContext: rightEncoderFrames)
    }
}
