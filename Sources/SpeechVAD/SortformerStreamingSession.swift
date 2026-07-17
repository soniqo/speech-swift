#if canImport(CoreML)
import CoreML
import Foundation

/// Incremental Sortformer diarization over a live audio stream.
///
/// Feed PCM in arbitrary sizes with ``push(audio:)``; whenever enough audio
/// for one streaming chunk (480 ms core + 560 ms lookahead) has accumulated,
/// the session runs one CoreML step, advances the Arrival-Order Speaker Cache
/// via ``SortformerStateUpdater``, and appends the newly confirmed frames to
/// the running result. Speaker IDs are cache slots ordered by first arrival,
/// so they remain stable for the lifetime of the session — no window
/// re-clustering and no renumbering.
///
/// ```swift
/// let session = try await SortformerStreamingSession.fromPretrained()
/// while let samples = captureNextBuffer() {
///     let snapshot = try session.push(audio: samples)
///     render(snapshot.segments)
/// }
/// let final = try session.finish()
/// ```
///
/// The session expects 16 kHz mono input (`SortformerConfig.sampleRate`).
/// Results are complete snapshots: every call returns all segments from the
/// start of the stream, mirroring how downstream consumers replace rather
/// than append transcript state.
public final class SortformerStreamingSession {

    private let model: SortformerCoreMLModel
    private let config: SortformerConfig
    private let updater: SortformerStateUpdater
    private let melExtractor: SortformerMelExtractor
    private var state: SortformerStreamingState

    /// Thresholds used when snapshots binarize the confirmed-frame history.
    /// Adjustable per session so consumers can trade misses against false
    /// alarms without reloading the model.
    public var binarization: DiarizationConfig = .default

    /// PCM retained for mel extraction, with `pcmBaseSample` giving the
    /// absolute index of `pcm[0]`. Only the span still needed for the next
    /// chunk (plus reflect-padding margin) is kept.
    private var pcm: [Float] = []
    private var pcmBaseSample = 0
    private var totalSamples = 0

    /// Next un-emitted core chunk start, in mel frames.
    private var sttFeat = 0
    /// Confirmed per-frame speaker probabilities from the stream start, flat
    /// `[confirmedFrames * maxSpeakers]`. 80 ms per frame — small enough to
    /// retain for hours and re-binarize per snapshot.
    private var confirmedProbs: [Float] = []
    private var finished = false

    /// Mel frames of extra margin on each side of an extracted span so the
    /// extractor's reflect padding cannot perturb the frames actually used.
    /// 2 frames × 160 samples ≥ the 200-sample pad of a 400-sample window.
    private static let melMarginFrames = 2

    init(model: SortformerCoreMLModel, config: SortformerConfig) {
        self.model = model
        self.config = config
        self.updater = SortformerStateUpdater(config: config)
        self.melExtractor = SortformerMelExtractor(config: config)
        self.state = SortformerStreamingState(config: config)
    }

    /// Download (or reuse) the streaming Sortformer variant and open a session.
    public static func fromPretrained(
        modelId: String = SortformerDiarizer.defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        config: SortformerConfig = .streaming,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SortformerStreamingSession {
        let diarizer = try await SortformerDiarizer.fromPretrained(
            modelId: modelId,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            config: config,
            computeUnits: computeUnits,
            progressHandler: progressHandler)
        return diarizer.makeStreamingSession()
    }

    /// Discard all stream state and start a fresh session.
    public func reset() {
        state = SortformerStreamingState(config: config)
        pcm.removeAll(keepingCapacity: true)
        pcmBaseSample = 0
        totalSamples = 0
        sttFeat = 0
        confirmedProbs.removeAll(keepingCapacity: true)
        finished = false
    }

    /// Ingest more audio and return the updated whole-stream snapshot.
    ///
    /// - Parameter audio: 16 kHz mono PCM. Any length, including empty.
    /// - Returns: all segments confirmed so far (lookahead means the last
    ///   ~1 s of pushed audio is still pending the next chunk).
    @discardableResult
    public func push(audio: [Float]) throws -> DiarizationResult {
        precondition(!finished, "push(audio:) after finish(); call reset() first")
        pcm.append(contentsOf: audio)
        totalSamples += audio.count
        try drainReadyChunks(flushing: false)
        return currentResult()
    }

    /// Flush the remaining buffered audio (with a shrinking right context,
    /// exactly like offline chunking at end of file) and return the final
    /// snapshot. The session must be ``reset()`` before further pushes.
    public func finish() throws -> DiarizationResult {
        guard !finished else { return currentResult() }
        finished = true
        try drainReadyChunks(flushing: true)
        return currentResult()
    }

    /// The whole-stream result at this instant.
    public func currentResult() -> DiarizationResult {
        let frames = confirmedProbs.count / config.maxSpeakers
        guard frames > 0 else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }
        let duration = Float(frames * config.subsamplingFactor)
            * Float(config.hopLength) / Float(config.sampleRate)
        let segments = SortformerDiarizer.binarize(
            probs: confirmedProbs,
            frameCount: frames,
            audioDuration: duration,
            config: config,
            thresholds: binarization)
        let usedSpeakers = Set(segments.map(\.speakerId))
        return DiarizationResult(
            segments: segments,
            numSpeakers: usedSpeakers.count,
            speakerEmbeddings: [])
    }

    // MARK: - Chunk loop

    private func drainReadyChunks(flushing: Bool) throws {
        let hop = config.hopLength
        let nMels = config.nMels
        let subFactor = config.subsamplingFactor
        let coreMelFrames = Int(config.chunkLenSeconds) * subFactor
        let leftCtxMel = Int(config.leftContextSeconds) * subFactor
        let rightCtxMel = Int(config.rightContextSeconds) * subFactor
        let margin = Self.melMarginFrames

        // Mel frame k centers on sample k*hop; a frame is only bit-identical
        // to whole-file extraction when the extracted span extends `margin`
        // frames past it on both sides (or hits the true stream start/end).
        // Whole-file extraction yields N/hop + 1 center-padded frames; the
        // final frame reflects off the true stream end, so it only exists
        // once the stream is finished — before that it would change when
        // more audio arrives.
        let totalMelFrames = finished
            ? totalSamples / hop + 1
            : totalSamples / hop
        while !finished || sttFeat < totalMelFrames {
            let endFeat = min(sttFeat + coreMelFrames, totalMelFrames)
            let rightAvailable = totalMelFrames - endFeat
            let rightOffset = min(rightCtxMel, rightAvailable)
            if flushing {
                guard sttFeat < totalMelFrames else { break }
            } else {
                // Wait until a full core chunk, its whole lookahead, and the
                // extraction margin are available.
                guard endFeat - sttFeat == coreMelFrames,
                      rightAvailable >= rightCtxMel + margin else { break }
            }
            let leftOffset = min(leftCtxMel, sttFeat)
            try runChunk(
                sttFeat: sttFeat,
                endFeat: endFeat,
                leftOffset: leftOffset,
                rightOffset: rightOffset)
            sttFeat = endFeat
            if flushing, endFeat >= totalMelFrames { break }
        }

        // Drop PCM that no future chunk (or its margin) can reference.
        let neededFromMelFrame = max(0, sttFeat - leftCtxMel - margin)
        let neededFromSample = neededFromMelFrame * hop
        if neededFromSample > pcmBaseSample {
            let drop = min(pcm.count, neededFromSample - pcmBaseSample)
            pcm.removeFirst(drop)
            pcmBaseSample += drop
        }
        _ = nMels
    }

    private func runChunk(
        sttFeat: Int, endFeat: Int, leftOffset: Int, rightOffset: Int
    ) throws {
        let hop = config.hopLength
        let nMels = config.nMels
        let subFactor = config.subsamplingFactor
        let margin = Self.melMarginFrames

        let chunkStartMel = sttFeat - leftOffset
        let chunkEndMel = endFeat + rightOffset

        // Extract with margin so reflect padding stays outside the used span.
        let extractStartMel = max(0, chunkStartMel - margin)
        let extractStartSample = extractStartMel * hop
        let extractEndSample = min(
            totalSamples, chunkEndMel * hop + margin * hop + config.nFFT)
        let sliceStart = extractStartSample - pcmBaseSample
        let sliceEnd = extractEndSample - pcmBaseSample
        guard sliceStart >= 0, sliceEnd <= pcm.count, sliceEnd > sliceStart else {
            return
        }
        let (melSpec, extractedFrames) = melExtractor.extract(
            Array(pcm[sliceStart..<sliceEnd]))
        let localStart = chunkStartMel - extractStartMel
        let localEnd = min(extractedFrames, chunkEndMel - extractStartMel)
        guard localEnd > localStart else { return }

        let actualLen = localEnd - localStart
        let coreMLInputFrames = config.coreMLInputFrames
        var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)
        let framesToCopy = min(actualLen, coreMLInputFrames)
        chunkMel.withUnsafeMutableBufferPointer { dst in
            melSpec.withUnsafeBufferPointer { src in
                memcpy(
                    dst.baseAddress!,
                    src.baseAddress! + localStart * nMels,
                    framesToCopy * nMels * MemoryLayout<Float>.stride)
            }
        }

        let dim = config.fcDModel
        var paddedSpkcache = state.spkcache
        paddedSpkcache.append(contentsOf: [Float](
            repeating: 0, count: config.spkcacheLen * dim - paddedSpkcache.count))
        var paddedFifo = state.fifo
        paddedFifo.append(contentsOf: [Float](
            repeating: 0, count: config.fifoLen * dim - paddedFifo.count))

        // Pool per prediction: the push loop has no draining run loop, and
        // without this the autoreleased IOSurface-backed CoreML buffers
        // accumulate across chunks until allocation fails on long streams.
        let output = try autoreleasepool {
            try model.predict(
                chunk: chunkMel,
                chunkLength: framesToCopy,
                spkcache: paddedSpkcache,
                spkcacheLength: state.spkcacheLength,
                fifo: paddedFifo,
                fifoLength: state.fifoLength)
        }

        let lcFrames = Int(Float(leftOffset) / Float(subFactor) + 0.5)
        let rcFrames = Int(ceil(Float(rightOffset) / Float(subFactor)))
        let update = updater.update(
            state: &state,
            chunkEmbs: output.encoderEmbs,
            preds: output.speakerPreds,
            leftContext: lcFrames,
            rightContext: rcFrames)
        confirmedProbs.append(contentsOf: update.confirmed)
    }
}

extension SortformerDiarizer {
    /// Open an incremental session sharing this diarizer's loaded model.
    /// Use the `.streaming` config preset for realtime-shaped chunks; other
    /// presets work but confirm output only every `chunkLenSeconds` seconds.
    public func makeStreamingSession() -> SortformerStreamingSession {
        SortformerStreamingSession(model: coreMLModel, config: configuration)
    }
}
#endif
