#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// End-to-end neural speaker diarization using NVIDIA Sortformer (CoreML).
///
/// Sortformer directly predicts per-frame speaker activity for up to 4 speakers
/// without requiring separate embedding extraction or clustering. The default
/// preset (`SortformerConfig.default`) runs on the Neural Engine at ~125–750×
/// RTF on M-series silicon (warm, depending on input length) — single CoreML
/// dispatch handles up to 30 s of audio per call. A small-chunk
/// `.streaming` preset is available for future realtime / low-latency
/// consumers but is significantly slower per-second-of-audio.
///
/// ```swift
/// let diarizer = try await SortformerDiarizer.fromPretrained()
/// let result = diarizer.diarize(audio: samples, sampleRate: 16000)
/// for seg in result.segments {
///     print("Speaker \(seg.speakerId): [\(seg.startTime)s - \(seg.endTime)s]")
/// }
/// ```
public final class SortformerDiarizer {

    /// Default HuggingFace model ID for the CoreML Sortformer model
    public static let defaultModelId = "aufklarer/Sortformer-Diarization-CoreML"

    private let model: SortformerCoreMLModel
    private let melExtractor: SortformerMelExtractor
    let config: SortformerConfig

    /// Frame duration from model metadata (0.08s = 80ms per diarization frame)
    private let frameDuration: Float = 0.08

    // MARK: - Streaming State

    private var state: SortformerStreamingState
    private let updater: SortformerStateUpdater

    init(model: SortformerCoreMLModel, config: SortformerConfig = .default) {
        self.model = model
        self.config = config
        self.melExtractor = SortformerMelExtractor(config: config)
        self.state = SortformerStreamingState(config: config)
        self.updater = SortformerStateUpdater(config: config)
    }

    /// Reset streaming state between different audio files.
    public func resetState() {
        state.reset()
    }

    // MARK: - Loading

    /// Load a pre-trained Sortformer model from HuggingFace.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - cacheDir: optional override for the local cache directory
    ///   - offlineMode: if true, only consult the local cache
    ///   - config: chunk-shape preset (`.default`, `.balanced`, `.streaming`)
    ///   - computeUnits: which CoreML backends to enable. Defaults to
    ///     `.cpuAndNeuralEngine` (ANE-only, no GPU/MPSGraph validation).
    ///     Set to `.cpuOnly` for instant first-load at the cost of ~10–50×
    ///     RTF instead of ~125–750× (useful during onboarding / on
    ///     simulator / on weak devices). `SPEECH_COREML_COMPUTE_UNITS` env
    ///     var overrides this when set.
    ///   - progressHandler: callback for download progress
    /// - Returns: ready-to-use diarizer
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        config: SortformerConfig = .default,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SortformerDiarizer {
        progressHandler?(0.0, "Downloading Sortformer model...")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        let modelFileName = config.coreMLModelFileName
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: ["\(modelFileName)/**", "config.json"],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.8, "Downloading Sortformer model...")
            }
        )

        progressHandler?(0.8, "Loading CoreML model...")

        let modelURL = cacheDir.appendingPathComponent(modelFileName, isDirectory: true)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "CoreML model not found at \(modelURL.path)")
        }

        let mlConfig = MLModelConfiguration()
        // `.cpuAndNeuralEngine` skips the GPU/MPSGraph backend during model
        // load validation — see #319 (M5 Max + macOS 26) and #328. The
        // caller can override via the `computeUnits` parameter or by
        // setting SPEECH_COREML_COMPUTE_UNITS in the environment.
        mlConfig.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)
        mlConfig.allowLowPrecisionAccumulationOnGPU = true

        let mlModel: MLModel
        do {
            mlModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
        } catch {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Failed to load CoreML model",
                underlying: error)
        }

        let coremlModel = SortformerCoreMLModel(model: mlModel, config: config)

        progressHandler?(1.0, "Ready")
        return SortformerDiarizer(model: coremlModel, config: config)
    }

    /// Prime the on-device CoreML cache without holding a diarizer instance.
    ///
    /// The first time a Sortformer `.mlmodelc` loads on a given device,
    /// CoreML compiles its MIL into ANE bytecode via the BNNS compiler.
    /// For `.default` (chunk_len=340) this can take minutes on cold cache;
    /// subsequent loads pull from `~/Library/Caches/com.apple.modelcompiler/`
    /// in seconds. Call this from a background task at app launch (or any
    /// non-interactive moment) so the user pays the compile cost during
    /// onboarding instead of on the first diarize call.
    ///
    /// The compiled artifact persists in the system cache after this
    /// returns; the temporary `MLModel` instance is discarded.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace model ID
    ///   - cacheDir: optional override for the local cache directory
    ///   - offlineMode: if true, only consult the local cache
    ///   - config: chunk-shape preset to warm
    ///   - computeUnits: backends to compile for (defaults to `.cpuAndNeuralEngine`)
    ///   - progressHandler: callback for download + load progress
    public static func preload(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        config: SortformerConfig = .default,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws {
        // Reuses fromPretrained so the load path is identical to what the
        // app would later hit; the diarizer is allowed to deallocate as
        // soon as this function returns — the BNNS compile cache survives.
        _ = try await fromPretrained(
            modelId: modelId,
            cacheDir: cacheDir,
            offlineMode: offlineMode,
            config: config,
            computeUnits: computeUnits,
            progressHandler: progressHandler)
    }

    // MARK: - Diarization

    /// Run speaker diarization on complete audio.
    ///
    /// Processes audio in streaming chunks matching NeMo's streaming_feat_loader:
    /// each chunk is 112 mel frames = (leftCtx + coreChunk + rightCtx) × subsampling.
    /// Core predictions are extracted per chunk and concatenated.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: optional override for diarization thresholds
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig = .default
    ) -> DiarizationResult {
        diarize(audio: audio, sampleRate: sampleRate, config: config, progressHandler: nil)
    }

    /// Run speaker diarization with progress reporting and optional cancellation.
    ///
    /// Same as `diarize(audio:sampleRate:config:)` but reports progress per chunk.
    /// The handler returns a `Bool`: `true` to continue, `false` to cancel.
    /// When cancelled, an empty `DiarizationResult` is returned immediately.
    ///
    /// - Parameters:
    ///   - audio: PCM Float32 audio samples
    ///   - sampleRate: sample rate of the input audio
    ///   - config: optional override for diarization thresholds
    ///   - progressHandler: called with (progress 0.0–1.0, stage description);
    ///     return `true` to continue or `false` to cancel
    /// - Returns: diarization result with speaker-labeled segments
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        config: DiarizationConfig = .default,
        progressHandler: ((Float, String) -> Bool)?
    ) -> DiarizationResult {
        let samples = DiarizationHelpers.resample(audio, from: sampleRate, to: self.config.sampleRate)

        guard !samples.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        resetState()

        // Extract mel features for the entire audio: [totalMelFrames, 128]
        let (melSpec, totalMelFrames) = melExtractor.extract(samples)

        guard totalMelFrames > 0 else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Streaming chunking parameters (matching NeMo)
        let subFactor = self.config.subsamplingFactor
        let chunkLen = Int(self.config.chunkLenSeconds)
        let leftCtx = Int(self.config.leftContextSeconds)
        let rightCtx = Int(self.config.rightContextSeconds)
        let coreMelFrames = chunkLen * subFactor
        let coreMLInputFrames = self.config.coreMLInputFrames
        let nMels = self.config.nMels
        let numSpeakers = self.config.maxSpeakers
        let dim = self.config.fcDModel
        let spkcacheCapacity = self.config.spkcacheLen
        let fifoCapacity = self.config.fifoLen

        var allChunkProbs = [[Float]]()
        let emptyResult = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])

        let totalChunks = max(1, (totalMelFrames + coreMelFrames - 1) / coreMelFrames)
        var chunkIndex = 0

        var sttFeat = 0
        var endFeat = 0

        while endFeat < totalMelFrames {
            chunkIndex += 1
            if progressHandler?(Float(chunkIndex) / Float(totalChunks), "Diarizing \(chunkIndex)/\(totalChunks)") == false {
                return emptyResult
            }
            let leftOffset = min(leftCtx * subFactor, sttFeat)
            endFeat = min(sttFeat + coreMelFrames, totalMelFrames)
            let rightOffset = min(rightCtx * subFactor, totalMelFrames - endFeat)

            let chunkStart = sttFeat - leftOffset
            let chunkEnd = endFeat + rightOffset
            let actualLen = chunkEnd - chunkStart

            // Build padded mel chunk [coreMLInputFrames, nMels]
            var chunkMel = [Float](repeating: 0, count: coreMLInputFrames * nMels)
            let framesToCopy = min(actualLen, coreMLInputFrames)
            if framesToCopy > 0 {
                let elements = framesToCopy * nMels
                let srcStart = chunkStart * nMels
                chunkMel.withUnsafeMutableBufferPointer { dst in
                    melSpec.withUnsafeBufferPointer { src in
                        memcpy(dst.baseAddress!, src.baseAddress! + srcStart,
                               elements * MemoryLayout<Float>.stride)
                    }
                }
            }

            // The CoreML inputs require fixed shapes — pad the dynamic state
            // arrays up to capacity for this call. The model fills the
            // unused tail with zeros via its internal length inputs.
            var paddedSpkcache = state.spkcache
            paddedSpkcache.append(contentsOf:
                [Float](repeating: 0,
                        count: spkcacheCapacity * dim - paddedSpkcache.count))
            var paddedFifo = state.fifo
            paddedFifo.append(contentsOf:
                [Float](repeating: 0,
                        count: fifoCapacity * dim - paddedFifo.count))

            do {
                // Pool per prediction: this loop has no draining run loop, so
                // autoreleased IOSurface-backed CoreML buffers otherwise
                // accumulate across chunks until allocation fails on long
                // inputs.
                let output = try autoreleasepool {
                    try model.predict(
                        chunk: chunkMel,
                        chunkLength: actualLen,
                        spkcache: paddedSpkcache,
                        spkcacheLength: state.spkcacheLength,
                        fifo: paddedFifo,
                        fifoLength: state.fifoLength
                    )
                }

                let lcFrames = Int(Float(leftOffset) / Float(subFactor) + 0.5)
                let rcFrames = Int(ceil(Float(rightOffset) / Float(subFactor)))

                let result = updater.update(
                    state: &state,
                    chunkEmbs: output.encoderEmbs,
                    preds: output.speakerPreds,
                    leftContext: lcFrames,
                    rightContext: rcFrames)
                allChunkProbs.append(result.confirmed)
            } catch {
                print("Warning: Sortformer inference failed on chunk at mel frame \(sttFeat): \(error)")
            }

            sttFeat = endFeat
        }

        guard !allChunkProbs.isEmpty else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }

        // Concatenate all core predictions
        let audioDuration = Float(samples.count) / Float(self.config.sampleRate)
        let segments = binarizeCorePredictions(
            allChunkProbs: allChunkProbs,
            audioDuration: audioDuration,
            numSpeakers: numSpeakers,
            onset: config.onset,
            offset: config.offset,
            minSpeechDuration: config.minSpeechDuration,
            minSilenceDuration: config.minSilenceDuration
        )

        let usedSpeakers = Set(segments.map(\.speakerId))
        return DiarizationResult(
            segments: segments,
            numSpeakers: usedSpeakers.count,
            speakerEmbeddings: []  // End-to-end model, no separate embeddings
        )
    }

    // MARK: - Binarization

    /// Concatenate per-chunk core predictions and binarize into segments.
    private func binarizeCorePredictions(
        allChunkProbs: [[Float]],
        audioDuration: Float,
        numSpeakers: Int,
        onset: Float,
        offset: Float,
        minSpeechDuration: Float,
        minSilenceDuration: Float
    ) -> [DiarizedSegment] {
        var allProbs = [Float]()
        for chunkProbs in allChunkProbs {
            allProbs.append(contentsOf: chunkProbs)
        }
        return Self.binarize(
            probs: allProbs,
            frameCount: allProbs.count / numSpeakers,
            audioDuration: audioDuration,
            config: config,
            thresholds: DiarizationConfig(
                onset: onset,
                offset: offset,
                minSpeechDuration: minSpeechDuration,
                minSilenceDuration: minSilenceDuration))
    }

    /// Turn flat per-frame speaker probabilities into labeled segments.
    /// Shared by whole-buffer diarization and the incremental streaming
    /// session, which re-binarizes its growing confirmed-frame history.
    static func binarize(
        probs: [Float],
        frameCount: Int,
        audioDuration: Float,
        config: SortformerConfig,
        thresholds: DiarizationConfig
    ) -> [DiarizedSegment] {
        guard frameCount > 0 else { return [] }
        let numSpeakers = config.maxSpeakers
        var allProbs = probs

        // Apply sigmoid if predictions are logits
        for i in 0..<allProbs.count {
            if allProbs[i] > 1.0 || allProbs[i] < 0.0 {
                allProbs[i] = 1.0 / (1.0 + exp(-allProbs[i]))
            }
        }

        let frameDuration = Float(config.subsamplingFactor * config.hopLength)
            / Float(config.sampleRate)
        var allSegments = [DiarizedSegment]()
        for spk in 0..<numSpeakers {
            var track = [Float](repeating: 0, count: frameCount)
            for f in 0..<frameCount {
                track[f] = allProbs[f * numSpeakers + spk]
            }

            let rawSegments = PowersetDecoder.binarize(
                probs: track,
                onset: thresholds.onset,
                offset: thresholds.offset,
                frameDuration: frameDuration
            )

            for seg in rawSegments {
                let duration = seg.endTime - seg.startTime
                guard duration >= thresholds.minSpeechDuration else { continue }
                allSegments.append(DiarizedSegment(
                    startTime: seg.startTime,
                    endTime: min(seg.endTime, audioDuration),
                    speakerId: spk
                ))
            }
        }

        allSegments.sort { $0.startTime < $1.startTime }
        let merged = DiarizationHelpers.mergeSegments(
            allSegments, minSilence: thresholds.minSilenceDuration)
        return DiarizationHelpers.compactSpeakerIds(merged)
    }

    // MARK: - Streaming session support

    var coreMLModel: SortformerCoreMLModel { model }
    var configuration: SortformerConfig { config }
}

// MARK: - SpeakerDiarizationModel

extension SortformerDiarizer: SpeakerDiarizationModel {
    public var inputSampleRate: Int { config.sampleRate }

    public func diarize(audio: [Float], sampleRate: Int) -> [DiarizedSegment] {
        diarize(audio: audio, sampleRate: sampleRate, config: .default).segments
    }
}
#endif
