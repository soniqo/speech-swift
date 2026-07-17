#if canImport(CoreML)
import AudioCommon
import CoreML
import Foundation

/// Native Apple runtime for pyannote `speaker-diarization-community-1`.
///
/// The published bundle contains Core ML segmentation and masked WeSpeaker
/// stages plus the PLDA tensors required by native VBx clustering. Unlike the
/// lightweight ``PyannoteDiarizationPipeline``, this class reproduces the full
/// Community-1 host pipeline: 1-second sliding chunks, hard powerset decoding,
/// overlap-aware embeddings, speaker counting, VBx, constrained assignment,
/// and overlap-add timeline reconstruction.
///
/// - Important: This class is not thread-safe. Create one instance per
///   concurrent diarization stream.
public final class Community1DiarizationPipeline {
    public static let defaultModelId = "aufklarer/Pyannote-Community-1-CoreML"

    private let models: Community1CoreMLModels
    private let plda: Community1PLDA
    public let config: Community1Config

    private init(
        models: Community1CoreMLModels,
        plda: Community1PLDA,
        config: Community1Config
    ) {
        self.models = models
        self.plda = plda
        self.config = config
    }

    /// Download and load the complete Community-1 Core ML bundle.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        config: Community1Config = .default,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Community1DiarizationPipeline {
        progressHandler?(0, "Downloading Community-1 bundle...")
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: directory,
            additionalFiles: [
                "segmentation.mlmodelc/**",
                "embedding.mlmodelc/**",
                "plda.safetensors",
                "config.json",
            ],
            offlineMode: offlineMode,
            progressHandler: { progress in
                progressHandler?(progress * 0.75, "Downloading Community-1 bundle...")
            }
        )
        return try fromLocal(
            directory: directory,
            config: config,
            computeUnits: computeUnits,
            progressHandler: { progress, message in
                progressHandler?(0.75 + progress * 0.25, message)
            }
        )
    }

    /// Load a previously downloaded or locally exported Community-1 bundle.
    public static func fromLocal(
        directory: URL,
        config: Community1Config = .default,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> Community1DiarizationPipeline {
        progressHandler?(0, "Validating Community-1 bundle...")
        let manifest = try Community1BundleManifest.load(from: directory)

        let pldaURL = directory.appendingPathComponent(manifest.plda.weights)
        guard FileManager.default.fileExists(atPath: pldaURL.path) else {
            throw Community1Error.invalidBundle("missing PLDA weights at \(pldaURL.path)")
        }
        let plda = try Community1PLDA.load(from: pldaURL)
        progressHandler?(0.25, "Loading Community-1 Core ML models...")
        let models = try Community1CoreMLModels(
            directory: directory,
            manifest: manifest,
            computeUnits: computeUnits
        )
        _ = progressHandler?(1, "Ready")
        return Community1DiarizationPipeline(models: models, plda: plda, config: config)
    }

    /// Run both fixed-shape Core ML graphs once to populate their execution caches.
    public func prewarm() throws {
        let silence = [Float](repeating: 0, count: Community1Config.chunkSamples)
        let segmentation = try models.segment(waveform: silence)
        _ = try models.embed(waveform: silence, masks: segmentation)
    }

    public func diarize(
        audio: [Float],
        sampleRate: Int,
        speakerBounds: Community1SpeakerBounds = .inferred
    ) throws -> DiarizationResult {
        try diarize(
            audio: audio,
            sampleRate: sampleRate,
            speakerBounds: speakerBounds,
            progressHandler: nil
        )
    }

    /// Diarize complete audio with stage progress and cooperative cancellation.
    ///
    /// The progress callback returns `true` to continue or `false` to stop. A
    /// cancelled run returns an empty result, matching the other diarizers.
    public func diarize(
        audio: [Float],
        sampleRate: Int,
        speakerBounds: Community1SpeakerBounds = .inferred,
        progressHandler: ((Float, String) -> Bool)?
    ) throws -> DiarizationResult {
        let empty = DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        guard !audio.isEmpty else { return empty }
        let samples = DiarizationHelpers.resample(
            audio, from: sampleRate, to: Community1Config.sampleRate
        )
        guard !samples.isEmpty else { return empty }

        let starts = Self.chunkStarts(sampleCount: samples.count)
        let totalWork = max(1, starts.count * 2 + 1)
        var completed = 0
        var segmentations = [[[Float]]]()
        segmentations.reserveCapacity(starts.count)

        for (index, start) in starts.enumerated() {
            if progressHandler?(
                Float(completed) / Float(totalWork),
                "Community-1 segmentation \(index + 1)/\(starts.count)"
            ) == false { return empty }
            let waveform = Self.chunk(samples, start: start)
            segmentations.append(try models.segment(waveform: waveform))
            completed += 1
        }

        let count = Self.speakerCount(segmentations)
        guard count.max() ?? 0 > 0 else { return empty }

        var embeddings = [[[Float]]]()
        embeddings.reserveCapacity(starts.count)
        for (index, start) in starts.enumerated() {
            if progressHandler?(
                Float(completed) / Float(totalWork),
                "Community-1 embeddings \(index + 1)/\(starts.count)"
            ) == false { return empty }
            let waveform = Self.chunk(samples, start: start)
            embeddings.append(try models.embed(
                waveform: waveform,
                masks: segmentations[index]
            ))
            completed += 1
        }

        if progressHandler?(
            Float(completed) / Float(totalWork), "Community-1 VBx clustering"
        ) == false { return empty }
        let clustered = try Community1Clustering.cluster(
            embeddings: embeddings,
            segmentations: segmentations,
            plda: plda,
            config: config,
            bounds: speakerBounds
        )
        guard !clustered.centroids.isEmpty else { return empty }

        // Local speakers with no active frame are never allowed to contribute
        // to reconstruction, even if constrained assignment gave them a slot.
        var hardClusters = clustered.hardClusters
        for chunk in segmentations.indices {
            for speaker in 0..<Community1Config.localSpeakers
            where segmentations[chunk].allSatisfy({ $0[speaker] == 0 }) {
                hardClusters[chunk][speaker] = -2
            }
        }

        let binary = Self.reconstruct(
            segmentations: segmentations,
            hardClusters: hardClusters,
            speakerCount: count,
            clusterCount: clustered.centroids.count
        )
        let rawSegments = Self.toSegments(binary)
        let compacted = DiarizationHelpers.compactSpeakerIdsAndEmbeddings(
            rawSegments,
            speakerEmbeddings: clustered.centroids,
            missingEmbeddingDimension: Community1Config.embeddingDimension
        )
        _ = progressHandler?(1, "Ready")
        return DiarizationResult(
            segments: compacted.segments,
            numSpeakers: compacted.speakerEmbeddings.count,
            speakerEmbeddings: compacted.speakerEmbeddings
        )
    }

    /// Start sample for every official 10-second / 1-second sliding chunk.
    static func chunkStarts(sampleCount: Int) -> [Int] {
        guard sampleCount > 0 else { return [] }
        let window = Community1Config.chunkSamples
        let step = Int(Community1Config.chunkStep * Float(Community1Config.sampleRate))
        let complete = sampleCount >= window ? (sampleCount - window) / step + 1 : 0
        var starts = (0..<complete).map { $0 * step }
        let hasLast = sampleCount < window || (sampleCount - window) % step > 0
        if hasLast { starts.append(complete * step) }
        return starts
    }

    private static func chunk(_ samples: [Float], start: Int) -> [Float] {
        var result = [Float](repeating: 0, count: Community1Config.chunkSamples)
        guard start < samples.count else { return result }
        let count = min(Community1Config.chunkSamples, samples.count - start)
        result.replaceSubrange(0..<count, with: samples[start..<(start + count)])
        return result
    }

    /// pyannote's overlap-averaged instantaneous speaker count.
    static func speakerCount(_ segmentations: [[[Float]]]) -> [Int] {
        guard !segmentations.isEmpty else { return [] }
        let outputFrames = aggregateFrameCount(chunkCount: segmentations.count)
        var totals = [Double](repeating: 0, count: outputFrames)
        var contributors = [Int](repeating: 0, count: outputFrames)
        for chunk in segmentations.indices {
            let start = closestFrame(
                Double(chunk) * Double(Community1Config.chunkStep)
                    + Double(Community1Config.frameDuration) / 2
            )
            for frame in 0..<Community1Config.framesPerChunk {
                let output = start + frame
                guard output >= 0, output < outputFrames else { continue }
                totals[output] += Double(segmentations[chunk][frame].reduce(0, +))
                contributors[output] += 1
            }
        }
        return totals.indices.map { frame in
            guard contributors[frame] > 0 else { return 0 }
            return Int((totals[frame] / Double(contributors[frame])).rounded(.toNearestOrEven))
        }
    }

    static func reconstruct(
        segmentations: [[[Float]]],
        hardClusters: [[Int]],
        speakerCount: [Int],
        clusterCount: Int
    ) -> [[Float]] {
        let maximumCount = speakerCount.max() ?? 0
        let outputSpeakers = max(clusterCount, maximumCount)
        var activations = Array(
            repeating: [Double](repeating: 0, count: outputSpeakers),
            count: speakerCount.count
        )
        for chunk in segmentations.indices {
            let start = closestFrame(
                Double(chunk) * Double(Community1Config.chunkStep)
                    + Double(Community1Config.frameDuration) / 2
            )
            for frame in 0..<Community1Config.framesPerChunk {
                let output = start + frame
                guard output >= 0, output < speakerCount.count else { continue }
                var chunkActivations = [Double](
                    repeating: -.infinity, count: outputSpeakers
                )
                for localSpeaker in 0..<Community1Config.localSpeakers {
                    let cluster = hardClusters[chunk][localSpeaker]
                    guard cluster >= 0, cluster < outputSpeakers else { continue }
                    let value = Double(segmentations[chunk][frame][localSpeaker])
                    chunkActivations[cluster] = max(chunkActivations[cluster], value)
                }
                for cluster in 0..<outputSpeakers where chunkActivations[cluster].isFinite {
                    // First take the maximum across local tracks assigned to
                    // this cluster, then sum overlapping chunk activations.
                    activations[output][cluster] += chunkActivations[cluster]
                }
            }
        }

        var binary = Array(
            repeating: [Float](repeating: 0, count: outputSpeakers),
            count: speakerCount.count
        )
        for frame in speakerCount.indices {
            let ordered = activations[frame].indices.sorted {
                if activations[frame][$0] == activations[frame][$1] { return $0 < $1 }
                return activations[frame][$0] > activations[frame][$1]
            }
            for speaker in ordered.prefix(min(speakerCount[frame], outputSpeakers)) {
                binary[frame][speaker] = 1
            }
        }
        return binary
    }

    /// Convert frame-center state transitions with pyannote's strict 0.5 rules.
    static func toSegments(_ binary: [[Float]]) -> [DiarizedSegment] {
        guard let speakerCount = binary.first?.count,
              speakerCount > 0,
              !binary.isEmpty else { return [] }
        let firstTimestamp = Community1Config.frameDuration / 2
        func timestamp(_ frame: Int) -> Float {
            firstTimestamp + Float(frame) * Community1Config.frameStep
        }

        var segments = [DiarizedSegment]()
        for speaker in 0..<speakerCount {
            var start = firstTimestamp
            var active = binary[0][speaker] > 0.5
            for frame in 1..<binary.count {
                let time = timestamp(frame)
                if active, binary[frame][speaker] < 0.5 {
                    if time > start {
                        segments.append(DiarizedSegment(
                            startTime: start, endTime: time, speakerId: speaker
                        ))
                    }
                    start = time
                    active = false
                } else if !active, binary[frame][speaker] > 0.5 {
                    start = time
                    active = true
                }
            }
            if active {
                let end = timestamp(binary.count - 1)
                if end > start {
                    segments.append(DiarizedSegment(
                        startTime: start, endTime: end, speakerId: speaker
                    ))
                }
            }
        }
        return segments.sorted {
            if $0.startTime == $1.startTime { return $0.speakerId < $1.speakerId }
            return $0.startTime < $1.startTime
        }
    }

    private static func aggregateFrameCount(chunkCount: Int) -> Int {
        let end = Double(Community1Config.chunkDuration)
            + Double(chunkCount - 1) * Double(Community1Config.chunkStep)
        return closestFrame(end + Double(Community1Config.frameDuration) / 2) + 1
    }

    private static func closestFrame(_ time: Double) -> Int {
        let centered = (time - Double(Community1Config.frameDuration) / 2)
            / Double(Community1Config.frameStep)
        return Int(centered.rounded(.toNearestOrEven))
    }
}
#endif
