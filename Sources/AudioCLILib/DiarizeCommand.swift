import Foundation
import ArgumentParser
import SpeechVAD
import AudioCommon
#if canImport(CoreML)
import CoreML
#endif

public struct DiarizeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "diarize",
        abstract: "Identify speakers and their speech segments in audio"
    )

    @Argument(help: "Audio file to analyze (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .long, help: "Enrollment audio for target speaker extraction")
    public var targetSpeaker: String?

    @Option(name: .long, help: "Diarization engine: pyannote (default), community1, sortformer, or nemotron3")
    public var engine: String = "pyannote"

    @Option(name: .long, help: "Nemotron 3 backend: coreml (default) or mlx")
    public var nemotron3Backend: String = "coreml"

    @Option(name: .long, help: "Optional local Nemotron 3 bundle directory")
    public var nemotron3Directory: String?

    @Option(name: .long, help: "Nemotron 3 Core ML compute units: ane (default), cpu, gpu, or all")
    public var nemotron3ComputeUnits: String = "ane"

    @Option(name: .long, help: "Sortformer variant: default (offline, ~125x RTF), balanced (faster first-load, ~hundreds-x RTF), or streaming (low-latency)")
    public var sortformerVariant: String = "default"

    @Option(name: .long, help: "Sortformer CoreML compute units: ane (default, ANE+CPU), cpu (instant first-load, ~20x RTF), gpu (cpu+gpu), all (every backend; rarely needed)")
    public var sortformerComputeUnits: String = "ane"

    @Option(name: .long, help: "Community-1 CoreML compute units: ane (default), cpu, gpu, or all")
    public var community1ComputeUnits: String = "ane"

    @Option(name: .long, help: "Known exact speaker count for Community-1")
    public var numSpeakers: Int?

    @Option(name: .long, help: "Minimum speaker count for Community-1 (default 1)")
    public var minSpeakers: Int = 1

    @Option(name: .long, help: "Maximum speaker count for Community-1")
    public var maxSpeakers: Int?

    @Option(name: .long, help: "Speaker embedding engine: mlx (default) or coreml")
    public var embeddingEngine: String = "mlx"

    @Flag(name: .long, help: "Output as JSON")
    public var json: Bool = false

    @Flag(name: .long, help: "Output as RTTM (standard diarization evaluation format)")
    public var rttm: Bool = false

    @Flag(name: .long, help: "Pre-filter with Silero VAD to reduce false alarms")
    public var vadFilter: Bool = false

    @Option(name: .long, help: "Reference RTTM file to score against (computes DER)")
    public var scoreAgainst: String?

    @Option(name: .long, help: "Minimum silence between segments in seconds (default 0.15)")
    public var minSilence: Float = 0.15

    @Option(name: .long, help: "Minimum speech segment duration in seconds (default 0.3)")
    public var minSpeech: Float = 0.3

    @Option(name: .long, help: "Speaker activity onset threshold (default 0.5)")
    public var onset: Float = 0.5

    @Option(name: .long, help: "Speaker activity offset threshold (default: 0.5 for Sortformer and Nemotron 3; 0.3 for pyannote)")
    public var offset: Float?

    @Option(name: .long, help: "Cosine distance threshold for speaker clustering (default 0.715, lower = fewer speakers)")
    public var clusterThreshold: Float = 0.715

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile), targetSampleRate: 16000)
            let duration = formatDuration(audio.count, sampleRate: 16000)
            print("  Loaded \(audio.count) samples (\(duration)s)")

            // Sortformer binarizes at NeMo's symmetric 0.5/0.5 by default; the
            // pyannote pipeline keeps its tuned 0.3 hysteresis offset.
            let config = DiarizationConfig(
                onset: onset,
                offset: offset ?? ((engine == "sortformer" || engine == "nemotron3") ? 0.5 : 0.3),
                minSpeechDuration: minSpeech,
                minSilenceDuration: minSilence,
                clusteringThreshold: clusterThreshold)

            if engine == "sortformer" {
                #if canImport(CoreML)
                try await runSortformer(audio: audio, config: config)
                #else
                print("Error: Sortformer requires CoreML (not available on this platform).")
                #endif
            } else if engine == "nemotron3" {
                #if canImport(CoreML)
                try await runNemotron3(audio: audio, config: config)
                #else
                print("Error: Nemotron 3 requires an Apple Silicon runtime.")
                #endif
            } else if engine == "community1" {
                #if canImport(CoreML)
                try await runCommunity1(audio: audio)
                #else
                print("Error: Community-1 requires CoreML (not available on this platform).")
                #endif
            } else if engine == "pyannote" {
                try await runPyannote(audio: audio, config: config)
            } else {
                print("Error: unknown engine '\(engine)'. Use 'pyannote', 'community1', 'sortformer', or 'nemotron3'.")
            }
        }
    }

    #if canImport(CoreML)
    private func runNemotron3(audio: [Float], config: DiarizationConfig) async throws {
        if targetSpeaker != nil {
            print("Warning: --target-speaker is not supported with Nemotron 3. Ignoring.")
        }
        let backend = nemotron3Backend.lowercased()
        let diarizer: Nemotron3Diarizer
        switch backend {
        case "coreml":
            let computeUnits: MLComputeUnits
            switch nemotron3ComputeUnits.lowercased() {
            case "ane", "cpuandneuralengine", "neuralengine":
                computeUnits = .cpuAndNeuralEngine
            case "cpu", "cpuonly":
                computeUnits = .cpuOnly
            case "gpu", "cpuandgpu":
                computeUnits = .cpuAndGPU
            case "all":
                computeUnits = .all
            default:
                print("Error: unknown Nemotron 3 compute units '\(nemotron3ComputeUnits)'. Use 'ane', 'cpu', 'gpu', or 'all'.")
                return
            }
            if let nemotron3Directory {
                diarizer = try Nemotron3Diarizer.fromCoreMLDirectory(
                    URL(fileURLWithPath: nemotron3Directory, isDirectory: true),
                    computeUnits: computeUnits)
            } else {
                diarizer = try await Nemotron3Diarizer.fromCoreMLPretrained(
                    computeUnits: computeUnits, progressHandler: reportProgress)
            }
        case "mlx":
            if let nemotron3Directory {
                diarizer = try Nemotron3Diarizer.fromMLXDirectory(
                    URL(fileURLWithPath: nemotron3Directory, isDirectory: true))
            } else {
                diarizer = try await Nemotron3Diarizer.fromMLXPretrained(
                    progressHandler: reportProgress)
            }
        default:
            print("Error: unknown Nemotron 3 backend '\(nemotron3Backend)'. Use 'coreml' or 'mlx'.")
            return
        }

        print("Running diarization (Nemotron 3 \(backend))...")
        let start = Date()
        var result = try diarizer.diarize(audio: audio, sampleRate: 16_000, config: config)
        if vadFilter {
            let vadModel = try await SileroVADModel.fromPretrained(
                progressHandler: reportProgress)
            let speech = vadModel.detectSpeech(audio: audio, sampleRate: 16_000)
            result = Self.maskedToSpeech(result, speech: speech)
        }
        outputResult(result, elapsed: Date().timeIntervalSince(start))
        if let scoreAgainst {
            try scoreDER(result: result, refFile: scoreAgainst)
        }
    }

    private func runSortformer(audio: [Float], config: DiarizationConfig) async throws {
        if targetSpeaker != nil {
            print("Warning: --target-speaker is not supported with Sortformer (no speaker embeddings). Ignoring.")
        }

        let sortformerConfig: SortformerConfig
        switch sortformerVariant.lowercased() {
        case "default":   sortformerConfig = .default
        case "balanced":  sortformerConfig = .balanced
        case "streaming": sortformerConfig = .streaming
        default:
            print("Error: unknown sortformer variant '\(sortformerVariant)'. Use 'default', 'balanced', or 'streaming'.")
            return
        }

        let computeUnits: MLComputeUnits
        switch sortformerComputeUnits.lowercased() {
        case "ane", "cpuandneuralengine", "neuralengine":
            computeUnits = .cpuAndNeuralEngine
        case "cpu", "cpuonly":
            computeUnits = .cpuOnly
        case "gpu", "cpuandgpu":
            computeUnits = .cpuAndGPU
        case "all":
            computeUnits = .all
        default:
            print("Error: unknown sortformer compute units '\(sortformerComputeUnits)'. Use 'ane', 'cpu', 'gpu', or 'all'.")
            return
        }

        print("Loading Sortformer model (variant: \(sortformerVariant.lowercased()), chunk=\(sortformerConfig.coreMLInputFrames) mel frames, compute units: \(sortformerComputeUnits.lowercased()))...")
        let diarizer = try await SortformerDiarizer.fromPretrained(
            config: sortformerConfig,
            computeUnits: computeUnits,
            progressHandler: reportProgress
        )

        print("Running diarization (Sortformer)...")
        let start = Date()
        var result = diarizer.diarize(audio: audio, sampleRate: 16000, config: config)
        if vadFilter {
            print("Applying Silero VAD filter...")
            let vadModel = try await SileroVADModel.fromPretrained(progressHandler: reportProgress)
            let speech = vadModel.detectSpeech(audio: audio, sampleRate: 16000)
            result = Self.maskedToSpeech(result, speech: speech)
        }
        let elapsed = Date().timeIntervalSince(start)

        outputResult(result, elapsed: elapsed)

        if let refFile = scoreAgainst {
            try scoreDER(result: result, refFile: refFile)
        }
    }

    /// Keep only diarized segments that mostly overlap Silero speech regions,
    /// trimmed to the speech bounds — the same rule the pyannote pipeline
    /// applies when its VAD filter is enabled.
    static func maskedToSpeech(
        _ result: DiarizationResult, speech: [SpeechSegment]
    ) -> DiarizationResult {
        let minDuration: Float = 0.3
        var kept: [DiarizedSegment] = []
        for segment in result.segments {
            var overlap: Float = 0
            var trimStart = segment.endTime
            var trimEnd = segment.startTime
            for region in speech {
                let s = max(segment.startTime, region.startTime)
                let e = min(segment.endTime, region.endTime)
                if s < e {
                    overlap += e - s
                    trimStart = min(trimStart, s)
                    trimEnd = max(trimEnd, e)
                }
            }
            guard segment.duration > 0,
                  overlap / segment.duration >= 0.5,
                  trimEnd - trimStart >= minDuration else { continue }
            kept.append(DiarizedSegment(
                startTime: trimStart, endTime: trimEnd, speakerId: segment.speakerId))
        }
        return DiarizationResult(
            segments: kept,
            numSpeakers: Set(kept.map(\.speakerId)).count,
            speakerEmbeddings: result.speakerEmbeddings)
    }

    private func runCommunity1(audio: [Float]) async throws {
        if targetSpeaker != nil {
            print("Warning: --target-speaker is not supported with Community-1. Ignoring.")
        }

        let computeUnits: MLComputeUnits
        switch community1ComputeUnits.lowercased() {
        case "ane", "cpuandneuralengine", "neuralengine":
            computeUnits = .cpuAndNeuralEngine
        case "cpu", "cpuonly":
            computeUnits = .cpuOnly
        case "gpu", "cpuandgpu":
            computeUnits = .cpuAndGPU
        case "all":
            computeUnits = .all
        default:
            print("Error: unknown Community-1 compute units '\(community1ComputeUnits)'. Use 'ane', 'cpu', 'gpu', or 'all'.")
            return
        }

        print("Loading Community-1 CoreML + native VBx pipeline...")
        let pipeline = try await Community1DiarizationPipeline.fromPretrained(
            computeUnits: computeUnits,
            progressHandler: reportProgress
        )
        let bounds = Community1SpeakerBounds(
            exact: numSpeakers,
            minimum: minSpeakers,
            maximum: maxSpeakers
        )
        print("Running diarization (Community-1)...")
        let start = Date()
        let result = try pipeline.diarize(
            audio: audio,
            sampleRate: 16000,
            speakerBounds: bounds
        )
        let elapsed = Date().timeIntervalSince(start)
        outputResult(result, elapsed: elapsed)

        if let refFile = scoreAgainst {
            try scoreDER(result: result, refFile: refFile)
        }
    }
    #endif

    private func runPyannote(audio: [Float], config: DiarizationConfig) async throws {
        guard let embEngine = WeSpeakerEngine(rawValue: embeddingEngine) else {
            print("Error: unknown embedding engine '\(embeddingEngine)'. Use 'mlx' or 'coreml'.")
            return
        }

        print("Loading diarization models (embedding engine: \(embEngine.rawValue)\(vadFilter ? ", VAD filter" : ""))...")
        let pipeline = try await DiarizationPipeline.fromPretrained(
            embeddingEngine: embEngine,
            useVADFilter: vadFilter,
            progressHandler: reportProgress
        )

        if let enrollmentFile = targetSpeaker {
            print("Loading enrollment audio: \(enrollmentFile)")
            let enrollAudio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: enrollmentFile), targetSampleRate: 16000)

            print("Extracting target speaker embedding...")
            let targetEmb = pipeline.embeddingModel.embed(
                audio: enrollAudio, sampleRate: 16000)

            print("Extracting target speaker segments...")
            let start = Date()
            let segments = pipeline.extractSpeaker(
                audio: audio, sampleRate: 16000,
                targetEmbedding: targetEmb, config: config
            )
            let elapsed = Date().timeIntervalSince(start)

            if json {
                printSpeechJSON(segments)
            } else {
                if segments.isEmpty {
                    print("Target speaker not found.")
                } else {
                    for seg in segments {
                        let s = String(format: "%.2f", seg.startTime)
                        let e = String(format: "%.2f", seg.endTime)
                        let d = String(format: "%.2f", seg.duration)
                        print("Target: [\(s)s - \(e)s] (\(d)s)")
                    }
                    let totalSpeech = segments.reduce(Float(0)) { $0 + $1.duration }
                    print("\n\(segments.count) segment(s), \(String(format: "%.2f", totalSpeech))s total")
                }
                print("Extraction took \(String(format: "%.2f", elapsed))s")
            }
        } else {
            print("Running diarization...")
            let start = Date()
            let result = pipeline.diarize(
                audio: audio, sampleRate: 16000, config: config)
            let elapsed = Date().timeIntervalSince(start)

            outputResult(result, elapsed: elapsed)

            if let refFile = scoreAgainst {
                try scoreDER(result: result, refFile: refFile)
            }
        }
    }

    private func outputResult(_ result: DiarizationResult, elapsed: TimeInterval) {
        if rttm {
            let basename = URL(fileURLWithPath: audioFile).deletingPathExtension().lastPathComponent
            let rttmSegments = toRTTM(segments: result.segments, filename: basename)
            print(formatRTTM(rttmSegments))
        } else if json {
            printDiarizeJSON(result)
        } else {
            if result.segments.isEmpty {
                print("No speech detected.")
            } else {
                for seg in result.segments {
                    let s = String(format: "%.2f", seg.startTime)
                    let e = String(format: "%.2f", seg.endTime)
                    let d = String(format: "%.2f", seg.duration)
                    print("Speaker \(seg.speakerId): [\(s)s - \(e)s] (\(d)s)")
                }
                print("\n--- \(result.numSpeakers) speaker(s) ---")
            }
            print("Diarization took \(String(format: "%.2f", elapsed))s")
        }
    }

    private func scoreDER(result: DiarizationResult, refFile: String) throws {
        let refContent = try String(contentsOfFile: refFile, encoding: .utf8)
        let refSegments = parseRTTM(refContent)
        let derResult = computeDERWithOptimalMapping(
            reference: refSegments, hypothesis: result.segments
        )
        print("\n--- DER Scoring ---")
        print("Total speech: \(String(format: "%.2f", derResult.totalSpeech))s")
        print("Missed speech: \(String(format: "%.2f", derResult.missedSpeech))s")
        print("False alarm: \(String(format: "%.2f", derResult.falseAlarm))s")
        print("Confusion: \(String(format: "%.2f", derResult.confusion))s")
        print("DER: \(String(format: "%.1f", derResult.derPercent))%")
    }

    private func printDiarizeJSON(_ result: DiarizationResult) {
        var items = [[String: Any]]()
        for seg in result.segments {
            items.append([
                "start": Double(String(format: "%.3f", seg.startTime))!,
                "end": Double(String(format: "%.3f", seg.endTime))!,
                "duration": Double(String(format: "%.3f", seg.duration))!,
                "speaker": seg.speakerId,
            ])
        }
        let output: [String: Any] = [
            "segments": items,
            "num_speakers": result.numSpeakers,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }

    private func printSpeechJSON(_ segments: [SpeechSegment]) {
        var items = [[String: Any]]()
        for seg in segments {
            items.append([
                "start": Double(String(format: "%.3f", seg.startTime))!,
                "end": Double(String(format: "%.3f", seg.endTime))!,
                "duration": Double(String(format: "%.3f", seg.duration))!,
            ])
        }
        if let data = try? JSONSerialization.data(withJSONObject: items, options: .prettyPrinted),
           let str = String(data: data, encoding: .utf8) {
            print(str)
        }
    }
}
