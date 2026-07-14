import ArgumentParser
import AudioCommon
import BenchmarkSupport
import Foundation
import SpeechVAD

#if canImport(CoreML)
import CoreML
#endif

@main
struct DiarizationBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "diarization-bench",
        abstract: "Benchmark diarization engines on a manifest of audio plus RTTM references."
    )

    @Option(name: .shortAndLong, help: "Manifest: <audio path>\\t<reference path>[\\t<id>]. References may be RTTM or start/end/speaker text.")
    var manifest: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Engines: community1-coreml, sortformer-default, sortformer-balanced, sortformer-streaming, pyannote-mlx, pyannote-coreml.")
    var engines: [String] = ["sortformer-default", "pyannote-mlx"]

    @Option(name: .shortAndLong, help: "Max files to process.")
    var limit: Int?

    @Option(name: .shortAndLong, help: "Optional path to write JSON report.")
    var output: String?

    @Option(name: .long, help: "Scoring collar in seconds.")
    var collar: Float = 0.25

    @Option(name: .long, help: "Frame scoring resolution in seconds.")
    var resolution: Float = 0.01

    mutating func run() async throws {
        setlinebuf(stdout)

        let entries = try BenchmarkManifest.load(url: URL(fileURLWithPath: manifest), limit: limit)
        print("Loaded \(entries.count) diarization benchmark file(s) from \(manifest)")

        let decoded = try entries.map { entry in
            let audio = try AudioFileLoader.load(url: entry.audioURL, targetSampleRate: 16000)
            let reference = try SegmentReference.loadDiarizedSegments(url: entry.referenceURL)
            return DiarizationInput(entry: entry, audio: audio, reference: reference)
        }
        let totalAudio = decoded.reduce(0.0) { $0 + $1.duration }
        print(String(format: "Total audio: %.1f s\n", totalAudio))

        var results: [DiarizationEngineResult] = []
        for engineID in engines {
            guard let engine = makeDiarizationEngine(engineID) else {
                fputs("Unknown diarization engine: \(engineID)\n", stderr)
                throw ExitCode(2)
            }
            results.append(try await runEngine(engine, inputs: decoded))
        }

        let report = DiarizationReport(
            machine: ProcessInfo.processInfo.hostName,
            manifestPath: manifest,
            files: decoded.count,
            collarSeconds: collar,
            resolutionSeconds: resolution,
            results: results
        )

        print("\n" + DiarizationReportPrinter.table(report))
        if let output {
            try DiarizationReportPrinter.writeJSON(report, to: URL(fileURLWithPath: output))
            print("Wrote JSON report to \(output)")
        }
    }

    private func runEngine(
        _ engine: DiarizationBenchEngine,
        inputs: [DiarizationInput]
    ) async throws -> DiarizationEngineResult {
        print("=== \(engine.name) ===")
        let preLoadRSS = benchmarkRSSBytes()
        var peakRSS = preLoadRSS

        let loadStart = Date()
        try await engine.load()
        if let warmup = inputs.first?.audio {
            _ = try engine.diarize(audio: Array(warmup.prefix(16_000)), sampleRate: 16000)
        }
        let loadElapsed = Date().timeIntervalSince(loadStart)
        peakRSS = max(peakRSS, benchmarkRSSBytes())
        print(String(format: "  loaded + warmed in %.1fs (RSS %.0f MB -> %.0f MB)",
                     loadElapsed,
                     Double(preLoadRSS) / (1024 * 1024),
                     Double(peakRSS) / (1024 * 1024)))

        var totalSpeech: Float = 0
        var falseAlarm: Float = 0
        var missedSpeech: Float = 0
        var confusion: Float = 0
        var derValues: [Double] = []
        var jerValues: [Double] = []
        var rtfValues: [Double] = []
        var elapsedTotal = 0.0
        var speakerCountCorrect = 0
        var failures: [String] = []

        for (idx, input) in inputs.enumerated() {
            do {
                let start = Date()
                let result = try engine.diarize(audio: input.audio, sampleRate: 16000)
                let elapsed = Date().timeIntervalSince(start)
                let der = computeDERWithOptimalMapping(
                    reference: input.reference,
                    hypothesis: result.segments,
                    collar: collar,
                    resolution: resolution
                )
                let jer = jaccardErrorPercent(
                    reference: input.reference,
                    hypothesis: result.segments,
                    collar: collar,
                    resolution: resolution
                )
                totalSpeech += der.totalSpeech
                falseAlarm += der.falseAlarm
                missedSpeech += der.missedSpeech
                confusion += der.confusion
                derValues.append(Double(der.derPercent))
                jerValues.append(Double(jer))
                rtfValues.append(elapsed / max(input.duration, 1e-6))
                elapsedTotal += elapsed

                let referenceSpeakers = Set(input.reference.map(\.speakerId)).count
                if referenceSpeakers == result.numSpeakers {
                    speakerCountCorrect += 1
                }

                if (idx + 1) % 5 == 0 || idx == inputs.count - 1 {
                    peakRSS = max(peakRSS, benchmarkRSSBytes())
                    let aggregateDER = totalSpeech > 0
                        ? Double((falseAlarm + missedSpeech + confusion) / totalSpeech * 100)
                        : 0
                    let processedAudio = inputs.prefix(idx + 1).reduce(0.0) { $0 + $1.duration }
                    print(String(format: "  [%4d/%4d] DER=%.2f%% xRT=%.1f peakRSS=%.0fMB",
                                 idx + 1, inputs.count,
                                 aggregateDER,
                                 elapsedTotal > 0 ? processedAudio / elapsedTotal : 0,
                                 Double(peakRSS) / (1024 * 1024)))
                }
            } catch {
                failures.append(input.entry.id)
                fputs("  file \(input.entry.id) failed: \(error)\n", stderr)
            }
        }

        let processedFiles = inputs.count - failures.count
        let audioTotal = inputs.reduce(0.0) { $0 + $1.duration }
        let derAggregate = totalSpeech > 0 ? Double((falseAlarm + missedSpeech + confusion) / totalSpeech * 100) : 0
        let rtfMean = BenchmarkMath.mean(rtfValues)
        let rtfMedian = BenchmarkMath.median(rtfValues)
        return DiarizationEngineResult(
            engine: engine.name,
            files: processedFiles,
            failures: failures,
            derPercent: derAggregate,
            derMeanPercent: BenchmarkMath.mean(derValues),
            derMedianPercent: BenchmarkMath.median(derValues),
            jerMeanPercent: BenchmarkMath.mean(jerValues),
            jerMedianPercent: BenchmarkMath.median(jerValues),
            missPercent: totalSpeech > 0 ? Double(missedSpeech / totalSpeech * 100) : 0,
            falseAlarmPercent: totalSpeech > 0 ? Double(falseAlarm / totalSpeech * 100) : 0,
            speakerErrorPercent: totalSpeech > 0 ? Double(confusion / totalSpeech * 100) : 0,
            totalSpeechSeconds: Double(totalSpeech),
            missedSpeechSeconds: Double(missedSpeech),
            falseAlarmSeconds: Double(falseAlarm),
            confusionSeconds: Double(confusion),
            speakerCountAccuracyPercent: BenchmarkMath.percent(Double(speakerCountCorrect), Double(max(processedFiles, 1))),
            rtfMean: rtfMean,
            rtfMedian: rtfMedian,
            throughputMeanXRT: rtfMean > 0 ? 1 / rtfMean : 0,
            throughputMedianXRT: rtfMedian > 0 ? 1 / rtfMedian : 0,
            throughputOverallXRT: elapsedTotal > 0 ? audioTotal / elapsedTotal : 0,
            loadElapsedSeconds: loadElapsed,
            audioSecondsTotal: audioTotal,
            elapsedSecondsTotal: elapsedTotal,
            peakRSSBytes: peakRSS,
            rssDeltaBytes: Int64(peakRSS) - Int64(preLoadRSS)
        )
    }
}

private struct DiarizationInput {
    let entry: BenchmarkManifestEntry
    let audio: [Float]
    let reference: [DiarizedSegment]
    var duration: Double { Double(audio.count) / 16000.0 }
}

private protocol DiarizationBenchEngine: AnyObject {
    var name: String { get }
    func load() async throws
    func diarize(audio: [Float], sampleRate: Int) throws -> DiarizationResult
}

private func makeDiarizationEngine(_ id: String) -> DiarizationBenchEngine? {
    switch id.lowercased() {
    case "community1-coreml":
        return Community1DiarizationBenchEngine()
    case "sortformer-default":
        return SortformerBenchEngine(name: "sortformer-default", variant: .default)
    case "sortformer-balanced":
        return SortformerBenchEngine(name: "sortformer-balanced", variant: .balanced)
    case "sortformer-streaming":
        return SortformerBenchEngine(name: "sortformer-streaming", variant: .streaming)
    case "pyannote-mlx":
        return PyannoteDiarizationBenchEngine(name: "pyannote-mlx", embeddingEngine: .mlx)
    case "pyannote-coreml":
        return PyannoteDiarizationBenchEngine(name: "pyannote-coreml", embeddingEngine: .coreml)
    default:
        return nil
    }
}

#if canImport(CoreML)
private final class Community1DiarizationBenchEngine: DiarizationBenchEngine {
    let name = "community1-coreml"
    var pipeline: Community1DiarizationPipeline?

    func load() async throws {
        pipeline = try await Community1DiarizationPipeline.fromPretrained(
            computeUnits: .cpuAndNeuralEngine
        )
        try pipeline?.prewarm()
    }

    func diarize(audio: [Float], sampleRate: Int) throws -> DiarizationResult {
        guard let pipeline else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }
        return try pipeline.diarize(audio: audio, sampleRate: sampleRate)
    }
}
#else
private final class Community1DiarizationBenchEngine: DiarizationBenchEngine {
    let name = "community1-coreml"
    func load() async throws {
        throw BenchmarkSupportError.unsupportedReference("Community-1 requires CoreML")
    }
    func diarize(audio: [Float], sampleRate: Int) throws -> DiarizationResult {
        DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
    }
}
#endif

private final class PyannoteDiarizationBenchEngine: DiarizationBenchEngine {
    let name: String
    let embeddingEngine: WeSpeakerEngine
    var pipeline: DiarizationPipeline?

    init(name: String, embeddingEngine: WeSpeakerEngine) {
        self.name = name
        self.embeddingEngine = embeddingEngine
    }

    func load() async throws {
        pipeline = try await DiarizationPipeline.fromPretrained(embeddingEngine: embeddingEngine)
    }

    func diarize(audio: [Float], sampleRate: Int) throws -> DiarizationResult {
        guard let pipeline else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }
        return pipeline.diarize(audio: audio, sampleRate: sampleRate)
    }
}

private final class SortformerBenchEngine: DiarizationBenchEngine {
    let name: String
    let variant: SortformerConfig
    var diarizer: SortformerDiarizer?

    init(name: String, variant: SortformerConfig) {
        self.name = name
        self.variant = variant
    }

    func load() async throws {
        #if canImport(CoreML)
        diarizer = try await SortformerDiarizer.fromPretrained(config: variant, computeUnits: .cpuAndNeuralEngine)
        #else
        throw BenchmarkSupportError.unsupportedReference("Sortformer requires CoreML")
        #endif
    }

    func diarize(audio: [Float], sampleRate: Int) throws -> DiarizationResult {
        guard let diarizer else {
            return DiarizationResult(segments: [], numSpeakers: 0, speakerEmbeddings: [])
        }
        return diarizer.diarize(audio: audio, sampleRate: sampleRate)
    }
}

public struct DiarizationEngineResult: Codable, Sendable {
    public let engine: String
    public let files: Int
    public let failures: [String]
    public let derPercent: Double
    public let derMeanPercent: Double
    public let derMedianPercent: Double
    public let jerMeanPercent: Double
    public let jerMedianPercent: Double
    public let missPercent: Double
    public let falseAlarmPercent: Double
    public let speakerErrorPercent: Double
    public let totalSpeechSeconds: Double
    public let missedSpeechSeconds: Double
    public let falseAlarmSeconds: Double
    public let confusionSeconds: Double
    public let speakerCountAccuracyPercent: Double
    public let rtfMean: Double
    public let rtfMedian: Double
    public let throughputMeanXRT: Double
    public let throughputMedianXRT: Double
    public let throughputOverallXRT: Double
    public let loadElapsedSeconds: Double
    public let audioSecondsTotal: Double
    public let elapsedSecondsTotal: Double
    public let peakRSSBytes: UInt64
    public let rssDeltaBytes: Int64
}

public struct DiarizationReport: Codable, Sendable {
    public let machine: String
    public let manifestPath: String
    public let files: Int
    public let collarSeconds: Float
    public let resolutionSeconds: Float
    public let results: [DiarizationEngineResult]
}

public enum DiarizationReportPrinter {
    public static func table(_ report: DiarizationReport) -> String {
        var s = ""
        s += "Machine: \(report.machine)\n"
        s += "Manifest: \(report.manifestPath)\n"
        s += "Files: \(report.files)\n\n"
        s += pad("Engine", 22) + "  " + pad("DER%", 7) + "  " + pad("JER%", 7) + "  " +
             pad("Miss%", 7) + "  " + pad("FA%", 7) + "  " + pad("SE%", 7) + "  " +
             pad("SpkAcc", 7) + "  " + pad("xRT", 8) + "  " + pad("Load(s)", 7) + "  " +
             pad("Peak(MB)", 8) + "\n"
        s += String(repeating: "-", count: 112) + "\n"
        for r in report.results {
            s += pad(r.engine, 22) + "  " +
                String(format: "%7.2f", r.derPercent) + "  " +
                String(format: "%7.2f", r.jerMeanPercent) + "  " +
                String(format: "%7.2f", r.missPercent) + "  " +
                String(format: "%7.2f", r.falseAlarmPercent) + "  " +
                String(format: "%7.2f", r.speakerErrorPercent) + "  " +
                String(format: "%7.1f", r.speakerCountAccuracyPercent) + "  " +
                String(format: "%8.1f", r.throughputOverallXRT) + "  " +
                String(format: "%7.1f", r.loadElapsedSeconds) + "  " +
                String(format: "%8.0f", Double(r.peakRSSBytes) / (1024 * 1024)) + "\n"
        }
        return s
    }

    public static func writeJSON(_ report: DiarizationReport, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url)
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return String(s.prefix(width)) }
        return s + String(repeating: " ", count: width - s.count)
    }
}

private func jaccardErrorPercent(
    reference: [DiarizedSegment],
    hypothesis: [DiarizedSegment],
    collar: Float,
    resolution: Float
) -> Float {
    let refSpeakers = Set(reference.map(\.speakerId)).sorted()
    let hypSpeakers = Set(hypothesis.map(\.speakerId)).sorted()
    guard !refSpeakers.isEmpty else { return hypothesis.isEmpty ? 0 : 100 }

    let all = reference + hypothesis
    guard let maxTime = all.map(\.endTime).max(), maxTime > 0 else { return 0 }
    let frames = max(1, Int(ceil(maxTime / resolution)))
    let scored = collarMask(reference: reference, frames: frames, resolution: resolution, collar: collar)
    let refFrameSets = speakerFrames(reference, speakers: refSpeakers, frames: frames, resolution: resolution, scored: scored)
    let hypFrameSets = speakerFrames(hypothesis, speakers: hypSpeakers, frames: frames, resolution: resolution, scored: scored)

    var matrix = [[Float]](repeating: [Float](repeating: 0, count: hypSpeakers.count), count: refSpeakers.count)
    for r in refSpeakers.indices {
        for h in hypSpeakers.indices {
            let intersection = refFrameSets[r].intersection(hypFrameSets[h]).count
            let union = refFrameSets[r].union(hypFrameSets[h]).count
            matrix[r][h] = union == 0 ? 0 : Float(intersection) / Float(union)
        }
    }

    let best = bestJaccardSum(matrix)
    return (1 - best / Float(refSpeakers.count)) * 100
}

private func collarMask(reference: [DiarizedSegment], frames: Int, resolution: Float, collar: Float) -> [Bool] {
    var scored = [Bool](repeating: true, count: frames)
    guard collar > 0 else { return scored }
    let collarFrames = Int(collar / resolution)
    for segment in reference {
        let start = Int(segment.startTime / resolution)
        let end = Int(segment.endTime / resolution)
        for idx in max(0, start - collarFrames)..<min(frames, start + collarFrames) {
            scored[idx] = false
        }
        for idx in max(0, end - collarFrames)..<min(frames, end + collarFrames) {
            scored[idx] = false
        }
    }
    return scored
}

private func speakerFrames(
    _ segments: [DiarizedSegment],
    speakers: [Int],
    frames: Int,
    resolution: Float,
    scored: [Bool]
) -> [Set<Int>] {
    let index = Dictionary(uniqueKeysWithValues: speakers.enumerated().map { ($1, $0) })
    var out = [Set<Int>](repeating: [], count: speakers.count)
    for segment in segments {
        guard let speakerIndex = index[segment.speakerId] else { continue }
        let start = max(0, Int(segment.startTime / resolution))
        let end = min(frames, Int(ceil(segment.endTime / resolution)))
        if end > start {
            for frame in start..<end where scored[frame] {
                out[speakerIndex].insert(frame)
            }
        }
    }
    return out
}

private func bestJaccardSum(_ matrix: [[Float]]) -> Float {
    guard !matrix.isEmpty else { return 0 }
    let refCount = matrix.count
    let hypCount = matrix[0].count
    guard hypCount > 0 else { return 0 }

    var used = [Bool](repeating: false, count: hypCount)
    var best: Float = 0

    func search(_ refIndex: Int, _ sum: Float) {
        if refIndex == refCount {
            best = max(best, sum)
            return
        }

        search(refIndex + 1, sum)
        for hypIndex in 0..<hypCount where !used[hypIndex] {
            used[hypIndex] = true
            search(refIndex + 1, sum + matrix[refIndex][hypIndex])
            used[hypIndex] = false
        }
    }

    if refCount <= 8 && hypCount <= 8 {
        search(0, 0)
        return best
    }

    var pairs: [(Float, Int, Int)] = []
    for r in 0..<refCount {
        for h in 0..<hypCount {
            pairs.append((matrix[r][h], r, h))
        }
    }
    pairs.sort { $0.0 > $1.0 }
    var usedRefs = Set<Int>()
    var usedHyps = Set<Int>()
    for (score, r, h) in pairs where !usedRefs.contains(r) && !usedHyps.contains(h) {
        best += score
        usedRefs.insert(r)
        usedHyps.insert(h)
    }
    return best
}
