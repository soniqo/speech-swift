import ArgumentParser
import AudioCommon
import BenchmarkSupport
import Foundation
import SpeechVAD

@main
struct VadBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vad-bench",
        abstract: "Benchmark VAD engines on a manifest of audio plus reference speech segments."
    )

    @Option(name: .shortAndLong, help: "Manifest: <audio path>\\t<reference path>[\\t<id>]. References may be RTTM or start/end text.")
    var manifest: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Engines: silero-coreml, silero-mlx, firered, pyannote.")
    var engines: [String] = ["silero-coreml", "silero-mlx", "firered", "pyannote"]

    @Option(name: .shortAndLong, help: "Max files to process.")
    var limit: Int?

    @Option(name: .shortAndLong, help: "Optional path to write JSON report.")
    var output: String?

    @Option(name: .long, help: "Frame scoring resolution in seconds.")
    var resolution: Double = 0.01

    @Option(name: .long, help: "Speech activity ratio required for file-level speech/no-speech scoring.")
    var fileActivityThreshold: Double = 0.1

    @Option(name: .long, help: "Silero onset threshold.")
    var sileroOnset: Float = VADConfig.sileroDefault.onset

    @Option(name: .long, help: "Silero offset threshold.")
    var sileroOffset: Float = VADConfig.sileroDefault.offset

    @Option(name: .long, help: "Silero minimum speech duration in seconds.")
    var sileroMinSpeech: Float = VADConfig.sileroDefault.minSpeechDuration

    @Option(name: .long, help: "Silero minimum silence duration in seconds.")
    var sileroMinSilence: Float = VADConfig.sileroDefault.minSilenceDuration

    mutating func run() async throws {
        setlinebuf(stdout)

        let manifestURL = URL(fileURLWithPath: manifest)
        let entries = try BenchmarkManifest.load(url: manifestURL, limit: limit)
        print("Loaded \(entries.count) VAD benchmark file(s) from \(manifest)")

        let decoded = try entries.map { entry in
            let audio = try AudioFileLoader.load(url: entry.audioURL, targetSampleRate: 16000)
            let reference = try SegmentReference.loadSpeechSegments(url: entry.referenceURL)
            return VADInput(entry: entry, audio: audio, reference: reference)
        }
        let totalAudio = decoded.reduce(0.0) { $0 + $1.duration }
        print(String(format: "Total audio: %.1f s\n", totalAudio))

        var results: [VADEngineResult] = []
        let sileroConfig = VADConfig(
            onset: sileroOnset,
            offset: sileroOffset,
            minSpeechDuration: sileroMinSpeech,
            minSilenceDuration: sileroMinSilence,
            windowDuration: VADConfig.sileroDefault.windowDuration,
            stepRatio: VADConfig.sileroDefault.stepRatio
        )
        for engineID in engines {
            guard let engine = makeEngine(engineID, sileroConfig: sileroConfig) else {
                fputs("Unknown VAD engine: \(engineID)\n", stderr)
                throw ExitCode(2)
            }
            results.append(try await runEngine(engine, inputs: decoded))
        }

        let report = VADReport(
            machine: ProcessInfo.processInfo.hostName,
            manifestPath: manifest,
            files: decoded.count,
            fileActivityThreshold: fileActivityThreshold,
            sileroOnset: sileroOnset,
            sileroOffset: sileroOffset,
            sileroMinSpeech: sileroMinSpeech,
            sileroMinSilence: sileroMinSilence,
            results: results
        )

        print("\n" + VADReportPrinter.table(report))
        if let output {
            try VADReportPrinter.writeJSON(report, to: URL(fileURLWithPath: output))
            print("Wrote JSON report to \(output)")
        }
    }

    private func runEngine(_ engine: VADBenchEngine, inputs: [VADInput]) async throws -> VADEngineResult {
        print("=== \(engine.name) ===")
        let preLoadRSS = benchmarkRSSBytes()
        var peakRSS = preLoadRSS
        let loadStart = Date()
        try await engine.load()
        if let warmup = inputs.first?.audio {
            _ = try engine.detect(audio: Array(warmup.prefix(16_000)), sampleRate: 16000)
        }
        let loadElapsed = Date().timeIntervalSince(loadStart)
        peakRSS = max(peakRSS, benchmarkRSSBytes())
        print(String(format: "  loaded + warmed in %.1fs (RSS %.0f MB -> %.0f MB)",
                     loadElapsed,
                     Double(preLoadRSS) / (1024 * 1024),
                     Double(peakRSS) / (1024 * 1024)))

        var counts = VADFrameCounts()
        var elapsedTotal = 0.0
        var perFileRTF: [Double] = []
        var perFileF1: [Double] = []
        var failures: [String] = []
        var fileTruePositive = 0
        var fileFalsePositive = 0
        var fileFalseNegative = 0
        var fileTrueNegative = 0

        for (idx, input) in inputs.enumerated() {
            do {
                let start = Date()
                let hypothesis = try engine.detect(audio: input.audio, sampleRate: 16000)
                let elapsed = Date().timeIntervalSince(start)
                let fileCounts = VADScoring.score(
                    reference: input.reference,
                    hypothesis: hypothesis,
                    duration: input.duration,
                    resolution: resolution
                )
                counts.add(fileCounts)
                elapsedTotal += elapsed
                perFileRTF.append(elapsed / max(input.duration, 1e-6))
                perFileF1.append(Self.f1Percent(fileCounts))
                let referenceHasSpeech = Self.activityRatio(input.reference, duration: input.duration) >= fileActivityThreshold
                let hypothesisHasSpeech = Self.activityRatio(hypothesis, duration: input.duration) >= fileActivityThreshold
                switch (hypothesisHasSpeech, referenceHasSpeech) {
                case (true, true):
                    fileTruePositive += 1
                case (true, false):
                    fileFalsePositive += 1
                case (false, true):
                    fileFalseNegative += 1
                case (false, false):
                    fileTrueNegative += 1
                }
                if (idx + 1) % 10 == 0 || idx == inputs.count - 1 {
                    peakRSS = max(peakRSS, benchmarkRSSBytes())
                    print(String(format: "  [%4d/%4d] F1=%.2f%% xRT=%.1f peakRSS=%.0fMB",
                                 idx + 1, inputs.count,
                                 Self.f1Percent(counts),
                                 elapsedTotal > 0 ? inputs.prefix(idx + 1).reduce(0.0) { $0 + $1.duration } / elapsedTotal : 0,
                                 Double(peakRSS) / (1024 * 1024)))
                }
            } catch {
                failures.append(input.entry.id)
                fputs("  file \(input.entry.id) failed: \(error)\n", stderr)
            }
        }

        let audioTotal = inputs.reduce(0.0) { $0 + $1.duration }
        let rtfMean = BenchmarkMath.mean(perFileRTF)
        let rtfMedian = BenchmarkMath.median(perFileRTF)
        let filePrecision = Self.percent(fileTruePositive, fileTruePositive + fileFalsePositive)
        let fileRecall = Self.percent(fileTruePositive, fileTruePositive + fileFalseNegative)
        return VADEngineResult(
            engine: engine.name,
            files: inputs.count - failures.count,
            failures: failures,
            truePositiveFrames: counts.truePositive,
            falsePositiveFrames: counts.falsePositive,
            falseNegativeFrames: counts.falseNegative,
            trueNegativeFrames: counts.trueNegative,
            fileTruePositives: fileTruePositive,
            fileFalsePositives: fileFalsePositive,
            fileFalseNegatives: fileFalseNegative,
            fileTrueNegatives: fileTrueNegative,
            fileAccuracyPercent: Self.percent(fileTruePositive + fileTrueNegative, max(inputs.count - failures.count, 1)),
            filePrecisionPercent: filePrecision,
            fileRecallPercent: fileRecall,
            fileF1Percent: filePrecision + fileRecall > 0 ? (2 * filePrecision * fileRecall) / (filePrecision + fileRecall) : 0,
            accuracyPercent: BenchmarkMath.percent(Double(counts.truePositive + counts.trueNegative), Double(max(counts.total, 1))),
            precisionPercent: BenchmarkMath.percent(Double(counts.truePositive), Double(max(counts.truePositive + counts.falsePositive, 1))),
            recallPercent: BenchmarkMath.percent(Double(counts.truePositive), Double(max(counts.truePositive + counts.falseNegative, 1))),
            f1Percent: Self.f1Percent(counts),
            f1MeanPercent: BenchmarkMath.mean(perFileF1),
            f1MedianPercent: BenchmarkMath.median(perFileF1),
            falseAlarmRatePercent: BenchmarkMath.percent(Double(counts.falsePositive), Double(max(counts.nonSpeech, 1))),
            missRatePercent: BenchmarkMath.percent(Double(counts.falseNegative), Double(max(counts.speech, 1))),
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

    private static func f1Percent(_ counts: VADFrameCounts) -> Double {
        let tp = Double(counts.truePositive)
        let fp = Double(counts.falsePositive)
        let fn = Double(counts.falseNegative)
        let denom = (2 * tp) + fp + fn
        return denom == 0 ? 0 : (2 * tp) / denom * 100
    }

    private static func percent(_ numerator: Int, _ denominator: Int) -> Double {
        denominator == 0 ? 0 : Double(numerator) / Double(denominator) * 100
    }

    private static func activityRatio(_ segments: [SpeechSegment], duration: Double) -> Double {
        guard duration > 0 else { return 0 }
        let speechSeconds = segments.reduce(0.0) { total, segment in
            total + max(0, Double(segment.endTime - segment.startTime))
        }
        return speechSeconds / duration
    }
}

private struct VADInput {
    let entry: BenchmarkManifestEntry
    let audio: [Float]
    let reference: [SpeechSegment]
    var duration: Double { Double(audio.count) / 16000.0 }
}

private protocol VADBenchEngine: AnyObject {
    var name: String { get }
    func load() async throws
    func detect(audio: [Float], sampleRate: Int) throws -> [SpeechSegment]
}

private func makeEngine(_ id: String, sileroConfig: VADConfig) -> VADBenchEngine? {
    switch id.lowercased() {
    case "silero-coreml":
        return SileroBenchEngine(engine: .coreml, config: sileroConfig)
    case "silero-mlx":
        return SileroBenchEngine(engine: .mlx, config: sileroConfig)
    case "firered":
        return FireRedBenchEngine()
    case "pyannote":
        return PyannoteBenchEngine()
    default:
        return nil
    }
}

private final class SileroBenchEngine: VADBenchEngine {
    let engine: SileroVADEngine
    let config: VADConfig
    var model: SileroVADModel?

    init(engine: SileroVADEngine, config: VADConfig) {
        self.engine = engine
        self.config = config
    }

    var name: String { "silero-\(engine.rawValue)" }

    func load() async throws {
        model = try await SileroVADModel.fromPretrained(engine: engine)
    }

    func detect(audio: [Float], sampleRate: Int) throws -> [SpeechSegment] {
        guard let model else { return [] }
        return model.detectSpeech(audio: audio, sampleRate: sampleRate, config: config)
    }
}

private final class FireRedBenchEngine: VADBenchEngine {
    var model: FireRedVADModel?
    let name = "firered-coreml"

    func load() async throws {
        model = try await FireRedVADModel.fromPretrained()
    }

    func detect(audio: [Float], sampleRate: Int) throws -> [SpeechSegment] {
        guard let model else { return [] }
        return model.detectSpeech(audio: audio, sampleRate: sampleRate)
    }
}

private final class PyannoteBenchEngine: VADBenchEngine {
    var model: PyannoteVADModel?
    let name = "pyannote-mlx"

    func load() async throws {
        model = try await PyannoteVADModel.fromPretrained()
    }

    func detect(audio: [Float], sampleRate: Int) throws -> [SpeechSegment] {
        guard let model else { return [] }
        return model.detectSpeech(audio: audio, sampleRate: sampleRate)
    }
}

public struct VADEngineResult: Codable, Sendable {
    public let engine: String
    public let files: Int
    public let failures: [String]
    public let truePositiveFrames: Int
    public let falsePositiveFrames: Int
    public let falseNegativeFrames: Int
    public let trueNegativeFrames: Int
    public let fileTruePositives: Int
    public let fileFalsePositives: Int
    public let fileFalseNegatives: Int
    public let fileTrueNegatives: Int
    public let fileAccuracyPercent: Double
    public let filePrecisionPercent: Double
    public let fileRecallPercent: Double
    public let fileF1Percent: Double
    public let accuracyPercent: Double
    public let precisionPercent: Double
    public let recallPercent: Double
    public let f1Percent: Double
    public let f1MeanPercent: Double
    public let f1MedianPercent: Double
    public let falseAlarmRatePercent: Double
    public let missRatePercent: Double
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

public struct VADReport: Codable, Sendable {
    public let machine: String
    public let manifestPath: String
    public let files: Int
    public let fileActivityThreshold: Double
    public let sileroOnset: Float
    public let sileroOffset: Float
    public let sileroMinSpeech: Float
    public let sileroMinSilence: Float
    public let results: [VADEngineResult]
}

public enum VADReportPrinter {
    public static func table(_ report: VADReport) -> String {
        var s = ""
        s += "Machine: \(report.machine)\n"
        s += "Manifest: \(report.manifestPath)\n"
        s += "Files: \(report.files)\n\n"
        s += pad("Engine", 20) + "  " + pad("F1%", 7) + "  " + pad("Prec%", 7) + "  " +
             pad("Rec%", 7) + "  " + pad("FAR%", 7) + "  " + pad("MR%", 7) + "  " +
             pad("xRT", 8) + "  " + pad("Load(s)", 7) + "  " + pad("Peak(MB)", 8) + "\n"
        s += String(repeating: "-", count: 95) + "\n"
        for r in report.results {
            s += pad(r.engine, 20) + "  " +
                String(format: "%7.2f", r.f1Percent) + "  " +
                String(format: "%7.2f", r.precisionPercent) + "  " +
                String(format: "%7.2f", r.recallPercent) + "  " +
                String(format: "%7.2f", r.falseAlarmRatePercent) + "  " +
                String(format: "%7.2f", r.missRatePercent) + "  " +
                String(format: "%8.1f", r.throughputOverallXRT) + "  " +
                String(format: "%7.1f", r.loadElapsedSeconds) + "  " +
                String(format: "%8.0f", Double(r.peakRSSBytes) / (1024 * 1024)) + "\n"
        }
        return s
    }

    public static func writeJSON(_ report: VADReport, to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(to: url)
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return String(s.prefix(width)) }
        return s + String(repeating: " ", count: width - s.count)
    }
}
