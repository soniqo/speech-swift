import ArgumentParser
import Foundation

@main
struct AsrBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "asr-bench",
        abstract: "Benchmark ASR engines on the same dataset (WER + RTF, normalized)."
    )

    @Option(name: .shortAndLong, help: "Path to LibriSpeech-style directory (e.g. .../test-clean) OR a .tsv manifest.")
    var dataset: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Engines to run. Valid: \(EngineID.allCases.map { $0.rawValue }.joined(separator: ", ")).")
    var engines: [String] = ["qwen3-coreml", "parakeet", "whisperkit-large-v3-turbo"]

    @Option(name: .shortAndLong, help: "Max utterances per engine (omit = full set).")
    var limit: Int?

    @Option(name: .shortAndLong, help: "Optional path to write the JSON report.")
    var output: String?

    @Option(name: .long, help: "ISO language hint passed to engines that accept one.")
    var language: String?

    @Flag(name: .long,
          help: "Run each engine in a separate process so RSS and physical-footprint high-water marks reflect one engine.")
    var isolated: Bool = false

    @Flag(name: .long, help: "Print reference/hypothesis pairs for utterances with WER errors.")
    var printErrors: Bool = false

    mutating func run() async throws {
        // stdout is block-buffered when the process runs with redirected
        // output (background tasks, tee, etc.); progress lines get lost if
        // the buffer never flushes. Line-buffer it so progress is live.
        setlinebuf(stdout)

        if isolated && engines.count > 1 {
            try await runIsolated()
            return
        }

        let datasetURL = URL(fileURLWithPath: dataset)
        let utterances: [Utterance]
        if datasetURL.pathExtension.lowercased() == "tsv" {
            utterances = try Dataset.loadManifest(url: datasetURL, limit: limit)
        } else {
            utterances = try Dataset.loadLibriSpeech(root: datasetURL, limit: limit)
        }
        print("Loaded \(utterances.count) utterances from \(dataset)")

        // Pre-decode all audio once — keeps each engine fair (audio I/O cost
        // is not what we're measuring) and matches what a streaming pipeline
        // would see.
        print("Decoding audio (16 kHz mono)...")
        let decoded: [(utt: Utterance, audio: [Float])] = try utterances.map {
            (utt: $0, audio: try Dataset.loadAudio16k($0))
        }
        let totalAudioSec = decoded.reduce(0.0) { $0 + Double($1.audio.count) / 16000.0 }
        print(String(format: "Total audio: %.1f s\n", totalAudioSec))

        // Resolve engines.
        var engineImpls: [BenchEngine] = []
        for name in engines {
            guard let id = EngineID(rawValue: name) else {
                fputs("Unknown engine: \(name)\n", stderr)
                throw ExitCode(2)
            }
            engineImpls.append(id.make())
        }

        // First utterance doubles as the warmup audio so every engine pays
        // its compile cost up front and the per-utterance loop measures
        // hot-path latency.
        let warmup = decoded[0].audio

        var results: [EngineResult] = []
        for engine in engineImpls {
            print("=== \(engine.name) ===")
            let preLoadMemory = currentProcessMemoryUsage()
            var peakRSS = preLoadMemory.residentBytes
            var peakPhysicalFootprint = preLoadMemory.peakPhysicalFootprintBytes
            do {
                try await engine.load(warmupAudio: warmup, sampleRate: 16000)
                let loadedMemory = currentProcessMemoryUsage()
                peakRSS = max(peakRSS, loadedMemory.residentBytes)
                peakPhysicalFootprint = max(
                    peakPhysicalFootprint,
                    loadedMemory.peakPhysicalFootprintBytes)
                print(String(format: "  loaded + warmed in %.1fs (RSS %.0f MB → %.0f MB, footprint %.0f MB)",
                             engine.loadElapsed,
                             Double(preLoadMemory.residentBytes) / (1024 * 1024),
                             Double(peakRSS) / (1024 * 1024),
                             Double(peakPhysicalFootprint) / (1024 * 1024)))
            } catch {
                fputs("  load failed: \(error)\n", stderr)
                continue
            }

            var subs = 0, ins = 0, dels = 0, refWords = 0
            var charErrors = 0, refChars = 0
            var perFileWERs: [Double] = []
            var perFileCERs: [Double] = []
            var rtfs: [Double] = []
            var totalElapsed = 0.0
            var processedAudioSec = 0.0
            for (idx, item) in decoded.enumerated() {
                do {
                    let (text, t) = try await engine.transcribe(audio: item.audio, sampleRate: 16000, language: language)
                    let hyp = Normalizer.normalize(text)
                    let ref = Normalizer.normalize(item.utt.reference)
                    let b = WER.compute(reference: ref, hypothesis: hyp)
                    let c = CER.compute(reference: ref, hypothesis: hyp)
                    if printErrors && b.totalErrors > 0 {
                        print(String(format: "  %@ WER=%.2f%% S/I/D=%d/%d/%d",
                                     item.utt.id as NSString,
                                     b.wer * 100,
                                     b.substitutions,
                                     b.insertions,
                                     b.deletions))
                        print("    ref: \(ref)")
                        print("    hyp: \(hyp)")
                    }
                    subs += b.substitutions
                    ins += b.insertions
                    dels += b.deletions
                    refWords += b.referenceWords
                    charErrors += c.editDistance
                    refChars += c.referenceCharacters
                    perFileWERs.append(b.wer * 100)
                    perFileCERs.append(c.cer * 100)
                    rtfs.append(t.rtf)
                    totalElapsed += t.elapsed
                    processedAudioSec += t.audioDuration
                    // Sample outside the timed transcription interval. Physical
                    // footprint catches file-backed MLX allocations that RSS can
                    // miss, and sampling every utterance avoids missing a short
                    // peak between progress updates.
                    let memory = currentProcessMemoryUsage()
                    peakRSS = max(peakRSS, memory.residentBytes)
                    peakPhysicalFootprint = max(
                        peakPhysicalFootprint,
                        memory.peakPhysicalFootprintBytes)
                    if (idx + 1) % 25 == 0 || idx == decoded.count - 1 {
                        print(String(format: "  [%4d/%4d] rolling WER=%.2f%% RTF=%.3f peakRSS=%.0fMB footprint=%.0fMB",
                                     idx + 1, decoded.count,
                                     Double(subs + ins + dels) / Double(max(refWords, 1)) * 100,
                                     rtfs.reduce(0, +) / Double(rtfs.count),
                                     Double(peakRSS) / (1024 * 1024),
                                     Double(peakPhysicalFootprint) / (1024 * 1024)))
                    }
                } catch {
                    fputs("  utterance \(item.utt.id) failed: \(error)\n", stderr)
                }
            }

            let meanRTF = mean(rtfs)
            let medianRTF = median(rtfs)
            let aggregateWER = Double(subs + ins + dels) / Double(max(refWords, 1)) * 100
            let aggregateCER = Double(charErrors) / Double(max(refChars, 1)) * 100
            let throughputMeanXRT = meanRTF > 0 ? 1.0 / meanRTF : 0
            let throughputMedianXRT = medianRTF > 0 ? 1.0 / medianRTF : 0
            let throughputOverallXRT = totalElapsed > 0 ? processedAudioSec / totalElapsed : 0
            results.append(EngineResult(
                engine: engine.name,
                utterances: rtfs.count,
                werPercent: aggregateWER,
                werAggregatePercent: aggregateWER,
                werMeanPercent: mean(perFileWERs),
                werMedianPercent: median(perFileWERs),
                cerPercent: aggregateCER,
                cerAggregatePercent: aggregateCER,
                cerMeanPercent: mean(perFileCERs),
                cerMedianPercent: median(perFileCERs),
                substitutions: subs,
                insertions: ins,
                deletions: dels,
                referenceWords: refWords,
                characterErrors: charErrors,
                referenceCharacters: refChars,
                rtfMean: meanRTF,
                rtfMedian: medianRTF,
                rtfFirst: rtfs.first ?? 0,
                loadElapsedSeconds: engine.loadElapsed,
                audioSecondsTotal: processedAudioSec,
                elapsedSecondsTotal: totalElapsed,
                peakRSSBytes: peakRSS,
                rssDeltaBytes: Int64(peakRSS) - Int64(preLoadMemory.residentBytes),
                peakPhysicalFootprintBytes: peakPhysicalFootprint,
                physicalFootprintDeltaBytes: Int64(peakPhysicalFootprint)
                    - Int64(preLoadMemory.physicalFootprintBytes),
                throughputXRT: throughputMeanXRT,
                throughputMeanXRT: throughputMeanXRT,
                throughputMedianXRT: throughputMedianXRT,
                throughputOverallXRT: throughputOverallXRT
            ))
        }

        let report = Report(
            machine: ProcessInfo.processInfo.hostName,
            datasetPath: dataset,
            utterances: decoded.count,
            results: results
        )

        print("\n" + ReportPrinter.table(report))

        if let output {
            try ReportPrinter.writeJSON(report, to: URL(fileURLWithPath: output))
            print("Wrote JSON report to \(output)")
        }
    }

    /// Re-executes the bench once per engine in a child process. Each child
    /// gets a fresh MLX/CoreML state, so its process-memory high-water marks
    /// reflect only its own allocations, not a previous engine's cache.
    private func runIsolated() async throws {
        let selfPath = CommandLine.arguments[0]
        let runID = UUID().uuidString.prefix(8)
        var merged: [EngineResult] = []
        var failures: [String] = []
        for engineName in engines {
            let tempOut = "/tmp/asr-bench-iso-\(runID)-\(engineName).json"
            var args: [String] = [
                "--dataset", dataset,
                "--engines", engineName,
                "--output", tempOut
            ]
            if let limit { args += ["--limit", "\(limit)"] }
            if let language { args += ["--language", language] }

            print("\n──── isolated subrun: \(engineName) ────")
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: selfPath)
            proc.arguments = args
            proc.standardOutput = FileHandle.standardOutput
            proc.standardError = FileHandle.standardError
            try proc.run()
            proc.waitUntilExit()

            if proc.terminationStatus == 0,
               let data = try? Data(contentsOf: URL(fileURLWithPath: tempOut)),
               let child = try? JSONDecoder().decode(Report.self, from: data) {
                merged.append(contentsOf: child.results)
                try? FileManager.default.removeItem(atPath: tempOut)
            } else {
                failures.append(engineName)
                fputs("subrun for \(engineName) failed (exit \(proc.terminationStatus))\n", stderr)
            }
        }

        let datasetURL = URL(fileURLWithPath: dataset)
        let uttCount: Int
        if datasetURL.pathExtension.lowercased() == "tsv" {
            uttCount = (try? Dataset.loadManifest(url: datasetURL, limit: limit).count) ?? 0
        } else {
            uttCount = (try? Dataset.loadLibriSpeech(root: datasetURL, limit: limit).count) ?? 0
        }
        let report = Report(
            machine: ProcessInfo.processInfo.hostName,
            datasetPath: dataset,
            utterances: uttCount,
            results: merged
        )

        print("\n" + ReportPrinter.table(report))
        if !failures.isEmpty {
            fputs("Failed subruns: \(failures.joined(separator: ", "))\n", stderr)
        }
        if let output {
            try ReportPrinter.writeJSON(report, to: URL(fileURLWithPath: output))
            print("Wrote merged JSON report to \(output)")
        }
    }
}

private func mean(_ values: [Double]) -> Double {
    values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
}

private func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    if sorted.count.isMultiple(of: 2) {
        return (sorted[mid - 1] + sorted[mid]) / 2
    }
    return sorted[mid]
}
