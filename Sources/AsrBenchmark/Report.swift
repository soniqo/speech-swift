import Foundation

public struct EngineResult: Codable, Sendable {
    public var engine: String
    public var utterances: Int
    public var werPercent: Double
    public var substitutions: Int
    public var insertions: Int
    public var deletions: Int
    public var referenceWords: Int
    public var rtfMean: Double          // mean per-utterance RTF after warmup
    public var rtfFirst: Double         // first transcription (cold-after-load)
    public var loadElapsedSeconds: Double
    public var audioSecondsTotal: Double
    public var elapsedSecondsTotal: Double
    public var peakRSSBytes: UInt64     // peak resident memory observed during run
    public var rssDeltaBytes: Int64     // peak RSS minus pre-load RSS
    public var throughputXRT: Double    // audio seconds per wall second (= 1 / rtf)
}

public struct Report: Codable, Sendable {
    public var machine: String
    public var datasetPath: String
    public var utterances: Int
    public var results: [EngineResult]
}

public enum ReportPrinter {
    public static func table(_ report: Report) -> String {
        var s = ""
        s += "Machine: \(report.machine)\n"
        s += "Dataset: \(report.datasetPath)\n"
        s += "Utterances: \(report.utterances)\n\n"
        // NOTE: Swift `String` is NOT a C string; `%s` with a Swift String is
        // undefined behavior (typically segfault). Use `%@` for objects, and
        // pad/truncate manually for strings to keep the table clean.
        s += pad("Engine", 32) + "  " + pad("WER%", 6) + "  " + pad("RTF", 6) + "  " +
             pad("xRT", 6) + "  " + pad("Load(s)", 7) + "  " + pad("Peak(MB)", 8) + "  " +
             pad("Δ(MB)", 7) + "\n"
        s += String(repeating: "-", count: 88) + "\n"
        for r in report.results {
            s += pad(r.engine, 32) + "  " +
                 String(format: "%6.2f", r.werPercent) + "  " +
                 String(format: "%6.3f", r.rtfMean) + "  " +
                 String(format: "%6.1f", r.throughputXRT) + "  " +
                 String(format: "%7.1f", r.loadElapsedSeconds) + "  " +
                 String(format: "%8.0f", Double(r.peakRSSBytes) / (1024 * 1024)) + "  " +
                 String(format: "%+7.0f", Double(r.rssDeltaBytes) / (1024 * 1024)) + "\n"
        }
        return s
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return String(s.prefix(width)) }
        return s + String(repeating: " ", count: width - s.count)
    }

    public static func writeJSON(_ report: Report, to url: URL) throws {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try enc.encode(report)
        try data.write(to: url)
    }
}
