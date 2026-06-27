import AudioCommon
import Darwin
import Foundation
import SpeechVAD

public struct BenchmarkManifestEntry: Sendable {
    public let id: String
    public let audioURL: URL
    public let referenceURL: URL

    public init(id: String, audioURL: URL, referenceURL: URL) {
        self.id = id
        self.audioURL = audioURL
        self.referenceURL = referenceURL
    }
}

public enum BenchmarkSupportError: Error, CustomStringConvertible {
    case emptyManifest(String)
    case malformedManifestLine(Int)
    case missingFile(String)
    case unsupportedReference(String)

    public var description: String {
        switch self {
        case .emptyManifest(let path): return "manifest has no entries: \(path)"
        case .malformedManifestLine(let line): return "manifest line \(line) is not <audio>\\t<reference>[\\t<id>]"
        case .missingFile(let path): return "file does not exist: \(path)"
        case .unsupportedReference(let path): return "unsupported reference format: \(path)"
        }
    }
}

public enum BenchmarkManifest {
    /// Loads `<audio path>\t<reference path>[\t<id>]`.
    ///
    /// Relative paths are resolved against the manifest directory. Reference
    /// files may be RTTM or simple segment text with `start end` per line.
    public static func load(url: URL, limit: Int? = nil) throws -> [BenchmarkManifestEntry] {
        let content = try String(contentsOf: url, encoding: .utf8)
        let base = url.deletingLastPathComponent()
        var entries: [BenchmarkManifestEntry] = []

        for (idx, raw) in content.split(separator: "\n").enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(separator: "\t", omittingEmptySubsequences: false).map(String.init)
            guard parts.count >= 2 else {
                throw BenchmarkSupportError.malformedManifestLine(idx + 1)
            }
            let audio = resolve(parts[0], relativeTo: base)
            let reference = resolve(parts[1], relativeTo: base)
            let id = parts.count >= 3 && !parts[2].isEmpty
                ? parts[2]
                : audio.deletingPathExtension().lastPathComponent

            guard FileManager.default.fileExists(atPath: audio.path) else {
                throw BenchmarkSupportError.missingFile(audio.path)
            }
            guard FileManager.default.fileExists(atPath: reference.path) else {
                throw BenchmarkSupportError.missingFile(reference.path)
            }

            entries.append(BenchmarkManifestEntry(id: id, audioURL: audio, referenceURL: reference))
            if let limit, entries.count >= limit { break }
        }

        if entries.isEmpty {
            throw BenchmarkSupportError.emptyManifest(url.path)
        }
        return entries
    }

    private static func resolve(_ path: String, relativeTo base: URL) -> URL {
        path.hasPrefix("/") ? URL(fileURLWithPath: path) : base.appendingPathComponent(path)
    }
}

public enum SegmentReference {
    public static func loadSpeechSegments(url: URL) throws -> [SpeechSegment] {
        if url.pathExtension.lowercased() == "rttm" {
            let diarized = try loadDiarizedSegments(url: url)
            return mergeSpeechSegments(diarized.map {
                SpeechSegment(startTime: $0.startTime, endTime: $0.endTime)
            })
        }

        let content = try String(contentsOf: url, encoding: .utf8)
        var segments: [SpeechSegment] = []
        for raw in content.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let fields = line
                .replacingOccurrences(of: ",", with: " ")
                .split(whereSeparator: \.isWhitespace)
                .map(String.init)
            guard fields.count >= 2,
                  let start = Float(fields[0]),
                  let end = Float(fields[1]),
                  end > start else { continue }
            segments.append(SpeechSegment(startTime: start, endTime: end))
        }
        return mergeSpeechSegments(segments)
    }

    public static func loadDiarizedSegments(url: URL) throws -> [DiarizedSegment] {
        if url.pathExtension.lowercased() == "rttm" {
            let content = try String(contentsOf: url, encoding: .utf8)
            return parseRTTM(content)
        }

        let content = try String(contentsOf: url, encoding: .utf8)
        var segments: [DiarizedSegment] = []
        var speakerMap: [String: Int] = [:]
        var nextSpeaker = 0
        for raw in content.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let fields = line
                .replacingOccurrences(of: ",", with: " ")
                .split(whereSeparator: \.isWhitespace)
                .map(String.init)
            guard fields.count >= 3,
                  let start = Float(fields[0]),
                  let end = Float(fields[1]),
                  end > start else { continue }
            let speaker = fields[2]
            if speakerMap[speaker] == nil {
                speakerMap[speaker] = nextSpeaker
                nextSpeaker += 1
            }
            segments.append(DiarizedSegment(startTime: start, endTime: end, speakerId: speakerMap[speaker]!))
        }
        return segments.sorted { $0.startTime < $1.startTime }
    }

    public static func mergeSpeechSegments(_ segments: [SpeechSegment], maxGap: Float = 0) -> [SpeechSegment] {
        let sorted = segments.sorted { $0.startTime < $1.startTime }
        guard var current = sorted.first else { return [] }
        var merged: [SpeechSegment] = []

        for segment in sorted.dropFirst() {
            if segment.startTime <= current.endTime + maxGap {
                current = SpeechSegment(
                    startTime: current.startTime,
                    endTime: max(current.endTime, segment.endTime)
                )
            } else {
                merged.append(current)
                current = segment
            }
        }
        merged.append(current)
        return merged
    }
}

public struct VADFrameCounts: Codable, Sendable {
    public var truePositive: Int
    public var falsePositive: Int
    public var falseNegative: Int
    public var trueNegative: Int

    public init(truePositive: Int = 0, falsePositive: Int = 0, falseNegative: Int = 0, trueNegative: Int = 0) {
        self.truePositive = truePositive
        self.falsePositive = falsePositive
        self.falseNegative = falseNegative
        self.trueNegative = trueNegative
    }

    public var total: Int { truePositive + falsePositive + falseNegative + trueNegative }
    public var speech: Int { truePositive + falseNegative }
    public var nonSpeech: Int { trueNegative + falsePositive }

    public mutating func add(_ other: VADFrameCounts) {
        truePositive += other.truePositive
        falsePositive += other.falsePositive
        falseNegative += other.falseNegative
        trueNegative += other.trueNegative
    }
}

public enum VADScoring {
    public static func score(
        reference: [SpeechSegment],
        hypothesis: [SpeechSegment],
        duration: Double,
        resolution: Double
    ) -> VADFrameCounts {
        let frames = max(1, Int(ceil(duration / resolution)))
        let refMask = mask(segments: reference, frames: frames, resolution: resolution)
        let hypMask = mask(segments: hypothesis, frames: frames, resolution: resolution)
        var counts = VADFrameCounts()

        for idx in 0..<frames {
            switch (refMask[idx], hypMask[idx]) {
            case (true, true): counts.truePositive += 1
            case (false, true): counts.falsePositive += 1
            case (true, false): counts.falseNegative += 1
            case (false, false): counts.trueNegative += 1
            }
        }
        return counts
    }

    private static func mask(segments: [SpeechSegment], frames: Int, resolution: Double) -> [Bool] {
        var out = [Bool](repeating: false, count: frames)
        for segment in segments {
            let start = max(0, Int(floor(Double(segment.startTime) / resolution)))
            let end = min(frames, Int(ceil(Double(segment.endTime) / resolution)))
            if end > start {
                for idx in start..<end { out[idx] = true }
            }
        }
        return out
    }
}

public enum BenchmarkMath {
    public static func mean(_ values: [Double]) -> Double {
        values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
    }

    public static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        if sorted.count.isMultiple(of: 2) {
            return (sorted[mid - 1] + sorted[mid]) / 2
        }
        return sorted[mid]
    }

    public static func percent(_ numerator: Double, _ denominator: Double) -> Double {
        denominator == 0 ? 0 : numerator / denominator * 100
    }
}

public func benchmarkRSSBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kerr = withUnsafeMutablePointer(to: &info) { ptr -> kern_return_t in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { p in
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), p, &count)
        }
    }
    guard kerr == KERN_SUCCESS else { return 0 }
    return UInt64(info.resident_size)
}
