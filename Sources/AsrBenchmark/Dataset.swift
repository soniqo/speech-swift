import Foundation
import AudioCommon

public struct Utterance: Sendable {
    public let id: String
    public let audioURL: URL
    public let reference: String
}

public enum Dataset {
    /// Walks a LibriSpeech-style directory (e.g. `LibriSpeech/test-clean`) and
    /// returns one `Utterance` per audio file. Layout:
    ///
    ///     test-clean/<speaker>/<chapter>/<speaker>-<chapter>.trans.txt
    ///     test-clean/<speaker>/<chapter>/<speaker>-<chapter>-NNNN.flac
    ///
    /// `trans.txt` is `<id> <transcript>` per line. Audio can be `.flac` or
    /// `.wav` — AVAudioFile handles both.
    public static func loadLibriSpeech(root: URL, limit: Int? = nil) throws -> [Utterance] {
        let fm = FileManager.default
        guard let walker = fm.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) else {
            throw BenchError.datasetEmpty(root.path)
        }
        var transcripts: [String: String] = [:]
        var audioPaths: [(id: String, url: URL)] = []
        for case let url as URL in walker {
            let ext = url.pathExtension.lowercased()
            if ext == "txt" && url.lastPathComponent.hasSuffix(".trans.txt") {
                let content = try String(contentsOf: url, encoding: .utf8)
                for raw in content.split(separator: "\n") {
                    let line = raw.trimmingCharacters(in: .whitespaces)
                    guard let spaceIdx = line.firstIndex(of: " ") else { continue }
                    let id = String(line[..<spaceIdx])
                    let text = String(line[line.index(after: spaceIdx)...])
                    transcripts[id] = text
                }
            } else if ext == "flac" || ext == "wav" {
                let id = url.deletingPathExtension().lastPathComponent
                audioPaths.append((id: id, url: url))
            }
        }
        audioPaths.sort { $0.id < $1.id }

        var out: [Utterance] = []
        for entry in audioPaths {
            guard let ref = transcripts[entry.id] else { continue }
            out.append(Utterance(id: entry.id, audioURL: entry.url, reference: ref))
            if let limit, out.count >= limit { break }
        }
        if out.isEmpty {
            throw BenchError.datasetEmpty(root.path)
        }
        return out
    }

    /// Loads a TSV manifest: `<audio path>\t<reference text>` per line.
    /// Audio paths can be absolute or relative to the manifest file.
    public static func loadManifest(url: URL, limit: Int? = nil) throws -> [Utterance] {
        let content = try String(contentsOf: url, encoding: .utf8)
        let base = url.deletingLastPathComponent()
        var out: [Utterance] = []
        for (idx, raw) in content.split(separator: "\n").enumerated() {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            guard parts.count == 2 else {
                throw BenchError.datasetMalformed("line \(idx + 1) is not <path>\\t<text>")
            }
            let p = String(parts[0])
            let audioURL = p.hasPrefix("/") ? URL(fileURLWithPath: p) : base.appendingPathComponent(p)
            out.append(Utterance(id: audioURL.deletingPathExtension().lastPathComponent,
                                 audioURL: audioURL,
                                 reference: String(parts[1])))
            if let limit, out.count >= limit { break }
        }
        if out.isEmpty { throw BenchError.datasetEmpty(url.path) }
        return out
    }

    /// Decodes an utterance to 16 kHz mono Float32 (the rate every engine
    /// here expects). Resampling is handled by `AudioFileLoader`.
    public static func loadAudio16k(_ utt: Utterance) throws -> [Float] {
        return try AudioFileLoader.load(url: utt.audioURL, targetSampleRate: 16000)
    }
}
