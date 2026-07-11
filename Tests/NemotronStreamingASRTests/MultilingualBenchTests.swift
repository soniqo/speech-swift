import XCTest
import Darwin
import Foundation
@testable import NemotronStreamingASR
@testable import AudioCommon
import CoreML

/// FLEURS bench from Swift: 6 languages × 50 samples, captures WER + CER +
/// RTF + per-utterance latency + peak RSS. Output JSON lands at
/// `/tmp/nem35-logs/wer_swift.json` so it can be diffed against the Python
/// reference (`/tmp/nem35-logs/wer_coreml_new_all.json`).
///
/// Requires the FLEURS data materialized at
/// `/Users/ivan/repos/speech-models/benchmarks/fleurs/<lang>/` and the local
/// CoreML bundle at `/tmp/Nemotron-3.5-CoreML-320ms` (or set
/// `NEMOTRON_35_LOCAL_BUNDLE`).
final class E2ENemotronMultilingualBench: XCTestCase {

    private static let fleursRoot = ProcessInfo.processInfo.environment["FLEURS_ROOT"]
        ?? "/Users/ivan/repos/speech-models/benchmarks/fleurs"
    private static let outputPath = "/tmp/nem35-logs/wer_swift.json"
    private static let languages: [(dir: String, tag: String)] = [
        ("en_us", "en-US"), ("de_de", "de-DE"), ("fr_fr", "fr-FR"),
        ("ar_eg", "ar"),   ("hi_in", "hi-IN"), ("ja_jp", "ja-JP"),
    ]
    private static let samplesPerLang = 50

    func testBenchAllLanguages() async throws {
        let bundlePath = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"]
            ?? "/tmp/Nemotron-3.5-CoreML-320ms"
        let bundleURL = URL(fileURLWithPath: bundlePath)
        guard FileManager.default.fileExists(atPath: bundleURL.path) else {
            throw XCTSkip("Local bundle not found at \(bundlePath)")
        }

        let rssBeforeLoad = currentRSS()
        // One initial load to measure cold-start cost; per-language reloads
        // happen in the loop below to avoid CoreML IOSurface exhaustion after
        // 200+ sequential predicts in a single MLModel lifetime.
        print("Loading model from \(bundlePath)… (cold start measurement)")
        let loadStart = Date()
        var model: NemotronStreamingASRModel? = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundleURL)
        let rssAfterLoad = currentRSS()
        let loadElapsed = Date().timeIntervalSince(loadStart)
        print("  loaded in \(String(format: "%.2f", loadElapsed))s, RSS Δ +\(rssMB(rssAfterLoad - rssBeforeLoad)) MB → \(rssMB(rssAfterLoad)) MB")
        try model!.warmUp()
        let rssAfterWarmup = currentRSS()
        model!.unload(); model = nil

        var perLang: [String: LangResult] = [:]
        var peakRSS = rssAfterWarmup
        for (dir, tag) in Self.languages {
            // Fresh model per language: avoids cumulative CoreML state that
            // segfaults around the 250th predict in a single process.
            let langModel = try await NemotronStreamingASRModel.fromLocal(bundleDir: bundleURL)
            let result = try benchLanguage(model: langModel, dir: dir, tag: tag, peak: &peakRSS)
            perLang[dir] = result
            langModel.unload()
        }

        let summary = BenchSummary(
            bundle: bundlePath,
            samplesPerLang: Self.samplesPerLang,
            rssPreloadMB: rssMB(rssBeforeLoad),
            rssPostloadMB: rssMB(rssAfterLoad),
            rssPostwarmupMB: rssMB(rssAfterWarmup),
            rssPeakMB: rssMB(peakRSS),
            modelLoadSeconds: loadElapsed,
            perLanguage: perLang
        )

        try writeSummary(summary)
        printSummary(summary)
    }

    // MARK: - Per-language bench loop

    private func benchLanguage(
        model: NemotronStreamingASRModel,
        dir: String,
        tag: String,
        peak: inout Int
    ) throws -> LangResult {
        let langDir = URL(fileURLWithPath: Self.fleursRoot).appendingPathComponent(dir)
        let tsvURL = langDir.appendingPathComponent("test.tsv")
        guard FileManager.default.fileExists(atPath: tsvURL.path) else {
            print("\n[\(dir)] missing — skipping")
            return LangResult.empty(tag: tag)
        }
        let items = try loadFLEURS(tsvURL: tsvURL, audioDir: langDir.appendingPathComponent("audio/test"),
                                   limit: Self.samplesPerLang)
        print("\n[\(dir)] (\(tag)) — \(items.count) samples")

        var refs: [String] = []
        var hyps: [String] = []
        var perUtteranceMs: [Double] = []
        var totalAudioSeconds: Double = 0
        let langStart = Date()

        for (i, item) in items.enumerated() {
            // Wrap each utterance in autoreleasepool so the CoreML-returned
            // MLMultiArrays (which sit on IOSurface-backed Metal buffers) get
            // released before the next predict. Without this, IOSurface pool
            // is exhausted after ~10–30 utterances and predict() throws.
            autoreleasepool {
                let audio: [Float]
                do {
                    audio = try AudioFileLoader.load(url: item.url, targetSampleRate: 16000)
                } catch {
                    print("  [\(i)] load fail: \(error)")
                    return
                }
                totalAudioSeconds += Double(audio.count) / 16000.0
                let t0 = Date()
                let hyp: String
                do {
                    // FLEURS is natural speech with ambient lead-in — skip the
                    // 100 ms silence padding that helps TTS-style sharp onsets.
                    hyp = try model.transcribeAudio(audio, sampleRate: 16000, language: tag, padSilence: false)
                } catch {
                    print("  [\(i)] transcribe fail: \(error)")
                    return
                }
                let dt = Date().timeIntervalSince(t0) * 1000
                perUtteranceMs.append(dt)
                refs.append(normalize(item.ref, langDir: dir))
                hyps.append(normalize(hyp, langDir: dir))
                let cur = currentRSS()
                if cur > peak { peak = cur }
            }
        }

        let wallMs = Date().timeIntervalSince(langStart) * 1000
        let werPct = werPercent(refs: refs, hyps: hyps)
        let cerPct = cerPercent(refs: refs, hyps: hyps)
        let rtf = totalAudioSeconds > 0 ? (wallMs / 1000.0) / totalAudioSeconds : 0
        let sortedMs = perUtteranceMs.sorted()
        let p50 = sortedMs.isEmpty ? 0 : sortedMs[sortedMs.count / 2]
        let p95 = sortedMs.isEmpty ? 0 : sortedMs[min(sortedMs.count - 1, Int(Double(sortedMs.count) * 0.95))]
        let p99 = sortedMs.isEmpty ? 0 : sortedMs[min(sortedMs.count - 1, Int(Double(sortedMs.count) * 0.99))]

        print(String(format: "  WER=%.2f%%  CER=%.2f%%  audio=%.1fs  wall=%.0fms  RTF=%.4f  p50=%.0fms  p99=%.0fms",
                     werPct, cerPct, totalAudioSeconds, wallMs, rtf, p50, p99))

        return LangResult(
            tag: tag, samples: refs.count,
            werPct: werPct, cerPct: cerPct,
            audioSeconds: totalAudioSeconds,
            wallMs: wallMs, rtf: rtf,
            p50Ms: p50, p95Ms: p95, p99Ms: p99
        )
    }

    // MARK: - FLEURS TSV loader

    private struct FleursItem {
        let url: URL
        let ref: String
    }

    private func loadFLEURS(tsvURL: URL, audioDir: URL, limit: Int) throws -> [FleursItem] {
        let content = try String(contentsOf: tsvURL, encoding: .utf8)
        var out: [FleursItem] = []
        // Use components(separatedBy: .newlines) so we handle both LF and
        // CRLF files (Swift's String.split treats "\r\n" as a single grapheme
        // cluster and won't split on the \n inside it).
        for rawLine in content.components(separatedBy: .newlines) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty { continue }
            let cols = line.split(separator: "\t", omittingEmptySubsequences: false)
            guard cols.count >= 4 else { continue }
            let wavName = String(cols[1])
            let normalized = String(cols[3])
            let wavURL = audioDir.appendingPathComponent(wavName)
            if FileManager.default.fileExists(atPath: wavURL.path) {
                out.append(FleursItem(url: wavURL, ref: normalized))
                if out.count >= limit { break }
            }
        }
        return out
    }

    // MARK: - Normalization (Whisper BasicTextNormalizer port)

    /// Languages without word boundaries / with complex grapheme clusters —
    /// score char-level (`splitLetters=true`) to match NVIDIA's methodology.
    private static let splitLetterLangs: Set<String> = [
        "ja_jp", "zh_cn", "ko_kr", "th_th", "hi_in",
    ]

    private static let langTagRegex = try! NSRegularExpression(pattern: "<[a-zA-Z-]+>")
    private static let bracketRegex = try! NSRegularExpression(pattern: "[\\[<][^\\]>]*[\\]>]")
    private static let parenRegex = try! NSRegularExpression(pattern: "\\(([^)]+?)\\)")
    private static let multiSpaceRegex = try! NSRegularExpression(pattern: "\\s+")
    private static let englishNormalizer = WhisperEnglishNormalizer()

    private func normalize(_ text: String, langDir: String) -> String {
        var t = text
        var r = NSRange(t.startIndex..., in: t)
        t = Self.langTagRegex.stringByReplacingMatches(in: t, range: r, withTemplate: "")
        // English: full WhisperEnglishNormalizer (contractions, fillers, etc.).
        if langDir == "en_us" {
            return Self.englishNormalizer(t)
        }
        // Other languages: BasicTextNormalizer behavior (NFKC + M/S/P → space).
        t = t.lowercased()
        r = NSRange(t.startIndex..., in: t)
        t = Self.bracketRegex.stringByReplacingMatches(in: t, range: r, withTemplate: "")
        r = NSRange(t.startIndex..., in: t)
        t = Self.parenRegex.stringByReplacingMatches(in: t, range: r, withTemplate: "")
        let nfkc = t.precomposedStringWithCompatibilityMapping
        var out = ""
        out.reserveCapacity(nfkc.unicodeScalars.count)
        for scalar in nfkc.unicodeScalars {
            switch scalar.properties.generalCategory {
            case .nonspacingMark, .enclosingMark, .spacingMark,
                 .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
                 .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
                 .initialPunctuation, .finalPunctuation, .otherPunctuation:
                out.unicodeScalars.append(Unicode.Scalar(0x20)!)
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        t = out.lowercased()
        if Self.splitLetterLangs.contains(langDir) {
            t = t.map { String($0) }.joined(separator: " ")
        }
        r = NSRange(t.startIndex..., in: t)
        t = Self.multiSpaceRegex.stringByReplacingMatches(in: t, range: r, withTemplate: " ")
        return t.trimmingCharacters(in: .whitespaces)
    }

    // MARK: - WER / CER

    private func werPercent(refs: [String], hyps: [String]) -> Double {
        guard refs.count == hyps.count, !refs.isEmpty else { return 0 }
        var totalDist = 0
        var totalRef = 0
        for (r, h) in zip(refs, hyps) {
            let rW = r.split(separator: " ").map(String.init)
            let hW = h.split(separator: " ").map(String.init)
            totalDist += levenshtein(rW, hW)
            totalRef += rW.count
        }
        return totalRef == 0 ? 0 : Double(totalDist) / Double(totalRef) * 100
    }

    private func cerPercent(refs: [String], hyps: [String]) -> Double {
        guard refs.count == hyps.count, !refs.isEmpty else { return 0 }
        var totalDist = 0
        var totalRef = 0
        for (r, h) in zip(refs, hyps) {
            let rC = Array(r)
            let hC = Array(h)
            totalDist += levenshtein(rC, hC)
            totalRef += rC.count
        }
        return totalRef == 0 ? 0 : Double(totalDist) / Double(totalRef) * 100
    }

    private func levenshtein<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        if a.isEmpty { return b.count }
        if b.isEmpty { return a.count }
        var prev = Array(0...b.count)
        var curr = [Int](repeating: 0, count: b.count + 1)
        for i in 1...a.count {
            curr[0] = i
            for j in 1...b.count {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                curr[j] = Swift.min(prev[j] + 1, Swift.min(curr[j - 1] + 1, prev[j - 1] + cost))
            }
            swap(&prev, &curr)
        }
        return prev[b.count]
    }

    // MARK: - RSS

    private func currentRSS() -> Int {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? Int(info.resident_size) : 0
    }

    private func rssMB(_ bytes: Int) -> Int { bytes / (1024 * 1024) }

    // MARK: - Output

    private struct LangResult: Codable {
        let tag: String
        let samples: Int
        let werPct: Double
        let cerPct: Double
        let audioSeconds: Double
        let wallMs: Double
        let rtf: Double
        let p50Ms: Double
        let p95Ms: Double
        let p99Ms: Double

        static func empty(tag: String) -> LangResult {
            LangResult(tag: tag, samples: 0, werPct: 0, cerPct: 0,
                       audioSeconds: 0, wallMs: 0, rtf: 0,
                       p50Ms: 0, p95Ms: 0, p99Ms: 0)
        }
    }

    private struct BenchSummary: Codable {
        let bundle: String
        let samplesPerLang: Int
        let rssPreloadMB: Int
        let rssPostloadMB: Int
        let rssPostwarmupMB: Int
        let rssPeakMB: Int
        let modelLoadSeconds: Double
        let perLanguage: [String: LangResult]
    }

    private func writeSummary(_ s: BenchSummary) throws {
        let dir = URL(fileURLWithPath: Self.outputPath).deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(s)
        try data.write(to: URL(fileURLWithPath: Self.outputPath))
        print("\nwrote \(Self.outputPath)")
    }

    private func printSummary(_ s: BenchSummary) {
        print("\n========== Swift FLEURS bench summary ==========")
        print(String(format: "  bundle=%@  load=%.2fs", s.bundle, s.modelLoadSeconds))
        print("  RSS pre=\(s.rssPreloadMB) MB  postload=\(s.rssPostloadMB) MB  postwarmup=\(s.rssPostwarmupMB) MB  peak=\(s.rssPeakMB) MB")
        print(String(format: "  %-8s | %7s | %7s | %7s | %7s | %7s", "lang", "WER%", "CER%", "RTF", "p50ms", "p99ms"))
        for (dir, _) in Self.languages {
            guard let r = s.perLanguage[dir] else { continue }
            print(String(format: "  %-8s | %7.2f | %7.2f | %7.4f | %7.0f | %7.0f",
                         dir, r.werPct, r.cerPct, r.rtf, r.p50Ms, r.p99Ms))
        }
    }
}
