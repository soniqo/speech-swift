import Foundation
import CoreML
import AudioCommon
import ParakeetStreamingASR
import OmnilingualASR
import SupertonicTTS
import KokoroTTS
import FunctionGemma

/// One benchmarked model's result. Codable → written to `results.json`.
struct BenchRow: Codable, Identifiable, Sendable {
    var id: String
    var label: String          // card row label, e.g. "Streaming ASR + EOU"
    var model: String          // e.g. "Parakeet-EOU 120M · CoreML INT8"
    var metric: String         // headline: "0.35 RTF" or "48 tok/s"
    var rtf: Double?
    var tokensPerSecond: Double?
    var audioSeconds: Double?
    var wallSeconds: Double?
    var peakMB: Double
    var loadSeconds: Double
    var error: String?
}

/// Runs each CoreML model in isolation (load → warm-up → N timed runs → release)
/// and reports RTF / tokens-per-second + peak `phys_footprint`. Plain class run
/// from a detached task so heavy inference never blocks the main thread.
final class BenchmarkCore: @unchecked Sendable {
    typealias StatusFn = @Sendable (String) -> Void
    typealias RowFn = @Sendable (BenchRow) -> Void

    private let ttsText =
        "The quick brown fox jumps over the lazy dog. On-device speech synthesis runs entirely on the Neural Engine."

    private var onStatus: StatusFn = { _ in }
    private var onRow: RowFn = { _ in }

    func run(onStatus: @escaping StatusFn, onRow: @escaping RowFn) async {
        self.onStatus = onStatus
        self.onRow = onRow
        log("=== iOS CoreML benchmark starting ===")
        // Supertonic last — its HF download has been flaky; ordering it last means a
        // stall there can't block the other four, and each model has a hard timeout.
        await bench { try await self.benchParakeetEOU() }
        await bench { try await self.benchOmnilingual() }
        await bench { try await self.benchKokoro() }
        await bench { try await self.benchFunctionGemma() }
        await bench { try await self.benchSupertonic() }
        writeJSON()
        log("=== benchmark DONE ===")
    }

    // MARK: - Per-model benchmarks

    private func benchParakeetEOU() async throws -> BenchRow {
        log("Parakeet-EOU: loading…")
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await ParakeetStreamingASRModel.fromPretrained { p, s in
            self.log(String(format: "Parakeet-EOU load %.0f%% %@", p * 100, s))
        }
        let load = CFAbsoluteTimeGetCurrent() - t0

        let (audio, seconds) = try loadClip("parakeet_eou", sampleRate: 16000)
        try? model.warmUp()
        _ = try model.transcribeAudio(audio, sampleRate: 16000)   // warm-up on real clip

        let (wall, peak) = try timed(runs: 5) {
            _ = try model.transcribeAudio(audio, sampleRate: 16000)
        }
        let rtf = wall / seconds
        return BenchRow(
            id: "asr-eou", label: "Streaming ASR + EOU",
            model: "Parakeet-EOU 120M · CoreML INT8",
            metric: String(format: "%.2f RTF", rtf),
            rtf: rtf, tokensPerSecond: nil, audioSeconds: seconds, wallSeconds: wall,
            peakMB: peak, loadSeconds: load, error: nil)
    }

    private func benchOmnilingual() async throws -> BenchRow {
        log("Omnilingual: loading…")
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await OmnilingualASRModel.fromPretrained { p, s in
            self.log(String(format: "Omnilingual load %.0f%% %@", p * 100, s))
        }
        let load = CFAbsoluteTimeGetCurrent() - t0

        let (audio, seconds) = try loadClip("omnilingual_en", sampleRate: 16000)
        _ = try model.transcribeAudio(audio, sampleRate: 16000, language: "en")   // warm-up

        let (wall, peak) = try timed(runs: 5) {
            _ = try model.transcribeAudio(audio, sampleRate: 16000, language: "en")
        }
        let rtf = wall / seconds
        return BenchRow(
            id: "asr-omni", label: "Multilingual ASR",
            model: "Omnilingual 300M · CoreML INT8",
            metric: String(format: "%.2f RTF", rtf),
            rtf: rtf, tokensPerSecond: nil, audioSeconds: seconds, wallSeconds: wall,
            peakMB: peak, loadSeconds: load, error: nil)
    }

    private func benchSupertonic() async throws -> BenchRow {
        log("Supertonic-3: loading…")
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await SupertonicTTSModel.fromPretrained { p, s in
            self.log(String(format: "Supertonic load %.0f%% %@", p * 100, s))
        }
        let load = CFAbsoluteTimeGetCurrent() - t0
        let sr = 44100.0

        _ = try model.synthesize(text: ttsText, language: "en")   // warm-up

        var seconds = 0.0
        let (wall, peak) = try timed(runs: 5) {
            let pcm = try model.synthesize(text: self.ttsText, language: "en")
            seconds = Double(pcm.count) / sr
        }
        let rtf = wall / seconds
        return BenchRow(
            id: "tts-supertonic", label: "TTS",
            model: "Supertonic-3 99M · CoreML",
            metric: String(format: "%.2f RTF", rtf),
            rtf: rtf, tokensPerSecond: nil, audioSeconds: seconds, wallSeconds: wall,
            peakMB: peak, loadSeconds: load, error: nil)
    }

    private func benchKokoro() async throws -> BenchRow {
        log("Kokoro-82M: loading…")
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await KokoroTTSModel.fromPretrained { p, s in
            self.log(String(format: "Kokoro load %.0f%% %@", p * 100, s))
        }
        let load = CFAbsoluteTimeGetCurrent() - t0
        let sr = Double(KokoroTTSModel.outputSampleRate)

        _ = try model.synthesize(text: ttsText)   // warm-up

        var seconds = 0.0
        let (wall, peak) = try timed(runs: 5) {
            let pcm = try model.synthesize(text: self.ttsText)
            seconds = Double(pcm.count) / sr
        }
        let rtf = wall / seconds
        return BenchRow(
            id: "tts-kokoro", label: "TTS",
            model: "Kokoro-82M · CoreML",
            metric: String(format: "%.2f RTF", rtf),
            rtf: rtf, tokensPerSecond: nil, audioSeconds: seconds, wallSeconds: wall,
            peakMB: peak, loadSeconds: load, error: nil)
    }

    private func benchFunctionGemma() async throws -> BenchRow {
        log("FunctionGemma: loading…")
        let t0 = CFAbsoluteTimeGetCurrent()
        let model = try await FunctionGemma.loadFromHub(computeUnits: .cpuAndNeuralEngine)
        let load = CFAbsoluteTimeGetCurrent() - t0

        let tool = FunctionDeclaration(
            name: "get_weather",
            description: "Get the current weather for a city",
            parameters: [
                "type": "object",
                "properties": ["city": ["type": "string", "description": "City name"]],
                "required": ["city"],
            ])
        let prompt = "What is the weather in Paris right now?"

        _ = try await model.generate(prompt: prompt, tools: [tool], maxNewTokens: 48)  // warm-up

        let monitor = MemoryMonitor(); monitor.reset(); monitor.start()
        var rates: [Double] = []
        for _ in 0..<3 {
            _ = try await model.generate(prompt: prompt, tools: [tool], maxNewTokens: 48)
            rates.append(model.lastMetrics.tokensPerSecond)
        }
        monitor.stop()
        let tps = median(rates)
        return BenchRow(
            id: "llm-gemma", label: "LLM tokens/s",
            model: "FunctionGemma 270M · CoreML ANE",
            metric: String(format: "%.0f tok/s", tps),
            rtf: nil, tokensPerSecond: tps, audioSeconds: nil, wallSeconds: nil,
            peakMB: monitor.peakMB, loadSeconds: load, error: nil)
    }

    // MARK: - Helpers

    private func bench(_ fn: @escaping @Sendable () async throws -> BenchRow) async {
        do {
            let row = try await withTimeout(seconds: 420, fn)
            log("✓ \(row.label): \(row.metric) · \(Int(row.peakMB)) MB (load \(String(format: "%.1f", row.loadSeconds))s)")
            collected.append(row)
            onRow(row)
        } catch {
            let row = BenchRow(
                id: "err-\(collected.count)", label: "(failed)",
                model: "\(error)", metric: "—",
                rtf: nil, tokensPerSecond: nil, audioSeconds: nil, wallSeconds: nil,
                peakMB: 0, loadSeconds: 0, error: "\(error)")
            log("✗ FAILED: \(error)")
            collected.append(row)
            onRow(row)
        }
        // Let memory settle before loading the next model.
        try? await Task.sleep(nanoseconds: 1_500_000_000)
    }

    /// Run `body` `runs` times while sampling peak memory; return median wall + peak MB.
    private func timed(runs: Int, _ body: () throws -> Void) rethrows -> (Double, Double) {
        let monitor = MemoryMonitor(); monitor.reset(); monitor.start()
        var walls: [Double] = []
        for _ in 0..<runs {
            let t0 = CFAbsoluteTimeGetCurrent()
            try body()
            walls.append(CFAbsoluteTimeGetCurrent() - t0)
        }
        monitor.stop()
        return (median(walls), monitor.peakMB)
    }

    private func loadClip(_ name: String, sampleRate: Int) throws -> ([Float], Double) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "wav") else {
            throw BenchError.missingResource(name)
        }
        let audio = try AudioFileLoader.load(url: url, targetSampleRate: sampleRate)
        return (audio, Double(audio.count) / Double(sampleRate))
    }

    private func median(_ xs: [Double]) -> Double {
        let s = xs.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    private func writeJSON() {
        let all = collected
        guard let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = dir.appendingPathComponent("results.json")
        do {
            let enc = JSONEncoder(); enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            try enc.encode(all).write(to: url)
            log("wrote \(url.path)")
        } catch {
            log("JSON write failed: \(error)")
        }
    }

    private var collected: [BenchRow] = []

    private func log(_ s: String) {
        print("[BENCH] \(s)")
        onStatus(s)
    }

    /// Race `op` against a timeout so a hung download can never block the suite.
    private func withTimeout(seconds: Double, _ op: @escaping @Sendable () async throws -> BenchRow) async throws -> BenchRow {
        try await withThrowingTaskGroup(of: BenchRow.self) { group in
            group.addTask { try await op() }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                throw BenchError.timeout(seconds)
            }
            guard let first = try await group.next() else { throw BenchError.timeout(seconds) }
            group.cancelAll()
            return first
        }
    }

    enum BenchError: Error, CustomStringConvertible {
        case missingResource(String)
        case timeout(Double)
        var description: String {
            switch self {
            case .missingResource(let n): return "missing bundled resource \(n).wav"
            case .timeout(let s): return "timed out (>\(Int(s))s)"
            }
        }
    }
}
