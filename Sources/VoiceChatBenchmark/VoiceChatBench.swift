import ArgumentParser
import Darwin
import Foundation
import MLX
import MLXNN
import VoiceChat

/// Benchmarks the VoiceChat perception front end: how long the streaming
/// FastConformer plus modality projection take to turn audio into language-model
/// embeddings, at a range of utterance lengths.
///
/// Measures the encode path only. The language backbone and the duplex loop are
/// separate concerns with separate cost profiles, and mixing them into one
/// number hides which half is slow.
///
/// Real-time factor is the figure that matters for a duplex model: the encoder
/// runs continuously while the user speaks, so anything at or above 1.0 means
/// the model cannot keep up with live audio.
@main
struct VoiceChatBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voicechat-bench",
        abstract: "Benchmark VoiceChat perception encode latency and real-time factor."
    )

    @Option(name: .shortAndLong, help: "Path to an exported bundle directory containing encoder/.")
    var model: String

    @Option(name: .shortAndLong, parsing: .upToNextOption,
            help: "Utterance lengths to test, in seconds.")
    var durations: [Double] = [1, 5, 15, 30]

    @Option(name: .long, help: "Timed iterations per length.")
    var iterations: Int = 5

    @Option(name: .long, help: "Warmup iterations, excluded from timings.")
    var warmup: Int = 2

    @Option(name: .shortAndLong, help: "Optional path to write a JSON report.")
    var output: String?

    @Option(name: .long, help: "Parity check: safetensors holding a `mel` array to encode instead of benchmarking.")
    var parityInput: String?

    @Option(name: .long, help: "Where to write the parity result (safetensors, key `embeddings`).")
    var parityOutput: String?

    @Flag(name: .long, help: "Parity-check the language backbone instead of the encoder. Input key `tokens`, output key `logits`.")
    var llm = false

    /// Resident set size of this process, in bytes.
    ///
    /// Peak GPU memory alone understates what a duplex model costs: weights are
    /// mapped into the process, so RSS is what actually decides whether a
    /// bundle fits alongside everything else on the machine.
    private func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    private func pad(_ text: String, _ width: Int) -> String {
        String(repeating: " ", count: max(0, width - text.count)) + text
    }

    // 16 kHz audio, 10 ms hop, matching the exported preprocessor.
    private static let melHopSeconds = 0.01

    struct Measurement: Codable {
        let seconds: Double
        let melFrames: Int
        let encodedFrames: Int
        let meanMs: Double
        let p50Ms: Double
        let minMs: Double
        let maxMs: Double
        let realTimeFactor: Double
        let msPerEncodedFrame: Double
    }

    struct Report: Codable {
        let bundle: String
        let quantization: String
        let encoderLayers: Int
        let dModel: Int
        let attContextSize: [Int]
        let peakMemoryGB: Double
        let residentMemoryGB: Double
        let measurements: [Measurement]
    }

    mutating func run() async throws {
        // Unbuffered: this benchmark is usually piped, and a block-buffered
        // stdout loses every line if a later stage crashes.
        setvbuf(stdout, nil, _IONBF, 0)
        let root = URL(fileURLWithPath: model)
        // The language backbone is only loaded when asked for: it is 10-19 GB
        // against the encoder's 1.25, so paying for it by default would make
        // every encoder run needlessly slow.
        let languageModel = llm
            ? try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))
            : nil
        // LLM parity needs no encoder, and loading 1.25 GB of weights that are
        // never used just slows the check down.
        let llmOnly = llm && parityInput != nil
        let perception = llmOnly
            ? nil
            : try VoiceChatPerception.load(from: root.appendingPathComponent("encoder"))

        guard let perception else {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: parityInput!))
            guard let tokens = arrays["tokens"] else {
                throw ValidationError("no `tokens` array in \(parityInput!)")
            }
            let logits = languageModel!(tokens)
            eval(logits)
            try MLX.save(arrays: ["logits": logits], url: URL(fileURLWithPath: parityOutput!))
            print("parity: tokens \(tokens.shape) -> logits \(logits.shape)")
            return
        }
        let config = perception.config
        let quantization = config.quantization.map { "\($0.bits)-bit g\($0.groupSize)" } ?? "fp16"
        print("bundle        \(model)")
        print("quantization  \(quantization)")
        print("encoder       \(config.encoder.nLayers) layers, d_model \(config.encoder.dModel), "
              + "attention context \(config.encoder.attContextSize)")
        print("")
        // Note: String(format:) with %s expects a C string, not a Swift String,
        // and crashes if given one. Pad manually instead.
        print(pad("audio", 8) + pad("mel", 10) + pad("encoded", 10)
              + pad("mean ms", 10) + pad("RTF", 9) + pad("ms/frame", 12))
        print(String(repeating: "-", count: 64))

        // Parity mode: encode one supplied input and write the result, so the
        // Swift port can be diffed against the Python reference element-wise.
        // A benchmark that measures a numerically wrong model is worse than no
        // benchmark, so this runs before any timing is trusted.
        if llm, let parityInput, let parityOutput {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: parityInput))
            guard let tokens = arrays["tokens"] else {
                throw ValidationError("no `tokens` array in \(parityInput)")
            }
            let logits = languageModel!(tokens)
            eval(logits)
            try MLX.save(arrays: ["logits": logits], url: URL(fileURLWithPath: parityOutput))
            print("parity: tokens \(tokens.shape) -> logits \(logits.shape) -> \(parityOutput)")
            return
        }

        if let parityInput, let parityOutput {
            let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: parityInput))
            guard let mel = arrays["mel"] else {
                throw ValidationError("no `mel` array in \(parityInput)")
            }
            let embeddings = perception(mel)
            eval(embeddings)
            try MLX.save(arrays: ["embeddings": embeddings],
                         url: URL(fileURLWithPath: parityOutput))
            print("parity: mel \(mel.shape) -> embeddings \(embeddings.shape) -> \(parityOutput)")
            return
        }

        var measurements: [Measurement] = []

        for seconds in durations {
            let melFrames = Int(seconds / Self.melHopSeconds)
            let input = MLXRandom.normal([1, melFrames, config.encoder.featIn])
            eval(input)

            for _ in 0 ..< warmup {
                let out = perception(input)
                eval(out)
            }

            var samples: [Double] = []
            var encodedFrames = 0
            for _ in 0 ..< iterations {
                let start = DispatchTime.now().uptimeNanoseconds
                let out = perception(input)
                eval(out)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                samples.append(elapsed)
                encodedFrames = out.shape[1]
            }

            let sorted = samples.sorted()
            let mean = samples.reduce(0, +) / Double(samples.count)
            let measurement = Measurement(
                seconds: seconds,
                melFrames: melFrames,
                encodedFrames: encodedFrames,
                meanMs: mean,
                p50Ms: sorted[sorted.count / 2],
                minMs: sorted.first ?? 0,
                maxMs: sorted.last ?? 0,
                realTimeFactor: (mean / 1000) / seconds,
                msPerEncodedFrame: mean / Double(max(encodedFrames, 1))
            )
            measurements.append(measurement)

            print(pad(String(format: "%.0fs", seconds), 8)
                  + pad("\(melFrames)", 10)
                  + pad("\(encodedFrames)", 10)
                  + pad(String(format: "%.1f", measurement.meanMs), 10)
                  + pad(String(format: "%.4f", measurement.realTimeFactor), 9)
                  + pad(String(format: "%.3f", measurement.msPerEncodedFrame), 12))
        }

        let peakGB = Double(GPU.peakMemory) / 1e9
        let rssGB = Double(residentBytes()) / 1e9
        print("")
        print(String(format: "peak GPU memory  %.2f GB", peakGB))
        print(String(format: "resident memory  %.2f GB", rssGB))

        // Where the time actually goes, at the longest length tested.
        if let longest = durations.max() {
            let mel = MLXRandom.normal([1, Int(longest / Self.melHopSeconds), config.encoder.featIn])
            eval(mel)
            _ = perception.profile(mel)                   // warm
            let stages = perception.profile(mel)
            print("")
            print("stage breakdown at \(Int(longest))s:")
            let total = stages.totalMs
            for (name, ms) in [("subsampling", stages.subsamplingMs),
                               ("conformer", stages.conformerMs),
                               ("projection", stages.projectionMs)] {
                print(String(format: "  %-12s %7.1f ms  %4.1f%%", (name as NSString).utf8String!,
                             ms, ms / total * 100))
            }
        }

        let slowest = measurements.map(\.realTimeFactor).max() ?? 0
        if slowest >= 1.0 {
            print("WARNING: real-time factor \(String(format: "%.2f", slowest)) — "
                  + "the encoder cannot keep up with live audio.")
        }

        if let output {
            let report = Report(
                bundle: model,
                quantization: quantization,
                encoderLayers: config.encoder.nLayers,
                dModel: config.encoder.dModel,
                attContextSize: config.encoder.attContextSize,
                peakMemoryGB: peakGB,
                residentMemoryGB: rssGB,
                measurements: measurements
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(report).write(to: URL(fileURLWithPath: output))
            print("wrote \(output)")
        }
    }
}
