import ArgumentParser
import AudioCommon
import Darwin
import Foundation
import MLX
import MLXNN
import VoiceChat

/// Benchmarks the VoiceChat perception front end or runs the complete
/// duplex speech-to-speech pipeline over a real input file.
///
/// The default mode measures how long the streaming
/// FastConformer plus modality projection take to turn audio into language-model
/// embeddings, at a range of utterance lengths.
///
/// Real-time factor is the figure that matters for a duplex model: the encoder
/// runs continuously while the user speaks, so anything at or above 1.0 means
/// the model cannot keep up with live audio.
@main
struct VoiceChatBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voicechat-bench",
        abstract: "Benchmark VoiceChat perception or run complete speech-to-speech inference."
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

    @Option(name: .long, help: "Run complete speech-to-speech inference on this audio file.")
    var e2eAudio: String?

    @Option(name: .long, help: "Write the complete model response as a 22.05 kHz WAV.")
    var responseOutput: String?

    @Option(name: .long, help: "Silent tail after E2E input, in seconds.")
    var tailSeconds: Double = 6

    @Option(name: .long, help: "Input frames per E2E streaming push (one frame is 80 ms).")
    var chunkFrames: Int = 1

    @Flag(name: .long, help: "Force BOS at the end of E2E input for controlled tests.")
    var forceTurnAtEnd = false

    private struct ProcessMemory {
        let residentBytes: UInt64
        let peakResidentBytes: UInt64
        let physicalFootprintBytes: UInt64
        let peakPhysicalFootprintBytes: UInt64
    }

    /// Current and kernel-recorded peak process memory, in bytes.
    ///
    /// RSS is retained because it is widely reported. Physical footprint is
    /// also shown because macOS RSS can omit file-backed MLX mappings and thus
    /// understate the unified-memory pressure that determines whether the
    /// complete bundle fits on a machine.
    private func processMemory() -> ProcessMemory {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size
                / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(
                    mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return ProcessMemory(
                residentBytes: 0, peakResidentBytes: 0,
                physicalFootprintBytes: 0, peakPhysicalFootprintBytes: 0)
        }
        let resident = UInt64(info.resident_size)
        let footprint = UInt64(info.phys_footprint)
        return ProcessMemory(
            residentBytes: resident,
            peakResidentBytes: max(resident, UInt64(info.resident_size_peak)),
            physicalFootprintBytes: footprint,
            peakPhysicalFootprintBytes: max(
                footprint, UInt64(max(0, info.ledger_phys_footprint_peak))))
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
        if let e2eAudio {
            try await runCompletePipeline(
                root: root, audioURL: URL(fileURLWithPath: e2eAudio))
            return
        }
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

        let peakGB = Double(Memory.peakMemory) / 1e9
        let rssGB = Double(processMemory().residentBytes) / 1e9
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

    private func runCompletePipeline(root: URL, audioURL: URL) async throws {
        guard tailSeconds.isFinite,
              tailSeconds >= 0,
              tailSeconds <= VoiceChatSession.maximumSilenceSeconds else {
            throw ValidationError(
                "--tail-seconds must be between 0 and "
                    + "\(Int(VoiceChatSession.maximumSilenceSeconds))")
        }
        guard chunkFrames > 0, chunkFrames <= 128 else {
            throw ValidationError("--chunk-frames must be between 1 and 128")
        }
        let samples = try AudioFileLoader.load(
            url: audioURL, targetSampleRate: VoiceChatSession.inputSampleRate)
        let baselineMemory = processMemory()
        print("loading complete VoiceChat bundle...")
        let loadStarted = DispatchTime.now().uptimeNanoseconds
        let fullModel = try await VoiceChatModel.load(from: root)
        let session = try await fullModel.startSession()
        let loadElapsed = Double(
            DispatchTime.now().uptimeNanoseconds - loadStarted) / 1_000_000
        let loadedMemory = processMemory()
        if forceTurnAtEnd {
            await session.forceTurn(
                atFrame: samples.count / VoiceChatSession.inputSamplesPerFrame)
        }

        let chunk = VoiceChatSession.inputSamplesPerFrame * chunkFrames
        var firstTextMilliseconds: Double?
        var firstAudioMilliseconds: Double?

        func recordFirstSpeaking(events: [VoiceChatFrameEvent]) {
            guard firstAudioMilliseconds == nil,
                  let first = events.first(where: \.speaking) else { return }
            // Input preparation + encoder + language decision is the point at
            // which the first spoken text token exists. Adding the EAR-TTS and
            // codec stage gives the first playable output frame.
            let textMilliseconds = first.perceptionLatencyMilliseconds
                + first.decisionLatencyMilliseconds
            firstTextMilliseconds = textMilliseconds
            firstAudioMilliseconds = textMilliseconds
                + first.synthesisLatencyMilliseconds
        }

        let started = DispatchTime.now().uptimeNanoseconds
        for start in stride(from: 0, to: samples.count, by: chunk) {
            let events = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + chunk)]))
            recordFirstSpeaking(events: events)
        }
        let silenceSamples = Int(
            (tailSeconds * Double(VoiceChatSession.inputSampleRate)).rounded())
        let silence = [Float](repeating: 0, count: silenceSamples)
        for start in stride(from: 0, to: silence.count, by: chunk) {
            let events = try await session.pushAudio(
                Array(silence[start ..< min(silence.count, start + chunk)]))
            recordFirstSpeaking(events: events)
        }
        let elapsed = Double(
            DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        let reply = await session.reply()
        let userTranscript = await session.userTranscript()
        let summary = await session.summary()
        let streamingMemory = processMemory()
        let timelineMilliseconds = Double(summary.frames * VoiceChatSession.frameMilliseconds)

        print(String(format: "load + warmup   %.0f ms", loadElapsed))
        print("user transcript \(String(reflecting: userTranscript))")
        print("model response  \(String(reflecting: reply))")
        print("frames          \(summary.frames) (\(summary.speakingFrames) speaking)")
        if let first = summary.firstSpeechMilliseconds {
            print(String(format: "opened turn     %.0f ms", first))
        }
        if let firstTextMilliseconds, let firstAudioMilliseconds {
            print(String(
                format: "first response  text %.1f ms  audio %.1f ms",
                firstTextMilliseconds, firstAudioMilliseconds))
        }
        print(String(
            format: "perception      p50 %.1f ms  p95 %.1f ms",
            summary.perceptionP50Milliseconds, summary.perceptionP95Milliseconds))
        print(String(
            format: "decision        p50 %.1f ms  p95 %.1f ms",
            summary.decisionP50Milliseconds, summary.decisionP95Milliseconds))
        print(String(
            format: "synthesis       p50 %.1f ms  p95 %.1f ms",
            summary.synthesisP50Milliseconds, summary.synthesisP95Milliseconds))
        print(String(
            format: "total/frame     p50 %.1f ms  p95 %.1f ms",
            summary.totalP50Milliseconds, summary.totalP95Milliseconds))
        print(String(
            format: "wall / timeline %.0f / %.0f ms  RTF %.2f",
            elapsed, timelineMilliseconds,
            elapsed / max(1, timelineMilliseconds)))
        print("real time       \(summary.realTime ? "yes" : "NO")")
        print(String(
            format: "streaming RSS   %.2f GB current  %.2f GB peak  (+%.2f GB)",
            Double(streamingMemory.residentBytes) / 1e9,
            Double(streamingMemory.peakResidentBytes) / 1e9,
            Double(streamingMemory.peakResidentBytes - baselineMemory.residentBytes) / 1e9))
        print(String(
            format: "memory pressure %.2f GB current  %.2f GB peak  (+%.2f GB)",
            Double(streamingMemory.physicalFootprintBytes) / 1e9,
            Double(streamingMemory.peakPhysicalFootprintBytes) / 1e9,
            Double(streamingMemory.peakPhysicalFootprintBytes
                - baselineMemory.physicalFootprintBytes) / 1e9))
        print(String(
            format: "MLX GPU peak    %.2f GB  loaded RSS %.2f GB",
            Double(Memory.peakMemory) / 1e9,
            Double(loadedMemory.residentBytes) / 1e9))

        if let responseOutput {
            let waveform = await session.renderedAudio()
            let url = URL(fileURLWithPath: responseOutput)
            try WAVWriter.write(
                samples: waveform,
                sampleRate: VoiceChatSession.outputSampleRate,
                to: url)
            print("wrote \(url.path)")
        }
    }
}
