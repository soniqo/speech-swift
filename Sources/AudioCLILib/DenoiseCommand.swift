import Foundation
import ArgumentParser
import SpeechEnhancement
import AudioCommon

public struct DenoiseCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "denoise",
        abstract: "Remove background noise from speech audio using DeepFilterNet3"
    )

    @Argument(help: "Input audio file (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Output file path (default: input_clean.wav)")
    public var output: String?

    @Option(name: .shortAndLong, help: "Model ID on HuggingFace")
    public var model: String = SpeechEnhancer.defaultModelId

    @Option(name: .long, help: "Window size in seconds when auto-chunking long inputs (default: 45.0)")
    public var chunkSeconds: Double = 45.0

    @Option(name: .long, help: "Crossfade overlap between chunks in milliseconds (default: 500)")
    public var overlapMs: Int = 500

    @Flag(name: .long, help: "Disable auto-chunking; error if input exceeds the model's 60s cap")
    public var noChunk: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading audio: \(audioFile)")
            let inputURL = URL(fileURLWithPath: audioFile)
            let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: 48000)
            let duration = Double(audio.count) / 48000.0
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s @ 48kHz)")

            print("Loading DeepFilterNet3 model: \(model)")
            let enhancer = try await SpeechEnhancer.fromPretrained(
                modelId: model,
                progressHandler: reportProgress
            )

            // Auto-chunk inputs above the single-shot 60s cap unless the
            // user explicitly opted out with --no-chunk. Inputs at or under
            // chunkSeconds flow through the fast single-shot path inside
            // enhanceChunked() and are bit-identical to plain enhance().
            let willChunk = !noChunk && duration > chunkSeconds
            if willChunk {
                let chunkCount = Int(ceil((duration - Double(overlapMs) / 1000) / (chunkSeconds - Double(overlapMs) / 1000)))
                print("Auto-chunking: \(String(format: "%.1f", duration))s → \(chunkCount) chunks of \(String(format: "%.1f", chunkSeconds))s with \(overlapMs)ms crossfade")
            } else {
                print("Enhancing audio...")
            }

            let start = Date()
            let enhanced: [Float]
            if noChunk {
                enhanced = try enhancer.enhance(audio: audio, sampleRate: 48000)
            } else {
                enhanced = try enhancer.enhanceChunked(
                    audio: audio, sampleRate: 48000,
                    chunkSeconds: chunkSeconds, overlapMs: overlapMs)
            }
            let elapsed = Date().timeIntervalSince(start)

            let enhancedDuration = Double(enhanced.count) / 48000.0
            let rtf = elapsed / enhancedDuration

            print("  Enhanced \(enhanced.count) samples (\(String(format: "%.2f", enhancedDuration))s)")
            print("  Processing: \(String(format: "%.3f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")

            // Determine output path
            let outputPath: String
            if let output {
                outputPath = output
            } else {
                let base = inputURL.deletingPathExtension().lastPathComponent
                let dir = inputURL.deletingLastPathComponent()
                outputPath = dir.appendingPathComponent("\(base)_clean.wav").path
            }

            let outputURL = URL(fileURLWithPath: outputPath)
            try WAVWriter.write(samples: enhanced, sampleRate: 48000, to: outputURL)
            print("  Saved: \(outputPath)")
        }
    }
}
