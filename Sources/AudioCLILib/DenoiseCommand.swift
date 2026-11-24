import Foundation
import ArgumentParser
import SpeechEnhancement
import AudioCommon

public struct DenoiseCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "denoise",
        abstract: "Remove background noise from speech with DeepFilterNet3 or Inworld Voice Isolator"
    )

    @Argument(help: "Input audio file (WAV, any sample rate)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Output file path (default: input_clean.wav)")
    public var output: String?

    @Option(name: .shortAndLong, help: "Hugging Face model ID for DeepFilterNet3, or local Core ML export directory for inworld")
    public var model: String?

    @Option(name: .long, help: "Engine: coreml or mlx (DeepFilterNet3), inworld (Voice Isolator v1, Core ML)")
    public var engine: String = "coreml"

    @Option(name: .long, help: "Chunk size in seconds (default: 45 for DeepFilterNet3, 8 for inworld)")
    public var chunkSeconds: Double?

    @Option(name: .long, help: "Crossfade in milliseconds (default: 500 for DeepFilterNet3, 2000 for inworld)")
    public var overlapMs: Int?

    @Flag(name: .long, help: "Disable auto-chunking (Core ML limits: 60 s for DeepFilterNet3, 8 s for inworld)")
    public var noChunk: Bool = false

    public init() {}

    public func validate() throws {
        let name = engine.lowercased()
        if name == "inworld" {
            guard let model, !model.isEmpty else {
                throw ValidationError("--engine inworld requires --model with a local Core ML export directory")
            }
        } else if SpeechEnhancerEngine(rawValue: name) == nil {
            throw ValidationError("Unknown engine '\(engine)' (expected coreml, mlx, or inworld)")
        }
    }

    public func run() throws {
        try runAsync {
            let useInworld = engine.lowercased() == "inworld"
            let sampleRate = 48_000
            print("Loading audio: \(audioFile)")
            let inputURL = URL(fileURLWithPath: audioFile)
            let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: sampleRate,
                                                 quality: useInworld ? .mastering : .standard)
            let duration = Double(audio.count) / Double(sampleRate)
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", duration))s @ 48kHz)")

            let enhance: () throws -> [Float]
            if useInworld, let model {
                if noChunk && duration > InworldVoiceIsolator.maxChunkSeconds {
                    throw ValidationError("Inworld Voice Isolator supports up to 8 seconds per chunk. Enable chunking for longer audio.")
                }
                let modelDirectory = URL(fileURLWithPath: model, isDirectory: true)
                print("Loading Inworld Voice Isolator v1: \(modelDirectory.path)")
                let isolator = try InworldVoiceIsolator(modelDirectory: modelDirectory)
                enhance = {
                    try isolator.enhanceChunked(
                        audio: audio, sampleRate: sampleRate,
                        chunkSeconds: noChunk ? InworldVoiceIsolator.maxChunkSeconds : chunkSeconds ?? InworldVoiceIsolator.maxChunkSeconds,
                        overlapMs: noChunk ? 0 : overlapMs ?? InworldVoiceIsolator.defaultOverlapMs)
                }
            } else {
                guard let resolvedEngine = SpeechEnhancerEngine(rawValue: engine.lowercased()) else {
                    throw ValidationError("Invalid denoise engine or model configuration")
                }
                let resolvedModel = model ?? (resolvedEngine == .mlx
                    ? SpeechEnhancer.defaultMLXModelId : SpeechEnhancer.defaultModelId)
                print("Loading DeepFilterNet3 model: \(resolvedModel) (\(resolvedEngine.rawValue))")
                let enhancer = try await SpeechEnhancer.fromPretrained(
                    modelId: resolvedModel, engine: resolvedEngine, progressHandler: reportProgress)
                enhance = {
                    if noChunk {
                        return try enhancer.enhance(audio: audio, sampleRate: sampleRate)
                    }
                    return try enhancer.enhanceChunked(
                        audio: audio, sampleRate: sampleRate,
                        chunkSeconds: chunkSeconds ?? 45, overlapMs: overlapMs ?? 500)
                }
            }

            print("Enhancing audio...")
            let start = Date()
            let enhanced = try enhance()
            let elapsed = Date().timeIntervalSince(start)

            let enhancedDuration = Double(enhanced.count) / Double(sampleRate)
            let rtf = enhancedDuration > 0 ? elapsed / enhancedDuration : 0

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
            if useInworld {
                try WAVWriter.writeFloat32(samples: enhanced, sampleRate: sampleRate, to: outputURL)
            } else {
                try WAVWriter.write(samples: enhanced, sampleRate: sampleRate, to: outputURL)
            }
            print("  Saved: \(outputPath)")
        }
    }
}
