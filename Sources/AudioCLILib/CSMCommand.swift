import Foundation
import ArgumentParser
import MLX
import CSM
import AudioCommon

public struct CSMCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "csm",
        abstract: "Conversational Speech Model (CSM-1B) — text→audio with voice cloning"
    )

    @Argument(help: "Text to synthesize")
    public var text: String

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Reference audio to clone the voice from (any sample rate)")
    public var refAudio: String

    @Option(name: .long, help: "Transcript of the reference audio")
    public var refText: String

    @Option(name: .long, help: "HuggingFace model ID")
    public var model: String = "aufklarer/CSM-1B-MLX-8bit"

    @Option(name: .long, help: "Sampling temperature (0 = greedy)")
    public var temperature: Float = 0.9

    @Option(name: .long, help: "Top-k sampling")
    public var topK: Int = 50

    public init() {}

    public func run() throws {
        try runAsync {
            print("Loading \(model)...")
            let pipeline = try await CSMPipeline.fromPretrained(model)

            let ref = try AudioFileLoader.load(
                url: URL(fileURLWithPath: refAudio), targetSampleRate: 24000)
            print("Synthesizing (\(ref.count) reference samples)...")

            let audio = pipeline.synthesize(
                text: text, refAudio: MLXArray(ref), refText: refText,
                temperature: temperature, topK: topK)
            let samples = audio.asArray(Float.self)

            try WAVWriter.write(samples: samples, sampleRate: 24000,
                               to: URL(fileURLWithPath: output))
            print("Wrote \(output) (\(String(format: "%.2f", Double(samples.count) / 24000))s)")
        }
    }
}
