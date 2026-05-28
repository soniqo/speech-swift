import Foundation
import ArgumentParser
import FlashSR
import AudioCommon

public struct UpsampleCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "upsample",
        abstract: "Audio super-resolution with FlashSR (one-step distilled AudioSR, MLX, 48 kHz)"
    )

    @Argument(help: "Input audio file (WAV, any sample rate — resampled internally to 48 kHz)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Output WAV path (48 kHz mono)")
    public var output: String = "hr.wav"

    @Option(name: .long, help: "Bundle variant: int4 or int8")
    public var variant: String = "int4"

    @Option(name: .long, help: "Diffusion timestep (default 999 for the 1-step distilled student)")
    public var timestep: Int = 999

    @Option(name: .long, help: "Random seed for reproducible noise initialisation")
    public var seed: UInt64?

    public init() {}

    public func run() throws {
        try runAsync {
            guard let variantEnum = FlashSRVariant(rawValue: variant) else {
                throw ValidationError("Unknown variant '\(variant)'. Use one of: \(FlashSRVariant.allCases.map { $0.rawValue }.joined(separator: ", "))")
            }

            let inputURL = URL(fileURLWithPath: audioFile)
            print("Loading audio: \(audioFile)")
            // Super-resolution: resample the input up to 48 kHz at mastering
            // quality before the model adds high-frequency detail.
            let audio = try AudioFileLoader.load(url: inputURL, targetSampleRate: FlashSR.sampleRate, quality: .mastering)
            let durIn = Double(audio.count) / Double(FlashSR.sampleRate)
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", durIn))s @ 48 kHz)")

            print("Loading FlashSR \(variant)…")
            let model = try await FlashSR.fromPretrained(
                variant: variantEnum,
                progressHandler: { reportProgress($0, "downloading") })

            print("Up-sampling \(String(format: "%.2f", durIn))s of audio in \(FlashSR.frameSamples) sample (5.12s) windows…")
            let start = Date()
            let params = FlashSRParams(timestep: timestep, seed: seed)
            // Use the lower-level windowed API when there's just one window; for
            // multi-window inputs we still use the convenience `enhance` path.
            let hr: [Float]
            if audio.count <= FlashSR.frameSamples {
                hr = model.upsampleWindow(samples: audio, params: params)
            } else {
                hr = try model.upsample(audio: audio, sampleRate: FlashSR.sampleRate)
            }
            let elapsed = Date().timeIntervalSince(start)
            let durOut = Double(hr.count) / Double(FlashSR.sampleRate)
            print("  Generated \(hr.count) samples (\(String(format: "%.2f", durOut))s @ \(FlashSR.sampleRate) Hz)")
            print("  Wall: \(String(format: "%.2f", elapsed))s  RTF: \(String(format: "%.2f", elapsed / durOut))")

            try WAVWriter.write(samples: hr, sampleRate: FlashSR.sampleRate,
                                to: URL(fileURLWithPath: output))
            print("  Saved: \(output)")
        }
    }
}
