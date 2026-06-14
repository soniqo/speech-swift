import Foundation
import ArgumentParser
import SpeechRestoration
import AudioCommon

/// `speech restore` — joint denoise + dereverb with Sidon (CoreML, 48 kHz out).
///
/// Opt-in speech restoration intended for cleaning a noisy/reverberant
/// voice-cloning reference before TTS. Preserves speaker identity (the design
/// goal of the upstream model), unlike generic enhancement.
public struct RestoreCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "restore",
        abstract: "Restore (denoise + dereverb) speech with Sidon — CoreML, 48 kHz output"
    )

    @Argument(help: "Input audio file (WAV, any sample rate — resampled to 16 kHz internally)")
    public var audioFile: String

    @Option(name: .shortAndLong, help: "Output file path (default: input_restored.wav, 48 kHz)")
    public var output: String?

    @Option(name: .long, help: "Precision variant: fp16 (default) or int8")
    public var variant: String = "fp16"

    @Option(name: .shortAndLong, help: "Model repo id on HuggingFace")
    public var model: String = SpeechRestorer.defaultModelId

    public init() {}

    public func run() throws {
        try runAsync {
            guard let variantEnum = SidonVariant(rawValue: variant) else {
                throw ValidationError(
                    "Unknown variant '\(variant)'. Use one of: "
                    + SidonVariant.allCases.map { $0.rawValue }.joined(separator: ", "))
            }

            let inputURL = URL(fileURLWithPath: audioFile)
            print("Loading audio: \(audioFile)")
            // Sidon's front-end is 16 kHz; load+resample there.
            let audio = try AudioFileLoader.load(
                url: inputURL, targetSampleRate: SpeechRestorer.inputSampleRate)
            let durIn = Double(audio.count) / Double(SpeechRestorer.inputSampleRate)
            print("  Loaded \(audio.count) samples (\(String(format: "%.2f", durIn))s @ 16 kHz)")

            print("Loading Sidon (\(variant))…")
            let restorer = try await SpeechRestorer.fromPretrained(
                variant: variantEnum,
                modelId: model,
                progressHandler: { reportProgress($0, $1) })

            print("Restoring (10 s windows)…")
            let start = Date()
            let restored = try restorer.restore(
                audio: audio, sampleRate: SpeechRestorer.inputSampleRate)
            let elapsed = Date().timeIntervalSince(start)
            let durOut = Double(restored.count) / Double(SpeechRestorer.outputSampleRate)
            print("  Restored \(restored.count) samples (\(String(format: "%.2f", durOut))s @ 48 kHz)")
            print("  Wall: \(String(format: "%.2f", elapsed))s  RTF: \(String(format: "%.2f", elapsed / Swift.max(durOut, 1e-6)))")

            let outputPath: String
            if let output {
                outputPath = output
            } else {
                let base = inputURL.deletingPathExtension().lastPathComponent
                let dir = inputURL.deletingLastPathComponent()
                outputPath = dir.appendingPathComponent("\(base)_restored.wav").path
            }
            try WAVWriter.write(
                samples: restored,
                sampleRate: SpeechRestorer.outputSampleRate,
                to: URL(fileURLWithPath: outputPath))
            print("  Saved: \(outputPath)")
        }
    }
}
