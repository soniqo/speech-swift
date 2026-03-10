import Foundation
import ArgumentParser
import KokoroTTS
import AudioCommon

public struct KokoroCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "kokoro",
        abstract: "Text-to-speech synthesis using Kokoro-82M (CoreML, Neural Engine)"
    )

    @Argument(help: "Text to synthesize (omit when using --list-voices)")
    public var text: String?

    @Option(name: .shortAndLong, help: "Output WAV file path")
    public var output: String = "output.wav"

    @Option(name: .long, help: "Voice preset (default: af_heart)")
    public var voice: String = "af_heart"

    @Option(name: .long, help: "Language code: en, fr, es, ja, zh, hi, pt, ko")
    public var language: String = "en"

    @Option(name: .long, help: "HuggingFace model ID")
    public var model: String = KokoroTTSModel.defaultModelId

    @Flag(name: .long, help: "List available voice presets and exit")
    public var listVoices: Bool = false

    @Flag(name: .long, help: "Show detailed timing info")
    public var verbose: Bool = false

    public init() {}

    public func validate() throws {
        if text == nil && !listVoices {
            throw ValidationError("Either a text argument or --list-voices must be provided")
        }
    }

    public func run() throws {
        try runAsync {
            print("Loading Kokoro-82M model...")
            let kokoroModel = try await KokoroTTSModel.fromPretrained(
                modelId: model, progressHandler: reportProgress)

            if listVoices {
                let voices = kokoroModel.availableVoices
                if voices.isEmpty {
                    print("No voice presets available.")
                } else {
                    print("Available voices (\(voices.count)):")
                    for name in voices {
                        print("  - \(name)")
                    }
                }
                return
            }

            guard let inputText = text else {
                print("Error: text argument required")
                throw ExitCode(1)
            }

            print("Synthesizing: \"\(inputText)\"")
            print("  Voice: \(voice), Language: \(language)")

            let startTime = CFAbsoluteTimeGetCurrent()
            let audio = try kokoroModel.synthesize(
                text: inputText, voice: voice, language: language)
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            guard !audio.isEmpty else {
                print("Error: No audio generated")
                throw ExitCode(1)
            }

            let duration = Double(audio.count) / Double(KokoroTTSModel.outputSampleRate)
            print(String(format: "  Duration: %.2fs, Time: %.3fs, RTFx: %.1f",
                         duration, elapsed, duration / elapsed))

            let outputURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: audio, sampleRate: KokoroTTSModel.outputSampleRate, to: outputURL)
            print("Saved to \(output)")
        }
    }
}
