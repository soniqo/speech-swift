import ArgumentParser
import AudioCommon
import Foundation
import SpeechLanguageID

public struct LanguageIDCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "language-id",
        abstract: "Identify the spoken language in an audio file"
    )

    @Argument(help: "Audio file to classify (WAV, FLAC, or another supported format)")
    public var audioFile: String

    @Option(name: .long, help: "Inference engine: mlx or coreml")
    public var engine: String = LanguageIDEngine.mlx.rawValue

    @Option(name: .long, help: "Override the Hugging Face model ID")
    public var model: String?

    @Option(name: .long, help: "Number of ranked language candidates to show")
    public var top: Int = 5

    @Flag(name: .long, help: "Emit the result as JSON")
    public var json: Bool = false

    public init() {}

    public func validate() throws {
        guard LanguageIDEngine(rawValue: engine) != nil else {
            throw ValidationError("--engine must be 'mlx' or 'coreml'")
        }
        guard (1...107).contains(top) else {
            throw ValidationError("--top must be between 1 and 107")
        }
    }

    public func run() throws {
        try runAsync {
            guard let selectedEngine = LanguageIDEngine(rawValue: engine) else {
                throw ValidationError("--engine must be 'mlx' or 'coreml'")
            }
            let audio = try AudioFileLoader.load(
                url: URL(fileURLWithPath: audioFile),
                targetSampleRate: SpeechBrainFbank.sampleRate
            )
            let identifier = try await SpeechLanguageIdentifier.fromPretrained(
                modelID: model,
                engine: selectedEngine,
                progressHandler: json ? nil : reportProgress
            )

            let started = Date()
            let result = try identifier.identify(
                audio: audio,
                sampleRate: SpeechBrainFbank.sampleRate,
                topK: top
            )
            let elapsed = Date().timeIntervalSince(started)

            if json {
                struct Output: Encodable {
                    let predictions: [LanguageIDPrediction]
                    let analyzedDuration: TimeInterval
                    let windowCount: Int
                    let inferenceSeconds: TimeInterval
                }
                let output = Output(
                    predictions: result.predictions,
                    analyzedDuration: result.analyzedDuration,
                    windowCount: result.windowCount,
                    inferenceSeconds: elapsed
                )
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
                print(String(decoding: try encoder.encode(output), as: UTF8.self))
                return
            }

            for (rank, prediction) in result.predictions.enumerated() {
                let percent = prediction.probability * 100
                print(
                    "\(rank + 1). \(prediction.label.name) "
                        + "[\(prediction.label.code)] "
                        + "\(String(format: "%.2f", percent))%"
                )
            }
            print(
                "Analyzed \(String(format: "%.2f", result.analyzedDuration))s "
                    + "in \(result.windowCount) window(s); "
                    + "inference \(String(format: "%.3f", elapsed))s"
            )
        }
    }
}
