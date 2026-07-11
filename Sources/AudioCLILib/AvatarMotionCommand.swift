import Foundation
import ArgumentParser
import AudioCommon
import Audio2Face3D

public struct AvatarMotionCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "avatar-motion",
        abstract: "Generate NVIDIA Audio2Face-3D coefficient frames from speech audio"
    )

    @Argument(help: "Input audio file (WAV, m4a, etc.; resampled to 16 kHz)")
    public var input: String

    @Option(name: .shortAndLong, help: "Output JSONL file path")
    public var output: String = "avatar-motion.jsonl"

    @Option(name: .long, help: "Backend: mlx")
    public var backend: Audio2Face3DBackend = .mlx

    @Option(name: .long, help: "Hugging Face model ID for the exported MLX bundle")
    public var model: String = Audio2Face3DConfiguration.defaultModelId

    @Option(name: .long, help: "Local exported model directory for --backend mlx")
    public var modelDir: String?

    @Flag(name: .long, help: "Print timing and frame count")
    public var verbose: Bool = false

    public init() {}

    public func run() throws {
        try runAsync {
            let inputURL = URL(fileURLWithPath: input)
            let audio = try AudioFileLoader.load(
                url: inputURL,
                targetSampleRate: 16_000,
                quality: .standard)

            let motionModel: Audio2Face3DModel
            if let modelDir {
                motionModel = try Audio2Face3DModel.fromLocal(
                    directory: URL(fileURLWithPath: modelDir, isDirectory: true),
                    backend: backend)
            } else {
                motionModel = try await Audio2Face3DModel.fromPretrained(
                    modelId: model,
                    backend: backend,
                    progressHandler: reportProgress)
            }

            let start = CFAbsoluteTimeGetCurrent()
            let frames = try motionModel.frames(for: audio, sampleRate: 16_000)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let lines = try frames.map { frame -> String in
                let data = try encoder.encode(frame)
                return String(decoding: data, as: UTF8.self)
            }.joined(separator: "\n")

            let outputURL = URL(fileURLWithPath: output)
            try (lines + "\n").write(to: outputURL, atomically: true, encoding: .utf8)

            if verbose {
                let duration = Double(audio.count) / 16_000.0
                print(String(format: "Generated %d coefficient frames for %.2fs audio in %.3fs (%.1f fps)",
                             frames.count, duration, elapsed, Double(frames.count) / max(elapsed, 0.001)))
            }
            print("Saved avatar motion to \(output)")
        }
    }
}

extension Audio2Face3DBackend: ExpressibleByArgument {}
