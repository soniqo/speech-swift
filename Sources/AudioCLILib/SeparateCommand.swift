import Foundation
import ArgumentParser
import SourceSeparation
import AudioCommon
import MLX

public struct SeparateCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "separate",
        abstract: "Separate a music track into stems (vocals, drums, bass, other)"
    )

    @Argument(help: "Input audio file (WAV, stereo 44.1kHz)")
    public var input: String

    @Option(name: .long, help: "Output directory (default: <input>_stems/)")
    public var outputDir: String?

    @Option(name: .long, help: "Stems to extract: vocals,drums,bass,other (default: all)")
    public var stems: String?

    @Option(name: .long, help: "Engine: umx (default) or htdemucs (Demucs v4, higher quality)")
    public var engine: String = "umx"

    @Option(name: .long, help: "Model variant: hq (default, 8.9M/stem) or l (28.3M/stem)")
    public var model: String = "hq"

    @Option(name: .long, help: "Local htdemucs weights dir (htdemucs_ft.safetensors + _config.json)")
    public var htdemucsDir: String?

    @Option(name: .long, help: "HTDemucs precision: fp16 (default) or int8 (smaller download)")
    public var htdemucsPrecision: String = "fp16"

    @Flag(name: .long, help: "Show timing info")
    public var verbose: Bool = false

    public init() {}

    public func run() throws {
        let inputURL = URL(fileURLWithPath: input)
        let baseName = inputURL.deletingPathExtension().lastPathComponent
        let outDir = URL(fileURLWithPath: outputDir ?? "\(baseName)_stems")

        // Parse target stems
        let targets: [SeparationTarget]
        if let stemsStr = stems {
            targets = stemsStr.split(separator: ",").compactMap {
                SeparationTarget(rawValue: String($0).trimmingCharacters(in: .whitespaces))
            }
        } else {
            targets = SeparationTarget.allCases
        }

        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        if engine.lowercased() == "htdemucs" {
            try runHTDemucs(inputURL: inputURL, outDir: outDir, targets: targets)
            return
        }

        try runAsync {
            let modelId = model.lowercased() == "l" ? SourceSeparator.largeModelId : SourceSeparator.defaultModelId
            print("Loading model (\(model.lowercased() == "l" ? "UMX-L" : "UMX-HQ"))...")
            let separator = try await SourceSeparator.fromPretrained(
                modelId: modelId, progressHandler: reportProgress)

            print("Loading audio: \(input)")
            let audio = try AudioFileLoader.loadStereo(url: inputURL, targetSampleRate: 44100, quality: .mastering)
            let duration = Double(audio[0].count) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")

            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let startTime = CFAbsoluteTimeGetCurrent()

            let results = separator.separate(audio: audio, targets: targets)

            let elapsed = CFAbsoluteTimeGetCurrent() - startTime
            let rtf = elapsed / duration

            for (target, stemAudio) in results {
                let outputURL = outDir.appendingPathComponent("\(target.rawValue).wav")
                try WAVWriter.writeStereo(left: stemAudio[0], right: stemAudio[1], sampleRate: 44100, to: outputURL)
                print("  Saved: \(outputURL.lastPathComponent)")
            }

            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }

    private func runHTDemucs(inputURL: URL, outDir: URL, targets: [SeparationTarget]) throws {
        try runAsync {
            let precision = HTDemucsSeparator.Precision(rawValue: htdemucsPrecision.lowercased()) ?? .fp16
            let separator: HTDemucsSeparator
            if let dir = htdemucsDir {
                print("Loading HTDemucs (Demucs v4, \(precision.rawValue)) from \(dir)...")
                separator = try HTDemucsSeparator.fromLocal(
                    directory: URL(fileURLWithPath: dir), modelName: precision.modelName)
            } else {
                print("Loading HTDemucs (Demucs v4, \(precision.rawValue))...")
                separator = try await HTDemucsSeparator.fromPretrained(
                    precision: precision, progressHandler: reportProgress)
            }

            print("Loading audio: \(input)")
            let audio = try AudioFileLoader.loadStereo(url: inputURL, targetSampleRate: 44100)
            let L = audio[0].count
            let duration = Double(L) / 44100.0
            print("  Duration: \(String(format: "%.1f", duration))s")
            let mix = MLXArray(audio[0] + audio[1], [1, 2, L])

            let wanted = Set(targets.map(\.rawValue))
            print("Separating into \(targets.map(\.rawValue).joined(separator: ", "))...")
            let start = CFAbsoluteTimeGetCurrent()
            let stems = separator.separate(mix)
            let elapsed = CFAbsoluteTimeGetCurrent() - start

            for (name, arr) in stems where wanted.contains(name) {
                let flat = arr.asArray(Float.self)             // [1, 2, L] → 2L
                let left = Array(flat[0..<L])
                let right = Array(flat[L..<(2 * L)])
                let outputURL = outDir.appendingPathComponent("\(name).wav")
                try WAVWriter.writeStereo(left: left, right: right, sampleRate: 44100, to: outputURL)
                print("  Saved: \(outputURL.lastPathComponent)")
            }
            let rtf = elapsed / duration
            print("Done in \(String(format: "%.1f", elapsed))s (RTF: \(String(format: "%.2f", rtf)))")
        }
    }
}
