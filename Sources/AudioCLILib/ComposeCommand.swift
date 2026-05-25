import Foundation
import ArgumentParser
import MAGNeTMusicGen
import StableAudio3MusicGen
import AudioCommon

public struct ComposeCommand: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "compose",
        abstract: "Generate music from a text prompt (Stable Audio 3 by default; MAGNeT also available, MLX, on-device)"
    )

    @Argument(help: "Text prompt describing the music to generate (e.g. \"lofi house loop\")")
    public var prompt: String

    @Option(name: .shortAndLong, help: "Output WAV path (44.1 kHz stereo for sa3, 32 kHz mono for magnet)")
    public var output: String = "music.wav"

    @Option(
        name: .long,
        help: "Music engine: sa3 (default — Stable Audio 3 Medium INT8, 44.1 kHz stereo, variable length) | magnet (MAGNeT, 32 kHz mono, fixed 30 s)"
    )
    public var engine: String = "sa3"

    // MARK: SA3 options
    @Option(name: .long, help: "SA3 variant: medium-int8 (default) | medium-int4 (Stable Audio 3 Small variants land in a follow-up)")
    public var sa3Variant: String = "medium-int8"

    @Option(name: .long, help: "SA3 output length in seconds (1.0–384.0)")
    public var seconds: Float = 30.0

    @Option(name: .long, help: "SA3 pingpong sampler steps (default 8 — distilled sweet spot)")
    public var sa3Steps: Int = 8

    @Option(name: .long, help: "SA3 classifier-free guidance (1.0 = off, >1 toward prompt)")
    public var sa3Cfg: Float = 1.0

    @Option(name: .long, help: "SA3 σmax — initial noise level (default 1.0)")
    public var sigmaMax: Float = 1.0

    @Option(name: .long, help: "SA3 local bundle override path — skip HF download and load from this directory (smoke-test / offline)")
    public var sa3BundleDir: String?

    // MARK: MAGNeT options
    @Option(name: .long, help: "MAGNeT variant: small-int4 | small-int8 | medium-int4 | medium-int8")
    public var magnetVariant: String = "small-int4"

    @Option(name: .long, help: "MAGNeT sampling temperature (annealed linearly per stage)")
    public var temperature: Float = 3.0

    @Option(name: .long, help: "MAGNeT top-p (nucleus) sampling threshold")
    public var topP: Float = 0.9

    @Option(name: .long, help: "MAGNeT max classifier-free guidance coefficient")
    public var cfgMax: Float = 10.0

    @Option(name: .long, help: "MAGNeT min classifier-free guidance coefficient")
    public var cfgMin: Float = 1.0

    @Option(
        name: .long,
        help: "MAGNeT comma-separated decoding steps per codebook (default: 20,10,10,10)"
    )
    public var magnetSteps: String = "20,10,10,10"

    @Option(name: .long, help: "Random seed for reproducibility")
    public var seed: UInt64?

    public init() {}

    public func run() throws {
        switch engine.lowercased() {
        case "sa3", "stableaudio3", "stable-audio-3":
            try runSA3()
        case "magnet":
            try runMAGNeT()
        default:
            throw ValidationError("Unknown --engine '\(engine)'. Use one of: sa3, magnet.")
        }
    }

    private func runSA3() throws {
        try runAsync {
            guard let variantEnum = StableAudio3Variant(rawValue: sa3Variant) else {
                throw ValidationError(
                    "Unknown --sa3-variant '\(sa3Variant)'. Use one of: "
                  + StableAudio3Variant.allCases.map(\.rawValue).joined(separator: ", "))
            }
            print("Loading Stable Audio 3 \(sa3Variant)…")
            let overrideURL = sa3BundleDir.map { URL(fileURLWithPath: $0) }
            let model = try await StableAudio3MusicGen.fromPretrained(
                variant: variantEnum,
                tLatHint: StableAudio3MusicGen.computeTLat(seconds: seconds),
                localBundleOverride: overrideURL,
                progressHandler: { reportProgress($0, "downloading") })

            print("Prompt: \"\(prompt)\"")
            let params = StableAudio3GenerationParams(
                seconds: seconds, steps: sa3Steps,
                cfgScale: sa3Cfg, apg: 1.0,
                sigmaMax: sigmaMax, seed: seed)

            print("Generating \(seconds)s of audio…")
            let start = Date()
            let (left, right) = model.generate(prompt: prompt, params: params)
            let elapsed = Date().timeIntervalSince(start)
            let audioSec = Double(left.count) / Double(StableAudio3MusicGen.sampleRate)
            print("  Generated \(left.count) frames per channel "
                + "(\(String(format: "%.2f", audioSec))s @ \(StableAudio3MusicGen.sampleRate) Hz stereo)")
            print("  Wall: \(String(format: "%.2f", elapsed))s  RTF: \(String(format: "%.2f", elapsed / audioSec))")

            let outURL = URL(fileURLWithPath: output)
            try WAVWriter.writeStereo(left: left, right: right,
                                       sampleRate: StableAudio3MusicGen.sampleRate, to: outURL)
            print("  Saved: \(outURL.path)")
        }
    }

    private func runMAGNeT() throws {
        try runAsync {
            guard let variantEnum = MAGNeTVariant(rawValue: magnetVariant) else {
                throw ValidationError("Unknown --magnet-variant '\(magnetVariant)'. Use one of: \(MAGNeTVariant.allCases.map { $0.rawValue }.joined(separator: ", "))")
            }
            let decodingSteps = magnetSteps.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            guard decodingSteps.count == 4 else {
                throw ValidationError("--magnet-steps must be 4 comma-separated integers (one per codebook), got \(decodingSteps.count)")
            }

            print("Loading MAGNeT \(magnetVariant)…")
            let model = try await MAGNeTMusicGen.fromPretrained(
                variant: variantEnum,
                progressHandler: { reportProgress($0, "downloading") })

            print("Prompt: \"\(prompt)\"")
            let params = MAGNeTGenerationParams(
                decodingSteps: decodingSteps,
                maxCfgCoef: cfgMax, minCfgCoef: cfgMin,
                temperature: temperature, topP: topP,
                annealTemp: true, seed: seed)

            print("Generating \(model.config.segmentDuration)s of audio…")
            let start = Date()
            let pcm = model.generate(text: prompt, params: params)
            let elapsed = Date().timeIntervalSince(start)
            let audioSec = Double(pcm.count) / Double(model.config.sampleRate)
            let rtf = elapsed / audioSec
            print("  Generated \(pcm.count) samples (\(String(format: "%.2f", audioSec))s @ \(model.config.sampleRate) Hz)")
            print("  Wall: \(String(format: "%.2f", elapsed))s  RTF: \(String(format: "%.2f", rtf))")

            let outURL = URL(fileURLWithPath: output)
            try WAVWriter.write(samples: pcm, sampleRate: model.config.sampleRate, to: outURL)
            print("  Saved: \(outURL.path)")
        }
    }
}

// Convenience accessor since StableAudio3MusicGen.sampleRate is the SA3 constant.
extension StableAudio3MusicGen {
    public static var sampleRate: Int { SA3Audio.sampleRate }
}
