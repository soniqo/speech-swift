import AudioCommon
import CoreML
import Foundation

/// Inworld Voice Isolator v1 for recordings, loaded from a local Core ML bundle.
public final class InworldVoiceIsolator: SpeechEnhancementModel {
    public static let sampleRate = 48_000
    public static let maxChunkSeconds = 8.0
    public static let defaultOverlapMs = 2_000
    public var inputSampleRate: Int { Self.sampleRate }

    private static let format = "speech-swift-inworld-voice-isolator-v1"
    static let bands = 46
    static let hiddenWidth = 512
    static let fullFrames = Int(maxChunkSeconds * Double(sampleRate)) / InworldIsolationDSP.hop + 1

    private let stageFiles: InworldCoreMLStages
    private let computeUnits: MLComputeUnits
    private let fullWindowFrames: Int?
    private let tiledBackbone: InworldTiledBackbone?
    private let scan: InworldMambaScan
    private let dsp: InworldIsolationDSP

    /// Load the seven stages from a local bundle of `.mlpackage` or `.mlmodelc` directories.
    /// Compute units control Core ML. The Mamba scan uses Metal.
    public init(modelDirectory: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let manifestURL = modelDirectory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
              [Self.format, InworldTiledBackbone.format].contains(manifest.format) else {
            throw InworldIsolationError.invalidBundle("expected a v1 manifest in \(modelDirectory.path)")
        }
        let tiled = manifest.format == InworldTiledBackbone.format
        let timeRows = manifest.timeRows ?? 0
        let bandRows = manifest.bandRows ?? 0
        if tiled && (!(1...12).contains(timeRows) || !(1...128).contains(bandRows)) {
            throw InworldIsolationError.invalidBundle("incorrect tiled row limits")
        }
        let coefficientsURL = modelDirectory.appendingPathComponent("scan_coefficients.bin")
        let loadedScan = try InworldMambaScan(coefficients: Data(contentsOf: coefficientsURL))
        let loadedDSP = try InworldIsolationDSP()

        let stageCount = tiled ? InworldTiledBackbone.stageCount : InworldMambaScan.stageCount + 1
        let stageFiles = try InworldCoreMLStages(directory: modelDirectory, count: stageCount)
        self.stageFiles = stageFiles
        self.computeUnits = computeUnits
        fullWindowFrames = manifest.fullWindowFrames
        tiledBackbone = tiled ? InworldTiledBackbone(timeRows: timeRows, bandRows: bandRows,
                                                     supportsOutputBackings: manifest.tiledBackings ?? false,
                                                     supportsStaticTiles: manifest.staticTiles ?? false,
                                                     urls: stageFiles.urls,
                                                     computeUnits: computeUnits, scan: loadedScan) : nil
        scan = loadedScan
        dsp = loadedDSP
    }

    /// Isolate foreground speech with the checkpoint's 8 s window and 2 s crossfade.
    /// The result is 48 kHz mono PCM, aligned to the resampled input.
    public func enhance(audio: [Float], sampleRate: Int) throws -> [Float] {
        try enhanceChunked(audio: audio, sampleRate: sampleRate)
    }

    /// Use a shorter window or a different crossfade for recorded audio.
    public func enhanceChunked(
        audio: [Float], sampleRate: Int,
        chunkSeconds: Double = InworldVoiceIsolator.maxChunkSeconds,
        overlapMs: Int = InworldVoiceIsolator.defaultOverlapMs
    ) throws -> [Float] {
        let window = try InworldIsolationWindow(seconds: chunkSeconds, overlapMs: overlapMs)
        guard sampleRate > 0, audio.allSatisfy(\.isFinite) else {
            throw InworldIsolationError.invalidAudio("expected finite mono PCM and a positive sample rate")
        }
        guard !audio.isEmpty else { return [] }
        let samples = sampleRate == Self.sampleRate ? audio : AudioFileLoader.resample(
            audio, from: sampleRate, to: Self.sampleRate, quality: .mastering)
        if let tiledBackbone {
            return try window.render(samples, pairWindows: tiledBackbone.supportsStaticTiles) { windows in
                let spectra = try windows.map(dsp.spectrum)
                let frames = spectra[0].shape[1].intValue
                let masks = try tiledBackbone.predict(spectra: spectra, frames: frames)
                return windows.indices.map { index in
                    dsp.synthesize(windows[index], spectrum: spectra[index], maskResidual: masks[index])
                }
            }
        }
        return try window.render(samples, process: isolate)
    }

    private func isolate(_ audio: ArraySlice<Float>) throws -> [Float] {
        let spectrum = try dsp.spectrum(audio)
        let frames = audio.count / InworldIsolationDSP.hop + 1
        let fullWindow = frames == fullWindowFrames
        let workspace = try scan.workspace(tokens: Self.bands * frames)
        var inputs = ["spectrum": spectrum]

        for index in 0..<InworldMambaScan.stageCount {
            try autoreleasepool {
                let acrossBands = !index.isMultiple(of: 2)
                let rows = acrossBands ? frames : Self.bands
                let steps = acrossBands ? Self.bands : frames
                let destination = try workspace.inputArray(rows: rows, steps: steps)
                let result = try predict(stage: index, inputs: inputs, fullWindow: fullWindow,
                                         scanBacking: fullWindow ? destination : nil)
                let features = try Self.array(result, named: "scan_features", shape: [rows, steps, Self.hiddenWidth])
                let projected = try Self.array(result, named: "scan_input", shape: [2, rows, steps, InworldMambaScan.packedWidth])
                if projected !== destination {
                    projected.transfer(to: destination)
                }
                let scanned = try scan.run(stage: index, rows: rows, steps: steps, workspace: workspace)
                inputs = ["features": features, "scan_output": scanned]
            }
        }

        return try autoreleasepool {
            let decoded = try predict(stage: InworldMambaScan.stageCount, inputs: inputs, fullWindow: fullWindow)
            let raw = try Self.array(decoded, named: "mask_residual", shape: [1, frames, InworldIsolationDSP.bins, 4])
            return dsp.synthesize(audio, spectrum: spectrum, maskResidual: raw)
        }
    }

    private func predict(
        stage index: Int, inputs: [String: MLMultiArray], fullWindow: Bool, scanBacking: MLMultiArray? = nil
    ) throws -> MLFeatureProvider {
        // Core ML retains prediction workspaces on each loaded model. Keep only
        // the active stage resident so long recordings stay within one workspace.
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        if fullWindow { configuration.functionName = "full_window" }
        let model = try CoreMLLoader.load(url: stageFiles.urls[index], configuration: configuration, name: "inworld-stage_\(index)")
        let options = MLPredictionOptions()
        if let scanBacking { options.outputBackings = ["scan_input": scanBacking] }
        return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs), options: options)
    }

    static func array(_ provider: MLFeatureProvider, named name: String, shape: [Int]) throws -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue,
              array.dataType == .float32, array.shape.map(\.intValue) == shape else {
            throw InworldIsolationError.predictionFailed("unexpected tensor '\(name)', expected Float32 \(shape)")
        }
        return array
    }

    private struct Manifest: Decodable {
        let format: String
        let fullWindowFrames: Int?
        let timeRows: Int?
        let bandRows: Int?
        let tiledBackings: Bool?
        let staticTiles: Bool?

        enum CodingKeys: String, CodingKey {
            case format
            case fullWindowFrames = "full_window_frames"
            case timeRows = "time_rows"
            case bandRows = "band_rows"
            case tiledBackings = "tiled_backings"
            case staticTiles = "static_tiles"
        }
    }
}

/// Resolve compiled stages for the lifetime of a loaded bundle.
final class InworldCoreMLStages {
    private(set) var urls: [URL] = []
    private var temporaryModels: [URL] = []

    init(directory: URL, count: Int) throws {
        for index in 0..<count {
            let name = "stage_\(index)"
            let compiled = directory.appendingPathComponent("\(name).mlmodelc")
            let package = directory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: compiled.path) {
                urls.append(compiled)
            } else if FileManager.default.fileExists(atPath: package.path) {
                let url = try MLModel.compileModel(at: package)
                temporaryModels.append(url)
                urls.append(url)
            } else {
                throw InworldIsolationError.invalidBundle("missing \(name).mlpackage or \(name).mlmodelc")
            }
        }
    }

    deinit {
        for url in temporaryModels { try? FileManager.default.removeItem(at: url) }
    }
}

enum InworldIsolationError: LocalizedError {
    case invalidBundle(String)
    case invalidAudio(String)
    case predictionFailed(String)

    var errorDescription: String? {
        switch self {
        case .invalidBundle(let reason): return "Invalid Inworld Voice Isolator bundle: \(reason)"
        case .invalidAudio(let reason): return "Invalid Inworld Voice Isolator audio settings: \(reason)"
        case .predictionFailed(let reason): return "Inworld Voice Isolator inference failed: \(reason)"
        }
    }
}

/// Anchor the last full window to the end of the recording, as in v1.
struct InworldIsolationWindow {
    let count: Int
    let overlap: Int

    init(seconds: Double, overlapMs: Int) throws {
        let sampleRate = InworldVoiceIsolator.sampleRate
        let minimumSeconds = Double(InworldIsolationDSP.hop) / Double(sampleRate)
        guard seconds >= minimumSeconds, seconds <= InworldVoiceIsolator.maxChunkSeconds, overlapMs >= 0 else {
            throw InworldIsolationError.invalidAudio("chunk length must be 0.01–8 seconds and overlap must be nonnegative")
        }
        count = Int(seconds * Double(sampleRate))
        overlap = min(count / 2, min(overlapMs, 4_000) * (sampleRate / 1_000))
    }

    func render(_ audio: [Float], process: (ArraySlice<Float>) throws -> [Float]) rethrows -> [Float] {
        try render(audio, pairWindows: false) { windows in try windows.map(process) }
    }

    func render(
        _ audio: [Float], pairWindows: Bool, processBatch: ([ArraySlice<Float>]) throws -> [[Float]]
    ) rethrows -> [Float] {
        guard !audio.isEmpty else { return [] }
        guard audio.count > count else { return try processBatch([audio[...]])[0] }

        var output: [Float] = []
        output.reserveCapacity(audio.count)
        let lastStart = audio.count - count
        var start = 0
        var previousEnd = 0
        while previousEnd < audio.count {
            let end = start + count
            var windows = [audio[start..<end]]
            if pairWindows && start < lastStart {
                let pairedStart = min(end - overlap, lastStart)
                windows.append(audio[pairedStart..<(pairedStart + count)])
            }

            let rendered = try processBatch(windows)
            for index in windows.indices {
                let window = windows[index]
                let next = rendered[index]
                if previousEnd > 0 {
                    let fadeStart = previousEnd - overlap
                    for i in 0..<overlap {
                        let t = Float(i) / Float(max(1, overlap - 1))
                        let offset = fadeStart + i
                        output[offset] = (1 - t) * output[offset] + t * next[offset - window.startIndex]
                    }
                }
                output.append(contentsOf: next[max(0, previousEnd - window.startIndex)...])
                previousEnd = window.endIndex
            }
            start = min(previousEnd - overlap, lastStart)
        }
        return output
    }
}
