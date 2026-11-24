import AudioCommon
import CoreML
import Foundation

/// Incremental 48 kHz mono voice isolation with an Inworld realtime bundle.
public final class InworldVoiceIsolatorStream {
    public static let sampleRate = InworldVoiceIsolator.sampleRate

    private let session: any InworldStreamingSession

    /// Create a session for 0, 2, or 4 frames of look-ahead. Serialize calls to this session.
    public init(modelDirectory: URL, lookaheadFrames: Int = 2, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
        let manifest = try InworldStreamingManifest(directory: modelDirectory)
        guard let lookaheadIndex = manifest.lookahead_frames_choices.firstIndex(of: lookaheadFrames) else {
            throw InworldIsolationError.invalidAudio("streaming look-ahead must be 0, 2, or 4 frames")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = CoreMLComputeUnitsResolver.resolved(default: computeUnits)
        guard configuration.computeUnits == .cpuAndGPU || configuration.computeUnits == .all else {
            throw InworldIsolationError.invalidBundle("the realtime model requires GPU compute")
        }

        switch manifest.format {
        case .stateful:
            let backbone = try InworldStreamingBackbone(modelDirectory: modelDirectory,
                blockFrames: manifest.block_frames, lookaheadIndex: lookaheadIndex, configuration: configuration)
            self.session = try InworldPCMStream(backbone: backbone, lookaheadFrames: lookaheadFrames)
        case .paddedAttention, .specializedAttention, .metalDecoder:
            let valueEncoding = manifest.kv_cache_bits ?? .float16
            let backbone = try InworldPairedStreamingBackbone(modelDirectory: modelDirectory,
                blockFrames: manifest.block_frames, lookaheadIndex: lookaheadIndex, configuration: configuration,
                useSingleLaneStages: manifest.format != .paddedAttention,
                useMetalDecoder: manifest.format == .metalDecoder,
                keyEncoding: manifest.key_cache_bits ?? valueEncoding, valueEncoding: valueEncoding)
            self.session = try InworldPCMStream(backbone: backbone, lookaheadFrames: lookaheadFrames)
        }
    }

    /// Analysis and look-ahead latency at 48 kHz, excluding inference and audio transport.
    public var latencySamples: Int { session.latencySamples }

    /// The next PCM block size in samples, accounting for analysis and crossfade boundaries.
    /// A closed session returns zero. Calls to `push` can use any input size.
    public var preferredPushSamples: Int { session.preferredPushSamples }

    /// Run one warm-up prediction before audio capture. Call this before pushing any samples.
    public func prepare() throws {
        try session.prepare()
    }

    /// Consume finite mono Float32 samples and return the samples that became final.
    public func push(_ samples: [Float]) throws -> [Float] {
        try session.push(samples)
    }

    /// Finish the session using zero-padded analysis and return the remaining samples.
    public func flush() throws -> [Float] {
        try session.flush()
    }
}

enum InworldStreamingLayout {
    static let windowFrames = 600
    static let crossfadeFrames = 200
    static let laneCacheFrames = windowFrames + crossfadeFrames
    static let pairedBlockFrames = 2
    static let pairedLanes = 2

    static func framesToBoundary(at position: Int) -> Int {
        let startOfFade = windowFrames - crossfadeFrames
        return (position < startOfFade ? startOfFade : windowFrames) - position
    }

    static func preferredPushSamples(analyzed: Int, received: Int, blockFrames: Int) -> Int {
        let frames = min(blockFrames, framesToBoundary(at: analyzed % windowFrames))
        let lastFrame = analyzed + frames - 1
        return lastFrame * InworldIsolationDSP.hop + InworldIsolationDSP.fftSize / 2 - received
    }
}

private struct InworldStreamingManifest: Decodable {
    enum Format: String, Decodable {
        case stateful = "inworld-voice-isolator-realtime-coreml-stateful-v1"
        case paddedAttention = "inworld-voice-isolator-realtime-coreml-padded-attention-v1"
        case specializedAttention = "inworld-voice-isolator-realtime-coreml-specialized-attention-v1"
        case metalDecoder = "inworld-voice-isolator-realtime-coreml-metal-decoder-v1"

        func supportsBlockFrames(_ frames: Int) -> Bool {
            switch self {
            case .stateful: (1...8).contains(frames)
            case .paddedAttention, .specializedAttention: frames == InworldStreamingLayout.pairedBlockFrames
            case .metalDecoder: [2, 4, 8].contains(frames)
            }
        }
    }

    let format: Format
    let sample_rate: Int
    let block_frames: Int
    let window_frames: Int
    let stateful_cache_frames: Int
    let lookahead_frames_choices: [Int]
    let kv_cache_bits: InworldStreamingAttention.CacheEncoding?
    let key_cache_bits: InworldStreamingAttention.CacheEncoding?

    private var supportsCacheEncoding: Bool {
        guard kv_cache_bits != .packed10, format != .stateful || kv_cache_bits != .packed12 else { return false }
        guard let key_cache_bits else { return true }
        return key_cache_bits == .packed10 && kv_cache_bits == .packed12 && format == .metalDecoder && block_frames == 8
    }

    init(directory: URL) throws {
        let url = directory.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url),
              let manifest = try? JSONDecoder().decode(Self.self, from: data),
              manifest.format.supportsBlockFrames(manifest.block_frames),
              manifest.supportsCacheEncoding,
              manifest.sample_rate == InworldVoiceIsolator.sampleRate,
              manifest.stateful_cache_frames == InworldStreamingLayout.laneCacheFrames,
              manifest.window_frames >= manifest.stateful_cache_frames,
              manifest.lookahead_frames_choices == [0, 2, 4] else {
            throw InworldIsolationError.invalidBundle("expected a v1 realtime manifest in \(directory.path)")
        }
        self = manifest
    }
}

protocol InworldStreamingInference {
    associatedtype Lane
    var blockFrames: Int { get }
    func makeLane() throws -> Lane
    func beginLane(copying active: Lane, reusing spare: Lane?) throws -> Lane
    func retireLane(_ lane: Lane)
    /// Warm the inference models and return an empty primary lane.
    func prepare(_ spectrum: MLMultiArray, lane: Lane) throws -> Lane
    /// Return decoder output in the requested lane order.
    func predict(_ spectrum: MLMultiArray, frames: Int, lanes: [Lane]) throws -> [MLMultiArray]
}

extension InworldStreamingInference {
    func prepare(_ spectrum: MLMultiArray, lane: Lane) throws -> Lane {
        _ = try predict(spectrum, frames: blockFrames, lanes: [lane])
        retireLane(lane)
        return try makeLane()
    }
}

private protocol InworldStreamingSession {
    var latencySamples: Int { get }
    var preferredPushSamples: Int { get }
    func prepare() throws
    func push(_ samples: [Float]) throws -> [Float]
    func flush() throws -> [Float]
}

private final class InworldPCMStream<Backbone: InworldStreamingInference>: InworldStreamingSession {
    private static var hop: Int { InworldIsolationDSP.hop }
    private static var pad: Int { InworldIsolationDSP.fftSize / 2 }

    private let backbone: Backbone
    private let dsp: InworldIsolationDSP
    private let lookaheadFrames: Int
    private var primary: Backbone.Lane
    private var secondary: Backbone.Lane?
    private var spare: Backbone.Lane?
    private var pending: [Frame] = []
    private var workspace = InworldIsolationDSP.SynthesisWorkspace()
    private var audio: [Float] = []
    private var audioBase = 0
    private var signal: [Float] = []
    private var weights: [Float] = []
    private var synthesisBase = 0
    private var analyzed = 0
    private var synthesized = 0
    private var released = 0
    private var closed = false
    private var prepared = false

    init(backbone: Backbone, lookaheadFrames: Int) throws {
        self.backbone = backbone
        self.lookaheadFrames = lookaheadFrames
        self.primary = try backbone.makeLane()
        self.dsp = try InworldIsolationDSP()
    }

    var latencySamples: Int { InworldIsolationDSP.fftSize + lookaheadFrames * Self.hop }

    var preferredPushSamples: Int {
        guard !closed else { return 0 }
        return InworldStreamingLayout.preferredPushSamples(analyzed: analyzed,
            received: audioBase + audio.count, blockFrames: backbone.blockFrames)
    }

    func prepare() throws {
        guard !closed, audioBase == 0, audio.isEmpty else {
            throw InworldIsolationError.invalidAudio("prepare the isolation stream before pushing audio")
        }
        guard !prepared else { return }
        do {
            try autoreleasepool {
                let spectrum = try analyze(frames: backbone.blockFrames)
                primary = try backbone.prepare(spectrum, lane: primary)
            }
            prepared = true
        } catch {
            closed = true
            throw error
        }
    }

    func push(_ samples: [Float]) throws -> [Float] {
        guard !closed else { throw InworldIsolationError.invalidAudio("the isolation stream is closed") }
        guard samples.allSatisfy(\.isFinite) else {
            throw InworldIsolationError.invalidAudio("expected finite mono PCM")
        }
        audio.append(contentsOf: samples)
        let total = audioBase + audio.count
        let limit = total >= Self.pad ? (total - Self.pad) / Self.hop + 1 : 0
        do {
            try advance(to: limit)
        } catch {
            closed = true
            throw error
        }
        let output = release(upTo: synthesized * Self.hop - Self.pad)
        trim()
        return output
    }

    func flush() throws -> [Float] {
        guard !closed else { return [] }
        closed = true
        let total = audioBase + audio.count
        guard total > 0 else { return [] }
        try advance(to: total / Self.hop + 1)
        synthesizePending(final: true)
        let output = release(upTo: total)
        trim()
        return output
    }

    private func advance(to limit: Int) throws {
        let window = InworldStreamingLayout.windowFrames
        let crossfade = InworldStreamingLayout.crossfadeFrames
        let fadeStart = window - crossfade
        while analyzed < limit {
            let position = analyzed % window
            let frames = min(backbone.blockFrames, limit - analyzed, InworldStreamingLayout.framesToBoundary(at: position))
            try autoreleasepool {
                let spectrum = try analyze(frames: frames)
                if position >= fadeStart && secondary == nil {
                    secondary = try backbone.beginLane(copying: primary, reusing: spare)
                    spare = nil
                }
                let lanes = secondary.map { [primary, $0] } ?? [primary]
                let decoded = try backbone.predict(spectrum, frames: frames, lanes: lanes)
                let raw = decoded[0]
                if let secondary {
                    Self.blend(raw, with: decoded[1], frames: frames, first: position - fadeStart, length: crossfade)
                    if position + frames == window {
                        backbone.retireLane(primary)
                        spare = primary
                        primary = secondary
                        self.secondary = nil
                    }
                }
                for frame in 0..<frames {
                    pending.append(Frame(spectrum: spectrum, raw: raw, index: frame))
                }
                analyzed += frames
                synthesizePending(final: false)
            }
        }
    }

    private func analyze(frames: Int) throws -> MLMultiArray {
        let first = analyzed * Self.hop - Self.pad
        let count = (frames - 1) * Self.hop + InworldIsolationDSP.fftSize
        var padded = [Float](repeating: 0, count: count)
        let start = max(audioBase, first)
        let end = min(audioBase + audio.count, first + count)
        if start < end {
            padded.replaceSubrange((start - first)..<(end - first), with: audio[(start - audioBase)..<(end - audioBase)])
        }
        return try dsp.spectrum(padded: padded, frames: frames)
    }

    private func synthesizePending(final: Bool) {
        let count = final ? pending.count : max(0, pending.count - lookaheadFrames)
        guard count > 0 else { return }
        let last = pending.count - 1
        let needed = (synthesized + count - 1) * Self.hop - synthesisBase + InworldIsolationDSP.fftSize
        if needed > signal.count {
            let padding = repeatElement(Float(0), count: needed - signal.count)
            signal.append(contentsOf: padding)
            weights.append(contentsOf: padding)
        }
        for i in 0..<count {
            let source = pending[i]
            let correction = pending[min(i + lookaheadFrames, last)]
            let offset = (synthesized + i) * Self.hop - synthesisBase
            dsp.addCorrection(source.spectrum, frame: source.index, mask: correction.raw,
                              maskFrame: correction.index, at: offset,
                              signal: &signal, weights: &weights, workspace: &workspace)
        }
        synthesized += count
        pending.removeFirst(count)
    }

    private func release(upTo limit: Int) -> [Float] {
        let end = min(limit, audioBase + audio.count)
        guard end > released else { return [] }
        let start = released
        released = end
        return (start..<end).map { sample in
            let index = sample + Self.pad - synthesisBase
            return audio[sample - audioBase] + signal[index] / max(weights[index], 1e-11)
        }
    }

    private func trim() {
        let keepAudio = min(released, max(0, analyzed * Self.hop - Self.pad))
        audio.removeFirst(keepAudio - audioBase)
        audioBase = keepAudio
        let keepSynthesis = min(synthesized * Self.hop, released + Self.pad)
        signal.removeFirst(keepSynthesis - synthesisBase)
        weights.removeFirst(keepSynthesis - synthesisBase)
        synthesisBase = keepSynthesis
    }

    private static func blend(_ firstLane: MLMultiArray, with secondLane: MLMultiArray, frames: Int,
                              first: Int, length: Int) {
        let secondStrides = secondLane.strides.map(\.intValue)
        secondLane.withUnsafeBufferPointer(ofType: Float.self) { next in
            firstLane.withUnsafeMutableBufferPointer(ofType: Float.self) { current, firstStrides in
                for frame in 0..<frames {
                    let weight = Float(first + frame + 1) / Float(length)
                    for bin in 0..<InworldIsolationDSP.bins {
                        for component in 0..<4 {
                            let a = frame * firstStrides[1] + bin * firstStrides[2] + component * firstStrides[3]
                            let b = frame * secondStrides[1] + bin * secondStrides[2] + component * secondStrides[3]
                            current[a] = (1 - weight) * current[a] + weight * next[b]
                        }
                    }
                }
            }
        }
    }

    private struct Frame {
        let spectrum: MLMultiArray
        let raw: MLMultiArray
        let index: Int
    }
}
