import CoreML
import Foundation

/// Wrapper around the nanocodec mlmodelc files. The bundle ships two
/// variants:
/// - `nanocodec_decoder.mlmodelc` — 64-frame batch window, used by
///   `synthesize()` (1 codec call per ~3 s of audio).
/// - `nanocodec_decoder_streaming.mlmodelc` — 8-frame streaming window,
///   used by `synthesizeStream()` for ~370 ms first-packet latency.
///
/// IO (both variants):
/// - input:  `latents` fp32 (1, 32, W) — FSQ-inverse latents in NCL layout
/// - output: `audio`   fp32 (1, W * 1024)
public final class MagpieCoreMLNanoCodec {
    private let model: MLModel
    public let windowFrames: Int

    public init(url: URL, windowFrames: Int = MagpieCoreMLConstants.nanocodecBatchFrames) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "nanocodec_decoder", kind: .codec)
        self.windowFrames = windowFrames
    }

    /// One-shot ANE warm-up. Runs a dummy zero-latent prediction so the
    /// first real call doesn't pay JIT/compile cost. ~250–500 ms on first
    /// invocation; subsequent calls are no-ops as CoreML caches.
    public func prewarm() {
        do {
            let zeros = [Float](
                repeating: 0,
                count: MagpieCoreMLConstants.fsqLatentDim * windowFrames)
            let arr = try MagpieCoreMLBridge.makeFp32(
                zeros,
                shape: [1,
                        NSNumber(value: MagpieCoreMLConstants.fsqLatentDim),
                        NSNumber(value: windowFrames)],
                label: "prewarm/latents")
            let features = try MLDictionaryFeatureProvider(dictionary: [
                "latents": MLFeatureValue(multiArray: arr),
            ])
            _ = try model.prediction(from: features)
        } catch {
            // Pre-warm is best-effort; swallow.
        }
    }

    /// Decode a `T_frames × 8` code matrix to 22.05 kHz mono Float32 PCM.
    /// `T_frames` may exceed the codec window — we chunk internally.
    public func decode(codes frames: [[Int32]]) throws -> [Float] {
        if frames.isEmpty { return [] }
        let samplesPerFrame = MagpieCoreMLConstants.samplesPerFrame

        var out: [Float] = []
        out.reserveCapacity(frames.count * samplesPerFrame)

        var idx = 0
        while idx < frames.count {
            let end = min(idx + windowFrames, frames.count)
            let slice = frames[idx..<end]
            let chunk = try decodeWindow(frames: slice)
            // Trim per-window output to the number of real frames in this
            // chunk (the codec emits W * 1024 samples regardless of how
            // many real frames are in the window; trailing zero-padded
            // frames produce meaningless audio that we drop).
            let validSamples = (end - idx) * samplesPerFrame
            out.append(contentsOf: chunk.prefix(validSamples))
            idx = end
        }
        return out
    }

    /// Decode a single window of exactly `windowFrames` codes (pads zeros
    /// if `frames.count < windowFrames`). Exposed for the streaming path
    /// which emits one chunk per codec call.
    public func decodeWindow(frames: ArraySlice<[Int32]>) throws -> [Float] {
        precondition(frames.count <= windowFrames,
                     "decodeWindow: expected ≤\(windowFrames) frames, got \(frames.count)")
        let latentsFlat = MagpieCoreMLFSQ.decodeWindow(
            frames: frames, windowFrames: windowFrames)
        let latentsArr = try MagpieCoreMLBridge.makeFp32(
            latentsFlat,
            shape: [1,
                    NSNumber(value: MagpieCoreMLConstants.fsqLatentDim),
                    NSNumber(value: windowFrames)],
            label: "nanocodec/latents")
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "latents": MLFeatureValue(multiArray: latentsArr),
        ])
        let pred: MLFeatureProvider
        do {
            pred = try model.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "nanocodec_decoder", underlying: String(describing: error))
        }
        guard let audio = pred.featureValue(for: "audio")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "nanocodec_decoder", underlying: "missing audio output")
        }
        return MagpieCoreMLBridge.toFloat32(audio)
    }
}
