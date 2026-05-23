import CoreML
import Foundation

/// Wrapper around `nanocodec_decoder.mlmodelc`.
///
/// IO (from `model.mil`):
/// - input:  `latents` fp32 (1, 32, 64) — FSQ-inverse latents in NCL layout
/// - output: `audio`   fp32 (1, 65536)  — 64 frames × 1024 samples
///
/// The model is traced at a fixed 64-frame window. Longer code sequences
/// are chunked here and the per-call PCM outputs are concatenated.
public final class MagpieCoreMLNanoCodec {
    private let model: MLModel

    public init(url: URL) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "nanocodec_decoder")
    }

    /// Decode a `T_frames × 8` code matrix to 22.05 kHz mono Float32 PCM.
    /// `T_frames` may exceed the codec window — we chunk internally.
    public func decode(codes frames: [[Int32]]) throws -> [Float] {
        if frames.isEmpty { return [] }
        let windowFrames = MagpieCoreMLConstants.nanocodecFramesPerWindow
        let samplesPerFrame = MagpieCoreMLConstants.samplesPerFrame

        var out: [Float] = []
        out.reserveCapacity(frames.count * samplesPerFrame)

        var idx = 0
        while idx < frames.count {
            let end = min(idx + windowFrames, frames.count)
            let slice = frames[idx..<end]
            let latentsFlat = MagpieCoreMLFSQ.decodeWindow(frames: slice)
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
            let pcm = MagpieCoreMLBridge.toFloat32(audio)
            // Trim per-window output to the number of real frames in this
            // chunk (the codec is invariant to the trailing zero-padded
            // frames, but the audio it generates for them is meaningless).
            let validSamples = (end - idx) * samplesPerFrame
            out.append(contentsOf: pcm.prefix(validSamples))
            idx = end
        }
        return out
    }
}
