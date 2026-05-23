import CoreML
import Foundation

/// Wrapper around `nanocodec_decoder.mlmodelc`.
///
/// IO (from bundle manifest):
/// - input:  `tokens` int32 (1, 8, 256)
/// - output: `audio`  fp32  (1, 262144)  = 256 frames × 1024 samples
///
/// The codec is a fixed-window batch decoder; you cannot stream with it. We
/// hard-cap the AR loop at 256 frames so the codec call always receives a
/// shape-compatible buffer.
public final class MagpieCoreMLNanoCodec {
    private let model: MLModel

    public init(url: URL) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "nanocodec_decoder")
    }

    /// Decode a `(T_frames, 8)` code matrix to a 22.05 kHz mono Float32 PCM
    /// waveform. Pads the time dimension up to 256 with `audio_eos_id` so the
    /// fixed-window codec doesn't choke; output is trimmed back to
    /// `T_frames * 1024` samples.
    public func decode(codes frames: [[Int32]]) throws -> [Float] {
        let maxFrames = MagpieCoreMLConstants.maxNanocodecFrames
        let K = MagpieCoreMLConstants.numCodebooks
        if frames.count == 0 { return [] }
        if frames.count > maxFrames {
            throw MagpieCoreMLError.audioTooLong(frames: frames.count, max: maxFrames)
        }
        // Pad/transpose into (1, 8, 256) int32. CoreML wants codebook on axis
        // 1, time on axis 2.
        var buffer = [Int32](repeating: MagpieCoreMLConstants.audioEosId,
                             count: K * maxFrames)
        for k in 0..<K {
            for t in 0..<frames.count {
                buffer[k * maxFrames + t] = frames[t][k]
            }
        }
        let arr = try MagpieCoreMLBridge.makeInt32(
            buffer,
            shape: [1, NSNumber(value: K), NSNumber(value: maxFrames)],
            label: "nanocodec/tokens")
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: arr)
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
        let allSamples = MagpieCoreMLBridge.toFloat32(audio)
        // Trim to the actual number of frames we generated. Codec emits 1024
        // samples per frame in C order.
        let trimCount = frames.count * MagpieCoreMLConstants.samplesPerFrame
        return Swift.Array(allSamples.prefix(trimCount))
    }
}
