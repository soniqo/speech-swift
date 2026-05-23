import Foundation
import MLX
import MagpieTTS

/// Lazy-loaded MLX MagpieTTS instance. We use it for two things:
///   1. The 8 audio embedding tables — averaged per frame to build the
///      next `audio_emb` input for the CoreML decoder.
///   2. The 1-layer LocalTransformer — refines `decoder_step`'s `h_last`
///      into the 8 codebook tokens (AR sampling head). Using MLX here is
///      a pragmatic shortcut so we don't ship a second copy of the LT
///      weights inside the CoreML bundle. iOS deployment that wants to
///      drop the MLX dependency will need a Swift-side LT and a CoreML
///      bundle revision that ships LT/audio-embedding .npy files.
///
/// Loads the INT4 MLX bundle (~247 MB) on first use; the OS caches it so
/// subsequent runs are no-ops.
public final class MagpieMLXHelper {
    private var model: MagpieTTS?
    private let lock = NSLock()

    public init() {}

    public func ensure() throws -> MagpieTTS {
        lock.lock()
        defer { lock.unlock() }
        if let m = model { return m }
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<MagpieTTS, Error> = .failure(
            MagpieCoreMLError.inferenceFailed(stage: "mlx_load",
                                              underlying: "task never resumed"))
        Task {
            do {
                let m = try await MagpieTTS.fromPretrained(variant: .int4)
                result = .success(m)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        let m = try result.get()
        self.model = m
        return m
    }
}

/// MLX-backed audio embedding source. Calls into the MLX MagpieTTS module
/// to compute per-frame averaged embeddings.
public final class MagpieMLXAudioEmbedSource: AudioEmbedSource {
    private let helper: MagpieMLXHelper

    public init(helper: MagpieMLXHelper) {
        self.helper = helper
    }

    public func averageBosEmbedding() throws -> [Float] {
        let bos = MagpieCoreMLConstants.audioBosId
        let codes = [Int32](repeating: bos, count: MagpieCoreMLConstants.numCodebooks)
        return try averageEmbedding(codes: codes)
    }

    public func averageEmbedding(codes: [Int32]) throws -> [Float] {
        let model = try helper.ensure()
        let codesArr = MLXArray(codes, [MagpieCoreMLConstants.numCodebooks])
            .reshaped([1, 1, MagpieCoreMLConstants.numCodebooks])
        let emb = model.decoder.embedAudioFrame(codesArr)
        // emb shape: (1, 1, D) — flatten.
        let flat: [Float] = emb.asArray(Float.self)
        precondition(flat.count == MagpieCoreMLConstants.dModel)
        return flat
    }
}

/// MLX-backed LocalTransformer sampler. Wraps the MLX MagpieTTS's
/// ``sampleLocalTransformer`` so the CoreML decoder's `h_last` (FP32
/// `(1, 1, 768)` MLMultiArray) can be fed straight into the validated
/// MLX sampler. Returns the 8 codebook tokens for one frame.
public final class MagpieMLXLocalSampler {
    private let helper: MagpieMLXHelper

    public init(helper: MagpieMLXHelper) {
        self.helper = helper
    }

    public func sample(hLastFlat: [Float],
                forbidEos: Bool,
                temperature: Float,
                topK: Int) throws -> [Int32] {
        let model = try helper.ensure()
        precondition(hLastFlat.count == MagpieCoreMLConstants.dModel)
        let hLast = MLXArray(hLastFlat, [MagpieCoreMLConstants.dModel])
            .reshaped([1, 1, MagpieCoreMLConstants.dModel])
        return model.sampleLocalTransformer(
            hLast: hLast,
            forbidEos: forbidEos,
            temperature: temperature,
            topK: topK)
    }
}
