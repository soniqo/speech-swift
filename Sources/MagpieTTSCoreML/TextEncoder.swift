import CoreML
import Foundation

/// Wrapper around `text_encoder.mlmodelc`.
///
/// IO (from `model.mil`):
/// - inputs:  `tokens` int32 (1, 256)
///            `mask`   fp32  (1, 256)
/// - output:  `encoder_output` fp32 (1, 256, 768)
public final class MagpieCoreMLTextEncoder {
    public struct Output {
        /// (1, 256, 768) FP32 — passed verbatim into `decoder_prefill` /
        /// `decoder_step` (both expect FP32 at IO).
        public let encoderOutput: MLMultiArray
        /// (1, 256) FP32 — 1.0 for real tokens, 0.0 for padding. We keep
        /// it as the same `MLMultiArray` we built for the encoder input
        /// so callers don't allocate it twice (decoder also wants it).
        public let encoderMask: MLMultiArray
    }

    private let model: MLModel

    public init(url: URL) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "text_encoder", kind: .decoder)
    }

    public func encode(tokens ids: [Int32]) throws -> Output {
        let maxLen = MagpieCoreMLConstants.maxTextTokens
        if ids.count > maxLen {
            throw MagpieCoreMLError.textTooLong(tokens: ids.count, max: maxLen)
        }
        var padded = ids
        var mask = [Float](repeating: 1.0, count: ids.count)
        if padded.count < maxLen {
            let pad = maxLen - padded.count
            padded.append(contentsOf: [Int32](repeating: 0, count: pad))
            mask.append(contentsOf: [Float](repeating: 0.0, count: pad))
        }
        let tokensArr = try MagpieCoreMLBridge.makeInt32(
            padded, shape: [1, NSNumber(value: maxLen)],
            label: "text_encoder/tokens")
        let maskArr = try MagpieCoreMLBridge.makeFp32(
            mask, shape: [1, NSNumber(value: maxLen)],
            label: "text_encoder/mask")

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokensArr),
            "mask":   MLFeatureValue(multiArray: maskArr),
        ])
        let pred: MLFeatureProvider
        do {
            pred = try model.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "text_encoder", underlying: String(describing: error))
        }
        guard let enc = pred.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "text_encoder", underlying: "missing encoder_output")
        }
        return Output(encoderOutput: enc, encoderMask: maskArr)
    }
}
