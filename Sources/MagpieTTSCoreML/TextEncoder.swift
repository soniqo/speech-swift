import CoreML
import Foundation

/// Wrapper around `text_encoder.mlmodelc`.
///
/// IO (from bundle manifest):
/// - inputs:  `text_tokens` int32  (1, 256)
///            `text_mask`   fp16   (1, 256)
/// - outputs: `encoder_output` fp16 (1, 256, 768)
///            `encoder_mask`   fp16 (1, 256)
public final class MagpieCoreMLTextEncoder {
    public struct Output {
        public let encoderOutput: MLMultiArray
        public let encoderMask: MLMultiArray
    }

    private let model: MLModel

    public init(url: URL) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "text_encoder")
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
            label: "text_encoder/text_tokens")
        let maskArr = try MagpieCoreMLBridge.makeFp16(
            mask, shape: [1, NSNumber(value: maxLen)],
            label: "text_encoder/text_mask")

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokensArr),
            "text_mask":   MLFeatureValue(multiArray: maskArr),
        ])
        let pred: MLFeatureProvider
        do {
            pred = try model.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "text_encoder", underlying: String(describing: error))
        }
        guard let enc = pred.featureValue(for: "encoder_output")?.multiArrayValue,
              let mOut = pred.featureValue(for: "encoder_mask")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "text_encoder", underlying: "missing encoder_output/encoder_mask")
        }
        return Output(encoderOutput: enc, encoderMask: mOut)
    }
}
