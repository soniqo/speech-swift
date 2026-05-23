import CoreML
import Foundation

/// Wraps `decoder_prefill.mlmodelc` + `decoder_step.mlmodelc` and manages the
/// 12-layer KV cache between them.
///
/// `decoder_prefill` runs once at the start of each utterance: input is the
/// 110-frame speaker context (already converted to embeddings), and the
/// text-encoder output + mask. It returns the last hidden state of the prefix
/// plus 12 packed `(2, 1, 512, 12, 64)` KV cache tensors and 12 int32 position
/// counters.
///
/// `decoder_step` is the AR hot loop: one `(1, 1, 768)` audio embedding per
/// call, 24 split KV inputs `cache_k* / cache_v*`, 12 fp16 position scalars.
/// Returns updated split-K/V tensors + new position scalars + decoder hidden
/// (which the LocalTransformer + sampler turn into the next codebook frame).
///
/// Mapping the irregular auto-generated MIL output names (`new_k`, `new_k_1`,
/// `new_k_3`, …, `new_k_21`) back to ordered slots 0…11 is the trickiest
/// part; see ``stepNewKeyName`` / ``stepNewValueName`` / ``stepPositionName``.
public final class MagpieCoreMLDecoder {

    public struct PrefillOutput {
        public let hiddenStates: MLMultiArray
        /// 12 split-K caches, ready to feed into ``step(...)`` as `cache_k0…11`.
        public let cacheK: [MLMultiArray]
        /// 12 split-V caches.
        public let cacheV: [MLMultiArray]
        /// 12 fp16 scalar positions (converted from the prefill's int32 outputs).
        public let positions: [MLMultiArray]
    }

    public struct StepOutput {
        /// `[dModel]` Float32 — the conditional decoder hidden ready for the
        /// LocalTransformer sampler.
        public let decoderHidden: [Float]
        public let cacheK: [MLMultiArray]
        public let cacheV: [MLMultiArray]
        public let positions: [MLMultiArray]
    }

    private let prefill: MLModel
    private let step: MLModel
    private let numLayers: Int

    public init(prefillURL: URL, stepURL: URL,
                numLayers: Int = MagpieCoreMLConstants.numDecoderLayers) throws {
        self.prefill = try MagpieCoreMLBridge.loadCompiled(at: prefillURL, label: "decoder_prefill")
        self.step = try MagpieCoreMLBridge.loadCompiled(at: stepURL, label: "decoder_step")
        self.numLayers = numLayers
    }

    // MARK: - Prefill

    /// Run the prefill pass once at the start of an utterance.
    /// - Parameter contextEmbedding: speaker context FP32 flat buffer, length
    ///   `speakerContextLength * dModel` (110 × 768).
    /// - Parameter encoderOutput: from `text_encoder`.
    /// - Parameter encoderMask: from `text_encoder`.
    public func prefill(contextEmbedding: [Float],
                         encoderOutput: MLMultiArray,
                         encoderMask: MLMultiArray) throws -> PrefillOutput {
        let T = MagpieCoreMLConstants.speakerContextLength
        let D = MagpieCoreMLConstants.dModel
        precondition(contextEmbedding.count == T * D,
                     "context embedding count = \(contextEmbedding.count), expected \(T * D)")
        let inputArr = try MagpieCoreMLBridge.makeFp16(
            contextEmbedding,
            shape: [1, NSNumber(value: T), NSNumber(value: D)],
            label: "decoder_prefill/input")

        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input":          MLFeatureValue(multiArray: inputArr),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "encoder_mask":   MLFeatureValue(multiArray: encoderMask),
        ])
        let pred: MLFeatureProvider
        do {
            pred = try prefill.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_prefill", underlying: String(describing: error))
        }
        guard let hidden = pred.featureValue(for: "hidden_states")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_prefill", underlying: "missing hidden_states output")
        }

        var cacheK: [MLMultiArray] = []
        var cacheV: [MLMultiArray] = []
        var positions: [MLMultiArray] = []
        cacheK.reserveCapacity(numLayers)
        cacheV.reserveCapacity(numLayers)
        positions.reserveCapacity(numLayers)
        for layer in 0..<numLayers {
            guard let packed = pred.featureValue(for: "cache_\(layer)")?.multiArrayValue else {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "missing cache_\(layer)")
            }
            let split = try MagpieCoreMLBridge.splitKVCache(packed, label: "prefill cache_\(layer)")
            cacheK.append(split.k)
            cacheV.append(split.v)

            guard let posInt = pred.featureValue(for: "position_\(layer)")?.multiArrayValue else {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "missing position_\(layer)")
            }
            let posFp16 = try MagpieCoreMLBridge.scalarInt32ToFp16(posInt, label: "position_\(layer)")
            positions.append(posFp16)
        }
        return PrefillOutput(
            hiddenStates: hidden, cacheK: cacheK, cacheV: cacheV, positions: positions)
    }

    // MARK: - Step

    /// One AR step. Inputs the previous frame's averaged audio embedding,
    /// current KV cache slices, and the encoder context. Returns the updated
    /// cache + new decoder hidden.
    public func step(audioEmbedding: [Float],
                      encoderOutput: MLMultiArray, encoderMask: MLMultiArray,
                      cacheK: [MLMultiArray], cacheV: [MLMultiArray],
                      positions: [MLMultiArray]) throws -> StepOutput {
        let D = MagpieCoreMLConstants.dModel
        precondition(audioEmbedding.count == D)
        precondition(cacheK.count == numLayers)
        precondition(cacheV.count == numLayers)
        precondition(positions.count == numLayers)

        let audioArr = try MagpieCoreMLBridge.makeFp16(
            audioEmbedding,
            shape: [1, 1, NSNumber(value: D)],
            label: "decoder_step/audio_embed")

        // decoder_step takes `encoder_mask` as bool, but feeding fp16 in works
        // because CoreML's bool MLMultiArrays are 1 byte each — manifest notes
        // dtype as bool. Our prefill encoder_mask is fp16, so we re-pack as
        // bool if needed below. Trying fp16 first and only re-packing on
        // failure is wasteful, so we always re-pack defensively.
        let maskBool = try makeStepEncoderMask(from: encoderMask)

        var dict: [String: MLFeatureValue] = [
            "audio_embed":     MLFeatureValue(multiArray: audioArr),
            "encoder_output":  MLFeatureValue(multiArray: encoderOutput),
            "encoder_mask":    MLFeatureValue(multiArray: maskBool),
        ]
        for layer in 0..<numLayers {
            dict["cache_k\(layer)"] = MLFeatureValue(multiArray: cacheK[layer])
            dict["cache_v\(layer)"] = MLFeatureValue(multiArray: cacheV[layer])
            dict["position\(layer)"] = MLFeatureValue(multiArray: positions[layer])
        }
        let features = try MLDictionaryFeatureProvider(dictionary: dict)
        let pred: MLFeatureProvider
        do {
            pred = try step.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_step", underlying: String(describing: error))
        }

        // Pull decoder hidden ("input" output per manifest) and convert to fp32 [D].
        guard let hidden = pred.featureValue(for: "input")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_step", underlying: "missing decoder hidden output")
        }
        let hiddenF32 = MagpieCoreMLBridge.toFloat32(hidden)
        precondition(hiddenF32.count == 1 * 1 * D)

        // Updated KV / positions: irregular auto-generated names.
        var newK: [MLMultiArray] = []
        var newV: [MLMultiArray] = []
        var newPos: [MLMultiArray] = []
        newK.reserveCapacity(numLayers)
        newV.reserveCapacity(numLayers)
        newPos.reserveCapacity(numLayers)
        for layer in 0..<numLayers {
            let kName = Self.stepNewKeyName(layer: layer)
            let vName = Self.stepNewValueName(layer: layer)
            let pName = Self.stepPositionName(layer: layer)
            guard let kArr = pred.featureValue(for: kName)?.multiArrayValue,
                  let vArr = pred.featureValue(for: vName)?.multiArrayValue,
                  let pArr = pred.featureValue(for: pName)?.multiArrayValue else {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing one of \(kName)/\(vName)/\(pName)")
            }
            newK.append(try MagpieCoreMLBridge.cloneFp16(kArr, label: kName))
            newV.append(try MagpieCoreMLBridge.cloneFp16(vArr, label: vName))
            newPos.append(try MagpieCoreMLBridge.cloneScalarFp16(pArr, label: pName))
        }
        return StepOutput(
            decoderHidden: hiddenF32,
            cacheK: newK, cacheV: newV, positions: newPos)
    }

    private func makeStepEncoderMask(from fp16Mask: MLMultiArray) throws -> MLMultiArray {
        let count = fp16Mask.count
        let arr = try MLMultiArray(shape: fp16Mask.shape, dataType: .float32)
        switch fp16Mask.dataType {
        case .float16:
            let src = fp16Mask.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count {
                dst[i] = NpyReader.float16ToFloat32(bits: src[i]) > 0 ? 1.0 : 0.0
            }
        case .float32:
            let src = fp16Mask.dataPointer.bindMemory(to: Float.self, capacity: count)
            let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] > 0 ? 1.0 : 0.0 }
        default:
            let src = MagpieCoreMLBridge.toFloat32(fp16Mask)
            let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { dst[i] = src[i] > 0 ? 1.0 : 0.0 }
        }
        return arr
    }

    // MARK: - Name mapping for the step model's irregular MIL outputs

    /// Updated split-K cache output names. From the FluidInference manifest:
    /// `new_k_1, new_k_3, new_k_5, …, new_k_21, new_k`. That is: layers 0..10
    /// use `new_k_{2*layer+1}`, layer 11 uses `new_k`.
    static func stepNewKeyName(layer: Int) -> String {
        if layer == 11 { return "new_k" }
        return "new_k_\(layer * 2 + 1)"
    }

    static func stepNewValueName(layer: Int) -> String {
        if layer == 11 { return "new_v" }
        return "new_v_\(layer * 2 + 1)"
    }

    /// Advanced position outputs: `var_169, var_339, …, var_2039`. They start
    /// at 169 and step by 170; layer 11 is the final one.
    static func stepPositionName(layer: Int) -> String {
        return "var_\(169 + layer * 170)"
    }
}
