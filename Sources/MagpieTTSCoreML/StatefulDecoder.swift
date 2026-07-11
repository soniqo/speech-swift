import CoreML
import Foundation

/// Wraps `decoder_step_stateful.mlmodelc` — the variant with KV cache
/// held in CoreML state (fp16, ANE-resident) instead of passed as
/// inputs/outputs each step. Reduces per-step IO marshalling: ~85 MB
/// of fp32 cache transfer per step → 0 (state stays in ANE memory).
///
/// IO of the stateful model:
/// - Inputs:  `audio_emb` (1,1,768) fp32, `encoder_output` (1,256,768) fp32,
///            `encoder_mask` (1,256) fp32, `position` (1,) int32,
///            12 × `xa_k_i` (1,256,1,128) fp32, 12 × `xa_v_i` (1,256,1,128) fp32
/// - State:   12 × `sa_k_i` (1,600,12,64) fp16 — read+written in-place each call
///            12 × `sa_v_i` (1,600,12,64) fp16 — read+written in-place each call
/// - Outputs: `logits` (1,1,8,2024) fp32, `h_last` (1,1,768) fp32
public final class MagpieCoreMLStatefulDecoderStep {
    private let model: MLModel
    public let numLayers: Int
    public let saStateShape: [NSNumber]

    /// Per-call state. Allocated once at init and reused across all AR
    /// steps within a synthesis. The orchestrator should call
    /// ``resetState(seedFromPrefillSaK:saV:)`` at the start of each
    /// utterance to populate it from the prefill outputs.
    private let state: MLState

    public init(url: URL,
                numLayers: Int = MagpieCoreMLConstants.numDecoderLayers) throws {
        self.model = try MagpieCoreMLBridge.loadCompiled(at: url, label: "decoder_step_stateful", kind: .decoder)
        self.numLayers = numLayers
        self.saStateShape = [
            1,
            NSNumber(value: MagpieCoreMLConstants.saCacheLength),
            NSNumber(value: MagpieCoreMLConstants.saSelfHeads),
            NSNumber(value: MagpieCoreMLConstants.saHeadDim),
        ]
        // Stateful models (iOS 18+, macOS 15+). Swift's refined name
        // for ObjC `-[MLModel newState]` is unavailable in the SDK
        // overlay (probably because `new`-prefixed methods collide
        // with Swift conventions). Fall back to dynamic dispatch.
        //
        // CRITICAL: `newState` is a `new`-prefix ObjC method, which per
        // ARC rules returns a *retained* object. We must call
        // `takeRetainedValue()`; using `takeUnretainedValue()` causes a
        // double-release crash on first state access.
        guard let s = model.perform(NSSelectorFromString("newState"))?
                .takeRetainedValue() as? MLState
        else {
            throw MagpieCoreMLError.modelLoadFailed(
                name: "decoder_step_stateful",
                underlying: "newState selector returned nil — model is not stateful")
        }
        self.state = s
    }

    /// Seed the KV state from the prefill model's fp32 outputs. Each
    /// prefill output is a 111-frame slice (110 baked speaker context +
    /// 1 BOS) that we copy into the fp16 state buffer's first 111 rows.
    /// Remaining 489 rows stay zero — the causal mask + position scalar
    /// gate which slots are read.
    public func resetState(prefillSaK: [MLMultiArray], prefillSaV: [MLMultiArray]) throws {
        precondition(prefillSaK.count == numLayers && prefillSaV.count == numLayers)
        for layer in 0..<numLayers {
            // CoreML's withMultiArray closure is NS_NOESCAPE — the
            // MLMultiArray reference is only valid inside the closure
            // and the state's underlying buffer is mapped read/write
            // while the closure runs.
            state.withMultiArray(for: "sa_k_\(layer)") { stateArr in
                try? seedStateBuffer(stateArr, fromFp32Prefill: prefillSaK[layer])
            }
            state.withMultiArray(for: "sa_v_\(layer)") { stateArr in
                try? seedStateBuffer(stateArr, fromFp32Prefill: prefillSaV[layer])
            }
        }
    }

    /// One AR step. Returns (logits, h_last) — the cache is updated
    /// in-place inside the state and isn't surfaced as a return value.
    public func step(audioEmbedding: [Float], position: Int32,
                      encoderOutput: MLMultiArray, encoderMask: MLMultiArray,
                      xaK: [MLMultiArray], xaV: [MLMultiArray]) throws
        -> (logitsFlat: [Float], hLast: MLMultiArray)
    {
        let D = MagpieCoreMLConstants.dModel
        precondition(audioEmbedding.count == D)
        precondition(xaK.count == numLayers && xaV.count == numLayers)

        let audioArr = try MagpieCoreMLBridge.makeFp32(
            audioEmbedding, shape: [1, 1, NSNumber(value: D)],
            label: "decoder_step_stateful/audio_emb")
        let positionArr = try MagpieCoreMLBridge.makeScalarInt32(
            position, label: "decoder_step_stateful/position")

        var dict: [String: MLFeatureValue] = [
            "audio_emb":      MLFeatureValue(multiArray: audioArr),
            "encoder_mask":   MLFeatureValue(multiArray: encoderMask),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "position":       MLFeatureValue(multiArray: positionArr),
        ]
        for layer in 0..<numLayers {
            dict["xa_k_\(layer)"] = MLFeatureValue(multiArray: xaK[layer])
            dict["xa_v_\(layer)"] = MLFeatureValue(multiArray: xaV[layer])
        }
        let features = try MLDictionaryFeatureProvider(dictionary: dict)
        let pred: MLFeatureProvider
        do {
            pred = try model.prediction(from: features, using: state)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_step_stateful", underlying: String(describing: error))
        }
        guard let logits = pred.featureValue(for: "logits")?.multiArrayValue,
              let hLast  = pred.featureValue(for: "h_last")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_step_stateful",
                underlying: "missing logits or h_last")
        }
        return (MagpieCoreMLBridge.toFloat32(logits), hLast)
    }

    /// Copy a fp32 (1, 111, 12, 64) prefill output into the fp16 state
    /// buffer's (1, 600, 12, 64) memory. First 111 frames filled, rest
    /// stays zero (or whatever was there).
    private func seedStateBuffer(_ stateArr: MLMultiArray, fromFp32Prefill src: MLMultiArray) throws {
        let srcShape = src.shape.map { $0.intValue }
        let stateShape = stateArr.shape.map { $0.intValue }
        let perTime = 12 * 64
        let prefillFrames = MagpieCoreMLConstants.speakerContextLength + 1  // 110 + 1 BOS = 111

        // The orchestrator runs prefill outputs through
        // ``MagpieCoreMLDecoder.padPrefillSaToStepShape`` which already
        // pads from (1, 111, 12, 64) up to (1, 600, 12, 64). Accept
        // either shape; both contain valid prefill data in the first
        // 111 time steps with zeros after.
        let srcValidT: Int
        if srcShape == [1, prefillFrames, 12, 64] {
            srcValidT = prefillFrames
        } else if srcShape == [1, MagpieCoreMLConstants.saCacheLength, 12, 64] {
            srcValidT = prefillFrames  // only the first 111 are meaningful
        } else {
            throw MagpieCoreMLError.invalidConstants(
                "stateful seed source has shape \(srcShape); "
                + "expected [1,\(prefillFrames),12,64] or [1,600,12,64]")
        }
        precondition(stateShape == [1, MagpieCoreMLConstants.saCacheLength, 12, 64],
                     "state shape \(stateShape), expected [1,600,12,64]")
        precondition(stateArr.dataType == .float16,
                     "state dtype \(stateArr.dataType.rawValue), expected fp16")

        let srcFloats = MagpieCoreMLBridge.toFloat32(src)
        let dstPtr = stateArr.dataPointer.bindMemory(
            to: UInt16.self, capacity: MagpieCoreMLConstants.saCacheLength * perTime)
        for i in 0..<(MagpieCoreMLConstants.saCacheLength * perTime) { dstPtr[i] = 0 }
        for t in 0..<srcValidT {
            for i in 0..<perTime {
                dstPtr[t * perTime + i] = float32ToFloat16Bits(srcFloats[t * perTime + i])
            }
        }
    }
}


/// Float32 → IEEE-754 binary16. Standalone (no Accelerate) so this file
/// stays minimal-dep.
@inline(__always)
private func float32ToFloat16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    var exp = Int32((bits >> 23) & 0xFF) - 127 + 15
    var mant = bits & 0x7F_FFFF
    if exp <= 0 {
        if exp < -10 { return sign }
        mant |= 0x80_0000
        let shift = 14 - exp
        return sign | UInt16((mant >> shift) & 0x3FF)
    } else if exp >= 31 {
        if mant != 0 { return sign | 0x7E00 }
        return sign | 0x7C00
    } else {
        let expBits = UInt16(exp & 0x1F) << 10
        let mantBits = UInt16((mant >> 13) & 0x3FF)
        return sign | expBits | mantBits
    }
}
