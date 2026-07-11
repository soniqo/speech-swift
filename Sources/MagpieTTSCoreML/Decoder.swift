import CoreML
import Foundation

/// Wraps `decoder_prefill.mlmodelc` + `decoder_step.mlmodelc`.
///
/// `decoder_prefill` runs once per utterance: takes `speaker_idx`
/// (selects the baked speaker context internally), the encoder output and
/// mask, and emits:
/// - `h_last` (1, 1, 768) — last hidden of the prefix
/// - 12 × `sa_k_*` (1, 111, 12, 64) — self-attention K cache (110 baked + 1 BOS)
/// - 12 × `sa_v_*` (1, 111, 12, 64) — self-attention V cache
/// - 12 × `xa_k_*` (1, 256, 1, 128) — cross-attention K (precomputed)
/// - 12 × `xa_v_*` (1, 256, 1, 128) — cross-attention V (precomputed)
///
/// `decoder_step` advances one frame: consumes `audio_emb`, encoder
/// output/mask, the shared `position` scalar, all 48 KV inputs, returns:
/// - `logits` (1, 1, 8, 2024) — parallel codebook head (used directly for sampling)
/// - `h_last` (1, 1, 768) — decoder hidden
/// - 12 × `sa_k_out_*` (1, 600, 12, 64)
/// - 12 × `sa_v_out_*` (1, 600, 12, 64)
///
/// Cache padding: prefill emits `(1, 111, 12, 64)` self-attention caches;
/// the step input expects `(1, 600, 12, 64)`. We zero-pad at the right
/// edge before the first step and reuse the model's `(1, 600, 12, 64)`
/// outputs verbatim afterwards.
public final class MagpieCoreMLDecoder {

    public struct PrefillOutput {
        public let hLast: MLMultiArray         // (1, 1, 768)
        public let saK: [MLMultiArray]         // 12 × (1, 600, 12, 64) — padded to step shape
        public let saV: [MLMultiArray]         // 12 × (1, 600, 12, 64)
        public let xaK: [MLMultiArray]         // 12 × (1, 256, 1, 128)
        public let xaV: [MLMultiArray]         // 12 × (1, 256, 1, 128)
        /// Initial AR position = 110 (baked frames) + 1 (BOS) = 111.
        public let initialPosition: Int32
    }

    public struct StepOutput {
        /// Flat `[8 * 2024]` Float32 logits — sampler reshapes per codebook.
        public let logitsFlat: [Float]
        public let hLast: MLMultiArray
        public let saK: [MLMultiArray]
        public let saV: [MLMultiArray]
    }

    private let prefill: MLModel
    private let step: MLModel
    private let numLayers: Int

    /// Back-buffers for `MLPredictionOptions.outputBackings`. We pre-
    /// allocate one set of MLMultiArrays matching every `sa_k_out_*` /
    /// `sa_v_out_*` output the decoder_step model writes, and tell
    /// CoreML to write into THOSE buffers instead of allocating fresh
    /// fp32 outputs on every step. For a 12-layer × 600-pos × 12-heads ×
    /// 64-dim × fp32 KV cache, that's 24 × 1.76 MB = ~42 MB of allocation
    /// per step that CoreML would otherwise pay (Magpie has fp32 IO at
    /// the boundaries, unlike FluidInference's fp16-IO bundle). Across a
    /// 200-frame synthesis that's ~8.4 GB of fresh allocator pressure.
    ///
    /// The buffers are ping-ponged: after each prediction we swap the
    /// back-buffers (just-written outputs) with the front-buffers
    /// (current cache state) so the next step's `sa_k_in_*` inputs are
    /// the same MLMultiArrays we just received the outputs into.
    private var saKBack: [MLMultiArray] = []
    private var saVBack: [MLMultiArray] = []

    /// Same idea for `logits` and `h_last` — modest savings but free.
    private var logitsBack: MLMultiArray?
    private var hLastBack: MLMultiArray?

    /// Set to `false` if our model doesn't accept output backings (some
    /// trace configurations don't carry explicit MultiArray shape/dtype
    /// constraints on outputs). The rejection is a static property of
    /// the model, so once we see it we permanently skip the fast path
    /// instead of throwing 500 times per utterance.
    private var useOutputBackings: Bool = true

    /// Internal counters for `MAGPIE_COREML_PROFILE_STEP=1` debug mode.
    /// Splits step() wall time into IO setup (build MLMultiArrays +
    /// FeatureProvider) vs the actual `prediction(from:)` call.
    public static var setupAccum: Double = 0
    public static var predAccum: Double = 0
    public static var extractAccum: Double = 0
    private static var warnedAboutSilentBackingIgnore = false

    public static func resetStepProfile() {
        setupAccum = 0
        predAccum = 0
        extractAccum = 0
    }

    public init(prefillURL: URL, stepURL: URL,
                numLayers: Int = MagpieCoreMLConstants.numDecoderLayers) throws {
        self.prefill = try MagpieCoreMLBridge.loadCompiled(at: prefillURL, label: "decoder_prefill", kind: .decoder)
        self.step = try MagpieCoreMLBridge.loadCompiled(at: stepURL, label: "decoder_step", kind: .decoder)
        self.numLayers = numLayers
        try allocOutputBackings()
    }

    /// Allocate the output back-buffers once at init. Same shapes as the
    /// step model's outputs (fp32 throughout for our bundle).
    private func allocOutputBackings() throws {
        let saShape: [NSNumber] = [
            1,
            NSNumber(value: MagpieCoreMLConstants.saCacheLength),
            NSNumber(value: MagpieCoreMLConstants.saSelfHeads),
            NSNumber(value: MagpieCoreMLConstants.saHeadDim),
        ]
        let D = MagpieCoreMLConstants.dModel
        let K = MagpieCoreMLConstants.numCodebooks
        let V = MagpieCoreMLConstants.numCodesPerCodebook
        for _ in 0..<numLayers {
            saKBack.append(try MLMultiArray(shape: saShape, dataType: .float32))
            saVBack.append(try MLMultiArray(shape: saShape, dataType: .float32))
        }
        logitsBack = try MLMultiArray(
            shape: [1, 1, NSNumber(value: K), NSNumber(value: V)],
            dataType: .float32)
        hLastBack = try MLMultiArray(
            shape: [1, 1, NSNumber(value: D)],
            dataType: .float32)
    }

    /// One-shot ANE warm-up. Run a dummy prediction on each model with
    /// zero-valued inputs so the first real call doesn't pay the
    /// JIT/binding cost (measured ~140 ms/frame for the first 1–3 steps
    /// vs ~16 ms steady-state on M4 Pro). Output is discarded.
    public func prewarm() {
        do { _ = try prewarmPrefill() } catch { /* swallow — not fatal */ }
        do { _ = try prewarmStep() } catch { /* swallow */ }
    }

    private func prewarmPrefill() throws {
        let zerosEnc = [Float](
            repeating: 0,
            count: MagpieCoreMLConstants.maxTextTokens * MagpieCoreMLConstants.dModel)
        let encOut = try MagpieCoreMLBridge.makeFp32(
            zerosEnc,
            shape: [1,
                    NSNumber(value: MagpieCoreMLConstants.maxTextTokens),
                    NSNumber(value: MagpieCoreMLConstants.dModel)],
            label: "prewarm/encoder_output")
        let zerosMask = [Float](repeating: 0, count: MagpieCoreMLConstants.maxTextTokens)
        let mask = try MagpieCoreMLBridge.makeFp32(
            zerosMask,
            shape: [1, NSNumber(value: MagpieCoreMLConstants.maxTextTokens)],
            label: "prewarm/encoder_mask")
        let speaker = try MagpieCoreMLBridge.makeScalarInt32(
            0, label: "prewarm/speaker_idx")
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_mask":   MLFeatureValue(multiArray: mask),
            "encoder_output": MLFeatureValue(multiArray: encOut),
            "speaker_idx":    MLFeatureValue(multiArray: speaker),
        ])
        _ = try prefill.prediction(from: features)
    }

    private func prewarmStep() throws {
        let D = MagpieCoreMLConstants.dModel
        let audioEmb = try MagpieCoreMLBridge.makeFp32(
            [Float](repeating: 0, count: D),
            shape: [1, 1, NSNumber(value: D)],
            label: "prewarm/audio_emb")
        let encOut = try MagpieCoreMLBridge.makeFp32(
            [Float](repeating: 0,
                    count: MagpieCoreMLConstants.maxTextTokens * D),
            shape: [1,
                    NSNumber(value: MagpieCoreMLConstants.maxTextTokens),
                    NSNumber(value: D)],
            label: "prewarm/encoder_output")
        let mask = try MagpieCoreMLBridge.makeFp32(
            [Float](repeating: 0, count: MagpieCoreMLConstants.maxTextTokens),
            shape: [1, NSNumber(value: MagpieCoreMLConstants.maxTextTokens)],
            label: "prewarm/encoder_mask")
        let position = try MagpieCoreMLBridge.makeScalarInt32(
            Int32(MagpieCoreMLConstants.speakerContextLength + 1),
            label: "prewarm/position")
        var dict: [String: MLFeatureValue] = [
            "audio_emb":      MLFeatureValue(multiArray: audioEmb),
            "encoder_mask":   MLFeatureValue(multiArray: mask),
            "encoder_output": MLFeatureValue(multiArray: encOut),
            "position":       MLFeatureValue(multiArray: position),
        ]
        let saShape: [NSNumber] = [
            1,
            NSNumber(value: MagpieCoreMLConstants.saCacheLength),
            NSNumber(value: MagpieCoreMLConstants.saSelfHeads),
            NSNumber(value: MagpieCoreMLConstants.saHeadDim)]
        let saZero = [Float](
            repeating: 0,
            count: MagpieCoreMLConstants.saCacheLength
                 * MagpieCoreMLConstants.saSelfHeads
                 * MagpieCoreMLConstants.saHeadDim)
        let xaShape: [NSNumber] = [
            1,
            NSNumber(value: MagpieCoreMLConstants.xaContextLength),
            1,
            NSNumber(value: MagpieCoreMLConstants.xaInnerDim)]
        let xaZero = [Float](
            repeating: 0,
            count: MagpieCoreMLConstants.xaContextLength
                 * MagpieCoreMLConstants.xaInnerDim)
        for layer in 0..<numLayers {
            let kArr = try MagpieCoreMLBridge.makeFp32(saZero, shape: saShape,
                                                       label: "prewarm/sa_k_\(layer)")
            let vArr = try MagpieCoreMLBridge.makeFp32(saZero, shape: saShape,
                                                       label: "prewarm/sa_v_\(layer)")
            let xkArr = try MagpieCoreMLBridge.makeFp32(xaZero, shape: xaShape,
                                                        label: "prewarm/xa_k_\(layer)")
            let xvArr = try MagpieCoreMLBridge.makeFp32(xaZero, shape: xaShape,
                                                        label: "prewarm/xa_v_\(layer)")
            dict["sa_k_in_\(layer)"] = MLFeatureValue(multiArray: kArr)
            dict["sa_v_in_\(layer)"] = MLFeatureValue(multiArray: vArr)
            dict["xa_k_\(layer)"]    = MLFeatureValue(multiArray: xkArr)
            dict["xa_v_\(layer)"]    = MLFeatureValue(multiArray: xvArr)
        }
        let features = try MLDictionaryFeatureProvider(dictionary: dict)
        _ = try step.prediction(from: features)
    }

    // MARK: - Prefill

    public func prefill(speakerIndex: Int,
                         encoderOutput: MLMultiArray,
                         encoderMask: MLMultiArray) throws -> PrefillOutput {
        let speakerArr = try MagpieCoreMLBridge.makeScalarInt32(
            Int32(speakerIndex), label: "decoder_prefill/speaker_idx")
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_mask":   MLFeatureValue(multiArray: encoderMask),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "speaker_idx":    MLFeatureValue(multiArray: speakerArr),
        ])
        let pred: MLFeatureProvider
        do {
            pred = try prefill.prediction(from: features)
        } catch {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_prefill", underlying: String(describing: error))
        }
        guard let hLast = pred.featureValue(for: "h_last")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_prefill", underlying: "missing h_last")
        }

        var saK: [MLMultiArray] = []
        var saV: [MLMultiArray] = []
        var xaK: [MLMultiArray] = []
        var xaV: [MLMultiArray] = []
        saK.reserveCapacity(numLayers)
        saV.reserveCapacity(numLayers)
        xaK.reserveCapacity(numLayers)
        xaV.reserveCapacity(numLayers)

        for layer in 0..<numLayers {
            guard let saKArr = pred.featureValue(for: "sa_k_\(layer)")?.multiArrayValue,
                  let saVArr = pred.featureValue(for: "sa_v_\(layer)")?.multiArrayValue,
                  let xaKArr = pred.featureValue(for: "xa_k_\(layer)")?.multiArrayValue,
                  let xaVArr = pred.featureValue(for: "xa_v_\(layer)")?.multiArrayValue else {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "missing one of sa_k_\(layer)/sa_v_\(layer)/xa_k_\(layer)/xa_v_\(layer)")
            }
            saK.append(try padPrefillSaToStepShape(saKArr, label: "sa_k_\(layer)"))
            saV.append(try padPrefillSaToStepShape(saVArr, label: "sa_v_\(layer)"))
            xaK.append(xaKArr)
            xaV.append(xaVArr)
        }
        return PrefillOutput(
            hLast: hLast, saK: saK, saV: saV, xaK: xaK, xaV: xaV,
            initialPosition: Int32(MagpieCoreMLConstants.speakerContextLength + 1))
    }

    // MARK: - Step

    public func step(audioEmbedding: [Float], position: Int32,
                      encoderOutput: MLMultiArray, encoderMask: MLMultiArray,
                      saK: [MLMultiArray], saV: [MLMultiArray],
                      xaK: [MLMultiArray], xaV: [MLMultiArray]) throws -> StepOutput {
        let D = MagpieCoreMLConstants.dModel
        precondition(audioEmbedding.count == D)
        precondition(saK.count == numLayers && saV.count == numLayers)
        precondition(xaK.count == numLayers && xaV.count == numLayers)

        let debug = ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE_STEP"] == "1"

        let tSetup = debug ? CFAbsoluteTimeGetCurrent() : 0
        let audioArr = try MagpieCoreMLBridge.makeFp32(
            audioEmbedding, shape: [1, 1, NSNumber(value: D)],
            label: "decoder_step/audio_emb")
        let positionArr = try MagpieCoreMLBridge.makeScalarInt32(
            position, label: "decoder_step/position")

        var dict: [String: MLFeatureValue] = [
            "audio_emb":      MLFeatureValue(multiArray: audioArr),
            "encoder_mask":   MLFeatureValue(multiArray: encoderMask),
            "encoder_output": MLFeatureValue(multiArray: encoderOutput),
            "position":       MLFeatureValue(multiArray: positionArr),
        ]
        for layer in 0..<numLayers {
            dict["sa_k_in_\(layer)"] = MLFeatureValue(multiArray: saK[layer])
            dict["sa_v_in_\(layer)"] = MLFeatureValue(multiArray: saV[layer])
            dict["xa_k_\(layer)"]    = MLFeatureValue(multiArray: xaK[layer])
            dict["xa_v_\(layer)"]    = MLFeatureValue(multiArray: xaV[layer])
        }
        let features = try MLDictionaryFeatureProvider(dictionary: dict)
        if debug { Self.setupAccum += CFAbsoluteTimeGetCurrent() - tSetup }

        let tPred = debug ? CFAbsoluteTimeGetCurrent() : 0
        // Fast path: pre-allocated output backings. CoreML writes directly
        // into our back-buffers, avoiding ~42 MB of fresh fp32 allocation
        // per step. We then ping-pong (swap front/back) so next call's
        // sa_k_in_* inputs are this call's freshly-written outputs.
        var pred: MLFeatureProvider!
        var fastPathSucceeded = false
        if useOutputBackings {
            let predOpts = MLPredictionOptions()
            var backings: [String: Any] = [:]
            for layer in 0..<numLayers {
                backings["sa_k_out_\(layer)"] = saKBack[layer]
                backings["sa_v_out_\(layer)"] = saVBack[layer]
            }
            if let lb = logitsBack { backings["logits"] = lb }
            if let hb = hLastBack  { backings["h_last"] = hb }
            predOpts.outputBackings = backings
            do {
                pred = try step.prediction(from: features, options: predOpts)
                fastPathSucceeded = true
                // Verify the fast path is actually engaged: when CoreML
                // honors outputBackings, the returned output MultiArray
                // dataPointer for, say, h_last should equal our back-buffer
                // pointer. If they differ, CoreML silently allocated fresh
                // outputs (likely because the model wasn't exported with
                // strict output type constraints) — surface that so we
                // know the optimization is dead code, not silently
                // helping.
                if let hBack = hLastBack,
                   let hOut = pred.featureValue(for: "h_last")?.multiArrayValue,
                   hOut.dataPointer != hBack.dataPointer,
                   ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE"] == "1",
                   !Self.warnedAboutSilentBackingIgnore {
                    Self.warnedAboutSilentBackingIgnore = true
                    FileHandle.standardError.write(Data(
                        "[magpie-coreml] outputBackings accepted but CoreML still allocated fresh outputs — model isn't exported with strict output type constraints; the optimization is a no-op for this bundle\n".utf8))
                }
            } catch {
                // Model exported without explicit output shape constraints
                // — latch the flag off so we don't throw 500 times.
                useOutputBackings = false
                if ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE_STEP"] == "1"
                    || ProcessInfo.processInfo.environment["MAGPIE_COREML_PROFILE"] == "1" {
                    FileHandle.standardError.write(Data(
                        "[magpie-coreml] outputBackings rejected: \(error.localizedDescription); falling back to fresh-alloc path\n".utf8))
                }
            }
        }
        if !fastPathSucceeded {
            do {
                pred = try step.prediction(from: features)
            } catch {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_step", underlying: String(describing: error))
            }
        }
        if debug { Self.predAccum += CFAbsoluteTimeGetCurrent() - tPred }

        guard let logits = pred.featureValue(for: "logits")?.multiArrayValue,
              let hLast  = pred.featureValue(for: "h_last")?.multiArrayValue else {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "decoder_step", underlying: "missing logits or h_last output")
        }

        let tExtract = debug ? CFAbsoluteTimeGetCurrent() : 0
        var newK: [MLMultiArray] = []
        var newV: [MLMultiArray] = []
        newK.reserveCapacity(numLayers)
        newV.reserveCapacity(numLayers)
        for layer in 0..<numLayers {
            guard let kArr = pred.featureValue(for: "sa_k_out_\(layer)")?.multiArrayValue,
                  let vArr = pred.featureValue(for: "sa_v_out_\(layer)")?.multiArrayValue else {
                throw MagpieCoreMLError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing sa_k_out_\(layer)/sa_v_out_\(layer)")
            }
            newK.append(kArr)
            newV.append(vArr)
        }
        let logitsFlat = MagpieCoreMLBridge.toFloat32(logits)
        if debug { Self.extractAccum += CFAbsoluteTimeGetCurrent() - tExtract }

        // Ping-pong: now that this step's outputs are in saKBack/saVBack
        // (via outputBackings), the orchestrator will pass those same
        // arrays back as next step's sa_k_in/sa_v_in. We need fresh
        // back-buffers for the next round — swap our internal saKBack
        // pointers with the OLD inputs (the saK/saV we received), which
        // are no longer needed. Cheap pointer swap, no copy.
        if fastPathSucceeded {
            for layer in 0..<numLayers {
                saKBack[layer] = saK[layer]
                saVBack[layer] = saV[layer]
            }
        }

        return StepOutput(
            logitsFlat: logitsFlat,
            hLast: hLast, saK: newK, saV: newV)
    }

    /// `decoder_prefill` emits self-attention caches as
    /// `(1, 111, 12, 64)`; `decoder_step` expects `(1, 600, 12, 64)`.
    /// Allocate a step-shaped buffer, copy the 111 valid positions in,
    /// and leave the rest zeroed. The model's causal mask + the
    /// `position` scalar gate which slots are read.
    private func padPrefillSaToStepShape(_ src: MLMultiArray, label: String) throws
        -> MLMultiArray
    {
        let srcShape = src.shape.map { $0.intValue }
        precondition(srcShape == [1, 111, 12, 64],
                     "\(label): expected prefill sa shape [1,111,12,64], got \(srcShape)")
        let outShape: [NSNumber] = [
            1,
            NSNumber(value: MagpieCoreMLConstants.saCacheLength),
            NSNumber(value: MagpieCoreMLConstants.saSelfHeads),
            NSNumber(value: MagpieCoreMLConstants.saHeadDim)]
        let dst = try MLMultiArray(shape: outShape, dataType: .float32)
        let dstPtr = dst.dataPointer.bindMemory(to: Float.self, capacity: dst.count)
        for i in 0..<dst.count { dstPtr[i] = 0 }

        // src layout: (1, 111, 12, 64) row-major. Each "time step" t is a
        // contiguous (12 * 64 = 768) chunk. dst's stride along T is also
        // (12 * 64). Direct memcpy by time step.
        let stride = MagpieCoreMLConstants.saSelfHeads * MagpieCoreMLConstants.saHeadDim
        let srcFloats = MagpieCoreMLBridge.toFloat32(src)
        for t in 0..<111 {
            let srcOff = t * stride
            let dstOff = t * stride
            for i in 0..<stride { dstPtr[dstOff + i] = srcFloats[srcOff + i] }
        }
        return dst
    }
}
