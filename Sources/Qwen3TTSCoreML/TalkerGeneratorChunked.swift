#if canImport(CoreML)
import CoreML
import Foundation

/// Chunked ANE-friendly CodeDecoder runner.
///
/// Mirrors ``TalkerGenerator`` but executes the decoder as N stateless
/// transformer chunks + 1 head model — the same chunking pattern that
/// lets the MultiCodeDecoder fit on ANE (see ``MultiCodeDecoderChunked``
/// for the design rationale).
///
/// Bundle layout (auto-detected by ``Qwen3TTSCoreMLModel.fromPretrained``):
///   - ``CodeDecoder_chunk{i}of{N}.mlmodelc`` — N stateless ≤4-layer chunks.
///     Inputs: ``hidden_in`` + cache_length + key_padding_mask + kv_cache_update_mask
///             + ``key_cache``/``value_cache`` [1, chunk_layers × kv_dim, 1, MAX_SEQ_LEN]
///     Outputs: ``hidden_out`` + ``new_k_slots`` + ``new_v_slots`` [1, chunk_layers × kv_dim, 1, 1]
///   - ``CodeDecoder_head.mlmodelc`` — final RMSNorm + codec_head Linear.
///     Inputs:  ``hidden_in`` [1, 1024, 1, 1]
///     Outputs: ``logits`` [1, 1, vocab] + ``hidden_states`` [1, 1024, 1, 1]
///              (pre-norm hidden, fed to MCD downstream)
final class TalkerGeneratorChunked {
    private let chunks: [MLModel]
    private let head: MLModel
    private let chunkLayerCounts: [Int]
    private let layerKvDim: Int    // per-layer kv_dim (== num_kv_heads × head_dim)
    private let maxSeqLen: Int
    private let hiddenSize: Int
    private let vocabSize: Int

    private var keyCaches: [MLMultiArray] = []
    private var valueCaches: [MLMultiArray] = []
    private var currentPos: Int = 0

    /// Last hidden state from the most recent forward pass — published so
    /// the synthesise loop can feed it into the MultiCodeDecoder.
    private(set) var lastHiddenState: MLMultiArray?

    init(chunks: [MLModel], head: MLModel,
         maxSeqLen: Int = 256, hiddenSize: Int = 1024,
         layerKvDim: Int = 1024) {
        precondition(!chunks.isEmpty)
        self.chunks = chunks
        self.head = head
        self.maxSeqLen = maxSeqLen
        self.hiddenSize = hiddenSize
        self.layerKvDim = layerKvDim
        // Read vocab from head's logits output shape.
        if let logitsDesc = head.modelDescription.outputDescriptionsByName["logits"],
           let constraint = logitsDesc.multiArrayConstraint {
            self.vocabSize = constraint.shape.last!.intValue
        } else {
            self.vocabSize = 3072
        }
        self.chunkLayerCounts = chunks.map { chunk in
            let desc = chunk.modelDescription.inputDescriptionsByName["key_cache"]!
            let totalC = desc.multiArrayConstraint!.shape[1].intValue
            return totalC / layerKvDim
        }
        resetCache()
    }

    func resetCache() {
        keyCaches.removeAll(keepingCapacity: true)
        valueCaches.removeAll(keepingCapacity: true)
        for layers in chunkLayerCounts {
            let totalC = layers * layerKvDim
            keyCaches.append(makeZeros(shape: [1, totalC, 1, maxSeqLen]))
            valueCaches.append(makeZeros(shape: [1, totalC, 1, maxSeqLen]))
        }
        currentPos = 0
    }

    func forward(embedArray: MLMultiArray) throws -> (logits: [Float], hidden: [Float16]) {
        return try forwardInternal(inputEmbeds: ensureNCHW(embedArray, channels: hiddenSize))
    }

    func forward(embed: [Float16]) throws -> (logits: [Float], hidden: [Float16]) {
        let inputEmbeds = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize), 1, 1], dataType: .float16)
        let embPtr = inputEmbeds.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<hiddenSize { embPtr[i] = embed[i] }
        return try forwardInternal(inputEmbeds: inputEmbeds)
    }

    private func forwardInternal(inputEmbeds: MLMultiArray) throws -> (logits: [Float], hidden: [Float16]) {
        guard currentPos < maxSeqLen else { throw TalkerError.cacheFull }

        let (keyMask, updateMask, cacheLen) = try makeMasks(position: currentPos)
        var h = inputEmbeds
        for i in 0..<chunks.count {
            let result = try runChunk(chunkIdx: i, hidden: h, cacheLen: cacheLen,
                                      keyMask: keyMask, updateMask: updateMask,
                                      kc: keyCaches[i], vc: valueCaches[i])
            h = result.hiddenOut
            scatterWrite(into: keyCaches[i], slots: result.newKSlots, position: currentPos)
            scatterWrite(into: valueCaches[i], slots: result.newVSlots, position: currentPos)
        }
        // Head produces logits + pre-norm hidden_states
        let headProv = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_in": MLFeatureValue(multiArray: h),
        ])
        let headOut = try head.prediction(from: headProv)
        let logitsArray = headOut.featureValue(for: "logits")!.multiArrayValue!
        let hiddenArray = headOut.featureValue(for: "hidden_states")!.multiArrayValue!

        currentPos += 1
        lastHiddenState = ensureNCHW(hiddenArray, channels: hiddenSize)

        // Extract logits (1, 1, vocab) → [Float] of vocab
        var logits = [Float](repeating: 0, count: vocabSize)
        let ndim = logitsArray.shape.count
        for i in 0..<vocabSize {
            var idx = [NSNumber](repeating: 0, count: ndim)
            idx[ndim - 1] = i as NSNumber
            logits[i] = logitsArray[idx].floatValue
        }
        var hidden = [Float16](repeating: 0, count: hiddenSize)
        let hndim = hiddenArray.shape.count
        for i in 0..<hiddenSize {
            var idx = [NSNumber](repeating: 0, count: hndim)
            if hndim >= 4 { idx[1] = i as NSNumber } else { idx[0] = i as NSNumber }
            hidden[i] = Float16(hiddenArray[idx].floatValue)
        }
        return (logits, hidden)
    }

    func prefill(embeds: [[Float16]]) throws -> (logits: [Float], hidden: [Float16]) {
        var lastLogits = [Float]()
        var lastHidden = [Float16]()
        for embed in embeds {
            (lastLogits, lastHidden) = try forward(embed: embed)
        }
        return (lastLogits, lastHidden)
    }

    enum TalkerError: Error { case cacheFull }

    // MARK: - Helpers

    private func runChunk(chunkIdx: Int, hidden: MLMultiArray, cacheLen: MLMultiArray,
                          keyMask: MLMultiArray, updateMask: MLMultiArray,
                          kc: MLMultiArray, vc: MLMultiArray)
        throws -> (hiddenOut: MLMultiArray, newKSlots: MLMultiArray, newVSlots: MLMultiArray) {
        let inputs: [String: MLFeatureValue] = [
            "hidden_in": MLFeatureValue(multiArray: hidden),
            "cache_length": MLFeatureValue(multiArray: cacheLen),
            "key_padding_mask": MLFeatureValue(multiArray: keyMask),
            "kv_cache_update_mask": MLFeatureValue(multiArray: updateMask),
            "key_cache": MLFeatureValue(multiArray: kc),
            "value_cache": MLFeatureValue(multiArray: vc),
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        let result = try chunks[chunkIdx].prediction(from: provider)
        return (
            result.featureValue(for: "hidden_out")!.multiArrayValue!,
            result.featureValue(for: "new_k_slots")!.multiArrayValue!,
            result.featureValue(for: "new_v_slots")!.multiArrayValue!
        )
    }

    private func makeMasks(position: Int) throws
        -> (keyMask: MLMultiArray, updateMask: MLMultiArray, cacheLen: MLMultiArray) {
        let cacheLen = try MLMultiArray(shape: [1], dataType: .int32)
        cacheLen.dataPointer.assumingMemoryBound(to: Int32.self)[0] = Int32(position)

        let keyMask = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        let kPtr = keyMask.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<maxSeqLen { kPtr[i] = i <= position ? Float16(0) : Float16(-1e4) }

        let updateMask = try MLMultiArray(shape: [1, NSNumber(value: maxSeqLen)], dataType: .float16)
        memset(updateMask.dataPointer, 0, maxSeqLen * 2)
        updateMask.dataPointer.assumingMemoryBound(to: Float16.self)[position] = Float16(1.0)
        return (keyMask, updateMask, cacheLen)
    }

    private func scatterWrite(into cache: MLMultiArray, slots: MLMultiArray, position: Int) {
        let channels = cache.shape[1].intValue
        let cachePtr = cache.dataPointer.assumingMemoryBound(to: Float16.self)
        let slotPtr = slots.dataPointer.assumingMemoryBound(to: Float16.self)
        for c in 0..<channels {
            cachePtr[c * maxSeqLen + position] = slotPtr[c]
        }
    }

    private func makeZeros(shape: [Int]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        memset(a.dataPointer, 0, shape.reduce(1, *) * 2)
        return a
    }
}

// Common interface so synthesize() can hold either flavour in one field.
protocol CodeDecoderInterface: AnyObject {
    var lastHiddenState: MLMultiArray? { get }
    func resetCache()
    func forward(embedArray: MLMultiArray) throws -> (logits: [Float], hidden: [Float16])
    func forward(embed: [Float16]) throws -> (logits: [Float], hidden: [Float16])
}

extension TalkerGenerator: CodeDecoderInterface {}
extension TalkerGeneratorChunked: CodeDecoderInterface {}
#endif
