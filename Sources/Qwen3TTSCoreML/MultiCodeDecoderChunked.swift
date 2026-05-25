#if canImport(CoreML)
import CoreML
import Foundation

/// Common interface so the synthesise loop can hold either the monolithic
/// (legacy ``MultiCodeDecoderCoreML``) or the chunked ANE form behind a
/// single field. ``Qwen3TTSCoreMLModel`` picks at load time based on which
/// .mlmodelc files exist in the bundle.
protocol MultiCodeDecoderInterface: AnyObject {
    func predict(
        hiddenState: MLMultiArray,
        cb0Token: Int32,
        codeEmbedder: CodeEmbedderModel,
        multiCodeEmbedder: MultiCodeEmbedderModel,
        temperature: Float,
        topK: Int
    ) throws -> [Int32]
}

extension MultiCodeDecoderCoreML: MultiCodeDecoderInterface {}

/// Chunked ANE-friendly MultiCodeDecoder.
///
/// Background: the monolithic 5-layer MCD won't compile for ANE (op-count
/// threshold; CoreML compiler silently falls back to CPU). Splitting into
/// 2 chunks of ≤4 transformer blocks each, plus a separate head model,
/// lets every sub-model land on ANE.
///
/// Layout in the bundle:
///   - ``MultiCodeDecoder_chunk{i}of{N}.mlmodelc`` — N stateless block chunks
///     Inputs:  ``hidden_in`` [1,1024,1,1] + cache_length + key_padding_mask
///              + kv_cache_update_mask + ``key_cache``/``value_cache``
///              [1, chunk_layers × kv_dim, 1, MCD_SEQ_LEN]
///     Outputs: ``hidden_out`` + ``new_k_slots`` + ``new_v_slots``
///              [1, chunk_layers × kv_dim, 1, 1] (just the new entry — Swift
///              writes it into position `cache_length` of the cache buffer
///              between calls. This keeps the model's output tensors small
///              enough for ANE to accept the graph.)
///
///   - ``MultiCodeDecoder_head.mlmodelc`` — final RMSNorm + 15 codec heads.
///     Input:  ``hidden_in`` [1,1024,1,1]
///     Output: ``all_logits`` [1, 15, 2048]
///
/// One ``predict()`` call (a single position in the 16-step autoregressive
/// loop) chains all N+1 sub-models. Per-chunk KV caches are kept in this
/// instance as plain MLMultiArrays — one (key, value) pair per chunk, sized
/// to that chunk's layer count.
final class MultiCodeDecoderChunked: MultiCodeDecoderInterface {
    private let chunks: [MLModel]
    private let head: MLModel
    /// kv_dim per layer (== num_kv_heads × head_dim). Same for all chunks.
    private let layerKvDim: Int
    /// Number of transformer layers in each chunk (read from input shape).
    private let chunkLayerCounts: [Int]
    private let maxSeqLen = 16
    private let hiddenSize = 1024
    private let numGroups = 15

    init(chunks: [MLModel], head: MLModel) {
        precondition(!chunks.isEmpty, "chunked MCD needs at least one chunk")
        self.chunks = chunks
        self.head = head

        // Discover per-chunk layer count from each chunk's key_cache input.
        // key_cache shape is [1, layers × kv_dim, 1, seq_len]. We don't know
        // (layers, kv_dim) separately from one chunk alone, but layer count
        // == chunk_channels / per_layer_kv_dim. We deduce per_layer_kv_dim
        // from the smallest chunk's channels divided by its layer count —
        // since we can't read both, we use the convention that the bundle
        // also exposes ``new_k_slots`` of shape [1, layers × kv_dim, 1, 1].
        // Per-layer kv_dim is therefore (slots_channels / layer_count).
        // Easier path: assume the standard Qwen3-TTS MCD layout — 8 KV heads
        // × 128 head_dim = 1024 per layer. Hard-coded matches our convert
        // script (anything else would need a re-export anyway).
        self.layerKvDim = 1024
        self.chunkLayerCounts = chunks.map { chunk in
            let desc = chunk.modelDescription.inputDescriptionsByName["key_cache"]!
            let totalC = desc.multiArrayConstraint!.shape[1].intValue
            return totalC / 1024
        }
    }

    func predict(
        hiddenState: MLMultiArray,
        cb0Token: Int32,
        codeEmbedder: CodeEmbedderModel,
        multiCodeEmbedder: MultiCodeEmbedderModel,
        temperature: Float = 0.9,
        topK: Int = 50
    ) throws -> [Int32] {
        // Per-chunk KV cache buffers, fresh per predict() (cache resets
        // cleanly between codec frames — same as the monolithic path).
        var keyCaches: [MLMultiArray] = []
        var valueCaches: [MLMultiArray] = []
        for layers in chunkLayerCounts {
            let totalC = layers * layerKvDim
            keyCaches.append(makeZeros(shape: [1, totalC, 1, maxSeqLen]))
            valueCaches.append(makeZeros(shape: [1, totalC, 1, maxSeqLen]))
        }

        // Position 0: feed hidden_states from CodeDecoder (no logits read)
        var hidden = ensureNCHW(hiddenState, channels: hiddenSize)
        hidden = try stepChunks(hidden: hidden, position: 0,
                                keyCaches: &keyCaches, valueCaches: &valueCaches)

        // Position 1: feed CodeEmbedder(CB0) → head → CB1 logits
        let cb0Embed = ensureNCHW(try codeEmbedder.embed(Int(cb0Token)), channels: hiddenSize)
        hidden = try stepChunks(hidden: cb0Embed, position: 1,
                                keyCaches: &keyCaches, valueCaches: &valueCaches)
        var logits = try runHead(hidden: hidden)
        var prevToken = TTSSampler.sample(
            logits: extractGroupLogits(logits, group: 0),
            temperature: temperature, topK: topK)
        var tokens = [prevToken]

        // Positions 2..15: autoregressive CB2..CB15
        for cbStep in 1..<numGroups {
            let embed = ensureNCHW(
                try multiCodeEmbedder.embed(codebookIdx: cbStep - 1, tokenId: Int(prevToken)),
                channels: hiddenSize)
            hidden = try stepChunks(hidden: embed, position: cbStep + 1,
                                    keyCaches: &keyCaches, valueCaches: &valueCaches)
            logits = try runHead(hidden: hidden)
            prevToken = TTSSampler.sample(
                logits: extractGroupLogits(logits, group: cbStep),
                temperature: temperature, topK: topK)
            tokens.append(prevToken)
        }
        return tokens
    }

    // MARK: - Sub-model calls

    /// Run all chunks for a single position; threads hidden state through and
    /// scatters the per-chunk new K/V slots into each chunk's cache at `position`.
    private func stepChunks(hidden: MLMultiArray, position: Int,
                            keyCaches: inout [MLMultiArray],
                            valueCaches: inout [MLMultiArray]) throws -> MLMultiArray {
        let (keyMask, updateMask, cacheLen) = try makeMasks(position: position)
        var h = hidden
        for i in 0..<chunks.count {
            let result = try runChunk(chunkIdx: i, hidden: h, cacheLen: cacheLen,
                                      keyMask: keyMask, updateMask: updateMask,
                                      kc: keyCaches[i], vc: valueCaches[i])
            h = result.hiddenOut
            scatterWrite(into: keyCaches[i], slots: result.newKSlots, position: position)
            scatterWrite(into: valueCaches[i], slots: result.newVSlots, position: position)
        }
        return h
    }

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

    private func runHead(hidden: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "hidden_in": MLFeatureValue(multiArray: hidden),
        ])
        let result = try head.prediction(from: provider)
        return result.featureValue(for: "all_logits")!.multiArrayValue!
    }

    // MARK: - Helpers

    /// Build the (key_mask, update_mask, cache_length) tuple shared across
    /// all chunks for a single position. Allocated once per position to
    /// avoid 3N tiny allocations across the per-position chunk loop.
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

    /// Scatter-write slots `[1, C, 1, 1]` into cache `[1, C, 1, S]` at index `position`
    /// along the trailing axis. Both are contiguous fp16 NCHW; the cache
    /// stride along the S axis is 1, so we copy C interleaved values.
    private func scatterWrite(into cache: MLMultiArray, slots: MLMultiArray, position: Int) {
        let channels = cache.shape[1].intValue
        precondition(slots.shape[1].intValue == channels, "slot channel mismatch")
        let cachePtr = cache.dataPointer.assumingMemoryBound(to: Float16.self)
        let slotPtr = slots.dataPointer.assumingMemoryBound(to: Float16.self)
        // Memory layout for NCHW with H=1 and S in W:
        //   cache[c, s] is at offset c * S + s
        //   slot[c]     is at offset c
        for c in 0..<channels {
            cachePtr[c * maxSeqLen + position] = slotPtr[c]
        }
    }

    /// Extract logits for a specific codebook group from [1, 15, 2048].
    private func extractGroupLogits(_ array: MLMultiArray, group: Int) -> [Float] {
        let vocabSize = 2048
        var result = [Float](repeating: 0, count: vocabSize)
        let ndim = array.shape.count
        for i in 0..<vocabSize {
            var idx = [NSNumber](repeating: 0, count: ndim)
            if ndim == 3 { idx[1] = group as NSNumber; idx[2] = i as NSNumber }
            else if ndim == 2 { idx[0] = group as NSNumber; idx[1] = i as NSNumber }
            result[i] = array[idx].floatValue
        }
        return result
    }

    private func makeZeros(shape: [Int]) -> MLMultiArray {
        let a = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        memset(a.dataPointer, 0, shape.reduce(1, *) * 2)
        return a
    }
}
#endif
