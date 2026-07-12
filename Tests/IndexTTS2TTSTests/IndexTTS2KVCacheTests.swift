import MLX
import XCTest

@testable import IndexTTS2TTS

/// The batched sampling paths decode against `BatchedGPTKVCache`
/// (preallocated chunks, in-place slice writes); `GPTKVCache` is the
/// concat-per-step reference semantics. Every history the batched cache
/// returns must be identical to the reference's.
final class IndexTTS2KVCacheTests: XCTestCase {
    private let beams = 3
    private let heads = 2
    private let headDim = 4

    private func step(_ t: Int, seed: Float) -> (keys: MLXArray, values: MLXArray) {
        let count = beams * heads * t * headDim
        let keys = MLXArray((0..<count).map { Float($0) * 0.01 + seed })
            .reshaped([beams, heads, t, headDim])
        return (keys, keys + 1000)
    }

    private func assertSame(_ a: MLXArray, _ b: MLXArray, _ label: String) {
        XCTAssertEqual(a.shape, b.shape, label)
        XCTAssertTrue(arrayEqual(a, b).item(Bool.self), label)
    }

    func testBatchedCacheMatchesConcatReferenceAcrossChunkGrowth() {
        let layers = 2
        var reference = IndexTTS2SemanticGPT.GPTKVCache(layerCount: layers)
        let batched = IndexTTS2SemanticGPT.BatchedGPTKVCache(layerCount: layers)

        // The prefill lands just under one 256-token chunk; the single-token
        // steps then fill it exactly and force a growth copy past 256.
        for (index, tokens) in [254, 1, 1, 1, 1].enumerated() {
            for layer in 0..<layers {
                let (k, v) = step(tokens, seed: Float(index * 10 + layer))
                let expected = reference.update(layer: layer, keys: k, values: v)
                let actual = batched.update(layer: layer, keys: k, values: v)
                assertSame(expected.keys, actual.keys, "keys step \(index) layer \(layer)")
                assertSame(expected.values, actual.values, "values step \(index) layer \(layer)")
            }
            reference.commit(tokens: tokens)
            batched.commit(tokens: tokens)
        }
    }

    func testBatchedCachePromotesHalfPrecisionHistoryToFloat32() {
        // Attention over the cached history runs in float32 (the fp16 SDPA
        // kernel degrades under real decode buffer states); the cache must
        // return float32 no matter what dtype the step produces.
        let batched = IndexTTS2SemanticGPT.BatchedGPTKVCache(layerCount: 1)
        let (k, v) = step(2, seed: 3)
        let out = batched.update(layer: 0, keys: k.asType(.float16), values: v.asType(.float16))
        XCTAssertEqual(out.keys.dtype, .float32)
        XCTAssertEqual(out.values.dtype, .float32)
        assertSame(out.keys, k.asType(.float16).asType(.float32), "promoted keys")
        assertSame(out.values, v.asType(.float16).asType(.float32), "promoted values")
    }

    func testReorderRowsMatchesGatherAndIdentityIsNoOp() {
        let batched = IndexTTS2SemanticGPT.BatchedGPTKVCache(layerCount: 1)
        let (k0, v0) = step(3, seed: 0)
        _ = batched.update(layer: 0, keys: k0, values: v0)
        batched.commit(tokens: 3)

        batched.reorderRows([0, 1, 2])
        let parents = [2, 0, 0]
        batched.reorderRows(parents)

        // Reference: gather the rows up front, then append the next step.
        let ids = MLXArray(parents.map(Int32.init))
        var reference = IndexTTS2SemanticGPT.GPTKVCache(layerCount: 1)
        _ = reference.update(layer: 0, keys: take(k0, ids, axis: 0), values: take(v0, ids, axis: 0))
        reference.commit(tokens: 3)

        let (k1, v1) = step(1, seed: 9)
        let expected = reference.update(layer: 0, keys: k1, values: v1)
        let actual = batched.update(layer: 0, keys: k1, values: v1)
        assertSame(expected.keys, actual.keys, "keys after reorder")
        assertSame(expected.values, actual.values, "values after reorder")
    }
}
