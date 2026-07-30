import XCTest
import Foundation
import MLX
import MLXRandom
import MLXCommon
@testable import Qwen3Chat

/// Geometry of Gemma 4's blocked attention, with no model involved.
///
/// `Gemma4Model.attentionBlocks` decides which keys a block of queries may reach; everything about
/// the speedup rests on that being exactly the set the full `[1,1,Tq,Tk]` mask would have kept.
///
/// Each case is checked twice. Against the full mask it replaces, where the two must agree to within
/// the difference between MLX kernels — a one-query block dispatches to the vector kernel while a
/// 512-query block dispatches to the full one, and those disagree in the last float32 digits for
/// reasons unrelated to masking. And against attention written out with plain precise ops, which
/// pins both paths to the same distance from explicit arithmetic rather than only to each other.
///
/// `testWrongBandIsCaught` is the other half of the argument: it shifts the window by one key and
/// shows both bounds reject it by orders of magnitude, so the tolerances are loose enough for
/// arithmetic and tight enough for the defect they guard.
final class Gemma4AttentionBlockTests: XCTestCase {
    private static let window = 512

    /// Blocked against the full mask. Observed worst case is 4.4e-4, where a trailing one-query
    /// block lands on a different kernel from the reference's single call.
    private static let kernelTolerance: Float = 2e-3

    /// Either fused path against explicit precise arithmetic. MLX's attention kernels are further
    /// from a written-out softmax (~1e-3 to 4e-3 here) than they are from each other, so this bound
    /// is the looser of the two and says nothing about blocking on its own — it is here to catch a
    /// band that is wrong in both paths at once, which comparing them to each other cannot.
    private static let exactTolerance: Float = 1e-2

    /// Attention over the whole key axis behind one full mask — the implementation being replaced.
    private func reference(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float,
                           queryLen: Int, keyLen: Int, offset: Int, window: Int?) -> MLXArray {
        SDPA.attendAndMerge(
            qHeads: q, kHeads: k, vHeads: v, scale: scale,
            mask: Gemma4Model.makeMask(queryLen: queryLen, keyLen: keyLen, offset: offset,
                                       windowSize: window, dtype: .float32))
    }

    /// Attention block by block, each against only the keys its queries can reach.
    private func blocked(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float,
                         queryLen: Int, keyLen: Int, offset: Int, window: Int?) -> MLXArray {
        let blocks = Gemma4Model.attentionBlocks(queryLen: queryLen, keyLen: keyLen, offset: offset,
                                                 windowSize: window, dtype: .float32)
        var parts: [MLXArray] = []
        for block in blocks {
            parts.append(SDPA.attendAndMerge(
                qHeads: q[0..., 0..., block.queryStart ..< block.queryEnd, 0...],
                kHeads: k[0..., 0..., block.keyStart ..< block.keyEnd, 0...],
                vHeads: v[0..., 0..., block.keyStart ..< block.keyEnd, 0...],
                scale: scale, mask: block.mask))
        }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 1)
    }

    /// Attention written out: score, mask, precise softmax, weight — no fused kernel, no blocking.
    /// The grouped kv heads are expanded by hand into the block layout MLX's own fallback uses (kv
    /// head `j` serves query heads `j*repeats ..< (j+1)*repeats`) so nothing here rests on how a
    /// kernel chooses to repeat them.
    private func exact(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, scale: Float,
                       queryLen: Int, keyLen: Int, offset: Int, window: Int?) -> MLXArray {
        let heads = q.dim(1), kvHeads = k.dim(1), headDim = q.dim(3)
        let repeats = heads / kvHeads
        func expand(_ kv: MLXArray) -> MLXArray {
            broadcast(kv.expandedDimensions(axis: 2), to: [1, kvHeads, repeats, keyLen, headDim])
                .reshaped(1, heads, keyLen, headDim)
        }
        let mask = Gemma4Model.makeMask(queryLen: queryLen, keyLen: keyLen, offset: offset,
                                        windowSize: window, dtype: .float32)
        let scores = matmul(q * scale, expand(k).transposed(0, 1, 3, 2)) + mask
        return SDPA.mergeHeads(matmul(softmax(scores, axis: -1, precise: true), expand(v)))
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        let d = MLX.abs(a - b).max()
        eval(d)
        return d.item(Float.self)
    }

    /// Runs all three implementations. `plannedWindow` is what the blocked path is told the window is
    /// — normally `window`, and deliberately wrong only in `testWrongBandIsCaught`.
    @discardableResult
    private func compare(queryLen: Int, keyLen: Int, offset: Int, window: Int?,
                         plannedWindow: Int?? = nil,
                         heads: Int = 8, kvHeads: Int = 2, headDim: Int = 64)
        -> (blockedVsReference: Float, blockedVsExact: Float, referenceVsExact: Float) {
        MLXRandom.seed(7)
        let q = MLXRandom.normal([1, heads, queryLen, headDim]).asType(.float32)
        let k = MLXRandom.normal([1, kvHeads, keyLen, headDim]).asType(.float32)
        let v = MLXRandom.normal([1, kvHeads, keyLen, headDim]).asType(.float32)
        // 1/sqrt(d) rather than Gemma 4's own scale of 1: with unnormalised N(0,1) inputs a scale of
        // 1 drives scores to a standard deviation of 8, and a softmax that peaked collapses onto a
        // single key, so every path returns nearly the same row whatever it attended over. A tame
        // softmax spreads the weight and makes a wrong band visible.
        let scale = 1.0 / Float(headDim).squareRoot()

        let truth = exact(q, k, v, scale: scale, queryLen: queryLen, keyLen: keyLen,
                          offset: offset, window: window)
        let ref = reference(q, k, v, scale: scale, queryLen: queryLen, keyLen: keyLen,
                            offset: offset, window: window)
        let band = blocked(q, k, v, scale: scale, queryLen: queryLen, keyLen: keyLen,
                           offset: offset, window: plannedWindow ?? window)
        return (maxAbsDiff(band, ref), maxAbsDiff(band, truth), maxAbsDiff(ref, truth))
    }

    private func check(queryLen: Int, keyLen: Int, offset: Int, window: Int?,
                      heads: Int = 8, kvHeads: Int = 2, headDim: Int = 64) {
        let d = compare(queryLen: queryLen, keyLen: keyLen, offset: offset, window: window,
                        heads: heads, kvHeads: kvHeads, headDim: headDim)
        let at = "q=\(queryLen) k=\(keyLen) offset=\(offset) "
            + "window=\(window.map(String.init) ?? "none")"
        XCTAssertLessThan(d.blockedVsReference, Self.kernelTolerance,
                          "blocked diverged from the full mask: \(at) diff=\(d.blockedVsReference)")
        XCTAssertLessThan(d.blockedVsExact, Self.exactTolerance,
                          "blocked diverged from explicit attention: \(at) "
                          + "diff=\(d.blockedVsExact) (full mask: \(d.referenceVsExact))")
    }

    /// Prefill at lengths that straddle the window: under it, exactly on it, one past it (the first
    /// genuinely banded block), and multiples over it (interior blocks that share one reused mask,
    /// plus a trailing block that needs its own).
    func testSlidingPrefillLengths() {
        for n in [1, 8, 100, 511, 512, 513, 1024, 1600, 2048, 4000] {
            check(queryLen: n, keyLen: n, offset: 0, window: Self.window)
        }
    }

    /// Global layers are unbanded but still blocked, and every one of their blocks is end-aligned
    /// causal — so this also checks that handing MLX `.causal` reproduces the explicit mask.
    func testGlobalPrefillLengths() {
        for n in [1, 100, 512, 513, 1600, 4000] {
            check(queryLen: n, keyLen: n, offset: 0, window: nil)
        }
    }

    /// Decode steps: one query against a cache far longer than the window. The sliding case reads
    /// the newest `window` entries with no mask at all, which must equal masking the whole cache.
    func testIncrementalDecodeAgainstLongCache() {
        for offset in [0, 1, 511, 512, 900, 2048] {
            check(queryLen: 1, keyLen: offset + 1, offset: offset, window: Self.window)
            check(queryLen: 1, keyLen: offset + 1, offset: offset, window: nil)
        }
    }

    /// A second prefill chunk appended to an existing cache — queries start mid-sequence, so the
    /// block bounds depend on the offset rather than on the local index.
    func testChunkedPrefillWithOffset() {
        for (offset, queries) in [(300, 200), (512, 600), (1000, 1024), (2048, 700)] {
            check(queryLen: queries, keyLen: offset + queries, offset: offset, window: Self.window)
            check(queryLen: queries, keyLen: offset + queries, offset: offset, window: nil)
        }
    }

    /// A window narrower than the query block, so the band is bounded inside the very first block.
    func testWindowNarrowerThanQueryBlock() {
        for n in [64, 300, 1024] {
            check(queryLen: n, keyLen: n, offset: 0, window: 128)
        }
    }

    /// The tolerances above are only worth something if they reject a band that reaches the wrong
    /// keys. One key too many or too few moves the output by ~9e-2 — 47× the kernel tolerance and 9×
    /// the looser one — so both bounds catch it with room to spare rather than by a whisker.
    func testWrongBandIsCaught() {
        let floor: Float = 4e-2
        for wrong in [Self.window - 1, Self.window + 1] {
            let d = compare(queryLen: 1600, keyLen: 1600, offset: 0, window: Self.window,
                            plannedWindow: .some(wrong))
            XCTAssertGreaterThan(d.blockedVsReference, floor,
                                 "window \(wrong) went unnoticed against the full mask")
            XCTAssertGreaterThan(d.blockedVsExact, floor,
                                 "window \(wrong) went unnoticed against explicit attention")
        }
    }

    /// Every block must cover its queries exactly once and reach no key outside the causal window.
    func testBlocksTileQueriesAndRespectBounds() {
        let window = Self.window
        for (offset, tq) in [(0, 4000), (0, 512), (0, 1), (777, 900), (2048, 1)] {
            let tk = offset + tq
            let blocks = Gemma4Model.attentionBlocks(queryLen: tq, keyLen: tk, offset: offset,
                                                     windowSize: window, dtype: .float32)
            XCTAssertEqual(blocks.first?.queryStart, 0)
            XCTAssertEqual(blocks.last?.queryEnd, tq)
            for (i, block) in blocks.enumerated() {
                if i > 0 { XCTAssertEqual(block.queryStart, blocks[i - 1].queryEnd, "blocks must tile") }
                XCTAssertLessThan(block.queryStart, block.queryEnd)
                // Causal upper bound and window lower bound, in absolute positions.
                XCTAssertEqual(block.keyEnd, min(offset + block.queryEnd, tk))
                XCTAssertEqual(block.keyStart, max(0, offset + block.queryStart - window + 1))
            }
        }
    }
}
