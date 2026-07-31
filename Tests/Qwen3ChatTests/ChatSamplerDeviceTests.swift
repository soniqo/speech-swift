import XCTest
import Foundation
import MLX
@testable import Qwen3Chat

/// `ChatSampler.sampleOnDevice` against the host sampler it replaces in the Gemma 4 decode loop.
///
/// The two are separate implementations of one set of rules, so the only thing that keeps them
/// honest is comparing them. `ChatSampler.sample(…uniform:)` exists for this: with the nucleus
/// draw lifted out of the sampler, both paths are pure functions of the same inputs and can be
/// compared token-for-token over a sweep of draws rather than in distribution.
///
/// Where a case has ties the comparison is deliberately weaker, and the reason is stated at each
/// such test: neither path breaks equal probabilities by index — the host sorts candidates with
/// `Array.sort`, the device with `argSort` — so on equal logits they may return different tokens
/// of *equal* probability. Asserting index equality there would be asserting an implementation
/// detail neither path promises. Greedy is exempt: both take the lowest index among equal maxima,
/// and `testGreedyTieTakesLowestIndex` pins that.
final class ChatSamplerDeviceTests: XCTestCase {

    /// Draws swept across `[0, 1)` — enough that every candidate of a few-dozen-wide nucleus is
    /// reached many times, so a disagreement about which ranks survive filtering shows up as a
    /// disagreement about which tokens are reachable.
    private static let sweep = 256

    private func uniforms(_ n: Int = sweep) -> [Float] {
        (0 ..< n).map { (Float($0) + 0.5) / Float(n) }
    }

    private func device(
        _ logits: [Float], _ config: ChatSamplingConfig,
        suppressing: [Int] = [], previousTokens: [Int] = [], vocabSize: Int? = nil,
        uniform: Float = 0
    ) -> Int {
        ChatSampler.sampleOnDevice(
            logits: MLXArray(logits), config: config, suppressing: suppressing,
            previousTokens: previousTokens, vocabSize: vocabSize, uniform: uniform
        ).item(Int.self)
    }

    /// The host decode loop suppressed end tokens into its `[Float]` before sampling; the device
    /// sampler folds that in. This is the host order, written out so the comparisons below are
    /// against the code that was actually replaced.
    private func host(
        _ logits: [Float], _ config: ChatSamplingConfig,
        suppressing: [Int] = [], previousTokens: [Int] = [], vocabSize: Int? = nil,
        uniform: Float = 0
    ) -> Int {
        var l = Array(logits.prefix(vocabSize ?? logits.count))
        for id in suppressing where id >= 0 && id < l.count { l[id] = -Float.greatestFiniteMagnitude }
        return ChatSampler.sample(logits: l, config: config, previousTokens: previousTokens,
                                  uniform: uniform)
    }

    /// Seeded so a failure is reproducible, and spread over roughly ±8 — the range real logits
    /// occupy, where a nucleus holds a few dozen candidates rather than collapsing onto one.
    private func randomLogits(_ n: Int, seed: UInt64) -> [Float] {
        var s = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return (0 ..< n).map { _ in
            s = s &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(Int32(truncatingIfNeeded: s >> 33)) / Float(1 << 28)
        }
    }

    private static let greedy = ChatSamplingConfig(
        temperature: 0, topK: 0, topP: 1.0, maxTokens: 1, repetitionPenalty: 1.0)

    // MARK: - Greedy

    /// The hard gate: with `temperature <= 0` the decision is deterministic, so any difference is
    /// a behavioural change. Random logits are drawn from a range wide enough that exact ties are
    /// vanishingly unlikely, which is the case the decode loop actually runs in.
    func testGreedyMatchesHostOnRandomLogits() {
        for seed in UInt64(1) ... 20 {
            let l = randomLogits(4096, seed: seed)
            XCTAssertEqual(device(l, Self.greedy), host(l, Self.greedy), "seed \(seed)")
        }
    }

    /// The host scans with a strict `>`, so the first of several equal maxima wins. `argMax` must
    /// agree — this is the one tie the two paths do promise to break the same way, and the greedy
    /// parity claim rests on it.
    func testGreedyTieTakesLowestIndex() {
        XCTAssertEqual(device([Float](repeating: 1.5, count: 64), Self.greedy), 0)
        XCTAssertEqual(host([Float](repeating: 1.5, count: 64), Self.greedy), 0)

        var l = [Float](repeating: -3, count: 64)
        for i in [7, 19, 40] { l[i] = 9 }
        XCTAssertEqual(device(l, Self.greedy), 7)
        XCTAssertEqual(host(l, Self.greedy), 7)
    }

    /// Logits wider than the tokenizer's vocabulary are trimmed on both paths — the host did it
    /// with `prefix(vocabSize)`. A padded row's extra columns must not be reachable.
    func testVocabSizeTrimMatchesHostPrefix() {
        var l = randomLogits(300, seed: 99)
        l[280] = 1_000                      // beyond the "vocabulary"
        l[137] = 500                        // the real maximum
        XCTAssertEqual(device(l, Self.greedy, vocabSize: 256), 137)
        XCTAssertEqual(host(l, Self.greedy, vocabSize: 256), 137)
    }

    // MARK: - Repetition penalty

    /// Same window (the last 64 ids), same asymmetric rule (divide when positive, multiply when
    /// not), same tolerance for ids that are duplicated or out of range.
    func testRepetitionPenaltyMatchesHost() {
        let config = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 1,
                                        repetitionPenalty: 1.7)
        for seed in UInt64(1) ... 10 {
            let l = randomLogits(512, seed: seed)
            // Longer than the window, with repeats and ids the sampler must ignore.
            var history = (0 ..< 200).map { Int(($0 &* 37 &+ Int(seed)) % 512) }
            history += [-1, 512, 99_999, history[0], history[0]]
            XCTAssertEqual(device(l, config, previousTokens: history),
                           host(l, config, previousTokens: history), "seed \(seed)")
        }
    }

    /// The rule has to move a negative logit *down*, not up — multiplying is what does that, and
    /// getting the branch backwards is the obvious way to break it.
    func testRepetitionPenaltyPushesBothSignsDown() {
        let config = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 1,
                                        repetitionPenalty: 2.0)
        //         0      1     2
        let l: [Float] = [10.0, 9.0, -1.0]
        XCTAssertEqual(device(l, config, previousTokens: [0]), 1)      // 10 → 5, so 9 wins
        XCTAssertEqual(host(l, config, previousTokens: [0]), 1)

        let neg: [Float] = [-1.0, -5.0, -1.5]
        XCTAssertEqual(device(neg, config, previousTokens: [0]), 2)    // -1 → -2, so -1.5 wins
        XCTAssertEqual(host(neg, config, previousTokens: [0]), 2)
    }

    /// A penalty of 1.0 is off, not "a penalty of one" — the host guards on `> 1.0` and so must
    /// the device path, or a default config would quietly rescale every seen logit.
    func testPenaltyOfOneChangesNothing() {
        let l: [Float] = [10.0, 1.0, 3.0]
        let off = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 1,
                                     repetitionPenalty: 1.0)
        XCTAssertEqual(device(l, off, previousTokens: [0, 0, 0]), 0)
        XCTAssertEqual(host(l, off, previousTokens: [0, 0, 0]), 0)
    }

    // MARK: - End-token suppression

    /// Suppressed ids must lose even when they are the argmax, and must stay lost through the
    /// penalty: a suppressed id that is also in the penalty window overflows to -inf on both
    /// paths, which is the interaction most likely to be broken by reordering the two steps.
    func testEndTokenSuppression() {
        var l = randomLogits(256, seed: 7)
        l[11] = 100
        l[12] = 90
        l[13] = 80

        XCTAssertEqual(device(l, Self.greedy, suppressing: [11]), 12)
        XCTAssertEqual(host(l, Self.greedy, suppressing: [11]), 12)
        XCTAssertEqual(device(l, Self.greedy, suppressing: [11, 12]), 13)
        XCTAssertEqual(host(l, Self.greedy, suppressing: [11, 12]), 13)

        let penalised = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 1,
                                           repetitionPenalty: 1.1)
        XCTAssertEqual(device(l, penalised, suppressing: [11, 12], previousTokens: [11, 12, 13]),
                       host(l, penalised, suppressing: [11, 12], previousTokens: [11, 12, 13]))
        XCTAssertNotEqual(device(l, penalised, suppressing: [11, 12], previousTokens: [11, 12]), 11)

        // Out-of-range suppressions are ignored rather than trapping.
        XCTAssertEqual(device(l, Self.greedy, suppressing: [-1, 256, 11]), 12)
    }

    /// Suppression survives sampling too, not just argmax — nothing may be drawn from a
    /// suppressed id at any point in the sweep.
    func testSuppressedIdIsUnreachableWhenSampling() {
        var l = [Float](repeating: -20, count: 128)
        l[3] = 10; l[4] = 9.5; l[5] = 9.0
        let config = ChatSamplingConfig(temperature: 1.0, topK: 0, topP: 1.0, maxTokens: 1,
                                        repetitionPenalty: 1.0)
        for u in uniforms() {
            XCTAssertNotEqual(device(l, config, suppressing: [3], uniform: u), 3)
        }
    }

    // MARK: - Sampling

    /// Temperature, top-K and top-P together, over a sweep of draws, on logits with no ties. This
    /// is the strongest statement available for the stochastic path: given the same draw the two
    /// samplers return the same token, so they differ in nothing but where the arithmetic ran.
    func testStochasticMatchesHostOverUniformSweep() {
        let configs = [
            ChatSamplingConfig(temperature: 0.7, topK: 50, topP: 0.9, maxTokens: 1,
                               repetitionPenalty: 1.1),
            ChatSamplingConfig(temperature: 1.0, topK: 0, topP: 1.0, maxTokens: 1,
                               repetitionPenalty: 1.0),
            ChatSamplingConfig(temperature: 0.3, topK: 20, topP: 0.8, maxTokens: 1,
                               repetitionPenalty: 1.0),
            ChatSamplingConfig(temperature: 2.0, topK: 8, topP: 0.5, maxTokens: 1,
                               repetitionPenalty: 1.3),
        ]
        for (c, config) in configs.enumerated() {
            let l = randomLogits(1024, seed: UInt64(100 + c))
            let history = (0 ..< 80).map { ($0 &* 13) % 1024 }
            var mismatches = 0
            for u in uniforms() {
                let d = device(l, config, previousTokens: history, uniform: u)
                let h = host(l, config, previousTokens: history, uniform: u)
                if d != h { mismatches += 1 }
            }
            XCTAssertEqual(mismatches, 0, "config \(c): \(mismatches)/\(Self.sweep) draws differ")
        }
    }

    /// Top-K is a hard bound on what can be drawn, and the two paths must bound it identically.
    /// Sweeping the draw enumerates the reachable set, which is the observable the filter decides.
    func testTopKReachableSetMatchesHost() {
        let l = randomLogits(512, seed: 41)
        for k in [1, 2, 5, 17, 64] {
            let config = ChatSamplingConfig(temperature: 1.0, topK: k, topP: 1.0, maxTokens: 1,
                                            repetitionPenalty: 1.0)
            var d = Set<Int>(), h = Set<Int>()
            for u in uniforms() {
                d.insert(device(l, config, uniform: u))
                h.insert(host(l, config, uniform: u))
            }
            XCTAssertEqual(d, h, "k=\(k)")
            XCTAssertLessThanOrEqual(d.count, k)

            // And it is the *top* k, not any k: every reachable token outranks every other.
            let ranked = l.enumerated().sorted { $0.element > $1.element }.prefix(k).map(\.offset)
            XCTAssertTrue(d.isSubset(of: Set(ranked)), "k=\(k) drew outside the top-k")
        }
    }

    /// Top-P with the threshold landing exactly on a cumulative boundary.
    ///
    /// The probabilities are made exactly representable — `n` equal logits give exactly `1/n` each
    /// for a power-of-two `n`, and the partial sums are exact — because a boundary that only lands
    /// on the threshold in real arithmetic is decided by the last float digit of each path
    /// independently, and agreeing there would be luck rather than equivalence.
    ///
    /// Equal logits mean the *identity* of the survivors is arbitrary on both paths, so the
    /// assertion is on how many survive: `topP = 0.5` over eight equal candidates keeps four, and
    /// an off-by-one in either direction is what this catches.
    func testTopPExactBoundaryKeepsSamePrefixLength() {
        let l = [Float](repeating: 2.25, count: 8)
        for (topP, expected) in [(Float(0.125), 1), (0.25, 2), (0.5, 4), (0.75, 6), (1.0, 8)] {
            let config = ChatSamplingConfig(temperature: 1.0, topK: 0, topP: topP, maxTokens: 1,
                                            repetitionPenalty: 1.0)
            var d = Set<Int>(), h = Set<Int>()
            for u in uniforms(1024) {
                d.insert(device(l, config, uniform: u))
                h.insert(host(l, config, uniform: u))
            }
            XCTAssertEqual(d.count, expected, "device kept \(d.count) at topP=\(topP)")
            XCTAssertEqual(h.count, expected, "host kept \(h.count) at topP=\(topP)")
        }
    }

    /// Top-P on well-separated probabilities, where the nucleus is unambiguous: here the two paths
    /// must agree on the tokens themselves, not only on how many there are.
    func testTopPReachableSetMatchesHost() {
        let l = randomLogits(512, seed: 63)
        for topP in [Float(0.1), 0.5, 0.9, 0.95, 0.99, 1.0] {
            let config = ChatSamplingConfig(temperature: 0.8, topK: 0, topP: topP, maxTokens: 1,
                                            repetitionPenalty: 1.0)
            var d = Set<Int>(), h = Set<Int>()
            for u in uniforms() {
                d.insert(device(l, config, uniform: u))
                h.insert(host(l, config, uniform: u))
            }
            XCTAssertEqual(d, h, "topP=\(topP)")
            XCTAssertFalse(d.isEmpty)
        }
    }

    /// All-equal logits: every token is equally likely, so the two paths cannot be compared by
    /// index — only by how much of the vocabulary stays reachable and by the draw staying inside
    /// the filters. A degenerate distribution must not collapse onto one token or escape top-K.
    func testAllEqualLogitsStayInDistribution() {
        let l = [Float](repeating: -0.5, count: 32)
        let config = ChatSamplingConfig(temperature: 0.9, topK: 10, topP: 1.0, maxTokens: 1,
                                        repetitionPenalty: 1.0)
        var d = Set<Int>(), h = Set<Int>()
        for u in uniforms(1024) {
            d.insert(device(l, config, uniform: u))
            h.insert(host(l, config, uniform: u))
        }
        XCTAssertEqual(d.count, 10, "device reached \(d.count) of the 10 equally likely candidates")
        XCTAssertEqual(h.count, 10, "host reached \(h.count) of the 10 equally likely candidates")
        XCTAssertTrue(d.allSatisfy { $0 >= 0 && $0 < 32 })
    }

    /// A single dominant token is the case a nucleus must never widen: the rest of the mass is
    /// below any threshold, so every draw lands on it.
    func testDominantTokenAlwaysWins() {
        var l = [Float](repeating: -100, count: 64)
        l[37] = 100
        let config = ChatSamplingConfig(temperature: 1.0, topK: 0, topP: 0.5, maxTokens: 1,
                                        repetitionPenalty: 1.0)
        for u in uniforms() {
            XCTAssertEqual(device(l, config, uniform: u), 37)
            XCTAssertEqual(host(l, config, uniform: u), 37)
        }
    }

    /// `uniform` is documented as `[0, 1)`; both ends have to behave. Zero is the first kept rank
    /// and anything just under one is the last, on both paths.
    func testDrawEndpointsMatchHost() {
        let l = randomLogits(256, seed: 5)
        let config = ChatSamplingConfig(temperature: 1.0, topK: 12, topP: 0.99, maxTokens: 1,
                                        repetitionPenalty: 1.0)
        for u in [Float(0), 0.5, Float(1).nextDown] {
            XCTAssertEqual(device(l, config, uniform: u), host(l, config, uniform: u), "u=\(u)")
        }
    }
}
