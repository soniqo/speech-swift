import XCTest
import Foundation
import MLX
import AudioCommon
@testable import Qwen3Chat

/// Gemma 4's KV cache appends in place and drops what a sliding layer can no longer read, so the
/// map from a cache index to an absolute token position is no longer the identity. Every mask and
/// key range in `attentionBlocks` is stated in absolute positions and translated through the
/// cache's `base`; get that translation wrong and attention reads the wrong keys, which produces
/// fluent, confidently wrong text and no crash at all.
///
/// So the cache is checked against implementations that cannot share the mistake:
///   • `testDecodeMatchesCachelessForward` — greedy decode against re-running the whole prompt
///     through the no-cache forward at every step. That path has no cache, no eviction and no base,
///     so agreeing with it over a prompt already past the window is a direct check of the mapping.
///   • `testPrefillAndDecodeAgree` — one prefill against the same tokens fed one at a time, at
///     lengths either side of the 512 window. The two drive eviction completely differently: a
///     prefill stores the whole prompt and trims once, a token-at-a-time run trims repeatedly.
///
/// `testGreedyParityAgainstBaseline` is the harness that compares a long greedy run against another
/// revision's token ids; it needs a file from that other revision, so it stays opt-in.
final class E2EGemma4KVCacheTests: XCTestCase {
    private static let modelDir: URL? = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return try? HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/gemma-4-E4B-it-MLX-4bit")
    }()

    private func loadChat() throws -> Gemma4Chat {
        guard let dir = Self.modelDir,
              FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { throw XCTSkip("Gemma 4 model dir unavailable") }
        do { return try Gemma4Chat.fromDirectory(dir) }
        catch { throw XCTSkip("model load failed (weights/metallib): \(error)") }
    }

    // MARK: - Helpers

    private func array(_ tokens: [Int]) -> MLXArray {
        MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
    }

    /// Greedy argmax of the final row, read off the GPU as one index rather than as 262k floats.
    private func argmax(_ logits: MLXArray) -> Int {
        let last = logits[0, logits.dim(1) - 1]
        let best = MLX.argMax(last, axis: -1)
        eval(best)
        return best.item(Int.self)
    }

    /// Greedy continuation through the incremental cache: one prefill, then one token per step.
    private func decodeThroughCache(_ chat: Gemma4Chat, prompt: [Int], count: Int) -> [Int] {
        var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        var next = argmax(chat.model.lastTokenLogits(inputIds: array(prompt), state: &state))
        var produced = [next]
        for _ in 1..<count {
            next = argmax(chat.model.lastTokenLogits(inputIds: array([next]), state: &state))
            produced.append(next)
        }
        return produced
    }

    // MARK: - Tests

    /// The cache against no cache at all. Every step re-runs the whole sequence through the
    /// single-pass forward, which knows nothing about eviction — so if a decode step reads keys
    /// from the wrong absolute positions, the two continuations diverge.
    ///
    /// One prompt sits under the 512-token window (nothing is ever evicted) and one well past it
    /// (every sliding layer is evicting from the first step), because those are different states.
    func testDecodeMatchesCachelessForward() throws {
        let chat = try loadChat()
        for promptLength in [300, 700] {
            let prompt = E2EGemma4SlidingWindowTests.promptTokens(promptLength, chat.gemmaTokenizer)
            let steps = 12

            var reference: [Int] = []
            var sequence = prompt
            for _ in 0..<steps {
                let next = argmax(chat.model.forward(inputIds: array(sequence)))
                reference.append(next)
                sequence.append(next)
                Memory.clearCache()
            }

            let cached = decodeThroughCache(chat, prompt: prompt, count: steps)
            print("[gemma4-kv] prompt=\(promptLength) cacheless=\(reference) cached=\(cached)")
            XCTAssertEqual(cached, reference,
                           "cached decode diverged from the no-cache forward at \(promptLength) tokens")
            Memory.clearCache()
        }
    }

    /// Prefill against one-token-at-a-time, at lengths either side of the window. A prefill writes
    /// the whole prompt into the cache and trims once; the same tokens fed singly trim over and
    /// over, so the two arrive at the same cache contents by different routes and must agree on
    /// what comes next.
    func testPrefillAndDecodeAgree() throws {
        let chat = try loadChat()
        for n in [400, 520, 700] {
            let tokens = E2EGemma4SlidingWindowTests.promptTokens(n, chat.gemmaTokenizer)

            var prefillState = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            let prefilled = argmax(chat.model.lastTokenLogits(
                inputIds: array(tokens), state: &prefillState))

            var stepState = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            var stepped = -1
            for token in tokens {
                stepped = argmax(chat.model.lastTokenLogits(
                    inputIds: array([token]), state: &stepState))
            }

            print("[gemma4-kv] tokens=\(n) prefill=\(prefilled) oneAtATime=\(stepped)")
            XCTAssertEqual(stepped, prefilled,
                           "prefill and token-at-a-time disagree at \(n) tokens")
            Memory.clearCache()
        }
    }

    /// A prompt prefilled in two chunks must match the same prompt prefilled in one. The split is
    /// placed so the second chunk starts mid-window, which is the case a cache that tracked its
    /// contents by count rather than by absolute position would get wrong.
    func testSplitPrefillMatchesWholePrefill() throws {
        let chat = try loadChat()
        let tokens = E2EGemma4SlidingWindowTests.promptTokens(900, chat.gemmaTokenizer)
        for split in [200, 600] {
            var whole = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            let once = argmax(chat.model.lastTokenLogits(inputIds: array(tokens), state: &whole))

            var chunked = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            eval(chat.model.lastTokenLogits(inputIds: array(Array(tokens.prefix(split))),
                                            state: &chunked))
            let twice = argmax(chat.model.lastTokenLogits(
                inputIds: array(Array(tokens.dropFirst(split))), state: &chunked))

            print("[gemma4-kv] split=\(split) whole=\(once) chunked=\(twice)")
            XCTAssertEqual(twice, once, "split prefill at \(split) changed the next token")
            Memory.clearCache()
        }
    }

    /// Long greedy runs recorded as token ids, for comparison against another revision.
    ///
    /// `GEMMA4_KV_TOKENS` names a JSON file: written when it does not exist, asserted against when
    /// it does. Greedy decoding is deterministic, so a single differing id is a real defect rather
    /// than sampling noise. Opt-in because it needs a file produced by a different checkout — the
    /// two tests above are the ones that stand on their own.
    func testGreedyParityAgainstBaseline() throws {
        guard let path = ProcessInfo.processInfo.environment["GEMMA4_KV_TOKENS"] else {
            throw XCTSkip("set GEMMA4_KV_TOKENS=<file> to record or compare greedy token ids")
        }
        let chat = try loadChat()
        // Under the window, past it, and one long enough that 600 generated tokens cross the
        // window boundary many times over.
        let cases: [(prompt: Int, generate: Int)] = [(300, 64), (1_536, 600), (2_048, 200)]

        var produced: [String: [Int]] = [:]
        for c in cases {
            let prompt = E2EGemma4SlidingWindowTests.promptTokens(c.prompt, chat.gemmaTokenizer)
            let tokens = decodeThroughCache(chat, prompt: prompt, count: c.generate)
            produced["\(c.prompt)x\(c.generate)"] = tokens
            print("[gemma4-kv] prompt=\(c.prompt) generated=\(tokens.count) head=\(tokens.prefix(8))")
            Memory.clearCache()
        }

        let url = URL(fileURLWithPath: path)
        guard let data = try? Data(contentsOf: url),
              let baseline = try? JSONDecoder().decode([String: [Int]].self, from: data)
        else {
            try JSONEncoder().encode(produced).write(to: url)
            print("[gemma4-kv] wrote baseline token ids to \(path)")
            return
        }
        XCTAssertEqual(Set(produced.keys), Set(baseline.keys))
        for (key, expected) in baseline {
            let actual = produced[key] ?? []
            if actual != expected {
                let firstDiff = zip(actual, expected).enumerated().first { $0.element.0 != $0.element.1 }
                XCTFail("greedy tokens differ for \(key) at index "
                        + "\(firstDiff?.offset.description ?? "length") "
                        + "(\(actual.count) vs \(expected.count) tokens)")
            } else {
                print("[gemma4-kv] \(key): \(expected.count) tokens identical")
            }
        }
    }

    /// Decode cost and cache residency against context length. Gated on `GEMMA4_DECODE_BENCH` — it
    /// loads the 4.2 GB checkpoint and prefills 17k tokens, so it is a measurement harness rather
    /// than a test.
    func testDecodeScaling() throws {
        guard ProcessInfo.processInfo.environment["GEMMA4_DECODE_BENCH"] != nil else {
            throw XCTSkip("set GEMMA4_DECODE_BENCH=1 to measure decode cost")
        }
        let chat = try loadChat()
        // Warm up prefill *and* decode before anything is timed. The first pass after a load pays
        // for paging 4.2 GB of weights in and for every kernel that has yet to be specialised, and
        // decode uses kernels prefill never touches — measured, that one-time cost is large enough
        // to make a 24-step average of the first context look slower than the last one, which is
        // not a thing decode can do.
        var warm = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        let warmPrompt = E2EGemma4SlidingWindowTests.promptTokens(2_048, chat.gemmaTokenizer)
        eval(chat.model.lastTokenLogits(inputIds: array(warmPrompt), state: &warm))
        for _ in 0..<16 {
            eval(chat.model.lastTokenLogits(inputIds: array([warmPrompt[0]]), state: &warm))
        }

        for n in [512, 2_560, 10_240, 17_408] {
            let tokens = E2EGemma4SlidingWindowTests.promptTokens(n, chat.gemmaTokenizer)
            Memory.clearCache()
            GPU.resetPeakMemory()

            var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            eval(chat.model.lastTokenLogits(inputIds: array(tokens), state: &state))
            let afterPrefill = Double(Memory.peakMemory) / 1_073_741_824

            let step = array([tokens[0]])
            // A few unmeasured steps first: the first decode off a fresh prefill is the one that
            // grows the global caches out of the buffer the prompt itself became.
            for _ in 0..<8 { eval(chat.model.lastTokenLogits(inputIds: step, state: &state)) }

            let steps = 64
            let start = CFAbsoluteTimeGetCurrent()
            for _ in 0..<steps { eval(chat.model.lastTokenLogits(inputIds: step, state: &state)) }
            let msPerToken = (CFAbsoluteTimeGetCurrent() - start) / Double(steps) * 1000
            let resident = Double(Memory.activeMemory) / 1_073_741_824

            print(String(format: "[gemma4-kv-bench] context=%5d decode=%6.2f ms/tok "
                         + "prefillPeak=%5.2fGB residentAfterDecode=%5.2fGB",
                         n, msPerToken, afterPrefill, resident))
        }
    }
}
