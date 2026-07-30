import XCTest
import Foundation
import MLX
import AudioCommon
@testable import Qwen3Chat

/// Blocked sliding-window attention must be a pure compute change.
///
/// `Gemma4Model` scores each query block against only the keys that block can reach, so a
/// sliding layer never builds a matrix wider than its 512-token window. The implementation it
/// replaced built one `[1,1,T,T]` mask per layer type and let all 42 layers compute the complete
/// score matrix before masking ~97% of it to `-inf`. Both paths are still reachable
/// (`Gemma4Model.usesReferenceFullMask`), so the two can be compared on the same weights, the same
/// prompt, and the same process — which is what these tests do.
///
/// Prompt lengths straddle the window deliberately: under it (100) the two paths are the same
/// single call, at it (512) the first block ends exactly on the boundary, one past it (513) adds
/// the first genuinely banded block, and 1600 / 4000 exercise interior blocks that share one
/// reused mask plus a short trailing block that needs its own.
final class E2EGemma4SlidingWindowTests: XCTestCase {
    private static let modelDir: URL? = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return try? HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/gemma-4-E4B-it-MLX-4bit")
    }()

    /// Logits come out in bfloat16, whose spacing at the softcap's ±30 is 0.125 — so no tolerance
    /// below ~0.13 is even expressible here, and one has to be stated in units of that grid. A change
    /// of tiling alone costs 1 to 2 of those steps: running the lm_head on one row instead of all of
    /// them, which cannot change a result, already moves logits by 0.19. Blocking, measured across
    /// these lengths, costs at most 0.69 ≈ 5 steps, accumulated over 42 layers of shortened score
    /// rows and softmax denominators. 1.0 is 8 steps.
    ///
    /// The argmax is asserted exactly, and separately shown to clear this tolerance by more than an
    /// order of magnitude — the smallest top-1/top-2 gap these prompts produce is 14.3 — so the token
    /// greedy sampling takes cannot turn on rounding. Deeper ranks are not asserted and must not be:
    /// candidates 4 and 5 come out within one bfloat16 step of each other, so their order is not
    /// reproducible under *any* retiling, including changes that provably cannot alter a result. The
    /// numbers are printed so a regression reads as a number rather than as a pass.
    private static let logitTolerance: Float = 1.0

    private func loadChat() throws -> Gemma4Chat {
        guard let dir = Self.modelDir,
              FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { throw XCTSkip("Gemma 4 model dir unavailable") }
        do { return try Gemma4Chat.fromDirectory(dir) }
        catch { throw XCTSkip("model load failed (weights/metallib): \(error)") }
    }

    /// Realistic token ids: tokenize a meeting-shaped paragraph and repeat it to length.
    static func promptTokens(_ count: Int, _ tok: Gemma4Tokenizer) -> [Int] {
        let paragraph = """
        We walked through the retrieval budget again. Ivan said the lexical arm returns nothing when \
        the planner writes a function word into the query, and Yeran wanted the passages per branch \
        raised from six to eight before we measure anything else. The index rebuild took eleven \
        minutes on the laptop and the outbox drained cleanly afterwards.
        """
        var ids: [Int] = [tok.bosTokenId]
        while ids.count < count { ids.append(contentsOf: tok.encode(paragraph)) }
        return Array(ids.prefix(count))
    }

    /// Softcapped logits for the final position, through the no-cache single forward.
    private func singleForwardLogits(_ chat: Gemma4Chat, _ tokens: [Int]) -> [Float] {
        let arr = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let logits = chat.model.forward(inputIds: arr)
        let last = logits[0, logits.dim(1) - 1].asType(.float32)
        eval(last)
        return Array(last.asArray(Float.self).prefix(chat.denseConfig.vocabSize))
    }

    /// Softcapped logits for the final position, through the incremental KV-cache prefill.
    private func cachedPrefillLogits(_ chat: Gemma4Chat, _ tokens: [Int]) -> [Float] {
        var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        let arr = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let logits = chat.model.lastTokenLogits(inputIds: arr, state: &state)
        let last = logits[0, 0].asType(.float32)
        eval(last)
        return Array(last.asArray(Float.self).prefix(chat.denseConfig.vocabSize))
    }

    private func withReferenceMask<R>(_ body: () throws -> R) rethrows -> R {
        Gemma4Model.usesReferenceFullMask = true
        defer { Gemma4Model.usesReferenceFullMask = false }
        return try body()
    }

    private func argmax(_ l: [Float]) -> Int {
        var best = 0
        for i in 1..<l.count where l[i] > l[best] { best = i }
        return best
    }

    /// The five most likely token ids in order — what sampling actually reads.
    private func top5(_ l: [Float]) -> [Int] {
        l.enumerated().sorted { $0.element > $1.element }.prefix(5).map(\.offset)
    }

    /// Top-1 minus top-2: the margin the argmax survives on, so the tolerance can be read against
    /// something rather than asserted in a vacuum.
    private func topMargin(_ l: [Float]) -> Float {
        var best = -Float.greatestFiniteMagnitude, second = -Float.greatestFiniteMagnitude
        for v in l where v > second { if v > best { second = best; best = v } else { second = v } }
        return best - second
    }

    func testBlockedAttentionMatchesFullMask() throws {
        let chat = try loadChat()
        for n in [100, 512, 513, 1600, 4000] {
            let tokens = Self.promptTokens(n, chat.gemmaTokenizer)
            XCTAssertEqual(tokens.count, n)

            let reference = withReferenceMask { singleForwardLogits(chat, tokens) }
            let blocked = singleForwardLogits(chat, tokens)
            let cached = cachedPrefillLogits(chat, tokens)

            var blockedDiff: Float = 0, cachedDiff: Float = 0
            for i in 0..<reference.count {
                blockedDiff = max(blockedDiff, abs(reference[i] - blocked[i]))
                cachedDiff = max(cachedDiff, abs(reference[i] - cached[i]))
            }
            let margin = topMargin(reference)
            print(String(format:
                "[gemma4-band] tokens=%d argmax ref=%d blocked=%d cached=%d "
                + "maxAbsDiff blocked=%.4f cached=%.4f topMargin=%.3f top5=%@",
                n, argmax(reference), argmax(blocked), argmax(cached),
                blockedDiff, cachedDiff, margin, "\(top5(reference))"))

            XCTAssertEqual(argmax(blocked), argmax(reference),
                           "blocked attention changed the next token at \(n) tokens")
            XCTAssertEqual(argmax(cached), argmax(reference),
                           "blocked KV-cache prefill changed the next token at \(n) tokens")
            XCTAssertLessThan(blockedDiff, Self.logitTolerance,
                              "blocked attention logits drifted at \(n) tokens")
            XCTAssertLessThan(cachedDiff, Self.logitTolerance,
                              "blocked KV-cache prefill logits drifted at \(n) tokens")
            // The argmax survives on a gap far wider than the tolerance, so it is not a coin toss.
            XCTAssertGreaterThan(margin, 10 * Self.logitTolerance,
                                 "top-1 margin at \(n) tokens is too narrow to conclude anything")
            Memory.clearCache()
        }
    }

    /// Decode, not just prefill: at one query the sliding layers now read the newest 512 cache
    /// entries with no mask at all instead of the whole cache behind one. Greedy generation over a
    /// prompt past the window must still produce the same tokens.
    func testBlockedDecodeMatchesFullMask() throws {
        let chat = try loadChat()
        let filler = Self.promptTokens(1200, chat.gemmaTokenizer)
        let transcript = chat.gemmaTokenizer.decode(filler)
        let messages = [
            ChatMessage(role: .system, content: "You are a helpful assistant. Give short direct answers."),
            ChatMessage(role: .user, content: transcript + "\n\nWho wanted more passages per branch?"),
        ]
        let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 24,
                                          repetitionPenalty: 1.0)

        let reference = try withReferenceMask { try chat.generate(messages: messages, sampling: sampling) }
        let blocked = try chat.generate(messages: messages, sampling: sampling)
        print("[gemma4-band] greedy reference=\(reference.debugDescription) blocked=\(blocked.debugDescription)")
        XCTAssertEqual(blocked, reference, "blocked attention changed greedy generation")
    }

    /// `lastTokenLogits` runs the 262k-wide lm_head on the final row only; it must agree with the
    /// final row of the all-positions `forward`.
    func testLastTokenLogitsMatchFullForward() throws {
        let chat = try loadChat()
        let tokens = Self.promptTokens(1600, chat.gemmaTokenizer)
        var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        let arr = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)

        let all = chat.model.forward(inputIds: arr, state: &state)
        let allLast = all[0, all.dim(1) - 1].asType(.float32)
        eval(allLast)
        let expected = Array(allLast.asArray(Float.self).prefix(chat.denseConfig.vocabSize))
        let actual = cachedPrefillLogits(chat, tokens)

        var diff: Float = 0
        for i in 0..<expected.count { diff = max(diff, abs(expected[i] - actual[i])) }
        print(String(format: "[gemma4-band] lastTokenLogits maxAbsDiff=%.4f", diff))
        XCTAssertEqual(argmax(actual), argmax(expected))
        XCTAssertLessThan(diff, Self.logitTolerance)
    }

    /// Prefill, timed, in a frame of its own so its logits are released before the caller decodes.
    /// The all-rows array is 9 GB at 17k tokens; holding it across the decode loop charged that
    /// variant's per-token cost for memory pressure rather than for the attention geometry being
    /// compared, and did so unreproducibly — 101 ms/tok in one run and 71 in the next.
    private func timedPrefill(_ chat: Gemma4Chat, _ arr: MLXArray,
                              state: inout Gemma4Model.InferenceState,
                              allPositions: Bool) -> Double {
        let start = CFAbsoluteTimeGetCurrent()
        let logits = allPositions
            ? chat.model.forward(inputIds: arr, state: &state)
            : chat.model.lastTokenLogits(inputIds: arr, state: &state)
        eval(logits)
        return CFAbsoluteTimeGetCurrent() - start
    }

    /// Wall-clock prefill and decode, decomposed over the two changes: the full mask with the
    /// lm_head over every position (what this replaced), the same mask with the lm_head on the final
    /// row alone, and blocked attention with it. Gated on `GEMMA4_PREFILL_BENCH` — it loads the 4.2 GB
    /// checkpoint and runs 17k-token forwards, so it is a measurement harness, not a test.
    func testPrefillScaling() throws {
        guard ProcessInfo.processInfo.environment["GEMMA4_PREFILL_BENCH"] != nil else {
            throw XCTSkip("set GEMMA4_PREFILL_BENCH=1 to measure prefill")
        }
        let chat = try loadChat()
        let variants: [(label: String, referenceMask: Bool, allPositionLogits: Bool)] = [
            ("full+allrows", true, true),
            ("full+lastrow", true, false),
            ("blocked", false, false),
        ]
        // Warm up both paths first: the first prefill after a load pays for paging 4.2 GB of weights
        // in and for whatever kernels have yet to be specialised, which at 1k tokens is most of the
        // measurement and would be charged to whichever variant happened to run first.
        for warmUp in [true, false] {
            Gemma4Model.usesReferenceFullMask = warmUp
            var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
            let tokens = Self.promptTokens(2048, chat.gemmaTokenizer)
            let arr = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
            eval(chat.model.lastTokenLogits(inputIds: arr, state: &state))
        }
        Gemma4Model.usesReferenceFullMask = false

        for n in [1024, 4096, 10240, 17408] {
            let tokens = Self.promptTokens(n, chat.gemmaTokenizer)
            for variant in variants {
                Gemma4Model.usesReferenceFullMask = variant.referenceMask
                defer { Gemma4Model.usesReferenceFullMask = false }
                Memory.clearCache()
                GPU.resetPeakMemory()

                let arr = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
                var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
                let prefillSeconds = timedPrefill(chat, arr, state: &state,
                                                  allPositions: variant.allPositionLogits)
                let prefillPeak = Double(Memory.peakMemory) / 1_073_741_824

                // Eight decode steps off that cache, so the per-step cost at long context is
                // measured rather than inferred.
                let step = MLXArray([Int32(tokens[0])]).expandedDimensions(axis: 0)
                let start = CFAbsoluteTimeGetCurrent()
                for _ in 0..<8 {
                    let out = chat.model.lastTokenLogits(inputIds: step, state: &state)
                    eval(out)
                }
                let decodeMs = (CFAbsoluteTimeGetCurrent() - start) / 8 * 1000

                print(String(format: "[gemma4-prefill] tokens=%5d %-12@ prefill=%6.2fs "
                             + "decode=%5.1fms/tok peak=%5.2fGB",
                             n, variant.label, prefillSeconds, decodeMs, prefillPeak))
            }
        }
    }
}
