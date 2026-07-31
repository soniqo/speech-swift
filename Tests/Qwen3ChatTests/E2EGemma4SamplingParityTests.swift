import XCTest
import Foundation
import MLX
import AudioCommon
@testable import Qwen3Chat

/// The Gemma 4 decode loop against the one it replaced, with the real model behind it.
///
/// `ChatSamplerDeviceTests` compares the two samplers on constructed logits, which says nothing
/// about the loop around them: the previous shape evaluated each step's logits before sampling,
/// and dropping that `eval` lets MLX schedule the model forward and the sampling as one graph.
/// Fusion can change how a kernel is dispatched, so the decision that greedy decoding is
/// unchanged has to be made against real weights, at real context lengths, over enough tokens
/// for a single flipped argmax to diverge into different text.
///
/// `reference` below is the previous loop copied verbatim — prefill, `eval`, pull all 262,144
/// logits to the host, suppress, sample there. It is the thing under comparison, so it is kept
/// as it was rather than tidied.
final class E2EGemma4SamplingParityTests: XCTestCase {

    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return (try? HuggingFaceDownloader.getCacheDirectory(
            for: "aufklarer/gemma-4-E4B-it-MLX-4bit"))
            ?? URL(fileURLWithPath: "/nonexistent")
    }()

    private func loadChat() throws -> Gemma4Chat {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }
        do { return try Gemma4Chat.fromDirectory(Self.modelDir) }
        catch { throw XCTSkip("model load failed (weights/metallib): \(error)") }
    }

    /// Greedy: no draw to differ, so the sequences must match token for token.
    private static let greedy = ChatSamplingConfig(
        temperature: 0, topK: 0, topP: 1.0, maxTokens: 200, repetitionPenalty: 1.1)

    // MARK: - The loop being replaced

    /// The decode loop as it stood before sampling moved onto the device, reproduced exactly:
    /// evaluate the step, copy the whole vocabulary to the host, suppress and sample there.
    private func reference(
        _ chat: Gemma4Chat, promptTokens: [Int], sampling: ChatSamplingConfig,
        onToken: (Int) -> Void = { _ in }
    ) -> [Int] {
        func lastLogits(_ logits: MLXArray) -> [Float] {
            let t = logits.dim(1)
            let last = logits[0, t - 1].asType(.float32)
            eval(last)
            let all: [Float] = last.asArray(Float.self)
            return Array(all.prefix(chat.denseConfig.vocabSize))
        }
        func suppressEndTokens(_ logits: inout [Float]) {
            let ninf = -Float.greatestFiniteMagnitude
            for id in chat.gemmaTokenizer.eosTokenIds where id >= 0 && id < logits.count {
                logits[id] = ninf
            }
        }

        chat.resetState()
        let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
        let prefillLogits = chat.model.lastTokenLogits(inputIds: promptArray, state: &chat.state)
        eval(prefillLogits)
        var logits = lastLogits(prefillLogits)

        var history = promptTokens
        var produced = false
        var generated: [Int] = []
        var filter = Gemma4AnswerFilter(tokenizer: chat.gemmaTokenizer)

        for _ in 0 ..< sampling.maxTokens {
            if !produced { suppressEndTokens(&logits) }

            let next = ChatSampler.sample(
                logits: logits, config: sampling, previousTokens: history)

            if chat.gemmaTokenizer.eosTokenIds.contains(next) { break }
            history.append(next)
            generated.append(next)
            onToken(next)

            let text = filter.consume(next)
            if !text.isEmpty { produced = true }

            let arr = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
            let step = chat.model.forward(inputIds: arr, state: &chat.state)
            eval(step)
            logits = lastLogits(step)
        }
        return generated
    }

    private func production(
        _ chat: Gemma4Chat, promptTokens: [Int], sampling: ChatSamplingConfig
    ) -> (tokens: [Int], text: String) {
        var tokens: [Int] = []
        var text = ""
        chat.decode(promptTokens: promptTokens, sampling: sampling,
                    onToken: { tokens.append($0) }, onText: { text += $0 })
        return (tokens, text)
    }

    // MARK: - Prompts

    /// Five prompts of increasing length, the last over 2,000 tokens so the comparison covers a
    /// prefill long enough to exercise the sliding-window layers' cache as well as a short one.
    ///
    /// Each asks for a long answer on purpose. A prompt the model can satisfy in a sentence stops
    /// at the end token after a handful of steps, and a handful of steps is not a parity test: one
    /// flipped argmax has to have room to turn into different text.
    private func prompts(_ chat: Gemma4Chat) -> [(name: String, tokens: [Int])] {
        let filler = """
            The meeting opened with a review of last quarter's numbers, which came in slightly \
            ahead of plan on revenue and slightly behind on margin. Procurement raised the cost \
            of the new tooling, engineering raised the schedule, and nobody raised their hand \
            when asked who would own the migration. We agreed to revisit it on Thursday.
            """
        func encode(_ user: String) -> [Int] {
            Gemma4ChatTemplate.encode(messages: [
                ChatMessage(role: .system,
                            content: "You are a helpful assistant. Answer thoroughly, at length, "
                                + "and in complete sentences."),
                ChatMessage(role: .user, content: user),
            ], tokenizer: chat.gemmaTokenizer)
        }
        return [
            ("tiny", encode("Count from one to sixty, spelling each number out, one per line.")),
            ("short", encode("Explain in detail how a hash table resolves collisions, covering "
                             + "chaining and open addressing, with worked examples.")),
            ("medium", encode("Write a detailed explanation of how TCP congestion control works, "
                              + "covering slow start, congestion avoidance, fast retransmit and "
                              + "fast recovery, in at least four paragraphs.")),
            ("list", encode("List twenty things to check before a long flight. Give each one its "
                            + "own numbered line with a full sentence of reasoning.")),
            ("long", encode(String(repeating: filler + "\n\n", count: 45)
                            + "Summarise the discussion above in detail, covering every point "
                            + "raised, in at least four hundred words.")),
        ]
    }

    // MARK: - Parity

    /// The gate. Both loops drive the model themselves, so this covers the removed per-step `eval`
    /// as well as the sampler — comparing the samplers on shared logits would not.
    func testGreedyDecodeMatchesPreviousLoop() throws {
        let chat = try loadChat()
        let cases = prompts(chat)
        XCTAssertGreaterThan(cases.last!.tokens.count, 2000,
                             "the long prompt must actually be long — got \(cases.last!.tokens.count)")

        for (name, tokens) in cases {
            let expected = reference(chat, promptTokens: tokens, sampling: Self.greedy)
            let actual = production(chat, promptTokens: tokens, sampling: Self.greedy)
            print("[gemma4-parity] \(name): prompt=\(tokens.count) generated=\(actual.tokens.count)")
            XCTAssertEqual(actual.tokens, expected,
                           "\(name) (prompt \(tokens.count) tokens) diverged from the host loop")
            // A short generation would pass this test without testing much; the prompts are
            // written to run the budget out.
            XCTAssertGreaterThanOrEqual(actual.tokens.count, 150,
                                        "\(name) stopped after \(actual.tokens.count) tokens")
        }
    }

    /// Greedy is the only comparison that can be exact end to end; with a draw in the loop the two
    /// paths consume different uniforms and separate immediately. What is checked instead is that
    /// each step agrees given the *same* draw — the equivalence `ChatSamplerDeviceTests` shows on
    /// constructed logits, here on the model's own, where the distribution is whatever the model
    /// produced rather than whatever a test built.
    ///
    /// One class of disagreement is allowed and counted rather than failed. The lm_head emits
    /// bfloat16, which carries eight mantissa bits, so among a few dozen candidates *exactly*
    /// equal logits are common — and neither path breaks equal probabilities by index. A step
    /// where the two picked different tokens carrying bit-identical logits picked between two
    /// tokens the samplers were equally entitled to. A step where they differ on anything else is
    /// a real divergence, and that is what this fails on.
    ///
    /// Measured over the 1,000 steps below: a handful land on such a tie, and none has ever
    /// differed for another reason.
    func testSampledStepsMatchHostGivenTheSameDraw() throws {
        let chat = try loadChat()
        let sampling = ChatSamplingConfig(temperature: 0.7, topK: 50, topP: 0.9, maxTokens: 200,
                                          repetitionPenalty: 1.1)
        var steps = 0, ties = 0

        // Fixed draws: a failure has to be reproducible, and `Float.random` would make the tie
        // count wander from run to run.
        var seed: UInt64 = 0x5EED_1234_ABCD_0001
        func nextUniform() -> Float {
            seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return Float(seed >> 40) / Float(1 << 24)
        }

        for (name, promptTokens) in prompts(chat) {
            chat.resetState()
            let promptArray = MLXArray(promptTokens.map { Int32($0) }).expandedDimensions(axis: 0)
            var logits = chat.model.lastTokenLogits(inputIds: promptArray, state: &chat.state)

            var history = promptTokens
            var produced = false
            var filter = Gemma4AnswerFilter(tokenizer: chat.gemmaTokenizer)
            let endTokens = Array(chat.gemmaTokenizer.eosTokenIds)
            var diverged = 0

            for _ in 0 ..< sampling.maxTokens {
                let suppressing = produced ? [] : endTokens
                let u = nextUniform()

                let onDevice = ChatSampler.sampleOnDevice(
                    logits: logits, config: sampling, suppressing: suppressing,
                    previousTokens: history, vocabSize: chat.denseConfig.vocabSize,
                    uniform: u
                ).item(Int.self)

                var host = logits[0, logits.dim(1) - 1].asType(.float32).asArray(Float.self)
                host = Array(host.prefix(chat.denseConfig.vocabSize))
                for id in suppressing where id >= 0 && id < host.count {
                    host[id] = -Float.greatestFiniteMagnitude
                }
                let onHost = ChatSampler.sample(logits: host, config: sampling,
                                                previousTokens: history, uniform: u)

                if onDevice != onHost {
                    // Compare what the samplers ranked, which is the penalised logit, not the raw
                    // one — the penalty can separate two candidates that arrived equal.
                    var penalised = host
                    let p = sampling.repetitionPenalty
                    for tid in Set(history.suffix(64)) where tid >= 0 && tid < penalised.count {
                        penalised[tid] = penalised[tid] > 0 ? penalised[tid] / p : penalised[tid] * p
                    }
                    if penalised[onDevice] == penalised[onHost] {
                        ties += 1
                    } else {
                        diverged += 1
                        print(String(format: "[gemma4-parity] %@ u=%.6f dev=%d(%.9g) host=%d(%.9g)",
                                     name, u, onDevice, penalised[onDevice],
                                     onHost, penalised[onHost]))
                    }
                }
                steps += 1

                let next = onHost
                if chat.gemmaTokenizer.eosTokenIds.contains(next) { break }
                history.append(next)
                if !filter.consume(next).isEmpty { produced = true }

                let arr = MLXArray([Int32(next)]).expandedDimensions(axis: 0)
                logits = chat.model.forward(inputIds: arr, state: &chat.state)
            }
            XCTAssertEqual(diverged, 0,
                           "\(name): \(diverged) sampled steps chose differently-weighted tokens")
        }
        print("[gemma4-parity] sampled steps compared: \(steps), equal-logit ties: \(ties)")
        XCTAssertGreaterThanOrEqual(steps, 500)
    }

    // MARK: - Throughput

    /// Both loops, back to back on one loaded model, at a short and a long context.
    ///
    /// Measured in the same process on the same weights so the comparison is not across machine
    /// states; the long context is what matters, since decode cost grows with the KV cache while
    /// the per-token vocabulary transfer this removes does not.
    ///
    /// The clock starts at the first sampled token and stops a fixed number of tokens later.
    ///
    /// Both bounds matter. A 10,000-token prefill takes seconds, so leaving it inside would make
    /// the figure mostly prefill and its run-to-run variance larger than the difference being
    /// measured — an earlier version differenced two budgets instead and reported a 1.27x that
    /// was prefill noise. And a fixed window rather than "however many tokens came out" is what
    /// makes the two loops comparable when sampling: they consume different draws, stop at
    /// different lengths, and per-token cost rises with the cache, so an unequal number of tokens
    /// compares different things.
    func testDecodeThroughput() throws {
        try XCTSkipUnless(ProcessInfo.processInfo.environment["GEMMA4_BENCH"] != nil,
                          "set GEMMA4_BENCH=1 to measure decode throughput")
        let chat = try loadChat()
        let window = 64
        let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 256,
                                          repetitionPenalty: 1.1)

        func msPerToken(_ loop: ((Int) -> Void) -> Void) -> Double {
            var start: Date?
            var end: Date?
            var counted = 0
            loop { _ in
                if start == nil { start = Date() } else { counted += 1 }
                if counted == window && end == nil { end = Date() }
            }
            guard let start, let end else { return .nan }
            return end.timeIntervalSince(start) * 1000 / Double(window)
        }

        // Assembled from token ids rather than by encoding a huge string. The prompt only has to
        // be long enough to load the KV cache to the right depth, and tokenising tens of thousands
        // of tokens would take longer than the decode being measured. The layout is
        // `Gemma4ChatTemplate`'s.
        let tok = chat.gemmaTokenizer
        let filler = tok.encode("The migration plan was reviewed again, and the schedule slipped "
                                + "again, and the owner was still unnamed. ")
        func prompt(targetTokens: Int) -> [Int] {
            var ids = [tok.bosTokenId]
            ids += tok.encode("<|turn>user\n")
            let tail = tok.encode("\n\nSummarise the above.") + tok.encode("<turn|>\n")
                + tok.encode("<|turn>model\n")
            while ids.count + filler.count + tail.count <= targetTokens { ids += filler }
            return ids + tail
        }

        // Both configurations, because they take different device paths: greedy stops at an
        // `argMax` over the vocabulary, while sampling ranks it and runs the nucleus filters.
        let sampled = ChatSamplingConfig(temperature: 0.7, topK: 50, topP: 0.9, maxTokens: 256,
                                         repetitionPenalty: 1.1)
        let warm = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 4,
                                      repetitionPenalty: 1.1)

        for target in [500, 10_000] {
            let tokens = prompt(targetTokens: target)
            // Warm both paths for this shape so the first timed run isn't paying for dispatch.
            _ = production(chat, promptTokens: tokens, sampling: warm)
            _ = reference(chat, promptTokens: tokens, sampling: warm)

            for (label, config) in [("greedy", sampling), ("sampled", sampled)] {
                // Interleaved and repeated, so a thermal drift over the run moves both figures
                // instead of only the second, and the best of each stands for the machine when it
                // was not busy with something else.
                var host = Double.infinity, device = Double.infinity
                for _ in 0 ..< 3 {
                    host = min(host, msPerToken { emit in
                        _ = reference(chat, promptTokens: tokens, sampling: config, onToken: emit)
                    })
                    device = min(device, msPerToken { emit in
                        chat.decode(promptTokens: tokens, sampling: config,
                                    onToken: emit, onText: { _ in })
                    })
                }
                print(String(format: "[gemma4-bench] context=%d %@ window=%d host=%.2f ms/tok "
                                     + "device=%.2f ms/tok speedup=%.3fx",
                             tokens.count, label, window, host, device, host / device))
                XCTAssertFalse(host.isNaN || device.isNaN,
                               "\(label) at \(tokens.count): fewer than \(window) tokens generated")
            }

            // Only greedy can be compared token for token: with a draw in the loop the two runs
            // consume different uniforms.
            XCTAssertEqual(production(chat, promptTokens: tokens, sampling: sampling).tokens,
                           reference(chat, promptTokens: tokens, sampling: sampling),
                           "diverged at context \(tokens.count)")
        }
    }
}
