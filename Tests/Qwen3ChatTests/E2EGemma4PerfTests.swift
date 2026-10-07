import XCTest
import Foundation
import MLX
import MLXNN
import MLXRandom
@testable import Qwen3Chat

/// Where a Gemma 4 request spends its time: reading the prompt, and each generated token, with and
/// without a JSON constraint, and the forward pass alone against the whole decode step.
///
/// A measurement, not a gate: it prints and asserts nothing about speed. Run it alone, with nothing
/// else on the GPU:
///
///     GEMMA4_PERF=1 GEMMA4_MODEL_DIR=<gemma-4-E4B-it-MLX-4bit> swift test --filter E2EGemma4PerfTests
final class E2EGemma4PerfTests: XCTestCase {
    private static var modelDir: URL? {
        guard ProcessInfo.processInfo.environment["GEMMA4_PERF"] == "1",
              let path = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] else { return nil }
        return URL(fileURLWithPath: path)
    }

    private func loadChat() throws -> Gemma4Chat {
        guard let dir = Self.modelDir,
              FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path)
        else { throw XCTSkip("set GEMMA4_PERF=1 and GEMMA4_MODEL_DIR") }
        return try Gemma4Chat.fromDirectory(dir)
    }

    /// A prompt of about `words` words, shaped like a Discover pass: a long fixed instruction and
    /// a transcript excerpt.
    private func prompt(_ chat: Gemma4Chat, words: Int) -> [Int] {
        let sentence = "The team agreed to move the pilot to the second week of March and to send the "
            + "revised security answers before the review, while the budget stays at forty thousand. "
        let body = String(repeating: sentence, count: max(1, words / 30))
        return Gemma4ChatTemplate.encode(
            messages: [
                ChatMessage(role: .system, content: "You judge transcript passages. " + body),
                ChatMessage(role: .user, content: "Question: what was agreed?\nPassage: " + sentence),
            ],
            tokenizer: chat.gemmaTokenizer)
    }

    private func seconds(_ body: () -> Void) -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        body()
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// One decode step's GPU time taken apart: the 42 MLPs chained with nothing else, the tied
    /// output projection, and the whole step. Each piece's bytes over its time is the bandwidth it
    /// reaches; what the whole step spends beyond its pieces is small-kernel work between them.
    func testWhereADecodeStepSpendsItsTime() throws {
        let chat = try loadChat()
        let model = chat.model
        let x = MLXRandom.normal([1, 1, chat.denseConfig.hiddenSize]).asType(.bfloat16)
        eval(x)
        func timed(_ label: String, bytes: Double, repeats: Int = 40, _ body: () -> MLXArray) {
            eval(body())
            let t = seconds { for _ in 0 ..< repeats { eval(body()) } } / Double(repeats)
            print(String(format: "[gemma4-perf] %@: %.2f ms, %.0f GB/s", label, t * 1000,
                         bytes / t / 1e9))
        }
        func weightBytes(_ module: Module) -> Double {
            Double(module.parameters().flattened().reduce(0) { $0 + $1.1.nbytes })
        }
        let mlpBytes = model.layers.reduce(0.0) { $0 + weightBytes($1.mlp) }
        timed("42 MLPs chained", bytes: mlpBytes) {
            var h = x
            for layer in model.layers { h = layer.mlp(h) }
            return h
        }
        timed("one MLP", bytes: weightBytes(model.layers[0].mlp)) { model.layers[0].mlp(x) }
        timed("output projection", bytes: weightBytes(model.embedTokens)) {
            model.embedTokens.asLinear(x)
        }
        let attnBytes = model.layers.reduce(0.0) { $0 + weightBytes($1.attn) }
        let allBytes = weightBytes(model) - weightBytes(model.embedTokensPerLayer)
        print(String(format: "[gemma4-perf] weights read per token: MLP %.2f GB, attention %.2f GB, all but PLE table %.2f GB",
                     mlpBytes / 1e9, attnBytes / 1e9, allBytes / 1e9))
        // The whole step, against a short cache.
        var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        eval(model.lastTokenLogits(
            inputIds: MLXArray((0 ..< 200).map { Int32(1000 + $0) }).expandedDimensions(axis: 0),
            state: &state))
        timed("whole decode step", bytes: allBytes) {
            model.forward(inputIds: MLXArray([Int32(100)]).expandedDimensions(axis: 0), state: &state)
        }
    }

    func testWhereARequestSpendsItsTime() throws {
        let chat = try loadChat()
        let warm = prompt(chat, words: 60)
        chat.decode(promptTokens: warm, sampling: .init(temperature: 0, maxTokens: 8), onText: { _ in })

        // Prefill alone: max one token, so the time is the prompt plus one step.
        for words in [300, 1000, 3000] {
            let tokens = prompt(chat, words: words)
            let prefill = seconds {
                chat.decode(promptTokens: tokens, sampling: .init(temperature: 0, maxTokens: 1),
                            onText: { _ in })
            }
            print(String(format: "[gemma4-perf] prefill %5d tokens: %.3f s (%.0f tok/s)",
                         tokens.count, prefill, Double(tokens.count) / prefill))
        }

        // Decode: the same prompt with and without 128 generated tokens, so the difference is the
        // generation alone. Product sampling (temperature 0.1, top-k 20, top-p 0.85, penalty 1.1).
        let tokens = prompt(chat, words: 1000)
        let product = ChatSamplingConfig(temperature: 0.1, topK: 20, topP: 0.85, maxTokens: 128,
                                         repetitionPenalty: 1.1)
        var oneStep = product; oneStep.maxTokens = 1
        let base = seconds { chat.decode(promptTokens: tokens, sampling: oneStep, onText: { _ in }) }
        var generated = 0
        let full = seconds {
            chat.decode(promptTokens: tokens, sampling: product, onToken: { _ in generated += 1 },
                        onText: { _ in })
        }
        print(String(format: "[gemma4-perf] decode %d tokens: %.1f ms/token (sampled)",
                     generated, (full - base) * 1000 / Double(max(1, generated - 1))))

        var greedy = product; greedy.temperature = 0; greedy.repetitionPenalty = 1.0
        generated = 0
        let fullGreedy = seconds {
            chat.decode(promptTokens: tokens, sampling: greedy, onToken: { _ in generated += 1 },
                        onText: { _ in })
        }
        print(String(format: "[gemma4-perf] decode %d tokens: %.1f ms/token (greedy, no penalty)",
                     generated, (fullGreedy - base) * 1000 / Double(max(1, generated - 1))))

        // Constrained: a Discover-shaped schema with a free-text field long enough to decode.
        let schema = #"{"type":"object","additionalProperties":false,"required":["decision","quote","reason"],"properties":{"decision":{"type":"string","enum":["KEEP","DROP"]},"quote":{"type":"string"},"reason":{"type":"string"}}}"#
        var constrained = product; constrained.responseFormat = .jsonSchema(schema)
        for engine in [JSONConstraintEngine.xgrammar, .swift] {
            let constraint = try chat.makeConstraint(constrained.responseFormat, engine: engine)
            generated = 0
            let t = seconds {
                chat.decode(promptTokens: tokens, sampling: constrained, constraint: constraint,
                            onToken: { _ in generated += 1 }, onText: { _ in })
            }
            print(String(format: "[gemma4-perf] constrained (%@) %d tokens: %.1f ms/token",
                         "\(engine)", generated, (t - base) * 1000 / Double(max(1, generated - 1))))
        }

        // The forward pass alone, one token at a time against a 1000-word cache: the floor any
        // sampler or constraint work sits on top of.
        var state = Gemma4Model.InferenceState.initial(config: chat.denseConfig)
        let promptArray = MLXArray(tokens.map { Int32($0) }).expandedDimensions(axis: 0)
        eval(chat.model.lastTokenLogits(inputIds: promptArray, state: &state))
        let steps = 64
        let forward = seconds {
            for _ in 0 ..< steps {
                let logits = chat.model.forward(
                    inputIds: MLXArray([Int32(100)]).expandedDimensions(axis: 0), state: &state)
                eval(logits)
            }
        }
        print(String(format: "[gemma4-perf] forward only: %.1f ms/token", forward * 1000 / Double(steps)))

        // The same steps split into building the lazy graph on the CPU and running it. Measured
        // 2026-09-28 on an M5 Pro: about 1 ms of about 18, so overlapping the two (pipelining
        // the step on the unread token) was tried and bought nothing measurable.
        var build = 0.0, run = 0.0
        for _ in 0 ..< steps {
            var logits: MLXArray!
            build += seconds {
                logits = chat.model.forward(
                    inputIds: MLXArray([Int32(100)]).expandedDimensions(axis: 0), state: &state)
            }
            run += seconds { eval(logits) }
        }
        print(String(format: "[gemma4-perf] per token: graph build %.1f ms, evaluation %.1f ms",
                     build * 1000 / Double(steps), run * 1000 / Double(steps)))
    }
}
