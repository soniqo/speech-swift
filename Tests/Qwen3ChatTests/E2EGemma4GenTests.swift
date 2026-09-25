import XCTest
import Foundation
@testable import Qwen3Chat

/// End-to-end streaming generation: the incremental KV-cache + chat-template path must produce a
/// coherent, thinking-free reply. Asserts the streamed answer to a factual prompt contains "Paris"
/// and carries no chat-template / reasoning-channel markup.
final class E2EGemma4GenTests: XCTestCase {
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath:
            "/Users/ivan/repos/runner-speech-models/speech-models/out/mlx/gemma-4-E2B-it-MLX-4bit")
    }()

    func testStreamingReplyAboutParis() throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }

        let chat: Gemma4Chat
        do {
            chat = try Gemma4Chat.fromDirectory(Self.modelDir)
        } catch {
            throw XCTSkip("model load failed (weights/metallib): \(error)")
        }

        let messages = [
            ChatMessage(role: .system, content: "You are a helpful assistant. Give short direct answers."),
            ChatMessage(role: .user, content: "What is the capital of France? Answer in one short sentence."),
        ]
        // Greedy for determinism; cap tokens so a runaway thought channel can't hang the test.
        let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1.0, maxTokens: 64,
                                          repetitionPenalty: 1.0)

        let reply = try chat.generate(messages: messages, sampling: sampling)
        print("[gemma4-gen] reply=\(reply.debugDescription)")

        XCTAssertFalse(reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                       "streamed reply must be non-empty")
        XCTAssertTrue(reply.contains("Paris"), "reply must name Paris — got: \(reply)")
        for marker in ["<|channel>", "<channel|>", "<|turn>", "<turn|>", "<start_of_turn>",
                       "<end_of_turn>", "thought", "<bos>", "<eos>"] {
            XCTAssertFalse(reply.contains(marker), "reply must not contain '\(marker)' — got: \(reply)")
        }
    }

    func testCooperativeCancellationStopsBeforeTheNextToken() throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }

        let chat: Gemma4Chat
        do {
            chat = try Gemma4Chat.fromDirectory(Self.modelDir)
        } catch {
            throw XCTSkip("model load failed (weights/metallib): \(error)")
        }

        let prompt = Gemma4ChatTemplate.encode(
            messages: [
                ChatMessage(role: .system, content: "Answer directly."),
                ChatMessage(role: .user, content: "Count upward forever."),
            ],
            tokenizer: chat.gemmaTokenizer
        )
        let sampling = ChatSamplingConfig(
            temperature: 0,
            topK: 0,
            topP: 1.0,
            maxTokens: 64,
            repetitionPenalty: 1.0
        )
        var admissionChecks = 0
        var emittedTokens = 0
        chat.decode(
            promptTokens: prompt,
            sampling: sampling,
            shouldContinue: {
                admissionChecks += 1
                return admissionChecks <= 2
            },
            onToken: { _ in emittedTokens += 1 },
            onText: { _ in }
        )

        XCTAssertEqual(admissionChecks, 3)
        XCTAssertEqual(emittedTokens, 2)
    }

    /// Deterministic (no model): the reasoning-channel filter drops a `<|channel>thought … <channel|>`
    /// block and emits only the answer text after it, decoding the SentencePiece byte-fallback tokens.
    func testAnswerFilterSuppressesThoughtChannel() throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("Gemma 4 tokenizer unavailable: \(Self.modelDir.path)")
        }
        let tok = Gemma4Tokenizer()
        try tok.load(from: Self.modelDir)

        // Build a synthetic stream: <|channel>(100) "thought\n let me think \n" <channel|>(101)
        // then the real answer "Paris." — only the answer must survive.
        var stream: [Int] = [100]
        stream += tok.encode("thought\nThe user wants the capital. Let me think.\n")
        stream += [101]
        stream += tok.encode("Paris.")

        var filter = Gemma4AnswerFilter(tokenizer: tok)
        var out = ""
        for id in stream { out += filter.consume(id) }
        if let tail = filter.flush() { out += tail }

        print("[gemma4-gen] filtered=\(out.debugDescription)")
        XCTAssertEqual(out, "Paris.", "thought channel must be fully suppressed")
        XCTAssertFalse(out.contains("think"))
        XCTAssertFalse(out.contains("<|channel>"))

        // And with no channel block, text passes through unchanged.
        var f2 = Gemma4AnswerFilter(tokenizer: tok)
        var out2 = ""
        for id in tok.encode("The capital of France is Paris.") { out2 += f2.consume(id) }
        if let t = f2.flush() { out2 += t }
        XCTAssertEqual(out2, "The capital of France is Paris.")
    }

    /// Sanity: the first token off the incremental KV-cache path must equal a single-forward argmax,
    /// proving the cache + cross-layer KV sharing reproduce the no-cache forward.
    func testCacheFirstTokenMatchesSingleForward() throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }
        let chat: Gemma4Chat
        do { chat = try Gemma4Chat.fromDirectory(Self.modelDir) }
        catch { throw XCTSkip("model load failed: \(error)") }

        let prompt = [818, 5279, 529, 7001, 563]   // same prompt as the parity test
        let singleForwardArgmax = chat.nextTokenArgmax(promptTokens: prompt).argmax
        let cacheArgmax = chat.firstTokenViaCache(promptTokens: prompt)
        print("[gemma4-gen] single=\(singleForwardArgmax) cache=\(cacheArgmax)")
        XCTAssertEqual(cacheArgmax, singleForwardArgmax,
                       "KV-cache prefill must match the single-forward argmax")
    }

    /// Sampled replies never open the reasoning channel, so a budget cannot be spent on text the
    /// filter will discard.
    func testSampledDecodeNeverOpensTheReasoningChannel() throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }
        let chat: Gemma4Chat
        do { chat = try Gemma4Chat.fromDirectory(Self.modelDir) }
        catch { throw XCTSkip("model load failed: \(error)") }
        let messages = [
            ChatMessage(role: .system, content: "Reason carefully before you answer."),
            ChatMessage(role: .user, content: "A train leaves at 9:40 and arrives at 13:05. How long is the trip?"),
        ]
        let prompt = Gemma4ChatTemplate.encode(messages: messages, tokenizer: chat.gemmaTokenizer)
        for _ in 0 ..< 3 {
            var tokens: [Int] = []
            var text = ""
            chat.decode(
                promptTokens: prompt,
                sampling: ChatSamplingConfig(temperature: 0.7, topK: 40, topP: 0.9, maxTokens: 96,
                                             repetitionPenalty: 1.1),
                onToken: { tokens.append($0) },
                onText: { text += $0 })
            XCTAssertFalse(tokens.contains(Gemma4AnswerFilter.channelOpen), "reasoning channel opened")
            XCTAssertFalse(text.isEmpty, "empty reply")
        }
    }

    /// Replays recorded requests (`GEMMA4_REPLAY_PROMPTS`: a JSON array of `{id, system, user,
    /// max_tokens, temperature}`) with production sampling and counts empty replies and
    /// reasoning-channel tokens. Measurement only; skipped without the file.
    func testReplayRecordedPromptsForEmptyReplies() throws {
        guard let path = ProcessInfo.processInfo.environment["GEMMA4_REPLAY_PROMPTS"] else {
            throw XCTSkip("set GEMMA4_REPLAY_PROMPTS")
        }
        let chat = try Gemma4Chat.fromDirectory(Self.modelDir)
        let entries = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [[String: Any]])
        let repeats = Int(ProcessInfo.processInfo.environment["GEMMA4_REPLAY_REPEATS"] ?? "3") ?? 3
        var runs = 0, empty = 0, opened = 0
        for entry in entries {
            let messages = [
                ChatMessage(role: .system, content: entry["system"] as? String ?? ""),
                ChatMessage(role: .user, content: entry["user"] as? String ?? ""),
            ]
            let prompt = Gemma4ChatTemplate.encode(messages: messages, tokenizer: chat.gemmaTokenizer)
            let sampling = ChatSamplingConfig(
                temperature: Float(entry["temperature"] as? Double ?? 0.1), topK: 20, topP: 0.85,
                maxTokens: entry["max_tokens"] as? Int ?? 256, repetitionPenalty: 1.1)
            for _ in 0 ..< repeats {
                var tokens: [Int] = []
                var text = ""
                chat.decode(promptTokens: prompt, sampling: sampling,
                            onToken: { tokens.append($0) }, onText: { text += $0 })
                runs += 1
                if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { empty += 1 }
                if tokens.contains(Gemma4AnswerFilter.channelOpen) { opened += 1 }
            }
        }
        print("[replay] \(entries.count) prompts x \(repeats): \(runs) runs, \(empty) empty, \(opened) opened the reasoning channel")
    }
}
