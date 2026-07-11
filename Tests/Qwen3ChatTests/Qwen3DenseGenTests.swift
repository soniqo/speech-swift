import XCTest
@testable import Qwen3Chat

/// End-to-end generation through the hand-written Qwen3 dense backend: chat template + tokenizer +
/// the optimised sampler must produce coherent text (the forward pass is already parity-verified).
final class Qwen3DenseGenTests: XCTestCase {
    func testGeneratesCoherentReply() async throws {
        let chat: Qwen3DenseChat
        do {
            chat = try await Qwen3DenseChat.fromPretrained(modelId: "aufklarer/Qwen3-4B-Instruct-2507-MLX-4bit")
        } catch {
            throw XCTSkip("model unavailable (download/metallib): \(error)")
        }
        let sampling = ChatSamplingConfig(temperature: 0.5, topK: 50, topP: 0.9, maxTokens: 40, repetitionPenalty: 1.1)

        for prompt in ["What is the capital of France?", "Say hello in one short sentence."] {
            var reply = ""
            for try await chunk in chat.generateStream(
                messages: [ChatMessage(role: .user, content: prompt)], sampling: sampling) {
                reply += chunk
            }
            reply = reply.trimmingCharacters(in: .whitespacesAndNewlines)
            print("[dense-gen] «\(prompt)» → «\(reply)»")
            XCTAssertFalse(reply.isEmpty, "should generate text")
            XCTAssertFalse(reply.contains("\u{FFFD}"), "no mojibake")
            XCTAssertFalse(reply.contains("<|"), "no leaked special tokens")
        }
    }
}
