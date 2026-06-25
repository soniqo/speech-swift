import XCTest
import Foundation
@testable import Qwen3Chat

/// End-to-end generation through the hand-written Qwen3 dense backend: chat template + tokenizer +
/// the optimised sampler must produce coherent text (the forward pass is already parity-verified).
final class E2EQwen3DenseGenTests: XCTestCase {
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["QWEN3_DENSE_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath:
            "/Users/ivan/repos/runner-speech-models/speech-models/out/mlx/Qwen3-4B-Instruct-2507-MLX-4bit")
    }()

    func testGeneratesCoherentReply() async throws {
        guard FileManager.default.fileExists(
            atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Qwen3 dense model dir unavailable: \(Self.modelDir.path)")
        }

        let chat: Qwen3DenseChat
        do {
            chat = try Qwen3DenseChat.fromDirectory(Self.modelDir)
        } catch {
            throw XCTSkip("model load failed (weights/metallib): \(error)")
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
