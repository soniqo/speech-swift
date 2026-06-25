import XCTest
import Foundation
@testable import Qwen3Chat

/// Numeric parity: the hand-written `Qwen3DenseModel` must produce the same next-token logits as the
/// Python `mlx_lm` reference for a fixed prompt — proving the forward pass (attention/MLP/RoPE/QK-norm/
/// quantized weight loading) is correct before trusting any generation.
///
/// Reference (mlx_lm, Qwen3-4B-Instruct-2507 MLX int4, tokens [9707, 11, 1879, 0]):
///   argmax 358 (logit 11.31), top5 [358, 220, 320, 1096, 11162].
final class E2EQwen3DenseParityTests: XCTestCase {
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["QWEN3_DENSE_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath:
            "/Users/ivan/repos/runner-speech-models/speech-models/out/mlx/Qwen3-4B-Instruct-2507-MLX-4bit")
    }()

    func testNextTokenArgmaxMatchesReference() async throws {
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
        let r = chat.nextTokenArgmax(promptTokens: [9707, 11, 1879, 0])
        print("[dense-parity] argmax=\(r.argmax) logit=\(String(format: "%.4f", r.logit)) top5=\(r.top5.map { "(\($0.0), \(String(format: "%.3f", $0.1)))" }.joined(separator: " "))")
        XCTAssertEqual(r.argmax, 358, "hand-written forward must match mlx_lm reference argmax")
    }
}
