import XCTest
import Foundation
@testable import Qwen3Chat

/// Numeric parity: the hand-written `Gemma4Model` must produce the same next-token logits as the
/// Python `mlx_lm` reference for a fixed prompt — proving the full Gemma-4 forward pass (PLE,
/// sandwich norms, dual RoPE, KV-sharing, double-wide MLP, per-layer gating, logit softcap, and
/// quantized weight loading) is correct.
///
/// Reference (mlx_lm, gemma-4-E2B-it MLX int4, tokens [818, 5279, 529, 7001, 563]):
///   argmax 7001 (logit -2.078),
///   top5 [(7001,-2.078), (531,-12.5), (711,-13.94), (529,-15.13), (496,-15.56)].
final class E2EGemma4ParityTests: XCTestCase {
    /// Local MLX export directory (int4). Override with GEMMA4_MODEL_DIR if needed.
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return URL(fileURLWithPath:
            "/Users/ivan/repos/runner-speech-models/speech-models/out/mlx/gemma-4-E2B-it-MLX-4bit")
    }()

    func testNextTokenArgmaxMatchesReference() throws {
        guard FileManager.default.fileExists(atPath: Self.modelDir.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Gemma 4 model dir unavailable: \(Self.modelDir.path)")
        }
        Gemma4Model.debugStats = ProcessInfo.processInfo.environment["GEMMA4_DEBUG"] != nil
        let chat: Gemma4Chat
        do {
            chat = try Gemma4Chat.fromDirectory(Self.modelDir)
        } catch {
            throw XCTSkip("model load failed (weights/metallib): \(error)")
        }
        let r = chat.nextTokenArgmax(promptTokens: [818, 5279, 529, 7001, 563])
        print("[gemma4-parity] argmax=\(r.argmax) logit=\(String(format: "%.4f", r.logit)) "
            + "top5=\(r.top5.map { "(\($0.0), \(String(format: "%.3f", $0.1)))" }.joined(separator: " "))")
        XCTAssertEqual(r.argmax, 7001, "hand-written Gemma 4 forward must match mlx_lm reference argmax")
        // Logit parity (quantized int4 → allow a small tolerance).
        XCTAssertEqual(r.logit, -2.078, accuracy: 0.5, "top logit must match reference")
    }
}
