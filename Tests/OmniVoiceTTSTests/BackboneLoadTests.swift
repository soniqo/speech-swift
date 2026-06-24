import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Gates the OmniVoice backbone port: the published `model.safetensors` must load
/// into `OmniVoiceModel` with `verify: .all` (so the `@ModuleInfo` keys match the
/// Qwen3 `llm.*` + audio embedding/head layout exactly), and a forward pass must
/// produce per-codebook logits of shape `[B, numCodebook, L, audioVocabSize]`.
///
/// Bundle: the local fp16 convert at `/tmp/omnivoice-mlx/model.safetensors`
/// (published as `aufklarer/OmniVoice-MLX-fp16`). Skips if not present.
final class BackboneLoadTests: XCTestCase {
    func testLoadAndForwardShape() throws {
        let bundle = "/tmp/omnivoice-mlx/model.safetensors"
        guard FileManager.default.fileExists(atPath: bundle) else {
            throw XCTSkip("missing \(bundle); run the OmniVoice converter")
        }
        let cfg = OmniVoiceConfig()
        let model = OmniVoiceModel(cfg)
        // verify: .all throws on any missing/extra/mismatched key — i.e. this is
        // the structural gate against the real checkpoint.
        try model.loadWeights(from: URL(fileURLWithPath: bundle))

        // Small synthetic forward: B=1, L=10, rows = 1 text + numCodebook audio.
        let b = 1, l = 10, rows = 1 + cfg.numAudioCodebook
        let ids = MLXArray((0 ..< b * rows * l).map { Int32($0 % 1000) }).reshaped([b, rows, l])
        let audioMask = MLXArray(Array(repeating: true, count: b * l)).reshaped([b, l])
        let logits = model(inputIds: ids, audioMask: audioMask)
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [b, cfg.numAudioCodebook, l, cfg.audioVocabSize])
    }
}
