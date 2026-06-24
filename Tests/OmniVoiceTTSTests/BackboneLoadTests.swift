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

        // Small synthetic forward: B=1, L=10, rows = numCodebook (8).
        let b = 1, l = 10, rows = cfg.numAudioCodebook
        let ids = MLXArray((0 ..< b * rows * l).map { Int32($0 % 1000) }).reshaped([b, rows, l])
        let audioMask = MLXArray(Array(repeating: Int32(1), count: b * l)).reshaped([b, l])
        let logits = model(inputIds: ids, audioMask: audioMask)
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [b, cfg.numAudioCodebook, l, cfg.audioVocabSize])
    }

    /// Numeric gate: the first-step forward must match the oracle's `first_logits`.
    /// Loads the exact captured inputs (`fwd_input_ids` [2,8,143] / `fwd_audio_mask`
    /// [2,143]) and the golden logits ([2,8,143,1025]) dumped from the torch model.
    func testFirstLogitsMatchesOracle() throws {
        let dir = "/tmp/omnivoice_golden"
        let bundle = "/tmp/omnivoice-mlx/model.safetensors"
        for p in [bundle, "\(dir)/fwd_input_ids.i32", "\(dir)/first_logits.f32"]
        where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run the OmniVoice golden capture")
        }
        func i32(_ name: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        func f32(_ name: String) throws -> [Float] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        let cfg = OmniVoiceConfig()
        let B = 2, L = 143, C = cfg.numAudioCodebook, V = cfg.audioVocabSize
        let ids = MLXArray(try i32("fwd_input_ids.i32")).reshaped([B, C, L])
        let mask = MLXArray(try i32("fwd_audio_mask.i32")).reshaped([B, L])
        let gold = try f32("first_logits.f32")

        let model = OmniVoiceModel(cfg)
        try model.loadWeights(from: URL(fileURLWithPath: bundle))
        let logits = model(inputIds: ids, audioMask: mask)
        MLX.eval(logits)
        XCTAssertEqual(logits.shape, [B, C, L, V])
        let got = logits.asType(.float32).reshaped([gold.count]).asArray(Float.self)

        var dot = 0.0, ng = 0.0, na = 0.0
        for i in 0 ..< gold.count {
            dot += Double(got[i]) * Double(gold[i])
            na += Double(got[i]) * Double(got[i])
            ng += Double(gold[i]) * Double(gold[i])
        }
        let cosine = dot / (na.squareRoot() * ng.squareRoot())
        print("[OmniVoice] first_logits cosine = \(cosine)")
        // fp16 bundle vs fp32 oracle weights → expect ~0.999+, not exact.
        XCTAssertGreaterThan(cosine, 0.99, "first_logits cosine=\(cosine)")
    }
}
