import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Measures the steady-state synthesis resident set for one precision variant
/// (backbone + codec decoder — the per-line path; the encoder is a one-time clone
/// cost). Pick the variant with `SONIQO_RSS_VARIANT` = fp32 | fp16 | int8 and run
/// under `/usr/bin/time -l` to read "maximum resident set size".
final class OmniVoiceRSSTests: XCTestCase {
    func testResidentSet() throws {
        let variant = ProcessInfo.processInfo.environment["SONIQO_RSS_VARIANT"] ?? "int8"
        let codecBundle = "/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors"  // fp16 codec, shipped
        let dir = "/tmp/omnivoice_golden"

        let (bundle, quant): (String, (Int, Int)?)
        switch variant {
        case "fp32": (bundle, quant) = ("/tmp/omnivoice-fp32/model.safetensors", nil)
        case "fp16": (bundle, quant) = ("/tmp/omnivoice-mlx/model.safetensors", nil)
        default: (bundle, quant) = ("/tmp/omnivoice-int8/model.safetensors", (64, 8))
        }
        for p in [bundle, codecBundle, "\(dir)/det16_input_ids.i32"]
        where !FileManager.default.fileExists(atPath: p) { throw XCTSkip("missing \(p)") }
        func i32(_ n: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(n)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }

        let cfg = OmniVoiceConfig()
        let C = cfg.numAudioCodebook
        let cond = MLXArray(try i32("det16_input_ids.i32")).reshaped([2, C, 143])[0 ..< 1]
        let mask = MLXArray(try i32("det16_audio_mask.i32")).reshaped([2, 143])[0 ..< 1]

        let model = OmniVoiceModel(cfg)
        try model.loadWeights(from: URL(fileURLWithPath: bundle),
            quantization: quant.map { (groupSize: $0.0, bits: $0.1) })
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))

        let tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: 63, numSteps: 16)
        let wav = codec.decode(tokens)
        MLX.eval(wav)
        print("[OmniVoice] RSS variant=\(variant): synthesis ran (\(wav.dim(0)) samples)")
    }
}
