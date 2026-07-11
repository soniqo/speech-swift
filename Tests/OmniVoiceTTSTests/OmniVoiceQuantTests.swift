import Foundation
import MLX
import MLXNN
import XCTest

@testable import OmniVoiceTTS

/// Measures the int8 lever for RTF: load fp32 weights, quantize the backbone
/// Linears to 8-bit in place, and compare generate latency + audio quality
/// against the fp32 baseline. The codec stays fp32 (it runs once and is cheap).
/// Writes the int8 waveform for an external ASR check.
final class OmniVoiceQuantTests: XCTestCase {
    func testInt8RTFAndQuality() throws {
        let bundle = "/tmp/omnivoice-fp32/model.safetensors"
        let codecBundle = "/tmp/omnivoice-fp32/audio_tokenizer/model.safetensors"
        let dir = "/tmp/omnivoice_golden"
        for p in [bundle, codecBundle, "\(dir)/det16_input_ids.i32"]
        where !FileManager.default.fileExists(atPath: p) { throw XCTSkip("missing \(p)") }
        func i32(_ n: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(n)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }

        let cfg = OmniVoiceConfig()
        let C = cfg.numAudioCodebook, condLen = 143, T = 63
        let cond = MLXArray(try i32("det16_input_ids.i32")).reshaped([2, C, condLen])[0 ..< 1]
        let mask = MLXArray(try i32("det16_audio_mask.i32")).reshaped([2, condLen])[0 ..< 1]
        let audioSeconds = Double(T * 960) / 24000.0

        func timeGenerate(_ model: OmniVoiceModel, runs: Int) -> (Double, MLXArray) {
            // warmup
            var tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: T, numSteps: 16)
            MLX.eval(tokens)
            var best = Double.greatestFiniteMagnitude
            for _ in 0 ..< runs {
                let t0 = DispatchTime.now().uptimeNanoseconds
                tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: T, numSteps: 16)
                MLX.eval(tokens)
                let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
                best = min(best, dt)
            }
            return (best, tokens)
        }

        // fp32 baseline.
        let m32 = OmniVoiceModel(cfg)
        try m32.loadWeights(from: URL(fileURLWithPath: bundle))
        let (t32, _) = timeGenerate(m32, runs: 3)

        // int8: load fp32 then quantize the Linears in place (8-bit, group 64).
        let m8 = OmniVoiceModel(cfg)
        try m8.loadWeights(from: URL(fileURLWithPath: bundle))
        quantize(model: m8, groupSize: 64, bits: 8)
        let (t8, tokens8) = timeGenerate(m8, runs: 3)

        print(String(
            format: "[OmniVoice] RTF — fp32: %.3fs (RTF %.3f) | int8: %.3fs (RTF %.3f) | speedup %.2fx",
            t32, t32 / audioSeconds, t8, t8 / audioSeconds, t32 / t8))

        // int8 audio → write for ASR.
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))
        let wav = codec.decode(tokens8)
        MLX.eval(wav)
        let samples = wav.asType(.float32).asArray(Float.self)
        var sumSq = 0.0
        for s in samples { sumSq += Double(s) * Double(s) }
        print(String(format: "[OmniVoice] int8 wav: %d samples rms=%.4f", samples.count, (sumSq / Double(samples.count)).squareRoot()))
        let out = "\(dir)/swift_int8_e2e.f32"
        samples.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: out)) }
        print("[OmniVoice] wrote \(out)")
        XCTAssertLessThan(t8, t32 * 1.05, "int8 should not be slower than fp32")
    }
}
