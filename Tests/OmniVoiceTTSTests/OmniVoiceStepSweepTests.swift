import Foundation
import MLX
import MLXNN
import XCTest

@testable import OmniVoiceTTS

/// Sweeps the diffusion step count on the int8 backbone to find the cheapest
/// setting that still produces intelligible audio. Fewer steps is a linear
/// speedup that stacks with int8. Writes one waveform per step count for an
/// external ASR check.
final class OmniVoiceStepSweepTests: XCTestCase {
    func testStepCountSweep() throws {
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

        let model = OmniVoiceModel(cfg)
        try model.loadWeights(from: URL(fileURLWithPath: bundle))
        quantize(model: model, groupSize: 64, bits: 8)
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))

        for steps in [16, 12, 10, 8] {
            // warm + timed
            var tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: T, numSteps: steps)
            MLX.eval(tokens)
            let t0 = DispatchTime.now().uptimeNanoseconds
            tokens = model.generateTokens(condInputIds: cond, audioMask: mask, targetLen: T, numSteps: steps)
            MLX.eval(tokens)
            let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            let wav = codec.decode(tokens)
            MLX.eval(wav)
            let samples = wav.asType(.float32).asArray(Float.self)
            samples.withUnsafeBytes {
                try? Data($0).write(to: URL(fileURLWithPath: "\(dir)/swift_int8_s\(steps).f32"))
            }
            print(String(format: "[OmniVoice] int8 %2d steps: %.3fs (RTF %.3f) → swift_int8_s%d.f32",
                steps, dt, dt / audioSeconds, steps))
        }
    }
}
