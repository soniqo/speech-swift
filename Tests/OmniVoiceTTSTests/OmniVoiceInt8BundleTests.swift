import Foundation
import MLX
import MLXNN
import XCTest

@testable import OmniVoiceTTS

/// Produces and round-trips a published-shaped int8 backbone bundle: quantize the
/// fp32 backbone to 8-bit, save the quantized parameters to safetensors, then load
/// them back into a freshly-quantized model and confirm generation still works.
/// This is the artifact for `aufklarer/OmniVoice-MLX-int8` (the codec stays fp16 —
/// it runs once and is already exact there). Writes the reload's waveform for ASR.
final class OmniVoiceInt8BundleTests: XCTestCase {
    func testQuantizeSaveReload() throws {
        let fp32 = "/tmp/omnivoice-fp32/model.safetensors"
        let codecBundle = "/tmp/omnivoice-fp32/audio_tokenizer/model.safetensors"
        let dir = "/tmp/omnivoice_golden"
        let outDir = "/tmp/omnivoice-int8"
        for p in [fp32, codecBundle, "\(dir)/det16_input_ids.i32"]
        where !FileManager.default.fileExists(atPath: p) { throw XCTSkip("missing \(p)") }
        func i32(_ n: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(n)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

        let cfg = OmniVoiceConfig()
        let groupSize = 64, bits = 8

        // 1. Quantize the fp32 backbone and save its parameters.
        let m = OmniVoiceModel(cfg)
        try m.loadWeights(from: URL(fileURLWithPath: fp32))
        quantize(model: m, groupSize: groupSize, bits: bits)
        var params: [String: MLXArray] = [:]
        for (k, v) in m.parameters().flattened() { params[k] = v }
        let outPath = "\(outDir)/model.safetensors"
        try MLX.save(arrays: params, url: URL(fileURLWithPath: outPath))

        let fp32Size = (try FileManager.default.attributesOfItem(atPath: fp32)[.size] as? Int) ?? 0
        let int8Size = (try FileManager.default.attributesOfItem(atPath: outPath)[.size] as? Int) ?? 0
        print(String(format: "[OmniVoice] int8 bundle: %.0f MB (fp32 %.0f MB, %.2fx smaller)",
            Double(int8Size) / 1e6, Double(fp32Size) / 1e6, Double(fp32Size) / Double(max(int8Size, 1))))

        // 2. Reload into a fresh quantized model (swap structure, then load params).
        let C = cfg.numAudioCodebook, condLen = 143, T = 63
        let cond = MLXArray(try i32("det16_input_ids.i32")).reshaped([2, C, condLen])[0 ..< 1]
        let mask = MLXArray(try i32("det16_audio_mask.i32")).reshaped([2, condLen])[0 ..< 1]

        // Reload through the real published-bundle path (quantize-then-load).
        let m2 = OmniVoiceModel(cfg)
        try m2.loadWeights(
            from: URL(fileURLWithPath: outPath), quantization: (groupSize, bits))

        let tokens = m2.generateTokens(condInputIds: cond, audioMask: mask, targetLen: T, numSteps: 16)
        MLX.eval(tokens)

        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))
        let wav = codec.decode(tokens)
        MLX.eval(wav)
        let samples = wav.asType(.float32).asArray(Float.self)
        XCTAssertEqual(samples.count, T * 960)
        var sumSq = 0.0
        for s in samples { sumSq += Double(s) * Double(s) }
        let rms = (sumSq / Double(samples.count)).squareRoot()
        print(String(format: "[OmniVoice] int8 reload wav: %d samples rms=%.4f", samples.count, rms))
        XCTAssertGreaterThan(rms, 0.01, "int8 reload produced (near) silence")
        let out = "\(dir)/swift_int8_reload.f32"
        samples.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: out)) }
        print("[OmniVoice] wrote \(out)")
    }
}
