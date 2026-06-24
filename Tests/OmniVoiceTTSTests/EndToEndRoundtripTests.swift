import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// End-to-end roundtrip of the ported pipeline: golden conditioning input ids
/// (style + text + reference, target all-mask) -> `OmniVoiceModel.generateTokens`
/// (16-step diffusion) -> `OmniVoiceCodec.decode` (Higgs-audio v2 decoder, 960x
/// upsample) -> 24 kHz waveform. The waveform is written to disk so an external
/// ASR pass can confirm it transcribes back to the source text
/// ("The quick brown fox jumps over the lazy dog.").
///
/// This exercises the two heavyweight stages (the bidirectional Qwen3 backbone
/// run as a discrete-diffusion sampler, and the RVQ + DAC decoder) together on
/// real weights — the numeric gates in the sibling tests check each stage against
/// the torch oracle; this one checks they compose into intelligible audio.
///
/// Bundles: `/tmp/omnivoice-mlx/model.safetensors` (backbone + audio I/O) and
/// `/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors` (codec). Golden input
/// under `/tmp/omnivoice_golden/`. Skips if any is missing. Output wav is written
/// to `/tmp/omnivoice_golden/swift_e2e.f32` (Float32 raw @ 24 kHz).
final class EndToEndRoundtripTests: XCTestCase {
    func testGenerateThenDecodeProducesAudio() throws {
        let modelBundle = "/tmp/omnivoice-mlx/model.safetensors"
        let codecBundle = "/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors"
        let dir = "/tmp/omnivoice_golden"
        for p in [modelBundle, codecBundle, "\(dir)/det_input_ids.i32", "\(dir)/det_audio_mask.i32"]
        where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run the OmniVoice golden capture")
        }
        func i32(_ name: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }

        let cfg = OmniVoiceConfig()
        let C = cfg.numAudioCodebook, condLen = 143, targetLen = 63
        // Conditioning input is captured as a [2, C, condLen] CFG pair; row 0 is
        // the conditional branch (style + text + reference + masked target).
        let cond = MLXArray(try i32("det_input_ids.i32")).reshaped([2, C, condLen])[0 ..< 1]
        let mask = MLXArray(try i32("det_audio_mask.i32")).reshaped([2, condLen])[0 ..< 1]

        // Stage 1: diffusion decode -> audio token matrix [1, C, targetLen].
        let model = OmniVoiceModel(cfg)
        try model.loadWeights(from: URL(fileURLWithPath: modelBundle))
        let tokens = model.generateTokens(
            condInputIds: cond, audioMask: mask, targetLen: targetLen, numSteps: 16)
        MLX.eval(tokens)
        XCTAssertEqual(tokens.shape, [1, C, targetLen])

        // Stage 2: codec decode -> waveform.
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))
        let wav = codec.decode(tokens)
        MLX.eval(wav)
        let samples = wav.asType(.float32).asArray(Float.self)

        XCTAssertEqual(samples.count, targetLen * 960)
        // Sanity: the audio must be non-silent (RMS well above the noise floor).
        var sumSq = 0.0
        for s in samples { sumSq += Double(s) * Double(s) }
        let rms = (sumSq / Double(samples.count)).squareRoot()
        print("[OmniVoice] e2e wav: \(samples.count) samples, rms=\(rms)")
        XCTAssertGreaterThan(rms, 0.01, "e2e wav is (near) silent, rms=\(rms)")

        // Persist for the external ASR roundtrip check.
        let out = "\(dir)/swift_e2e.f32"
        samples.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: out)) }
        print("[OmniVoice] wrote \(out)")
    }
}
