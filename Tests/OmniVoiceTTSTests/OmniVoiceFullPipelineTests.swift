import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// The **full** text→speech pipeline, end to end in Swift with no oracle fixtures
/// in the loop — only a raw reference clip and the text go in:
///
///   ref wav ─► OmniVoiceCodecEncoder.encode ─► reference tokens [8, R]
///   text + refText + lang + refTokens ─► OmniVoiceInputBuilder.buildInputs ─► input_ids
///   input_ids ─► OmniVoiceModel.generateTokens (16-step diffusion) ─► audio tokens
///   audio tokens ─► OmniVoiceCodec.decode ─► 24 kHz waveform
///
/// Uses the max-precision fp32 weights. Writes the waveform to
/// `/tmp/omnivoice_golden/swift_true_e2e.f32` for an external ASR check (it must
/// transcribe back to the source text). This is the real product path; the
/// per-stage numeric gates in the sibling tests prove each box is exact.
final class OmniVoiceFullPipelineTests: XCTestCase {
    func testTextToSpeechEndToEnd() async throws {
        let weightsDir = "/tmp/omnivoice-fp32"
        let tokDir = "/tmp/omnivoice-mlx"  // tokenizer.json
        let dir = "/tmp/omnivoice_golden"
        let modelBundle = "\(weightsDir)/model.safetensors"
        let codecBundle = "\(weightsDir)/audio_tokenizer/model.safetensors"
        for p in [modelBundle, codecBundle, "\(tokDir)/tokenizer.json", "\(dir)/ref_wav_in.f32"]
        where !FileManager.default.fileExists(atPath: p) { throw XCTSkip("missing \(p)") }

        let text = "The quick brown fox jumps over the lazy dog."
        let refText = "Hello there, this is a cloned voice."

        // 1. Reference clip → reference audio tokens.
        let wavData = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/ref_wav_in.f32"))
        let wav = wavData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let refWav = MLXArray(wav).reshaped([1, 1, wav.count])
        let encoder = OmniVoiceCodecEncoder()
        try encoder.loadWeights(from: URL(fileURLWithPath: codecBundle))
        let refTokens = encoder.encode(refWav)  // [1, 8, R]
        MLX.eval(refTokens)
        let R = refTokens.dim(refTokens.ndim - 1)
        print("[OmniVoice] e2e: encoded reference → \(refTokens.shape) (R=\(R))")

        // 2. Text + reference → conditioning input ids.
        let tok = try await OmniVoiceTokenizer.load(from: URL(fileURLWithPath: tokDir))
        let builder = OmniVoiceInputBuilder(tokenizer: tok)
        let targetLen = 63  // pinned to match the reference clip's duration here
        let (inputIds, audioMask) = builder.buildInputs(
            text: text, refText: refText, lang: "en", refAudioTokens: refTokens,
            targetLen: targetLen, denoise: true, instruct: nil)
        MLX.eval(inputIds, audioMask)
        print("[OmniVoice] e2e: built inputs \(inputIds.shape)")

        // 3. Diffusion decode → audio tokens.
        let model = OmniVoiceModel()
        try model.loadWeights(from: URL(fileURLWithPath: modelBundle))
        let tokens = model.generateTokens(
            condInputIds: inputIds, audioMask: audioMask, targetLen: targetLen, numSteps: 16)
        MLX.eval(tokens)

        // 4. Codec decode → waveform.
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: codecBundle))
        let outWav = codec.decode(tokens)
        MLX.eval(outWav)
        let samples = outWav.asType(.float32).asArray(Float.self)

        XCTAssertEqual(samples.count, targetLen * 960)
        var sumSq = 0.0
        for s in samples { sumSq += Double(s) * Double(s) }
        let rms = (sumSq / Double(samples.count)).squareRoot()
        print("[OmniVoice] e2e: \(samples.count) samples, rms=\(rms)")
        XCTAssertGreaterThan(rms, 0.01, "e2e wav is (near) silent")

        let out = "\(dir)/swift_true_e2e.f32"
        samples.withUnsafeBytes { try? Data($0).write(to: URL(fileURLWithPath: out)) }
        print("[OmniVoice] e2e: wrote \(out)")
    }
}
