import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Gates the OmniVoice codec **decoder** port (Higgs-audio v2 tokenizer decode
/// path): audio codes `[1, 8, 63]` -> 24 kHz waveform of `63 * 960 = 60480`
/// samples, compared to the torch oracle by cosine similarity.
///
/// Bundle: `/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors`. Golden inputs/
/// outputs under `/tmp/omnivoice_golden/`. Skips if either is missing.
final class CodecDecodeTests: XCTestCase {
    func testDecodeMatchesOracle() throws {
        let bundle = "/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors"
        let dir = "/tmp/omnivoice_golden"
        for p in [bundle, "\(dir)/det_final_tokens.i32", "\(dir)/det_decode_wav.f32"]
        where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run the OmniVoice codec golden capture")
        }
        func i32(_ name: String) throws -> [Int32] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
        }
        func f32(_ name: String) throws -> [Float] {
            let d = try Data(contentsOf: URL(fileURLWithPath: "\(dir)/\(name)"))
            return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }

        let codesFlat = try i32("det_final_tokens.i32")  // [1, 8, 63]
        let T = 63, C = 8
        XCTAssertEqual(codesFlat.count, C * T)
        let codes = MLXArray(codesFlat).reshaped([1, C, T])
        let gold = try f32("det_decode_wav.f32")  // 60480 samples

        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: URL(fileURLWithPath: bundle))
        let wav = codec.decode(codes)
        MLX.eval(wav)
        let got = wav.asType(.float32).asArray(Float.self)

        XCTAssertEqual(got.count, T * 960, "expected \(T * 960) samples, got \(got.count)")
        XCTAssertEqual(gold.count, T * 960, "golden has \(gold.count) samples")

        var dot = 0.0, ng = 0.0, na = 0.0
        for i in 0 ..< min(got.count, gold.count) {
            dot += Double(got[i]) * Double(gold[i])
            na += Double(got[i]) * Double(got[i])
            ng += Double(gold[i]) * Double(gold[i])
        }
        let cosine = dot / (na.squareRoot() * ng.squareRoot())
        print("[OmniVoice] codec decode cosine = \(cosine)")
        XCTAssertGreaterThan(cosine, 0.95, "codec decode cosine=\(cosine)")
    }
}
