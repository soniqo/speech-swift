import Foundation
import MLX
import XCTest

@testable import OmniVoiceTTS

/// Gates the OmniVoice codec **encoder** port (Higgs-audio v2 tokenizer encode
/// path): 24 kHz reference waveform -> 8-codebook audio tokens, gated stage by
/// stage against the torch oracle goldens under `/tmp/omnivoice_golden/`.
///
/// Bundle: `/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors`. Skips if the
/// bundle or goldens are missing.
final class CodecEncodeTests: XCTestCase {
    static let bundle = "/tmp/omnivoice-mlx/audio_tokenizer/model.safetensors"
    static let dir = "/tmp/omnivoice_golden"

    func f32(_ name: String) throws -> [Float] {
        let d = try Data(contentsOf: URL(fileURLWithPath: "\(Self.dir)/\(name)"))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    func i32(_ name: String) throws -> [Int32] {
        let d = try Data(contentsOf: URL(fileURLWithPath: "\(Self.dir)/\(name)"))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
    func npyF32(_ name: String) throws -> [Float] {
        // Minimal .npy reader for 1-D float32 arrays (header is ASCII, data raw).
        let d = try Data(contentsOf: URL(fileURLWithPath: "\(Self.dir)/\(name)"))
        // header length is at bytes 8..10 (little-endian uint16), data after.
        let headerLen = Int(d[8]) | (Int(d[9]) << 8)
        let dataStart = 10 + headerLen
        let body = d.subdata(in: dataStart ..< d.count)
        return body.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func cosine(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    func loadEncoder() throws -> OmniVoiceCodecEncoder {
        let enc = OmniVoiceCodecEncoder()
        try enc.loadWeights(from: URL(fileURLWithPath: Self.bundle))
        return enc
    }

    func skipIfMissing(_ extra: [String] = []) throws {
        let paths = [Self.bundle] + extra.map({ "\(Self.dir)/\($0)" })
        for p in paths where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("missing \(p); run the OmniVoice codec encode golden capture")
        }
    }

    /// Gate 3: DAC acoustic encoder from the raw 24 kHz ref wav -> [1, 256, 52].
    func testAcousticEncoderGate3() throws {
        try skipIfMissing(["ref_wav_in.npy", "enc_e_acoustic.f32"])
        let enc = try loadEncoder()
        let wav = try npyF32("ref_wav_in.npy")  // [49920]
        let x = MLXArray(wav).reshaped([1, 1, wav.count])
        let out = enc.encodeAcoustic(x, semanticT: 52)  // [1, 256, 52]
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 256, 52])
        let got = out.asType(.float32).asArray(Float.self)
        let gold = try f32("enc_e_acoustic.f32")
        let c = cosine(got, gold)
        print("[OmniVoice] enc Gate3 acoustic cosine = \(c)")
        XCTAssertGreaterThan(c, 0.99, "acoustic encoder cosine=\(c)")
    }

    /// Gate 2: encoder_semantic from golden semantic feats -> [1, 768, 52].
    func testEncoderSemanticGate2() throws {
        try skipIfMissing(["enc_semantic_feats.f32", "enc_e_semantic.f32"])
        let enc = try loadEncoder()
        let feats = try f32("enc_semantic_feats.f32")  // [1, 52, 768]
        let x = MLXArray(feats).reshaped([1, 52, 768])
        let out = enc.encodeSemantic(x)  // [1, 768, 52]
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 768, 52])
        let got = out.asType(.float32).asArray(Float.self)
        let gold = try f32("enc_e_semantic.f32")
        let c = cosine(got, gold)
        print("[OmniVoice] enc Gate2 encoder_semantic cosine = \(c)")
        XCTAssertGreaterThan(c, 0.99, "encoder_semantic cosine=\(c)")
    }

    /// Gate 4: fc + cat from golden acoustic + semantic -> [1, 1024, 52].
    func testFuseEmbeddingsGate4() throws {
        try skipIfMissing(["enc_e_acoustic.f32", "enc_e_semantic.f32", "enc_embeddings.f32"])
        let enc = try loadEncoder()
        let ac = MLXArray(try f32("enc_e_acoustic.f32")).reshaped([1, 256, 52])
        let se = MLXArray(try f32("enc_e_semantic.f32")).reshaped([1, 768, 52])
        let out = enc.fuseEmbeddings(acoustic: ac, semantic: se)  // [1, 1024, 52]
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 1024, 52])
        let got = out.asType(.float32).asArray(Float.self)
        let gold = try f32("enc_embeddings.f32")
        let c = cosine(got, gold)
        print("[OmniVoice] enc Gate4 embeddings cosine = \(c)")
        XCTAssertGreaterThan(c, 0.99, "fuse embeddings cosine=\(c)")
    }

    /// RVQ token match given golden embeddings -> [1, 8, 52] codes.
    func testQuantizeFromGoldenEmbeddings() throws {
        try skipIfMissing(["enc_embeddings.f32", "enc_audio_codes.i32"])
        let enc = try loadEncoder()
        let emb = MLXArray(try f32("enc_embeddings.f32")).reshaped([1, 1024, 52])
        let codes = enc.quantize(emb)  // [1, 8, 52]
        MLX.eval(codes)
        XCTAssertEqual(codes.shape, [1, 8, 52])
        let got = codes.asType(.int32).asArray(Int32.self)
        let gold = try i32("enc_audio_codes.i32")
        let match = tokenMatch(got, gold)
        print("[OmniVoice] enc RVQ-from-golden-embeddings token match = \(match)")
        XCTAssertGreaterThan(match, 0.95, "RVQ token match=\(match)")
    }

    /// Gate 1: HuBERT mean-hidden features from the 16k padded golden -> [1,52,768].
    func testSemanticFeaturesGate1() throws {
        try skipIfMissing(["enc_resampled16k_padded.f32", "enc_semantic_feats.f32"])
        let enc = try loadEncoder()
        let padded = try f32("enc_resampled16k_padded.f32")  // [1, 33600]
        let x = MLXArray(padded).reshaped([1, padded.count])
        let out = enc.semanticFeatures16k(x)  // [1, 52, 768]
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 52, 768])
        let got = out.asType(.float32).asArray(Float.self)
        let gold = try f32("enc_semantic_feats.f32")
        let c = cosine(got, gold)
        print("[OmniVoice] enc Gate1 HuBERT semantic feats cosine = \(c)")
        XCTAssertGreaterThan(c, 0.99, "HuBERT semantic feats cosine=\(c)")
    }

    /// FINAL gate: full encode from the 24 kHz ref wav -> [1, 8, 52] codes,
    /// including the 24k->16k sinc resample. Token match >= 95%.
    func testFullEncodeTokenMatch() throws {
        try skipIfMissing(["ref_wav_in.npy", "enc_audio_codes.i32"])
        let enc = try loadEncoder()
        let wav = try npyF32("ref_wav_in.npy")  // [49920]
        let x = MLXArray(wav).reshaped([1, 1, wav.count])
        let codes = enc.encode(x)  // [1, 8, 52]
        MLX.eval(codes)
        XCTAssertEqual(codes.shape, [1, 8, 52])
        let got = codes.asType(.int32).asArray(Int32.self)
        let gold = try i32("enc_audio_codes.i32")
        let match = tokenMatch(got, gold)
        print("[OmniVoice] enc FINAL full-encode token match = \(match)")
        XCTAssertGreaterThan(match, 0.95, "full encode token match=\(match)")
    }

    private func tokenMatch(_ a: [Int32], _ b: [Int32]) -> Double {
        let n = min(a.count, b.count)
        guard n > 0 else { return 0 }
        var hit = 0
        for i in 0 ..< n where a[i] == b[i] { hit += 1 }
        return Double(hit) / Double(n)
    }
}
