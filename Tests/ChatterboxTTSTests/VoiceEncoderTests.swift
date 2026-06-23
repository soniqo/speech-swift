import MLX
import MLXNN
import XCTest

@testable import ChatterboxTTS

/// End-to-end numeric gate for the VoiceEncoder: mel → 3-layer LSTM → proj →
/// ReLU → partial-window mean → L2, validated against the golden 256-d speaker
/// embedding from the reference on the SAME prepared 16 kHz input (so
/// resample/trim are isolated out). Weights come from a small ve-only
/// safetensors to keep the test off the 1.3 GB bundle.
final class VoiceEncoderTests: XCTestCase {
    private let weightsPath = "/tmp/cbx_ve.safetensors"
    private let inputPath = "/tmp/cbx_ve_input16k.f32"
    private let goldenPath = "/tmp/cbx_ve_embed.f32"

    private func readF32(_ path: String) throws -> [Float] {
        let d = try Data(contentsOf: URL(fileURLWithPath: path))
        return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    func testEmbeddingMatchesOracle() throws {
        for p in [weightsPath, inputPath, goldenPath] where !FileManager.default.fileExists(atPath: p) {
            throw XCTSkip("golden \(p) not present; run /tmp/cbx_ve_golden.py")
        }

        let raw = try MLX.loadArrays(url: URL(fileURLWithPath: weightsPath))
        let weights = raw.mapValues { $0.asType(.float32) }
        let ve = ChatterboxVoiceEncoder()
        try ve.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        MLX.eval(ve.parameters())

        let samples = try readF32(inputPath)
        XCTAssertEqual(samples.count, 51200)
        let emb = ve.embed(samples: samples)
        MLX.eval(emb)
        let got = emb.asArray(Float.self)
        XCTAssertEqual(got.count, 256)

        let gold = try readF32(goldenPath)
        XCTAssertEqual(gold.count, 256)

        // Both are L2-normalised, so the dot product is the cosine similarity.
        var dot: Float = 0
        for i in 0 ..< 256 { dot += got[i] * gold[i] }
        XCTAssertGreaterThan(dot, 0.9999, "speaker-embedding cosine = \(dot)")
    }
}
