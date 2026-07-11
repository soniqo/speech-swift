import XCTest
@testable import MAGNeTMusicGen

final class MAGNeTConfigTests: XCTestCase {
    func testDecodeMediumInt8Config() throws {
        let json = """
        {
          "n_q": 4, "card": 2048, "dim": 1536,
          "num_heads": 24, "num_layers": 48, "ffn_dim": 6144,
          "segment_duration": 30, "frame_rate": 50,
          "subcodes_context": 5, "span_len": 3,
          "sample_rate": 32000,
          "t5_name": "t5-base", "t5_dim": 768,
          "encodec_name": "encodec-32khz",
          "quantization": {
            "mode": "mlx_affine", "bits": 8, "group_size": 64,
            "targets": ["self_attn.q_proj.weight"]
          },
          "format": "int8"
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MAGNeTConfig.self, from: json)
        XCTAssertEqual(cfg.nQ, 4)
        XCTAssertEqual(cfg.card, 2048)
        XCTAssertEqual(cfg.dim, 1536)
        XCTAssertEqual(cfg.numHeads, 24)
        XCTAssertEqual(cfg.numLayers, 48)
        XCTAssertEqual(cfg.ffnDim, 6144)
        XCTAssertEqual(cfg.seqLen, 1500)
        XCTAssertEqual(cfg.maskTokenId, 2048)
        XCTAssertEqual(cfg.quantization?.bits, 8)
        XCTAssertEqual(cfg.quantization?.groupSize, 64)
    }

    func testVariantRepoIds() {
        XCTAssertEqual(MAGNeTVariant.smallInt4.huggingFaceRepoId,
                       "aufklarer/MAGNeT-Small-30secs-MLX-4bit")
        XCTAssertEqual(MAGNeTVariant.smallInt8.huggingFaceRepoId,
                       "aufklarer/MAGNeT-Small-30secs-MLX-8bit")
        XCTAssertEqual(MAGNeTVariant.mediumInt4.huggingFaceRepoId,
                       "aufklarer/MAGNeT-Medium-30secs-MLX-4bit")
        XCTAssertEqual(MAGNeTVariant.mediumInt8.huggingFaceRepoId,
                       "aufklarer/MAGNeT-Medium-30secs-MLX-8bit")
        XCTAssertEqual(MAGNeTVariant.smallInt4.bits, 4)
        XCTAssertEqual(MAGNeTVariant.mediumInt8.bits, 8)
    }
}
