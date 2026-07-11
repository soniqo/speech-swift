import XCTest
import MLX
@testable import MAGNeTMusicGen

final class MAGNeTUtilitiesTests: XCTestCase {
    func testSinEmbeddingShape() {
        let emb = magnetSinEmbedding(seqLen: 100, dim: 128)
        XCTAssertEqual(emb.shape, [1, 100, 128])
        // First half is cos (cos(0)=1 at position 0); second half is sin (sin(0)=0).
        let row0 = emb[0, 0, 0...].asArray(Float.self)
        XCTAssertEqual(row0[0], 1.0, accuracy: 1e-5,
                       "cos(0) should be 1 at pos 0")
        XCTAssertEqual(row0[64], 0.0, accuracy: 1e-5,
                       "sin(0) should be 0 at pos 0")
    }

    func testPositionsToMask() {
        // Place indices {0, 3} into an N=8 mask.
        let indices = MLXArray([Int32(0), Int32(3)]).reshaped([1, 1, 2])
        let mask = positionsToMask(indices: indices, N: 8)
        XCTAssertEqual(mask.shape, [1, 1, 8])
        let values = mask.asArray(Bool.self)
        XCTAssertEqual(values, [true, false, false, true,
                                false, false, false, false])
    }

    func testStageWriteOnlyTouchesTargetStage() {
        // gen: [1, K=4, T=3] with sentinel value 99 everywhere.
        let gen = MLXArray.full([1, 4, 3], values: MLXArray(Int32(99)))
        // Replace stage 2 with values [1, 2, 3].
        let stageSeq = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 1, 3])
        let out = stageWrite(gen: gen, stageSeq: stageSeq, stage: 2)
        let arr = out.asArray(Int32.self)
        // Stage 2 row should be [1, 2, 3]; others stay 99.
        XCTAssertEqual(Array(arr[0..<3]),  [99, 99, 99])       // stage 0
        XCTAssertEqual(Array(arr[3..<6]),  [99, 99, 99])       // stage 1
        XCTAssertEqual(Array(arr[6..<9]),  [1, 2, 3])          // stage 2
        XCTAssertEqual(Array(arr[9..<12]), [99, 99, 99])       // stage 3
    }
}

final class MAGNeTEncodecConfigTests: XCTestCase {
    func testDecodeEncodecConfig() throws {
        let json = """
        {
          "audio_channels": 1, "chunk_length_s": null,
          "codebook_dim": 128, "codebook_size": 2048, "compress": 2,
          "dilation_growth_rate": 2, "hidden_size": 128, "kernel_size": 7,
          "last_kernel_size": 7, "norm_type": "weight_norm", "normalize": false,
          "num_filters": 64, "num_lstm_layers": 2, "num_residual_layers": 1,
          "overlap": null, "pad_mode": "reflect", "residual_kernel_size": 3,
          "sampling_rate": 32000, "target_bandwidths": [2.2],
          "trim_right_ratio": 1.0, "upsampling_ratios": [8, 5, 4, 4],
          "use_causal_conv": false, "use_conv_shortcut": false
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(EncodecModelConfig.self, from: json)
        XCTAssertEqual(cfg.audioChannels, 1)
        XCTAssertEqual(cfg.codebookSize, 2048)
        XCTAssertEqual(cfg.upsamplingRatios, [8, 5, 4, 4])
        XCTAssertEqual(cfg.padMode, "reflect")
        XCTAssertFalse(cfg.useCausalConv)
        XCTAssertFalse(cfg.useConvShortcut)
    }
}
