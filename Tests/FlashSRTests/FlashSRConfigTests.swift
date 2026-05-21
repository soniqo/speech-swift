import XCTest
@testable import FlashSR

final class FlashSRConfigTests: XCTestCase {
    func testVariantRepoIds() {
        XCTAssertEqual(FlashSRVariant.int4.huggingFaceRepoId, "aufklarer/FlashSR-MLX-4bit")
        XCTAssertEqual(FlashSRVariant.int8.huggingFaceRepoId, "aufklarer/FlashSR-MLX-8bit")
        XCTAssertEqual(FlashSRVariant.int4.bits, 4)
        XCTAssertEqual(FlashSRVariant.int8.bits, 8)
    }

    func testConfigDecodeMatchesPublishedSchema() throws {
        // Minimal subset of the real config.json published with the bundle.
        let json = """
        {
          "vae": {
            "in_channels": 1, "out_ch": 1, "ch": 128, "ch_mult": [1, 2, 4, 8],
            "num_res_blocks": 2, "attn_resolutions": [], "double_z": true,
            "z_channels": 16, "embed_dim": 16, "dropout": 0.1,
            "resolution": 256, "mel_bins": 256, "scale_factor_z": 0.3342
          },
          "ldm": {
            "in_channels": 32, "model_channels": 128, "out_channels": 16,
            "num_res_blocks": 2
          },
          "mel": {
            "n_fft": 2048, "hop": 480, "sr": 48000, "n_mels": 256,
            "fmin": 20, "fmax": 24000
          },
          "audio": {
            "sample_rate": 48000, "frame_samples": 245760, "frame_sec": 5.12
          },
          "format": "int4",
          "quantization": {
            "mode": "mlx_affine_flat", "bits": 4, "group_size": 64,
            "rule": "every weight with ndim>=2 and (prod(shape[1:]) % group_size == 0)"
          },
          "quantized_shapes": {
            "ldm.input_blocks.1.0.emb_layers.1.weight": [128, 512]
          }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(FlashSRConfig.self, from: json)
        XCTAssertEqual(cfg.vae.ch, 128)
        XCTAssertEqual(cfg.vae.chMult, [1, 2, 4, 8])
        XCTAssertEqual(cfg.vae.zChannels, 16)
        XCTAssertEqual(cfg.vae.scaleFactorZ, 0.3342, accuracy: 1e-6)
        XCTAssertEqual(cfg.ldm.inChannels, 32)
        XCTAssertEqual(cfg.ldm.attentionResolutions, [8, 4, 2])
        XCTAssertEqual(cfg.ldm.channelMult, [1, 2, 3, 5])
        XCTAssertEqual(cfg.mel.nFft, 2048)
        XCTAssertEqual(cfg.mel.hop, 480)
        XCTAssertEqual(cfg.audio.sampleRate, 48000)
        XCTAssertEqual(cfg.audio.frameSamples, 245760)
        XCTAssertEqual(cfg.quantization?.bits, 4)
        XCTAssertEqual(cfg.quantization?.groupSize, 64)
        XCTAssertEqual(cfg.quantizedShapes?["ldm.input_blocks.1.0.emb_layers.1.weight"], [128, 512])
    }
}
