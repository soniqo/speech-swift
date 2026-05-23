import XCTest
@testable import MagpieTTS

final class MagpieConfigTests: XCTestCase {
    func testVariantHFIDs() {
        XCTAssertEqual(MagpieTTSVariant.int4.huggingFaceRepoId,
                       "aufklarer/Magpie-TTS-Multilingual-357M-MLX-4bit")
        XCTAssertEqual(MagpieTTSVariant.int8.huggingFaceRepoId,
                       "aufklarer/Magpie-TTS-Multilingual-357M-MLX-8bit")
        XCTAssertEqual(MagpieTTSVariant.int4.bits, 4)
        XCTAssertEqual(MagpieTTSVariant.int8.bits, 8)
    }

    func testSpeakerNames() {
        XCTAssertEqual(MagpieSpeaker(named: "sofia"), .sofia)
        XCTAssertEqual(MagpieSpeaker(named: "Aria"), .aria)
        XCTAssertEqual(MagpieSpeaker(named: "JOHN"), .johnVanStan)
        XCTAssertNil(MagpieSpeaker(named: "nope"))
        XCTAssertEqual(MagpieSpeaker.sofia.displayName, "Sofia")
        XCTAssertEqual(MagpieSpeaker.johnVanStan.displayName, "John Van Stan")
    }

    func testLanguageCodes() {
        XCTAssertEqual(MagpieLanguage(code: "en"), .english)
        XCTAssertEqual(MagpieLanguage(code: "english"), .english)
        XCTAssertEqual(MagpieLanguage(code: "es"), .spanish)
        XCTAssertEqual(MagpieLanguage(code: "castellano"), .spanish)
        XCTAssertEqual(MagpieLanguage(code: "ZH"), .chinese)
        XCTAssertEqual(MagpieLanguage(code: "mandarin"), .chinese)
        XCTAssertEqual(MagpieLanguage(code: "ja"), .japanese)
        XCTAssertNil(MagpieLanguage(code: "klingon"))
    }

    func testDecoderConfigParse() throws {
        let json = """
        {
          "model": "magpie_decoder_prefill",
          "d_model": 768, "d_ffn": 3072, "n_layers": 12, "n_heads": 12,
          "kernel_size": 1,
          "xa_d_memory": 768, "xa_n_heads": 1, "xa_d_head": 128,
          "max_len": 2048,
          "num_codebooks": 8, "vocab_per_codebook": 2024,
          "audio_bos_id": 2016, "audio_eos_id": 2017,
          "num_baked_speakers": 5, "baked_T": 110,
          "local_transformer": { "n_layers": 1, "n_heads": 1, "d_model": 256 }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MagpieDecoderConfig.self, from: json)
        XCTAssertEqual(cfg.dModel, 768)
        XCTAssertEqual(cfg.nLayers, 12)
        XCTAssertEqual(cfg.xaDHead, 128)
        XCTAssertEqual(cfg.bakedT, 110)
        XCTAssertEqual(cfg.audioBosId, 2016)
        XCTAssertEqual(cfg.localTransformer.dModel, 256)
        XCTAssertNil(cfg.quantization)
    }

    func testQuantizedDecoderConfigParse() throws {
        let json = """
        {
          "model": "magpie_decoder_prefill",
          "d_model": 768, "d_ffn": 3072, "n_layers": 12, "n_heads": 12,
          "kernel_size": 1,
          "xa_d_memory": 768, "xa_n_heads": 1, "xa_d_head": 128,
          "max_len": 2048,
          "num_codebooks": 8, "vocab_per_codebook": 2024,
          "audio_bos_id": 2016, "audio_eos_id": 2017,
          "num_baked_speakers": 5, "baked_T": 110,
          "local_transformer": { "n_layers": 1, "n_heads": 1, "d_model": 256 },
          "quantization": { "bits": 4, "group_size": 64, "mode": "mlx_affine_flat" },
          "quantized_shapes": { "final_proj.weight": [16192, 768] }
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MagpieDecoderConfig.self, from: json)
        XCTAssertEqual(cfg.quantization?.bits, 4)
        XCTAssertEqual(cfg.quantization?.groupSize, 64)
        XCTAssertEqual(cfg.quantization?.mode, "mlx_affine_flat")
        XCTAssertEqual(cfg.quantizedShapes?["final_proj.weight"], [16192, 768])
    }

    func testNanoCodecConfigParse() throws {
        let json = """
        {
          "model": "nano_codec_decoder",
          "sample_rate": 22050, "samples_per_frame": 1024,
          "num_codebooks": 8, "vocab_per_codebook": 2024,
          "fsq_num_levels": [8, 7, 6, 6],
          "up_sample_rates": [8, 8, 4, 2, 2],
          "base_channels": 864
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(MagpieNanoCodecConfig.self, from: json)
        XCTAssertEqual(cfg.sampleRate, 22050)
        XCTAssertEqual(cfg.upSampleRates, [8, 8, 4, 2, 2])
        XCTAssertEqual(cfg.fsqNumLevels, [8, 7, 6, 6])
        XCTAssertEqual(cfg.baseChannels, 864)
    }

    func testParamsDefaults() {
        let p = MagpieTTSParams()
        XCTAssertEqual(p.temperature, 0.6, accuracy: 1e-6)
        XCTAssertEqual(p.topK, 80)
        XCTAssertEqual(p.maxSteps, 500)
        XCTAssertEqual(p.minFrames, 4)
        XCTAssertNil(p.seed)
    }
}
