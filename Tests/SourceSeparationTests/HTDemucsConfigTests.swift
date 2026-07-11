import XCTest
@testable import SourceSeparation

/// Unit tests for HTDemucs config decoding (no GPU / model downloads).
final class HTDemucsConfigTests: XCTestCase {

    private static let json = """
    {
      "model_name": "htdemucs_ft",
      "dtype": "fp16",
      "sources": ["drums", "bass", "other", "vocals"],
      "num_models": 4,
      "weights": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "samplerate": 44100,
      "segment": 7.8,
      "audio_channels": 2,
      "arch": {
        "channels": 48, "growth": 2, "nfft": 4096, "cac": true, "depth": 4,
        "rewrite": true, "freq_emb": 0.2, "emb_scale": 10, "emb_smooth": true,
        "kernel_size": 8, "stride": 4, "time_stride": 2, "context": 1,
        "context_enc": 0, "norm_starts": 4, "norm_groups": 4, "dconv_mode": 3,
        "dconv_depth": 2, "dconv_comp": 8, "bottom_channels": 512,
        "t_layers": 5, "t_heads": 8, "t_hidden_scale": 4.0, "t_emb": "sin",
        "t_max_period": 10000.0, "t_layer_scale": true, "t_gelu": true,
        "t_norm_in": true, "t_norm_first": true, "t_norm_out": true,
        "t_weight_pos_embed": 1.0, "t_cross_first": false
      }
    }
    """

    func testDecode() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("htd_cfg_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try Self.json.data(using: .utf8)!.write(to: url)

        let cfg = try HTDemucsConfig.load(from: url)
        XCTAssertEqual(cfg.sources, ["drums", "bass", "other", "vocals"])
        XCTAssertEqual(cfg.numModels, 4)
        XCTAssertEqual(cfg.samplerate, 44100)
        XCTAssertEqual(cfg.arch.bottomChannels, 512)
        XCTAssertEqual(cfg.arch.tLayers, 5)
        XCTAssertEqual(cfg.arch.tHeads, 8)
        XCTAssertEqual(cfg.arch.nfft, 4096)
        XCTAssertEqual(cfg.hopLength, 1024)                 // nfft/4
        XCTAssertEqual(cfg.trainingLength, Int(7.8 * 44100)) // 343980
        XCTAssertEqual(cfg.weights?[1], [0, 1, 0, 0])        // diagonal bag
    }
}
