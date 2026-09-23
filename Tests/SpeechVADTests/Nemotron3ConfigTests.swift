import Foundation
import XCTest
@testable import SpeechVAD

final class Nemotron3ConfigTests: XCTestCase {
    func testFinalBundleConfigLoadsAndPreviewRevisionIsRejected() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var fields: [String: Any] = [
            "model_type": "nemotron3_diarization",
            "source_model": "nvidia/Nemotron-3-Diarization",
            "source_revision": "a435e9867d79e789e90053f9b6d6834053af564a",
            "dtype": "int8",
            "sample_rate": 16_000,
            "n_mels": 128,
            "d_model": 512,
            "tf_model": 192,
            "num_layers": 31,
            "num_heads": 8,
            "num_speakers": 8,
            "subsampling_factor": 8,
            "upsample_factor": 8,
            "spkcache_len": 264,
            "fifo_len": 40,
            "chunk_len": 340,
            "right_context": 40,
            "spkcache_update_period": 300,
            "quantization": ["group_size": 64, "bits": 8, "mode": "affine"],
        ]
        let url = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: fields).write(to: url)
        let loaded = try Nemotron3ArtifactConfiguration.load(from: directory)
        XCTAssertEqual(loaded.numSpeakers, 8)

        fields["source_revision"] = "56d02ca97ad538d937f823e830ac5116de0acbb1"
        try JSONSerialization.data(withJSONObject: fields).write(to: url)
        XCTAssertThrowsError(try Nemotron3ArtifactConfiguration.load(from: directory))
    }
}
