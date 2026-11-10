import Foundation
import MLX
import XCTest
@testable import Qwen3ASR

final class FeatureExtractorBatchTests: XCTestCase {
    func testPartialSTFTBatchMatchesHuggingFace() {
        // 258 centered frames leave a partial second batch. Whisper drops
        // the final frame, retaining frame 256 with its reflected right edge.
        let audio = (0..<41120).map {
            Float(sin(2 * Double.pi * 440 * Double($0) / 16000) * 0.5)
        }
        let expected: [[Float]] = [
            [-0.514618, -0.514618, 1.344505, 1.337491, 1.485354, 1.275244, 1.287393, -0.514618, -0.514618],
            [0.105587, 0.059733, 1.344851, 1.337560, 1.485382, 1.274828, 1.286977, 0.101382, -0.006189],
        ]
        let extractor = WhisperFeatureExtractor()
        let raw = extractor.extractFeaturesRaw(audio)
        let mlx = extractor.extractFeatures(audio)
        XCTAssertEqual(raw.timeFrames, 257)
        XCTAssertEqual(mlx.shape, [128, 257])
        for (offset, values) in expected.enumerated() {
            let frame = 255 + offset
            for (binOffset, value) in values.enumerated() {
                let bin = 14 + binOffset
                XCTAssertEqual(raw.data[bin * raw.timeFrames + frame], value, accuracy: 1e-3)
                XCTAssertEqual(mlx[bin, frame].item(Float.self), value, accuracy: 1e-3)
            }
        }
    }
}
