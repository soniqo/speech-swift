import CoreML
import XCTest
@testable import CanaryASR

/// Holds Canary's mel front end to the same NeMo reference the Parakeet side
/// uses — both checkpoints are trained by the same preprocessor.
///
/// Isolates the front end from the decode loop: an empty transcript can come
/// from either, and this says which.
final class MelContractTests: XCTestCase {

    private struct Reference: Decodable {
        let frames: Int
        let bins: Int
        let validFrames: Int
        let melFrames: Int
        let mel: [Float]
    }

    private func referenceSignal(count: Int) -> [Float] {
        var state: UInt32 = 12345
        var samples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 1664525 &+ 1013904223
            samples[i] = (Float(state) / 2147483648.0 - 1.0) * 0.3
        }
        return samples
    }

    func testMelMatchesTheTrainingPreprocessor() throws {
        guard let url = Bundle.module.url(forResource: "nemo_mel_reference", withExtension: "json") else {
            throw XCTSkip("nemo_mel_reference.json not in test resources")
        }
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))

        let preprocessor = MelPreprocessor(config: .default)
        let audio = referenceSignal(count: 16000 * 2)
        let (mel, frames) = try preprocessor.extract(audio, window: 3000)

        XCTAssertEqual(frames, reference.validFrames, "valid frame count")
        XCTAssertEqual(mel.dataType, .float32)

        let pointer = mel.dataPointer.assumingMemoryBound(to: Float.self)
        let stride = mel.shape[2].intValue
        var worst: Float = 0
        var sumSquared: Double = 0
        for t in 0..<reference.melFrames {
            for b in 0..<reference.bins {
                let delta = abs(reference.mel[t * reference.bins + b] - pointer[b * stride + t])
                sumSquared += Double(delta * delta)
                worst = max(worst, delta)
            }
        }
        let rms = sqrt(sumSquared / Double(reference.melFrames * reference.bins))

        XCTAssertLessThan(worst, 0.05, "worst mel deviation \(worst); RMS \(rms)")
        XCTAssertLessThan(rms, 0.01, "RMS mel deviation \(rms)")
    }
}
