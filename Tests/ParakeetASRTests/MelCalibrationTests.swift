import CoreML
import XCTest
@testable import ParakeetASR

final class MelCalibrationTests: XCTestCase {
    private struct Fixture: Decodable {
        let sourceCommit: String
        let references: [Reference]
    }

    private struct Reference: Decodable {
        let amplitude: Float
        let samples: Int
        let bins: Int
        let validFrames: Int
        let bufferFrames: Int
        let mel: [Float] // Frame-major, including the zero padding frame.
    }

    func testDefaultFeaturesRetainLegacyCalibrationAtNormalAndQuietLevels() throws {
        let url = try XCTUnwrap(Bundle.module.url(
            forResource: "legacy_mel_reference", withExtension: "json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertEqual(fixture.sourceCommit, "64bcdc960d78f968ab6fe5767797c2ac0dad9434")
        XCTAssertEqual(fixture.references.count, 2)

        for reference in fixture.references {
            var state: UInt32 = 12345
            let audio: [Float] = (0..<reference.samples).map { _ in
                state = state &* 1664525 &+ 1013904223
                return (Float(state) / 2147483648 - 1) * reference.amplitude
            }
            let (mel, length) = try MelPreprocessor(config: .default).extract(audio)
            XCTAssertEqual(length, reference.validFrames)
            XCTAssertEqual(mel.shape, [1, reference.bins as NSNumber, reference.bufferFrames as NSNumber])
            XCTAssertEqual(reference.mel.count, reference.bins * reference.bufferFrames)
            let values = mel.dataPointer.assumingMemoryBound(to: Float16.self)
            var worst: Float = 0
            var squared = 0.0
            for frame in 0..<reference.bufferFrames {
                for bin in 0..<reference.bins {
                    let expected = reference.mel[frame * reference.bins + bin]
                    let delta = abs(Float(values[bin * reference.bufferFrames + frame]) - expected)
                    worst = max(worst, delta)
                    squared += Double(delta * delta)
                }
            }
            let rms = sqrt(squared / Double(reference.mel.count))
            XCTAssertLessThan(worst, 0.01, "Legacy calibration at amplitude \(reference.amplitude)")
            XCTAssertLessThan(rms, 0.001, "Legacy calibration at amplitude \(reference.amplitude)")
        }
    }
}
