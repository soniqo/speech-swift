import CoreML
import XCTest
@testable import ParakeetASR

final class MelWindowNormalizationTests: XCTestCase {
    func testWindowsUseTheirOwnMeanAndSampleDeviation() {
        var features: [Float] = [0, 2, 100, 104, 999, 10, 14, -100, -98, 999]
        MelPreprocessor.normalizeLogMel(
            &features, bins: 2, bufferFrames: 5, validFrames: 4, windowLength: 2)
        let expected: [Float] = [
            -1 / (sqrt(2) + 1e-5), 1 / (sqrt(2) + 1e-5),
            -2 / (sqrt(8) + 1e-5), 2 / (sqrt(8) + 1e-5), 0,
            -2 / (sqrt(8) + 1e-5), 2 / (sqrt(8) + 1e-5),
            -1 / (sqrt(2) + 1e-5), 1 / (sqrt(2) + 1e-5), 0,
        ]
        for (actual, expected) in zip(features, expected) {
            XCTAssertEqual(actual, expected, accuracy: 1e-6)
        }
    }

    func testOneFrameTailAndPaddingAreZero() {
        var features: [Float] = [0, 2, 99, 123]
        MelPreprocessor.normalizeLogMel(
            &features, bins: 1, bufferFrames: 4, validFrames: 3, windowLength: 2)
        XCTAssertTrue(features.allSatisfy(\.isFinite))
        XCTAssertEqual(features[2], 0)
        XCTAssertEqual(features[3], 0)
        XCTAssertEqual(features[0], -features[1])
    }

    func testConstantWindowsNormalizeToZero() {
        var features = [Float](repeating: log(pow(2, -24)), count: 1001)
        MelPreprocessor.normalizeLogMel(
            &features, bins: 1, bufferFrames: 1001, validFrames: 1000, windowLength: 500)
        XCTAssertTrue(features.allSatisfy { $0 == 0 })
    }

    func testNoValidFramesProducesOnlyPadding() {
        var features: [Float] = [1, 2, 3, 4]
        MelPreprocessor.normalizeLogMel(
            &features, bins: 2, bufferFrames: 2, validFrames: 0, windowLength: 2)
        XCTAssertEqual(features, [0, 0, 0, 0])
    }

    func testSingleWindowRetainsUtteranceFeatures() throws {
        let audio = signal(count: 16000, quietAfter: 16000)
        let preprocessor = MelPreprocessor(config: .default)
        let (utterance, length) = try preprocessor.extract(audio)
        let (window, windowLength) = try preprocessor.extract(audio, normalizationWindowLength: 500)
        XCTAssertEqual(length, windowLength)
        XCTAssertEqual(utterance.shape, window.shape)
        let lhs = utterance.dataPointer.assumingMemoryBound(to: Float16.self)
        let rhs = window.dataPointer.assumingMemoryBound(to: Float16.self)
        XCTAssertEqual(Array(UnsafeBufferPointer(start: lhs, count: utterance.count)),
                       Array(UnsafeBufferPointer(start: rhs, count: window.count)))
    }

    func testExtractionCentersEachWindowAcrossALevelChange() throws {
        let audio = signal(count: 32000, quietAfter: 16000)
        let (mel, length) = try MelPreprocessor(config: .default).extract(
            audio, normalizationWindowLength: 100)
        XCTAssertEqual(length, 200)
        let pointer = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let frames = mel.shape[2].intValue
        for bin in 0..<128 {
            for start in [0, 100] {
                let mean = (start..<(start + 100)).reduce(Float.zero) {
                    $0 + Float(pointer[bin * frames + $1])
                } / 100
                XCTAssertEqual(mean, 0, accuracy: 0.01, "Bin \(bin), window \(start)")
            }
            XCTAssertEqual(pointer[bin * frames + length], 0)
        }
    }

    func testExtractionHandlesASingleFrameFinalWindow() throws {
        let (mel, length) = try MelPreprocessor(config: .default).extract(
            signal(count: 101 * 160, quietAfter: 16000), normalizationWindowLength: 100)
        XCTAssertEqual(length, 101)
        let pointer = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let frames = mel.shape[2].intValue
        for index in 0..<mel.count { XCTAssertTrue(pointer[index].isFinite) }
        for bin in 0..<128 {
            XCTAssertEqual(pointer[bin * frames + 100], 0)
            XCTAssertEqual(pointer[bin * frames + 101], 0)
        }
    }

    func testFullWindowPrefixKeepsTheFirstAndLastValidFrames() throws {
        let mel = try MLMultiArray(shape: [1, 128, 501], dataType: .float16)
        let values = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        for bin in 0..<128 {
            for frame in 0..<501 { values[bin * 501 + frame] = Float16(frame) }
        }
        let prefix = try MelPreprocessor.validPrefix(mel, length: 500)
        XCTAssertEqual(prefix.shape, [1, 128, 500])
        let output = prefix.dataPointer.assumingMemoryBound(to: Float16.self)
        for bin in 0..<128 {
            XCTAssertEqual(output[bin * 500], 0)
            XCTAssertEqual(output[bin * 500 + 499], 499)
        }
    }

    func testPrefixWithoutPaddingRetainsTheArray() throws {
        let mel = try MLMultiArray(shape: [1, 128, 500], dataType: .float16)
        XCTAssertTrue(try MelPreprocessor.validPrefix(mel, length: 500) === mel)
    }

    func testEmptyAudioAndInvalidWindowAreRejected() {
        let preprocessor = MelPreprocessor(config: .default)
        XCTAssertThrowsError(try preprocessor.extract([]))
        XCTAssertThrowsError(try preprocessor.extract([0, 1], normalizationWindowLength: 0))
    }

    private func signal(count: Int, quietAfter: Int) -> [Float] {
        var state: UInt32 = 12345
        return (0..<count).map { index in
            state = state &* 1664525 &+ 1013904223
            let amplitude: Float = index < quietAfter ? 0.3 : 0.03
            return (Float(state) / 2147483648 - 1) * amplitude
        }
    }
}
