import AudioCommon
import Foundation
import SpeechEnhancement
import XCTest

final class E2EInworldVoiceIsolatorStreamTests: XCTestCase {
    func testGoldenRendersWithIncrementalInput() throws {
        try checkGoldenRenders(modelVariable: "INWORLD_VOICE_ISOLATOR_STREAM_MODEL_DIR")
    }

    func testLowMemoryGoldenRendersWithIncrementalInput() throws {
        try checkGoldenRenders(modelVariable: "INWORLD_VOICE_ISOLATOR_LOW_MEMORY_MODEL_DIR")
    }

    func testFastGoldenRendersWithIncrementalInput() throws {
        try checkGoldenRenders(modelVariable: "INWORLD_VOICE_ISOLATOR_FAST_MODEL_DIR", pushSize: 3_840)
    }

    func testFastPreparedStreamUsesAlignedBlocks() throws {
        try checkGoldenRenders(modelVariable: "INWORLD_VOICE_ISOLATOR_FAST_MODEL_DIR", pushSize: 3_840, prepare: true)
    }

    func testFastStreamRendersThroughTwoCrossfades() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment["INWORLD_VOICE_ISOLATOR_FAST_MODEL_DIR"],
              let fixtures = environment["INWORLD_VOICE_ISOLATOR_FIXTURES"] else {
            throw XCTSkip("Set INWORLD_VOICE_ISOLATOR_FAST_MODEL_DIR and INWORLD_VOICE_ISOLATOR_FIXTURES")
        }
        let fixture = URL(fileURLWithPath: fixtures).appendingPathComponent("realtime_stream_11s")
        let rate = InworldVoiceIsolatorStream.sampleRate
        let audio = try AudioFileLoader.load(url: fixture.appendingPathComponent("input.wav"), targetSampleRate: rate)
        let extended = audio + audio.prefix(2 * rate)
        let stream = try InworldVoiceIsolatorStream(modelDirectory: URL(fileURLWithPath: model))
        var actual: [Float] = []
        for start in stride(from: 0, to: extended.count, by: 3_840) {
            let end = min(start + 3_840, extended.count)
            actual.append(contentsOf: try stream.push(Array(extended[start..<end])))
        }
        actual.append(contentsOf: try stream.flush())
        XCTAssertEqual(actual.count, extended.count)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        let expected = try loadGolden(fixture.appendingPathComponent("expected.npy"))
        // Leave room for the different look-ahead and analysis at the end of the fixture.
        let prefix = expected.count - 2 * stream.latencySamples
        let maxError = zip(actual.prefix(prefix), expected.prefix(prefix)).reduce(Float(0)) {
            max($0, abs($1.0 - $1.1))
        }
        XCTAssertLessThanOrEqual(maxError, 1e-4)
    }

    func testSpecializedAttentionGoldenRendersWithIncrementalInput() throws {
        try checkGoldenRenders(modelVariable: "INWORLD_VOICE_ISOLATOR_SPECIALIZED_MODEL_DIR")
    }

    private func checkGoldenRenders(modelVariable: String, pushSize: Int = 960, prepare: Bool = false) throws {
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment[modelVariable],
              let fixtures = environment["INWORLD_VOICE_ISOLATOR_FIXTURES"] else {
            throw XCTSkip("Set \(modelVariable) and INWORLD_VOICE_ISOLATOR_FIXTURES")
        }
        for (name, pushSizes) in [("realtime_stream_3s", [7, pushSize, 1_537, 137]), ("realtime_stream_11s", [pushSize])] {
            try autoreleasepool {
                let fixture = URL(fileURLWithPath: fixtures).appendingPathComponent(name)
                let audio = try AudioFileLoader.load(url: fixture.appendingPathComponent("input.wav"), targetSampleRate: 48_000)
                let stream = try InworldVoiceIsolatorStream(modelDirectory: URL(fileURLWithPath: model))
                XCTAssertEqual(stream.latencySamples, 3_008)
                XCTAssertThrowsError(try stream.push([.nan]))
                XCTAssertEqual(try stream.push([]), [])
                if prepare {
                    XCTAssertEqual(stream.preferredPushSamples, 4_384)
                    try stream.prepare()
                    try stream.prepare()
                    XCTAssertEqual(stream.preferredPushSamples, 4_384)
                }
                var actual: [Float] = []
                var position = 0
                var push = 0
                while position < audio.count {
                    let count = prepare ? stream.preferredPushSamples : pushSizes[push % pushSizes.count]
                    if prepare, position > 0 { XCTAssertEqual(count, pushSize) }
                    let end = min(position + count, audio.count)
                    actual.append(contentsOf: try stream.push(Array(audio[position..<end])))
                    if prepare, position == 0 { XCTAssertThrowsError(try stream.prepare()) }
                    position = end
                    push += 1
                }
                XCTAssertLessThan(actual.count, audio.count, "Flush must release the look-ahead tail.")
                actual.append(contentsOf: try stream.flush())
                XCTAssertEqual(try stream.flush(), [])
                XCTAssertEqual(stream.preferredPushSamples, 0)
                XCTAssertThrowsError(try stream.push([]))
                let expected = try loadGolden(fixture.appendingPathComponent("expected.npy"))
                XCTAssertEqual(actual.count, expected.count, name)
                XCTAssertTrue(actual.allSatisfy(\.isFinite), name)
                let maxError = zip(actual, expected).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
                print("\(name): maximum absolute error \(maxError)")
                XCTAssertLessThanOrEqual(maxError, 1e-4, name)
            }
        }
    }

    /// These fixtures use NumPy v1 with a single little-endian Float32 axis.
    private func loadGolden(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count >= 10, Array(data.prefix(8)) == [0x93, 78, 85, 77, 80, 89, 1, 0] else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let offset = 10 + Int(data[8]) + 256 * Int(data[9])
        guard offset < data.count, (data.count - offset).isMultiple(of: 4),
              String(decoding: data[10..<offset], as: UTF8.self).contains("'descr': '<f4'") else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return data.withUnsafeBytes { bytes in
            stride(from: offset, to: bytes.count, by: 4).map { index in
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: index, as: UInt32.self)))
            }
        }
    }
}
