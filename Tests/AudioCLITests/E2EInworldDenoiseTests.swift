import ArgumentParser
import AudioCommon
import Foundation
import XCTest
@testable import AudioCLILib

final class E2EInworldDenoiseTests: XCTestCase {
    func testOfflineGoldenThroughDenoiseCommand() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let model = environment["INWORLD_VOICE_ISOLATOR_MODEL_DIR"],
              let fixtures = environment["INWORLD_VOICE_ISOLATOR_FIXTURES"] else {
            throw XCTSkip("Set INWORLD_VOICE_ISOLATOR_MODEL_DIR and INWORLD_VOICE_ISOLATOR_FIXTURES")
        }

        for name in ["offline_3s", "offline_11s"] {
            let fixture = URL(fileURLWithPath: fixtures).appendingPathComponent(name)
            let input = fixture.appendingPathComponent("input.wav")
            let output = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).wav")
            defer { try? FileManager.default.removeItem(at: output) }
            var command = try AudioCLI.parseAsRoot([
                "denoise", input.path, "--engine", "inworld", "--model", model, "-o", output.path,
            ])
            XCTAssertTrue(command is DenoiseCommand)
            try command.run()

            let expected = try loadGolden(fixture.appendingPathComponent("expected.npy"))
            let actual = try AudioFileLoader.load(url: output, targetSampleRate: 48_000)
            XCTAssertEqual(actual.count, expected.count, name)
            XCTAssertTrue(actual.allSatisfy(\.isFinite), name)
            let maxError = zip(actual, expected).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            print("\(name): maximum absolute error \(maxError)")
            XCTAssertLessThanOrEqual(maxError, 1e-4, name)
        }
    }

    /// The pinned upstream fixtures use NumPy v1 with a single little-endian Float32 axis.
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
