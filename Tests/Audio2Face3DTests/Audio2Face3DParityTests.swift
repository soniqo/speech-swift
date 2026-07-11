import XCTest
@testable import Audio2Face3D

final class Audio2Face3DParityTests: XCTestCase {
    func testMLXForwardMatchesONNXFixtureWhenAvailable() throws {
        guard let fixtureDir = ProcessInfo.processInfo.environment["AUDIO2FACE3D_FIXTURE_DIR"] else {
            throw XCTSkip("set AUDIO2FACE3D_FIXTURE_DIR to an exported Audio2Face3D MLX bundle")
        }

        let dir = URL(fileURLWithPath: fixtureDir, isDirectory: true)
        let input = try readFloat32(dir.appendingPathComponent("parity_input.f32"))
        let emotion = try readFloat32(dir.appendingPathComponent("parity_emotion.f32"))
        let expected = try readFloat32(dir.appendingPathComponent("parity_result.f32"))

        let configuration = try Audio2Face3DDownloader.configuration(from: dir)
        let runtime = try Audio2Face3DMLXRuntime(directory: dir, configuration: configuration)
        let actual = try runtime.coefficients(forWindow: input, emotion: emotion)

        XCTAssertEqual(actual.count, expected.count)
        var maxAbs: Float = 0
        var sumAbs: Float = 0
        for (lhs, rhs) in zip(actual, expected) {
            let diff = abs(lhs - rhs)
            maxAbs = max(maxAbs, diff)
            sumAbs += diff
        }
        let meanAbs = sumAbs / Float(expected.count)
        XCTAssertLessThan(maxAbs, 0.05)
        XCTAssertLessThan(meanAbs, 0.005)
    }

    private func readFloat32(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        precondition(data.count % MemoryLayout<Float>.stride == 0)
        return data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }
}
