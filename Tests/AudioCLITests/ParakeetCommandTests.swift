import Foundation
import XCTest
@testable import AudioCLILib

final class ParakeetCommandTests: XCTestCase {
    func testSingleAndBatchSelectUltra() throws {
        let single = try TranscribeCommand.parse(["speech.wav", "--engine", "parakeet", "--model", "Ultra"])
        let batch = try TranscribeBatchCommand.parse(["clips", "--engine", "parakeet", "--model", "ultra"])

        for value in [single.model, batch.model] {
            let resolved = try resolveParakeetModel(value)
            XCTAssertEqual(resolved.modelId, "aufklarer/Parakeet-Ultra-CoreML-INT8-30s")
            XCTAssertNil(resolved.directory)
        }
    }

    func testPlatformDefaultAndExplicitHubID() throws {
        let single = try TranscribeCommand.parse(["speech.wav", "--engine", "parakeet"])
        let batch = try TranscribeBatchCommand.parse(["clips", "--engine", "parakeet"])
        for value in [single.model, batch.model, "v3", "default"] {
            let resolved = try resolveParakeetModel(value)
            XCTAssertNil(resolved.modelId)
            XCTAssertNil(resolved.directory)
        }
        XCTAssertEqual(try resolveParakeetModel("org/custom-model").modelId, "org/custom-model")
    }

    func testLocalBundleAndInvalidSelection() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: directory) }

        let local = try resolveParakeetModel(directory.path)
        XCTAssertEqual(local.directory?.path, directory.path)
        XCTAssertNil(local.modelId)
        XCTAssertThrowsError(try TranscribeCommand.parse([
            "speech.wav", "--engine", "parakeet", "--model", directory.appendingPathComponent("missing").path,
        ]))
        XCTAssertThrowsError(try TranscribeBatchCommand.parse([
            "clips", "--engine", "parakeet", "--model", "unknown",
        ]))
    }
}
