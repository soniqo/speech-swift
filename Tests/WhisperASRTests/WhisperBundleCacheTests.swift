import Foundation
@testable import WhisperASR
import XCTest

final class WhisperBundleCacheTests: XCTestCase {
    private var temporaryDirectory: URL!

    override func setUpWithError() throws {
        temporaryDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: temporaryDirectory,
            withIntermediateDirectories: true
        )
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: temporaryDirectory)
        temporaryDirectory = nil
    }

    func testCompleteBundleIsAccepted() throws {
        try createCompleteBundle()

        XCTAssertTrue(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testMissingPayloadIsRejected() throws {
        try createCompleteBundle()
        try FileManager.default.removeItem(
            at: temporaryDirectory.appendingPathComponent("AudioEncoder.mlmodelc/model.mil")
        )

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    func testEmptyPayloadIsRejected() throws {
        try createCompleteBundle()
        try Data().write(
            to: temporaryDirectory.appendingPathComponent("TextDecoder.mlmodelc/weights/weight.bin")
        )

        XCTAssertFalse(WhisperASRModel.hasCompleteCachedBundle(in: temporaryDirectory))
    }

    private func createCompleteBundle() throws {
        for relativePath in WhisperASRModel.requiredBundleFiles {
            let fileURL = temporaryDirectory.appendingPathComponent(relativePath)
            try FileManager.default.createDirectory(
                at: fileURL.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try Data([0x01]).write(to: fileURL)
        }
    }
}
