import Foundation
import XCTest
@testable import Qwen3TTS

final class LocalModelLoadingTests: XCTestCase {

    func testLocalBundleValidationAcceptsSeparateCompleteDirectories() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }

        XCTAssertNoThrow(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: fixture.modelDirectory,
                tokenizerDirectory: fixture.tokenizerDirectory))
    }

    func testLocalBundleValidationRejectsNonFileURLBeforeFileSystemAccess() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        let remoteURL = try XCTUnwrap(URL(string: "https://example.com/model"))

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: remoteURL,
                tokenizerDirectory: fixture.tokenizerDirectory)
        ) { error in
            XCTAssertEqual(error as? Qwen3TTSLoadingError, .nonFileURL(remoteURL))
        }
    }

    func testLocalBundleValidationRejectsMissingDirectory() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        let missing = fixture.root.appendingPathComponent("missing", isDirectory: true)

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: missing,
                tokenizerDirectory: fixture.tokenizerDirectory)
        ) { error in
            XCTAssertEqual(error as? Qwen3TTSLoadingError, .directoryNotFound(missing))
        }
    }

    func testLocalBundleValidationRejectsRegularFileAsDirectory() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        let file = fixture.root.appendingPathComponent("not-a-directory")
        try Data().write(to: file)

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: fixture.modelDirectory,
                tokenizerDirectory: file)
        ) { error in
            XCTAssertEqual(error as? Qwen3TTSLoadingError, .pathIsNotDirectory(file))
        }
    }

    func testLocalBundleValidationRejectsMissingRequiredFiles() throws {
        for fileName in ["config.json", "vocab.json"] {
            let fixture = try LocalBundleFixture()
            defer { fixture.remove() }
            let file = fixture.modelDirectory.appendingPathComponent(fileName)
            try FileManager.default.removeItem(at: file)

            XCTAssertThrowsError(
                try Qwen3TTSModel.validateLocalBundle(
                    modelDirectory: fixture.modelDirectory,
                    tokenizerDirectory: fixture.tokenizerDirectory)
            ) { error in
                XCTAssertEqual(
                    error as? Qwen3TTSLoadingError,
                    .requiredFileUnavailable(file))
            }
        }
    }

    func testLocalBundleValidationRejectsMissingModelWeights() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try FileManager.default.removeItem(at: fixture.modelWeights)

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: fixture.modelDirectory,
                tokenizerDirectory: fixture.tokenizerDirectory)
        ) { error in
            XCTAssertEqual(
                error as? Qwen3TTSLoadingError,
                .weightsUnavailable(fixture.modelDirectory))
        }
    }

    func testLocalBundleValidationRejectsMissingTokenizerWeights() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try FileManager.default.removeItem(at: fixture.tokenizerWeights)

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: fixture.modelDirectory,
                tokenizerDirectory: fixture.tokenizerDirectory)
        ) { error in
            XCTAssertEqual(
                error as? Qwen3TTSLoadingError,
                .weightsUnavailable(fixture.tokenizerDirectory))
        }
    }

    func testLocalBundleValidationRejectsIncompleteShardedCheckpoint() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try FileManager.default.removeItem(at: fixture.modelWeights)
        let firstShard = "model-00001-of-00002.safetensors"
        try Data().write(
            to: fixture.modelDirectory.appendingPathComponent(firstShard))
        let index = """
            {
              "weight_map": {
                "talker.weight": "\(firstShard)",
                "predictor.weight": "model-00002-of-00002.safetensors"
              }
            }
            """
        try Data(index.utf8).write(
            to: fixture.modelDirectory.appendingPathComponent(
                "model.safetensors.index.json"))

        XCTAssertThrowsError(
            try Qwen3TTSModel.validateLocalBundle(
                modelDirectory: fixture.modelDirectory,
                tokenizerDirectory: fixture.tokenizerDirectory)
        ) { error in
            XCTAssertEqual(
                error as? Qwen3TTSLoadingError,
                .weightsUnavailable(fixture.modelDirectory))
        }
    }

    func testNoneWiredMemoryPolicyLeavesProcessStateUntouched() throws {
        var requestedFractions: [Double] = []

        try Qwen3TTSModel.applyWiredMemoryPolicy(.none) { fraction in
            requestedFractions.append(fraction)
        }

        XCTAssertTrue(requestedFractions.isEmpty)
    }

    func testPinWiredMemoryPolicyForwardsValidatedFraction() throws {
        var requestedFractions: [Double] = []

        try Qwen3TTSModel.applyWiredMemoryPolicy(.pin(fraction: 0.75)) { fraction in
            requestedFractions.append(fraction)
        }

        XCTAssertEqual(requestedFractions, [0.75])
    }

    func testPinWiredMemoryPolicyRejectsInvalidFractionsWithoutMutation() {
        for fraction in [0, -0.1, 1.1, .infinity, .nan] {
            var didPin = false

            XCTAssertThrowsError(
                try Qwen3TTSModel.applyWiredMemoryPolicy(.pin(fraction: fraction)) { _ in
                    didPin = true
                }
            ) { error in
                guard let loadingError = error as? Qwen3TTSLoadingError,
                    case .invalidWiredMemoryFraction(let actual) = loadingError
                else {
                    return XCTFail("Unexpected error: \(error)")
                }
                if fraction.isNaN {
                    XCTAssertTrue(actual.isNaN)
                } else {
                    XCTAssertEqual(actual, fraction)
                }
            }
            XCTAssertFalse(didPin)
        }
    }
}

private final class LocalBundleFixture {
    let root: URL
    let modelDirectory: URL
    let tokenizerDirectory: URL
    let modelWeights: URL
    let tokenizerWeights: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-local-loader-\(UUID().uuidString)", isDirectory: true)
        modelDirectory = root.appendingPathComponent("model", isDirectory: true)
        tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        modelWeights = modelDirectory.appendingPathComponent("model.safetensors")
        tokenizerWeights = tokenizerDirectory.appendingPathComponent("tokenizer.safetensors")

        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(
            to: modelDirectory.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(
            to: modelDirectory.appendingPathComponent("vocab.json"))
        try Data().write(to: modelWeights)
        try Data().write(to: tokenizerWeights)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }
}
