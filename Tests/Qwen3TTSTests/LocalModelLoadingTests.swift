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

    func testBundleConfigurationOverridesWrongCallerSizeAndQuantization() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig(
            """
            {
              "model_size": "0.6B",
              "quantization_config": { "bits": 8, "group_size": 32 },
              "talker_config": {
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_hidden_layers": 28
              }
            }
            """)
        var wrongFallback = Qwen3TTSConfig.config(for: .large, bits: 0)
        wrongFallback.speechTokenizerDecoder.sampleRate = 16_000

        let resolved = try Qwen3TTSModel.resolveLocalConfiguration(
            from: fixture.configFile,
            fallback: wrongFallback)

        XCTAssertEqual(resolved.talker.hiddenSize, 1024)
        XCTAssertEqual(resolved.talker.bits, 8)
        XCTAssertEqual(resolved.talker.groupSize, 32)
        XCTAssertEqual(resolved.codePredictor.embeddingDim, 1024)
        XCTAssertEqual(resolved.codePredictor.bits, 8)
        XCTAssertEqual(resolved.codePredictor.groupSize, 32)
        XCTAssertEqual(resolved.speechTokenizerDecoder.sampleRate, 16_000)
    }

    func testMissingQuantizationMetadataMeansUnquantized() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig(
            """
            {
              "model_size": "1.7B",
              "talker_config": { "hidden_size": 2048 }
            }
            """)

        let resolved = try Qwen3TTSModel.resolveLocalConfiguration(
            from: fixture.configFile,
            fallback: .config(for: .small, bits: 8))

        XCTAssertEqual(resolved.talker.hiddenSize, 2048)
        XCTAssertEqual(resolved.talker.bits, 0)
        XCTAssertEqual(resolved.codePredictor.embeddingDim, 2048)
        XCTAssertEqual(resolved.codePredictor.bits, 0)
    }

    func testTalkerMetadataCanInferSizeAndConfigureCodePredictor() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig(
            """
            {
              "talker_config": {
                "hidden_size": 1024,
                "rope_scaling": { "mrope_section": [16, 16, 32] },
                "code_predictor_config": {
                  "hidden_size": 1024,
                  "num_hidden_layers": 6,
                  "num_code_groups": 12
                }
              }
            }
            """)

        let resolved = try Qwen3TTSModel.resolveLocalConfiguration(
            from: fixture.configFile)

        XCTAssertEqual(resolved.talker.hiddenSize, 1024)
        XCTAssertEqual(resolved.talker.bits, 0)
        XCTAssertEqual(resolved.talker.mropeSections, [16, 16, 32])
        XCTAssertEqual(resolved.codePredictor.numLayers, 6)
        XCTAssertEqual(resolved.codePredictor.numCodeGroups, 12)
    }

    func testBundleConfigurationRejectsContradictoryModelSize() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig(
            """
            {
              "model_size": "0.6B",
              "talker_config": { "hidden_size": 2048 }
            }
            """)

        XCTAssertThrowsError(
            try Qwen3TTSModel.resolveLocalConfiguration(from: fixture.configFile)
        ) { error in
            guard case .invalidConfiguration(let url, let reason) =
                error as? Qwen3TTSLoadingError
            else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url, fixture.configFile)
            XCTAssertTrue(reason.contains("conflicts"))
        }
    }

    func testBundleConfigurationRejectsUnsupportedQuantizationBeforeAllocation() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig(
            """
            {
              "model_size": "0.6B",
              "quantization_config": { "bits": 3 }
            }
            """)

        XCTAssertThrowsError(
            try Qwen3TTSModel.resolveLocalConfiguration(from: fixture.configFile)
        ) { error in
            guard case .invalidConfiguration(let url, let reason) =
                error as? Qwen3TTSLoadingError
            else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url, fixture.configFile)
            XCTAssertTrue(reason.contains("bits"))
        }
    }

    func testBundleConfigurationWrapsMalformedJSONInTypedError() throws {
        let fixture = try LocalBundleFixture()
        defer { fixture.remove() }
        try fixture.writeConfig("{not-json")

        XCTAssertThrowsError(
            try Qwen3TTSModel.resolveLocalConfiguration(from: fixture.configFile)
        ) { error in
            guard case .invalidConfiguration(let url, _) = error as? Qwen3TTSLoadingError else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(url, fixture.configFile)
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
    let configFile: URL

    init() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-local-loader-\(UUID().uuidString)", isDirectory: true)
        modelDirectory = root.appendingPathComponent("model", isDirectory: true)
        tokenizerDirectory = root.appendingPathComponent("tokenizer", isDirectory: true)
        modelWeights = modelDirectory.appendingPathComponent("model.safetensors")
        tokenizerWeights = tokenizerDirectory.appendingPathComponent("tokenizer.safetensors")
        configFile = modelDirectory.appendingPathComponent("config.json")

        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: tokenizerDirectory, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: configFile)
        try Data("{}".utf8).write(
            to: modelDirectory.appendingPathComponent("vocab.json"))
        try Data().write(to: modelWeights)
        try Data().write(to: tokenizerWeights)
    }

    func remove() {
        try? FileManager.default.removeItem(at: root)
    }

    func writeConfig(_ json: String) throws {
        try Data(json.utf8).write(to: configFile)
    }
}
