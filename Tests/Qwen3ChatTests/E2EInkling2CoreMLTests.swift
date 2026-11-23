import CoreML
import Foundation
import XCTest
@testable import Qwen3Chat

/// CPU Core ML checks use locally converted tiny partitions, with no downloads.
final class E2EInkling2CoreMLParityTests: XCTestCase {
    private func directory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["INKLING2_COREML_TINY_DIR"] else {
            throw XCTSkip("Set INKLING2_COREML_TINY_DIR to the converted tiny Core ML fixture")
        }
        return URL(fileURLWithPath: path)
    }

    func testSwiftVisitsCachesAndResetMatchPythonCoreML() throws {
        let directory = try directory()
        let model = try Inkling2CoreMLModel(directory: directory, contextLimit: 64, computeUnits: .cpuOnly)
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf:
            directory.appendingPathComponent("tiny-validation.json"))) as! [String: Any]
        let vectors = root["vectors"] as! [[String: Any]]
        var state = model.initialState()
        for item in vectors {
            let name = item["name"] as! String, tokens = item["tokens"] as! [Int]
            if ["prefill", "sequential_0", "grouped_prefill", "chunk_0_13"].contains(name) {
                state = model.initialState()
            }
            let expected = (item["coreml_logits"] as! [[[NSNumber]]])[0]
            for (index, token) in tokens.enumerated() {
                let actual = try model.forward([token], state: &state)
                let target = expected[index].map(\.floatValue)
                XCTAssertEqual(actual.count, target.count)
                let error = zip(actual, target).map { abs($0 - $1) }.max() ?? .infinity
                XCTAssertLessThan(error, 2e-5, "\(name), token \(index)")
                XCTAssertEqual(actual.indices.max { actual[$0] < actual[$1] },
                               target.indices.max { target[$0] < target[$1] })
            }
            XCTAssertEqual(state.layers.count, 16)
            for (visit, layer) in model.config.layerVisits.enumerated() {
                let cache = try XCTUnwrap(state.layers[visit])
                let expectedLength = model.config.localLayers.contains(layer) ? min(state.position, 4) : state.position
                XCTAssertEqual(InklingCoreMLTensor.shape(cache.keys)[2], expectedLength)
                XCTAssertEqual(cache.histories.count, 4)
                XCTAssertTrue(cache.histories.allSatisfy { $0.dataType == .float32 })
            }
        }
    }

    func testInvalidInputAndCancellationLeaveRequestUnadvanced() throws {
        let model = try Inkling2CoreMLModel(directory: directory(), contextLimit: 64, computeUnits: .cpuOnly)
        var state = model.initialState()
        for tokens in [[], [-1], [64], [1, 2]] {
            XCTAssertThrowsError(try model.forward(tokens, state: &state))
            XCTAssertEqual(state.position, 0)
        }
        XCTAssertThrowsError(try model.forward([1], state: &state,
            checkCancellation: { throw CancellationError() })) { error in
            XCTAssertTrue(error is CancellationError)
        }
        XCTAssertEqual(state.position, 0)
        XCTAssertTrue(state.layers.allSatisfy { $0 == nil })
    }

    func testPortableCompilationArtifactsFollowModelLifetime() throws {
        let source = try directory()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for name in ["config.json", "coreml-manifest.json", "embedding.mlpackage", "head.mlpackage"]
            + (0..<8).map({ "layer-\($0).mlpackage" }) {
            try FileManager.default.copyItem(at: source.appendingPathComponent(name),
                                            to: folder.appendingPathComponent(name))
        }
        var temporary: URL?
        try autoreleasepool {
            let model = try Inkling2CoreMLModel(directory: folder, contextLimit: 64, computeUnits: .cpuOnly)
            temporary = try XCTUnwrap(model.temporaryCompilationDirectory)
            XCTAssertTrue(FileManager.default.fileExists(atPath: temporary!.path))
            var state = model.initialState()
            XCTAssertEqual(try model.forward([1], state: &state).count, 61)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: try XCTUnwrap(temporary).path))

        let tempRoot = FileManager.default.temporaryDirectory
        let ownedDirectories = {
            try Set(FileManager.default.contentsOfDirectory(atPath: tempRoot.path)
                .filter { $0.hasPrefix("speech-swift-inkling2-coreml-") })
        }
        let before = try ownedDirectories()
        try FileManager.default.removeItem(at: folder.appendingPathComponent("head.mlpackage"))
        XCTAssertThrowsError(try Inkling2CoreMLModel(directory: folder, contextLimit: 64, computeUnits: .cpuOnly))
        XCTAssertEqual(try ownedDirectories(), before)
    }
}
