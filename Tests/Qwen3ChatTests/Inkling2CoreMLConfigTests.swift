import XCTest
import CoreML
@testable import Qwen3Chat

final class Inkling2CoreMLConfigTests: XCTestCase {
    private func architecture() throws -> Inkling2Config {
        try Inkling2Config(data: JSONSerialization.data(withJSONObject: Inkling2ConfigTests.fixture()))
    }
    private func manifest() -> [String: Any] {
        let names = ["embedding", "head"] + (0..<6).map { "layer-\($0)" }
        let parts = Dictionary(uniqueKeysWithValues: names.map {
            ($0, ["package": "\($0).mlpackage", "compiled": "compiled/\($0).mlmodelc"])
        })
        return ["format": Inkling2CoreMLManifest.format, "status": "converted", "weight_bits": 8,
            "activation_dtype": "float16", "convolution_history_dtype": "float32", "physical_layers": 6,
            "context_limit": 64, "prefill_size": 1, "image_patch_limit": 0, "expert_group_size": 32, "parts": parts]
    }
    private func parse(_ root: [String: Any]) throws -> Inkling2CoreMLManifest {
        try Inkling2CoreMLManifest(data: JSONSerialization.data(withJSONObject: root), architecture: architecture())
    }
    func testINT8ManifestPreservesVisitArchitectureAndHistoryPrecision() throws {
        let parsed = try parse(manifest())
        XCTAssertEqual(parsed.parts.count, 8)
        XCTAssertEqual(parsed.contextLimit, 64)
        XCTAssertEqual(parsed.prefillSize, 1)
        XCTAssertEqual(parsed.activationType, .float16)
        XCTAssertEqual(parsed.groupSize, 32)
    }
    func testRejectsIncompleteAndUnsupportedPrecision() {
        for (key, value) in [("status", "in_progress"), ("convolution_history_dtype", "float16"),
                             ("activation_dtype", "int8"), ("format", "unsupported")] {
            var root = manifest(); root[key] = value
            XCTAssertThrowsError(try parse(root))
        }
        var root = manifest(); root["weight_bits"] = 4
        XCTAssertThrowsError(try parse(root))
    }
    func testFP32ActivationsAreAcceptedWithINT8Storage() throws {
        var root = manifest(); root["activation_dtype"] = "float32"
        XCTAssertEqual(try parse(root).activationType, .float32)
    }
    func testRejectsMissingLayersAndWrongVisitPlan() {
        var root = manifest()
        var parts = root["parts"] as! [String: [String: String]]
        parts.removeValue(forKey: "layer-3"); root["parts"] = parts
        XCTAssertThrowsError(try parse(root))
        root = manifest(); root["layer_visits"] = [0, 1, 2, 3, 4, 5]
        XCTAssertThrowsError(try parse(root))
    }
    func testRejectsPathsOutsideBundle() {
        for invalid in ["../layer-0.mlpackage", "/tmp/layer-0.mlpackage", "nested/layer-0.mlpackage"] {
            var root = manifest()
            var parts = root["parts"] as! [String: [String: String]]
            parts["layer-0"]!["package"] = invalid; root["parts"] = parts
            XCTAssertThrowsError(try parse(root))
        }
        var root = manifest()
        var parts = root["parts"] as! [String: [String: String]]
        parts["head"]!["compiled"] = "../head.mlmodelc"; root["parts"] = parts
        XCTAssertThrowsError(try parse(root))
    }
    func testRejectsInvalidBudgets() {
        for (key, value) in [("context_limit", 0), ("prefill_size", 0), ("prefill_size", 65),
                             ("image_patch_limit", 1025), ("expert_group_size", 16)] {
            var root = manifest(); root[key] = value
            XCTAssertThrowsError(try parse(root))
        }
    }
    func testContextRejectedBeforeLoadingCoreMLModels() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: Inkling2ConfigTests.fixture()).write(to: directory.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: manifest()).write(to: directory.appendingPathComponent("coreml-manifest.json"))
        for context in [0, -1, 65] {
            XCTAssertThrowsError(try Inkling2CoreMLChat.fromDirectory(directory, contextLimit: context)) { error in
                XCTAssertTrue(error is Inkling2Error)
                XCTAssertTrue(String(describing: error).contains("context"))
            }
        }
    }
    func testTensorSlicesKeepChronologicalCacheOrder() throws {
        let values = (0..<24).map(Float.init)
        let source = try InklingCoreMLTensor.make([1, 2, 3, 4], type: .float16, values: values)
        let tail = try InklingCoreMLTensor.slice(source, axis: 2, range: 1..<3)
        XCTAssertEqual(InklingCoreMLTensor.shape(tail), [1, 2, 2, 4])
        XCTAssertEqual(try InklingCoreMLTensor.floats(tail), Array(values[4..<12]) + Array(values[16..<24]))
    }
    func testNoncontiguousCoreMLOutputsKeepLogicalElementOrder() throws {
        let backing = try InklingCoreMLTensor.make([8], values: (0..<8).map(Float.init))
        defer { withExtendedLifetime(backing) {} }
        let view = try MLMultiArray(dataPointer: backing.dataPointer, shape: [3, 2], dataType: .float32,
                                    strides: [1, 5], deallocator: { _ in })
        XCTAssertEqual(try InklingCoreMLTensor.floats(view), [0, 5, 1, 6, 2, 7])
        let tail = try InklingCoreMLTensor.slice(view, axis: 0, range: 1..<3)
        XCTAssertEqual(try InklingCoreMLTensor.floats(tail), [1, 6, 2, 7])
    }
    func testResidualBankPreservesTokenAndBlockOrdering() throws {
        let a = try InklingCoreMLTensor.make([1, 2, 1, 2], values: [1, 2, 3, 4])
        let b = try InklingCoreMLTensor.make([1, 2, 2, 2], values: [10, 11, 12, 13, 20, 21, 22, 23])
        let result = try InklingCoreMLTensor.bank([a, b], tokens: 2, width: 2)
        XCTAssertEqual(try InklingCoreMLTensor.floats(result), [1, 2, 10, 11, 12, 13, 3, 4, 20, 21, 22, 23])
    }
    func testTensorSizeGuardsRejectBeforeAllocation() {
        XCTAssertThrowsError(try InklingCoreMLTensor.make([0, 10]))
        XCTAssertThrowsError(try InklingCoreMLTensor.make([Int.max, 2]))
        XCTAssertThrowsError(try InklingCoreMLTensor.make([2], values: [1]))
        XCTAssertThrowsError(try InklingCoreMLTensor.integers([], shape: [Int.max, 2]))
        XCTAssertThrowsError(try InklingCoreMLTensor.integers([1], shape: [-1, -1]))
        XCTAssertThrowsError(try InklingCoreMLTensor.integers([Int.max], shape: [1]))
    }
}
