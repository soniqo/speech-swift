#if canImport(CoreML)
import CoreML
import Foundation
import XCTest
@testable import Qwen3ASR

/// Bundle selection and cache migration, without loading model weights.
final class CoreMLForcedAlignerBundleTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try FileManager.default.removeItem(at: directory)
    }

    private func makeBundle(_ name: String, suffix: String = "mlmodelc") throws -> URL {
        let url = directory.appendingPathComponent("\(name).\(suffix)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    func testPublishedCPUDownloadsUseVersionedRepairedBundles() {
        for variant in ["INT8", "FP16"] {
            XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
                modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-\(variant)",
                computeUnits: .cpuOnly), "text_decoder_cpu_v1")
        }
    }

    func testPublishedGPUDownloadsKeepTheExistingDecoder() {
        for units in [MLComputeUnits.cpuAndGPU, .cpuAndNeuralEngine, .all] {
            XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
                modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
                computeUnits: units), "text_decoder")
        }
    }

    func testCustomModelDownloadNamesRemainCompatible() {
        XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
            modelId: "example/custom-aligner", computeUnits: .cpuOnly), "text_decoder")
    }

    func testOldCachedDecoderCannotMaskCPURepair() throws {
        _ = try makeBundle("text_decoder")
        let repaired = try makeBundle("text_decoder_cpu_v1")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly), repaired)
    }

    func testGPUStillUsesOriginalGraphWhenBothBundlesAreCached() throws {
        let original = try makeBundle("text_decoder")
        _ = try makeBundle("text_decoder_cpu_v1")
        for units in [MLComputeUnits.cpuAndGPU, .cpuAndNeuralEngine, .all] {
            XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
                in: directory, computeUnits: units), original)
        }
    }

    func testRepairedCPUCanLoadWithoutTheLegacyDecoder() throws {
        let repaired = try makeBundle("text_decoder_cpu_v1")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly), repaired)
    }

    func testRepairedSourcePackageIsPreferredOverOldCompiledDecoder() throws {
        _ = try makeBundle("text_decoder")
        let repaired = try makeBundle("text_decoder_cpu_v1", suffix: "mlpackage")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly), repaired)
    }

    func testCompiledRepairedBundleIsPreferredOverItsSource() throws {
        let repaired = try makeBundle("text_decoder_cpu_v1")
        _ = try makeBundle("text_decoder_cpu_v1", suffix: "mlpackage")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly), repaired)
    }

    func testLegacyLocalCPUExportsRemainLoadable() throws {
        let original = try makeBundle("text_decoder")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly), original)
    }

    func testMissingDecoderThrowsInsteadOfChoosingAnUnrelatedBundle() throws {
        _ = try makeBundle("unrelated")
        XCTAssertThrowsError(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuOnly))
    }
}
#endif
