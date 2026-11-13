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
        for variant in ["INT8", "FP16"] {
            XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
                modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-\(variant)",
                computeUnits: .cpuAndGPU), "text_decoder")
        }
    }

    func testCustomModelDownloadNamesRemainCompatible() {
        for units in [MLComputeUnits.cpuOnly, .cpuAndGPU, .cpuAndNeuralEngine, .all] {
            XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
                modelId: "example/custom-aligner", computeUnits: units), "text_decoder")
        }
    }

    func testPublishedNeuralEngineDownloadsUseVersionedRepairedBundles() {
        for variant in ["INT8", "FP16"] {
            for units in [MLComputeUnits.cpuAndNeuralEngine, .all] {
                XCTAssertEqual(CoreMLForcedAlignerDecoder.downloadedBundleName(
                    modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-\(variant)",
                    computeUnits: units), "text_decoder_ane_v1")
            }
        }
    }

    func testOldCachedDecoderCannotMaskNeuralEngineRepair() throws {
        _ = try makeBundle("text_decoder")
        _ = try makeBundle("text_decoder_cpu_v1")
        let repaired = try makeBundle("text_decoder_ane_v1")
        for units in [MLComputeUnits.cpuAndNeuralEngine, .all] {
            XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
                in: directory, computeUnits: units), repaired)
        }
    }

    func testRepairedNeuralEngineSourceWinsOverLegacyCompiledDecoder() throws {
        _ = try makeBundle("text_decoder")
        let repaired = try makeBundle("text_decoder_ane_v1", suffix: "mlpackage")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuAndNeuralEngine), repaired)
    }

    func testCompiledNeuralEngineRepairWinsOverItsSource() throws {
        _ = try makeBundle("text_decoder_ane_v1", suffix: "mlpackage")
        let repaired = try makeBundle("text_decoder_ane_v1")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuAndNeuralEngine), repaired)
    }

    func testPublishedNeuralEngineRequiresRepairEvenWithLegacyCache() throws {
        _ = try makeBundle("text_decoder")
        _ = try makeBundle("text_decoder_cpu_v1")
        for units in [MLComputeUnits.cpuAndNeuralEngine, .all] {
            XCTAssertThrowsError(try CoreMLForcedAlignerDecoder.resolveDownloadedBundleURL(
                in: directory, modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
                computeUnits: units)) { error in
                XCTAssertTrue(String(describing: error).contains("text_decoder_ane_v1"))
            }
        }
    }

    func testOfflinePublishedNeuralEngineCacheFailsBeforeLoadingModels() async throws {
        _ = try makeBundle("text_decoder")
        do {
            _ = try await CoreMLForcedAligner.fromPretrained(
                modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-INT8",
                decoderComputeUnits: .cpuAndNeuralEngine,
                cacheDir: directory, offlineMode: true)
            XCTFail("Legacy offline caches must not load the faulty Neural Engine decoder")
        } catch {
            XCTAssertTrue(String(describing: error).contains("text_decoder_ane_v1"))
        }
    }

    func testPublishedNeuralEngineCanLoadWithoutTheLegacyDecoder() throws {
        let repaired = try makeBundle("text_decoder_ane_v1")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveDownloadedBundleURL(
            in: directory, modelId: "aufklarer/Qwen3-ForcedAligner-0.6B-CoreML-FP16",
            computeUnits: .cpuAndNeuralEngine), repaired)
    }

    func testCustomNeuralEngineExportsRetainLegacyCompatibility() throws {
        let original = try makeBundle("text_decoder")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveDownloadedBundleURL(
            in: directory, modelId: "example/custom-aligner",
            computeUnits: .cpuAndNeuralEngine), original)
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
        _ = try makeBundle("text_decoder_ane_v1")
        XCTAssertEqual(try CoreMLForcedAlignerDecoder.resolveBundleURL(
            in: directory, computeUnits: .cpuAndGPU), original)
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
