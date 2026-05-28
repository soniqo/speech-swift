import XCTest
@testable import AudioCommon

final class HuggingFaceDownloaderTests: XCTestCase {

    // MARK: - offlineMode

    func testOfflineModeSkipsDownloadWhenWeightsExist() async throws {
        // Create a temp directory with a fake safetensors file
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let fakeWeights = tmpDir.appendingPathComponent("model.safetensors")
        try Data([0x00]).write(to: fakeWeights)

        // offlineMode=true should return immediately without network
        var progressReported = false
        try await HuggingFaceDownloader.downloadWeights(
            modelId: "fake/model",
            to: tmpDir,
            offlineMode: true,
            progressHandler: { progress in
                if progress >= 1.0 { progressReported = true }
            }
        )
        XCTAssertTrue(progressReported, "Progress should reach 1.0 in offline mode")
    }

    func testOfflineModeWithoutWeightsFallsThrough() async {
        // Empty directory — offlineMode should still attempt download (and fail)
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_empty_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: "nonexistent/model-that-does-not-exist",
                to: tmpDir,
                offlineMode: true
            )
            XCTFail("Should have thrown an error for nonexistent model")
        } catch {
            // Expected — no cached weights, so download is attempted and fails
        }
    }

    func testOfflineModeFalseDoesNotSkip() async {
        // offlineMode=false (default) should not skip even if weights exist
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_false_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let fakeWeights = tmpDir.appendingPathComponent("model.safetensors")
        try? Data([0x00]).write(to: fakeWeights)

        // offlineMode=false should attempt network (and fail for fake model)
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: "nonexistent/model-that-does-not-exist",
                to: tmpDir,
                offlineMode: false
            )
            XCTFail("Should have thrown for nonexistent model with offlineMode=false")
        } catch {
            // Expected — network download attempted and failed
        }
    }

    // MARK: - weightsExist

    func testWeightsExistReturnsTrueForSafetensors() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_exist_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: tmpDir))

        let fakeWeights = tmpDir.appendingPathComponent("model.safetensors")
        try Data([0x00]).write(to: fakeWeights)

        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    func testWeightsExistReturnsFalseForEmptyDirectory() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_empty_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    func testWeightsExistReturnsFalseForNonexistentDirectory() {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent_\(UUID().uuidString)")
        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    // MARK: - weightsExist — Apple CoreML bundle layouts

    /// CoreML-only repositories (e.g. `aufklarer/WeSpeaker-ResNet34-LM-CoreML`)
    /// ship a `.mlmodelc/` directory and no `.safetensors` files. The
    /// pre-fix `weightsExist` returned false for this layout, causing
    /// every `offlineMode: true` load to fall through to `hub.snapshot()`
    /// — which in turn issued an HTTP HEAD to huggingface.co even when
    /// every byte of the model was on disk.
    func testWeightsExistReturnsTrueForMlmodelcDirectory() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_mlmodelc_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: tmpDir))

        let mlmodelc = tmpDir.appendingPathComponent("wespeaker.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: mlmodelc, withIntermediateDirectories: true)

        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir),
            "Directories ending in .mlmodelc must satisfy weightsExist — that's the cached-CoreML layout HF ships")
    }

    /// Multi-component CoreML models (e.g. Parakeet's encoder + decoder + joint)
    /// ship multiple `.mlmodelc/` directories under the same repo. The
    /// existence check fires on any one of them.
    func testWeightsExistReturnsTrueForMultipleMlmodelcDirs() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_multi_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        for name in ["encoder.mlmodelc", "decoder.mlmodelc", "joint.mlmodelc"] {
            try FileManager.default.createDirectory(
                at: tmpDir.appendingPathComponent(name, isDirectory: true),
                withIntermediateDirectories: true
            )
        }

        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    /// `.mlpackage/` is the uncompiled CoreML container. Less common
    /// in HF caches but recognised for symmetry.
    func testWeightsExistReturnsTrueForMlpackageDirectory() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_mlpackage_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let mlpackage = tmpDir.appendingPathComponent("model.mlpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: mlpackage, withIntermediateDirectories: true)

        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    /// Mixed-layout repos that ship both `.safetensors` and `.mlmodelc/`
    /// (rare but possible) must continue to satisfy `weightsExist`.
    /// Pins that the broadened recogniser doesn't accidentally introduce
    /// a regression on the canonical safetensors path.
    func testWeightsExistReturnsTrueForMixedSafetensorsAndMlmodelc() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_mixed_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Data([0x00]).write(to: tmpDir.appendingPathComponent("model.safetensors"))
        try FileManager.default.createDirectory(
            at: tmpDir.appendingPathComponent("encoder.mlmodelc", isDirectory: true),
            withIntermediateDirectories: true
        )

        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    /// A directory containing only unrelated files (`config.json`,
    /// `.cache/`, `tokenizer.json`) must NOT satisfy `weightsExist`.
    /// Preserves the "incomplete cache → fall through to download"
    /// semantics that downstream consumers rely on.
    func testWeightsExistReturnsFalseForDirectoryWithoutWeightFiles() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("weights_unrelated_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Data("{}".utf8).write(to: tmpDir.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: tmpDir.appendingPathComponent("tokenizer.json"))
        try FileManager.default.createDirectory(
            at: tmpDir.appendingPathComponent(".cache", isDirectory: true),
            withIntermediateDirectories: true
        )

        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: tmpDir),
            "Unrelated files (config, tokenizer, .cache) must NOT satisfy weightsExist — preserves 'incomplete cache → download' semantics")
    }

    // MARK: - offlineMode integration with CoreML caches

    /// The behavioural counterpart to `testOfflineModeSkipsDownloadWhenWeightsExist`
    /// — verifies that `downloadWeights(offlineMode: true)` short-circuits
    /// (no network) when ONLY `.mlmodelc/` directories are present, without
    /// any `.safetensors` files. This is the field-reported scenario that
    /// motivated the patch.
    func testOfflineModeSkipsDownloadWhenMlmodelcExists() async throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_mlmodelc_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Populate cache with the WeSpeaker-style single-mlmodelc layout
        // (no safetensors). Pre-fix, this would NOT short-circuit.
        let mlmodelc = tmpDir.appendingPathComponent("wespeaker.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: mlmodelc, withIntermediateDirectories: true)

        var progressReported = false
        try await HuggingFaceDownloader.downloadWeights(
            modelId: "fake/coreml-only-model",
            to: tmpDir,
            offlineMode: true,
            progressHandler: { progress in
                if progress >= 1.0 { progressReported = true }
            }
        )
        XCTAssertTrue(progressReported,
            "offlineMode: true must short-circuit (no network) when only .mlmodelc/ caches are present — same contract as for .safetensors")
    }

    // MARK: - cacheDir (custom cache directory)

    func testCustomCacheDirSkipsDefaultResolution() async throws {
        // Create a temp directory with a fake safetensors file
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("custom_cache_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let fakeWeights = tmpDir.appendingPathComponent("model.safetensors")
        try Data([0x00]).write(to: fakeWeights)

        // With custom cacheDir + offlineMode, should succeed without any network or default path resolution
        var progressReported = false
        try await HuggingFaceDownloader.downloadWeights(
            modelId: "fake/model",
            to: tmpDir,
            offlineMode: true,
            progressHandler: { progress in
                if progress >= 1.0 { progressReported = true }
            }
        )
        XCTAssertTrue(progressReported)
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: tmpDir))
    }

    // MARK: - Download stall guard

    /// A stalled operation (reports progress once, then sleeps forever)
    /// must be aborted by the guard rather than hanging. Uses a 1 s
    /// stall window so the test is fast.
    func testStallGuardAbortsWedgedDownload() async throws {
        let start = Date()
        do {
            try await HuggingFaceDownloader.withDownloadStallGuard(
                modelId: "fake/wedged", stallTimeoutSeconds: 1
            ) { reportProgress in
                reportProgress(0.1)  // one tick, then never again
                try await Task.sleep(for: .seconds(60))  // simulate a wedged transfer
            }
            XCTFail("expected stall guard to throw")
        } catch let error as DownloadError {
            guard case .stalled(let modelId, let seconds) = error else {
                return XCTFail("expected .stalled, got \(error)")
            }
            XCTAssertEqual(modelId, "fake/wedged")
            XCTAssertEqual(seconds, 1)
        }
        // Must abort within a few seconds, not wait out the 60 s sleep.
        XCTAssertLessThan(Date().timeIntervalSince(start), 10,
                          "stall guard should abort promptly after the window")
    }

    /// An operation that keeps reporting progress must NOT be tripped by
    /// the guard even when it runs longer than the stall window.
    func testStallGuardAllowsProgressingDownload() async throws {
        try await HuggingFaceDownloader.withDownloadStallGuard(
            modelId: "fake/healthy", stallTimeoutSeconds: 1
        ) { reportProgress in
            // Tick every 200 ms for ~1.6 s (> the 1 s stall window) so the
            // clock keeps resetting; should complete without stalling.
            for i in 1...8 {
                try await Task.sleep(for: .milliseconds(200))
                reportProgress(Double(i) / 8.0)
            }
        }
    }
}
