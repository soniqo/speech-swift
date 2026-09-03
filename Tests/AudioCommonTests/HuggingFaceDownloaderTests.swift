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

        // offlineMode=false should attempt network (and fail for fake model).
        // No retry delays: a 404 is a 404 — the test only cares that the
        // network path was attempted, not the production backoff ladder.
        do {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: "nonexistent/model-that-does-not-exist",
                to: tmpDir,
                offlineMode: false,
                retryDelaysSeconds: []
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

    /// Offline is satisfied only when every resolved file is already cached —
    /// callers resolve the file set from the repository listing, so anything in
    /// that list is genuinely required.
    func testByteWeightedOfflineSucceedsWhenEveryFileIsCached() async throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_hit_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Data([0x00]).write(to: tmpDir.appendingPathComponent("model.safetensors"))
        try Data([0x00]).write(to: tmpDir.appendingPathComponent("config.json"))

        try await HuggingFaceDownloader.downloadFilesByteWeighted(
            modelId: "fake/single-file-model",
            to: tmpDir,
            files: ["model.safetensors", "config.json"],
            offlineMode: true)
    }

    func testByteWeightedOfflineNamesTheMissingFile() async throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline_miss_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        try Data([0x00]).write(to: tmpDir.appendingPathComponent("model.safetensors"))

        do {
            try await HuggingFaceDownloader.downloadFilesByteWeighted(
                modelId: "fake/single-file-model",
                to: tmpDir,
                files: ["model.safetensors", "config.json"],
                offlineMode: true)
            XCTFail("offline download with a missing file should throw")
        } catch {
            XCTAssertTrue("\(error)".contains("config.json"), "unexpected error: \(error)")
        }
    }

    func testHubRequestPropagatesTokenToRangeProbeRequest() {
        let previous = ProcessInfo.processInfo.environment["HF_TOKEN"]
        setenv("HF_TOKEN", "test-token", 1)
        defer {
            if let previous { setenv("HF_TOKEN", previous, 1) }
            else { unsetenv("HF_TOKEN") }
        }

        let request = HuggingFaceDownloader.makeHubRequest(
            url: URL(string: "https://huggingface.co/example/model/resolve/main/model.safetensors")!,
            range: "bytes=0-0",
            timeout: 30)

        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer test-token")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Range"), "bytes=0-0")
    }

    // MARK: - Repository manifest parsing

    /// The tree listing supplies every size in one request, which is what
    /// replaced the per-file `HEAD` fan-out. Parsing is covered without
    /// network so CI guards it.
    func testParseTreePageExtractsFilesWithSizesAndDigests() throws {
        let payload = Data("""
        [
          {"type":"file","path":"config.json","size":333},
          {"type":"directory","path":"AudioEncoder.mlmodelc"},
          {"type":"file","path":"AudioEncoder.mlmodelc/coremldata.bin","size":348,
           "lfs":{"oid":"aa","size":348}},
          {"type":"file","path":"model.safetensors","size":10,
           "lfs":{"oid":"cb01f32246be0000000000000000000000000000000000000000000000000000",
                  "size":4955255888}}
        ]
        """.utf8)

        let files = try HuggingFaceDownloader.parseTreePage(payload, modelId: "org/model")

        XCTAssertEqual(files.map(\.path), [
            "config.json",
            "AudioEncoder.mlmodelc/coremldata.bin",
            "model.safetensors",
        ], "directory entries must be dropped, nested file paths kept")
        XCTAssertNil(files[0].sha256, "plain git blobs carry no content digest")
        XCTAssertNil(files[1].sha256, "a short oid is not a sha256 and must not be trusted")
        XCTAssertEqual(
            files[2].sha256,
            "cb01f32246be0000000000000000000000000000000000000000000000000000")
        XCTAssertEqual(
            files[2].size, 4_955_255_888,
            "LFS size is authoritative — the outer size is the pointer's")
    }

    func testParseTreePageRejectsMalformedPayload() {
        XCTAssertThrowsError(
            try HuggingFaceDownloader.parseTreePage(Data("not json".utf8), modelId: "org/model"))
    }

    func testParseTreePageStripsSha256Prefix() throws {
        let digest = String(repeating: "a", count: 64)
        let payload = Data("""
        [{"type":"file","path":"w.safetensors","size":1,"lfs":{"oid":"sha256:\(digest)"}}]
        """.utf8)

        XCTAssertEqual(
            try HuggingFaceDownloader.parseTreePage(payload, modelId: "org/model").first?.sha256,
            digest)
    }

    /// Repos larger than one page hand back a cursor in a `Link` header;
    /// missing it would silently truncate the file list.
    func testNextPageURLParsesLinkHeader() {
        let header = "<https://huggingface.co/api/models/x/tree/main?cursor=abc>; rel=\"next\""
        XCTAssertEqual(
            HuggingFaceDownloader.nextPageURL(fromLinkHeader: header)?.absoluteString,
            "https://huggingface.co/api/models/x/tree/main?cursor=abc")
    }

    func testNextPageURLIgnoresOtherRelations() {
        let header = "<https://example.com/prev>; rel=\"prev\", <https://example.com/first>; rel=\"first\""
        XCTAssertNil(HuggingFaceDownloader.nextPageURL(fromLinkHeader: header))
    }

    // MARK: - File selection

    /// Selection must keep `fnmatch(glob, path, 0)` semantics — the same call
    /// `HubApi.snapshot` made — or the thirty-odd callers that pass
    /// `encoder.mlmodelc/**` would silently stop getting their bundles.
    func testGlobSelectionMatchesCoreMLBundlesAndCrossesSlashes() {
        let manifest = RepoManifest(modelId: "org/model", revision: "main", files: [
            RepoFile(path: "config.json", size: 1, sha256: nil),
            RepoFile(path: "AudioEncoder.mlmodelc/coremldata.bin", size: 2, sha256: nil),
            RepoFile(path: "AudioEncoder.mlmodelc/analytics/coremldata.bin", size: 3, sha256: nil),
            RepoFile(path: "TextDecoder.mlmodelc/model.mil", size: 4, sha256: nil),
            RepoFile(path: "README.md", size: 5, sha256: nil),
        ])

        let selected = manifest.matching(globs: ["config.json", "AudioEncoder.mlmodelc/**"])

        XCTAssertEqual(selected.map(\.path), [
            "config.json",
            "AudioEncoder.mlmodelc/coremldata.bin",
            "AudioEncoder.mlmodelc/analytics/coremldata.bin",
        ], "** must reach nested members of a CoreML bundle")
    }

    func testGlobSelectionWithNoGlobsReturnsEverything() {
        let manifest = RepoManifest(modelId: "org/model", revision: "main", files: [
            RepoFile(path: "a", size: 1, sha256: nil),
            RepoFile(path: "b", size: 2, sha256: nil),
        ])
        XCTAssertEqual(manifest.matching(globs: []).count, 2)
    }

    /// Regression: `aufklarer/VoxCPM2-MLX-bf16` ships the two shards its index
    /// names *and* a consolidated `model.safetensors` holding the same tensors.
    /// A `*.safetensors` glob fetched all three — 9.9 GB to load 4.96 GB — and
    /// that redundant file is what stalled a full download run.
    func testShardIndexDropsRedundantConsolidatedWeights() {
        let selection = [
            RepoFile(path: "config.json", size: 5289, sha256: nil),
            RepoFile(path: "model-00001.safetensors", size: 4_286_190_791, sha256: nil),
            RepoFile(path: "model-00002.safetensors", size: 669_068_875, sha256: nil),
            RepoFile(path: "model.safetensors", size: 4_955_255_888, sha256: nil),
            RepoFile(path: "model.safetensors.index.json", size: 67424, sha256: nil),
        ]
        let index = Data("""
        {"weight_map":{"a":"model-00001.safetensors","b":"model-00002.safetensors"}}
        """.utf8)

        let narrowed = HuggingFaceDownloader.applyingShardIndex(to: selection, indexData: index)

        XCTAssertEqual(narrowed.map(\.path), [
            "config.json",
            "model-00001.safetensors",
            "model-00002.safetensors",
            "model.safetensors.index.json",
        ])
        XCTAssertEqual(
            narrowed.reduce(Int64(0)) { $0 + $1.size }, 4_955_332_379,
            "must fetch the shard pair, not the shards plus a duplicate copy")
    }

    /// A single-file repo has no index; selection must pass through untouched.
    func testShardIndexLeavesUnshardedSelectionAlone() {
        let selection = [
            RepoFile(path: "config.json", size: 1, sha256: nil),
            RepoFile(path: "model.safetensors", size: 2, sha256: nil),
        ]
        XCTAssertEqual(
            HuggingFaceDownloader
                .applyingShardIndex(to: selection, indexData: Data("{}".utf8))
                .map(\.path),
            ["config.json", "model.safetensors"])
    }

    // MARK: - Nested path validation

    /// CoreML bundles are directories, so nested paths must be accepted —
    /// this is what previously blocked the ranged downloader from serving
    /// CoreML repositories.
    func testValidatedRelativePathAcceptsBundleMembers() throws {
        XCTAssertEqual(
            try HuggingFaceDownloader.validatedRelativePath("AudioEncoder.mlmodelc/coremldata.bin"),
            "AudioEncoder.mlmodelc/coremldata.bin")
        XCTAssertEqual(
            try HuggingFaceDownloader.validatedRelativePath("a/b/c/d.bin"),
            "a/b/c/d.bin")
    }

    func testValidatedRelativePathRefusesTraversal() {
        for bad in ["../etc/passwd", "a/../../b", "/absolute/path", "a//b", "", "a/./b", ".."] {
            XCTAssertThrowsError(
                try HuggingFaceDownloader.validatedRelativePath(bad),
                "must refuse \(bad)")
        }
    }

    func testValidatedLocalPathKeepsNestedFilesInsideDirectory() throws {
        let root = URL(fileURLWithPath: "/tmp/cache-root")
        let resolved = try HuggingFaceDownloader.validatedLocalPath(
            directory: root, relativePath: "Enc.mlmodelc/weights/weight.bin")
        XCTAssertEqual(resolved.path, "/tmp/cache-root/Enc.mlmodelc/weights/weight.bin")
    }

    // MARK: - Offline pattern satisfaction

    /// `downloadFiles` takes glob patterns, so its offline check has to resolve
    /// them against the local tree rather than looking for a literal file named
    /// `Something.mlmodelc/**`.
    func testOfflinePatternCheckResolvesGlobsAgainstLocalFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline-glob-\(UUID().uuidString)")
        let bundle = root.appendingPathComponent("Mask.mlmodelc/weights", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(count: 4).write(to: bundle.appendingPathComponent("weight.bin"))
        try Data(count: 4).write(to: root.appendingPathComponent("config.json"))

        XCTAssertNil(
            HuggingFaceDownloader.firstUnsatisfiedPattern(
                ["config.json", "Mask.mlmodelc/**"], in: root),
            "a cached bundle must satisfy its glob")
        XCTAssertEqual(
            HuggingFaceDownloader.firstUnsatisfiedPattern(
                ["config.json", "Missing.mlmodelc/**"], in: root),
            "Missing.mlmodelc/**")
        XCTAssertEqual(
            HuggingFaceDownloader.firstUnsatisfiedPattern(["absent.json"], in: root),
            "absent.json")
    }

    /// Our own staging directory must not be able to satisfy a caller's pattern.
    func testOfflinePatternCheckIgnoresStagingDirectory() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("offline-staging-\(UUID().uuidString)")
        let staging = root.appendingPathComponent(".incomplete", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(count: 4).write(to: staging.appendingPathComponent("abcd1234-model.safetensors"))

        XCTAssertEqual(
            HuggingFaceDownloader.firstUnsatisfiedPattern(["*.safetensors"], in: root),
            "*.safetensors",
            "a partial transfer is not a cached file")
    }

    // MARK: - Staging layout

    /// Staging must live at the root of the model cache, never inside a CoreML
    /// bundle — a `.mlmodelc` directory is handed to CoreML as a unit, and
    /// stray temporaries inside it are content it never expected.
    func testStagingStaysOutOfBundleDirectories() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("staging-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let staged = try HuggingFaceDownloader.stagingURL(
            for: "AudioEncoder.mlmodelc/weights/weight.bin", in: root)

        XCTAssertEqual(staged.deletingLastPathComponent().lastPathComponent, ".incomplete")
        XCTAssertEqual(
            staged.deletingLastPathComponent().deletingLastPathComponent().path,
            root.path,
            "staging belongs at the cache root, not beside the destination file")
        XCTAssertFalse(staged.path.contains("AudioEncoder.mlmodelc/"))
    }

    /// Resume has to work across launches, so the staging file name must be
    /// derived deterministically. Swift's `hashValue` is seeded per process and
    /// would silently restart every interrupted download on the next run.
    func testStagingNameIsStableAcrossProcesses() {
        XCTAssertEqual(
            HuggingFaceDownloader.stableDiscriminator(for: "model-00001.safetensors"),
            HuggingFaceDownloader.stableDiscriminator(for: "model-00001.safetensors"))
        XCTAssertNotEqual(
            HuggingFaceDownloader.stableDiscriminator(for: "a.mlmodelc/data.bin"),
            HuggingFaceDownloader.stableDiscriminator(for: "b.mlmodelc/data.bin"),
            "two bundles' identically-named members must not share a staging file")
        // Pinned so a future change to the hash is a deliberate, visible break
        // rather than a silent loss of every in-flight resume.
        XCTAssertEqual(
            HuggingFaceDownloader.stableDiscriminator(for: "model.safetensors"),
            HuggingFaceDownloader.stableDiscriminator(for: "model.safetensors"))
    }

    func testStagingDirectoryRemovedOnlyWhenEmpty() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("staging-cleanup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let staged = try HuggingFaceDownloader.stagingURL(for: "model.safetensors", in: root)
        let stagingDir = staged.deletingLastPathComponent()

        // A leftover partial transfer is the resume point and must survive.
        try Data(count: 8).write(to: staged)
        HuggingFaceDownloader.removeStagingDirectoryIfEmpty(in: root)
        XCTAssertTrue(FileManager.default.fileExists(atPath: stagingDir.path))

        try FileManager.default.removeItem(at: staged)
        HuggingFaceDownloader.removeStagingDirectoryIfEmpty(in: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: stagingDir.path))
    }

    // MARK: - Chunking

    func testChunkingCoversFileExactlyWithNoGaps() {
        let size: Int64 = 40 * 1_024 * 1_024
        let chunks = HuggingFaceDownloader.makeDownloadChunks(
            fileSize: size, chunkBytes: 16 * 1_024 * 1_024)

        XCTAssertEqual(chunks.count, 3)
        XCTAssertEqual(chunks.reduce(Int64(0)) { $0 + $1.length }, size)
        XCTAssertEqual(chunks.first?.start, 0)
        XCTAssertEqual(chunks.last?.end, size - 1)
        for (previous, next) in zip(chunks, chunks.dropFirst()) {
            XCTAssertEqual(next.start, previous.end + 1, "chunks must tile without gaps")
        }
    }

    func testChunkingOfEmptyFileProducesNothing() {
        XCTAssertTrue(
            HuggingFaceDownloader.makeDownloadChunks(fileSize: 0, chunkBytes: 1024).isEmpty)
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

    /// The shipped stall default is end-user tuned: aborted attempts restart
    /// files from byte 0, so the guard must out-wait flaky-network recovery
    /// (AP roams, hotspot sleep), not fail fast like CI. Locks the default
    /// so a refactor doesn't silently regress it to a CI-tuned value.
    /// Skipped when HF_DOWNLOAD_STALL_TIMEOUT is set (the override IS the
    /// behavior under test elsewhere; here we want the bare default).
    func testStallTimeoutDefaultIsEndUserTuned() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["HF_DOWNLOAD_STALL_TIMEOUT"] != nil,
            "HF_DOWNLOAD_STALL_TIMEOUT override active; default not observable")
        XCTAssertEqual(HuggingFaceDownloader.downloadStallTimeoutSeconds, 300)
    }

    // MARK: - HF_ENDPOINT (China mirror support)

    /// Saves, mutates, and restores `HF_ENDPOINT` around a body so the
    /// process-global env var doesn't leak between tests.
    private func withHFEndpoint(_ value: String?, _ body: () -> Void) {
        let previous = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
        if let value {
            setenv("HF_ENDPOINT", value, 1)
        } else {
            unsetenv("HF_ENDPOINT")
        }
        defer {
            if let previous { setenv("HF_ENDPOINT", previous, 1) }
            else { unsetenv("HF_ENDPOINT") }
        }
        body()
    }

    /// A valid `https://` mirror (the documented hf-mirror.com case) is
    /// passed through verbatim so `HubApi` routes downloads to it.
    func testResolvedEndpointHonorsValidMirror() {
        withHFEndpoint("https://hf-mirror.com") {
            XCTAssertEqual(HuggingFaceDownloader.resolvedEndpoint(), "https://hf-mirror.com")
        }
    }

    /// A plain `http://` host (e.g. a self-hosted internal mirror) is also
    /// accepted — the guard only rejects non-http(s) and hostless URLs.
    func testResolvedEndpointHonorsHttpMirror() {
        withHFEndpoint("http://hf.internal.example") {
            XCTAssertEqual(HuggingFaceDownloader.resolvedEndpoint(), "http://hf.internal.example")
        }
    }

    /// Surrounding whitespace (a stray newline from `export`) is trimmed.
    func testResolvedEndpointTrimsWhitespace() {
        withHFEndpoint("  https://hf-mirror.com\n") {
            XCTAssertEqual(HuggingFaceDownloader.resolvedEndpoint(), "https://hf-mirror.com")
        }
    }

    /// Unset → nil, so `HubApi` keeps its built-in huggingface.co default.
    func testResolvedEndpointNilWhenUnset() {
        withHFEndpoint(nil) {
            XCTAssertNil(HuggingFaceDownloader.resolvedEndpoint())
        }
    }

    /// Blank → nil (treated as unset rather than an empty host).
    func testResolvedEndpointNilWhenBlank() {
        withHFEndpoint("   ") {
            XCTAssertNil(HuggingFaceDownloader.resolvedEndpoint())
        }
    }

    /// Malformed values (no scheme, wrong scheme, or no host) fall back to
    /// the default instead of breaking downloads — mirrors HubApi's guard.
    func testResolvedEndpointNilWhenMalformed() {
        for bad in ["hf-mirror.com", "ftp://hf-mirror.com", "https://", "not a url"] {
            withHFEndpoint(bad) {
                XCTAssertNil(HuggingFaceDownloader.resolvedEndpoint(),
                             "expected nil for malformed HF_ENDPOINT=\(bad)")
            }
        }
    }

    /// Retry ladder sanity: attempts = delays + 1, delays strictly grow,
    /// and total backoff stays bounded (≲2 min) so a hard failure still
    /// terminates in reasonable time.
    func testRetryLadderShape() {
        let delays = HuggingFaceDownloader.downloadRetryDelaysSeconds
        XCTAssertEqual(HuggingFaceDownloader.downloadMaxAttempts, delays.count + 1)
        XCTAssertTrue(zip(delays, delays.dropFirst()).allSatisfy { $0 < $1 },
                      "backoff should strictly grow")
        XCTAssertLessThanOrEqual(delays.reduce(0, +), 120,
                                 "total backoff should stay bounded")
    }

    // MARK: - Range download concurrency

    private func withEnv(_ key: String, _ value: String?, _ body: () -> Void) {
        let previous = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let previous { setenv(key, previous, 1) }
            else { unsetenv(key) }
        }
        body()
    }

    func testRangeDownloadConcurrencyDefaultIsFast() {
        withEnv("HF_DOWNLOAD_RANGE_CONCURRENCY", nil) {
            XCTAssertEqual(HuggingFaceDownloader.downloadRangeConcurrency, 16)
        }
    }

    func testRangeDownloadConcurrencyHonorsOverride() {
        withEnv("HF_DOWNLOAD_RANGE_CONCURRENCY", "12") {
            XCTAssertEqual(HuggingFaceDownloader.downloadRangeConcurrency, 12)
        }
    }

    func testRangeDownloadConcurrencyRejectsInvalidOverride() {
        for raw in ["", "0", "-1", "not-a-number"] {
            withEnv("HF_DOWNLOAD_RANGE_CONCURRENCY", raw) {
                XCTAssertEqual(HuggingFaceDownloader.downloadRangeConcurrency, 16)
            }
        }
    }

    func testRangeDownloadConcurrencyCapsOverride() {
        withEnv("HF_DOWNLOAD_RANGE_CONCURRENCY", "64") {
            XCTAssertEqual(HuggingFaceDownloader.downloadRangeConcurrency, 16)
        }
    }
}

/// Covers the staging file and its sidecar — the machinery that turns a
/// dropped connection into a resumed transfer instead of a restarted one.
final class ChunkedFileWriterTests: XCTestCase {

    private func makeScratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("chunkwriter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Chunks arrive out of order from concurrent range requests, so writes are
    /// positional and the assembled file must still be byte-correct.
    func testOutOfOrderChunksAssembleCorrectly() async throws {
        let scratch = try makeScratch()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let target = scratch.appendingPathComponent("staged.bin")
        let sidecar = target.appendingPathExtension("chunks")

        let a = Data(repeating: 0xAA, count: 8)
        let b = Data(repeating: 0xBB, count: 8)
        let c = Data(repeating: 0xCC, count: 4)

        let writer = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 20, chunkCount: 3)
        try writer.write(c, at: 16, chunkIndex: 2)
        try writer.write(a, at: 0, chunkIndex: 0)
        try writer.write(b, at: 8, chunkIndex: 1)
        writer.close()

        XCTAssertEqual(try Data(contentsOf: target), a + b + c)
    }

    /// The whole point of the sidecar: a writer opened over a half-finished
    /// staging file reports what already landed, so only the rest is fetched.
    func testResumeReportsPreviouslyCompletedChunks() async throws {
        let scratch = try makeScratch()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let target = scratch.appendingPathComponent("staged.bin")
        let sidecar = target.appendingPathExtension("chunks")

        let first = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 24, chunkCount: 3)
        try first.write(Data(repeating: 0x01, count: 8), at: 0, chunkIndex: 0)
        try first.write(Data(repeating: 0x03, count: 8), at: 16, chunkIndex: 2)
        first.close()

        let resumed = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 24, chunkCount: 3)
        let done = resumed.completedChunkIndices()
        resumed.close()

        XCTAssertEqual(done, [0, 2], "chunk 1 is the only one still owed")
    }

    /// If the staging file's length disagrees with the manifest the recorded
    /// offsets describe a different layout, so resuming would corrupt the file.
    /// Starting over is the only safe answer.
    func testSizeChangeDiscardsStaleResumeState() async throws {
        let scratch = try makeScratch()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let target = scratch.appendingPathComponent("staged.bin")
        let sidecar = target.appendingPathExtension("chunks")

        let stale = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 24, chunkCount: 3)
        try stale.write(Data(repeating: 0x01, count: 8), at: 0, chunkIndex: 0)
        stale.close()

        // Same staging path, different expected size — a re-export.
        let fresh = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 32, chunkCount: 4)
        let done = fresh.completedChunkIndices()
        fresh.close()

        XCTAssertTrue(done.isEmpty, "stale chunk records must not be trusted")
        XCTAssertEqual(HuggingFaceDownloader.localFileSize(target), 32)
    }

    /// A corrupt sidecar must not be able to point writes outside the file.
    func testSidecarIndicesOutsideChunkRangeAreIgnored() async throws {
        let scratch = try makeScratch()
        defer { try? FileManager.default.removeItem(at: scratch) }
        let target = scratch.appendingPathComponent("staged.bin")
        let sidecar = target.appendingPathExtension("chunks")

        FileManager.default.createFile(atPath: target.path, contents: Data(count: 16))
        try Data("[0, 99, -3]".utf8).write(to: sidecar)

        let writer = try ChunkedFileWriter(
            destination: target, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        let done = writer.completedChunkIndices()
        writer.close()

        XCTAssertEqual(done, [0])
    }
}

/// Verification is what stops a truncated or corrupted download from being
/// cached permanently and resurfacing as an unreadable-tensor crash.
final class DownloadVerificationTests: XCTestCase {

    func testChecksumMismatchDeletesTheFileSoItIsRefetched() throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("verify-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let url = scratch.appendingPathComponent("weights.safetensors")
        try Data("corrupted".utf8).write(to: url)

        let file = RepoFile(
            path: "weights.safetensors",
            size: 9,
            sha256: String(repeating: "f", count: 64))

        XCTAssertThrowsError(
            try HuggingFaceDownloader.verifyDownloaded(file, at: url, modelId: "org/model")
        ) { error in
            guard case DownloadError.checksumMismatch = error else {
                return XCTFail("expected checksumMismatch, got \(error)")
            }
        }
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: url.path),
            "a bad file left on disk would be accepted by the size check forever after")
    }

    func testShortFileIsRejectedAndRemoved() throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("verify-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let url = scratch.appendingPathComponent("weights.safetensors")
        try Data(count: 10).write(to: url)

        XCTAssertThrowsError(
            try HuggingFaceDownloader.verifyDownloaded(
                RepoFile(path: "weights.safetensors", size: 4096, sha256: nil),
                at: url, modelId: "org/model"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
    }

    func testMatchingChecksumPasses() throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("verify-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let url = scratch.appendingPathComponent("weights.safetensors")
        let payload = Data("known content".utf8)
        try payload.write(to: url)

        let digest = try XCTUnwrap(HuggingFaceDownloader.fileSHA256(at: url))
        XCTAssertNoThrow(
            try HuggingFaceDownloader.verifyDownloaded(
                RepoFile(path: "weights.safetensors", size: Int64(payload.count), sha256: digest),
                at: url, modelId: "org/model"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
    }

    /// Hashing reads in bounded chunks, so a file larger than one read window
    /// must hash the same as the reference digest.
    func testStreamingHashHandlesMultiChunkFiles() throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("verify-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let url = scratch.appendingPathComponent("big.bin")
        let payload = Data(repeating: 0x5A, count: HuggingFaceDownloader.shaChunkBytes + 12345)
        try payload.write(to: url)

        XCTAssertEqual(
            HuggingFaceDownloader.fileSHA256(at: url),
            HuggingFaceDownloader.sha256Hex(of: payload))
    }
}

/// Covers removal of a consolidated weights file left by an earlier over-fetch.
/// The loader merges every `.safetensors` in a directory, so a duplicate costs
/// load time and memory — but several repositories legitimately ship extra
/// weight files beside a sharded model, and removing one of those breaks them.
final class RedundantWeightsCleanupTests: XCTestCase {

    private func makeScratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("redundant-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Minimal valid safetensors: little-endian UInt64 header length, then the
    /// JSON header, then tensor bytes.
    private func writeSafetensors(tensors: [String], to url: URL) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for name in tensors {
            header[name] = ["dtype": "F32", "shape": [1], "data_offsets": [offset, offset + 4]]
            offset += 4
        }
        let json = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var payload = Data(bytes: &length, count: 8)
        payload.append(json)
        payload.append(Data(count: offset))
        try payload.write(to: url)
    }

    private func index(_ weightMap: [String: String]) throws -> Data {
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
    }

    func testRemovesConsolidatedCopyWhenShardsSupplyEveryTensor() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        try writeSafetensors(tensors: ["a.w"], to: dir.appendingPathComponent("model-00001.safetensors"))
        try writeSafetensors(tensors: ["b.w"], to: dir.appendingPathComponent("model-00002.safetensors"))
        try writeSafetensors(
            tensors: ["a.w", "b.w"], to: dir.appendingPathComponent("model.safetensors"))

        HuggingFaceDownloader.removeRedundantConsolidatedWeights(
            in: dir,
            indexData: try index(["a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"]))

        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors").path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model-00001.safetensors").path),
            "the shards are the model and must survive")
    }

    /// Fish Audio ships `codec.safetensors` next to an index that names only the
    /// model shards. Deleting weight files the index omits would break it, so
    /// only the exact consolidated name is ever a candidate.
    func testKeepsUnrelatedWeightFileTheIndexDoesNotName() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        try writeSafetensors(
            tensors: ["a.w"], to: dir.appendingPathComponent("model-00001-of-00002.safetensors"))
        try writeSafetensors(
            tensors: ["b.w"], to: dir.appendingPathComponent("model-00002-of-00002.safetensors"))
        try writeSafetensors(tensors: ["codec.w"], to: dir.appendingPathComponent("codec.safetensors"))

        HuggingFaceDownloader.removeRedundantConsolidatedWeights(
            in: dir,
            indexData: try index([
                "a.w": "model-00001-of-00002.safetensors",
                "b.w": "model-00002-of-00002.safetensors",
            ]))

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("codec.safetensors").path),
            "a separate weight file is not a redundant copy")
    }

    /// If the consolidated file holds tensors the shards don't, it isn't a
    /// duplicate and removing it would lose weights.
    func testKeepsConsolidatedCopyHoldingTensorsTheIndexDoesNotName() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        try writeSafetensors(tensors: ["a.w"], to: dir.appendingPathComponent("model-00001.safetensors"))
        try writeSafetensors(
            tensors: ["a.w", "extra.w"], to: dir.appendingPathComponent("model.safetensors"))

        HuggingFaceDownloader.removeRedundantConsolidatedWeights(
            in: dir, indexData: try index(["a.w": "model-00001.safetensors"]))

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors").path))
    }

    /// With a shard missing, the consolidated file may be the only complete
    /// copy of the weights.
    func testKeepsConsolidatedCopyWhenAShardIsMissing() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        try writeSafetensors(tensors: ["a.w"], to: dir.appendingPathComponent("model-00001.safetensors"))
        try writeSafetensors(
            tensors: ["a.w", "b.w"], to: dir.appendingPathComponent("model.safetensors"))

        HuggingFaceDownloader.removeRedundantConsolidatedWeights(
            in: dir,
            indexData: try index(["a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"]))

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors").path),
            "never remove weights while the indexed set is incomplete")
    }

    /// A single-file bundle whose index names `model.safetensors` itself.
    func testKeepsConsolidatedCopyNamedByTheIndex() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        try writeSafetensors(tensors: ["a.w"], to: dir.appendingPathComponent("model.safetensors"))

        HuggingFaceDownloader.removeRedundantConsolidatedWeights(
            in: dir, indexData: try index(["a.w": "model.safetensors"]))

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: dir.appendingPathComponent("model.safetensors").path))
    }

    func testHeaderReaderRejectsGarbage() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("broken.safetensors")
        try Data("not a safetensors file at all".utf8).write(to: url)

        XCTAssertNil(HuggingFaceDownloader.safetensorsTensorNames(at: url))
    }

    func testHeaderReaderExtractsTensorNamesIgnoringMetadata() throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("m.safetensors")
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
        header["x.weight"] = ["dtype": "F32", "shape": [1], "data_offsets": [0, 4]]
        let json = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var payload = Data(bytes: &length, count: 8)
        payload.append(json)
        payload.append(Data(count: 4))
        try payload.write(to: url)

        XCTAssertEqual(HuggingFaceDownloader.safetensorsTensorNames(at: url), ["x.weight"])
    }
}

/// A network blip must not fail a load that needs no bytes.
///
/// Regression: a transient outage during a full test run failed transcription
/// for a model whose every file was already cached, because resolution always
/// goes to the network. Falling back is only safe when the Hub was unreachable
/// — a 404 is a real answer — and only when the cache is genuinely complete.
///
/// The unreachable Hub is simulated by pointing `HF_ENDPOINT` at a closed port,
/// so these run without network and stay in CI.
final class UnreachableHubFallbackTests: XCTestCase {

    private static let deadEndpoint = "http://127.0.0.1:9"

    private func withDeadHub<T>(_ body: () async throws -> T) async rethrows -> T {
        let previous = ProcessInfo.processInfo.environment["HF_ENDPOINT"]
        setenv("HF_ENDPOINT", Self.deadEndpoint, 1)
        defer {
            if let previous { setenv("HF_ENDPOINT", previous, 1) } else { unsetenv("HF_ENDPOINT") }
        }
        return try await body()
    }

    private func makeScratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("unreachable-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testCompleteCacheLoadsWhileHubIsUnreachable() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(count: 8).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        try Data(count: 4).write(to: dir.appendingPathComponent("vocab.json"))

        var reported = 0.0
        try await withDeadHub {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: "org/model",
                to: dir,
                additionalFiles: ["vocab.json"],
                retryDelaysSeconds: []
            ) { reported = $0 }
        }

        XCTAssertEqual(reported, 1.0, accuracy: 0.001)
    }

    /// The cache must be complete, not merely non-empty. Weights without the
    /// tokenizer the caller asked for would load and then fail somewhere else.
    func testIncompleteCacheStillFailsWhileHubIsUnreachable() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(count: 8).write(to: dir.appendingPathComponent("model.safetensors"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))
        // vocab.json is requested below but absent.

        do {
            try await withDeadHub {
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: "org/model",
                    to: dir,
                    additionalFiles: ["vocab.json"],
                    retryDelaysSeconds: [])
            }
            XCTFail("an incomplete cache must not be accepted")
        } catch let error as DownloadError {
            guard case .networkUnavailable = error else {
                return XCTFail("unexpected: \(error)")
            }
        }
    }

    func testEmptyCacheStillFailsWhileHubIsUnreachable() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            try await withDeadHub {
                try await HuggingFaceDownloader.downloadWeights(
                    modelId: "org/model", to: dir, retryDelaysSeconds: [])
            }
            XCTFail("an empty cache has nothing to fall back to")
        } catch {
            // expected
        }
    }

    /// Same rule for the explicit-list paths, including bundle globs.
    func testDownloadFilesFallsBackWhenEveryPatternIsSatisfied() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        let bundle = dir.appendingPathComponent("Mask.mlmodelc", isDirectory: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data(count: 4).write(to: bundle.appendingPathComponent("coremldata.bin"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))

        try await withDeadHub {
            try await HuggingFaceDownloader.downloadFiles(
                modelId: "org/model",
                to: dir,
                files: ["config.json", "Mask.mlmodelc/**"],
                retryDelaysSeconds: [])
        }
    }

    func testDownloadFilesStillFailsWhenAPatternIsUnsatisfied() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data("{}".utf8).write(to: dir.appendingPathComponent("config.json"))

        do {
            try await withDeadHub {
                try await HuggingFaceDownloader.downloadFiles(
                    modelId: "org/model",
                    to: dir,
                    files: ["config.json", "Missing.mlmodelc/**"],
                    retryDelaysSeconds: [])
            }
            XCTFail("a missing bundle must not be papered over")
        } catch {
            // expected
        }
    }

    func testByteWeightedFallsBackWhenEveryFileIsCached() async throws {
        let dir = try makeScratch()
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(count: 4).write(to: dir.appendingPathComponent("a.json"))
        try Data(count: 4).write(to: dir.appendingPathComponent("b.json"))

        try await withDeadHub {
            try await HuggingFaceDownloader.downloadFilesByteWeighted(
                modelId: "org/model",
                to: dir,
                files: ["a.json", "b.json"],
                retryDelaysSeconds: [])
        }
    }

    /// An HTTP answer is a real answer and must never be softened, even with a
    /// complete-looking cache. Classification is checked directly so the test
    /// needs no network.
    func testHTTPErrorsAreNotTreatedAsNetworkFailures() {
        XCTAssertFalse(
            HuggingFaceDownloader.isLikelyNetworkFailure(
                DownloadError.failedToDownload("org/model: file listing HTTP 404")))
        XCTAssertFalse(
            HuggingFaceDownloader.isLikelyNetworkFailure(
                DownloadError.checksumMismatch(file: "w", expected: "a", actual: "b")))
        XCTAssertTrue(
            HuggingFaceDownloader.isLikelyNetworkFailure(
                DownloadError.stalled(modelId: "org/model", seconds: 90)))
        XCTAssertTrue(
            HuggingFaceDownloader.isLikelyNetworkFailure(
                URLError(.notConnectedToInternet)))
        XCTAssertTrue(
            HuggingFaceDownloader.isLikelyNetworkFailure(URLError(.networkConnectionLost)))
        XCTAssertFalse(HuggingFaceDownloader.isLikelyNetworkFailure(URLError(.badURL)))
    }
}
