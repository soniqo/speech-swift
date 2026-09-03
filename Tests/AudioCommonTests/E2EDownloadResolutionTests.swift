import XCTest
@testable import AudioCommon

/// Live-network checks for the download path.
///
/// These hit the Hub but stay cheap: resolution is metadata-only, and the one
/// real transfer forces the chunked path onto a small file rather than pulling
/// a multi-gigabyte shard to prove the same code works.
final class E2EDownloadResolutionTests: XCTestCase {

    private static let coreMLRepo = "aufklarer/Whisper-Large-v3-Turbo-CoreML"
    private static let shardedRepo = "aufklarer/VoxCPM2-MLX-bf16"

    private func scratchDirectory(_ label: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(label)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Resolution

    /// CoreML repositories ship bundles as directories of files. Resolving them
    /// requires a recursive listing and nested-path support — the combination
    /// that previously confined these repos to the slower downloader.
    func testCoreMLBundleResolvesNestedMembers() async throws {
        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.coreMLRepo)

        let bundle = manifest.matching(globs: ["AudioEncoder.mlmodelc/**"])
        XCTAssertFalse(bundle.isEmpty, "glob must reach inside the bundle directory")
        XCTAssertTrue(
            bundle.contains { $0.path.contains("/") },
            "nested members must survive resolution")
        XCTAssertTrue(
            bundle.allSatisfy { $0.size > 0 },
            "every resolved file must carry a size")
        XCTAssertTrue(
            manifest.files.allSatisfy { !$0.path.hasSuffix("/") },
            "directory entries must be filtered out — they are not downloadable")
    }

    /// Regression guard for the over-fetch. This repo publishes the two shards
    /// its index names *and* a consolidated copy of the same tensors, so a bare
    /// `*.safetensors` glob doubles the download.
    func testShardedRepoSelectsIndexShardsNotTheDuplicate() async throws {
        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.shardedRepo)
        let globbed = manifest.matching(globs: [
            "config.json", "*.safetensors", "model.safetensors.index.json",
        ])

        let fetchedIndex = await HuggingFaceDownloader.fetchShardIndex(
            modelId: Self.shardedRepo, manifest: manifest)
        let indexData = try XCTUnwrap(fetchedIndex, "this repo ships an index")
        let narrowed = HuggingFaceDownloader.applyingShardIndex(
            to: globbed, indexData: indexData)

        let globbedBytes = globbed.reduce(Int64(0)) { $0 + $1.size }
        let narrowedBytes = narrowed.reduce(Int64(0)) { $0 + $1.size }

        XCTAssertFalse(
            narrowed.contains { $0.path == "model.safetensors" },
            "the consolidated duplicate must be dropped")
        XCTAssertTrue(
            narrowed.contains { $0.path == "model-00001.safetensors" },
            "shards named by the index must be kept")
        XCTAssertLessThan(
            narrowedBytes, globbedBytes,
            "narrowing must actually reduce the transfer")
        // Guards the specific failure: ~9.9 GB globbed vs ~4.96 GB needed.
        XCTAssertLessThan(Double(narrowedBytes), Double(globbedBytes) * 0.6)
    }

    /// Weight files are LFS-backed and must carry a content digest, otherwise
    /// integrity checking silently does nothing on exactly the files that matter.
    func testLargeWeightsCarryContentDigests() async throws {
        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.shardedRepo)
        let shards = manifest.files.filter { $0.path.hasSuffix(".safetensors") }

        XCTAssertFalse(shards.isEmpty)
        for shard in shards {
            XCTAssertNotNil(shard.sha256, "\(shard.path) must publish a sha256")
            XCTAssertEqual(shard.sha256?.count, 64)
            XCTAssertGreaterThan(shard.size, 100_000_000)
        }
    }

    // MARK: - Transfer

    /// Drives the chunked path end to end against the real CDN — concurrent
    /// ranges, positional writes, reassembly and digest verification — using a
    /// few-megabyte file with the threshold lowered, so the test is fast but
    /// the code under test is the same code that moves multi-GB shards.
    func testRangedTransferProducesAByteCorrectFile() async throws {
        setenv("HF_DOWNLOAD_RANGE_THRESHOLD", "1", 1)
        setenv("HF_DOWNLOAD_RANGE_CHUNK", "262144", 1)
        defer {
            unsetenv("HF_DOWNLOAD_RANGE_THRESHOLD")
            unsetenv("HF_DOWNLOAD_RANGE_CHUNK")
        }

        let directory = try scratchDirectory("ranged")
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.shardedRepo)
        let target = try XCTUnwrap(manifest.file(at: "tokenizer.json"))

        try await HuggingFaceDownloader.downloadManifestFiles(
            [target], modelId: Self.shardedRepo, to: directory, progressHandler: nil)

        let written = directory.appendingPathComponent("tokenizer.json")
        XCTAssertEqual(HuggingFaceDownloader.localFileSize(written), target.size)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(".incomplete").path),
            "staging directory must be cleaned up on success")

        // The file must be intact, not merely the right length.
        let payload = try Data(contentsOf: written)
        XCTAssertNotNil(
            try? JSONSerialization.jsonObject(with: payload),
            "a mis-ordered chunk write would produce the right size and invalid content")
    }

    /// A transfer interrupted partway must resume from what already landed
    /// rather than starting over — the behaviour thirty-odd modules did not
    /// previously have.
    func testInterruptedRangedTransferResumes() async throws {
        setenv("HF_DOWNLOAD_RANGE_THRESHOLD", "1", 1)
        setenv("HF_DOWNLOAD_RANGE_CHUNK", "262144", 1)
        defer {
            unsetenv("HF_DOWNLOAD_RANGE_THRESHOLD")
            unsetenv("HF_DOWNLOAD_RANGE_CHUNK")
        }

        let directory = try scratchDirectory("resume")
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.shardedRepo)
        let target = try XCTUnwrap(manifest.file(at: "tokenizer.json"))

        // Simulate an attempt that got partway: stage the file and record the
        // first chunk as already written, using real bytes for that range.
        let staging = try HuggingFaceDownloader.stagingURL(
            for: target.path, in: directory)
        let sidecar = staging.appendingPathExtension("chunks")
        let writer = try ChunkedFileWriter(
            destination: staging,
            sidecar: sidecar,
            totalSize: target.size,
            chunkCount: 999)
        let url = try HuggingFaceDownloader.resolveURL(
            modelId: Self.shardedRepo, file: target.path)
        var head = HuggingFaceDownloader.makeHubRequest(url: url, timeout: 60)
        head.setValue("bytes=0-262143", forHTTPHeaderField: "Range")
        let (firstChunk, _) = try await URLSession.shared.data(for: head)
        try writer.write(firstChunk, at: 0, chunkIndex: 0)
        writer.close()

        XCTAssertTrue(FileManager.default.fileExists(atPath: sidecar.path))

        try await HuggingFaceDownloader.downloadManifestFiles(
            [target], modelId: Self.shardedRepo, to: directory, progressHandler: nil)

        let written = directory.appendingPathComponent("tokenizer.json")
        XCTAssertEqual(HuggingFaceDownloader.localFileSize(written), target.size)
        XCTAssertNotNil(
            try? JSONSerialization.jsonObject(with: Data(contentsOf: written)),
            "resumed content must be valid, not a mix of two attempts")
    }

    /// A file already on disk at the manifest's size must not be re-fetched.
    func testCachedFileIsSkipped() async throws {
        let directory = try scratchDirectory("cached")
        defer { try? FileManager.default.removeItem(at: directory) }

        let manifest = try await HuggingFaceDownloader.fetchManifest(modelId: Self.shardedRepo)
        let target = try XCTUnwrap(manifest.file(at: "config.json"))

        try await HuggingFaceDownloader.downloadManifestFiles(
            [target], modelId: Self.shardedRepo, to: directory, progressHandler: nil)

        let written = directory.appendingPathComponent("config.json")
        let firstModified = try FileManager.default
            .attributesOfItem(atPath: written.path)[.modificationDate] as? Date

        try await HuggingFaceDownloader.downloadManifestFiles(
            [target], modelId: Self.shardedRepo, to: directory, progressHandler: nil)

        let secondModified = try FileManager.default
            .attributesOfItem(atPath: written.path)[.modificationDate] as? Date
        XCTAssertEqual(firstModified, secondModified, "a complete file must not be rewritten")
    }

    /// A full fresh-cache download of a CoreML bundle.
    ///
    /// This is the shape the ranged downloader previously could not express at
    /// all: its path validation accepted only single file names, so every
    /// CoreML repository was confined to the slower path. Small model (~2.5 MB)
    /// chosen so the check stays cheap while covering the real flow.
    func testFreshCoreMLBundleDownloadsIntact() async throws {
        let directory = try scratchDirectory("coreml-fresh")
        defer { try? FileManager.default.removeItem(at: directory) }

        var lastFraction = 0.0
        try await HuggingFaceDownloader.downloadWeights(
            modelId: "aufklarer/DeepFilterNet3-CoreML",
            to: directory,
            additionalFiles: ["*.mlmodelc/**", "config.json"],
            retryDelaysSeconds: []
        ) { lastFraction = $0 }

        let entries = FileManager.default.enumerator(atPath: directory.path)?
            .compactMap { $0 as? String } ?? []

        XCTAssertTrue(
            entries.contains("DeepFilterNet3.mlmodelc/weights/weight.bin"),
            "bundle members must land inside their directories")
        XCTAssertTrue(entries.contains("config.json"))
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: directory))
        XCTAssertFalse(
            entries.contains { $0.hasPrefix(".incomplete") },
            "staging must not be left behind in a bundle CoreML will load")
        XCTAssertEqual(lastFraction, 1.0, accuracy: 0.001)
    }

    /// Regression guard: `downloadFiles` entries are glob patterns, not literal
    /// names. `LocalVQEEchoCanceller` asks for
    /// `LocalVQEAECResidualMask.mlmodelc/**`, and treating that as a filename
    /// makes the request resolve to nothing and fail.
    func testDownloadFilesAcceptsBundleGlobs() async throws {
        let directory = try scratchDirectory("glob-files")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await HuggingFaceDownloader.downloadFiles(
            modelId: "aufklarer/DeepFilterNet3-CoreML",
            to: directory,
            files: ["config.json", "DeepFilterNet3.mlmodelc/**"],
            retryDelaysSeconds: [])

        let entries = FileManager.default.enumerator(atPath: directory.path)?
            .compactMap { $0 as? String } ?? []
        XCTAssertTrue(entries.contains("DeepFilterNet3.mlmodelc/weights/weight.bin"))
        XCTAssertTrue(entries.contains("config.json"))
    }

    /// A pattern matching nothing contributes nothing and must not fail the
    /// call — several callers pass optional assets this way.
    func testDownloadFilesToleratesPatternsThatMatchNothing() async throws {
        let directory = try scratchDirectory("glob-empty")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await HuggingFaceDownloader.downloadFiles(
            modelId: "aufklarer/DeepFilterNet3-CoreML",
            to: directory,
            files: ["config.json", "definitely-not-here-*.bin"],
            retryDelaysSeconds: [])

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("config.json").path))
    }
}
