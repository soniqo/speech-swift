import XCTest
@testable import AudioCommon

/// Regression tests for two large-bundle download failure modes observed
/// with VoxCPM2 bf16 (4.3 GB shard):
/// 1. `weightsExist` accepted a partial multi-shard bundle (stall left
///    shard 1 missing, shard 2 complete) → model loaded half-initialized
///    and synthesized near-silence.
/// 2. The stall guard only ticked on `hub.snapshot` progress callbacks,
///    which fire at file completion — any shard needing longer than the
///    stall window produced zero ticks and a healthy transfer was killed
///    on every retry.
final class DownloaderRobustnessTests: XCTestCase {

    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("dl-robust-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private func write(_ name: String, _ contents: String, in dir: URL) throws {
        try contents.data(using: .utf8)!.write(to: dir.appendingPathComponent(name))
    }

    func testWeightsExistRejectsMissingShard() throws {
        let dir = try makeTempDir()
        try write("model.safetensors.index.json",
                  #"{"weight_map": {"a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"}}"#,
                  in: dir)
        try write("model-00002.safetensors", "shard2", in: dir)
        XCTAssertFalse(HuggingFaceDownloader.weightsExist(in: dir),
                       "partial multi-shard bundle must not count as cached")
    }

    func testWeightsExistAcceptsCompleteShardedBundle() throws {
        let dir = try makeTempDir()
        try write("model.safetensors.index.json",
                  #"{"weight_map": {"a.w": "model-00001.safetensors", "b.w": "model-00002.safetensors"}}"#,
                  in: dir)
        try write("model-00001.safetensors", "shard1", in: dir)
        try write("model-00002.safetensors", "shard2", in: dir)
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: dir))
    }

    func testWeightsExistAcceptsSingleFileWithoutIndex() throws {
        let dir = try makeTempDir()
        try write("model.safetensors", "weights", in: dir)
        XCTAssertTrue(HuggingFaceDownloader.weightsExist(in: dir))
    }

    /// A download that reports no fractional progress but keeps writing
    /// bytes to the destination must survive the stall guard.
    func testStallGuardTicksOnDiskGrowth() async throws {
        let dir = try makeTempDir()
        let file = dir.appendingPathComponent("model.safetensors.incomplete")
        FileManager.default.createFile(atPath: file.path, contents: Data())

        try await HuggingFaceDownloader.withDownloadStallGuard(
            modelId: "test/growing", stallTimeoutSeconds: 2, watchDirectory: dir
        ) { _ in
            // 5s of transfer with zero progress callbacks, but the file grows.
            for _ in 0..<10 {
                try await Task.sleep(for: .milliseconds(500))
                let handle = try FileHandle(forWritingTo: file)
                try handle.seekToEnd()
                try handle.write(contentsOf: Data(repeating: 7, count: 64))
                try handle.close()
            }
        }
    }

    /// No progress callbacks and no disk growth is a genuine stall.
    func testStallGuardStillFiresWithoutGrowth() async throws {
        let dir = try makeTempDir()
        do {
            try await HuggingFaceDownloader.withDownloadStallGuard(
                modelId: "test/stalled", stallTimeoutSeconds: 1, watchDirectory: dir
            ) { _ in
                try await Task.sleep(for: .seconds(10))
            }
            XCTFail("expected stall")
        } catch let error as DownloadError {
            guard case .stalled = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }
}
