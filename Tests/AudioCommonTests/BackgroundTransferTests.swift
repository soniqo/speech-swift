import XCTest
@testable import AudioCommon

/// A background transfer is delivered to whichever process the system chooses,
/// which may be a fresh launch with no memory of starting it. Everything needed
/// to file the bytes therefore has to survive in the task's own description and
/// on disk — that is what these cover.
final class BackgroundTransferItemTests: XCTestCase {

    private func chunkItem(index: Int = 3) -> BackgroundTransferItem {
        BackgroundTransferItem(
            label: "org/model/weights.safetensors",
            destination: "/tmp/cache/.incomplete/weights-abc",
            sidecar: "/tmp/cache/.incomplete/weights-abc.chunks",
            expectedSize: 33_554_432,
            chunkCount: 4,
            chunkIndex: index,
            rangeStart: Int64(index) * 8_388_608,
            rangeEnd: Int64(index + 1) * 8_388_608 - 1)
    }

    private var wholeItem: BackgroundTransferItem {
        BackgroundTransferItem(
            label: "org/model/config.json",
            destination: "/tmp/cache/config.json",
            sidecar: nil,
            expectedSize: 1_024,
            chunkCount: 1,
            chunkIndex: nil,
            rangeStart: nil,
            rangeEnd: nil)
    }

    func testAChunkSurvivesTheRoundTripThroughATaskDescription() throws {
        let item = chunkItem()
        let description = try XCTUnwrap(item.encoded())

        XCTAssertEqual(BackgroundTransferItem.decoded(from: description), item)
    }

    func testAWholeFileSurvivesTheRoundTripThroughATaskDescription() throws {
        let item = wholeItem
        let description = try XCTUnwrap(item.encoded())

        XCTAssertEqual(BackgroundTransferItem.decoded(from: description), item)
    }

    /// Tasks started by something other than this downloader carry descriptions
    /// that mean nothing here, and must not be mistaken for ours.
    func testAnUnreadableDescriptionDecodesToNothing() {
        XCTAssertNil(BackgroundTransferItem.decoded(from: nil))
        XCTAssertNil(BackgroundTransferItem.decoded(from: ""))
        XCTAssertNil(BackgroundTransferItem.decoded(from: "some other task"))
    }

    func testAChunkReportsTheLengthItsRangeAsksFor() {
        XCTAssertEqual(chunkItem().length, 8_388_608)
    }

    /// A whole file has no range, so the file's own size is what is owed.
    func testAWholeFileReportsTheFileSize() {
        XCTAssertEqual(wholeItem.length, 1_024)
    }

    /// Completions are matched to the caller by what they are assembling. A
    /// task identifier cannot do it: those are not stable across launches.
    func testChunksOfOneFileShareAGroupAndKeepDistinctSlots() {
        XCTAssertEqual(chunkItem(index: 0).groupKey, chunkItem(index: 1).groupKey)
        XCTAssertEqual(chunkItem(index: 0).slot, 0)
        XCTAssertEqual(chunkItem(index: 1).slot, 1)
        XCTAssertEqual(wholeItem.slot, 0, "a whole file is a group of one")
    }
}

/// A range can land while no process is assembling its file — the system
/// relaunches the app to hand it over, and the caller that wanted it is long
/// gone. Those bytes wait on disk instead of being thrown away.
final class HeldChunkSpliceTests: XCTestCase {

    private func makeScratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("held-chunk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    private func chunks() -> [DownloadChunk] {
        [
            DownloadChunk(index: 0, start: 0, end: 7),
            DownloadChunk(index: 1, start: 8, end: 15),
        ]
    }

    func testAHeldRangeIsSplicedAndThenRemoved() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        let sidecar = staging.appendingPathExtension("chunks")
        let held = BackgroundTransferCoordinator.heldChunkURL(staging: staging, chunkIndex: 1)
        let payload = Data(repeating: 0xBB, count: 8)
        try payload.write(to: held)

        let writer = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        BackgroundTransferCoordinator.spliceHeldChunks(
            staging: staging, chunks: chunks(), writer: writer)
        let done = writer.completedChunkIndices()
        writer.close()

        XCTAssertEqual(done, [1], "the held range counts as written")
        XCTAssertFalse(FileManager.default.fileExists(atPath: held.path))
        let assembled = try Data(contentsOf: staging)
        XCTAssertEqual(assembled.subdata(in: 8..<16), payload)
    }

    /// A held file of the wrong length describes a different export, and
    /// splicing it would put the wrong bytes at a valid offset.
    func testAHeldRangeOfTheWrongLengthIsIgnored() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        let sidecar = staging.appendingPathExtension("chunks")
        let held = BackgroundTransferCoordinator.heldChunkURL(staging: staging, chunkIndex: 0)
        try Data(repeating: 0xCC, count: 5).write(to: held)

        let writer = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        BackgroundTransferCoordinator.spliceHeldChunks(
            staging: staging, chunks: chunks(), writer: writer)
        let done = writer.completedChunkIndices()
        writer.close()

        XCTAssertTrue(done.isEmpty)
    }

    func testSplicingWithNothingHeldChangesNothing() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        let sidecar = staging.appendingPathExtension("chunks")

        let writer = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        BackgroundTransferCoordinator.spliceHeldChunks(
            staging: staging, chunks: chunks(), writer: writer)
        let done = writer.completedChunkIndices()
        writer.close()

        XCTAssertTrue(done.isEmpty)
    }

    /// The dangerous case. A re-exported file changes size, the staging file
    /// resets, and its sidecar is cleared — but a held range from the previous
    /// export is still on disk under the same name, and an interior range of a
    /// fixed-size chunking has exactly the right length to splice cleanly. It
    /// would be recorded as complete and corrupt the result.
    func testHeldRangesAreDiscardedWhenTheStagingFileResets() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        let sidecar = staging.appendingPathExtension("chunks")

        let previousExport = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        previousExport.close()
        XCTAssertFalse(previousExport.resumedExistingFile)

        let held = BackgroundTransferCoordinator.heldChunkURL(staging: staging, chunkIndex: 1)
        try Data(repeating: 0xBB, count: 8).write(to: held)

        // Same path, different size: a re-export.
        let reExport = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 24, chunkCount: 3)
        XCTAssertFalse(
            reExport.resumedExistingFile,
            "a size change is a different file, not a resume")

        BackgroundTransferCoordinator.discardHeldChunks(staging: staging, chunks: chunks())
        BackgroundTransferCoordinator.spliceHeldChunks(
            staging: staging, chunks: chunks(), writer: reExport)
        let done = reExport.completedChunkIndices()
        reExport.close()

        XCTAssertTrue(done.isEmpty, "a range from the previous export must not be spliced")
        XCTAssertFalse(FileManager.default.fileExists(atPath: held.path))
    }

    /// A staging file that survived at its manifest size is a resume, and its
    /// held ranges are still describing it.
    func testAStagingFileAtItsManifestSizeCountsAsAResume() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        let sidecar = staging.appendingPathExtension("chunks")

        let first = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        try first.write(Data(repeating: 0xAA, count: 8), at: 0, chunkIndex: 0)
        first.close()

        let resumed = try ChunkedFileWriter(
            destination: staging, sidecar: sidecar, totalSize: 16, chunkCount: 2)
        defer { resumed.close() }

        XCTAssertTrue(resumed.resumedExistingFile)
        XCTAssertEqual(resumed.completedChunkIndices(), [0])
    }

    /// A range delivered after the file was assembled has nothing to splice
    /// into, and would otherwise keep the staging directory alive forever.
    func testDiscardingRemovesEveryHeldRangeForAFile() throws {
        let scratch = try makeScratch()
        let staging = scratch.appendingPathComponent("staged.bin")
        for index in 0...1 {
            try Data(repeating: 0xEE, count: 8).write(
                to: BackgroundTransferCoordinator.heldChunkURL(
                    staging: staging, chunkIndex: index))
        }

        BackgroundTransferCoordinator.discardHeldChunks(staging: staging, chunks: chunks())

        for index in 0...1 {
            let held = BackgroundTransferCoordinator.heldChunkURL(
                staging: staging, chunkIndex: index)
            XCTAssertFalse(FileManager.default.fileExists(atPath: held.path))
        }
    }

    /// Held ranges sit beside the staging file rather than inside the cache
    /// tree, where a partial transfer could be mistaken for repo content.
    func testHeldRangesAreNamedBesideTheStagingFile() {
        let staging = URL(fileURLWithPath: "/tmp/cache/.incomplete/weights-abc")
        let held = BackgroundTransferCoordinator.heldChunkURL(staging: staging, chunkIndex: 7)

        XCTAssertEqual(held.deletingLastPathComponent(), staging.deletingLastPathComponent())
        XCTAssertEqual(held.lastPathComponent, "weights-abc.held-7")
    }
}

/// Out-of-process transfer is opt-in, so every caller that is never suspended
/// keeps the in-process session it has today.
final class BackgroundTransferOptInTests: XCTestCase {

    func testTheInProcessSessionIsTheDefault() {
        XCTAssertNil(HuggingFaceDownloader.backgroundTransfer)
    }

    func testTheConfigurationRoundTripsAndCanBeCleared() {
        let previous = HuggingFaceDownloader.backgroundTransfer
        defer { HuggingFaceDownloader.backgroundTransfer = previous }

        HuggingFaceDownloader.backgroundTransfer = BackgroundTransferConfiguration(
            sessionIdentifier: "test.session.\(UUID().uuidString)")
        XCTAssertNotNil(HuggingFaceDownloader.backgroundTransfer)

        HuggingFaceDownloader.backgroundTransfer = nil
        XCTAssertNil(HuggingFaceDownloader.backgroundTransfer)
    }

    /// A transfer that a slow link genuinely needs hours for must not be
    /// cancelled by a deadline meant for a stalled one.
    func testTheDefaultResourceDeadlineIsGenerous() {
        let configuration = BackgroundTransferConfiguration(sessionIdentifier: "test.session")

        XCTAssertGreaterThanOrEqual(configuration.resourceTimeout, 6 * 60 * 60)
        XCTAssertFalse(
            configuration.isDiscretionary,
            "a model download blocks the feature that needs it")
    }

    /// The system requires its completion handler to be called. An identifier
    /// this process knows nothing about is answered rather than dropped.
    func testAnUnknownSessionIdentifierStillAnswersTheSystem() {
        let previous = HuggingFaceDownloader.backgroundTransfer
        defer { HuggingFaceDownloader.backgroundTransfer = previous }
        HuggingFaceDownloader.backgroundTransfer = nil

        let answered = expectation(description: "system completion handler called")
        HuggingFaceDownloader.handleBackgroundSessionEvents(
            identifier: "not.a.session.this.process.owns"
        ) {
            answered.fulfill()
        }

        wait(for: [answered], timeout: 1)
    }
}
