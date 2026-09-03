import Foundation
import os

// MARK: - Transfer engine
//
// One downloader for every caller. Resolution happens against a repo manifest
// (see `HuggingFaceRepoManifest.swift`), so this layer only ever deals in
// files whose size is already known, which is what makes ranged transfer,
// resume and byte-accurate progress possible for all of them rather than the
// handful that previously passed an explicit list.

extension HuggingFaceDownloader {

    /// Download `files` into `directory`, skipping anything already complete.
    ///
    /// Progress is byte-weighted across the whole set. Large files are fetched
    /// as concurrent ranges and can resume mid-file; small ones are fetched in
    /// one request.
    static func downloadManifestFiles(
        _ files: [RepoFile],
        modelId: String,
        to directory: URL,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) async throws {
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        var completedBytes: Int64 = 0

        for file in files {
            let localURL = try validatedLocalPath(directory: directory, relativePath: file.path)
            try FileManager.default.createDirectory(
                at: localURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)

            // A file already at the manifest's size is complete. Re-hashing
            // cached weights on every launch would cost seconds per gigabyte
            // for no benefit — verification happens when bytes are written,
            // below, not when they are found.
            if localFileSize(localURL) == file.size {
                completedBytes += file.size
                reportByteProgress(
                    completedBytes: completedBytes,
                    totalBytes: totalBytes,
                    fileName: file.path,
                    progressHandler: progressHandler)
                continue
            }

            reportByteProgress(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                fileName: file.path,
                progressHandler: progressHandler)

            if file.size >= rangedDownloadThresholdBytes {
                try await downloadRanged(
                    file,
                    modelId: modelId,
                    to: localURL,
                    stagingRoot: directory,
                    completedBeforeFile: completedBytes,
                    totalBytes: totalBytes,
                    progressHandler: progressHandler)
            } else {
                try await downloadWhole(file, modelId: modelId, to: localURL)
            }

            try verifyDownloaded(file, at: localURL, modelId: modelId)

            completedBytes += file.size
            reportByteProgress(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                fileName: file.path,
                progressHandler: progressHandler)
        }

        removeStagingDirectoryIfEmpty(in: directory)

        if totalBytes == 0 {
            progressHandler?(1.0, 0, 0, "")
        }
    }

    /// Drop the staging directory once nothing is left in it.
    ///
    /// Only when empty: a failed transfer leaves its staging file behind on
    /// purpose, and that file is the resume point for the next attempt.
    static func removeStagingDirectoryIfEmpty(in directory: URL) {
        let staging = directory.appendingPathComponent(stagingDirectoryName, isDirectory: true)
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(atPath: staging.path),
              contents.isEmpty
        else { return }
        try? fm.removeItem(at: staging)
    }

    /// Check what was written, and delete it if wrong.
    ///
    /// Leaving a bad file on disk is what makes a corrupted download permanent:
    /// the size check at the top of the loop accepts it forever after, and the
    /// failure resurfaces as an unreadable-tensor error deep inside weight
    /// loading. Removing it means the retry ladder — or the next launch —
    /// simply fetches it again.
    static func verifyDownloaded(_ file: RepoFile, at url: URL, modelId: String) throws {
        let written = localFileSize(url)
        guard written == file.size else {
            try? FileManager.default.removeItem(at: url)
            throw DownloadError.failedToDownload(
                "\(modelId)/\(file.path): wrote \(written) bytes, expected \(file.size)")
        }

        guard let expected = file.sha256 else { return }
        guard let actual = fileSHA256(at: url) else { return }
        guard actual == expected else {
            try? FileManager.default.removeItem(at: url)
            throw DownloadError.checksumMismatch(
                file: "\(modelId)/\(file.path)", expected: expected, actual: actual)
        }
    }

    // MARK: - Whole-file transfer

    /// Fetch a file in a single request, staged through a temp file so an
    /// interrupted transfer never leaves a short file at the real path.
    static func downloadWhole(_ file: RepoFile, modelId: String, to destination: URL) async throws {
        let url = try resolveURL(modelId: modelId, file: file.path)
        if let background = HuggingFaceDownloader.backgroundTransfer {
            try await BackgroundTransferCoordinator.coordinator(for: background).download(
                wholeFile: file,
                from: url,
                label: "\(modelId)/\(file.path)",
                to: destination)
            return
        }
        let request = makeHubRequest(url: url, timeout: 120)
        let (tempURL, response) = try await URLSession.shared.download(for: request)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("\(modelId)/\(file.path): missing HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw DownloadError.failedToDownload(
                "\(modelId)/\(file.path): HTTP \(http.statusCode)")
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    // MARK: - Ranged transfer

    /// Fetch a large file as concurrent byte ranges, resuming whatever a
    /// previous attempt already wrote.
    ///
    /// Chunks are written straight into the staging file at their own offset
    /// and recorded in a sidecar. The previous design wrote each chunk to its
    /// own `.part` file and then concatenated them, which needed twice the
    /// file's size in free space (20 GB transient for a 10 GB bundle) and read
    /// and rewrote every byte a second time.
    static func downloadRanged(
        _ file: RepoFile,
        modelId: String,
        to destination: URL,
        stagingRoot: URL,
        completedBeforeFile: Int64,
        totalBytes: Int64,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) async throws {
        let staging = try stagingURL(for: file.path, in: stagingRoot)
        let sidecar = staging.appendingPathExtension("chunks")
        let chunks = makeDownloadChunks(fileSize: file.size, chunkBytes: rangedDownloadChunkBytes)

        let writer = try ChunkedFileWriter(
            destination: staging,
            sidecar: sidecar,
            totalSize: file.size,
            chunkCount: chunks.count)
        if HuggingFaceDownloader.backgroundTransfer != nil {
            if writer.resumedExistingFile {
                // A background transfer can land while no process is assembling
                // this file. Those ranges wait beside the staging file, and are
                // spliced here before anything decides what is still owed.
                BackgroundTransferCoordinator.spliceHeldChunks(
                    staging: staging, chunks: chunks, writer: writer)
            } else {
                // The staging file was reset, so it describes a different
                // export than whatever is held. Held ranges of a matching
                // length would splice cleanly and corrupt the result.
                BackgroundTransferCoordinator.discardHeldChunks(
                    staging: staging, chunks: chunks)
            }
        }
        let alreadyDone = writer.completedChunkIndices()

        let state = RangedDownloadProgress(
            completedBeforeFile: completedBeforeFile,
            totalBytes: totalBytes,
            fileName: file.path,
            progressHandler: progressHandler)
        for chunk in chunks where alreadyDone.contains(chunk.index) {
            await state.addCompletedBytes(chunk.length)
        }

        let pending = chunks.filter { !alreadyDone.contains($0.index) }
        if !pending.isEmpty, let background = HuggingFaceDownloader.backgroundTransfer {
            let url = try resolveURL(modelId: modelId, file: file.path)
            do {
                try await BackgroundTransferCoordinator.coordinator(for: background).download(
                    chunks: pending,
                    from: url,
                    label: "\(modelId)/\(file.path)",
                    writer: writer,
                    staging: staging,
                    sidecar: sidecar,
                    expectedSize: file.size,
                    chunkCount: chunks.count,
                    progress: state)
            } catch {
                // Keep the staging file and sidecar: they are the resume point.
                writer.close()
                throw error
            }
        } else if !pending.isEmpty {
            let url = try resolveURL(modelId: modelId, file: file.path)
            let concurrency = downloadRangeConcurrency
            let configuration = URLSessionConfiguration.default
            configuration.httpMaximumConnectionsPerHost = max(concurrency, 1)
            let session = URLSession(configuration: configuration)
            defer { session.invalidateAndCancel() }

            do {
                try await withThrowingTaskGroup(of: Void.self) { group in
                    var nextIndex = 0
                    for _ in 0..<min(concurrency, pending.count) {
                        let chunk = pending[nextIndex]
                        nextIndex += 1
                        group.addTask {
                            try await downloadRangeChunk(
                                url: url,
                                label: "\(modelId)/\(file.path)",
                                chunk: chunk,
                                writer: writer,
                                state: state,
                                session: session)
                        }
                    }
                    while try await group.next() != nil {
                        guard nextIndex < pending.count else { continue }
                        let chunk = pending[nextIndex]
                        nextIndex += 1
                        group.addTask {
                            try await downloadRangeChunk(
                                url: url,
                                label: "\(modelId)/\(file.path)",
                                chunk: chunk,
                                writer: writer,
                                state: state,
                                session: session)
                        }
                    }
                }
            } catch {
                // Keep the staging file and sidecar: they are the resume point.
                writer.close()
                throw error
            }
        }

        writer.close()

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: staging, to: destination)
        try? FileManager.default.removeItem(at: sidecar)
        if HuggingFaceDownloader.backgroundTransfer != nil {
            // A range delivered after the file was already assembled has
            // nothing left to splice into, and would otherwise keep the
            // staging directory alive forever.
            BackgroundTransferCoordinator.discardHeldChunks(
                staging: staging, chunks: chunks)
        }
    }

    static func downloadRangeChunk(
        url: URL,
        label: String,
        chunk: DownloadChunk,
        writer: ChunkedFileWriter,
        state: RangedDownloadProgress,
        session: URLSession
    ) async throws {
        let request = makeHubRequest(
            url: url,
            range: "bytes=\(chunk.start)-\(chunk.end)",
            timeout: 120)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("\(label) part \(chunk.index): missing HTTP response")
        }
        guard http.statusCode == 206 else {
            throw DownloadError.failedToDownload(
                "\(label) part \(chunk.index): expected HTTP 206, got \(http.statusCode)")
        }
        guard Int64(data.count) == chunk.length else {
            throw DownloadError.failedToDownload(
                "\(label) part \(chunk.index): got \(data.count) bytes, expected \(chunk.length)")
        }
        try writer.write(data, at: chunk.start, chunkIndex: chunk.index)
        await state.addCompletedBytes(chunk.length)
    }

    static func makeDownloadChunks(fileSize: Int64, chunkBytes: Int64) -> [DownloadChunk] {
        guard fileSize > 0, chunkBytes > 0 else { return [] }
        var chunks: [DownloadChunk] = []
        var start: Int64 = 0
        var index = 0
        while start < fileSize {
            let end = min(fileSize - 1, start + chunkBytes - 1)
            chunks.append(DownloadChunk(index: index, start: start, end: end))
            start = end + 1
            index += 1
        }
        return chunks
    }

    static let stagingDirectoryName = ".incomplete"

    /// Staging path for an in-flight file.
    ///
    /// Everything stages in one hidden directory at the root of the model
    /// cache, never beside the destination file. A CoreML bundle is a
    /// directory that CoreML loads as a unit, so writing temporaries inside
    /// `AudioEncoder.mlmodelc/` would put unexpected content in a bundle we
    /// then hand to the framework. Keeping staging out of the tree also means
    /// a partial transfer can never be mistaken for repo content.
    ///
    /// Nested repo paths are flattened, with a digest of the full path keeping
    /// identically-named members of two different bundles apart.
    static func stagingURL(for repoPath: String, in directory: URL) throws -> URL {
        let staging = directory.appendingPathComponent(stagingDirectoryName, isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        let flattened = repoPath.replacingOccurrences(of: "/", with: "_")
        let discriminator = stableDiscriminator(for: repoPath)
        return staging.appendingPathComponent("\(discriminator)-\(flattened)")
    }

    /// Stable across processes, unlike `hashValue`, which is seeded per launch
    /// — a staging file has to be findable by the *next* run to be resumable.
    static func stableDiscriminator(for repoPath: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in repoPath.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return String(format: "%08x", UInt32(truncatingIfNeeded: hash))
    }

    static func reportByteProgress(
        completedBytes: Int64,
        totalBytes: Int64,
        fileName: String,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) {
        guard totalBytes > 0 else {
            progressHandler?(1.0, completedBytes, totalBytes, fileName)
            return
        }
        let clamped = min(max(completedBytes, 0), totalBytes)
        progressHandler?(Double(clamped) / Double(totalBytes), clamped, totalBytes, fileName)
    }

    static func localFileSize(_ url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else {
            return 0
        }
        return size.int64Value
    }

    // MARK: - Shared request helpers

    static func resolveURL(modelId: String, file: String) throws -> URL {
        let endpoint = (resolvedEndpoint() ?? "https://huggingface.co")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let escaped = file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file
        guard let url = URL(string: "\(endpoint)/\(modelId)/resolve/main/\(escaped)") else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): invalid URL")
        }
        return url
    }

    static func makeHubRequest(
        url: URL,
        method: String? = nil,
        range: String? = nil,
        timeout: TimeInterval? = nil
    ) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let range {
            request.setValue(range, forHTTPHeaderField: "Range")
        }
        if let timeout {
            request.timeoutInterval = timeout
        }
        applyHubAuth(to: &request)
        return request
    }

    static func applyHubAuth(to request: inout URLRequest) {
        let env = ProcessInfo.processInfo.environment
        let token = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"]
        if let token, !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }
}

struct DownloadChunk: Sendable {
    let index: Int
    let start: Int64
    let end: Int64

    var length: Int64 {
        end - start + 1
    }
}

/// Serializes positional writes into the staging file and records which chunks
/// have landed, so an interrupted transfer resumes instead of restarting.
///
/// A lock rather than an actor: the background session delivers a finished
/// range inside a delegate callback and deletes the file it handed over as soon
/// as that callback returns, so the splice has to happen there and then. An
/// actor can only be reached by suspending, which is exactly what a delegate
/// callback cannot do.
final class ChunkedFileWriter: @unchecked Sendable {
    /// Whether this writer picked up a staging file a previous attempt left,
    /// rather than starting one.
    ///
    /// A reset means the manifest's size disagreed with what is on disk, so
    /// everything recorded about the old file describes a different export.
    /// Anything else holding bytes for it has to be discarded too.
    let resumedExistingFile: Bool

    private let lock = NSLock()
    private let handle: FileHandle
    private let sidecar: URL
    private let chunkCount: Int
    private var completed: Set<Int>
    private var closed = false

    init(destination: URL, sidecar: URL, totalSize: Int64, chunkCount: Int) throws {
        self.sidecar = sidecar
        self.chunkCount = chunkCount

        let fm = FileManager.default
        let resumable = fm.fileExists(atPath: destination.path)
            && HuggingFaceDownloader.localFileSize(destination) == totalSize
        self.resumedExistingFile = resumable
        if resumable {
            self.completed = Self.readSidecar(sidecar, chunkCount: chunkCount)
        } else {
            // The staging file's length disagrees with the manifest, so any
            // recorded offsets describe a different layout and resuming from
            // them would corrupt the result. Start over.
            self.completed = []
            try? fm.removeItem(at: destination)
            try? fm.removeItem(at: sidecar)
            guard fm.createFile(atPath: destination.path, contents: nil) else {
                throw DownloadError.failedToDownload(
                    "could not create staging file at \(destination.path)")
            }
        }

        guard let handle = try? FileHandle(forWritingTo: destination) else {
            throw DownloadError.failedToDownload(
                "could not open staging file at \(destination.path)")
        }
        // Give the file its full length up front so every chunk offset is
        // valid regardless of the order chunks arrive in.
        try? handle.truncate(atOffset: UInt64(totalSize))
        self.handle = handle
    }

    func completedChunkIndices() -> Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return completed
    }

    func write(_ data: Data, at offset: Int64, chunkIndex: Int) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else {
            throw DownloadError.failedToDownload("write after close on chunk \(chunkIndex)")
        }
        try handle.seek(toOffset: UInt64(offset))
        try handle.write(contentsOf: data)
        completed.insert(chunkIndex)
        persist()
    }

    func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        try? handle.synchronize()
        try? handle.close()
    }

    private func persist() {
        guard let data = try? JSONSerialization.data(withJSONObject: Array(completed).sorted())
        else { return }
        try? data.write(to: sidecar, options: .atomic)
    }

    private static func readSidecar(_ url: URL, chunkCount: Int) -> Set<Int> {
        guard let data = try? Data(contentsOf: url),
              let indices = try? JSONSerialization.jsonObject(with: data) as? [Int]
        else { return [] }
        return Set(indices.filter { $0 >= 0 && $0 < chunkCount })
    }
}

/// Coalesces byte-progress callbacks to at most one per reported megabyte.
actor RangedDownloadProgress {
    private let completedBeforeFile: Int64
    private let totalBytes: Int64
    private let fileName: String
    private let progressHandler: ((Double, Int64, Int64, String) -> Void)?
    private var fileCompletedBytes: Int64 = 0
    private var lastReportedMegabytes: Int64 = -1

    init(
        completedBeforeFile: Int64,
        totalBytes: Int64,
        fileName: String,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) {
        self.completedBeforeFile = completedBeforeFile
        self.totalBytes = totalBytes
        self.fileName = fileName
        self.progressHandler = progressHandler
    }

    func addCompletedBytes(_ bytes: Int64) {
        fileCompletedBytes += bytes
        let completed = completedBeforeFile + fileCompletedBytes
        let displayedMegabytes = Int64((Double(completed) / 1_000_000.0).rounded())
        if completed < totalBytes, displayedMegabytes == lastReportedMegabytes {
            return
        }
        lastReportedMegabytes = displayedMegabytes
        HuggingFaceDownloader.reportByteProgress(
            completedBytes: completed,
            totalBytes: totalBytes,
            fileName: fileName,
            progressHandler: progressHandler)
    }
}
