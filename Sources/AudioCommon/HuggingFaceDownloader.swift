import Foundation
import Hub
import os

/// Download errors
public enum DownloadError: Error, LocalizedError {
    case failedToDownload(String)
    case invalidRemoteFileName(String)
    /// A download attempt made no progress for `seconds` and was aborted
    /// so the caller's retry loop can fire instead of hanging.
    case stalled(modelId: String, seconds: Int)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        case .stalled(let modelId, let seconds):
            return "Download stalled for \(modelId): no progress in \(seconds)s"
        }
    }
}

/// HuggingFace model downloader — shared between ASR, TTS, VAD, etc.
///
/// Uses `HubApi` from the swift-transformers `Hub` module for downloads,
/// which provides HF token auth and metadata tracking. Files that finished
/// downloading are skipped on retry (etag/commit-hash check), but a file
/// interrupted mid-transfer restarts from byte 0 — there is no usable
/// mid-file resume in the current Hub stack, which is why the stall guard
/// and retry ladder below favor patience over fast abort.
public enum HuggingFaceDownloader {

    // MARK: - Cache Directory

    /// Get cache directory for a model.
    ///
    /// Returns the old flat cache path if it already contains model files (preserving
    /// ~10 GB of existing cached models), otherwise returns the new Hub-style path.
    public static func getCacheDirectory(for modelId: String, basePath: URL? = nil, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = basePath ?? resolveBaseCacheDir(cacheDirName: cacheDirName)
        let fm = FileManager.default

        // Check old (flat) cache path for backward compat:
        //   ~/Library/Caches/qwen3-speech/aufklarer_Qwen3-ASR-0.6B-MLX-4bit/
        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if weightsExist(in: oldDir) {
            return oldDir
        }

        // New Hub-style path:
        //   ~/Library/Caches/qwen3-speech/models/aufklarer/Qwen3-ASR-0.6B-MLX-4bit/
        let hub = HubApi(downloadBase: base)
        let repo = Hub.Repo(id: modelId)
        let dir = hub.localRepoLocation(repo)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - Weight Existence Check

    /// Extensions recognised as cached model weights: the canonical
    /// HF `.safetensors` layout plus Apple CoreML bundle directories
    /// (`.mlmodelc`, `.mlpackage`) shipped by CoreML-only repos.
    public static let weightFileExtensions: Set<String> = [
        "safetensors", "mlmodelc", "mlpackage"
    ]

    /// Returns `true` when `directory` contains at least one entry
    /// whose extension matches `weightFileExtensions`. Used by
    /// `downloadWeights` to short-circuit network requests when
    /// `offlineMode: true` is set on caches that contain only CoreML
    /// bundles and no `.safetensors` files.
    public static func weightsExist(in directory: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.path) else { return false }
        let contents: [URL]
        do {
            contents = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        } catch {
            AudioLog.download.debug("Could not list directory \(directory.path): \(error)")
            contents = []
        }
        return contents.contains { weightFileExtensions.contains($0.pathExtension) }
    }

    // MARK: - Download

    /// Download model files from HuggingFace using `HubApi.snapshot()`.
    ///
    /// Builds glob patterns from the file list:
    /// - Always includes `config.json`
    /// - If `additionalFiles` doesn't contain `.safetensors` files, adds `*.safetensors`
    ///   and `model.safetensors.index.json` to discover sharded weights automatically
    /// - All entries in `additionalFiles` are added as-is (they work as glob patterns)
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        offlineMode: Bool = false,
        retryDelaysSeconds: [Int]? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        // Skip network requests when weights are already cached
        if offlineMode && weightsExist(in: directory) {
            progressHandler?(1.0)
            return
        }

        var globs: [String] = ["config.json"]

        let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
        if !hasExplicitWeights {
            globs.append("*.safetensors")
            globs.append("model.safetensors.index.json")
        }
        for file in additionalFiles where !globs.contains(file) {
            globs.append(file)
        }

        // Derive the download base from the directory.
        // getCacheDirectory returns either:
        //   old: base/cacheKey         (flat, already has weights — won't reach here)
        //   new: base/models/org/model  (Hub-style)
        // For Hub API we need `base` as downloadBase.
        //
        // Forward `offlineMode` explicitly so HubApi doesn't fall through to
        // its internal NWPathMonitor auto-detect, which on macOS can briefly
        // report `.unsatisfied` and then refuse to download (manifesting as
        // "Offline mode error: No files available locally for this repository"
        // for a freshly-requested model).
        let hub = makeHubApi(for: modelId, repoDir: directory, offlineMode: offlineMode)
        let repo = Hub.Repo(id: modelId)

        // Retry with capped backoff — HuggingFace can timeout on slow
        // connections or rate-limit, and flaky networks (hotspots, captive
        // portals) drop out for minutes at a time. Each attempt is wrapped
        // in a progress-stall guard so a wedged mid-transfer (which
        // `hub.snapshot` won't surface on its own) aborts and retries
        // instead of hanging until the CI job is killed.
        //
        // No retries in offline mode: the failure is a deterministic local
        // cache miss, and 110 s of backoff can't change what's on disk.
        let delays = offlineMode ? [] : (retryDelaysSeconds ?? downloadRetryDelaysSeconds)
        let maxAttempts = delays.count + 1
        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                try await withDownloadStallGuard(modelId: modelId) { reportProgress in
                    try await hub.snapshot(from: repo, matching: globs) { progress in
                        reportProgress(progress.fractionCompleted)
                        progressHandler?(progress.fractionCompleted)
                    }
                }
                return  // Success
            } catch {
                lastError = error
                if attempt < maxAttempts {
                    try await Task.sleep(for: .seconds(delays[attempt - 1]))
                }
            }
        }
        throw DownloadError.failedToDownload(
            "\(modelId) after \(maxAttempts) attempt\(maxAttempts == 1 ? "" : "s") "
                + "(target: \(directory.path)): "
                + (lastError?.localizedDescription ?? "unknown"))
    }

    /// Download an explicit list of files from HuggingFace without adding any
    /// implicit weight globs. This is useful for overlaying tokenizer or config
    /// assets from a second repository on top of an existing cache.
    public static func downloadFiles(
        modelId: String,
        to directory: URL,
        files: [String],
        offlineMode: Bool = false,
        retryDelaysSeconds: [Int]? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        if files.isEmpty {
            progressHandler?(1.0)
            return
        }

        let hub = makeHubApi(for: modelId, repoDir: directory, offlineMode: offlineMode)
        let repo = Hub.Repo(id: modelId)

        let globs = files.map { $0 }
        // Same retry semantics as downloadWeights, including the offline
        // no-retry rule — keep the two loops in lockstep.
        let delays = offlineMode ? [] : (retryDelaysSeconds ?? downloadRetryDelaysSeconds)
        let maxAttempts = delays.count + 1
        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                try await withDownloadStallGuard(modelId: modelId) { reportProgress in
                    try await hub.snapshot(from: repo, matching: globs) { progress in
                        reportProgress(progress.fractionCompleted)
                        progressHandler?(progress.fractionCompleted)
                    }
                }
                return
            } catch {
                lastError = error
                if attempt < maxAttempts {
                    try await Task.sleep(for: .seconds(delays[attempt - 1]))
                }
            }
        }
        throw DownloadError.failedToDownload(
            "\(modelId) after \(maxAttempts) attempt\(maxAttempts == 1 ? "" : "s") "
                + "(target: \(directory.path)): "
                + (lastError?.localizedDescription ?? "unknown"))
    }

    /// Download an explicit file list with progress weighted by byte size.
    ///
    /// `HubApi.snapshot()` reports useful progress for many repos, but for
    /// large Xet-backed checkpoints it can advance by manifest/file count and
    /// then remain quiet while multi-GB shards download. This path resolves
    /// each file with `HEAD`, sums real `Content-Length` values, skips already
    /// complete files, and reports progress from actual transferred bytes.
    public static func downloadFilesByteWeighted(
        modelId: String,
        to directory: URL,
        files: [String],
        expectedSizes: [String: Int64]? = nil,
        offlineMode: Bool = false,
        retryDelaysSeconds: [Int]? = nil,
        progressHandler: ((Double, Int64, Int64, String) -> Void)? = nil
    ) async throws {
        if files.isEmpty {
            progressHandler?(1.0, 0, 0, "")
            return
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let safeFiles = try files.map(validatedRemoteFileName)
        if offlineMode {
            let missing = safeFiles.first {
                !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
            }
            if let missing {
                throw DownloadError.failedToDownload(
                    "\(modelId) offline cache miss: \(missing) is not present in \(directory.path)")
            }
            progressHandler?(1.0, 1, 1, "")
            return
        }

        let delays = retryDelaysSeconds ?? downloadRetryDelaysSeconds
        let maxAttempts = delays.count + 1
        var lastError: Error?
        for attempt in 1...maxAttempts {
            do {
                let remoteFiles = try await resolveRemoteFiles(
                    modelId: modelId,
                    files: safeFiles,
                    expectedSizes: expectedSizes)
                try await downloadResolvedFilesByteWeighted(
                    remoteFiles,
                    modelId: modelId,
                    to: directory,
                    progressHandler: progressHandler)
                return
            } catch {
                lastError = error
                if attempt < maxAttempts {
                    try await Task.sleep(for: .seconds(delays[attempt - 1]))
                }
            }
        }

        throw DownloadError.failedToDownload(
            "\(modelId) after \(maxAttempts) attempt\(maxAttempts == 1 ? "" : "s") "
                + "(target: \(directory.path)): "
                + (lastError?.localizedDescription ?? "unknown"))
    }

    // MARK: - Retry ladder

    /// Delays between download attempts. One more attempt than entries:
    /// 5 attempts with 5/15/30/60 s pauses (~110 s of backoff on top of the
    /// per-attempt stall patience). Generous on purpose — abandoned attempts
    /// restart files from byte 0 with the current Hub stack, so the cheap
    /// resource here is wall-clock, not bytes. A network that's down for a
    /// couple of minutes (AP roam, hotspot sleep, captive-portal re-auth)
    /// should not kill a 2.75 GB first-run download.
    static let downloadRetryDelaysSeconds = [5, 15, 30, 60]

    /// Total attempts per download (retries + the initial try).
    static var downloadMaxAttempts: Int { downloadRetryDelaysSeconds.count + 1 }

    // MARK: - Download stall guard

    /// Seconds of zero download progress after which an attempt is
    /// considered wedged and aborted. `hub.snapshot` reports
    /// `fractionCompleted` continuously while bytes flow, so a healthy
    /// (even slow) transfer keeps resetting the clock; only a genuinely
    /// stalled connection trips this.
    ///
    /// The default is tuned for end users, not CI: aborted attempts restart
    /// each file from byte 0 (the Hub stack's mid-file resume never engages
    /// on a fresh download), so firing the guard on a connection that would
    /// have recovered throws away every byte of that attempt. Flaky networks
    /// — AP roams, captive-portal re-auth, hotspot sleep — routinely stall
    /// for 1–3 minutes and then recover, hence 300 s. CI pins
    /// `HF_DOWNLOAD_STALL_TIMEOUT=90` to keep failing fast (app users can't
    /// set env vars; CI can).
    static var downloadStallTimeoutSeconds: Int {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_STALL_TIMEOUT"],
           let v = Int(raw), v > 0 {
            return v
        }
        return 300
    }

    /// Concurrent range requests used by the byte-weighted downloader for
    /// large files. Fish Audio and similar multi-GB checkpoints are hosted
    /// behind HF/CDN endpoints where one stream can under-fill the user's
    /// connection; bounded parallel ranges improve first-run download time
    /// while still avoiding unbounded request fan-out.
    static var downloadRangeConcurrency: Int {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_RANGE_CONCURRENCY"],
           let v = Int(raw), v > 0 {
            return min(v, 16)
        }
        return 16
    }

    /// Resolve the HuggingFace Hub endpoint, honoring the `HF_ENDPOINT`
    /// environment variable (the same name Python's `huggingface_hub` uses).
    ///
    /// Users behind the Great Firewall set `HF_ENDPOINT=https://hf-mirror.com`
    /// to fetch models from the China mirror. The cache layout is keyed by repo
    /// id, not host, so switching the endpoint reuses any already-downloaded
    /// weights and needs no re-download.
    ///
    /// Returns `nil` (so `HubApi` keeps its default `https://huggingface.co`)
    /// when the variable is unset, blank, or not a valid `http(s)://host` URL.
    /// The validation mirrors `HubApi`'s own guard so a malformed value falls
    /// back to the default instead of breaking downloads.
    ///
    /// Public so modules that construct `HubApi` directly (e.g.
    /// `FunctionGemma.loadFromHub`, which calls `snapshot(from:)` instead of
    /// going through `downloadWeights`) can still honor the mirror.
    public static func resolvedEndpoint() -> String? {
        guard let raw = ProcessInfo.processInfo.environment["HF_ENDPOINT"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty,
              let url = URL(string: raw),
              let scheme = url.scheme, scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty
        else { return nil }
        return raw
    }

    /// Thread-safe last-progress timestamp. `hub.snapshot`'s progress
    /// callback may fire from a background queue, so guard with a lock.
    private final class ProgressClock: @unchecked Sendable {
        private let lock = NSLock()
        private var last = Date()
        func tick() { lock.lock(); last = Date(); lock.unlock() }
        func idleSeconds() -> Double {
            lock.lock(); defer { lock.unlock() }
            return Date().timeIntervalSince(last)
        }
    }

    /// Run a download `operation` that reports fractional progress, and
    /// abort it if progress stalls for `downloadStallTimeoutSeconds`.
    /// On stall the in-flight `hub.snapshot` task is cancelled (URLSession
    /// honors cancellation) and `DownloadError.stalled` is thrown so the
    /// caller's retry loop fires instead of hanging indefinitely.
    static func withDownloadStallGuard(
        modelId: String,
        stallTimeoutSeconds: Int? = nil,
        _ operation: @escaping (@escaping @Sendable (Double) -> Void) async throws -> Void
    ) async throws {
        let stall = stallTimeoutSeconds ?? downloadStallTimeoutSeconds
        let clock = ProgressClock()

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await operation { _ in clock.tick() }
            }
            group.addTask {
                // Poll on a fraction of the window so we detect a stall
                // within ~stall..stall+pollStep seconds.
                let pollStep = max(1, stall / 3)
                while true {
                    try await Task.sleep(for: .seconds(pollStep))
                    if clock.idleSeconds() >= Double(stall) {
                        throw DownloadError.stalled(modelId: modelId, seconds: stall)
                    }
                }
            }
            // Whichever finishes first wins; cancel the other (the poller
            // on success, or the download on stall).
            defer { group.cancelAll() }
            try await group.next()
        }
    }

    // MARK: - Security Helpers (kept for backward compat + security tests)

    /// Convert an arbitrary modelId into a single, safe path component for on-disk caching.
    public static func sanitizedCacheKey(for modelId: String) -> String {
        let replaced = modelId.replacingOccurrences(of: "/", with: "_")

        let allowed = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        var scalars: [UnicodeScalar] = []
        scalars.reserveCapacity(replaced.unicodeScalars.count)
        for s in replaced.unicodeScalars {
            scalars.append(allowed.contains(s) ? s : "_")
        }

        var cleaned = String(String.UnicodeScalarView(scalars))
        cleaned = cleaned.trimmingCharacters(in: CharacterSet(charactersIn: "._"))

        if cleaned.isEmpty || cleaned == "." || cleaned == ".." {
            cleaned = "model"
        }

        return cleaned
    }

    /// Validate that a remote file name is safe.
    public static func validatedRemoteFileName(_ file: String) throws -> String {
        let base = URL(fileURLWithPath: file).lastPathComponent
        guard base == file else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard !base.isEmpty, !base.hasPrefix("."), !base.contains("..") else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        guard base.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
            throw DownloadError.invalidRemoteFileName(file)
        }
        return base
    }

    /// Validate that a local path stays within the expected directory.
    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        let local = directory.appendingPathComponent(fileName, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(fileName)
        }
        return local
    }

    // MARK: - Private Helpers

    /// Resolve the base cache directory from env vars or system default.
    private static func resolveBaseCacheDir(cacheDirName: String) -> URL {
        let fm = FileManager.default
        let root: URL
        if let override = ProcessInfo.processInfo.environment["QWEN3_CACHE_DIR"],
           !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else if let override = ProcessInfo.processInfo.environment["QWEN3_ASR_CACHE_DIR"],
                  !override.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Legacy env var support
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fm.urls(for: .cachesDirectory, in: .userDomainMask).first!
        }
        return root.appendingPathComponent(cacheDirName, isDirectory: true)
    }

    /// Create a `HubApi` whose `downloadBase` is derived from the repo directory that
    /// `getCacheDirectory` returned (strips the `models/<org>/<model>` suffix).
    ///
    /// `offlineMode` is forwarded as `useOfflineMode` so callers get the mode
    /// they asked for instead of relying on `NWPathMonitor` auto-detection,
    /// which can spuriously report `.unsatisfied` on macOS.
    private static func makeHubApi(for modelId: String, repoDir: URL, offlineMode: Bool) -> HubApi {
        // repoDir is  base/models/org/model
        // We need     base
        let repo = Hub.Repo(id: modelId)
        let suffix = "/\(repo.type.rawValue)/\(repo.id)"
        let repoDirPath = repoDir.path
        let downloadBase: URL
        if repoDirPath.hasSuffix(suffix) {
            let basePath = String(repoDirPath.dropLast(suffix.count))
            downloadBase = URL(fileURLWithPath: basePath, isDirectory: true)
        } else {
            // Fallback: old-style flat dir — use its parent as downloadBase.
            // Hub won't match this path, so we derive base from env/defaults.
            downloadBase = resolveBaseCacheDir(cacheDirName: repoDir.deletingLastPathComponent().lastPathComponent)
        }
        return HubApi(downloadBase: downloadBase, endpoint: resolvedEndpoint(), useOfflineMode: offlineMode)
    }
}

// MARK: - Byte-weighted explicit downloads

private struct ResolvedRemoteFile: Sendable {
    let fileName: String
    let url: URL
    let size: Int64
}

private extension HuggingFaceDownloader {
    static let rangedDownloadThresholdBytes: Int64 = 64 * 1_024 * 1_024
    static let rangedDownloadChunkBytes: Int64 = 16 * 1_024 * 1_024

    static func resolveRemoteFiles(
        modelId: String,
        files: [String],
        expectedSizes: [String: Int64]?
    ) async throws -> [ResolvedRemoteFile] {
        if let expectedSizes {
            return try files.map { file in
                guard let size = expectedSizes[file], size > 0 else {
                    throw DownloadError.failedToDownload("\(modelId)/\(file): missing expected size")
                }
                return ResolvedRemoteFile(
                    fileName: file,
                    url: try resolveURL(modelId: modelId, file: file),
                    size: size)
            }
        }

        var result: [ResolvedRemoteFile] = []
        result.reserveCapacity(files.count)
        for file in files {
            result.append(try await resolveRemoteFile(modelId: modelId, file: file))
        }
        return result
    }

    static func resolveRemoteFile(modelId: String, file: String) async throws -> ResolvedRemoteFile {
        let url = try resolveURL(modelId: modelId, file: file)
        var request = URLRequest(url: url)
        request.httpMethod = "HEAD"
        applyHubAuth(to: &request)
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): missing HTTP response")
        }
        guard (200..<300).contains(http.statusCode) else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): HTTP \(http.statusCode)")
        }
        guard let resolved = http.url else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): missing resolved URL")
        }
        let size = headerInt64(http, "Content-Length")
            ?? headerInt64(http, "x-linked-size")
            ?? http.expectedContentLength
        guard size > 0 else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): unknown remote size")
        }
        return ResolvedRemoteFile(fileName: file, url: resolved, size: size)
    }

    static func downloadResolvedFilesByteWeighted(
        _ files: [ResolvedRemoteFile],
        modelId: String,
        to directory: URL,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) async throws {
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        var completedBytes: Int64 = 0

        for file in files {
            let localURL = directory.appendingPathComponent(file.fileName, isDirectory: false)
            let currentSize = localFileSize(localURL)
            if currentSize == file.size {
                completedBytes += file.size
                reportByteProgress(
                    completedBytes: completedBytes,
                    totalBytes: totalBytes,
                    fileName: file.fileName,
                    progressHandler: progressHandler)
                continue
            }
            if currentSize > 0 {
                try? FileManager.default.removeItem(at: localURL)
            }

            reportByteProgress(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                fileName: file.fileName,
                progressHandler: progressHandler)
            if file.size >= rangedDownloadThresholdBytes {
                try await downloadRangedResolvedFile(
                    file,
                    to: localURL,
                    completedBeforeFile: completedBytes,
                    totalBytes: totalBytes,
                    progressHandler: progressHandler)
            } else {
                try await downloadSingleResolvedFile(
                    file,
                    to: localURL,
                    completedBeforeFile: completedBytes,
                    totalBytes: totalBytes,
                    progressHandler: progressHandler)
            }
            completedBytes += file.size
            reportByteProgress(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                fileName: file.fileName,
                progressHandler: progressHandler)

            let writtenSize = localFileSize(localURL)
            guard writtenSize == file.size else {
                throw DownloadError.failedToDownload(
                    "\(modelId)/\(file.fileName): wrote \(writtenSize) bytes, expected \(file.size)")
            }
        }
    }

    static func downloadSingleResolvedFile(
        _ file: ResolvedRemoteFile,
        to destination: URL,
        completedBeforeFile: Int64,
        totalBytes: Int64,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) async throws {
        var request = URLRequest(url: file.url)
        applyHubAuth(to: &request)
        let tempURL = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).download")
        try? FileManager.default.removeItem(at: tempURL)

        let delegate = ByteWeightedDownloadDelegate(
            destination: tempURL,
            expectedSize: file.size,
            completedBeforeFile: completedBeforeFile,
            totalBytes: totalBytes,
            fileName: file.fileName,
            progressHandler: progressHandler)
        let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)
        defer {
            session.invalidateAndCancel()
            try? FileManager.default.removeItem(at: tempURL)
        }

        try await withTaskCancellationHandler {
            try await delegate.download(request: request, session: session)
        } onCancel: {
            session.invalidateAndCancel()
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)
    }

    static func downloadRangedResolvedFile(
        _ file: ResolvedRemoteFile,
        to destination: URL,
        completedBeforeFile: Int64,
        totalBytes: Int64,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) async throws {
        let partsDirectory = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).parts", isDirectory: true)
        try FileManager.default.createDirectory(at: partsDirectory, withIntermediateDirectories: true)

        let chunks = makeDownloadChunks(fileSize: file.size, chunkBytes: rangedDownloadChunkBytes)
        let state = RangedDownloadProgress(
            completedBeforeFile: completedBeforeFile,
            totalBytes: totalBytes,
            fileName: file.fileName,
            progressHandler: progressHandler)
        let concurrency = downloadRangeConcurrency
        let configuration = URLSessionConfiguration.default
        configuration.httpMaximumConnectionsPerHost = max(concurrency, 1)
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }

        var missingChunks: [DownloadChunk] = []
        missingChunks.reserveCapacity(chunks.count)
        for chunk in chunks {
            let partURL = partFileURL(partsDirectory: partsDirectory, chunk: chunk)
            let partSize = localFileSize(partURL)
            if partSize == chunk.length {
                await state.addCompletedBytes(chunk.length)
                continue
            }
            if partSize > 0 {
                try? FileManager.default.removeItem(at: partURL)
            }
            missingChunks.append(chunk)
        }

        if !missingChunks.isEmpty {
            try await withThrowingTaskGroup(of: Void.self) { group in
                var nextIndex = 0
                let initial = min(concurrency, missingChunks.count)
                for _ in 0..<initial {
                    let chunk = missingChunks[nextIndex]
                    nextIndex += 1
                    group.addTask {
                        try await downloadRangeChunk(
                            file,
                            chunk: chunk,
                            to: partFileURL(partsDirectory: partsDirectory, chunk: chunk),
                            state: state,
                            session: session)
                    }
                }

                while try await group.next() != nil {
                    if nextIndex < missingChunks.count {
                        let chunk = missingChunks[nextIndex]
                        nextIndex += 1
                        group.addTask {
                            try await downloadRangeChunk(
                                file,
                                chunk: chunk,
                                to: partFileURL(partsDirectory: partsDirectory, chunk: chunk),
                                state: state,
                                session: session)
                        }
                    }
                }
            }
        }

        let tempURL = destination
            .deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).download")
        try? FileManager.default.removeItem(at: tempURL)
        FileManager.default.createFile(atPath: tempURL.path, contents: nil)
        let output = try FileHandle(forWritingTo: tempURL)
        defer { try? output.close() }

        for chunk in chunks {
            let partURL = partFileURL(partsDirectory: partsDirectory, chunk: chunk)
            let data = try Data(contentsOf: partURL)
            guard Int64(data.count) == chunk.length else {
                throw DownloadError.failedToDownload(
                    "\(file.fileName): part \(chunk.index) has \(data.count) bytes, expected \(chunk.length)")
            }
            try output.write(contentsOf: data)
        }

        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: tempURL, to: destination)
        try? FileManager.default.removeItem(at: partsDirectory)
    }

    static func downloadRangeChunk(
        _ file: ResolvedRemoteFile,
        chunk: DownloadChunk,
        to destination: URL,
        state: RangedDownloadProgress,
        session: URLSession
    ) async throws {
        var request = URLRequest(url: file.url)
        request.setValue("bytes=\(chunk.start)-\(chunk.end)", forHTTPHeaderField: "Range")
        applyHubAuth(to: &request)

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw DownloadError.failedToDownload("\(file.fileName) part \(chunk.index): missing HTTP response")
        }
        guard http.statusCode == 206 else {
            throw DownloadError.failedToDownload(
                "\(file.fileName) part \(chunk.index): expected HTTP 206, got \(http.statusCode)")
        }
        guard Int64(data.count) == chunk.length else {
            throw DownloadError.failedToDownload(
                "\(file.fileName) part \(chunk.index): got \(data.count) bytes, expected \(chunk.length)")
        }
        try data.write(to: destination, options: .atomic)
        await state.addCompletedBytes(chunk.length)
    }

    static func makeDownloadChunks(fileSize: Int64, chunkBytes: Int64) -> [DownloadChunk] {
        guard fileSize > 0 else { return [] }
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

    static func partFileURL(partsDirectory: URL, chunk: DownloadChunk) -> URL {
        partsDirectory.appendingPathComponent(String(format: "%06d.part", chunk.index))
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

    static func resolveURL(modelId: String, file: String) throws -> URL {
        let endpoint = (resolvedEndpoint() ?? "https://huggingface.co")
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let escapedFile = file.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file
        guard let url = URL(string: "\(endpoint)/\(modelId)/resolve/main/\(escapedFile)") else {
            throw DownloadError.failedToDownload("\(modelId)/\(file): invalid URL")
        }
        return url
    }

    static func applyHubAuth(to request: inout URLRequest) {
        let env = ProcessInfo.processInfo.environment
        let token = env["HF_TOKEN"] ?? env["HUGGING_FACE_HUB_TOKEN"]
        if let token, !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    static func headerInt64(_ response: HTTPURLResponse, _ key: String) -> Int64? {
        for (rawKey, rawValue) in response.allHeaderFields {
            guard String(describing: rawKey).caseInsensitiveCompare(key) == .orderedSame else {
                continue
            }
            if let value = rawValue as? NSNumber {
                return value.int64Value
            }
            return Int64(String(describing: rawValue))
        }
        return nil
    }

    static func localFileSize(_ url: URL) -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? NSNumber else {
            return 0
        }
        return size.int64Value
    }
}

private struct DownloadChunk: Sendable {
    let index: Int
    let start: Int64
    let end: Int64

    var length: Int64 {
        end - start + 1
    }
}

private actor RangedDownloadProgress {
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

private final class ByteWeightedDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let destination: URL
    private let expectedSize: Int64
    private let completedBeforeFile: Int64
    private let totalBytes: Int64
    private let fileName: String
    private let progressHandler: ((Double, Int64, Int64, String) -> Void)?
    private var continuation: CheckedContinuation<Void, Error>?
    private var completed = false
    private var lastReportedMegabytes: Int64 = -1

    init(
        destination: URL,
        expectedSize: Int64,
        completedBeforeFile: Int64,
        totalBytes: Int64,
        fileName: String,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?
    ) {
        self.destination = destination
        self.expectedSize = expectedSize
        self.completedBeforeFile = completedBeforeFile
        self.totalBytes = totalBytes
        self.fileName = fileName
        self.progressHandler = progressHandler
    }

    func download(request: URLRequest, session: URLSession) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            lock.lock()
            self.continuation = continuation
            lock.unlock()
            session.downloadTask(with: request).resume()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let fileBytes = totalBytesWritten >= 0 ? totalBytesWritten : 0
        let completed = completedBeforeFile + min(fileBytes, expectedSize)
        let displayedMegabytes = Int64((Double(completed) / 1_000_000.0).rounded())
        lock.lock()
        if completed < totalBytes, displayedMegabytes == lastReportedMegabytes {
            lock.unlock()
            return
        }
        lastReportedMegabytes = displayedMegabytes
        lock.unlock()
        HuggingFaceDownloader.reportByteProgress(
            completedBytes: completed,
            totalBytes: totalBytes,
            fileName: fileName,
            progressHandler: progressHandler)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let http = downloadTask.response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            complete(.failure(DownloadError.failedToDownload("\(fileName): HTTP \(http.statusCode)")))
            return
        }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)
            let writtenSize = HuggingFaceDownloader.localFileSize(destination)
            guard writtenSize == expectedSize else {
                throw DownloadError.failedToDownload(
                    "\(fileName): wrote \(writtenSize) bytes, expected \(expectedSize)")
            }
            complete(.success(()))
        } catch {
            complete(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            complete(.failure(error))
        }
    }

    private func complete(_ result: Result<Void, Error>) {
        lock.lock()
        guard !completed else {
            lock.unlock()
            return
        }
        completed = true
        let continuation = self.continuation
        self.continuation = nil
        lock.unlock()

        switch result {
        case .success:
            continuation?.resume()
        case .failure(let error):
            continuation?.resume(throwing: error)
        }
    }
}
