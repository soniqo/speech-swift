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
    /// Downloaded bytes did not match the digest the Hub published for them.
    case checksumMismatch(file: String, expected: String, actual: String)
    /// Every attempt failed for a reason that looks like the network being
    /// unreachable, rather than the repository being wrong. Distinguished from
    /// `failedToDownload` so a complete cache can still be used.
    case networkUnavailable(modelId: String, detail: String)

    public var errorDescription: String? {
        switch self {
        case .failedToDownload(let file):
            return "Failed to download: \(file)"
        case .invalidRemoteFileName(let file):
            return "Refusing to write unsafe remote file name: \(file)"
        case .stalled(let modelId, let seconds):
            return "Download stalled for \(modelId): no progress in \(seconds)s"
        case .checksumMismatch(let file, let expected, let actual):
            return "Checksum mismatch for \(file): expected \(expected), got \(actual)"
        case .networkUnavailable(let modelId, let detail):
            return "Network unavailable while fetching \(modelId): \(detail)"
        }
    }
}

/// HuggingFace model downloader — shared between ASR, TTS, VAD, etc.
///
/// Every entry point resolves the repository once against the Hub tree API and
/// then transfers through the same engine: large files as concurrent byte
/// ranges that resume mid-file, progress weighted by real bytes, and LFS
/// digests checked before a file is accepted.
///
/// The three functions differ only in how they choose files — glob discovery
/// (`downloadWeights`) or an explicit list (`downloadFiles`,
/// `downloadFilesByteWeighted`) — not in how they fetch them.
public enum HuggingFaceDownloader {

    // MARK: - Cache Directory

    /// Get cache directory for a model.
    ///
    /// Returns the old flat cache path if it already contains model files
    /// (preserving existing cached models), otherwise the Hub-style path.
    ///
    /// The legacy probe is memoized: it walks a directory and may parse a shard
    /// index, and this is called on essentially every model load. The layout of
    /// a cache directory does not change under a running process, so answering
    /// it once per model is enough.
    public static func getCacheDirectory(for modelId: String, basePath: URL? = nil, cacheDirName: String = "qwen3-speech") throws -> URL {
        let base = basePath ?? resolveBaseCacheDir(cacheDirName: cacheDirName)

        //   old: ~/Library/Caches/qwen3-speech/aufklarer_Model/
        //   new: ~/Library/Caches/qwen3-speech/models/aufklarer/Model/
        let oldDir = base.appendingPathComponent(sanitizedCacheKey(for: modelId), isDirectory: true)
        if legacyCacheDirectoryIsPopulated(oldDir) {
            return oldDir
        }

        let hub = HubApi(downloadBase: base)
        let dir = hub.localRepoLocation(Hub.Repo(id: modelId))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static let legacyCacheProbe = OSAllocatedUnfairLock(initialState: [String: Bool]())

    private static func legacyCacheDirectoryIsPopulated(_ directory: URL) -> Bool {
        legacyCacheProbe.withLock { cache in
            if let known = cache[directory.path] { return known }
            let populated = weightsExist(in: directory)
            cache[directory.path] = populated
            return populated
        }
    }

    // MARK: - Weight Existence Check

    /// Extensions recognised as cached model weights: the canonical
    /// HF `.safetensors` layout plus Apple CoreML bundle directories
    /// (`.mlmodelc`, `.mlpackage`) shipped by CoreML-only repos.
    public static let weightFileExtensions: Set<String> = [
        "safetensors", "mlmodelc", "mlpackage"
    ]

    /// Returns `true` when `directory` contains at least one entry
    /// whose extension matches `weightFileExtensions`, and — for a sharded
    /// bundle — every shard its index names.
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
        guard contents.contains(where: { weightFileExtensions.contains($0.pathExtension) }) else {
            return false
        }
        // A multi-shard bundle is only complete when every shard named in
        // its index is present — a stalled download can leave later shards
        // fully written while earlier ones are missing, and a half-loaded
        // model runs with random weights (observed as near-silent TTS).
        let indexURL = directory.appendingPathComponent(safetensorsIndexName)
        if fm.fileExists(atPath: indexURL.path),
            let data = try? Data(contentsOf: indexURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let weightMap = json["weight_map"] as? [String: String]
        {
            for shard in Set(weightMap.values)
            where !fm.fileExists(atPath: directory.appendingPathComponent(shard).path) {
                AudioLog.download.debug("weightsExist: missing shard \(shard) in \(directory.path)")
                return false
            }
        }
        return true
    }

    // MARK: - Download

    /// Download model files from HuggingFace, discovering weights by glob.
    ///
    /// Builds glob patterns from the file list:
    /// - Always includes `config.json`
    /// - If `additionalFiles` doesn't contain `.safetensors` files, adds
    ///   `*.safetensors` and `model.safetensors.index.json` to discover
    ///   sharded weights automatically
    /// - All entries in `additionalFiles` are added as-is (they work as glob
    ///   patterns, including `encoder.mlmodelc/**` for CoreML bundles)
    public static func downloadWeights(
        modelId: String,
        to directory: URL,
        additionalFiles: [String] = [],
        offlineMode: Bool = false,
        retryDelaysSeconds: [Int]? = nil,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        if offlineMode {
            guard weightsExist(in: directory) else {
                throw DownloadError.failedToDownload(
                    "\(modelId) offline cache miss: no model weights in \(directory.path)")
            }
            progressHandler?(1.0)
            return
        }

        let globs: [String] = {
            var patterns: [String] = ["config.json"]
            let hasExplicitWeights = additionalFiles.contains { $0.hasSuffix(".safetensors") }
            if !hasExplicitWeights {
                patterns.append("*.safetensors")
                patterns.append(safetensorsIndexName)
            }
            for file in additionalFiles where !patterns.contains(file) {
                patterns.append(file)
            }
            return patterns
        }()

        do {
            try await performDownload(
                modelId: modelId,
                directory: directory,
                retryDelaysSeconds: retryDelaysSeconds,
                progressHandler: { fraction, _, _, _ in progressHandler?(fraction) }
            ) {
                let manifest = try await fetchManifest(modelId: modelId)
                var selection = manifest.matching(globs: globs)
                // Globbing is discovery, so it needs narrowing when a repo
                // publishes the same tensors twice — see `applyingShardIndex`.
                if let indexData = await fetchShardIndex(modelId: modelId, manifest: manifest) {
                    selection = applyingShardIndex(to: selection, indexData: indexData)
                }
                return selection
            }
        } catch let error as DownloadError {
            // The Hub was unreachable. If everything this call would have
            // fetched is already on disk, the model can still load — a network
            // blip should not fail a load that needs no bytes. Anything else
            // (a 404, a checksum failure) is a real answer and propagates.
            guard case .networkUnavailable = error,
                  cachedBundleSatisfies(globs: globs, in: directory)
            else { throw error }
            AudioLog.download.warning(
                "\(modelId, privacy: .public): Hub unreachable, using the complete cache in \(directory.path, privacy: .public)")
            progressHandler?(1.0)
            return
        }

        // Caches populated before weight selection consulted the index may hold
        // a consolidated copy of the shards, which the loader would read on top
        // of them. Cleaning up after the fact is the only way those caches ever
        // stop paying for it.
        if let indexData = try? Data(
            contentsOf: directory.appendingPathComponent(safetensorsIndexName)) {
            removeRedundantConsolidatedWeights(in: directory, indexData: indexData)
        }
    }

    /// Whether the cache holds everything a `downloadWeights` call asked for.
    ///
    /// Stricter than `weightsExist` alone, which only answers "are there any
    /// weights here" — a cache with weights but no tokenizer would pass that
    /// and then fail at load, blaming the wrong thing. The weight globs are
    /// checked by `weightsExist` (which also verifies every shard an index
    /// names); the caller's own patterns are checked literally.
    static func cachedBundleSatisfies(globs: [String], in directory: URL) -> Bool {
        guard weightsExist(in: directory) else { return false }
        let discoveryPatterns: Set<String> = ["*.safetensors", safetensorsIndexName]
        let callerPatterns = globs.filter { !discoveryPatterns.contains($0) }
        return firstUnsatisfiedPattern(callerPatterns, in: directory) == nil
    }

    /// Download a list of files without adding any implicit weight globs.
    /// Useful for overlaying tokenizer or config assets from a second
    /// repository on top of an existing cache.
    ///
    /// Entries are glob patterns, not literal names — `LocalVQEEchoCanceller`
    /// asks for `LocalVQEAECResidualMask.mlmodelc/**`, for instance. A pattern
    /// that matches nothing contributes nothing and is not an error, matching
    /// what this function did when it was backed by `HubApi.snapshot`; callers
    /// that need a file to exist find out when they try to load it.
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

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        if offlineMode {
            if let missing = firstUnsatisfiedPattern(files, in: directory) {
                throw DownloadError.failedToDownload(
                    "\(modelId) offline cache miss: nothing matching \(missing) in \(directory.path)")
            }
            progressHandler?(1.0)
            return
        }

        do {
            try await performDownload(
                modelId: modelId,
                directory: directory,
                retryDelaysSeconds: retryDelaysSeconds,
                progressHandler: { fraction, _, _, _ in progressHandler?(fraction) }
            ) {
                try await fetchManifest(modelId: modelId).matching(globs: files)
            }
        } catch let error as DownloadError {
            guard case .networkUnavailable = error,
                  firstUnsatisfiedPattern(files, in: directory) == nil
            else { throw error }
            AudioLog.download.warning(
                "\(modelId, privacy: .public): Hub unreachable, every requested file is already cached")
            progressHandler?(1.0)
            return
        }
    }

    /// First pattern in `patterns` with no match on disk, or `nil` if all are
    /// satisfied. Used to answer offline requests without the network.
    static func firstUnsatisfiedPattern(_ patterns: [String], in directory: URL) -> String? {
        var localPaths: [String]?
        for pattern in patterns {
            if pattern.contains("*") || pattern.contains("?") || pattern.contains("[") {
                // Resolving a pattern needs the local file list; build it once
                // and only if some pattern actually requires it.
                let paths = localPaths ?? localRelativePaths(in: directory)
                localPaths = paths
                if !paths.contains(where: { fnmatch(pattern, $0, 0) == 0 }) {
                    return pattern
                }
            } else if !FileManager.default.fileExists(
                atPath: directory.appendingPathComponent(pattern).path) {
                return pattern
            }
        }
        return nil
    }

    /// Repo-relative paths of the files cached under `directory`, ignoring our
    /// own staging area.
    static func localRelativePaths(in directory: URL) -> [String] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: directory.path) else { return [] }
        var paths: [String] = []
        for case let entry as String in enumerator {
            if entry == stagingDirectoryName || entry.hasPrefix(stagingDirectoryName + "/") {
                enumerator.skipDescendants()
                continue
            }
            paths.append(entry)
        }
        return paths
    }

    /// Download an explicit file list with progress weighted by byte size.
    ///
    /// - Parameter expectedSizes: ignored. Sizes now come from the repository
    ///   listing, which cannot go stale the way a hand-maintained table does.
    ///   Kept so existing callers continue to compile.
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
        let safeFiles = try files.map(validatedRelativePath)

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

        do {
            try await performDownload(
                modelId: modelId,
                directory: directory,
                retryDelaysSeconds: retryDelaysSeconds,
                progressHandler: progressHandler
            ) {
                let manifest = try await fetchManifest(modelId: modelId)
                return try safeFiles.map { path in
                    guard let file = manifest.file(at: path) else {
                        throw DownloadError.failedToDownload(
                            "\(modelId)/\(path): not present in repository")
                    }
                    return file
                }
            }
        } catch let error as DownloadError {
            let fm = FileManager.default
            let allCached = safeFiles.allSatisfy {
                fm.fileExists(atPath: directory.appendingPathComponent($0).path)
            }
            guard case .networkUnavailable = error, allCached else { throw error }
            AudioLog.download.warning(
                "\(modelId, privacy: .public): Hub unreachable, every requested file is already cached")
            progressHandler?(1.0, 1, 1, "")
            return
        }
    }

    /// Shared retry ladder + stall guard around one resolve-then-transfer pass.
    static func performDownload(
        modelId: String,
        directory: URL,
        retryDelaysSeconds: [Int]?,
        progressHandler: ((Double, Int64, Int64, String) -> Void)?,
        resolve: @escaping @Sendable () async throws -> [RepoFile]
    ) async throws {
        let delays = retryDelaysSeconds ?? downloadRetryDelaysSeconds
        let maxAttempts = delays.count + 1
        var lastError: Error?

        for attempt in 1...maxAttempts {
            do {
                try await withDownloadStallGuardIfInProcess(modelId: modelId) { tick in
                    let files = try await resolve()
                    tick(0)
                    try await downloadManifestFiles(
                        files,
                        modelId: modelId,
                        to: directory
                    ) { fraction, completed, total, name in
                        tick(fraction)
                        progressHandler?(fraction, completed, total, name)
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

        // Distinguish "we couldn't reach the Hub" from "the Hub said no". Only
        // the former is safe to answer from cache: a 404 means the caller asked
        // for something that isn't there, and silently loading whatever happens
        // to be on disk would attribute the resulting failure to the wrong thing.
        if let lastError, isLikelyNetworkFailure(lastError) {
            throw DownloadError.networkUnavailable(
                modelId: modelId, detail: lastError.localizedDescription)
        }
        throw DownloadError.failedToDownload(
            "\(modelId) after \(maxAttempts) attempt\(maxAttempts == 1 ? "" : "s") "
                + "(target: \(directory.path)): "
                + (lastError?.localizedDescription ?? "unknown"))
    }

    /// Whether `error` looks like the network being unreachable rather than the
    /// server rejecting the request.
    ///
    /// A stall counts: bytes stopped flowing, which is a transport problem, not
    /// a statement about the repository.
    static func isLikelyNetworkFailure(_ error: Error) -> Bool {
        if let downloadError = error as? DownloadError {
            if case .stalled = downloadError { return true }
            if case .networkUnavailable = downloadError { return true }
            return false
        }
        guard let urlError = error as? URLError else { return false }
        switch urlError.code {
        case .notConnectedToInternet, .networkConnectionLost, .timedOut,
             .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed,
             .internationalRoamingOff, .dataNotAllowed, .secureConnectionFailed,
             .resourceUnavailable:
            return true
        default:
            return false
        }
    }

    // MARK: - Background transfer

    private static let backgroundTransferLock = NSLock()
    nonisolated(unsafe) private static var storedBackgroundTransfer: BackgroundTransferConfiguration?

    /// Transfer model files through a background `URLSession` that keeps
    /// running while the process is suspended.
    ///
    /// `nil` — the default — keeps the in-process session every caller has
    /// today. Set it once, before the first download, from a host that can be
    /// suspended; see `BackgroundTransferConfiguration` for what it costs.
    public static var backgroundTransfer: BackgroundTransferConfiguration? {
        get {
            backgroundTransferLock.lock()
            defer { backgroundTransferLock.unlock() }
            return storedBackgroundTransfer
        }
        set {
            backgroundTransferLock.lock()
            storedBackgroundTransfer = newValue
            backgroundTransferLock.unlock()
        }
    }

    /// Hand back the completion handler the system supplies when it relaunches
    /// the process to deliver finished background transfers.
    ///
    /// The handler must be called, so an identifier this process knows nothing
    /// about is answered immediately rather than dropped.
    public static func handleBackgroundSessionEvents(
        identifier: String,
        completionHandler: @escaping @Sendable () -> Void
    ) {
        if let existing = BackgroundTransferCoordinator.existingCoordinator(for: identifier) {
            existing.addSessionFinishedHandler(completionHandler)
            return
        }
        guard let configuration = backgroundTransfer,
              configuration.sessionIdentifier == identifier
        else {
            completionHandler()
            return
        }
        BackgroundTransferCoordinator.coordinator(for: configuration)
            .addSessionFinishedHandler(completionHandler)
    }

    // MARK: - Retry ladder

    /// Delays between download attempts. One more attempt than entries:
    /// 5 attempts with 5/15/30/60 s pauses. A network that's down for a couple
    /// of minutes (AP roam, hotspot sleep, captive-portal re-auth) should not
    /// kill a multi-GB first-run download. Retries are cheaper than they were:
    /// a large file resumes from the last completed chunk rather than byte 0.
    static let downloadRetryDelaysSeconds = [5, 15, 30, 60]

    /// Total attempts per download (retries + the initial try).
    static var downloadMaxAttempts: Int { downloadRetryDelaysSeconds.count + 1 }

    // MARK: - Download stall guard

    /// Seconds of zero download progress after which an attempt is considered
    /// wedged and aborted.
    ///
    /// Progress is now reported per megabyte of real transferred bytes, so a
    /// healthy transfer keeps resetting the clock however slow it is, and only
    /// a genuinely stalled connection trips this. CI pins
    /// `HF_DOWNLOAD_STALL_TIMEOUT=90` to fail fast; the default is tuned for
    /// end users on flaky networks, who can't set environment variables.
    static var downloadStallTimeoutSeconds: Int {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_STALL_TIMEOUT"],
           let v = Int(raw), v > 0 {
            return v
        }
        return 300
    }

    /// Concurrent range requests used for large files. Bounded parallel ranges
    /// improve first-run download time on connections one stream under-fills,
    /// while avoiding unbounded request fan-out.
    static var downloadRangeConcurrency: Int {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_RANGE_CONCURRENCY"],
           let v = Int(raw), v > 0 {
            return min(v, 16)
        }
        return 16
    }

    /// Size at or above which a file is fetched as concurrent ranges rather
    /// than one request.
    ///
    /// Deliberately low. A single-request download reports nothing until it
    /// finishes, so a file just under the threshold on a slow link produces no
    /// progress at all and can trip the stall guard; and it cannot resume.
    /// Chunking anything past a few megabytes means resume and progress are the
    /// normal case rather than the large-file case. `HF_DOWNLOAD_RANGE_THRESHOLD`
    /// overrides it, which is also how tests drive the chunked path cheaply.
    static var rangedDownloadThresholdBytes: Int64 {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_RANGE_THRESHOLD"],
           let v = Int64(raw), v > 0 {
            return v
        }
        return 8 * 1_024 * 1_024
    }

    /// Bytes per range request.
    ///
    /// Each in-flight chunk is buffered whole, so this multiplied by
    /// `downloadRangeConcurrency` is the transient memory a large download
    /// costs — 128 MB at the defaults. It also sets how often progress is
    /// reported and how much is re-fetched after an interruption.
    /// `HF_DOWNLOAD_RANGE_CHUNK` overrides it.
    static var rangedDownloadChunkBytes: Int64 {
        if let raw = ProcessInfo.processInfo.environment["HF_DOWNLOAD_RANGE_CHUNK"],
           let v = Int64(raw), v > 0 {
            return v
        }
        return 8 * 1_024 * 1_024
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

    /// Thread-safe last-progress timestamp — progress may be reported from a
    /// background task.
    private final class ProgressClock: @unchecked Sendable {
        private let lock = NSLock()
        private var last = Date()
        func tick() { lock.lock(); last = Date(); lock.unlock() }
        func idleSeconds() -> Double {
            lock.lock(); defer { lock.unlock() }
            return Date().timeIntervalSince(last)
        }
    }

    /// Apply the stall guard only when this process is the one transferring.
    ///
    /// The guard measures wall-clock time since the last reported byte, which
    /// says nothing about a background session: its clock keeps running while
    /// the process is suspended, so a transfer the system was performing
    /// correctly the whole time would look wedged the moment the process came
    /// back. A background transfer is bounded by the session's own resource
    /// timeout instead, which the system enforces out of process.
    static func withDownloadStallGuardIfInProcess(
        modelId: String,
        _ operation: @escaping (@escaping @Sendable (Double) -> Void) async throws -> Void
    ) async throws {
        guard backgroundTransfer == nil else {
            try await operation { _ in }
            return
        }
        try await withDownloadStallGuard(modelId: modelId, operation)
    }

    /// Run a download `operation` that reports progress, and abort it if
    /// progress stalls for `downloadStallTimeoutSeconds`.
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
                // Poll on a fraction of the window so a stall is detected
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

    // MARK: - Security Helpers

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

    /// Validate that a remote file name is a safe single path component.
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

    /// Validate a repo-relative path that may name a file inside a directory.
    ///
    /// CoreML bundles are directories (`AudioEncoder.mlmodelc/coremldata.bin`),
    /// so the download path has to accept nested paths — the previous
    /// single-component rule is exactly why the ranged downloader could not
    /// serve CoreML repositories. Traversal is still refused: no absolute
    /// paths, no empty components, and no `.` or `..` anywhere.
    public static func validatedRelativePath(_ path: String) throws -> String {
        guard !path.isEmpty, !path.hasPrefix("/") else {
            throw DownloadError.invalidRemoteFileName(path)
        }
        let components = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !components.isEmpty else {
            throw DownloadError.invalidRemoteFileName(path)
        }
        for component in components {
            guard !component.isEmpty, component != ".", component != ".." else {
                throw DownloadError.invalidRemoteFileName(path)
            }
            guard component.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil else {
                throw DownloadError.invalidRemoteFileName(path)
            }
        }
        return components.joined(separator: "/")
    }

    /// Validate that a local path stays within the expected directory.
    public static func validatedLocalPath(directory: URL, fileName: String) throws -> URL {
        try validatedLocalPath(directory: directory, relativePath: fileName)
    }

    /// Resolve a repo-relative path under `directory`, refusing anything that
    /// escapes it. Belt and braces on top of `validatedRelativePath`.
    public static func validatedLocalPath(directory: URL, relativePath: String) throws -> URL {
        let safe = try validatedRelativePath(relativePath)
        let local = directory.appendingPathComponent(safe, isDirectory: false)
        let dirPath = directory.standardizedFileURL.path
        let localPath = local.standardizedFileURL.path
        let prefix = dirPath.hasSuffix("/") ? dirPath : (dirPath + "/")
        guard localPath.hasPrefix(prefix) else {
            throw DownloadError.invalidRemoteFileName(relativePath)
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
}
