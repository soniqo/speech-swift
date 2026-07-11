import Foundation
import MLX

public struct ChatterboxMemoryOptions: Sendable, Equatable {
    /// Temporary MLX cache limit while running Chatterbox generation.
    /// `nil` preserves the caller's existing global cache limit.
    public var cacheLimitBytes: Int?

    /// Clear reusable MLX buffers between the autoregressive T3 stage, S3Gen
    /// flow stage, and vocoder stage. Active tensors are retained.
    public var clearCacheBetweenStages: Bool

    /// Clear reusable MLX buffers before returning generated audio.
    public var clearCacheOnCompletion: Bool

    public init(
        cacheLimitBytes: Int? = 512 * 1024 * 1024,
        clearCacheBetweenStages: Bool = true,
        clearCacheOnCompletion: Bool = true
    ) {
        self.cacheLimitBytes = cacheLimitBytes
        self.clearCacheBetweenStages = clearCacheBetweenStages
        self.clearCacheOnCompletion = clearCacheOnCompletion
    }

    /// Memory-conscious default for app/runtime use.
    public static let balanced = ChatterboxMemoryOptions()

    /// Preserve MLX's default cache behavior for maximum-throughput experiments.
    public static let unrestricted = ChatterboxMemoryOptions(
        cacheLimitBytes: nil,
        clearCacheBetweenStages: false,
        clearCacheOnCompletion: false
    )
}

enum ChatterboxMemory {
    private static let lock = NSRecursiveLock()

    static func withOptions<T>(_ options: ChatterboxMemoryOptions, _ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }

        let previousCacheLimit = options.cacheLimitBytes == nil ? nil : Memory.cacheLimit
        if let cacheLimitBytes = options.cacheLimitBytes {
            Memory.cacheLimit = min(Memory.cacheLimit, max(0, cacheLimitBytes))
            Memory.clearCache()
        }
        defer {
            if let previousCacheLimit {
                Memory.cacheLimit = previousCacheLimit
            }
            if options.clearCacheOnCompletion {
                Memory.clearCache()
            }
        }
        return try body()
    }

    static func clearStageCache(_ options: ChatterboxMemoryOptions) {
        if options.clearCacheBetweenStages {
            Memory.clearCache()
        }
    }
}
