import Foundation
import AudioCommon

/// Downloads the single FlashSR bundle (model.safetensors + config.json).
public enum FlashSRDownloader {

    public struct Paths: Sendable {
        public let bundleDir: URL
    }

    public static func ensureDownloaded(
        variant: FlashSRVariant,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Paths {
        let repoId = variant.huggingFaceRepoId
        let dir = try HuggingFaceDownloader.getCacheDirectory(for: repoId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: repoId,
            to: dir,
            additionalFiles: ["model.safetensors", "config.json"],
            offlineMode: false,
            progressHandler: progressHandler
        )
        return Paths(bundleDir: dir)
    }
}
