import Foundation
import AudioCommon

/// Downloads the three repos MAGNeT needs (bundle + T5 + EnCodec).
public enum MAGNeTDownloader {

    public struct BundlePaths: Sendable {
        public let bundleDir: URL
        public let t5Dir: URL
        public let encodecDir: URL
    }

    /// Ensure all three repos are cached locally. Returns the directories.
    public static func ensureDownloaded(
        variant: MAGNeTVariant,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> BundlePaths {
        let bundleId = variant.huggingFaceRepoId
        let bundleDir = try HuggingFaceDownloader.getCacheDirectory(for: bundleId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: bundleId,
            to: bundleDir,
            additionalFiles: ["model.safetensors", "config.json"],
            offlineMode: false,
            progressHandler: { progressHandler?($0 * 0.4) }
        )

        let t5Dir = try HuggingFaceDownloader.getCacheDirectory(for: "t5-base")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: "t5-base",
            to: t5Dir,
            additionalFiles: [
                "model.safetensors", "config.json",
                "spiece.model", "tokenizer.json", "tokenizer_config.json"
            ],
            offlineMode: false,
            progressHandler: { progressHandler?(0.4 + $0 * 0.3) }
        )

        let encId = "mlx-community/encodec-32khz-float32"
        let encodecDir = try HuggingFaceDownloader.getCacheDirectory(for: encId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: encId,
            to: encodecDir,
            additionalFiles: ["model.safetensors", "config.json"],
            offlineMode: false,
            progressHandler: { progressHandler?(0.7 + $0 * 0.3) }
        )

        return BundlePaths(bundleDir: bundleDir, t5Dir: t5Dir, encodecDir: encodecDir)
    }
}
