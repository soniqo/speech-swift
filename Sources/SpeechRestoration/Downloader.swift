import Foundation
import AudioCommon

/// Downloads the Sidon CoreML bundles (predictor + vocoder) for a variant.
///
/// Mirrors the other CoreML downloaders: it fetches both the compiled
/// `.mlmodelc/**` (preferred) and the `.mlpackage/**` (provisional / fallback)
/// trees so the engine works regardless of which layout the published repo
/// ships. The Hub glob skips files that don't exist, so requesting both is safe.
public enum SidonDownloader {

    public struct Paths: Sendable {
        /// Directory that contains the variant's model bundles.
        public let bundleDir: URL
    }

    /// Ensure the variant's models are present locally, downloading on first use.
    ///
    /// - Parameters:
    ///   - modelId: HuggingFace repo id (provisional: `aufklarer/Sidon-CoreML`).
    ///   - variant: precision variant (selects the in-repo subfolder).
    public static func ensureDownloaded(
        modelId: String,
        variant: SidonVariant,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Paths {
        let repoDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let sub = variant.subfolder

        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: repoDir,
            additionalFiles: [
                // Suppress the default *.safetensors glob — this is a CoreML repo.
                "no_weights.safetensors",
                // Preferred precompiled bundles (per variant subfolder).
                "\(sub)/\(SidonConfig.predictorCompiledName)/**",
                "\(sub)/\(SidonConfig.vocoderCompiledName)/**",
                // Provisional .mlpackage bundles (compiled on-device if the
                // .mlmodelc isn't published yet).
                "\(sub)/\(SidonConfig.predictorPackageName)/**",
                "\(sub)/\(SidonConfig.vocoderPackageName)/**",
            ],
            offlineMode: offlineMode,
            progressHandler: progressHandler
        )

        return Paths(bundleDir: repoDir.appendingPathComponent(sub, isDirectory: true))
    }
}
