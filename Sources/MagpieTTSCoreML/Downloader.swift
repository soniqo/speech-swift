import Foundation
import AudioCommon

/// Downloads the soniqo CoreML Magpie bundle (4 mlmodelc directories + a
/// small manifest from HuggingFace). Tokenizer JSONs come from the MLX
/// bundle and are fetched separately — they're shared with the MLX backend
/// and the cache short-circuits if the user already has them.
public enum MagpieCoreMLDownloader {

    public struct Paths: Sendable {
        public let bundleRoot: URL
        public let textEncoderCompiled: URL
        public let decoderPrefillCompiled: URL
        public let decoderStepCompiled: URL
        public let nanocodecCompiled: URL
        /// `tokenizer/` directory from the MLX bundle (shared 2360-token vocab JSONs).
        public let mlxTokenizerDir: URL
    }

    private static let modelDirectories: [String] = [
        "text_encoder.mlmodelc",
        "decoder_prefill.mlmodelc",
        "decoder_step.mlmodelc",
        "nanocodec_decoder.mlmodelc",
    ]

    /// MLX-bundle files we need for tokenization (the CoreML bundle ships
    /// no tokenizer assets — see README on the HuggingFace repo).
    private static let mlxTokenizerFiles: [String] = [
        "tokenizer/manifest.json",
        "tokenizer/en.json",
        "tokenizer/es.json",
        "tokenizer/de.json",
        "tokenizer/fr.json",
        "tokenizer/it.json",
        "tokenizer/vi.json",
        "tokenizer/zh.json",
        "tokenizer/hi.json",
    ]

    public static func ensureDownloaded(
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Paths {
        let repoId = MagpieCoreMLConstants.huggingFaceRepo
        let coreMLDir = try HuggingFaceDownloader.getCacheDirectory(for: repoId)

        var allFiles: [String] = ["manifest.json"]
        for mdir in modelDirectories {
            allFiles.append("\(mdir)/coremldata.bin")
            allFiles.append("\(mdir)/model.mil")
            allFiles.append("\(mdir)/weights/weight.bin")
            allFiles.append("\(mdir)/analytics/coremldata.bin")
        }

        try await HuggingFaceDownloader.downloadWeights(
            modelId: repoId,
            to: coreMLDir,
            additionalFiles: allFiles,
            offlineMode: false,
            progressHandler: progressHandler.map { p in
                { progress in p(progress * 0.95) }
            })

        let mlxRepoId = "aufklarer/Magpie-TTS-Multilingual-357M-MLX-4bit"
        let mlxDir = try HuggingFaceDownloader.getCacheDirectory(for: mlxRepoId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: mlxRepoId,
            to: mlxDir,
            additionalFiles: mlxTokenizerFiles,
            offlineMode: false,
            progressHandler: progressHandler.map { p in
                { progress in p(0.95 + progress * 0.05) }
            })

        return Paths(
            bundleRoot: coreMLDir,
            textEncoderCompiled: coreMLDir.appendingPathComponent("text_encoder.mlmodelc"),
            decoderPrefillCompiled: coreMLDir.appendingPathComponent("decoder_prefill.mlmodelc"),
            decoderStepCompiled: coreMLDir.appendingPathComponent("decoder_step.mlmodelc"),
            nanocodecCompiled: coreMLDir.appendingPathComponent("nanocodec_decoder.mlmodelc"),
            mlxTokenizerDir: mlxDir.appendingPathComponent("tokenizer"))
    }
}
