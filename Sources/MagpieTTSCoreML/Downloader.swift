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
        public let decoderStepStatefulCompiled: URL
        public let nanocodecCompiled: URL
        public let nanocodecStreamingCompiled: URL
        /// `tokenizer/` directory from the MLX bundle (shared 2360-token vocab JSONs).
        public let mlxTokenizerDir: URL
    }

    private static let modelDirectories: [String] = [
        "text_encoder.mlmodelc",
        "decoder_prefill.mlmodelc",
        "decoder_step.mlmodelc",                 // fp32 IO, 48 cache inputs (cache-as-IO path)
        "decoder_step_stateful.mlmodelc",        // fp16 KV state (iOS 18+ MLState path)
        "nanocodec_decoder.mlmodelc",            // 64-frame batch codec
        "nanocodec_decoder_streaming.mlmodelc",  // 8-frame streaming codec
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

        // Our bundle doesn't ship a manifest.json (FluidInference's did,
        // ours doesn't), so the file list is just the inner contents of
        // each compiled `.mlmodelc/` directory.
        var allFiles: [String] = []
        for mdir in modelDirectories {
            allFiles.append("\(mdir)/coremldata.bin")
            allFiles.append("\(mdir)/model.mil")
            allFiles.append("\(mdir)/weights/weight.bin")
            allFiles.append("\(mdir)/analytics/coremldata.bin")
        }

        // Hot path: if every required file is already on disk, skip the
        // 20× HEAD-request round-trip the HF downloader otherwise makes
        // on every CLI invocation (measured 10+ s for this bundle on a
        // residential connection). Cold path is unchanged.
        let coreMLCached = filesPresent(in: coreMLDir, relativePaths: allFiles)
        if !coreMLCached {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: repoId,
                to: coreMLDir,
                additionalFiles: allFiles,
                offlineMode: false,
                progressHandler: progressHandler.map { p in
                    { progress in p(progress * 0.95) }
                })
        }

        let mlxRepoId = "aufklarer/Magpie-TTS-Multilingual-357M-MLX-4bit"
        let mlxDir = try HuggingFaceDownloader.getCacheDirectory(for: mlxRepoId)
        let mlxCached = filesPresent(in: mlxDir, relativePaths: mlxTokenizerFiles)
        if !mlxCached {
            try await HuggingFaceDownloader.downloadWeights(
                modelId: mlxRepoId,
                to: mlxDir,
                additionalFiles: mlxTokenizerFiles,
                offlineMode: false,
                progressHandler: progressHandler.map { p in
                    { progress in p(0.95 + progress * 0.05) }
                })
        }
        progressHandler?(1.0)

        return Paths(
            bundleRoot: coreMLDir,
            textEncoderCompiled: coreMLDir.appendingPathComponent("text_encoder.mlmodelc"),
            decoderPrefillCompiled: coreMLDir.appendingPathComponent("decoder_prefill.mlmodelc"),
            decoderStepCompiled: coreMLDir.appendingPathComponent("decoder_step.mlmodelc"),
            decoderStepStatefulCompiled: coreMLDir.appendingPathComponent("decoder_step_stateful.mlmodelc"),
            nanocodecCompiled: coreMLDir.appendingPathComponent("nanocodec_decoder.mlmodelc"),
            nanocodecStreamingCompiled: coreMLDir.appendingPathComponent("nanocodec_decoder_streaming.mlmodelc"),
            mlxTokenizerDir: mlxDir.appendingPathComponent("tokenizer"))
    }

    private static func filesPresent(in baseDir: URL, relativePaths: [String]) -> Bool {
        for relPath in relativePaths {
            let full = baseDir.appendingPathComponent(relPath)
            if !FileManager.default.fileExists(atPath: full.path) {
                return false
            }
        }
        return true
    }
}
