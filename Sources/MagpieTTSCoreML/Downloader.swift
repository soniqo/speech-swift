import Foundation
import AudioCommon

/// Downloads the FluidInference CoreML Magpie bundle (mlmodelc + constants/
/// npy + tokenizer/) from HuggingFace. Also lazily downloads the MLX tokenizer
/// JSONs we reuse for G2P (a small ~10 MB subset of the MLX bundle).
public enum MagpieCoreMLDownloader {

    public struct Paths: Sendable {
        public let bundleRoot: URL
        public let textEncoderCompiled: URL
        public let decoderPrefillCompiled: URL
        public let decoderStepCompiled: URL
        public let nanocodecCompiled: URL
        public let constantsDir: URL
        public let localTransformerDir: URL
        public let tokenizerDir: URL  // FluidInference's own tokenizer/ (currently unused — we reuse MLX-bundle tokenizers via mlxTokenizerDir)
        public let mlxTokenizerDir: URL  // tokenizer/ from the MLX bundle (reused for our MagpieTokenizer)
    }

    /// File list pulled from the CoreML repo. Includes the 4 compiled
    /// `.mlmodelc/` directories, constants, and the FluidInference tokenizer/
    /// directory (we don't use those JSONs at runtime, but they're small and
    /// keep the bundle self-contained for offline inspection).
    private static let coreMLFiles: [String] = [
        "manifest.json",
        // Constants
        "constants/constants.json",
        "constants/speaker_info.json",
        "constants/tokenizer_metadata.json",
        "constants/text_embedding.npy",
        "constants/speaker_embeddings_raw.npy",
        "constants/speaker_0.npy",
        "constants/speaker_1.npy",
        "constants/speaker_2.npy",
        "constants/speaker_3.npy",
        "constants/speaker_4.npy",
        "constants/audio_embedding_0.npy",
        "constants/audio_embedding_1.npy",
        "constants/audio_embedding_2.npy",
        "constants/audio_embedding_3.npy",
        "constants/audio_embedding_4.npy",
        "constants/audio_embedding_5.npy",
        "constants/audio_embedding_6.npy",
        "constants/audio_embedding_7.npy",
        // Local transformer weights (Swift-side, runs outside CoreML)
        "constants/local_transformer/in_proj_weight.npy",
        "constants/local_transformer/in_proj_bias.npy",
        "constants/local_transformer/pos_emb.npy",
        "constants/local_transformer/norm1_weight.npy",
        "constants/local_transformer/norm2_weight.npy",
        "constants/local_transformer/sa_qkv_weight.npy",
        "constants/local_transformer/sa_o_weight.npy",
        "constants/local_transformer/ffn_conv1_weight.npy",
        "constants/local_transformer/ffn_conv2_weight.npy",
        "constants/local_transformer/out_proj_0_weight.npy",
        "constants/local_transformer/out_proj_0_bias.npy",
        "constants/local_transformer/out_proj_1_weight.npy",
        "constants/local_transformer/out_proj_1_bias.npy",
        "constants/local_transformer/out_proj_2_weight.npy",
        "constants/local_transformer/out_proj_2_bias.npy",
        "constants/local_transformer/out_proj_3_weight.npy",
        "constants/local_transformer/out_proj_3_bias.npy",
        "constants/local_transformer/out_proj_4_weight.npy",
        "constants/local_transformer/out_proj_4_bias.npy",
        "constants/local_transformer/out_proj_5_weight.npy",
        "constants/local_transformer/out_proj_5_bias.npy",
        "constants/local_transformer/out_proj_6_weight.npy",
        "constants/local_transformer/out_proj_6_bias.npy",
        "constants/local_transformer/out_proj_7_weight.npy",
        "constants/local_transformer/out_proj_7_bias.npy",
    ]

    /// The 4 compiled CoreML model directories. Each ships as a folder with
    /// `coremldata.bin` + `metadata.json` + `model.mil` + `weights/` — we
    /// download all the inner files via the standard mlmodelc patterns
    /// embedded in HuggingFaceDownloader.
    private static let modelDirectories: [String] = [
        "text_encoder.mlmodelc",
        "decoder_prefill.mlmodelc",
        "decoder_step.mlmodelc",
        "nanocodec_decoder.mlmodelc",
    ]

    /// MLX-bundle files we need for tokenization (the CoreML bundle ships its
    /// own per-language token2id JSONs, but we reuse our existing
    /// MagpieTokenizer + G2P infra which expects the original NeMo-style
    /// vocab JSON format).
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

        var allFiles = coreMLFiles
        // Add every file inside each mlmodelc/ directory (they ship as
        // multi-file packages, so the HF API serves a flat set of paths).
        for mdir in modelDirectories {
            allFiles.append("\(mdir)/coremldata.bin")
            allFiles.append("\(mdir)/metadata.json")
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
                { progress in p(progress * 0.95) }  // leave 5% for tokenizer download
            })

        // MLX tokenizer JSONs (small — ~10 MB total). Cached separately so a
        // user who already installed the MLX bundle gets a no-op here.
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
            constantsDir: coreMLDir.appendingPathComponent("constants"),
            localTransformerDir: coreMLDir
                .appendingPathComponent("constants")
                .appendingPathComponent("local_transformer"),
            tokenizerDir: coreMLDir.appendingPathComponent("tokenizer"),
            mlxTokenizerDir: mlxDir.appendingPathComponent("tokenizer"))
    }
}
