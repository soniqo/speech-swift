import Foundation
import AudioCommon

/// Downloads the 4-bundle MagpieTTS layout (text_encoder / decoder_prefill /
/// decoder_step / nanocodec_decoder + tokenizer/) from HuggingFace.
public enum MagpieTTSDownloader {

    public struct Paths: Sendable {
        public let bundleRoot: URL
        public let textEncoderDir: URL
        public let decoderPrefillDir: URL
        public let decoderStepDir: URL
        public let nanocodecDir: URL
        public let tokenizerDir: URL
    }

    public static func ensureDownloaded(
        variant: MagpieTTSVariant = .int8,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> Paths {
        let repoId = variant.huggingFaceRepoId
        let dir = try HuggingFaceDownloader.getCacheDirectory(for: repoId)
        try await HuggingFaceDownloader.downloadWeights(
            modelId: repoId,
            to: dir,
            additionalFiles: [
                "manifest.json",
                "text_encoder/config.json",
                "text_encoder/model.safetensors",
                "decoder_prefill/config.json",
                "decoder_prefill/model.safetensors",
                "decoder_step/config.json",
                "decoder_step/model.safetensors",
                "nanocodec_decoder/config.json",
                "nanocodec_decoder/model.safetensors",
                "tokenizer/manifest.json",
                "tokenizer/en.json",
                "tokenizer/es.json",
                "tokenizer/de.json",
                "tokenizer/fr.json",
                "tokenizer/it.json",
                "tokenizer/vi.json",
                "tokenizer/zh.json",
                "tokenizer/hi.json",
            ],
            offlineMode: false,
            progressHandler: progressHandler)
        return Paths(
            bundleRoot: dir,
            textEncoderDir: dir.appendingPathComponent("text_encoder"),
            decoderPrefillDir: dir.appendingPathComponent("decoder_prefill"),
            decoderStepDir: dir.appendingPathComponent("decoder_step"),
            nanocodecDir: dir.appendingPathComponent("nanocodec_decoder"),
            tokenizerDir: dir.appendingPathComponent("tokenizer"))
    }
}
