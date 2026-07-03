import Foundation
import AudioCommon
import PersonaPlex
import Qwen3ASR
import SpeechVAD

enum PersonaPlexDemoCachePolicy {
    static func personaPlexOfflineModeAvailable(
        modelId: String = PersonaPlexModel.modelId8bit
    ) -> Bool {
        guard let directory = try? HuggingFaceDownloader.getCacheDirectory(for: modelId) else {
            return false
        }

        return cacheComplete(
            in: directory,
            requiredFiles: [
                "temporal.safetensors",
                "depformer.safetensors",
                "embeddings.safetensors",
                "mimi.safetensors",
                "tokenizer_spm_32k_3.model",
                "config.json",
            ] + PersonaPlexVoice.allCases.map { "voices/\($0.rawValue).safetensors" },
            requiresWeights: false)
    }

    static func asrOfflineModeAvailable(
        modelId: String = Qwen3ASRModel.defaultModelId
    ) -> Bool {
        guard let directory = try? HuggingFaceDownloader.getCacheDirectory(for: modelId) else {
            return false
        }

        return cacheComplete(
            in: directory,
            requiredFiles: [
                "config.json",
                "model.safetensors",
                "vocab.json",
                "merges.txt",
                "tokenizer_config.json",
            ],
            requiresWeights: true)
    }

    static func vadOfflineModeAvailable(
        modelId: String = SileroVADModel.defaultModelId
    ) -> Bool {
        guard let directory = try? HuggingFaceDownloader.getCacheDirectory(for: modelId) else {
            return false
        }

        return cacheComplete(
            in: directory,
            requiredFiles: ["config.json", "model.safetensors"],
            requiresWeights: true)
    }

    static func cacheComplete(
        in directory: URL,
        requiredFiles: [String],
        requiresWeights: Bool
    ) -> Bool {
        let fm = FileManager.default

        guard fm.fileExists(atPath: directory.path) else {
            return false
        }

        if requiresWeights && !HuggingFaceDownloader.weightsExist(in: directory) {
            return false
        }

        for file in requiredFiles {
            let url = directory.appendingPathComponent(file)
            guard fm.fileExists(atPath: url.path) else {
                return false
            }
        }

        return true
    }
}
