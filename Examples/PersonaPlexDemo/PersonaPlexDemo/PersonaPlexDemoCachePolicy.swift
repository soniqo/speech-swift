import Foundation
import AudioCommon
import PersonaPlex
import Qwen3ASR
import SpeechVAD

struct CacheDirectoryRequirement {
    var relativePath: String
    var fileExtension: String
    var minimumCount: Int
}

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
            ],
            requiresWeights: false,
            directoryRequirements: [
                CacheDirectoryRequirement(
                    relativePath: "voices",
                    fileExtension: "safetensors",
                    minimumCount: PersonaPlexVoice.allCases.count)
            ])
    }

    static func asrOfflineModeAvailable(
        modelId: String = Qwen3ASRModel.defaultModelId
    ) -> Bool {
        guard let directory = try? HuggingFaceDownloader.getCacheDirectory(for: modelId) else {
            return false
        }

        return cacheComplete(
            in: directory,
            requiredFiles: ["vocab.json", "merges.txt", "tokenizer_config.json"],
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
            requiredFiles: ["config.json"],
            requiresWeights: true)
    }

    static func cacheComplete(
        in directory: URL,
        requiredFiles: [String],
        requiresWeights: Bool,
        directoryRequirements: [CacheDirectoryRequirement] = []
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

        for requirement in directoryRequirements {
            let dir = directory.appendingPathComponent(requirement.relativePath, isDirectory: true)
            let entries: [URL]
            do {
                entries = try fm.contentsOfDirectory(
                    at: dir,
                    includingPropertiesForKeys: nil,
                    options: [.skipsHiddenFiles])
            } catch {
                return false
            }

            let count = entries.filter { $0.pathExtension == requirement.fileExtension }.count
            guard count >= requirement.minimumCount else {
                return false
            }
        }

        return true
    }
}
