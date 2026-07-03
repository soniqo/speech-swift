import Foundation
import AudioCommon
import PersonaPlex
import Qwen3ASR
import SpeechVAD

struct CacheFileRequirement {
    var relativePath: String
    var byteCount: UInt64
}

enum PersonaPlexDemoCachePolicy {
    static func personaPlexOfflineModeAvailable(
        modelId: String = PersonaPlexModel.modelId8bit
    ) -> Bool {
        guard let directory = try? HuggingFaceDownloader.getCacheDirectory(for: modelId) else {
            return false
        }

        guard personaPlexVoiceByteCounts.count == PersonaPlexVoice.allCases.count else {
            return false
        }

        return cacheComplete(
            in: directory,
            requiredFiles: personaPlexFileRequirements + personaPlexVoiceRequirements,
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
            requiredFiles: asrFileRequirements,
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
            requiredFiles: vadFileRequirements,
            requiresWeights: true)
    }

    static func cacheComplete(
        in directory: URL,
        requiredFiles: [CacheFileRequirement],
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
            let url = directory.appendingPathComponent(file.relativePath)
            guard fm.fileExists(atPath: url.path) else {
                return false
            }

            guard byteCount(of: url) == file.byteCount else {
                return false
            }
        }

        return true
    }

    private static let personaPlexFileRequirements: [CacheFileRequirement] = [
        CacheFileRequirement(relativePath: "config.json", byteCount: 1_462),
        CacheFileRequirement(relativePath: "depformer.safetensors", byteCount: 1_381_596_968),
        CacheFileRequirement(relativePath: "embeddings.safetensors", byteCount: 988_459_536),
        CacheFileRequirement(relativePath: "mimi.safetensors", byteCount: 384_644_900),
        CacheFileRequirement(relativePath: "temporal.safetensors", byteCount: 6_988_291_888),
        CacheFileRequirement(relativePath: "tokenizer_spm_32k_3.model", byteCount: 552_778),
    ]

    private static let personaPlexVoiceByteCounts: [PersonaPlexVoice: UInt64] = [
        .NATF0: 418_216,
        .NATF1: 393_640,
        .NATF2: 418_216,
        .NATF3: 418_216,
        .NATM0: 410_024,
        .NATM1: 418_216,
        .NATM2: 410_024,
        .NATM3: 377_256,
        .VARF0: 557_480,
        .VARF1: 410_024,
        .VARF2: 418_216,
        .VARF3: 508_328,
        .VARF4: 467_368,
        .VARM0: 352_680,
        .VARM1: 377_256,
        .VARM2: 565_672,
        .VARM3: 377_256,
        .VARM4: 450_984,
    ]

    private static var personaPlexVoiceRequirements: [CacheFileRequirement] {
        PersonaPlexVoice.allCases.compactMap { voice in
            guard let byteCount = personaPlexVoiceByteCounts[voice] else {
                return nil
            }
            return CacheFileRequirement(
                relativePath: "voices/\(voice.rawValue).safetensors",
                byteCount: byteCount)
        }
    }

    private static let asrFileRequirements: [CacheFileRequirement] = [
        CacheFileRequirement(relativePath: "config.json", byteCount: 7_187),
        CacheFileRequirement(relativePath: "merges.txt", byteCount: 1_671_853),
        CacheFileRequirement(relativePath: "model.safetensors", byteCount: 708_236_945),
        CacheFileRequirement(relativePath: "tokenizer_config.json", byteCount: 12_487),
        CacheFileRequirement(relativePath: "vocab.json", byteCount: 2_776_833),
    ]

    private static let vadFileRequirements: [CacheFileRequirement] = [
        CacheFileRequirement(relativePath: "config.json", byteCount: 456),
        CacheFileRequirement(relativePath: "model.safetensors", byteCount: 1_237_580),
    ]

    private static func byteCount(of url: URL) -> UInt64? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else {
            return nil
        }

        switch attributes[.size] {
        case let value as NSNumber:
            return value.uint64Value
        case let value as UInt64:
            return value
        case let value as Int where value >= 0:
            return UInt64(value)
        default:
            return nil
        }
    }
}
