import AudioCommon
import Foundation

public struct VoiceCloneBundleFile: Codable, Equatable, Sendable {
    public let path: String
    public let bytes: Int64

    public init(path: String, bytes: Int64) {
        self.path = path
        self.bytes = bytes
    }
}

public struct VoiceCloneAuxiliaryManifest: Codable, Equatable, Sendable {
    public let key: String
    public let displayName: String
    public let sourceRepo: String
    public let sourceRevision: String?
    public let purpose: String
    public let convertedFiles: [String]
    public let copiedFiles: [String]
    public let notes: [String]

    enum CodingKeys: String, CodingKey {
        case key
        case displayName = "display_name"
        case sourceRepo = "source_repo"
        case sourceRevision = "source_revision"
        case purpose
        case convertedFiles = "converted_files"
        case copiedFiles = "copied_files"
        case notes
    }
}

public struct VoiceCloneExportManifest: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    public let artifactType: String
    public let modelKey: String
    public let displayName: String
    public let sourceRepo: String
    public let sourceRevision: String?
    public let publishRepo: String?
    public let outputName: String
    public let license: String?
    public let licensePosture: String?
    public let parameterCount: String?
    public let sampleRateHz: Int?
    public let languages: [String]
    public let voiceConditioning: String?
    public let streaming: String?
    public let format: String?
    public let runtimeStatus: String?
    public let convertedFiles: [String]
    public let copiedFiles: [String]
    public let auxiliaryModels: [VoiceCloneAuxiliaryManifest]
    public let notes: [String]
    public let files: [VoiceCloneBundleFile]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case artifactType = "artifact_type"
        case modelKey = "model_key"
        case displayName = "display_name"
        case sourceRepo = "source_repo"
        case sourceRevision = "source_revision"
        case publishRepo = "publish_repo"
        case outputName = "output_name"
        case license
        case licensePosture = "license_posture"
        case parameterCount = "parameter_count"
        case sampleRateHz = "sample_rate_hz"
        case languages
        case voiceConditioning = "voice_conditioning"
        case streaming
        case format
        case runtimeStatus = "runtime_status"
        case convertedFiles = "converted_files"
        case copiedFiles = "copied_files"
        case auxiliaryModels = "auxiliary_models"
        case notes
        case files
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try container.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        artifactType = try container.decodeIfPresent(String.self, forKey: .artifactType) ?? "voice_cloning_tts_candidate"
        modelKey = try container.decode(String.self, forKey: .modelKey)
        displayName = try container.decodeIfPresent(String.self, forKey: .displayName) ?? modelKey
        sourceRepo = try container.decodeIfPresent(String.self, forKey: .sourceRepo) ?? ""
        sourceRevision = try container.decodeIfPresent(String.self, forKey: .sourceRevision)
        publishRepo = try container.decodeIfPresent(String.self, forKey: .publishRepo)
        outputName = try container.decodeIfPresent(String.self, forKey: .outputName) ?? displayName
        license = try container.decodeIfPresent(String.self, forKey: .license)
        licensePosture = try container.decodeIfPresent(String.self, forKey: .licensePosture)
        parameterCount = try container.decodeIfPresent(String.self, forKey: .parameterCount)
        sampleRateHz = try container.decodeIfPresent(Int.self, forKey: .sampleRateHz)
        languages = try container.decodeIfPresent([String].self, forKey: .languages) ?? []
        voiceConditioning = try container.decodeIfPresent(String.self, forKey: .voiceConditioning)
        streaming = try container.decodeIfPresent(String.self, forKey: .streaming)
        format = try container.decodeIfPresent(String.self, forKey: .format)
        runtimeStatus = try container.decodeIfPresent(String.self, forKey: .runtimeStatus)
        convertedFiles = try container.decodeIfPresent([String].self, forKey: .convertedFiles) ?? []
        copiedFiles = try container.decodeIfPresent([String].self, forKey: .copiedFiles) ?? []
        auxiliaryModels = try container.decodeIfPresent(
            [VoiceCloneAuxiliaryManifest].self,
            forKey: .auxiliaryModels) ?? []
        notes = try container.decodeIfPresent([String].self, forKey: .notes) ?? []
        files = try container.decodeIfPresent([VoiceCloneBundleFile].self, forKey: .files) ?? []
    }
}

public struct VoiceCloneBundleInfo: Equatable, Sendable {
    public let directory: URL
    public let manifest: VoiceCloneExportManifest
    public let weightMemory: Int

    public var sampleRate: Int {
        manifest.sampleRateHz ?? 24_000
    }

    public init(directory: URL, manifest: VoiceCloneExportManifest, weightMemory: Int) {
        self.directory = directory
        self.manifest = manifest
        self.weightMemory = weightMemory
    }
}

public enum VoiceCloneBundleError: Error, LocalizedError, Equatable {
    case missingManifest(URL)
    case invalidManifest(URL, String)
    case unexpectedModelKey(expected: String, actual: String)
    case missingRequiredFile(String)
    case emptyWeightSet(URL)

    public var errorDescription: String? {
        switch self {
        case .missingManifest(let url):
            return "Missing voice-cloning bundle manifest: \(url.path)"
        case .invalidManifest(let url, let reason):
            return "Invalid voice-cloning bundle manifest at \(url.path): \(reason)"
        case .unexpectedModelKey(let expected, let actual):
            return "Unexpected voice-cloning model key '\(actual)' (expected '\(expected)')"
        case .missingRequiredFile(let path):
            return "Voice-cloning bundle is missing required file: \(path)"
        case .emptyWeightSet(let url):
            return "Voice-cloning bundle has no converted safetensors weights: \(url.path)"
        }
    }
}

public enum VoiceCloneBundleLoader {
    public static let manifestFileName = "soniqo_manifest.json"

    public static func download(
        modelId: String,
        requiredFiles: [String],
        cacheDir: URL?,
        offlineMode: Bool,
        progressHandler: ((Double, String) -> Void)?
    ) async throws -> URL {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(
            modelId: modelId,
            to: directory,
            files: requiredFiles,
            offlineMode: offlineMode
        ) { progress in
            progressHandler?(progress * 0.85, "Downloading \(modelId)")
        }
        return directory
    }

    public static func load(
        from directory: URL,
        expectedModelKey: String,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> VoiceCloneBundleInfo {
        progressHandler?(0.05, "Loading voice-cloning manifest")
        let manifestURL = directory.appendingPathComponent(manifestFileName)
        let manifest = try loadManifest(from: manifestURL)

        guard manifest.modelKey == expectedModelKey else {
            throw VoiceCloneBundleError.unexpectedModelKey(
                expected: expectedModelKey,
                actual: manifest.modelKey)
        }

        progressHandler?(0.35, "Validating voice-cloning bundle")
        for path in manifest.convertedFiles + manifest.copiedFiles {
            let fileURL = directory.appendingPathComponent(path)
            guard FileManager.default.fileExists(atPath: fileURL.path) else {
                throw VoiceCloneBundleError.missingRequiredFile(path)
            }
        }

        let weightFiles = manifest.convertedFiles.filter { $0.hasSuffix(".safetensors") }
        guard !weightFiles.isEmpty else {
            throw VoiceCloneBundleError.emptyWeightSet(directory)
        }

        let memory = weightFiles.reduce(0) { total, path in
            let fileURL = directory.appendingPathComponent(path)
            let size = ((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            return total + size
        }

        progressHandler?(1.0, "Voice-cloning bundle ready")
        return VoiceCloneBundleInfo(directory: directory, manifest: manifest, weightMemory: memory)
    }

    public static func loadManifest(from url: URL) throws -> VoiceCloneExportManifest {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw VoiceCloneBundleError.missingManifest(url)
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(VoiceCloneExportManifest.self, from: data)
        } catch let error as VoiceCloneBundleError {
            throw error
        } catch {
            throw VoiceCloneBundleError.invalidManifest(url, error.localizedDescription)
        }
    }

    public static func runtimeNotImplemented(modelName: String) -> AudioModelError {
        AudioModelError.inferenceFailed(
            operation: "\(modelName) synthesis",
            reason: "The bundle loader is implemented, but native Swift inference for this model has not been ported yet.")
    }
}
