import Foundation
import AudioCommon

public final class Audio2Face3DModel: @unchecked Sendable {
    public let backend: Audio2Face3DBackend
    public let configuration: Audio2Face3DConfiguration

    private let mlxRuntime: Audio2Face3DMLXRuntime?

    private init(
        backend: Audio2Face3DBackend,
        configuration: Audio2Face3DConfiguration,
        mlxRuntime: Audio2Face3DMLXRuntime?
    ) {
        self.backend = backend
        self.configuration = configuration
        self.mlxRuntime = mlxRuntime
    }

    /// Load a model for avatar motion generation.
    ///
    /// This loads the exported NVIDIA Audio2Face-3D tensors and runs the
    /// hand-written MLX graph. No heuristic/fallback avatar motion is exposed
    /// from this API.
    public static func fromLocal(
        directory: URL,
        backend: Audio2Face3DBackend = .mlx,
        configuration: Audio2Face3DConfiguration = Audio2Face3DConfiguration()
    ) throws -> Audio2Face3DModel {
        let resolvedConfiguration = try Audio2Face3DDownloader.configuration(
            from: directory,
            fallback: configuration)
        switch backend {
        case .mlx:
            let weights = directory.appendingPathComponent("audio2face3d.safetensors")
            guard FileManager.default.fileExists(atPath: weights.path) else {
                throw Audio2Face3DError.missingExportedWeights(weights.path)
            }
            return Audio2Face3DModel(
                backend: backend,
                configuration: resolvedConfiguration,
                mlxRuntime: try Audio2Face3DMLXRuntime(
                    directory: directory,
                    configuration: resolvedConfiguration))
        }
    }

    /// Download the exported MLX bundle from Hugging Face into the normal
    /// speech cache and load it.
    public static func fromPretrained(
        modelId: String = Audio2Face3DConfiguration.defaultModelId,
        backend: Audio2Face3DBackend = .mlx,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> Audio2Face3DModel {
        var configuration = Audio2Face3DConfiguration(modelId: modelId)
        let dir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        progressHandler?(0.0, "Downloading Audio2Face-3D...")
        try await Audio2Face3DDownloader.ensureDownloaded(
            modelId: modelId,
            to: dir,
            offlineMode: offlineMode
        ) { progress, _, _, file in
            progressHandler?(progress * 0.90, file.isEmpty ? "Downloading Audio2Face-3D..." : "Downloading \(file)...")
        }
        configuration = try Audio2Face3DDownloader.configuration(from: dir, fallback: configuration)
        progressHandler?(0.95, "Loading Audio2Face-3D...")
        return try fromLocal(directory: dir, backend: backend, configuration: configuration)
    }

    public func frames(
        for audio: [Float],
        sampleRate: Int,
        hopLength: Int? = nil,
        emotion: [Float]? = nil
    ) throws -> [Audio2Face3DFrame] {
        guard sampleRate > 0 else { throw Audio2Face3DError.invalidSampleRate(sampleRate) }
        switch backend {
        case .mlx:
            guard let mlxRuntime else {
                throw Audio2Face3DError.unsupportedBackend(backend)
            }
            return try mlxRuntime.frames(
                for: audio,
                sampleRate: sampleRate,
                hopLength: hopLength,
                emotion: emotion)
        }
    }

    public func emotionVector(explicit: [Float]) throws -> [Float] {
        switch backend {
        case .mlx:
            guard let mlxRuntime else {
                throw Audio2Face3DError.unsupportedBackend(backend)
            }
            return try mlxRuntime.emotionVector(explicit: explicit)
        }
    }
}
