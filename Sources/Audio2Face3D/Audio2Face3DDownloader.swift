import Foundation
import AudioCommon

public enum Audio2Face3DDownloader {
    public static let requiredFiles: [String] = [
        "audio2face3d.safetensors",
        "network_info.json",
        "model_config.json",
        "default_emotion.f32",
    ]

    /// Exact byte sizes of the published bundles, keyed by model id. Passing
    /// sizes skips the per-file HEAD resolution, which fails on small non-LFS
    /// files when the CDN omits Content-Length. Unknown model ids fall back
    /// to HEAD resolution.
    public static let expectedSizesByModelId: [String: [String: Int64]] = [
        Audio2Face3DConfiguration.markModelId: [
            "audio2face3d.safetensors": 73_099_520,
            "network_info.json": 1_015,
            "model_config.json": 984,
            "default_emotion.f32": 104,
        ],
        Audio2Face3DConfiguration.claireModelId: [
            "audio2face3d.safetensors": 157_778_224,
            "network_info.json": 1_017,
            "model_config.json": 980,
            "default_emotion.f32": 104,
        ],
        Audio2Face3DConfiguration.jamesModelId: [
            "audio2face3d.safetensors": 157_778_224,
            "network_info.json": 1_016,
            "model_config.json": 979,
            "default_emotion.f32": 104,
        ],
    ]

    public static func ensureDownloaded(
        modelId: String = Audio2Face3DConfiguration.defaultModelId,
        to directory: URL,
        offlineMode: Bool = false,
        progressHandler: ((Double, Int64, Int64, String) -> Void)? = nil
    ) async throws {
        try await HuggingFaceDownloader.downloadFilesByteWeighted(
            modelId: modelId,
            to: directory,
            files: requiredFiles,
            expectedSizes: expectedSizesByModelId[modelId],
            offlineMode: offlineMode,
            progressHandler: progressHandler)
    }

    public static func configuration(
        from directory: URL,
        fallback: Audio2Face3DConfiguration = Audio2Face3DConfiguration()
    ) throws -> Audio2Face3DConfiguration {
        let url = directory.appendingPathComponent("network_info.json")
        guard FileManager.default.fileExists(atPath: url.path) else { return fallback }
        let data = try Data(contentsOf: url)
        let info = try JSONDecoder().decode(NetworkInfo.self, from: data)
        let modelConfig = try? ModelConfig.from(directory: directory)
        return Audio2Face3DConfiguration(
            modelId: fallback.modelId,
            inputSampleRate: info.audioParams.samplerate,
            bufferLength: info.audioParams.bufferLen,
            hopLength: info.audioParams.bufferOfs,
            framesPerSecond: fallback.framesPerSecond,
            implicitEmotionCount: info.params.implicitEmotionLen,
            explicitEmotionCount: info.params.explicitEmotions.count,
            coefficientLayout: Audio2Face3DCoefficientLayout(
                skinCount: info.params.numShapesSkin,
                tongueCount: info.params.numShapesTongue,
                jawCount: info.params.resultJawSize,
                eyeCount: info.params.resultEyesSize),
            inputStrength: modelConfig?.config.inputStrength ?? fallback.inputStrength
        )
    }
}

private struct ModelConfig: Decodable {
    let config: Config

    struct Config: Decodable {
        let inputStrength: Float

        enum CodingKeys: String, CodingKey {
            case inputStrength = "input_strength"
        }
    }

    static func from(directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("model_config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ModelConfig.self, from: data)
    }
}

private struct NetworkInfo: Decodable {
    let params: Params
    let audioParams: AudioParams

    enum CodingKeys: String, CodingKey {
        case params
        case audioParams = "audio_params"
    }

    struct Params: Decodable {
        let implicitEmotionLen: Int
        let explicitEmotions: [String]
        let numShapesSkin: Int
        let numShapesTongue: Int
        let resultJawSize: Int
        let resultEyesSize: Int

        enum CodingKeys: String, CodingKey {
            case implicitEmotionLen = "implicit_emotion_len"
            case explicitEmotions = "explicit_emotions"
            case numShapesSkin = "num_shapes_skin"
            case numShapesTongue = "num_shapes_tongue"
            case resultJawSize = "result_jaw_size"
            case resultEyesSize = "result_eyes_size"
        }
    }

    struct AudioParams: Decodable {
        let bufferLen: Int
        let bufferOfs: Int
        let samplerate: Int

        enum CodingKeys: String, CodingKey {
            case bufferLen = "buffer_len"
            case bufferOfs = "buffer_ofs"
            case samplerate
        }
    }
}
