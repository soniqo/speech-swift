import Foundation

/// Configuration emitted beside the complete VoiceChat `tts/` MLX bundle.
public struct VoiceChatSpeechConfiguration: Codable, Sendable {
    public let modelType: String
    public let sampleRate: Int
    public let frameSamples: Int
    public let frameSeconds: Double
    public let speaker: String
    public let promptFrames: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let latentSize: Int
    public let numQuantizers: Int
    public let codebookSize: Int
    public let numIter: Int
    public let guidanceScale: Double
    public let topP: Double
    public let noiseScale: Double
    public let codecDenseFP16: Bool
    public let weightLayout: String
    public let quantization: QuantizationConfig?
    public let quantizationPerTensor: [String: QuantizationConfig]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case sampleRate = "sample_rate"
        case frameSamples = "frame_samples"
        case frameSeconds = "frame_seconds"
        case speaker
        case promptFrames = "prompt_frames"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case latentSize = "latent_size"
        case numQuantizers = "num_quantizers"
        case codebookSize = "codebook_size"
        case numIter = "num_iter"
        case guidanceScale = "guidance_scale"
        case topP = "top_p"
        case noiseScale = "noise_scale"
        case codecDenseFP16 = "codec_dense_fp16"
        case weightLayout = "weight_layout"
        case quantization
        case quantizationPerTensor = "quantization_per_tensor"
    }

    public static func load(from directory: URL) throws -> VoiceChatSpeechConfiguration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        return try JSONDecoder().decode(VoiceChatSpeechConfiguration.self, from: data)
    }
}
