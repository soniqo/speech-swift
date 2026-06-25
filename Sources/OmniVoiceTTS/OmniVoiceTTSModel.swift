import AudioCommon
import Foundation
import MLX
import MLXNN

/// High-level OmniVoice voice-cloning TTS. Downloads / loads the bundle and
/// synthesizes cloned speech from a reference clip + target text, wrapping the
/// Qwen3 diffusion backbone, the Higgs-audio v2 codec (encode + decode), and the
/// text front-end. This is the entry point the Studio sidecar drives.
public final class OmniVoiceTTSModel {
    /// Default published bundle (int8 backbone + fp16 codec). The fp16 backbone
    /// bundle is `aufklarer/OmniVoice-MLX-fp16`.
    public static let defaultModelId = "aufklarer/OmniVoice-MLX-int8"

    let cfg: OmniVoiceConfig
    let model: OmniVoiceModel
    let codec: OmniVoiceCodec
    let encoder: OmniVoiceCodecEncoder
    let builder: OmniVoiceInputBuilder

    /// Output sample rate (24 kHz).
    public var sampleRate: Int { cfg.sampleRate }
    /// Codec frame rate (Hz): each audio token decodes to `hop_length` (960)
    /// samples at 24 kHz, so 24000 / 960 = 25 tokens per second. Used to convert a
    /// requested duration to a token count.
    public let frameRate = 25.0

    init(cfg: OmniVoiceConfig, model: OmniVoiceModel, codec: OmniVoiceCodec,
        encoder: OmniVoiceCodecEncoder, builder: OmniVoiceInputBuilder) {
        self.cfg = cfg
        self.model = model
        self.codec = codec
        self.encoder = encoder
        self.builder = builder
    }

    /// Load from a local bundle directory: `model.safetensors` (backbone),
    /// `audio_tokenizer/model.safetensors` (codec), `tokenizer.json`, `config.json`.
    /// A `quantization` block in `config.json` selects the int8 load path.
    public static func fromBundle(_ dir: URL) async throws -> OmniVoiceTTSModel {
        let cfg = OmniVoiceConfig()
        let quant = readQuantization(dir.appendingPathComponent("config.json"))

        let model = OmniVoiceModel(cfg)
        try model.loadWeights(
            from: dir.appendingPathComponent("model.safetensors"), quantization: quant)

        let codecURL = dir.appendingPathComponent("audio_tokenizer/model.safetensors")
        let codec = OmniVoiceCodec()
        try codec.loadWeights(from: codecURL)
        let encoder = OmniVoiceCodecEncoder()
        try encoder.loadWeights(from: codecURL)

        let tokenizer = try await OmniVoiceTokenizer.load(from: dir)
        let builder = OmniVoiceInputBuilder(tokenizer: tokenizer, config: cfg)
        return OmniVoiceTTSModel(
            cfg: cfg, model: model, codec: codec, encoder: encoder, builder: builder)
    }

    /// Download the bundle from Hugging Face (cached) and load it.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> OmniVoiceTTSModel {
        let dir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.appendingPathComponent("model.safetensors").path) {
            progressHandler?(0.0, "Downloading \(modelId)...")
            // No explicit `.safetensors` in additionalFiles keeps the automatic
            // `*.safetensors` glob (fetches the top-level backbone); `audio_tokenizer/*`
            // pulls the codec + its config.
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: dir,
                additionalFiles: ["config.json", "tokenizer.json", "audio_tokenizer/*"],
                offlineMode: offlineMode
            ) { progressHandler?($0 * 0.9, "Downloading model...") }
        }
        progressHandler?(0.92, "Loading weights...")
        let m = try await fromBundle(dir)
        progressHandler?(1.0, "Ready")
        return m
    }

    /// Synthesize cloned speech.
    /// - Parameters:
    ///   - text: target text to speak.
    ///   - referenceAudio: reference clip to clone (any format/rate; resampled to 24 kHz).
    ///   - referenceText: optional transcript of the reference (improves prosody).
    ///   - language: OmniVoice language id (e.g. "en"); the model covers 600+.
    ///   - duration: optional fixed output length in seconds; `nil` estimates from text.
    ///   - numSteps: diffusion steps (16 default; 12 is a faster, near-equal setting).
    /// - Returns: mono Float samples at `sampleRate`.
    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String?,
        language: String = "en",
        duration: Double? = nil,
        numSteps: Int = 16
    ) throws -> [Float] {
        let refSamples = try AudioFileLoader.load(
            url: referenceAudio, targetSampleRate: cfg.sampleRate)
        let refWav = MLXArray(refSamples).reshaped([1, 1, refSamples.count])
        let refTokens = encoder.encode(refWav)

        let targetLen = duration.map { max(1, Int(($0 * frameRate).rounded())) }
            ?? builder.estimateTargetLen(text: text, lang: language, frameRate: frameRate)

        let (ids, mask) = builder.buildInputs(
            text: text, refText: referenceText, lang: language,
            refAudioTokens: refTokens, targetLen: targetLen, denoise: true, instruct: nil)
        let tokens = model.generateTokens(
            condInputIds: ids, audioMask: mask, targetLen: targetLen, numSteps: numSteps)
        let wav = codec.decode(tokens)
        MLX.eval(wav)
        return wav.asType(.float32).asArray(Float.self)
    }

    private static func readQuantization(_ url: URL) -> (groupSize: Int, bits: Int)? {
        guard let data = try? Data(contentsOf: url),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let q = json["quantization"] as? [String: Any],
            let g = q["group_size"] as? Int, let b = q["bits"] as? Int
        else { return nil }
        return (g, b)
    }
}
