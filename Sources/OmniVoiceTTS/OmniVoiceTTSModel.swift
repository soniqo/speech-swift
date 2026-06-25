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
        // Repair incomplete or pre-tokenizer-config caches. The HF snapshot fetch
        // is incremental, so already-present blobs are not refetched.
        let requiredFiles = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "audio_tokenizer/model.safetensors",
        ]
        let hasMissingFile = requiredFiles.contains {
            !fm.fileExists(atPath: dir.appendingPathComponent($0).path)
        }
        if hasMissingFile {
            progressHandler?(0.0, "Downloading \(modelId)...")
            // No explicit `.safetensors` in additionalFiles keeps the automatic
            // `*.safetensors` glob (fetches the top-level backbone); `audio_tokenizer/*`
            // pulls the codec + its config. `tokenizer_config.json` carries the
            // tokenizer class the loader needs.
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: dir,
                additionalFiles: [
                    "config.json", "tokenizer.json", "tokenizer_config.json", "audio_tokenizer/*",
                ],
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
    ///   - instruct: optional restricted OmniVoice style item (accent / age / gender /
    ///     pitch / whisper); `nil` for neutral.
    ///   - duration: optional fixed output length in seconds; `nil` estimates from text.
    ///   - numSteps: diffusion steps (16 default; 12 is a faster, near-equal setting).
    /// - Returns: mono Float samples at `sampleRate`.
    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String?,
        language: String = "en",
        instruct: String? = nil,
        duration: Double? = nil,
        numSteps: Int = 16
    ) throws -> [Float] {
        let refSamples = try AudioFileLoader.load(
            url: referenceAudio, targetSampleRate: cfg.sampleRate)
        let refWav = MLXArray(refSamples).reshaped([1, 1, refSamples.count])
        let refTokens = encoder.encode(refWav)

        // Target length: an explicit duration wins; otherwise estimate the token
        // count from the reference's pace (its text weight vs its token count),
        // matching OmniVoice's `_estimate_target_tokens`.
        let targetLen: Int
        if let d = duration {
            targetLen = max(1, Int((d * frameRate).rounded()))
        } else {
            let refTokenCount = refTokens.dim(refTokens.ndim - 1)
            targetLen = RuleDurationEstimator.shared.estimateTargetTokens(
                targetText: text, refText: referenceText, numRefAudioTokens: refTokenCount)
        }

        let (ids, mask) = builder.buildInputs(
            text: text, refText: referenceText, lang: language,
            refAudioTokens: refTokens, targetLen: targetLen, denoise: true, instruct: instruct)
        let tokens = model.generateTokens(
            condInputIds: ids, audioMask: mask, targetLen: targetLen, numSteps: numSteps)
        let wav = codec.decode(tokens)
        MLX.eval(wav)
        let samples = wav.asType(.float32).asArray(Float.self)
        let trimmed = Self.trimLeadingNoise(samples, sampleRate: cfg.sampleRate)
        return Self.applyEdgeFades(trimmed, sampleRate: cfg.sampleRate)
    }

    /// Raised-cosine fade in/out over the edges, matching the reference's
    /// `fade_and_pad_audio` (`fade_duration` 0.1 s). The long fade ramps over the
    /// kept leading silence so the diffusion+codec onset transient is inaudible,
    /// and removes any tail click, without shortening the speech (the fade sits on
    /// the silence the trim left, not on the words).
    static func applyEdgeFades(_ input: [Float], sampleRate: Int) -> [Float] {
        var s = input
        let n = s.count
        let fade = min(sampleRate / 10, n / 2)  // ~100 ms (or half the clip)
        guard fade > 0 else { return s }
        for i in 0 ..< fade {
            let w = Float(0.5 * (1.0 - cos(Double.pi * Double(i) / Double(fade))))
            s[i] *= w
            s[n - 1 - i] *= w
        }
        return s
    }

    /// Drop the brief codec/diffusion startup transient and any leading silence:
    /// find where the signal first stays above a small energy threshold for a
    /// sustained span (real speech, not the millisecond startup burst) and cut
    /// just before it, keeping a short lead-in. Returns the input unchanged when
    /// no sustained onset is found (e.g. an already-clean or silent clip).
    static func trimLeadingNoise(_ s: [Float], sampleRate: Int) -> [Float] {
        guard !s.isEmpty else { return s }
        let win = max(1, sampleRate / 100)  // 10 ms windows
        let sustain = 6  // require ~60 ms continuously above threshold = real speech
        let threshold: Float = 0.02
        func windowRMS(_ start: Int) -> Float {
            let end = min(start + win, s.count)
            var acc: Float = 0
            for j in start ..< end { acc += s[j] * s[j] }
            return (acc / Float(max(1, end - start))).squareRoot()
        }
        var start = 0
        var found = false
        while start + win * sustain <= s.count {
            var ok = true
            var k = 0
            while k < sustain {
                if windowRMS(start + k * win) < threshold { ok = false; break }
                k += 1
            }
            if ok { found = true; break }
            start += win
        }
        guard found, start > 0 else { return s }
        // Keep 100 ms of leading silence (matches the reference's `lead_sil`); the
        // 100 ms fade then ramps over it, reaching the speech onset gently.
        let cut = max(0, start - sampleRate / 10)
        return Array(s[cut...])
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
