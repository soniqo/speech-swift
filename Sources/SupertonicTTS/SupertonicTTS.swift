#if canImport(CoreML)
import CoreML
import Foundation
import AudioCommon

/// SupertonicTTS-3 on Apple via CoreML — 99M non-autoregressive flow-matching multilingual TTS
/// (44.1 kHz, 31 languages, **G2P-free**). Mirrors the C++ `LiteRTSupertonicTts` but, because the
/// CoreML graphs carry a `RangeDim` latent axis, the host runs at the **true** latent length per
/// utterance — no fixed-window truncation.
///
///     let tts = try await SupertonicTTSModel.fromPretrained()
///     let pcm = try tts.synthesize(text: "Hello there", voice: "F1", language: "en")  // [Float] @ 44.1 kHz
public final class SupertonicTTSModel: @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Supertonic-3-CoreML"

    struct VoiceStyle { let ttl: [Float]; let dp: [Float] }   // [1,50,256], [1,8,16]

    private let config: SupertonicConfig
    private let tokenizer: SupertonicTokenizer
    private let graphs: SupertonicGraphs
    private let voices: [String: VoiceStyle]
    public let defaultVoice: String

    /// Voice ids available (e.g. "F1"…"F5", "M1"…"M5").
    public var availableVoices: [String] { voices.keys.sorted() }

    /// Inter-chunk silence in seconds.
    public var chunkSilenceSeconds: Double = 0.3

    init(directory: URL, computeUnits: MLComputeUnits, config: SupertonicConfig = .default) throws {
        self.config = config
        self.tokenizer = try SupertonicTokenizer.load(
            from: directory.appendingPathComponent("unicode_indexer.json"))
        self.graphs = try SupertonicGraphs(dir: directory, cacheDir: directory, computeUnits: computeUnits)
        self.voices = try Self.loadVoices(dir: directory, config: config)
        guard !voices.isEmpty else {
            throw SupertonicError.badAsset("no voice styles under \(directory.path)/voice_styles")
        }
        self.defaultVoice = voices["F1"] != nil ? "F1" : voices.keys.sorted().first!
    }

    /// Download `aufklarer/Supertonic-3-CoreML` (or `modelId`) and load it.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        localPath: String? = nil,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> SupertonicTTSModel {
        let dir: URL
        if let localPath {
            dir = URL(fileURLWithPath: localPath, isDirectory: true)
        } else {
            dir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
            progressHandler?(0.0, "Downloading SupertonicTTS…")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId, to: dir,
                additionalFiles: [
                    "DurationPredictor.mlpackage/**", "TextEncoder.mlpackage/**",
                    "VectorEstimator.mlpackage/**", "Vocoder.mlpackage/**",
                    "tts.json", "unicode_indexer.json", "voice_styles/*.json",
                ],
                offlineMode: offlineMode
            ) { p in progressHandler?(p * 0.85, "Downloading SupertonicTTS…") }
        }
        progressHandler?(0.9, "Loading model…")
        let model = try SupertonicTTSModel(directory: dir, computeUnits: computeUnits)
        progressHandler?(1.0, "Ready")
        return model
    }

    // MARK: - synthesis

    /// Synthesize `text` in ISO `language` with `voice` → mono Float32 PCM at `sampleRate`.
    public func synthesize(text: String,
                           voice: String? = nil,
                           language: String = "en",
                           options: SupertonicOptions = .default) throws -> [Float] {
        let voiceId = voice ?? defaultVoice
        guard let style = voices[voiceId] else { throw SupertonicError.voiceNotFound(voiceId) }
        guard tokenizer.supports(language) else { throw SupertonicError.unsupportedLanguage(language) }

        let chunks = tokenizer.chunk(text, lang: language, textLength: config.textLength)
        let baseSeed = options.seed != 0 ? options.seed : UInt64.random(in: 1...UInt64.max)
        let silence = Int(chunkSilenceSeconds * Double(config.sampleRate))

        var out: [Float] = []
        for (ci, chunk) in chunks.enumerated() {
            // Decorrelate the latent noise per chunk.
            let seed = baseSeed &+ (0x9E3779B97F4A7C15 &* UInt64(ci + 1))
            let pcm = try synthChunk(chunk, lang: language, voice: style, options: options, seed: seed)
            if ci > 0, silence > 0 { out.append(contentsOf: [Float](repeating: 0, count: silence)) }
            out.append(contentsOf: pcm)
        }
        return out
    }

    private func synthChunk(_ chunk: String, lang: String, voice: VoiceStyle,
                            options: SupertonicOptions, seed: UInt64) throws -> [Float] {
        let T = config.textLength, C = config.latentChannels, chunkSize = config.chunkSamples
        let tok = try tokenizer.process(chunk, lang: lang, textLength: T)

        let textIds = try SupertonicBridge.int32(tok.ids, shape: [1, T])
        let textMask = try SupertonicBridge.fp32(tok.mask, shape: [1, 1, T])
        let styleTtl = try SupertonicBridge.fp32(voice.ttl, shape: [1, 50, 256])
        let styleDp = try SupertonicBridge.fp32(voice.dp, shape: [1, 8, 16])

        // 1) duration → seconds, /speed.
        var dur = try graphs.predictDuration(textIds: textIds, styleDp: styleDp, textMask: textMask)
        dur /= options.speed
        guard dur > 0, dur.isFinite else { return [] }

        // 2) text embedding (reused across the ODE steps).
        let textEmb = try graphs.encodeText(textIds: textIds, styleTtl: styleTtl, textMask: textMask)

        // 3) latent geometry — TRUE length (RangeDim), floored at the vector_estimator minimum.
        let wavLen = Int(Double(dur) * Double(config.sampleRate))        // truncate, matches infer.py
        let lTrue = (wavLen + chunkSize - 1) / chunkSize
        let L = max(lTrue, config.latentMin)
        let lFill = min(max(lTrue, 1), L)

        var latentMask = [Float](repeating: 0, count: L)
        for t in 0..<lFill { latentMask[t] = 1 }

        var rng = GaussianRNG(seed: seed)
        var xt = [Float](repeating: 0, count: C * L)
        for c in 0..<C {
            let base = c * L
            for t in 0..<L { xt[base + t] = rng.nextGaussian() * latentMask[t] }
        }

        let latentMaskArr = try SupertonicBridge.fp32(latentMask, shape: [1, 1, L])
        let totalStepArr = try SupertonicBridge.fp32([Float(options.totalStep)], shape: [1])

        // 4) flow-matching ODE — feed xt forward.
        for step in 0..<options.totalStep {
            let noisy = try SupertonicBridge.fp32(xt, shape: [1, C, L])
            let cur = try SupertonicBridge.fp32([Float(step)], shape: [1])
            let denoised = try graphs.vectorStep(
                noisy: noisy, textEmb: textEmb, styleTtl: styleTtl,
                latentMask: latentMaskArr, textMask: textMask, currentStep: cur, totalStep: totalStepArr)
            xt = SupertonicBridge.toFloat32(denoised)
        }

        // 5) vocode + trim to floor(SR*dur).
        let latent = try SupertonicBridge.fp32(xt, shape: [1, C, L])
        let wav = try graphs.vocode(latent: latent)
        var n = Int(Double(config.sampleRate) * Double(dur))
        n = min(n, chunkSize * lFill)
        n = min(n, wav.count)
        return Array(wav.prefix(n))
    }

    // MARK: - voices

    private static func flattenFloats(_ obj: Any) -> [Float] {
        if let arr = obj as? [Any] { return arr.flatMap { flattenFloats($0) } }
        if let n = obj as? NSNumber { return [n.floatValue] }
        return []
    }

    private static func loadVoices(dir: URL, config: SupertonicConfig) throws -> [String: VoiceStyle] {
        let vdir = dir.appendingPathComponent("voice_styles", isDirectory: true)
        let files = (try? FileManager.default.contentsOfDirectory(
            at: vdir, includingPropertiesForKeys: nil)) ?? []
        var voices: [String: VoiceStyle] = [:]
        for f in files where f.pathExtension == "json" {
            guard let obj = try? JSONSerialization.jsonObject(with: Data(contentsOf: f)),
                  let json = obj as? [String: Any],
                  let ttlNode = json["style_ttl"] as? [String: Any], let ttlData = ttlNode["data"],
                  let dpNode = json["style_dp"] as? [String: Any], let dpData = dpNode["data"]
            else { continue }
            let ttl = flattenFloats(ttlData), dp = flattenFloats(dpData)
            guard ttl.count == config.styleTtlCount, dp.count == config.styleDpCount else { continue }
            voices[f.deletingPathExtension().lastPathComponent] = VoiceStyle(ttl: ttl, dp: dp)
        }
        return voices
    }
}

/// Deterministic Gaussian sampler (SplitMix64 + Box–Muller). Runtime noise is stochastic anyway
/// (flow-matching: "trajectory divergence, not degradation"); a fixed seed makes it reproducible.
struct GaussianRNG {
    private var state: UInt64
    init(seed: UInt64) { state = seed }

    private mutating func nextU64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }
    private mutating func nextUniform() -> Float {
        Float((nextU64() >> 40) &+ 1) / Float(1 << 24)   // (0,1]
    }
    mutating func nextGaussian() -> Float {
        let u1 = nextUniform(), u2 = nextUniform()
        return sqrtf(-2 * logf(u1)) * cosf(2 * .pi * u2)
    }
}
#endif
