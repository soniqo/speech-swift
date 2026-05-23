import CoreML
import Foundation
import AudioCommon
import MagpieTTS

/// CoreML port of Magpie-TTS Multilingual 357M, backed by the
/// `aufklarer/Magpie-TTS-Multilingual-357M-CoreML-8bit` bundle.
///
/// Pipeline per synthesis call:
///   1. Per-language tokenizer (reused from MagpieTTS) → shared 2360-token vocab IDs.
///   2. `text_encoder.mlmodelc` runs once over the prompt.
///   3. `decoder_prefill.mlmodelc` selects the baked speaker by index,
///      seeds the 12-layer self-attention KV cache (110 + 1 BOS = 111
///      positions), and emits the precomputed cross-attention K/V.
///   4. AR loop: the MLX-loaded LocalTransformer samples the 8 codebook
///      tokens per frame from `decoder_step`'s `h_last`. We ignore the
///      CoreML graph's parallel-head logits — the LT is what NeMo
///      trains as the real sampling head and parallel-head sampling
///      produces noticeably worse codebook coherence. Sampled codes are
///      averaged through the audio embedding tables (also MLX-loaded)
///      to build the next `audio_emb` input.
///   5. ``MagpieCoreMLFSQ.decodeWindow`` turns the `(T, 8)` code matrix
///      into `(1, 32, T)` FSQ-inverse latents, then
///      `nanocodec_decoder.mlmodelc` produces 22.05 kHz mono PCM in
///      64-frame chunks.
///
/// Streaming is **not** supported because the codec is traced at a fixed
/// 64-frame window. ``synthesize(...)`` returns the complete waveform.
public final class MagpieTTSCoreML {

    public let textEncoder: MagpieCoreMLTextEncoder
    public let decoder: MagpieCoreMLDecoder
    public let nanoCodec: MagpieCoreMLNanoCodec

    /// MLX MagpieTTS instance, lazy-loaded on first synthesis. Source of:
    ///   * the 8 audio embedding tables (averaged per frame for `audio_emb`),
    ///   * the 1-layer LocalTransformer + sampling head (refines
    ///     `decoder_step`'s `h_last` into the next frame's 8 codebooks).
    /// Adding this MLX dependency at runtime is the trade-off until we
    /// ship LT + audio-embedding `.npy` files inside the CoreML bundle
    /// and replace the LT with a pure-Swift implementation.
    private let mlxHelper: MagpieMLXHelper
    private let audioEmbedSource: AudioEmbedSource
    private let localSampler: MagpieMLXLocalSampler

    private let tokenizerDir: URL
    private var tokenizers: [MagpieCoreMLLanguage: MagpieTokenizer] = [:]
    private let tokenizerLock = NSLock()

    public static let sampleRate = MagpieCoreMLConstants.sampleRate

    public init(textEncoder: MagpieCoreMLTextEncoder,
                decoder: MagpieCoreMLDecoder,
                nanoCodec: MagpieCoreMLNanoCodec,
                mlxHelper: MagpieMLXHelper,
                audioEmbedSource: AudioEmbedSource,
                localSampler: MagpieMLXLocalSampler,
                tokenizerDir: URL) {
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.nanoCodec = nanoCodec
        self.mlxHelper = mlxHelper
        self.audioEmbedSource = audioEmbedSource
        self.localSampler = localSampler
        self.tokenizerDir = tokenizerDir
    }

    // MARK: - Loading

    public static func fromPretrained(
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MagpieTTSCoreML {
        let paths = try await MagpieCoreMLDownloader.ensureDownloaded(
            progressHandler: progressHandler)
        let te = try MagpieCoreMLTextEncoder(url: paths.textEncoderCompiled)
        let dec = try MagpieCoreMLDecoder(
            prefillURL: paths.decoderPrefillCompiled,
            stepURL: paths.decoderStepCompiled)
        let codec = try MagpieCoreMLNanoCodec(url: paths.nanocodecCompiled)
        let helper = MagpieMLXHelper()
        return MagpieTTSCoreML(
            textEncoder: te, decoder: dec, nanoCodec: codec,
            mlxHelper: helper,
            audioEmbedSource: MagpieMLXAudioEmbedSource(helper: helper),
            localSampler: MagpieMLXLocalSampler(helper: helper),
            tokenizerDir: paths.mlxTokenizerDir)
    }

    // MARK: - Tokenizer access

    public func tokenizer(for language: MagpieCoreMLLanguage) throws -> MagpieTokenizer {
        tokenizerLock.lock()
        defer { tokenizerLock.unlock() }
        if let cached = tokenizers[language] { return cached }
        let url = tokenizerDir.appendingPathComponent("\(language.rawValue).json")
        let tok = try MagpieTokenizer.load(from: url, language: language.mlx)
        tokenizers[language] = tok
        return tok
    }

    // MARK: - Synthesis

    public func synthesize(text: String,
                            speaker: MagpieCoreMLSpeaker = .sofia,
                            language: MagpieCoreMLLanguage = .english,
                            prephonemized: Bool = false,
                            params: MagpieCoreMLParams = MagpieCoreMLParams()) throws -> [Float] {
        let tok = try tokenizer(for: language)
        let ids = tok.tokenize(text, prephonemized: prephonemized)
        if ids.isEmpty {
            throw MagpieCoreMLError.inferenceFailed(
                stage: "tokenize", underlying: "empty token sequence")
        }
        let encOut = try textEncoder.encode(tokens: ids.map(Int32.init))
        let prefill = try decoder.prefill(
            speakerIndex: speaker.rawValue,
            encoderOutput: encOut.encoderOutput,
            encoderMask: encOut.encoderMask)

        // Seed MLX's RNG when requested so LT sampling is reproducible
        // (mirrors what MagpieTTS does for its own AR loop).
        if let seed = params.seed { MagpieMLXSeed.seed(seed) }

        var saK = prefill.saK
        var saV = prefill.saV
        let xaK = prefill.xaK
        let xaV = prefill.xaV
        var position = prefill.initialPosition
        var audioEmb = try audioEmbedSource.averageBosEmbedding()
        var frames: [[Int32]] = []
        let stepCap = min(params.maxSteps, MagpieCoreMLConstants.maxARSteps)

        for step in 0..<stepCap {
            let stepOut = try decoder.step(
                audioEmbedding: audioEmb, position: position,
                encoderOutput: encOut.encoderOutput,
                encoderMask: encOut.encoderMask,
                saK: saK, saV: saV, xaK: xaK, xaV: xaV)
            saK = stepOut.saK
            saV = stepOut.saV
            position += 1
            // Use the MLX-loaded LocalTransformer for per-frame sampling.
            // We ignore the parallel head logits the CoreML graph also
            // emits — parallel head sampling produces noticeably worse
            // codebook coherence (the LT is what's trained as the real
            // sampling head).
            let hLastFlat = MagpieCoreMLBridge.toFloat32(stepOut.hLast)
            let codes = try localSampler.sample(
                hLastFlat: hLastFlat,
                forbidEos: step < params.minFrames,
                temperature: params.temperature, topK: params.topK)
            if step >= params.minFrames, codes.contains(MagpieCoreMLConstants.audioEosId) {
                break
            }
            frames.append(codes)
            audioEmb = try audioEmbedSource.averageEmbedding(codes: codes)
        }
        if frames.isEmpty { return [] }
        return try nanoCodec.decode(codes: frames)
    }
}

// MARK: - Audio embedding source

/// Per-frame `audio_emb` builder. The CoreML decoder's input expects a
/// `(1, 1, 768)` audio embedding which Magpie computes as the average of
/// the previous frame's 8 sampled codebook tokens projected through the
/// per-codebook embedding tables (`embed_audio_frame` in NeMo).
public protocol AudioEmbedSource {
    /// `[768]` for the AR-loop bootstrap frame (codebook BOS).
    func averageBosEmbedding() throws -> [Float]
    /// `[768]` — average of the 8 codebooks' embeddings for the given
    /// sampled codes.
    func averageEmbedding(codes: [Int32]) throws -> [Float]
}
