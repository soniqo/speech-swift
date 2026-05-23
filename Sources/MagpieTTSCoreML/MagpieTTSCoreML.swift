import CoreML
import Foundation
import AudioCommon
import MagpieTTS

/// CoreML port of Magpie-TTS Multilingual 357M.
///
/// Pipeline per synthesis call:
///   1. Per-language tokenizer (reused from the MLX MagpieTTS module) →
///      shared 2360-token vocab IDs.
///   2. `text_encoder.mlmodelc` runs once over the prompt.
///   3. `decoder_prefill.mlmodelc` seeds the 12-layer KV cache from the
///      110-frame baked speaker context.
///   4. AR loop: ``MagpieCoreMLSampler`` (Swift LocalTransformer + sampler)
///      produces the 8 codebook tokens, then `decoder_step.mlmodelc` advances
///      one frame and refreshes the cache.
///   5. `nanocodec_decoder.mlmodelc` decodes the accumulated `(T, 8)` code
///      matrix to a 22.05 kHz mono waveform.
///
/// Streaming is **not supported**: the bundled NanoCodec is a fixed 256-frame
/// window decoder, and overlap-window streaming is documented to yield
/// <15 dB SNR by the upstream exporter. ``synthesize(...)`` returns the
/// complete waveform in one call.
public final class MagpieTTSCoreML {

    public let textEncoder: MagpieCoreMLTextEncoder
    public let decoder: MagpieCoreMLDecoder
    public let nanoCodec: MagpieCoreMLNanoCodec
    public let localTransformer: MagpieCoreMLLocalTransformer
    public let sampler: MagpieCoreMLSampler

    /// Speaker context bank: 5 × `[110 * 768]` Float32. Indexed by
    /// ``MagpieCoreMLSpeaker`` raw value.
    public let speakerBank: [[Float]]

    /// 8 × `[2024 * 768]` Float32 codebook embedding tables.
    public let audioEmbeddings: [[Float]]

    /// Tokenizers reused from the MLX MagpieTTS module. Lazy-loaded on first
    /// `synthesize` for a given language.
    private let tokenizerDir: URL
    private var tokenizers: [MagpieCoreMLLanguage: MagpieTokenizer] = [:]
    private let tokenizerLock = NSLock()

    public static let sampleRate = MagpieCoreMLConstants.sampleRate

    public init(textEncoder: MagpieCoreMLTextEncoder,
                decoder: MagpieCoreMLDecoder,
                nanoCodec: MagpieCoreMLNanoCodec,
                localTransformer: MagpieCoreMLLocalTransformer,
                sampler: MagpieCoreMLSampler,
                speakerBank: [[Float]],
                audioEmbeddings: [[Float]],
                tokenizerDir: URL) {
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.nanoCodec = nanoCodec
        self.localTransformer = localTransformer
        self.sampler = sampler
        self.speakerBank = speakerBank
        self.audioEmbeddings = audioEmbeddings
        self.tokenizerDir = tokenizerDir
    }

    // MARK: - Loading

    public static func fromPretrained(
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MagpieTTSCoreML {
        let paths = try await MagpieCoreMLDownloader.ensureDownloaded(
            progressHandler: progressHandler)
        return try load(from: paths)
    }

    public static func load(from paths: MagpieCoreMLDownloader.Paths) throws -> MagpieTTSCoreML {
        let te = try MagpieCoreMLTextEncoder(url: paths.textEncoderCompiled)
        let dec = try MagpieCoreMLDecoder(
            prefillURL: paths.decoderPrefillCompiled,
            stepURL: paths.decoderStepCompiled)
        let codec = try MagpieCoreMLNanoCodec(url: paths.nanocodecCompiled)
        let speakers = try MagpieCoreMLAssets.loadSpeakerBank(
            constantsDir: paths.constantsDir)
        let audioEmbeds = try MagpieCoreMLAssets.loadAudioEmbeddings(
            constantsDir: paths.constantsDir)
        let ltWeights = try MagpieCoreMLLocalTransformerLoader.load(
            from: paths.localTransformerDir)
        let lt = MagpieCoreMLLocalTransformer(weights: ltWeights)
        let smp = MagpieCoreMLSampler(localTransformer: lt, audioEmbeddings: audioEmbeds)
        return MagpieTTSCoreML(
            textEncoder: te, decoder: dec, nanoCodec: codec,
            localTransformer: lt, sampler: smp,
            speakerBank: speakers, audioEmbeddings: audioEmbeds,
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
        let speakerCtx = speakerBank[speaker.rawValue]
        let prefill = try decoder.prefill(
            contextEmbedding: speakerCtx,
            encoderOutput: encOut.encoderOutput,
            encoderMask: encOut.encoderMask)

        let rng = MagpieCoreMLSamplerRng(seed: params.seed)

        // Initial decoder hidden from prefill: take the last row of the
        // (1, 110, 768) hidden_states output. The first AR step samples codes
        // for frame 0 from that hidden, then advances.
        let hiddenFlat = MagpieCoreMLBridge.toFloat32(prefill.hiddenStates)
        let D = MagpieCoreMLConstants.dModel
        let T = MagpieCoreMLConstants.speakerContextLength
        precondition(hiddenFlat.count == T * D)
        var hLast = Swift.Array(hiddenFlat[(T - 1) * D ..< T * D])

        var cacheK = prefill.cacheK
        var cacheV = prefill.cacheV
        var positions = prefill.positions
        var frames: [[Int32]] = []
        let stepCap = min(params.maxSteps, MagpieCoreMLConstants.maxNanocodecFrames)

        for step in 0..<stepCap {
            let codes = sampler.sample(
                decoderHidden: hLast,
                uncondDecoderHidden: nil,
                forbidEos: step < params.minFrames,
                params: params, rng: rng)
            if step >= params.minFrames, codes.contains(MagpieCoreMLConstants.audioEosId) {
                break
            }
            frames.append(codes)
            // Average the 8 codebook embeddings into a single (1,1,D) audio_embed.
            let avg = averageCodebooks(codes: codes)
            let stepOut = try decoder.step(
                audioEmbedding: avg,
                encoderOutput: encOut.encoderOutput,
                encoderMask: encOut.encoderMask,
                cacheK: cacheK, cacheV: cacheV, positions: positions)
            cacheK = stepOut.cacheK
            cacheV = stepOut.cacheV
            positions = stepOut.positions
            hLast = stepOut.decoderHidden
        }
        if frames.isEmpty { return [] }
        return try nanoCodec.decode(codes: frames)
    }

    private func averageCodebooks(codes: [Int32]) -> [Float] {
        let D = MagpieCoreMLConstants.dModel
        let K = MagpieCoreMLConstants.numCodebooks
        var out = [Float](repeating: 0, count: D)
        for k in 0..<K {
            let row = Int(codes[k]) * D
            let table = audioEmbeddings[k]
            for d in 0..<D {
                out[d] += table[row + d]
            }
        }
        let inv = 1.0 / Float(K)
        for d in 0..<D { out[d] *= inv }
        return out
    }
}
