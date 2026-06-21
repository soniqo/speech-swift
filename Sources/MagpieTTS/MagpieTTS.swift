import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCommon

/// Magpie-TTS Multilingual 357M (NVIDIA) — MLX port for Apple Silicon.
///
/// Pipeline per synthesis call:
///   1. Per-language tokeniser → vocab IDs (T text tokens).
///   2. ``MagpieTextEncoder`` runs once over the prompt (6 layers).
///   3. ``MagpieDecoder.prefill(...)`` seeds the 110-frame baked speaker
///      context + BOS frame into the KV cache.
///   4. AR loop: at each step the LocalTransformer samples the 8 codebooks
///      sequentially, then the next decoder hidden is produced by
///      ``MagpieDecoder.step(...)``. EOS detection on the per-codebook
///      parallel head.
///   5. ``MagpieNanoCodec`` decodes the (T_frames, 8) code matrix to a 22.05
///      kHz waveform.
public final class MagpieTTS {

    // Sub-modules
    public let textEncoder: MagpieTextEncoder
    public let decoder: MagpieDecoder
    public let localTransformer: MagpieLocalTransformer
    public let nanoCodec: MagpieNanoCodec

    // Tokenisers per language (loaded lazily on first use).
    private let tokenizerDir: URL
    private var tokenizers: [MagpieLanguage: MagpieTokenizer] = [:]

    // Configs (kept around for downstream queries — sample rate, codebook count…)
    public let decoderConfig: MagpieDecoderConfig
    public let nanoCodecConfig: MagpieNanoCodecConfig

    public static let sampleRate = 22050
    public static let codecFramesPerSecond: Double = Double(22050) / Double(1024)  // 21.5

    public init(textEncoder: MagpieTextEncoder, decoder: MagpieDecoder,
                localTransformer: MagpieLocalTransformer, nanoCodec: MagpieNanoCodec,
                decoderConfig: MagpieDecoderConfig, nanoCodecConfig: MagpieNanoCodecConfig,
                tokenizerDir: URL) {
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.localTransformer = localTransformer
        self.nanoCodec = nanoCodec
        self.decoderConfig = decoderConfig
        self.nanoCodecConfig = nanoCodecConfig
        self.tokenizerDir = tokenizerDir
    }

    // MARK: - Loading

    public static func fromPretrained(
        variant: MagpieTTSVariant = .int8,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MagpieTTS {
        let paths = try await MagpieTTSDownloader.ensureDownloaded(
            variant: variant, progressHandler: progressHandler)
        return try MagpieTTS.load(from: paths)
    }

    public static func load(from paths: MagpieTTSDownloader.Paths) throws -> MagpieTTS {
        let teCfg = try decode(
            MagpieTextEncoderConfig.self,
            from: paths.textEncoderDir.appendingPathComponent("config.json"))
        let decCfg = try decode(
            MagpieDecoderConfig.self,
            from: paths.decoderPrefillDir.appendingPathComponent("config.json"))
        let codecCfg = try decode(
            MagpieNanoCodecConfig.self,
            from: paths.nanocodecDir.appendingPathComponent("config.json"))

        let te = try MagpieWeightLoader.loadTextEncoder(
            bundleDir: paths.textEncoderDir, config: teCfg)
        let (dec, lt) = try MagpieWeightLoader.loadDecoder(
            bundleDir: paths.decoderPrefillDir, config: decCfg)
        let codec = try MagpieWeightLoader.loadNanoCodec(
            bundleDir: paths.nanocodecDir, config: codecCfg)

        return MagpieTTS(textEncoder: te, decoder: dec,
                         localTransformer: lt, nanoCodec: codec,
                         decoderConfig: decCfg, nanoCodecConfig: codecCfg,
                         tokenizerDir: paths.tokenizerDir)
    }

    private static func decode<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieTTSError.missingFile(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(T.self, from: data)
    }

    // MARK: - Tokeniser access

    public func tokenizer(for language: MagpieLanguage) throws -> MagpieTokenizer {
        if let cached = tokenizers[language] { return cached }
        // Japanese has no shipped JSON; reuse the EN vocab + a katakana
        // transliteration fall-back inside the tokenizer.
        let fileLang = (language == .japanese) ? MagpieLanguage.english : language
        let url = tokenizerDir.appendingPathComponent("\(fileLang.rawValue).json")
        let tok = try MagpieTokenizer.load(from: url, language: language)
        tokenizers[language] = tok
        return tok
    }

    // MARK: - Synthesis (batch)

    /// Synthesize a complete utterance in one pass. Returns 22.05 kHz PCM
    /// (Float32, ±1.0).
    public func synthesize(text: String,
                           speaker: MagpieSpeaker = .sofia,
                           language: MagpieLanguage = .english,
                           prephonemized: Bool = false,
                           params: MagpieTTSParams = MagpieTTSParams()) throws -> [Float] {
        if let seed = params.seed { MLXRandom.seed(seed) }
        let tok = try tokenizer(for: language)
        let ids = tok.tokenize(text, prephonemized: prephonemized)
        if ids.isEmpty {
            throw MagpieTTSError.textEncodingFailed("empty token sequence after tokenisation")
        }
        let codes = try sampleCodeSequence(
            textIds: ids, speaker: speaker, params: params)
        if codes.isEmpty { return [] }
        let codeMatrix = MLXArray(codes.flatMap { $0 }, [codes.count, MagpieNumCodebooks])
            .reshaped([1, codes.count, MagpieNumCodebooks])
        let audio = nanoCodec(codeMatrix)
        eval(audio)
        return audio[0].asArray(Float.self)
    }

    // MARK: - Synthesis (streaming)

    /// Streaming variant. Audio is emitted every ``framesPerChunk`` codec
    /// frames (≈ 46 ms each). The first chunk uses ``firstChunkFrames`` so
    /// callers can choose lower first-packet latency.
    public func synthesizeStream(
        text: String,
        speaker: MagpieSpeaker = .sofia,
        language: MagpieLanguage = .english,
        prephonemized: Bool = false,
        params: MagpieTTSParams = MagpieTTSParams(),
        firstChunkFrames: Int = 8,
        framesPerChunk: Int = 25
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    try self.runStreaming(
                        text: text, speaker: speaker, language: language,
                        prephonemized: prephonemized, params: params,
                        firstChunkFrames: firstChunkFrames,
                        framesPerChunk: framesPerChunk,
                        continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - Core AR loop

    /// Run the AR loop and return the sampled `T_frames × 8` code matrix as a
    /// flat list of frames. Used by the batch path; the streaming path
    /// inlines the same logic so it can emit codec output incrementally.
    private func sampleCodeSequence(textIds: [Int],
                                     speaker: MagpieSpeaker,
                                     params: MagpieTTSParams) throws -> [[Int32]] {
        let textIdsArr = MLXArray(textIds.map(Int32.init), [textIds.count])
            .reshaped([1, textIds.count])
        let memMask = MLXArray.ones(textIdsArr.shape, dtype: .float32)
        let encOut = textEncoder(textIdsArr, mask: memMask)
        eval(encOut)

        var (hLast, caches) = decoder.prefill(
            speakerIdx: speaker.rawValue,
            encoderOutput: encOut,
            encoderMask: memMask)
        eval(hLast)

        var frames: [[Int32]] = []
        var position = decoderConfig.bakedT
        for step in 0..<params.maxSteps {
            let codes = sampleLocalTransformer(
                hLast: hLast,
                forbidEos: (step < params.minFrames),
                temperature: params.temperature,
                topK: params.topK)
            if ProcessInfo.processInfo.environment["MAGPIE_DEBUG"] == "1", step < 5 {
                print("[magpie] frame \(step): \(codes)")
            }
            if step >= params.minFrames, codes.contains(MagpieAudioEosId) {
                break
            }
            frames.append(codes)
            // Advance decoder by one frame.
            let codesArr = MLXArray(codes, [MagpieNumCodebooks])
                .reshaped([1, 1, MagpieNumCodebooks])
            let audioEmb = decoder.embedAudioFrame(codesArr)
            let (_, h, _) = decoder.step(
                audioEmb: audioEmb,
                encoderOutput: encOut,
                encoderMask: memMask,
                caches: caches,
                position: position)
            hLast = h
            position += 1
            eval(hLast)
        }
        return frames
    }

    /// Streaming variant of ``sampleCodeSequence``: emits codec output
    /// every ``framesPerChunk`` frames (with ``firstChunkFrames`` for the
    /// very first chunk). Internally re-runs the codec on the *full*
    /// accumulated frame buffer at each emit, which is fine: the codec is
    /// causal so the new samples are deterministic suffixes.
    private func runStreaming(text: String,
                               speaker: MagpieSpeaker,
                               language: MagpieLanguage,
                               prephonemized: Bool,
                               params: MagpieTTSParams,
                               firstChunkFrames: Int,
                               framesPerChunk: Int,
                               continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation) throws {
        if let seed = params.seed { MLXRandom.seed(seed) }
        let tok = try tokenizer(for: language)
        let ids = tok.tokenize(text, prephonemized: prephonemized)
        if ids.isEmpty { return }

        let textIdsArr = MLXArray(ids.map(Int32.init), [ids.count])
            .reshaped([1, ids.count])
        let memMask = MLXArray.ones(textIdsArr.shape, dtype: .float32)
        let encOut = textEncoder(textIdsArr, mask: memMask)
        eval(encOut)

        var (hLast, caches) = decoder.prefill(
            speakerIdx: speaker.rawValue,
            encoderOutput: encOut, encoderMask: memMask)
        eval(hLast)

        let t0 = CFAbsoluteTimeGetCurrent()
        var frames: [[Int32]] = []
        var emittedSamples = 0
        var nextEmitAt = firstChunkFrames
        var position = decoderConfig.bakedT

        func emit(isFinal: Bool) {
            if frames.isEmpty { return }
            let codes = frames.flatMap { $0 }
            let codeMatrix = MLXArray(codes, [frames.count, MagpieNumCodebooks])
                .reshaped([1, frames.count, MagpieNumCodebooks])
            let audio = nanoCodec(codeMatrix)
            eval(audio)
            let full = audio[0].asArray(Float.self)
            if emittedSamples >= full.count {
                if isFinal {
                    continuation.yield(AudioChunk(
                        samples: [], sampleRate: Self.sampleRate,
                        frameIndex: frames.count, isFinal: true,
                        elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
                }
                return
            }
            let slice = Array(full[emittedSamples..<full.count])
            emittedSamples = full.count
            continuation.yield(AudioChunk(
                samples: slice, sampleRate: Self.sampleRate,
                frameIndex: frames.count, isFinal: isFinal,
                elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
        }

        for step in 0..<params.maxSteps {
            let codes = sampleLocalTransformer(
                hLast: hLast,
                forbidEos: (step < params.minFrames),
                temperature: params.temperature,
                topK: params.topK)
            if step >= params.minFrames, codes.contains(MagpieAudioEosId) {
                break
            }
            frames.append(codes)
            if frames.count >= nextEmitAt {
                emit(isFinal: false)
                nextEmitAt = frames.count + framesPerChunk
            }
            let codesArr = MLXArray(codes, [MagpieNumCodebooks])
                .reshaped([1, 1, MagpieNumCodebooks])
            let audioEmb = decoder.embedAudioFrame(codesArr)
            let (_, h, _) = decoder.step(
                audioEmb: audioEmb,
                encoderOutput: encOut, encoderMask: memMask,
                caches: caches, position: position)
            hLast = h
            position += 1
            eval(hLast)
        }
        emit(isFinal: true)
    }

    // MARK: - LocalTransformer per-frame sampling

    /// Sample the 8 codebooks for one frame autoregressively.
    /// Public so the CoreML backend (which runs the big models on
    /// CoreML/ANE but borrows the MLX-loaded LocalTransformer for
    /// per-frame sampling) can drive sampling from a CoreML-produced
    /// `h_last` without re-implementing the LT in pure Swift.
    public func sampleLocalTransformer(hLast: MLXArray,
                                        forbidEos: Bool,
                                        temperature: Float,
                                        topK: Int) -> [Int32] {
        let cache = MagpieKVCache()
        var x = localTransformer.inProjection(hLast)
        var codes = [Int32]()
        codes.reserveCapacity(MagpieNumCodebooks)
        for k in 0..<MagpieNumCodebooks {
            let h = localTransformer(x, position: k, cache: cache)
            // The last position's hidden produces the codebook-k logits.
            let tail = h[0..., (h.dim(1) - 1)..<h.dim(1), 0...]
            var logits = localTransformer.outProjections[k](tail[0])  // (1, V)
            var forbid: [Int] = [Int(MagpieAudioBosId)]
            if forbidEos { forbid.append(Int(MagpieAudioEosId)) }
            logits = forbidIds(logits, ids: forbid)
            let tok = sampleTopK(logits, temperature: temperature, k: topK)
            eval(tok)
            let value = Int32(tok.asArray(Int32.self)[0])
            codes.append(value)
            if k < MagpieNumCodebooks - 1 {
                let idArr = MLXArray([value], [1, 1])  // int32
                let emb = decoder.audioEmbeddings[k](idArr)  // (1, 1, d_dec)
                x = localTransformer.inProjection(emb)
            }
        }
        return codes
    }
}
