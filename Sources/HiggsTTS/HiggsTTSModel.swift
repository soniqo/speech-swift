import AudioCommon
import Foundation
import MLX
import MLXCommon
import MLXNN
import MLXRandom

/// Sampling controls for Higgs TTS 3 generation, mirroring the reference
/// implementation's surface. Temperatures at or below ~0 sample greedily.
public struct HiggsTTSSynthesisOptions: Equatable, Sendable {
    public let temperature: Float
    public let topP: Float?
    public let topK: Int?
    public let maxNewTokens: Int
    public let seed: UInt64

    public init(
        temperature: Float = 1.0,
        topP: Float? = nil,
        topK: Int? = nil,
        maxNewTokens: Int = 2048,
        seed: UInt64 = 0
    ) throws {
        guard temperature.isFinite, temperature >= 0 else {
            throw HiggsTTSError.invalidCodes("temperature must be finite and non-negative")
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw HiggsTTSError.invalidCodes("topP must be in (0, 1]")
            }
        }
        if let topK {
            guard topK > 0 else {
                throw HiggsTTSError.invalidCodes("topK must be positive")
            }
        }
        guard maxNewTokens > 0 else {
            throw HiggsTTSError.invalidCodes("maxNewTokens must be positive")
        }
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.maxNewTokens = maxNewTokens
        self.seed = seed
    }

    public static let `default` = try! HiggsTTSSynthesisOptions()
}

/// Higgs TTS 3 runtime: Qwen3 backbone, fused codebook interface, and the
/// Higgs codec (decode for synthesis, encode for reference cloning).
public final class HiggsTTSModel: SpeechGenerationModel, ModelMemoryManageable, @unchecked Sendable {
    public static let defaultModelId = "aufklarer/Higgs-TTS-3-4B-MLX-bf16"
    public static let modelKey = "higgs-tts-3"

    private static let requiredFiles = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    public let bundleDirectory: URL
    public let config: HiggsTTSConfig
    public let tokenizer: HiggsTTSTokenizer
    public var sampleRate: Int { config.sampleRate }

    private let promptBuilder: HiggsTTSPromptBuilder
    private var backbone: HiggsTTSBackbone?
    private var fused: HiggsTTSFusedCodebook?
    private var codecWeights: [String: MLXArray]?
    private var codec: HiggsTTSCodec?
    private var weightMemory: Int = 0
    private var loaded = true

    private init(
        bundleDirectory: URL,
        config: HiggsTTSConfig,
        tokenizer: HiggsTTSTokenizer
    ) {
        self.bundleDirectory = bundleDirectory
        self.config = config
        self.tokenizer = tokenizer
        self.promptBuilder = HiggsTTSPromptBuilder(specials: tokenizer.specials) { [tokenizer] text in
            tokenizer.encode(text)
        }
    }

    public static func fromPretrained(
        modelId: String = defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> HiggsTTSModel {
        let directory = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)
        try await HuggingFaceDownloader.downloadFiles(
            modelId: modelId,
            to: directory,
            files: requiredFiles,
            offlineMode: offlineMode
        ) { progress in
            progressHandler?(progress * 0.9, "Downloading Higgs TTS 3")
        }
        return try await fromBundle(directory) { progress, message in
            progressHandler?(0.9 + progress * 0.1, message)
        }
    }

    public static func fromBundle(
        _ directory: URL,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> HiggsTTSModel {
        progressHandler?(0.05, "Loading Higgs TTS 3 bundle")
        for file in requiredFiles {
            let url = directory.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw HiggsTTSError.missingRequiredFile(file)
            }
        }
        let config = try HiggsTTSConfig.load(from: directory.appendingPathComponent("config.json"))
        let tokenizer = try await HiggsTTSTokenizer.load(from: directory)
        progressHandler?(1.0, "Higgs TTS 3 bundle ready")
        return HiggsTTSModel(bundleDirectory: directory, config: config, tokenizer: tokenizer)
    }

    public var isLoaded: Bool { loaded }

    public var memoryFootprint: Int { loaded ? weightMemory : 0 }

    public func unload() {
        backbone = nil
        fused = nil
        codecWeights = nil
        codec = nil
        loaded = false
    }

    public func prepareRuntime(progressHandler: ((Double, String) -> Void)? = nil) throws {
        try ensureLoaded()
        _ = try runtime(progressHandler: progressHandler)
    }

    /// Generates delay-patterned codec rows for `text`, optionally cloning
    /// from delay-patterned reference codes. Rows include the trailing
    /// EOC-countdown frames; feed them to `HiggsTTSDelayPattern.reverse`
    /// after trimming for codec decoding.
    public func generateDelayedCodes(
        text: String,
        references: [HiggsTTSReference] = [],
        options: HiggsTTSSynthesisOptions = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> [[Int32]] {
        try ensureLoaded()
        let (backbone, fused) = try runtime(progressHandler: progressHandler)
        MLXRandom.seed(options.seed)

        progressHandler?(0.72, "Building Higgs prompt")
        let prompt = promptBuilder.build(text: text, references: references)
        let embeddings = try promptEmbeddings(prompt, backbone: backbone, fused: fused)

        progressHandler?(0.76, "Sampling Higgs audio frames")
        var state = HiggsTTSBackbone.InferenceState.initial(config: config.textConfig)
        let (hidden, prefilled) = backbone.forward(embeddings: embeddings, state: state)
        state = prefilled
        var last = hidden[0..., -1, 0...]

        var sampler = HiggsTTSSamplerState(
            codebooks: config.audioNumCodebooks,
            bocId: config.audioBOCTokenId,
            eocId: config.audioEOCTokenId)
        var rows: [[Int32]] = []
        for step in 0..<options.maxNewTokens {
            let logits = fused.logits(last).reshaped(
                config.audioNumCodebooks, config.audioCodebookSize)
            let sampled = sample(logits: logits, options: options)
            let codes = try sampler.advance(sampled)
            rows.append(codes)
            if sampler.isDone { break }

            let next = fused.embed(
                MLXArray(codes).reshaped(1, 1, config.audioNumCodebooks))
            let (h, advanced) = backbone.forward(embeddings: next, state: state)
            state = advanced
            last = h[0..., -1, 0...]
            if step % 64 == 63 {
                progressHandler?(min(0.76 + Double(step) / Double(options.maxNewTokens) * 0.2, 0.95),
                                 "Sampling Higgs audio frames")
            }
        }
        progressHandler?(1.0, "Higgs frame generation complete")
        return rows
    }

    /// Protocol entry point: reference-free synthesis. Higgs is multilingual
    /// with automatic language handling, so `language` is ignored.
    public func generate(text: String, language: String?) async throws -> [Float] {
        try generate(text: text)
    }

    /// Encodes a reference clip into delay-patterned codes for cloning.
    /// Accepts any sample rate; resampled to 24 kHz and padded to at least
    /// one second like the reference implementation.
    public func encodeReference(
        audio url: URL,
        text: String? = nil
    ) throws -> HiggsTTSReference {
        let samples = try AudioFileLoader.load(url: url, targetSampleRate: sampleRate)
        return try encodeReference(samples: samples, sampleRate: sampleRate, text: text)
    }

    public func encodeReference(
        samples: [Float],
        sampleRate inputRate: Int,
        text: String? = nil
    ) throws -> HiggsTTSReference {
        try ensureLoaded()
        var audio = inputRate == sampleRate
            ? samples
            : AudioFileLoader.resample(samples, from: inputRate, to: sampleRate)
        if audio.count < sampleRate {
            audio += [Float](repeating: 0, count: sampleRate - audio.count)
        }
        let raw = try loadedCodec().encode(waveform24k: audio)
        let delayed = try HiggsTTSDelayPattern.apply(
            raw,
            codebooks: config.audioNumCodebooks,
            bocId: config.audioBOCTokenId,
            eocId: config.audioEOCTokenId)
        return HiggsTTSReference(delayedCodes: delayed, text: text)
    }

    /// Zero-shot voice cloning from a reference clip and its transcript.
    public func generate(
        text: String,
        referenceAudio: URL,
        referenceText: String,
        options: HiggsTTSSynthesisOptions = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> [Float] {
        progressHandler?(0.02, "Encoding Higgs reference audio")
        let reference = try encodeReference(audio: referenceAudio, text: referenceText)
        return try generate(
            text: text,
            references: [reference],
            options: options,
            progressHandler: progressHandler)
    }

    /// Synthesizes 24 kHz mono audio for `text`, optionally cloning from
    /// delay-patterned reference codes, with the reference implementation's
    /// 30 ms fade-in / 15 ms fade-out applied.
    public func generate(
        text: String,
        references: [HiggsTTSReference] = [],
        options: HiggsTTSSynthesisOptions = .default,
        progressHandler: ((Double, String) -> Void)? = nil
    ) throws -> [Float] {
        let rows = try generateDelayedCodes(
            text: text,
            references: references,
            options: options
        ) { progress, message in
            progressHandler?(progress * 0.93, message)
        }
        guard rows.count >= config.audioNumCodebooks else {
            return []
        }
        progressHandler?(0.94, "Decoding Higgs waveform")
        let raw = try HiggsTTSDelayPattern.reverse(rows, codebooks: config.audioNumCodebooks)
        let wav = try loadedCodec().decode(raw).asType(.float32)
        eval(wav)
        var samples = wav.asArray(Float.self)

        let fadeIn = min(30 * config.sampleRate / 1000, samples.count)
        for index in 0..<fadeIn {
            samples[index] *= Float(index) / Float(max(fadeIn - 1, 1))
        }
        let fadeOut = min(15 * config.sampleRate / 1000, samples.count)
        for index in 0..<fadeOut {
            samples[samples.count - 1 - index] *= Float(index) / Float(max(fadeOut - 1, 1))
        }
        progressHandler?(1.0, "Higgs synthesis complete")
        return samples
    }

    /// Decodes raw (de-delayed) codec frames to samples — parity-test surface.
    func decodeCodes(_ codes: [[Int32]]) throws -> [Float] {
        let wav = try loadedCodec().decode(codes).asType(.float32)
        eval(wav)
        return wav.asArray(Float.self)
    }

    private func loadedCodec() throws -> HiggsTTSCodec {
        if let codec {
            return codec
        }
        let built = try HiggsTTSCodec(rawWeights: loadedCodecWeights())
        codec = built
        return built
    }

    /// Teacher-forced replay for parity testing: feeds `forcedRows` as the
    /// autoregressive inputs and returns the raw fused-head logits
    /// (`[N * V]` float32 per step) before each row is consumed.
    func teacherForcedLogits(
        text: String,
        references: [HiggsTTSReference] = [],
        forcedRows: [[Int32]]
    ) throws -> [[Float]] {
        try ensureLoaded()
        let (backbone, fused) = try runtime(progressHandler: nil)
        let prompt = promptBuilder.build(text: text, references: references)
        let embeddings = try promptEmbeddings(prompt, backbone: backbone, fused: fused)

        var state = HiggsTTSBackbone.InferenceState.initial(config: config.textConfig)
        let (hidden, prefilled) = backbone.forward(embeddings: embeddings, state: state)
        state = prefilled
        var last = hidden[0..., -1, 0...]

        var collected: [[Float]] = []
        for row in forcedRows {
            let logits = fused.logits(last).reshaped(
                config.audioNumCodebooks * config.audioCodebookSize).asType(.float32)
            eval(logits)
            collected.append(logits.asArray(Float.self))
            let next = fused.embed(MLXArray(row).reshaped(1, 1, config.audioNumCodebooks))
            let (h, advanced) = backbone.forward(embeddings: next, state: state)
            state = advanced
            last = h[0..., -1, 0...]
        }
        return collected
    }

    /// Raw codec tensors from the checkpoint, keyed without the checkpoint
    /// prefix — consumed by the codec once it lands.
    func loadedCodecWeights() throws -> [String: MLXArray] {
        _ = try runtime(progressHandler: nil)
        guard let codecWeights else {
            throw HiggsTTSError.missingRequiredFile("codec weights")
        }
        return codecWeights
    }

    // MARK: - Internals

    private func promptEmbeddings(
        _ prompt: HiggsTTSPrompt,
        backbone: HiggsTTSBackbone,
        fused: HiggsTTSFusedCodebook
    ) throws -> MLXArray {
        var pieces: [MLXArray] = []
        var cursor = 0
        for (start, delayedCodes) in prompt.audioSegments {
            let textIds = Array(prompt.tokenIds[cursor..<start])
            if !textIds.isEmpty {
                pieces.append(backbone.embedText(MLXArray(textIds).reshaped(1, textIds.count)))
            }
            guard delayedCodes.allSatisfy({ $0.count == config.audioNumCodebooks }) else {
                throw HiggsTTSError.invalidCodes(
                    "reference codes must have \(config.audioNumCodebooks) codebooks per row")
            }
            let flat = delayedCodes.flatMap { $0 }
            let codes = MLXArray(flat).reshaped(1, delayedCodes.count, config.audioNumCodebooks)
            pieces.append(fused.embed(codes))
            cursor = start + delayedCodes.count
        }
        let tail = Array(prompt.tokenIds[cursor...])
        guard !tail.contains(HiggsTTSPrompt.audioPlaceholderId) else {
            throw HiggsTTSError.invalidCodes("unresolved audio placeholder in prompt")
        }
        if !tail.isEmpty {
            pieces.append(backbone.embedText(MLXArray(tail).reshaped(1, tail.count)))
        }
        return concatenated(pieces, axis: 1)
    }

    /// Independent per-codebook sampling over `[N, V]` logits.
    private func sample(logits: MLXArray, options: HiggsTTSSynthesisOptions) -> [Int32] {
        if options.temperature <= 1e-5 || options.topK == 1 {
            let ids = argMax(logits, axis: -1).asType(.int32)
            eval(ids)
            return ids.asArray(Int32.self)
        }
        var scaled = logits / options.temperature
        if let topK = options.topK, topK < config.audioCodebookSize {
            let sorted = MLX.sorted(scaled, axis: -1)
            let index = sorted.dim(-1) - topK
            let threshold = sorted[0..., index..<(index + 1)]
            scaled = MLX.which(scaled .< threshold, -Float.infinity, scaled)
        }
        if let topP = options.topP, topP < 1 {
            let probs = softmax(scaled, axis: -1)
            let order = argSort(probs, axis: -1)
            let sortedProbs = takeAlong(probs, order, axis: -1)
            let cumulative = cumsum(sortedProbs, axis: -1)
            // Mask the low-probability tail whose cumulative mass stays
            // below 1 - topP; the top token always survives.
            let maskSorted = cumulative .<= (1 - topP)
            var mask = MLXArray.zeros(like: maskSorted)
            mask = putAlong(mask, order, values: maskSorted, axis: -1)
            scaled = MLX.which(mask, -Float.infinity, scaled)
        }
        let ids = MLXRandom.categorical(scaled, axis: -1).asType(.int32)
        eval(ids)
        return ids.asArray(Int32.self)
    }

    private func runtime(
        progressHandler: ((Double, String) -> Void)?
    ) throws -> (HiggsTTSBackbone, HiggsTTSFusedCodebook) {
        if let backbone, let fused {
            return (backbone, fused)
        }

        progressHandler?(0.1, "Loading Higgs TTS 3 weights")
        let raw = try CommonWeightLoader.loadSafetensors(
            url: bundleDirectory.appendingPathComponent("model.safetensors"))

        var backboneWeights: [String: MLXArray] = [:]
        var fusedWeights: [String: MLXArray] = [:]
        var codec: [String: MLXArray] = [:]
        for (key, value) in raw {
            guard let (component, mapped) = HiggsTTSWeightMap.remap(key) else { continue }
            switch component {
            case .backbone: backboneWeights[mapped] = value
            case .fusedEmbedding: fusedWeights["embedding." + mapped] = value
            case .codec: codec[mapped] = value
            }
        }

        progressHandler?(0.4, "Building Higgs TTS 3 graph")
        let backbone = HiggsTTSBackbone(config: config.textConfig)
        try HiggsTTSWeightLoading.apply(backboneWeights, to: backbone)
        let fused = HiggsTTSFusedCodebook(
            numCodebooks: config.audioNumCodebooks,
            codebookSize: config.audioCodebookSize,
            hiddenSize: config.textConfig.hiddenSize)
        try HiggsTTSWeightLoading.apply(fusedWeights, to: fused)

        weightMemory = raw.values.reduce(0) { $0 + $1.nbytes }
        self.backbone = backbone
        self.fused = fused
        self.codecWeights = codec
        progressHandler?(0.7, "Higgs TTS 3 runtime ready")
        return (backbone, fused)
    }

    private func ensureLoaded() throws {
        if !loaded {
            throw HiggsTTSError.unloaded
        }
    }
}
