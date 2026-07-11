import Foundation
import MLX
import MLXNN
import MLXRandom
import AudioCommon
import Tokenizers
import MLXCommon

// MARK: - Public errors

public enum MAGNeTError: Error, LocalizedError {
    case missingFile(String)
    case tokenizerLoadFailed(String)
    case weightLoadFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f): return "MAGNeT: missing file \(f)"
        case .tokenizerLoadFailed(let m): return "MAGNeT: tokenizer load failed: \(m)"
        case .weightLoadFailed(let m): return "MAGNeT: weight load failed: \(m)"
        }
    }
}

// MARK: - Generation parameters

public struct MAGNeTGenerationParams: Sendable {
    public var decodingSteps: [Int]
    public var maxCfgCoef: Float
    public var minCfgCoef: Float
    public var temperature: Float
    public var topP: Float
    public var annealTemp: Bool
    public var seed: UInt64?

    public init(
        decodingSteps: [Int] = [20, 10, 10, 10],
        maxCfgCoef: Float = 10.0,
        minCfgCoef: Float = 1.0,
        temperature: Float = 3.0,
        topP: Float = 0.9,
        annealTemp: Bool = true,
        seed: UInt64? = nil
    ) {
        self.decodingSteps = decodingSteps
        self.maxCfgCoef = maxCfgCoef
        self.minCfgCoef = minCfgCoef
        self.temperature = temperature
        self.topP = topP
        self.annealTemp = annealTemp
        self.seed = seed
    }
}

// MARK: - Main entry point

public final class MAGNeTMusicGen {
    public let config: MAGNeTConfig
    public let encodecConfig: EncodecModelConfig
    public let t5Config: T5ModelConfig
    internal let _lm: MAGNeTLM
    internal let _t5: T5Encoder
    internal let _encodec: EncodecModelMLX
    internal let _textProjWeight: MLXArray  // [dim, t5_dim]
    internal let _textProjBias: MLXArray    // [dim]
    internal let _tokenizer: Tokenizer

    private init(
        config: MAGNeTConfig,
        encodecConfig: EncodecModelConfig,
        t5Config: T5ModelConfig,
        lm: MAGNeTLM,
        t5: T5Encoder,
        encodec: EncodecModelMLX,
        textProjWeight: MLXArray,
        textProjBias: MLXArray,
        tokenizer: Tokenizer
    ) {
        self.config = config
        self.encodecConfig = encodecConfig
        self.t5Config = t5Config
        self._lm = lm
        self._t5 = t5
        self._encodec = encodec
        self._textProjWeight = textProjWeight
        self._textProjBias = textProjBias
        self._tokenizer = tokenizer
    }

    // MARK: - Loading

    public static func fromPretrained(
        variant: MAGNeTVariant,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws -> MAGNeTMusicGen {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: variant, progressHandler: progressHandler)

        // MAGNeT bundle.
        let cfg = try MAGNeTConfig.load(from: paths.bundleDir.appendingPathComponent("config.json"))
        let bundleWeights = try loadSafetensors(paths.bundleDir.appendingPathComponent("model.safetensors"))

        // T5 encoder.
        let t5Cfg = try T5ModelConfig.load(from: paths.t5Dir.appendingPathComponent("config.json"))
        let t5Raw = try loadSafetensors(paths.t5Dir.appendingPathComponent("model.safetensors"))
        let t5Weights = T5Encoder.sanitize(t5Raw)

        // EnCodec.
        let encCfg = try EncodecModelConfig.load(from: paths.encodecDir.appendingPathComponent("config.json"))
        let encWeights = try loadSafetensors(paths.encodecDir.appendingPathComponent("model.safetensors"))

        // Build modules.
        let lm = MAGNeTLM(config: cfg)
        var lmStripped: [String: MLXArray] = [:]
        for (k, v) in bundleWeights where k.hasPrefix("lm.") {
            lmStripped[String(k.dropFirst("lm.".count))] = v
        }
        try lm.loadWeights(lmStripped)

        let t5 = T5Encoder(config: t5Cfg)
        try t5.loadSanitizedWeights(t5Weights)

        let encodec = EncodecModelMLX(config: encCfg, numQuantizers: cfg.nQ)
        try loadFlatWeights(into: encodec, mapping: encWeights)

        // text_conditioner.output_proj
        guard let projW = bundleWeights["text_conditioner.output_proj.weight"],
              let projB = bundleWeights["text_conditioner.output_proj.bias"]
        else {
            throw MAGNeTError.weightLoadFailed("text_conditioner.output_proj.{weight,bias} missing")
        }

        // Tokenizer (T5 unigram via swift-transformers).
        let tokenizer: Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: paths.t5Dir)
        } catch {
            throw MAGNeTError.tokenizerLoadFailed("\(error)")
        }

        return MAGNeTMusicGen(
            config: cfg, encodecConfig: encCfg, t5Config: t5Cfg,
            lm: lm, t5: t5, encodec: encodec,
            textProjWeight: projW, textProjBias: projB,
            tokenizer: tokenizer)
    }

    // MARK: - Generation

    /// Generate a waveform conditioned on `text`. Returns mono Float32 PCM at
    /// `config.sampleRate` (32 kHz). The output length is exactly
    /// `config.segmentDuration * config.sampleRate` samples (30 s × 32k = 960k).
    public func generate(
        text: String,
        params: MAGNeTGenerationParams = MAGNeTGenerationParams()
    ) -> [Float] {
        if let seed = params.seed { MLXRandom.seed(seed) }
        precondition(params.decodingSteps.count == config.nQ,
                     "decodingSteps count must equal n_q=\(config.nQ)")

        // Text conditioning.
        let ids = _tokenizer.encode(text: text)
        let tokenArray = MLXArray(ids.map { Int32($0) }).expandedDimensions(axis: 0)  // [1, L]
        let t5Out = _t5(tokenArray)                                                    // [1, L, t5_dim]
        let cond = matmul(t5Out, _textProjWeight.T) + _textProjBias                    // [1, L, dim]
        let uncond = MLXArray.zeros(cond.shape, dtype: cond.dtype)
        let conditioning = concatenated([cond, uncond], axis: 0)                       // [2, L, dim]

        let T = config.seqLen
        let K = config.nQ
        let maskId = config.maskTokenId
        var gen = MLXArray.full([1, K, T], values: MLXArray(Int32(maskId)))            // [1, K, T] int32

        for stage in 0..<K {
            gen = generateStage(
                gen: gen, conditioning: conditioning,
                stage: stage, timesteps: params.decodingSteps[stage],
                params: params)
        }

        // EnCodec decode. Input shape: [B, K, T].
        let audio = _encodec.decode(gen)                                               // [B, samples, 1]
        let mono = audio[0, 0..., 0]                                                    // [samples]
        eval(mono)
        return mono.asArray(Float.self)
    }

    // MARK: - Stage decode loop

    private func generateStage(
        gen genIn: MLXArray, conditioning: MLXArray,
        stage: Int, timesteps: Int,
        params: MAGNeTGenerationParams
    ) -> MLXArray {
        var gen = genIn
        let B = gen.dim(0); precondition(B == 1)
        var T = gen.dim(2)
        let maskId = config.maskTokenId
        let spanLen = config.spanLen
        let chunkMasking = spanLen > 1
        let dontRemask = MLXArray(Float(-1e4))

        let nChunks = T / spanLen
        if chunkMasking && T % spanLen != 0 {
            T = spanLen * nChunks
            gen = gen[0..., 0..., 0..<T]
        }
        let numUnits = chunkMasking ? nChunks : T
        var scores = MLXArray.zeros([B, 1, numUnits], dtype: .float32)
        var stageSeq = MLXArray.full([B, 1, T], values: MLXArray(Int32(maskId)))

        for tIdx in 0..<timesteps {
            let timestep = Float(tIdx) / Float(max(timesteps - 1, 1))
            let maskP = cosf(timestep * Float.pi * 0.5)
            let numMasked = max(Int(maskP * Float(numUnits)), 1)

            // Pick top-`numMasked` highest-scored units.
            // argSort is ascending; the largest scores sit at the tail.
            let sortedIdx = argSort(scores, axis: -1)
            let maskedIdx = sortedIdx[0..., 0..., (numUnits - numMasked)..<numUnits]

            let tokenMask: MLXArray
            if chunkMasking {
                let chunksMask = positionsToMask(indices: maskedIdx, N: nChunks)       // [B,1,nChunks]
                tokenMask = repeated(chunksMask, count: spanLen, axis: -1)             // [B,1,T]
            } else {
                tokenMask = positionsToMask(indices: maskedIdx, N: T)
            }
            stageSeq = MLX.where(tokenMask, MLXArray(Int32(maskId)), stageSeq)

            // Write current stage into the full grid and run cond+uncond forward.
            gen = stageWrite(gen: gen, stageSeq: stageSeq, stage: stage)
            let lmInput = concatenated([gen, gen], axis: 0).transposed(0, 2, 1)        // [2, T, K]
            let allLogits = _lm(lmInput, conditioning: conditioning, stage: stage)      // [2, T, K, card]
            let condLogits = allLogits[0..<1]
            let uncondLogits = allLogits[1..<2]
            let cfgCoef = maskP * params.maxCfgCoef + (1 - maskP) * params.minCfgCoef
            let logits = uncondLogits + (condLogits - uncondLogits) * MLXArray(cfgCoef)
            let stageLogits = logits[.ellipsis, stage, 0...].reshaped([1, T, config.card])

            let stepsLeft = timesteps - tIdx - 1
            let tTemp = params.annealTemp
                ? params.temperature * Float(stepsLeft) / Float(timesteps)
                : params.temperature
            let sampled = sampleTopP(stageLogits, topP: params.topP, temperature: tTemp)
                .reshaped([1, 1, T]).asType(.int32)

            stageSeq = MLX.where(tokenMask, sampled, stageSeq)
            gen = stageWrite(gen: gen, stageSeq: stageSeq, stage: stage)

            // Update scores from sampled prob.
            let probs = softmax(stageLogits / MLXArray(max(tTemp, 1e-2)), axis: -1)
            let sampledFlat = sampled.reshaped([1, T]).expandedDimensions(axis: -1)
            let sampledProbs = takeAlong(probs, sampledFlat, axis: -1).reshaped([1, 1, T])
            if chunkMasking {
                let reshaped = sampledProbs.reshaped([B, 1, nChunks, spanLen])
                let chunkScores = MLXArray(Float(1.0)) - reshaped.max(axis: -1)
                let chunksMask = positionsToMask(indices: maskedIdx, N: nChunks)
                scores = MLX.where(chunksMask, chunkScores, dontRemask)
            } else {
                let tokScores = -log(sampledProbs + MLXArray(Float(1e-12)))
                scores = MLX.where(tokenMask, tokScores, dontRemask)
            }
            eval(gen, scores)
        }
        return gen
    }
}

// MARK: - Weight loading helpers

private func loadSafetensors(_ url: URL) throws -> [String: MLXArray] {
    do {
        return try MLX.loadArrays(url: url)
    } catch {
        throw MAGNeTError.weightLoadFailed("\(url.lastPathComponent): \(error)")
    }
}

/// Load a flat key→array dictionary into an MLX module via `update(parameters:)`.
/// Passes the dictionary directly (matches VoxCPM2 pattern) and verifies that
/// every parameter shape matches — surfaces silent loading failures.
private func loadFlatWeights(into module: Module, mapping: [String: MLXArray]) throws {
    let params = ModuleParameters.unflattened(mapping)
    try module.update(parameters: params, verify: .shapeMismatch)
}

/// Load only weights whose key starts with `prefix`, stripping it.
private func loadModuleWeights(into module: Module, all: [String: MLXArray], prefix: String) throws {
    var sub: [String: MLXArray] = [:]
    for (k, v) in all where k.hasPrefix(prefix) {
        sub[String(k.dropFirst(prefix.count))] = v
    }
    try loadFlatWeights(into: module, mapping: sub)
}
