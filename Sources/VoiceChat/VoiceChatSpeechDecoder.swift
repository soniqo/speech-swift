import Foundation
import MLX
import MLXRandom

public enum VoiceChatGenerationError: Error, CustomStringConvertible {
    case speechDecoderNotPrimed
    case invalidSpeechConfiguration(String)

    public var description: String {
        switch self {
        case .speechDecoderNotPrimed:
            return "call warmup() before generating VoiceChat speech frames"
        case .invalidSpeechConfiguration(let message):
            return "invalid VoiceChat speech configuration: \(message)"
        }
    }
}

public struct VoiceChatSpeechGenerationParameters: Sendable {
    public var guidance: Float
    public var topP: Float
    public var noise: Float
    public var iterations: Int

    public init(
        guidance: Float = 0.2,
        topP: Float = 0.95,
        noise: Float = 0.001,
        iterations: Int = 8
    ) {
        self.guidance = guidance
        self.topP = topP
        self.noise = noise
        self.iterations = iterations
    }
}

/// Mutable generation state owned by one duplex session.
///
/// Keeping both attention and character-embedding caches here lets multiple
/// sessions share the immutable decoder weights without resetting or advancing
/// each other's speech timeline.
public final class VoiceChatSpeechDecoderState {
    let attention: [VoiceChatSpeechAttentionCache]
    var textEmbeddings: [Int: MLXArray] = [:]

    init(attention: [VoiceChatSpeechAttentionCache]) {
        self.attention = attention
    }
}

/// VoiceChat's EAR-TTS decoder: one text-channel token in and one 80 ms frame
/// of 31 residual-codebook ids out.
///
/// This is a standalone text-conditioned decoder. It does not consume the
/// 4,480-wide language-model state and there is no projection between those
/// dimensions. Generation is MaskGIT-style over eight unmasking iterations.
public final class VoiceChatSpeechDecoder {
    public static let speechPad = 1_024
    public static let speechEOS = 1_025
    public static let speechBOS = 1_026
    public static let controlCodes = Set([speechPad, speechEOS, speechBOS])

    public let configuration: VoiceChatSpeechConfiguration
    public let silenceCodes: MLXArray

    private let backbone: VoiceChatSpeechBackbone
    private let head: VoiceChatMoGHead
    private let textEncoder: VoiceChatCharacterEncoder
    private let fusion: VoiceChatGatedFusion
    private let codeEmbedding: VoiceChatMatrix
    private let beginningEmbedding: MLXArray
    private let nullEmbedding: MLXArray
    private let residualCodebooks: MLXArray
    private let paddedCodebooks: MLXArray
    private let speakerPrompt: MLXArray
    private let textPadID: Int
    private let textEOSID: Int

    init(
        weights: [String: MLXArray],
        configuration: VoiceChatSpeechConfiguration,
        tokenizer: VoiceChatTokenizer
    ) throws {
        guard configuration.modelType == "nemotron_voicechat_tts",
              configuration.sampleRate == VoiceChatCodec.sampleRate,
              configuration.frameSamples == VoiceChatCodec.samplesPerFrame,
              abs(configuration.frameSeconds - 0.08) < 1e-9,
              configuration.speaker == "Aria",
              configuration.hiddenSize == 1_152,
              configuration.numHiddenLayers == 28,
              configuration.numAttentionHeads == 16,
              configuration.headDim == 72,
              configuration.latentSize == 512,
              configuration.numQuantizers == 31,
              configuration.codebookSize == 1_024,
              configuration.promptFrames == 37,
              configuration.numIter == 8,
              abs(configuration.guidanceScale - 0.2) < 1e-9,
              abs(configuration.topP - 0.95) < 1e-9,
              abs(configuration.noiseScale - 0.001) < 1e-9,
              configuration.codecDenseFP16,
              configuration.weightLayout == "nemo" else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "unsupported architecture in tts/config.json")
        }

        self.configuration = configuration
        textPadID = tokenizer.padID
        textEOSID = tokenizer.eosID
        let speechWeights = weights.filter {
            $0.key.hasPrefix("tts_model.tts_model.")
                || $0.key.hasPrefix("tts_model.audio_prompt_latents.")
                || $0.key == "tts_model.codec_silence_tokens"
        }
        let store = VoiceChatSpeechWeightStore(
            values: speechWeights,
            quantization: configuration.quantizationPerTensor ?? [:],
            defaultQuantization: configuration.quantization)

        backbone = try VoiceChatSpeechBackbone(store, configuration: configuration)
        head = try VoiceChatMoGHead(store)
        textEncoder = try VoiceChatCharacterEncoder(
            store, tokenizer: tokenizer, configuration: configuration)
        fusion = try VoiceChatGatedFusion(store, configuration: configuration)
        codeEmbedding = try store.matrix("tts_model.tts_model.embed_code.weight")
        beginningEmbedding = try store.dense("tts_model.tts_model.bos_emb")
        nullEmbedding = try store.dense("tts_model.tts_model.null_emb")
        residualCodebooks = try store.dense("tts_model.tts_model.rvq_embs")
        speakerPrompt = try store.dense("tts_model.audio_prompt_latents.Aria")

        guard let silence = speechWeights["tts_model.codec_silence_tokens"] else {
            throw VoiceChatLoadError.unexpectedKeys(["tts_model.codec_silence_tokens"])
        }
        silenceCodes = silence.asType(.int32)
        paddedCodebooks = MLX.concatenated([
            residualCodebooks,
            MLXArray.zeros(
                [configuration.numQuantizers, 1, configuration.latentSize],
                dtype: .float32),
        ], axis: 1)
    }

    public static func load(
        from root: URL,
        tokenizer: VoiceChatTokenizer
    ) throws -> VoiceChatSpeechDecoder {
        let directory = root.lastPathComponent == "tts"
            ? root
            : root.appendingPathComponent("tts")
        let configuration = try VoiceChatSpeechConfiguration.load(from: directory)
        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw VoiceChatLoadError.missingWeights(weightsURL)
        }
        return try VoiceChatSpeechDecoder(
            weights: MLX.loadArrays(url: weightsURL),
            configuration: configuration,
            tokenizer: tokenizer)
    }

    /// MaskGIT codebook assignments per iteration. For the published eight-step
    /// generation config this is [0, 0, 0, 1, 1, 3, 4, 22].
    public static func maskGITAssignmentCounts(
        quantizers: Int = 31,
        iterations: Int = 8
    ) -> [Int] {
        precondition(iterations > 0)
        let counts = (0 ..< iterations).map { index -> Int in
            let rate = Float(index) / Float(iterations)
            let masking = Foundation.pow(
                Double(1 - Foundation.pow(rate, 3)), 1.0 / 3.0)
            return Int(Foundation.ceil(masking * Double(quantizers)))
        }
        return counts.enumerated().map { index, value in
            value - (index + 1 < counts.count ? counts[index + 1] : 0)
        }
    }

    /// Sum the selected RVQ entry from each of the 31 codebooks.
    public func depthSum(_ codes: MLXArray) -> MLXArray {
        let clamped = MLX.minimum(
            MLX.maximum(codes.asType(.int32), MLXArray(Int32(0))),
            MLXArray(Int32(configuration.codebookSize)))
        var output = MLXArray.zeros(
            [codes.dim(0), codes.dim(1), configuration.latentSize],
            dtype: .float32)
        for index in 0 ..< configuration.numQuantizers {
            output = output + paddedCodebooks[index][clamped[0..., 0..., index]]
        }
        return output
    }

    /// Prefill a session's 28 attention caches with the checkpoint's 37-frame
    /// Aria prompt. The final prompt code is the first generation input.
    public func warmup(
        guidance: Bool = true
    ) -> (state: VoiceChatSpeechDecoderState, previousCode: MLXArray) {
        let state = VoiceChatSpeechDecoderState(attention: backbone.makeCache())

        let codes = MLX.broadcast(
            silenceCodes, to: [1, configuration.promptFrames, configuration.numQuantizers])
            + MLXArray(Int32(0))
        codes[0..., 0, 0...] = MLXArray(Int32(Self.speechPad))
        codes[0..., configuration.promptFrames - 1, 0...] =
            MLXArray(Int32(Self.speechPad))

        let shifted = MLX.concatenated([
            MLXArray.full(
                [1, 1, configuration.numQuantizers],
                values: MLXArray(Int32(Self.speechPad)), dtype: .int32),
            codes[0..., 0 ..< (configuration.promptFrames - 1), 0...],
        ], axis: 1)

        let fused = fuse(
            code: shifted,
            token: nil,
            guidance: guidance,
            state: state,
            promptLatents: speakerPrompt,
            beginningAt: configuration.promptFrames - 1,
            conditioning: warmupConditioning(state: state))
        let promptHidden = backbone(fused, cache: state.attention)
        let promptState = state.attention.flatMap { cache in
            [cache.keys, cache.values].compactMap { $0 }
        }
        // MLX is lazy. Evaluating only the returned previous code leaves the
        // 37-frame prompt prefill attached to the first live synthesis step,
        // incorrectly charging session warmup to end-to-end streaming RTF.
        MLX.eval([promptHidden] + promptState)
        return (
            state,
            codes[0..., (configuration.promptFrames - 1)..., 0...])
    }

    /// Generate one 80 ms frame of codec ids.
    public func step(
        state: VoiceChatSpeechDecoderState,
        previousCode: MLXArray,
        textToken: Int,
        parameters: VoiceChatSpeechGenerationParameters = .init()
    ) throws -> MLXArray {
        guard parameters.iterations > 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "MaskGIT iterations must be positive")
        }

        let useGuidance = parameters.guidance > 0
        let conditionedPrevious = condition(previousCode, for: textToken)
        let input = fuse(
            code: conditionedPrevious, token: textToken,
            guidance: useGuidance, state: state)
        let hidden = backbone(input, cache: state.attention)

        var codes = MLXArray.full(
            [1, 1, configuration.numQuantizers],
            values: MLXArray(Int32(configuration.codebookSize)), dtype: .int32)
        // Masked codebooks contribute exact zero vectors. Carry the selected
        // RVQ sum forward instead of gathering and adding all 31 codebooks
        // again before every MaskGIT head pass.
        var codeLatent = MLXArray.zeros(
            [1, 1, configuration.latentSize], dtype: .float32)
        var filled = 0
        for count in Self.maskGITAssignmentCounts(
            quantizers: configuration.numQuantizers,
            iterations: parameters.iterations
        ) where count > 0 {
            var embeddings = codeEmbedding(codeLatent)
            if useGuidance {
                embeddings = MLX.concatenated([
                    embeddings + hidden[0..<1, 0..., 0...],
                    embeddings + hidden[1..., 0..., 0...],
                ], axis: 0)
            } else {
                embeddings = embeddings + hidden
            }

            let (mean, logs) = head.infer(
                embeddings, guidance: parameters.guidance, topP: parameters.topP)
            let latent = mean + MLX.exp(logs)
                * MLXRandom.normal(mean.shape, dtype: mean.dtype)
                * MLXArray(parameters.noise)
            let assignment = Self.assignRVQCodes(
                residualCodebooks: residualCodebooks,
                latent: latent, to: codes, startingAt: filled, count: count,
                retainEmbeddings: filled + count < configuration.numQuantizers)
            codes = assignment.codes
            for selected in assignment.embeddings {
                codeLatent = codeLatent + selected
            }
            filled += count
        }
        return codes
    }

    /// Replace speech control ids with the checkpoint's canonical silence frame,
    /// then convert generated codes into the codec's 512-wide latent space.
    public func latents(for codes: MLXArray) -> MLXArray {
        let silence = MLX.broadcast(silenceCodes, to: codes.shape)
        var controls = MLXArray.zeros(codes.shape, dtype: .bool)
        for value in Self.controlCodes {
            controls = MLX.logicalOr(controls, codes .== MLXArray(Int32(value)))
        }
        return depthSum(MLX.where(controls, silence, codes))
    }

    private func condition(_ previous: MLXArray, for textToken: Int) -> MLXArray {
        if textToken == textEOSID {
            return MLX.broadcast(silenceCodes, to: previous.shape)
        }
        return previous
    }

    private func warmupConditioning(
        state: VoiceChatSpeechDecoderState
    ) -> MLXArray {
        let conditioning = MLXArray.zeros(
            [1, configuration.promptFrames, configuration.hiddenSize],
            dtype: .float32)
        conditioning[0..., (configuration.promptFrames - 2)..<(configuration.promptFrames - 1), 0...] =
            encodedText(textPadID, state: state)
        conditioning[0..., (configuration.promptFrames - 1)..., 0...] =
            encodedText(textEOSID, state: state)
        return conditioning
    }

    private func fuse(
        code: MLXArray,
        token: Int?,
        guidance: Bool,
        state: VoiceChatSpeechDecoderState,
        promptLatents: MLXArray? = nil,
        beginningAt: Int? = nil,
        conditioning: MLXArray? = nil
    ) -> MLXArray {
        var embeddings = codeEmbedding(depthSum(code))
        if let promptLatents {
            let count = promptLatents.dim(1)
            embeddings = MLX.concatenated([
                promptLatents[0..., 0 ..< (count - 1), 0...],
                embeddings[0..., (count - 1)..., 0...],
            ], axis: 1)
        }
        if let beginningAt {
            let addition = MLXArray.zeros(like: embeddings)
            addition[0..., beginningAt, 0...] = beginningEmbedding
            embeddings = embeddings + addition
        }

        let conditional: MLXArray
        if let conditioning {
            precondition(conditioning.shape == [
                1, embeddings.dim(1), configuration.hiddenSize
            ])
            conditional = conditioning
        } else if let token {
            conditional = MLX.broadcast(
                encodedText(token, state: state),
                to: [1, embeddings.dim(1), configuration.hiddenSize])
        } else {
            conditional = MLXArray.zeros(
                [1, embeddings.dim(1), configuration.hiddenSize],
                dtype: .float32)
        }

        if guidance {
            embeddings = MLX.concatenated([embeddings, embeddings], axis: 0)
            let unconditioned = MLX.broadcast(nullEmbedding, to: conditional.shape)
            return fusion(
                audio: embeddings,
                text: MLX.concatenated([conditional, unconditioned], axis: 0))
        }
        return fusion(audio: embeddings, text: conditional)
    }

    private func encodedText(
        _ tokenID: Int,
        state: VoiceChatSpeechDecoderState
    ) -> MLXArray {
        if let cached = state.textEmbeddings[tokenID] { return cached }
        let embedding = textEncoder(tokenID)
        eval(embedding)
        state.textEmbeddings[tokenID] = embedding
        return embedding
    }

    static func assignRVQCodes(
        residualCodebooks: MLXArray,
        latent: MLXArray,
        to initialCodes: MLXArray,
        startingAt start: Int,
        count: Int,
        retainEmbeddings: Bool
    ) -> (codes: MLXArray, embeddings: [MLXArray]) {
        var residual = latent
        let codes = initialCodes
        var embeddings: [MLXArray] = []
        if retainEmbeddings { embeddings.reserveCapacity(count) }
        for index in start ..< (start + count) {
            let codebook = residualCodebooks[index]
            let distances = MLX.sum(codebook.square(), axis: -1)
                - MLXArray(Float(2)) * MLX.matmul(residual, codebook.transposed())
            let selected = MLX.argMin(distances, axis: -1).asType(.int32)
            codes[.ellipsis, index] = selected
            let selectedEmbedding = codebook[selected]
            if retainEmbeddings { embeddings.append(selectedEmbedding) }
            residual = residual - selectedEmbedding
        }
        return (codes, embeddings)
    }
}
