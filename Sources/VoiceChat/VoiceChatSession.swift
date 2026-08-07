import Foundation
import MLX
import MLXLMCommon
import MLXRandom

public struct VoiceChatTextSamplingParameters: Sendable {
    public var temperature: Float
    public var topP: Float
    public var repetitionPenalty: Float
    public var presencePenalty: Float

    public init(
        temperature: Float = 0,
        topP: Float = 1,
        repetitionPenalty: Float = 1,
        presencePenalty: Float = 0
    ) {
        self.temperature = temperature
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
    }
}

public struct VoiceChatFrameEvent: Sendable {
    public let index: Int
    public let textToken: Int
    public let functionToken: Int
    public let text: String
    public let speaking: Bool
    public let audioPositionMilliseconds: Double
    /// Input preparation plus perception-encoder cost attributed to this frame.
    public let perceptionLatencyMilliseconds: Double
    public let decisionLatencyMilliseconds: Double
    public let synthesisLatencyMilliseconds: Double
    /// Live 22.05 kHz mono output for this 80 ms frame.
    public let audio: [Float]
}

public struct VoiceChatSessionSummary: Sendable {
    public let frames: Int
    public let speakingFrames: Int
    public let firstSpeechFrame: Int?
    public let firstSpeechMilliseconds: Double?
    public let perceptionP50Milliseconds: Double
    public let perceptionP95Milliseconds: Double
    public let decisionP50Milliseconds: Double
    public let decisionP95Milliseconds: Double
    public let synthesisP50Milliseconds: Double
    public let synthesisP95Milliseconds: Double
    public let totalP50Milliseconds: Double
    public let totalP95Milliseconds: Double
    public let realTime: Bool
}

/// Stateful audio-in/audio-out VoiceChat conversation.
///
/// Audio is accepted at 16 kHz mono. Every complete 1,280-sample input frame
/// produces one language decision and one 1,764-sample 22.05 kHz model-audio
/// frame on the same 80 ms timeline.
public actor VoiceChatSession {
    public static let inputSampleRate = 16_000
    public static let outputSampleRate = 22_050
    public static let frameMilliseconds = 80
    public static let inputSamplesPerFrame = 1_280
    public static let outputSamplesPerFrame = 1_764
    /// `pushSilence` is a finite-file tail helper, not an unbounded clock.
    public static let maximumSilenceSeconds: Double = 60

    public static let baseSystemPrompt =
        "You are an AI voice assistant developed by NVIDIA. "
        + "Your name is NVIDIA Voice Chat. "
        + "Answer in a spoken, conversational style rather than a written one. "
        + "Do not repeat the same sentence over and over again."
    /// Neutral by default: a greeting instruction changes measured turn onset.
    public static let defaultSystemPrompt = baseSystemPrompt
    public static let greetingSystemPrompt =
        baseSystemPrompt + " Start the conversation by greeting the user."

    // Three stride-2 causal subsampling stages have a 15-mel-frame receptive
    // field (140 ms of history). The conformer now owns its longer history in
    // bounded per-layer caches, so two preceding 80 ms input frames are enough
    // for the recomputed frontend window.
    static let frontendContextFrames = 2
    static let codecContextFrames = 8

    private let model: VoiceChatModel
    private let sampling: VoiceChatTextSamplingParameters
    private let speechParameters: VoiceChatSpeechGenerationParameters
    private var languageCache: [KVCache]
    private var encoderState: VoiceChatEncoderStreamState
    private var inputAudio: [Float] = []
    private var completedFrames = 0
    private var previousText: Int
    private var previousFunction: Int
    private var speechState: VoiceChatSpeechDecoderState?
    private var previousSpeechCode: MLXArray?
    private var emittedTokens: [Int] = []
    private var generatedCodes: [MLXArray] = []
    private var recordedEvents: [VoiceChatFrameEvent] = []
    private var forcedTurnFrame: Int?

    private init(
        model: VoiceChatModel,
        sampling: VoiceChatTextSamplingParameters,
        speechParameters: VoiceChatSpeechGenerationParameters
    ) {
        self.model = model
        self.sampling = sampling
        self.speechParameters = speechParameters
        self.languageCache = model.languageModel.newCache()
        self.encoderState = model.perception.encoder.newStreamState()
        self.previousText = model.tokenizer.padID
        self.previousFunction = model.tokenizer.padID
    }

    static func create(
        model: VoiceChatModel,
        systemPrompt: String,
        sampling: VoiceChatTextSamplingParameters,
        speechParameters: VoiceChatSpeechGenerationParameters
    ) async throws -> VoiceChatSession {
        try validate(sampling: sampling, speech: speechParameters)
        let session = VoiceChatSession(
            model: model, sampling: sampling,
            speechParameters: speechParameters)
        try await session.initialize(systemPrompt: systemPrompt)
        return session
    }

    private func initialize(systemPrompt: String) throws {
        try prime(systemPrompt)
        let warmup = model.speechDecoder.warmup(
            guidance: speechParameters.guidance > 0)
        speechState = warmup.state
        previousSpeechCode = warmup.previousCode
        eval(warmup.previousCode)
    }

    static func validate(
        sampling: VoiceChatTextSamplingParameters,
        speech: VoiceChatSpeechGenerationParameters
    ) throws {
        guard sampling.temperature.isFinite, sampling.temperature >= 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "text temperature must be finite and non-negative")
        }
        guard sampling.topP.isFinite, sampling.topP > 0, sampling.topP <= 1 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "text topP must be finite and in (0, 1]")
        }
        guard sampling.repetitionPenalty.isFinite,
              sampling.repetitionPenalty > 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "text repetitionPenalty must be finite and positive")
        }
        guard sampling.presencePenalty.isFinite else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "text presencePenalty must be finite")
        }
        guard speech.guidance.isFinite, speech.guidance >= 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "speech guidance must be finite and non-negative")
        }
        guard speech.topP.isFinite, speech.topP > 0, speech.topP <= 1 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "speech topP must be finite and in (0, 1]")
        }
        guard speech.noise.isFinite, speech.noise >= 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "speech noise must be finite and non-negative")
        }
        guard speech.iterations > 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "MaskGIT iterations must be positive")
        }
    }

    /// Force the model to open its turn at an audio-frame index if it has not
    /// emitted BOS itself. Useful for controlled evaluations, not normal chat.
    public func forceTurn(atFrame frame: Int?) {
        forcedTurnFrame = frame.map { max(0, $0) }
    }

    /// Feed 16 kHz mono samples. Partial 80 ms frames remain buffered.
    @discardableResult
    public func pushAudio(_ samples: [Float]) throws -> [VoiceChatFrameEvent] {
        guard samples.allSatisfy(\.isFinite) else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "input audio contains non-finite samples")
        }
        guard !samples.isEmpty else { return [] }
        inputAudio.append(contentsOf: samples)

        let perceptionStart = DispatchTime.now().uptimeNanoseconds
        let contextStart = max(
            0, completedFrames - Self.frontendContextFrames)
        let startSample = contextStart * Self.inputSamplesPerFrame
        let window = Array(inputAudio[startSample...])
        // NeMo's centred STFT needs enough signal to reveal a stable mel frame.
        guard window.count >= Self.inputSampleRate / 10 else { return [] }

        let mel = model.transcriber.logMel(window)
        let subsampled = model.perception.encoder.preEncode(mel)
        let alreadyProduced = completedFrames - contextStart
        let complete = inputAudio.count / Self.inputSamplesPerFrame
        let limit = min(subsampled.dim(1), complete - contextStart)
        guard limit > alreadyProduced else { return [] }

        var embeddings: [MLXArray] = []
        embeddings.reserveCapacity(limit - alreadyProduced)
        for offset in alreadyProduced ..< limit {
            let hidden = model.perception.encoder.stream(
                subsampled[0..., offset..<(offset + 1), 0...],
                state: encoderState)
            let embedding = model.perception.modalityProj(hidden).asType(.float32)
            MLX.eval([embedding] + encoderState.evaluatedArrays)
            embeddings.append(embedding)
        }
        let perceptionPerFrame = milliseconds(since: perceptionStart)
            / Double(limit - alreadyProduced)

        var events: [VoiceChatFrameEvent] = []
        for embedding in embeddings {
            let event = try advance(
                audioEmbedding: embedding,
                record: true,
                forceSilent: completedFrames == 0,
                audioMilliseconds: Double(completedFrames * Self.frameMilliseconds),
                perceptionLatencyMilliseconds: perceptionPerFrame)
            completedFrames += 1
            events.append(event)
        }
        return events
    }

    /// Feed silent tail frames so an answer that starts near the end of the
    /// user's clip can finish instead of being cut mid-sentence.
    @discardableResult
    public func pushSilence(seconds: Double) throws -> [VoiceChatFrameEvent] {
        let sampleCount = try Self.silenceSampleCount(seconds: seconds)
        return try pushAudio(
            [Float](repeating: 0, count: sampleCount))
    }

    static func silenceSampleCount(seconds: Double) throws -> Int {
        guard seconds.isFinite,
              seconds >= 0,
              seconds <= Self.maximumSilenceSeconds else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "silence duration must be finite and between 0 and "
                    + "\(Int(Self.maximumSilenceSeconds)) seconds")
        }
        let sampleCount = (seconds * Double(Self.inputSampleRate)).rounded()
        return Int(sampleCount)
    }

    public func reply() -> String {
        let spoken = emittedTokens.filter {
            !model.tokenizer.specialIDs.contains($0)
        }
        return model.tokenizer.decode(spoken).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func userTranscript() -> String {
        model.transcriber.transcribe(inputAudio)
    }

    public func events() -> [VoiceChatFrameEvent] {
        recordedEvents
    }

    /// Exact offline decode of every generated frame. This is the artifact to
    /// save or compare; per-event audio uses a bounded causal codec window.
    public func renderedAudio() -> [Float] {
        guard !generatedCodes.isEmpty else { return [] }
        let codes = MLX.concatenated(generatedCodes, axis: 1)
        let waveform = model.codec.decode(
            latents: model.speechDecoder.latents(for: codes))[0]
        eval(waveform)
        return waveform.asArray(Float.self)
    }

    public func turnTakingLatencyMilliseconds(
        userStoppedAtMilliseconds: Double
    ) -> Double? {
        recordedEvents.first {
            $0.speaking
                && $0.audioPositionMilliseconds >= userStoppedAtMilliseconds
        }.map { $0.audioPositionMilliseconds - userStoppedAtMilliseconds }
    }

    public func summary() -> VoiceChatSessionSummary {
        let speaking = recordedEvents.filter(\.speaking)
        let perception = recordedEvents.map(\.perceptionLatencyMilliseconds).sorted()
        let decision = recordedEvents.map(\.decisionLatencyMilliseconds).sorted()
        let synthesis = recordedEvents.map(\.synthesisLatencyMilliseconds).sorted()
        let total = recordedEvents.map {
            $0.perceptionLatencyMilliseconds
                + $0.decisionLatencyMilliseconds
                + $0.synthesisLatencyMilliseconds
        }.sorted()

        func percentile(_ values: [Double], _ fraction: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            return values[min(values.count - 1, Int(Double(values.count) * fraction))]
        }

        let totalP95 = percentile(total, 0.95)
        return VoiceChatSessionSummary(
            frames: recordedEvents.count,
            speakingFrames: speaking.count,
            firstSpeechFrame: speaking.first?.index,
            firstSpeechMilliseconds: speaking.first?.audioPositionMilliseconds,
            perceptionP50Milliseconds: percentile(perception, 0.5),
            perceptionP95Milliseconds: percentile(perception, 0.95),
            decisionP50Milliseconds: percentile(decision, 0.5),
            decisionP95Milliseconds: percentile(decision, 0.95),
            synthesisP50Milliseconds: percentile(synthesis, 0.5),
            synthesisP95Milliseconds: percentile(synthesis, 0.95),
            totalP50Milliseconds: percentile(total, 0.5),
            totalP95Milliseconds: totalP95,
            realTime: totalP95 < Double(Self.frameMilliseconds))
    }

    private func prime(_ systemPrompt: String) throws {
        let ids = [model.tokenizer.bosID]
            + model.tokenizer.encode(systemPrompt)
            + [model.tokenizer.eosID]
        let embeddings = model.languageModel.embed(MLXArray(ids))
            .expandedDimensions(axis: 0).asType(.float32)
        for position in 0 ..< ids.count {
            _ = try advance(
                audioEmbedding: embeddings[0..., position..<(position + 1), 0...],
                record: false, forceSilent: true, audioMilliseconds: 0,
                perceptionLatencyMilliseconds: 0)
        }
    }

    private func advance(
        audioEmbedding: MLXArray,
        record: Bool,
        forceSilent: Bool,
        audioMilliseconds: Double,
        perceptionLatencyMilliseconds: Double
    ) throws -> VoiceChatFrameEvent {
        var feedback = forceSilent ? model.tokenizer.padID : previousText
        if let forcedTurnFrame,
           record,
           recordedEvents.count == forcedTurnFrame,
           !emittedTokens.contains(model.tokenizer.bosID) {
            feedback = model.tokenizer.bosID
        }

        let decisionStart = DispatchTime.now().uptimeNanoseconds
        // NeMo's add fusion weights are text=1, user audio=1, function=2.
        let fused = audioEmbedding
            + model.languageModel.embed(MLXArray([feedback]))
                .expandedDimensions(axis: 0)
            + MLXArray(Float(2))
                * model.languageModel.embed(MLXArray([previousFunction]))
                    .expandedDimensions(axis: 0)
        let output = model.languageModel.call(
            embeddings: fused, cache: languageCache)
        let textToken = sampleTextToken(output.logits[0, -1, 0...])
        let functionToken: Int
        if let functionHead = model.languageModel.functionHead {
            functionToken = MLX.argMax(
                functionHead(output.hidden[0, -1, 0...]), axis: -1)
                .item(Int.self)
        } else {
            functionToken = model.tokenizer.padID
        }
        eval(output.logits)
        let decisionLatency = milliseconds(since: decisionStart)

        previousText = textToken
        previousFunction = functionToken

        let recordedToken = forceSilent ? model.tokenizer.padID : textToken
        var synthesisLatency = 0.0
        var audio: [Float] = []
        if record {
            emittedTokens.append(recordedToken)
            guard let speechState, let previousSpeechCode else {
                throw VoiceChatGenerationError.speechDecoderNotPrimed
            }
            let synthesisStart = DispatchTime.now().uptimeNanoseconds
            let code = try model.speechDecoder.step(
                state: speechState,
                previousCode: previousSpeechCode,
                textToken: recordedToken,
                parameters: speechParameters)
            eval(code)
            self.previousSpeechCode = code
            generatedCodes.append(code)
            audio = liveAudioFrame()
            synthesisLatency = milliseconds(since: synthesisStart)
        }

        let event = VoiceChatFrameEvent(
            index: recordedEvents.count,
            textToken: recordedToken,
            functionToken: functionToken,
            text: model.tokenizer.decode([recordedToken], skipSpecialTokens: false),
            speaking: !model.tokenizer.specialIDs.contains(recordedToken),
            audioPositionMilliseconds: audioMilliseconds,
            perceptionLatencyMilliseconds: perceptionLatencyMilliseconds,
            decisionLatencyMilliseconds: decisionLatency,
            synthesisLatencyMilliseconds: synthesisLatency,
            audio: audio)
        if record { recordedEvents.append(event) }
        return event
    }

    private func sampleTextToken(_ inputLogits: MLXArray) -> Int {
        let logits = inputLogits.asType(.float32)
        let greedy = MLX.argMax(logits, axis: -1).item(Int.self)
        if sampling.temperature == 0
            || model.tokenizer.specialIDs.contains(greedy) {
            return greedy
        }

        var adjusted = logits
        if !emittedTokens.isEmpty
            && (sampling.repetitionPenalty != 1 || sampling.presencePenalty != 0) {
            for token in Set(emittedTokens)
            where !model.tokenizer.specialIDs.contains(token) {
                let value = adjusted[token].item(Float.self)
                let repeated = sampling.repetitionPenalty == 1
                    ? value
                    : (value > 0
                        ? value / sampling.repetitionPenalty
                        : value * sampling.repetitionPenalty)
                adjusted[token] = MLXArray(repeated - sampling.presencePenalty)
            }
        }

        adjusted = adjusted / MLXArray(sampling.temperature)
        adjusted = voiceChatTopPFilter(adjusted, topP: sampling.topP)
        return MLX.argMax(
            adjusted + MLXRandom.gumbel(adjusted.shape), axis: -1)
            .item(Int.self)
    }

    private func liveAudioFrame() -> [Float] {
        let start = max(0, generatedCodes.count - Self.codecContextFrames)
        let context = MLX.concatenated(Array(generatedCodes[start...]), axis: 1)
        let waveform = model.codec.decode(
            latents: model.speechDecoder.latents(for: context))[0]
        let count = waveform.dim(0)
        let frame = waveform[(count - Self.outputSamplesPerFrame)..<count]
        eval(frame)
        return frame.asArray(Float.self)
    }

    private func milliseconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }
}
