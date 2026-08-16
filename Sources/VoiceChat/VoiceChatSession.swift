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
    /// Completed JSON tool-call payload from the function channel. This is set
    /// only on the frame that emits the end-of-tool-call token.
    public let functionCall: String?
    public let text: String
    /// Complete append-only user transcript so far when streaming
    /// transcription was enabled for the session; otherwise `nil`.
    public let userTranscript: String?
    /// First-prediction RNN-T activity for turn-taking and confirmation logic.
    /// `nil` when the session has neither captions nor RNN-T control enabled.
    public let rnntIsBlank: Bool?
    /// Whether RNN-T turn-taking overrode the model's text-channel decision.
    public let turnTakingAction: VoiceChatTurnTakingAction
    public let speaking: Bool
    public let audioPositionMilliseconds: Double
    /// Input preparation plus perception-encoder cost attributed to this frame.
    public let perceptionLatencyMilliseconds: Double
    public let decisionLatencyMilliseconds: Double
    public let synthesisLatencyMilliseconds: Double
    /// Live 22.05 kHz mono output for this 80 ms frame.
    public let audio: [Float]
    /// Whether a live driver should enqueue `audio` for playback. Deferred
    /// microphone frames are replayed through the shared model timeline after
    /// tool work completes. Their already-elapsed idle audio must not be
    /// played a second time, while any newly generated assistant audio remains
    /// observable.
    public let playbackRequired: Bool
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

struct VoiceChatDeferredMicrophoneStatistics: Sendable, Equatable {
    let bufferedFrames: Int
    let replayedFrames: Int
    let droppedFrames: Int
    let pendingSpeechCacheFrames: Int
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

    public static let personaSystemPrompt =
        "You are Soniqo, an AI voice assistant. "
        + "Your name is Soniqo. "
        + "Answer in a spoken, conversational style rather than a written one. "
        + "Do not repeat the same sentence over and over again."
    public static let baseSystemPrompt =
        personaSystemPrompt + " "
        + "You can answer questions and converse, but you cannot access apps, "
        + "calendars, reminders, accounts, devices, or external services. "
        + "Never claim to schedule, set, send, call, purchase, or complete an "
        + "external action. If asked to do one, clearly say that you cannot do "
        + "it, do not ask for confirmation, and briefly explain what the user "
        + "can do instead."
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
    static let idleSpeechBatchFrames = 8
    static let promptPrefillChunkTokens = 64
    /// Every trained native call starts with this exact JSON prefix. Once the
    /// function head has independently selected `<SPECIAL_20>`, replaying the
    /// deterministic syntax as one causal prefill preserves model state while
    /// avoiding serial 11B decode/projection steps. The literal protocol
    /// marker is part of the trained payload and must remain paired with the
    /// model-generated closing marker. Tool selection and all argument values
    /// remain model-generated.
    static let functionCallJSONPrefix = #"<TOOLCALL>[{"name":""#
    static let maximumFunctionCallTokens = 256
    /// NVIDIA's server path permits 2,000 asynchronous positions. The local
    /// 11B runtime uses a smaller token and wall-clock bound so a malformed
    /// PAD-only call cannot leave the assistant permanently occupied.
    static let maximumAsynchronousFunctionCallSteps = 256
    static let maximumAsynchronousFunctionCallSeconds: Double = 8
    static let maximumFunctionResponseTokens = 512
    /// A tool result is caller-provided and therefore fully known before phase
    /// two starts. Prefill it in bounded causal chunks instead of paying one
    /// 11B decode and vocabulary projection per response token. Keeping chunks
    /// bounded avoids a large temporary attention matrix for long MCP results.
    static let functionResponsePrefillChunkTokens = 16
    static let maximumAsynchronousFunctionResponseSteps =
        maximumFunctionResponseTokens
    static let maximumAsynchronousFunctionResponseSeconds: Double = 8
    /// Retain two frames before the first RNN-T activity so the language model
    /// receives the onset of speech that began while the function channel was
    /// busy. Long provider silence is compressed after a bounded tail instead
    /// of being replayed as seconds of duplicate idle model positions.
    static let deferredMicrophonePreRollFrames = 2
    static let deferredMicrophoneTrailingBlankFrames = 16
    /// Replay at most one deferred modality embedding per live microphone
    /// callback. Draining the complete queue synchronously can turn one 80 ms
    /// input tick into a multi-hundred-millisecond actor stall, which then
    /// overflows the microphone buffer and loses more speech.
    static let maximumDeferredMicrophoneReplayFramesPerPush = 1
    /// This bounds evaluated modality embeddings, not raw microphone audio.
    /// At the checkpoint's 4,096-wide Float32 projection, 30 seconds is about
    /// 6 MiB and covers the longest useful spoken interruption without making
    /// an unresponsive provider an unbounded memory sink.
    static let maximumDeferredMicrophoneFrames = 375

    private struct DeferredMicrophoneFrame {
        let embedding: MLXArray
        let userTranscript: String?
        let rnntIsBlank: Bool?
        let rnntHasLexicalToken: Bool
        let audioMilliseconds: Double
    }

    private let model: VoiceChatModel
    private let sampling: VoiceChatTextSamplingParameters
    private var speechParameters: VoiceChatSpeechGenerationParameters
    private let streamUserTranscript: Bool
    private let functionCallingEnabled: Bool
    private let functionStartID: Int?
    private let functionEndID: Int?
    private let functionResponseEndID: Int?
    private let functionCallPrefixTokens: [Int]
    /// Dequantized [PAD, `<SPECIAL_20>`] rows, about 36 KB for this model.
    /// The full function head is 518 MB and remains packed/dormant until this
    /// exact necessary condition says the start token may be the global argmax.
    private let functionDetectionRows: MLXArray?
    private var languageCache: [KVCache]
    private var encoderState: VoiceChatEncoderStreamState
    private var transcriptionState: VoiceChatTranscriber.StreamState?
    private var turnTakingState: VoiceChatRNNTTurnTakingState
    private var streamingUserTranscript = ""
    private var inputAudio: [Float] = []
    private var completedFrames = 0
    private var previousText: Int
    private var previousFunction: Int
    private var collectingFunctionCall = false
    private var functionCallTokens: [Int] = []
    private var functionCallPrefixPending = false
    private let functionHeadScheduler = VoiceChatFunctionHeadScheduler()
    private var functionProbeFrames = 0
    private var functionFullVerificationFrames = 0
    private var functionOpenCallFrames = 0
    private var asynchronousFunctionCallSteps = 0
    private var asynchronousFunctionResponseSteps = 0
    private var asynchronousFunctionCallTimeouts = 0
    private var asynchronousFunctionResponseTimeouts = 0
    private var asynchronousFunctionCallInterruptions = 0
    private var microphoneFramesDuringAsynchronousFunctionWork = 0
    private var functionVerificationWinners: [Int: Int] = [:]
    private var deferredFunctionStartHidden: MLXArray?
    private var functionSilenceEmbedding: MLXArray?
    private var functionFastPathTask: Task<Void, Never>?
    private var functionFastPathProgressWaiters:
        [CheckedContinuation<Void, Never>] = []
    private var functionFastPathLastProgressNanoseconds: UInt64?
    private var functionFastPathInterrupted = false
    private var functionFastPathSuppressedForTesting = false
    private var deferredMicrophoneFrames: [DeferredMicrophoneFrame] = []
    private var deferredMicrophonePreRoll: [DeferredMicrophoneFrame] = []
    private var deferredMicrophoneCaptureActive = false
    private var deferredMicrophoneTrailingBlankCount = 0
    private var deferredMicrophoneDroppedFrames = 0
    private var replayedDeferredMicrophoneFrames = 0
    /// Idle EAR-TTS positions created while deferred microphone embeddings are
    /// replayed. Keep them separate so replay cannot trigger an eight-frame
    /// cache flush inside a live 80 ms capture callback.
    private var deferredReplayPendingIdleSpeechFrames = 0
    private var consecutiveFunctionInterruptionSpeechFrames = 0
    private var asynchronousFunctionResponsePrefillBatches = 0
    private var functionCallDecodeStartedAtNanoseconds: UInt64?
    private var functionCallDecodeSteps = 0
    private var functionCallModelMilliseconds = 0.0
    private var functionCallSpeechCacheMilliseconds = 0.0
    private var lastFunctionCallDecodeMetrics:
        VoiceChatFunctionCallDecodeMetrics?
    private var functionResponseStartedAtNanoseconds: UInt64?
    private var functionResponseSteps = 0
    private var functionResponsePrefillBatches = 0
    private var functionResponseLanguageCacheMilliseconds = 0.0
    private var functionResponseSpeechCacheMilliseconds = 0.0
    private var lastFunctionResponseMetrics: VoiceChatFunctionResponseMetrics?
    private var pendingCompletedFunctionCall: String?
    private var awaitingFunctionResponse = false
    private var awaitingPostFunctionAssistantTurn = false
    private var postFunctionAssistantTurnStarted = false
    /// A confirmation-required result must produce a spoken model turn before
    /// the same write can be proposed again. Otherwise the function head may
    /// immediately re-enter SOTC on the result frame and silently loop without
    /// ever giving the user an opportunity to confirm.
    private var functionStartGate = VoiceChatFunctionStartGate()
    private var forcedFunctionTokens: [Int] = []
    private var forcedFunctionTokenIndex = 0
    private var injectingFunctionResponse = false
    private var speechState: VoiceChatSpeechDecoderState?
    private var previousSpeechCode: MLXArray?
    private var speechTurnState: VoiceChatSpeechTurnState
    private var pendingIdleSpeechFrames = 0
    private var emittedTokens: [Int] = []
    private var generatedCodes: [MLXArray] = []
    private var recordedEvents: [VoiceChatFrameEvent] = []
    private var forcedTurnFrame: Int?

    private init(
        model: VoiceChatModel,
        sampling: VoiceChatTextSamplingParameters,
        speechParameters: VoiceChatSpeechGenerationParameters,
        streamUserTranscript: Bool,
        turnTaking: VoiceChatTurnTakingParameters,
        functionCallingEnabled: Bool
    ) {
        self.model = model
        self.sampling = sampling
        self.speechParameters = speechParameters
        self.streamUserTranscript = streamUserTranscript
        self.functionCallingEnabled = functionCallingEnabled
        self.functionStartID = functionCallingEnabled
            ? model.tokenizer.tokenID("<SPECIAL_20>")
            : nil
        self.functionEndID = functionCallingEnabled
            ? model.tokenizer.tokenID("<SPECIAL_21>")
            : nil
        self.functionResponseEndID = functionCallingEnabled
            ? model.tokenizer.tokenID("<SPECIAL_22>")
            : nil
        self.functionCallPrefixTokens = functionCallingEnabled
            ? model.tokenizer.encode(Self.functionCallJSONPrefix)
            : []
        if functionCallingEnabled,
           let functionHead = model.languageModel.functionHead,
           let functionStartID = self.functionStartID
        {
            self.functionDetectionRows = functionHead.outputRows(MLXArray([
                model.tokenizer.padID,
                functionStartID,
            ])).asType(.float32)
        } else {
            self.functionDetectionRows = nil
        }
        self.languageCache = model.languageModel.newCache()
        self.encoderState = model.perception.encoder.newStreamState()
        self.transcriptionState = streamUserTranscript || turnTaking.enabled
            ? model.transcriber.makeStreamState()
            : nil
        self.turnTakingState = VoiceChatRNNTTurnTakingState(
            parameters: turnTaking)
        self.speechTurnState = VoiceChatSpeechTurnState(
            compactExtendedPads: speechParameters.realtimeIdleOptimization)
        self.previousText = model.tokenizer.padID
        self.previousFunction = model.tokenizer.padID
        if let functionDetectionRows {
            eval(functionDetectionRows)
        }
    }

    static func create(
        model: VoiceChatModel,
        systemPrompt: String,
        sampling: VoiceChatTextSamplingParameters,
        speechParameters: VoiceChatSpeechGenerationParameters,
        streamUserTranscript: Bool,
        turnTaking: VoiceChatTurnTakingParameters,
        functionCallingEnabled: Bool
    ) async throws -> VoiceChatSession {
        try validate(
            sampling: sampling,
            speech: speechParameters,
            turnTaking: turnTaking)
        let session = VoiceChatSession(
            model: model, sampling: sampling,
            speechParameters: speechParameters,
            streamUserTranscript: streamUserTranscript,
            turnTaking: turnTaking,
            functionCallingEnabled: functionCallingEnabled)
        if functionCallingEnabled {
            guard model.languageModel.functionHead != nil,
                  model.tokenizer.tokenID("<SPECIAL_20>") == 20,
                  model.tokenizer.tokenID("<SPECIAL_21>") == 21,
                  model.tokenizer.tokenID("<SPECIAL_22>") == 22,
                  !session.functionCallPrefixTokens.isEmpty,
                  model.tokenizer.decode(
                    session.functionCallPrefixTokens,
                    skipSpecialTokens: false) == Self.functionCallJSONPrefix
            else {
                throw VoiceChatLoadError.unexpectedKeys([
                    "function-calling head, protocol tokens, or JSON prefix tokenizer parity",
                ])
            }
        }
        try await session.initialize(systemPrompt: systemPrompt)
        return session
    }

    private func initialize(systemPrompt: String) throws {
        try prime(systemPrompt)
        if functionCallingEnabled {
            // NVIDIA encodes one second of zero PCM once and reuses the first
            // perception frame while generating function tokens at full LLM
            // speed. A literal zero vector or a frozen microphone frame does
            // not reproduce the checkpoint's training-time silence input.
            let silence = [Float](
                repeating: 0, count: Self.inputSampleRate)
            let encoded = model.perception(
                model.transcriber.logMel(silence)).asType(.float32)
            functionSilenceEmbedding = encoded[0..., 0 ..< 1, 0...]
            if let functionSilenceEmbedding {
                eval(functionSilenceEmbedding)
            }
        }
        let warmup = model.speechDecoder.warmup(
            guidance: speechParameters.guidance > 0,
            recentContextFrames: speechParameters.recentContextFrames)
        speechState = warmup.state
        previousSpeechCode = warmup.previousCode
        eval(warmup.previousCode)
    }

    static func validate(
        sampling: VoiceChatTextSamplingParameters,
        speech: VoiceChatSpeechGenerationParameters,
        turnTaking: VoiceChatTurnTakingParameters = .modelNative
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
        if let recentContextFrames = speech.recentContextFrames,
           recentContextFrames <= 0 {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "recent EAR-TTS context frames must be positive")
        }
        if turnTaking.enabled {
            guard turnTaking.endOfUtteranceFrames > 0,
                  turnTaking.beginOfUtteranceFrames > 0,
                  turnTaking.functionCallEndOfUtteranceFrames.map({ $0 > 0 })
                    ?? true,
                  turnTaking.minimumSpeechFrames > 0,
                  turnTaking.firstTurnMinimumSpeechFrames > 0,
                  turnTaking.noiseResetFrames > 0 else {
                throw VoiceChatGenerationError.invalidSpeechConfiguration(
                    "RNN-T turn-taking frame thresholds must be positive")
            }
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

        var frames: [(
            embedding: MLXArray,
            userTranscript: String?,
            rnntIsBlank: Bool?,
            rnntHasLexicalToken: Bool
        )] = []
        frames.reserveCapacity(limit - alreadyProduced)
        for offset in alreadyProduced ..< limit {
            let hidden = model.perception.encoder.stream(
                subsampled[0..., offset..<(offset + 1), 0...],
                state: encoderState)
            var transcript: String?
            var rnntIsBlank: Bool?
            var rnntHasLexicalToken = false
            if var state = transcriptionState {
                let result = model.transcriber.transcribeStreamingFrame(
                    hidden[0], state: &state)
                transcriptionState = state
                streamingUserTranscript = result.transcript
                transcript = streamUserTranscript ? result.transcript : nil
                rnntIsBlank = result.isBlank
                rnntHasLexicalToken = result.hasLexicalToken
            }
            let embedding = model.perception.modalityProj(hidden).asType(.float32)
            MLX.eval([embedding] + encoderState.evaluatedArrays)
            frames.append((
                embedding,
                transcript,
                rnntIsBlank,
                rnntHasLexicalToken))
        }
        let perceptionPerFrame = milliseconds(since: perceptionStart)
            / Double(limit - alreadyProduced)

        var events = try replayDeferredMicrophoneInputIfReady()
        let synchronizedDeferredSpeechCache = events.isEmpty
            ? try synchronizeDeferredReplaySpeechCacheIfReady()
            : false
        let deferredTimelineWorkThisPush =
            !events.isEmpty || synchronizedDeferredSpeechCache
        for frame in frames {
            if Self.shouldDeferLiveMicrophoneFrame(
                fastPathRunning: functionFastPathTask != nil,
                hasFastPathWork: hasFunctionFastPathWork,
                awaitingFunctionResponse: awaitingFunctionResponse,
                deferredTimelineWorkThisPush: deferredTimelineWorkThisPush,
                hasDeferredInput: !deferredMicrophoneFrames.isEmpty
                    || deferredReplayPendingIdleSpeechFrames > 0)
            {
                deferMicrophoneFrame(DeferredMicrophoneFrame(
                    embedding: frame.embedding,
                    userTranscript: frame.userTranscript,
                    rnntIsBlank: frame.rnntIsBlank,
                    rnntHasLexicalToken: frame.rnntHasLexicalToken,
                    audioMilliseconds: Double(
                        completedFrames * Self.frameMilliseconds)))
                let event = functionMicrophoneEvent(
                    perceptionLatencyMilliseconds: perceptionPerFrame,
                    userTranscript: frame.userTranscript,
                    rnntIsBlank: frame.rnntIsBlank,
                    rnntHasLexicalToken: frame.rnntHasLexicalToken)
                completedFrames += 1
                events.append(event)
                continue
            }
            let event = try advance(
                audioEmbedding: frame.embedding,
                record: true,
                forceSilent: completedFrames == 0,
                audioMilliseconds: Double(completedFrames * Self.frameMilliseconds),
                perceptionLatencyMilliseconds: perceptionPerFrame,
                userTranscript: frame.userTranscript,
                rnntIsBlank: frame.rnntIsBlank,
                rnntHasLexicalToken: frame.rnntHasLexicalToken)
            completedFrames += 1
            events.append(event)
            startFunctionFastPathIfNeeded()
        }
        return events
    }

    /// Assist at most one bounded unit of stalled function work.
    ///
    /// Actor reentrancy keeps microphone capture possible, but synthetic or
    /// finite-input callers can submit frames faster than the 80 ms live clock.
    /// Such callers may use this helper to await one progress boundary after a
    /// genuine stall. The live microphone path deliberately never waits here.
    public func assistFunctionFastPathIfStalled(
        maximumStallMilliseconds: Double = 160
    ) async {
        guard functionFastPathTask != nil,
              hasFunctionFastPathWork,
              let lastProgress = functionFastPathLastProgressNanoseconds,
              Self.functionFastPathNeedsProgressAssist(
                fastPathRunning: true,
                hasFastPathWork: true,
                lastProgressNanoseconds: lastProgress,
                nowNanoseconds: DispatchTime.now().uptimeNanoseconds,
                maximumStallMilliseconds: maximumStallMilliseconds)
        else { return }
        await withCheckedContinuation { continuation in
            functionFastPathProgressWaiters.append(continuation)
        }
    }

    static func functionFastPathNeedsProgressAssist(
        fastPathRunning: Bool,
        hasFastPathWork: Bool,
        lastProgressNanoseconds: UInt64?,
        nowNanoseconds: UInt64,
        maximumStallMilliseconds: Double
    ) -> Bool {
        guard fastPathRunning, hasFastPathWork,
              maximumStallMilliseconds >= 0,
              let lastProgressNanoseconds,
              nowNanoseconds >= lastProgressNanoseconds else { return false }
        return Double(nowNanoseconds - lastProgressNanoseconds) / 1_000_000
            >= maximumStallMilliseconds
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
        if transcriptionState != nil {
            return streamingUserTranscript
        }
        return model.transcriber.transcribe(inputAudio)
    }

    /// Change only the iterative EAR-TTS refinement budget for future model
    /// frames. The live CLI uses this as a reversible realtime safeguard.
    public func setSpeechIterations(_ iterations: Int) throws {
        guard iterations > 0 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "MaskGIT iterations must be positive")
        }
        speechParameters.iterations = iterations
    }

    public func speechIterations() -> Int {
        speechParameters.iterations
    }

    /// Queue a completed tool result for the model's function channel. The
    /// caller supplies one compact JSON value; VoiceChat adds the training
    /// protocol wrapper and silences normal speech while it is injected.
    public func injectFunctionResponse(
        _ responseJSON: String,
        requireAssistantReplyBeforeNextFunctionCall: Bool = false
    ) throws {
        guard functionCallingEnabled else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "function calling is not enabled for this session")
        }
        guard Self.functionResponseInjectionAvailable(
            collectingCall: collectingFunctionCall,
            injectingResponse: injectingFunctionResponse,
            forcedTokenIndex: forcedFunctionTokenIndex,
            forcedTokenCount: forcedFunctionTokens.count) else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "another function-channel operation is already active")
        }
        guard responseJSON.unicodeScalars.allSatisfy({ $0.isASCII }) else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "function responses must be ASCII JSON")
        }
        guard let data = responseJSON.data(using: .utf8),
              (try? JSONSerialization.jsonObject(with: data)) != nil else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "function response is not valid JSON")
        }

        let wrapped = "<TOOL_RESPONSE>[\(responseJSON)]</TOOL_RESPONSE>"
        var tokens = model.tokenizer.encode(wrapped)
        guard let functionResponseEndID else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "function response marker is unavailable")
        }
        tokens.append(functionResponseEndID)
        guard tokens.count > 1,
              tokens.count <= Self.maximumFunctionResponseTokens else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "function response exceeds \(Self.maximumFunctionResponseTokens) tokens")
        }
        forcedFunctionTokens = tokens
        forcedFunctionTokenIndex = 0
        functionStartGate.requireAssistantTurn(
            requireAssistantReplyBeforeNextFunctionCall)
        beginFunctionResponseMetrics()
        deferredFunctionStartHidden = nil
        awaitingFunctionResponse = false
        injectingFunctionResponse = true
        startFunctionFastPathIfNeeded()
    }

    /// Never splice a response into a native call that is still being decoded.
    static func functionResponseInjectionAvailable(
        collectingCall: Bool,
        injectingResponse: Bool,
        forcedTokenIndex: Int,
        forcedTokenCount: Int
    ) -> Bool {
        !collectingCall
            && !injectingResponse
            && forcedTokenIndex >= forcedTokenCount
    }

    /// With RNN-T turn taking, a tool-start candidate is cheap to remember but
    /// expensive to verify against the 131k-row function head. A model-native
    /// BOS is the learned end-of-turn signal and may arrive before the hard
    /// RNN-T fallback; either signal is safe to use for final verification.
    static func functionStartCommitReady(
        parameters: VoiceChatTurnTakingParameters,
        speechConfirmed: Bool,
        consecutiveBlankFrames: Int,
        rnntIsBlank: Bool?,
        proposedTextToken: Int? = nil,
        bosID: Int? = nil
    ) -> Bool {
        guard parameters.enabled else { return true }
        if let proposedTextToken, let bosID,
           proposedTextToken == bosID, speechConfirmed {
            return true
        }
        guard speechConfirmed, rnntIsBlank == true else { return false }
        let requiredFrames = parameters.functionCallEndOfUtteranceFrames
            ?? parameters.endOfUtteranceFrames
        return consecutiveBlankFrames + 1 >= requiredFrames
    }

    /// Verification needs a model-native candidate from either the current
    /// probe or the last frame of the same uninterrupted speech segment.
    static func functionStartVerificationReady(
        commitReady: Bool,
        currentProbeProposesStart: Bool,
        hasDeferredCandidate: Bool
    ) -> Bool {
        commitReady
            && (currentProbeProposesStart || hasDeferredCandidate)
    }

    /// A retained function candidate is valid only for one uninterrupted user
    /// speech segment. New user speech starts a different semantic decision;
    /// output injection and an active assistant turn likewise make the old
    /// candidate causally stale.
    static func shouldDiscardDeferredFunctionCandidate(
        injectingOutput: Bool,
        agentSpeaking: Bool,
        rnntIsBlank: Bool?
    ) -> Bool {
        injectingOutput || agentSpeaking || rnntIsBlank == false
    }

    public func isWaitingForFunctionResponse() -> Bool {
        awaitingFunctionResponse
    }

    /// Whether the checkpoint is currently producing a model-native tool call
    /// on NVIDIA's asynchronous function-channel path.
    public func isGeneratingFunctionCall() -> Bool {
        collectingFunctionCall
    }

    /// Restore a usable conversational state when a host cannot inject either
    /// the real tool result or its compact failure result. Without this escape
    /// hatch the session would remain permanently parked in
    /// `awaitingFunctionResponse` even though no response can arrive.
    public func recoverFromFunctionResponseFailure() async {
        let activeTask = functionFastPathTask
        activeTask?.cancel()
        if let activeTask { await activeTask.value }
        functionFastPathTask = nil
        abortFunctionFastPath()
        resumeFunctionFastPathProgressWaiters()
        functionFastPathLastProgressNanoseconds = nil
    }

    /// Current or most recently completed model-native tool-call decode. This
    /// is wall time, not audio-frame compute, and is therefore intentionally
    /// separate from realtime factor.
    public func functionCallDecodeMetrics()
        -> VoiceChatFunctionCallDecodeMetrics?
    {
        guard let started = functionCallDecodeStartedAtNanoseconds else {
            return lastFunctionCallDecodeMetrics
        }
        return VoiceChatFunctionCallDecodeMetrics(
            active: true,
            completed: false,
            elapsedMilliseconds: Self.elapsedMilliseconds(since: started),
            tokenSteps: functionCallDecodeSteps,
            modelMilliseconds: functionCallModelMilliseconds,
            speechCacheMilliseconds: functionCallSpeechCacheMilliseconds)
    }

    /// Current or most recently completed replay of an external result through
    /// the trained function channel. This background cache synchronization is
    /// intentionally excluded from foreground microphone RTF.
    public func functionResponseMetrics()
        -> VoiceChatFunctionResponseMetrics?
    {
        guard let started = functionResponseStartedAtNanoseconds else {
            return lastFunctionResponseMetrics
        }
        return VoiceChatFunctionResponseMetrics(
            active: true,
            completed: false,
            elapsedMilliseconds: Self.elapsedMilliseconds(since: started),
            tokenSteps: functionResponseSteps,
            prefillBatches: functionResponsePrefillBatches,
            languageCacheMilliseconds:
                functionResponseLanguageCacheMilliseconds,
            speechCacheMilliseconds:
                functionResponseSpeechCacheMilliseconds)
    }

    /// Return all UI-facing function-channel state in one actor hop. This is
    /// observational only and does not advance either model channel.
    package func functionRuntimeStatus() -> VoiceChatFunctionRuntimeStatus {
        VoiceChatFunctionRuntimeStatus(
            generatingCall: collectingFunctionCall,
            waitingForResponse: awaitingFunctionResponse,
            callDecode: functionCallDecodeMetrics(),
            responseSync: functionResponseMetrics())
    }

    func functionHeadEvaluationStatistics()
        -> VoiceChatFunctionHeadEvaluationStatistics
    {
        VoiceChatFunctionHeadEvaluationStatistics(
            probeFrames: functionProbeFrames,
            fullVerificationFrames: functionFullVerificationFrames,
            openCallFrames: functionOpenCallFrames,
            asynchronousCallSteps: asynchronousFunctionCallSteps,
            asynchronousResponseSteps: asynchronousFunctionResponseSteps,
            asynchronousResponsePrefillBatches:
                asynchronousFunctionResponsePrefillBatches,
            asynchronousCallTimeouts: asynchronousFunctionCallTimeouts,
            asynchronousResponseTimeouts: asynchronousFunctionResponseTimeouts,
            asynchronousCallInterruptions: asynchronousFunctionCallInterruptions,
            microphoneFramesDuringAsyncWork:
                microphoneFramesDuringAsynchronousFunctionWork,
            verificationWinners: functionVerificationWinners)
    }

    /// Whether a finite-input caller must keep advancing silent model frames
    /// to finish an in-flight native call or injected result.
    public func hasPendingFunctionOutput() -> Bool {
        collectingFunctionCall
            || functionFastPathTask != nil
            || pendingCompletedFunctionCall != nil
            || awaitingFunctionResponse
            || awaitingPostFunctionAssistantTurn
            || forcedFunctionTokenIndex < forcedFunctionTokens.count
            || injectingFunctionResponse
            || !deferredMicrophoneFrames.isEmpty
            || deferredReplayPendingIdleSpeechFrames > 0
    }

    /// Mark a discontinuity created by bounded live-input load shedding.
    /// Decoder and language history are preserved. Fresh RNN-T counters and
    /// predictor text are reset across the discontinuity, while an already
    /// confirmed user turn remains armed so a brief overload cannot erase the
    /// whole request and leave the assistant permanently silent.
    public func resynchronizeLiveInput() {
        deferredFunctionStartHidden = nil
        turnTakingState.resynchronizeInput(
            preserveSpeechConfirmation: true)
        if transcriptionState != nil {
            transcriptionState = model.transcriber.makeStreamState()
            streamingUserTranscript = ""
        }
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
        let padEmbedding = model.languageModel.embed(
            MLXArray([model.tokenizer.padID]))
            .expandedDimensions(axis: 0).asType(.float32)
        // System-prompt tokens are conditioning inputs. Both generated
        // channels remain PAD over the prefix, so the full fused sequence can
        // be causally prefetched in bounded chunks without computing either
        // 131k-vocabulary output head for every prompt position.
        for start in stride(
            from: 0,
            to: ids.count,
            by: Self.promptPrefillChunkTokens
        ) {
            let end = min(ids.count, start + Self.promptPrefillChunkTokens)
            let tokenEmbeddings = model.languageModel.embed(
                MLXArray(Array(ids[start ..< end])))
                .expandedDimensions(axis: 0).asType(.float32)
            let fused = tokenEmbeddings + MLXArray(Float(3)) * padEmbedding
            let hidden = model.languageModel.prefill(
                embeddings: fused,
                cache: languageCache)
            eval(hidden)
        }
        previousText = model.tokenizer.padID
        previousFunction = model.tokenizer.padID
    }

    private func advance(
        audioEmbedding: MLXArray,
        record: Bool,
        recordEvent: Bool = true,
        forceSilent: Bool,
        audioMilliseconds: Double,
        perceptionLatencyMilliseconds: Double,
        userTranscript: String?,
        rnntIsBlank: Bool?,
        rnntHasLexicalToken: Bool = false,
        deferredInputReplay: Bool = false,
        deferIdleSpeechCache: Bool = false
    ) throws -> VoiceChatFrameEvent {
        if rnntHasLexicalToken {
            turnTakingState.observeRecognizedSpeechToken()
        }
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
        let injectingOutput = forcedFunctionTokenIndex < forcedFunctionTokens.count
            || awaitingFunctionResponse
            || injectingFunctionResponse
        let functionEvaluation = functionHeadScheduler.evaluation(
            enabled: functionCallingEnabled
                && functionStartGate.allowsFunctionStarts,
            recording: record,
            forceSilent: forceSilent,
            collectingCall: collectingFunctionCall,
            outputInjectionActive: injectingOutput,
            turnTakingEnabled: turnTakingState.parameters.enabled,
            userSpeechConfirmed: turnTakingState.speechConfirmed,
            assistantTurnActive: turnTakingState.agentSpeaking)
        // Both an open native call and a pending/injected tool response force
        // the text channel to PAD below. Preserve the shared causal backbone
        // state, but do not materialize a 131k-row text projection whose
        // selection cannot be observed. Provider-wait microphone frames take
        // the perception-only two-phase path before reaching this function.
        let textDecisionSuppressed = Self.textDecisionIsSuppressed(
            functionEvaluation: functionEvaluation,
            outputInjectionActive: injectingOutput)
            || deferredInputReplay
        let functionHidden: MLXArray
        let greedyText: MLXArray?
        let proposedTextToken: Int
        if textDecisionSuppressed {
            let hidden = model.languageModel.callFunctionBackbone(
                embeddings: fused, cache: languageCache)
            functionHidden = hidden[0, -1, 0...].asType(.float32)
            greedyText = nil
            proposedTextToken = model.tokenizer.padID
        } else {
            let output = model.languageModel.call(
                embeddings: fused, cache: languageCache)
            let textLogits = output.logits[0, -1, 0...]
            let textSelection = MLX.argMax(
                textLogits.asType(.float32), axis: -1)
            functionHidden = output.hidden[0, -1, 0...].asType(.float32)
            greedyText = textSelection
            proposedTextToken = sampleTextToken(
                textLogits, greedy: textSelection.item(Int.self))
        }
        if functionEvaluation == .startProbe {
            functionProbeFrames += 1
        } else if functionEvaluation == .fullVocabulary {
            functionOpenCallFrames += 1
            // Count the projection that produced this token before handling
            // an end token below, so completed metrics include their final
            // decode step. Start probes are intentionally excluded.
            functionCallDecodeSteps += 1
        }
        let functionProbe = functionEvaluation == .startProbe
            ? functionDetectionRows.map {
                MLX.argMax(MLX.matmul(functionHidden, $0.transposed()), axis: -1)
            }
            : nil
        var functionSelection = functionEvaluation == .fullVocabulary
            ? model.languageModel.functionHead.map {
                MLX.argMax($0(functionHidden), axis: -1)
            }
            : nil
        let functionStartCommitReady = !deferredInputReplay
            && Self.functionStartCommitReady(
                parameters: turnTakingState.parameters,
                speechConfirmed: turnTakingState.speechConfirmed,
                consecutiveBlankFrames: turnTakingState.consecutiveBlankFrames,
                rnntIsBlank: rnntIsBlank,
                proposedTextToken: proposedTextToken,
                bosID: model.tokenizer.bosID)
        // A deferred candidate belongs to the uninterrupted speech segment
        // that produced it. Fresh RNN-T speech invalidates it before this frame
        // may contribute a new candidate.
        if Self.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: injectingOutput,
            agentSpeaking: turnTakingState.agentSpeaking,
            rnntIsBlank: rnntIsBlank)
        {
            deferredFunctionStartHidden = nil
        }
        // The text projection and tiny two-row probe share one GPU
        // synchronization. Probe index 1 means `<SPECIAL_20>` beats PAD, which
        // is necessary (but not sufficient) for it to be the full-head argmax.
        let backboneOnlyEvaluation = textDecisionSuppressed
            && functionEvaluation != .fullVocabulary
            ? functionHidden
            : nil
        MLX.eval(
            [greedyText].compactMap { $0 }
                + [functionProbe, functionSelection, backboneOnlyEvaluation]
                    .compactMap { $0 })
        if functionEvaluation == .startProbe {
            let currentProbeProposesStart = functionProbe?.item(Int.self) == 1
            if Self.functionStartVerificationReady(
                commitReady: functionStartCommitReady,
                currentProbeProposesStart: currentProbeProposesStart,
                hasDeferredCandidate: deferredFunctionStartHidden != nil
            ) {
                // The two-row result is only a necessary condition; the full
                // projection still verifies the exact global argmax. Prefer a
                // current candidate; otherwise use the final candidate from
                // this uninterrupted speech segment, because this checkpoint's
                // function signal commonly fades during the silence window.
                let verificationHidden = currentProbeProposesStart
                    ? functionHidden
                    : deferredFunctionStartHidden!
                functionSelection = model.languageModel.functionHead.map {
                    MLX.argMax($0(verificationHidden), axis: -1)
                }
                if let functionSelection { MLX.eval(functionSelection) }
                deferredFunctionStartHidden = nil
                functionFullVerificationFrames += 1
            } else if currentProbeProposesStart {
                // Retain only the latest 4096-value candidate state. It is
                // cleared above if the user resumes speaking.
                MLX.eval(functionHidden)
                deferredFunctionStartHidden = functionHidden
                functionSelection = nil
            } else {
                functionSelection = nil
            }
        }
        let selectedFunctionToken = functionSelection?.item(Int.self)
            ?? model.tokenizer.padID
        if functionEvaluation == .startProbe,
           functionSelection != nil
        {
            functionVerificationWinners[selectedFunctionToken, default: 0] += 1
        }
        // Outside an open call, only the protocol's explicitly verified start
        // token may enter autoregressive feedback. Other probe/full-head
        // outcomes remain PAD.
        let naturalFunctionToken = collectingFunctionCall
            || selectedFunctionToken == functionStartID
            ? selectedFunctionToken
            : model.tokenizer.padID
        var functionToken = naturalFunctionToken
        var completedFunctionCall: String?
        if recordEvent, let pendingCompletedFunctionCall {
            completedFunctionCall = pendingCompletedFunctionCall
            self.pendingCompletedFunctionCall = nil
        }
        var functionSilencesText = false
        if functionCallingEnabled, record, !forceSilent {
            if forcedFunctionTokenIndex < forcedFunctionTokens.count {
                functionToken = forcedFunctionTokens[forcedFunctionTokenIndex]
                forcedFunctionTokenIndex += 1
                functionResponseSteps += 1
                functionResponsePrefillBatches += 1
                functionSilencesText = true
                if functionToken == functionResponseEndID,
                   forcedFunctionTokenIndex == forcedFunctionTokens.count
                {
                    completeFunctionResponseInjection()
                }
            } else if awaitingFunctionResponse {
                functionToken = model.tokenizer.padID
                functionSilencesText = true
            } else {
                if injectingFunctionResponse {
                    injectingFunctionResponse = false
                    forcedFunctionTokens.removeAll(keepingCapacity: true)
                    forcedFunctionTokenIndex = 0
                }

                if functionToken == functionStartID {
                    collectingFunctionCall = true
                    functionCallTokens.removeAll(keepingCapacity: true)
                    functionCallPrefixPending = true
                    consecutiveFunctionInterruptionSpeechFrames = 0
                    if functionCallDecodeStartedAtNanoseconds == nil {
                        beginFunctionCallDecode()
                    }
                    turnTakingState.beginFunctionCall()
                    functionSilencesText = true
                } else if collectingFunctionCall {
                    functionSilencesText = true
                    // If a caller deliberately suppresses or outruns the
                    // background fast path, preserve the checkpoint's natural
                    // token stream instead of injecting the prefix late.
                    functionCallPrefixPending = false
                    if functionToken == functionEndID
                        || functionCallTokens.count
                            >= Self.maximumFunctionCallTokens
                    {
                        completedFunctionCall = model.tokenizer.decode(
                            functionCallTokens,
                            skipSpecialTokens: false)
                        collectingFunctionCall = false
                        awaitingFunctionResponse = true
                        finishFunctionCallDecode(
                            completed: functionToken == functionEndID)
                    } else if functionToken != model.tokenizer.padID {
                        functionCallTokens.append(functionToken)
                    }
                }
            }
        }

        let textToken: Int
        let turnTakingAction: VoiceChatTurnTakingAction
        if functionSilencesText {
            textToken = model.tokenizer.padID
            turnTakingAction = .none
        } else if deferredInputReplay {
            if let rnntIsBlank {
                turnTakingState.observeDeferredInput(
                    rnntIsBlank: rnntIsBlank)
            }
            textToken = model.tokenizer.padID
            turnTakingAction = .none
        } else if record, !forceSilent, let rnntIsBlank {
            let result = turnTakingState.selectToken(
                proposedToken: proposedTextToken,
                rnntIsBlank: rnntIsBlank,
                padID: model.tokenizer.padID,
                bosID: model.tokenizer.bosID,
                eosID: model.tokenizer.eosID)
            textToken = result.token
            turnTakingAction = result.action
        } else {
            textToken = proposedTextToken
            turnTakingAction = .none
        }
        // Once the model opens the confirmation prompt, `agentSpeaking`
        // suppresses function starts for that turn; after EOS, fresh RNN-T
        // user activity is required before another start probe.
        functionStartGate.observeTextToken(
            textToken, bosID: model.tokenizer.bosID)
        if awaitingPostFunctionAssistantTurn {
            if textToken == model.tokenizer.bosID {
                postFunctionAssistantTurnStarted = true
            } else if textToken == model.tokenizer.eosID,
                      postFunctionAssistantTurnStarted {
                awaitingPostFunctionAssistantTurn = false
                postFunctionAssistantTurnStarted = false
            }
        }
        let decisionLatency = milliseconds(since: decisionStart)

        let feedbackAfterStep = Self.channelFeedbackAfterStep(
            record: record,
            textToken: textToken,
            functionToken: functionToken,
            padID: model.tokenizer.padID)
        previousText = feedbackAfterStep.text
        previousFunction = feedbackAfterStep.function

        let recordedToken = forceSilent ? model.tokenizer.padID : textToken
        var synthesisLatency = 0.0
        var playbackRequired = true
        var audio: [Float] = []
        if record {
            emittedTokens.append(recordedToken)
            guard let speechState, let previousSpeechCode else {
                throw VoiceChatGenerationError.speechDecoderNotPrimed
            }
            let synthesisStart = DispatchTime.now().uptimeNanoseconds
            let turnForcesSpeechSilence = speechTurnState.shouldForceSilence(
                textToken: recordedToken,
                padID: model.tokenizer.padID,
                bosID: model.tokenizer.bosID,
                eosID: model.tokenizer.eosID)
            // Deferred microphone positions already elapsed on the wall clock.
            // They update the shared semantic and voice caches causally, but
            // must never restart an old acoustic tail or emit duplicate audio.
            let forceSpeechSilence = deferredInputReplay
                || turnForcesSpeechSilence
            playbackRequired = !deferredInputReplay || !forceSpeechSilence
            if speechParameters.realtimeIdleOptimization,
               forceSpeechSilence,
               recordedToken == model.tokenizer.padID {
                pendingIdleSpeechFrames += 1
                if deferIdleSpeechCache {
                    deferredReplayPendingIdleSpeechFrames += 1
                } else if pendingIdleSpeechFrames >= Self.idleSpeechBatchFrames {
                    try flushIdleSpeechFrames(
                        state: speechState,
                        previousCode: previousSpeechCode)
                }
                let code = MLX.broadcast(
                    model.speechDecoder.silenceCodes,
                    to: previousSpeechCode.shape).asType(.int32)
                generatedCodes.append(code)
                audio = [Float](repeating: 0, count: Self.outputSamplesPerFrame)
            } else {
                try flushIdleSpeechFrames(
                    state: speechState,
                    previousCode: previousSpeechCode)
                let code = try model.speechDecoder.step(
                    state: speechState,
                    previousCode: self.previousSpeechCode!,
                    textToken: recordedToken,
                    parameters: speechParameters,
                    forceSilence: forceSpeechSilence)
                evaluateSpeech(code, state: speechState)
                self.previousSpeechCode = code
                generatedCodes.append(code)
                audio = forceSpeechSilence
                    ? [Float](repeating: 0, count: Self.outputSamplesPerFrame)
                    : liveAudioFrame()
            }
            synthesisLatency = milliseconds(since: synthesisStart)
        }

        let event = VoiceChatFrameEvent(
            index: recordedEvents.count,
            textToken: recordedToken,
            functionToken: functionToken,
            functionCall: completedFunctionCall,
            text: model.tokenizer.decode([recordedToken], skipSpecialTokens: false),
            userTranscript: userTranscript,
            rnntIsBlank: rnntIsBlank,
            turnTakingAction: turnTakingAction,
            speaking: !model.tokenizer.specialIDs.contains(recordedToken),
            audioPositionMilliseconds: audioMilliseconds,
            perceptionLatencyMilliseconds: perceptionLatencyMilliseconds,
            decisionLatencyMilliseconds: decisionLatency,
            synthesisLatencyMilliseconds: synthesisLatency,
            audio: audio,
            playbackRequired: playbackRequired)
        if recordEvent { recordedEvents.append(event) }
        return event
    }

    /// Start NVIDIA's function-channel fast path after `<SPECIAL_20>` wins or
    /// when a tool result is ready for phase-two injection. Each generated
    /// call token or known-response prefill chunk yields the actor so fresh
    /// microphone frames can continue through perception and RNN-T while the
    /// 11B cache advances on cached silence.
    private func startFunctionFastPathIfNeeded() {
        guard hasFunctionFastPathWork,
              functionFastPathTask == nil,
              !functionFastPathSuppressedForTesting else { return }
        guard functionSilenceEmbedding != nil else {
            abortFunctionFastPath()
            return
        }
        functionFastPathInterrupted = false
        functionFastPathLastProgressNanoseconds = DispatchTime.now()
            .uptimeNanoseconds
        // Use the caller's normal priority. A lower `.utility` priority lets a
        // continuous stream of microphone actor jobs starve function decoding;
        // the bounded token/batch work and explicit yields provide cooperative
        // interleaving without creating that priority inversion.
        functionFastPathTask = Task { [weak self] in
            await self?.runFunctionFastPath()
        }
    }

    /// Advance the fixed opening syntax of a native call in one causal batch.
    ///
    /// The model has already made the semantic decision to call a tool by
    /// emitting `<SPECIAL_20>`. The training prompt requires every subsequent
    /// payload to begin with `<TOOLCALL>[{"name":"`, so evaluating a full
    /// vocabulary projection for each of those known positions cannot affect
    /// the call.
    /// The prefix is still inserted into both language and EAR-TTS histories;
    /// only its redundant autoregressive selection is skipped.
    private func prefillFunctionCallPrefix(
        audioEmbedding: MLXArray
    ) throws -> Int {
        guard functionCallPrefixPending,
              collectingFunctionCall,
              functionCallTokens.isEmpty,
              !functionCallPrefixTokens.isEmpty else { return 0 }
        guard let speechState, let previousSpeechCode else {
            throw VoiceChatGenerationError.speechDecoderNotPrimed
        }

        let tokens = functionCallPrefixTokens
        let count = tokens.count
        let padID = model.tokenizer.padID
        let functionFeedback = Self.functionResponseFeedbackTokens(
            previousFunction: previousFunction,
            responseTokens: tokens)
        let textEmbedding = model.languageModel.embed(
            MLXArray([Int](repeating: padID, count: count)))
            .expandedDimensions(axis: 0)
        let functionEmbedding = model.languageModel.embed(
            MLXArray(functionFeedback))
            .expandedDimensions(axis: 0)
        let fused = MLX.broadcast(
            audioEmbedding,
            to: [1, count, audioEmbedding.dim(2)])
            + textEmbedding
            + MLXArray(Float(2)) * functionEmbedding

        let languageStarted = DispatchTime.now().uptimeNanoseconds
        let hidden = model.languageModel.prefill(
            embeddings: fused, cache: languageCache)
        MLX.eval(hidden)
        functionCallModelMilliseconds += Self.elapsedMilliseconds(
            since: languageStarted)

        let pendingPrefixFrames = pendingIdleSpeechFrames
        pendingIdleSpeechFrames = 0
        let speechStarted = DispatchTime.now().uptimeNanoseconds
        let code = model.speechDecoder.advanceIdleSilence(
            state: speechState,
            previousCode: previousSpeechCode,
            frames: pendingPrefixFrames + count,
            guidance: speechParameters.guidance)
        evaluateSpeech(code, state: speechState)
        functionCallSpeechCacheMilliseconds += Self.elapsedMilliseconds(
            since: speechStarted)

        self.previousSpeechCode = code
        previousText = padID
        previousFunction = tokens.last ?? previousFunction
        functionCallTokens.append(contentsOf: tokens)
        functionCallDecodeSteps += count
        let silenceCode = MLX.broadcast(
            model.speechDecoder.silenceCodes,
            to: previousSpeechCode.shape).asType(.int32)
        emittedTokens.append(contentsOf: repeatElement(padID, count: count))
        generatedCodes.append(contentsOf: repeatElement(
            silenceCode, count: count))
        for _ in 0 ..< count {
            _ = speechTurnState.shouldForceSilence(
                textToken: padID,
                padID: padID,
                bosID: model.tokenizer.bosID,
                eosID: model.tokenizer.eosID)
        }
        functionCallPrefixPending = false
        return count
    }

    /// Replay the known phase-two function response causally in chunks.
    ///
    /// This is equivalent to repeated `advance` calls for cache and channel
    /// feedback purposes: agent text remains PAD, each function token becomes
    /// the next step's function feedback, and the trained end marker releases
    /// native result-conditioned continuation. Unlike phase-one call generation,
    /// phase two needs neither text logits nor the 131k-row function head.
    private func prefillFunctionResponse(
        audioEmbedding: MLXArray,
        deadlineNanoseconds: UInt64? = nil
    ) async throws -> Int {
        guard forcedFunctionTokenIndex < forcedFunctionTokens.count else {
            return 0
        }
        guard let speechState, let previousSpeechCode else {
            throw VoiceChatGenerationError.speechDecoderNotPrimed
        }

        // `advance` would flush an incomplete idle batch before a non-idle
        // speech step. Phase two is agent-idle throughout, so fold that pending
        // prefix into the same batched TTS cache update.
        let pendingPrefixFrames = pendingIdleSpeechFrames
        pendingIdleSpeechFrames = 0
        let padID = model.tokenizer.padID
        var processed = 0
        var speechPrefixFrames = pendingPrefixFrames
        var speechCode = previousSpeechCode

        while forcedFunctionTokenIndex < forcedFunctionTokens.count {
            try Task.checkCancellation()
            if let deadlineNanoseconds,
               DispatchTime.now().uptimeNanoseconds >= deadlineNanoseconds
            {
                throw VoiceChatGenerationError.invalidSpeechConfiguration(
                    "function response prefill timed out")
            }
            let start = forcedFunctionTokenIndex
            let end = min(
                forcedFunctionTokens.count,
                start + Self.functionResponsePrefillChunkTokens)
            let responseTokens = Array(forcedFunctionTokens[start ..< end])
            let count = responseTokens.count

            // Position i consumes the previous position's text/function
            // feedback. Text is forced PAD for the complete function cycle.
            let functionFeedback = Self.functionResponseFeedbackTokens(
                previousFunction: previousFunction,
                responseTokens: responseTokens)
            let textEmbedding = model.languageModel.embed(
                MLXArray([Int](repeating: padID, count: count)))
                .expandedDimensions(axis: 0)
            let functionEmbedding = model.languageModel.embed(
                MLXArray(functionFeedback))
                .expandedDimensions(axis: 0)
            let fused = MLX.broadcast(
                audioEmbedding,
                to: [1, count, audioEmbedding.dim(2)])
                + textEmbedding
                + MLXArray(Float(2)) * functionEmbedding
            let languageStarted = DispatchTime.now().uptimeNanoseconds
            let hidden = model.languageModel.prefill(
                embeddings: fused, cache: languageCache)
            MLX.eval(hidden)
            functionResponseLanguageCacheMilliseconds +=
                Self.elapsedMilliseconds(since: languageStarted)

            previousText = padID
            previousFunction = responseTokens.last ?? previousFunction
            forcedFunctionTokenIndex = end
            processed += count
            asynchronousFunctionResponseSteps += count
            asynchronousFunctionResponsePrefillBatches += 1
            functionResponseSteps += count
            functionResponsePrefillBatches += 1

            // EAR-TTS and the codec timeline still receive one canonical-silent
            // position per hidden function token. Advance the attention cache
            // in the same bounded chunks and retain the silence codes used by
            // the live codec's causal context window.
            let speechFrames = speechPrefixFrames + count
            let speechStarted = DispatchTime.now().uptimeNanoseconds
            let code = model.speechDecoder.advanceIdleSilence(
                state: speechState,
                previousCode: speechCode,
                frames: speechFrames,
                guidance: speechParameters.guidance)
            evaluateSpeech(code, state: speechState)
            functionResponseSpeechCacheMilliseconds +=
                Self.elapsedMilliseconds(since: speechStarted)
            speechCode = code
            speechPrefixFrames = 0
            let silenceCode = MLX.broadcast(
                model.speechDecoder.silenceCodes,
                to: previousSpeechCode.shape).asType(.int32)
            emittedTokens.append(contentsOf: repeatElement(
                padID, count: count))
            generatedCodes.append(contentsOf: repeatElement(
                silenceCode, count: count))
            for _ in 0 ..< count {
                _ = speechTurnState.shouldForceSilence(
                    textToken: padID,
                    padID: padID,
                    bosID: model.tokenizer.bosID,
                    eosID: model.tokenizer.eosID)
            }

            if end == forcedFunctionTokens.count,
               responseTokens.last == functionResponseEndID
            {
                completeFunctionResponseInjection()
            }
            if forcedFunctionTokenIndex < forcedFunctionTokens.count {
                resumeFunctionFastPathProgressWaiters()
                await Task.yield()
            }
        }
        self.previousSpeechCode = speechCode

        injectingFunctionResponse = false
        forcedFunctionTokens.removeAll(keepingCapacity: true)
        forcedFunctionTokenIndex = 0
        return processed
    }

    static func functionResponseFeedbackTokens(
        previousFunction: Int,
        responseTokens: [Int]
    ) -> [Int] {
        guard !responseTokens.isEmpty else { return [] }
        return [previousFunction] + responseTokens.dropLast()
    }

    static func textDecisionIsSuppressed(
        functionEvaluation: VoiceChatFunctionHeadEvaluation,
        outputInjectionActive: Bool
    ) -> Bool {
        functionEvaluation == .fullVocabulary || outputInjectionActive
    }

    /// NVIDIA's two-phase path freezes the shared LLM/TTS timeline after EOTC
    /// while the external tool is running. Live microphone frames still update
    /// perception and RNN-T, but they are not inserted as synthetic PAD
    /// positions between the native call and its eventual response.
    static func usesPerceptionOnlyMicrophonePath(
        fastPathRunning: Bool,
        hasFastPathWork: Bool,
        awaitingFunctionResponse: Bool
    ) -> Bool {
        (fastPathRunning && hasFastPathWork) || awaitingFunctionResponse
    }

    /// Keep newly captured audio behind older deferred speech in the shared
    /// language/TTS timeline. A replayed frame owns this callback's one model
    /// step even when it emptied the queue; otherwise the callback would do
    /// two full 11B advances and recreate the capture stall we are avoiding.
    static func shouldDeferLiveMicrophoneFrame(
        fastPathRunning: Bool,
        hasFastPathWork: Bool,
        awaitingFunctionResponse: Bool,
        deferredTimelineWorkThisPush: Bool,
        hasDeferredInput: Bool
    ) -> Bool {
        usesPerceptionOnlyMicrophonePath(
            fastPathRunning: fastPathRunning,
            hasFastPathWork: hasFastPathWork,
            awaitingFunctionResponse: awaitingFunctionResponse)
            || deferredTimelineWorkThisPush
            || hasDeferredInput
    }

    private func runFunctionFastPath() async {
        guard let functionSilenceEmbedding else {
            abortFunctionFastPath()
            functionFastPathTask = nil
            resumeFunctionFastPathProgressWaiters()
            functionFastPathLastProgressNanoseconds = nil
            return
        }

        let callStarted = DispatchTime.now().uptimeNanoseconds
        let maximumCallNanoseconds = UInt64(
            Self.maximumAsynchronousFunctionCallSeconds * 1_000_000_000)
        let maximumResponseNanoseconds = UInt64(
            Self.maximumAsynchronousFunctionResponseSeconds * 1_000_000_000)
        var callSteps = 0
        var responseSteps = 0
        var responseStarted: UInt64?
        var failed = false

        if collectingFunctionCall, functionCallPrefixPending {
            do {
                let prefixed = try prefillFunctionCallPrefix(
                    audioEmbedding: functionSilenceEmbedding)
                callSteps += prefixed
                asynchronousFunctionCallSteps += prefixed
                resumeFunctionFastPathProgressWaiters()
                // Give a captured microphone frame a chance to run before the
                // first model-selected tool-name token.
                await Task.yield()
            } catch {
                failed = true
            }
        }

        // NVIDIA's two-phase path injects an already-known tool response at
        // full LLM speed. A causal prefill produces the same cache history but
        // avoids one decode and vocabulary projection per response token.
        if injectingFunctionResponse,
           forcedFunctionTokenIndex < forcedFunctionTokens.count
        {
            do {
                let responsePrefillStarted = DispatchTime.now().uptimeNanoseconds
                responseSteps = try await prefillFunctionResponse(
                    audioEmbedding: functionSilenceEmbedding,
                    deadlineNanoseconds:
                        responsePrefillStarted + maximumResponseNanoseconds)
                responseStarted = responsePrefillStarted
            } catch {
                failed = true
            }
        }

        while !failed,
              !Task.isCancelled,
              hasFunctionFastPathWork,
              (!collectingFunctionCall || !functionFastPathInterrupted)
        {
            let now = DispatchTime.now().uptimeNanoseconds
            if collectingFunctionCall {
                guard callSteps < Self.maximumAsynchronousFunctionCallSteps,
                      now - callStarted < maximumCallNanoseconds else { break }
            } else {
                let responseStart = responseStarted ?? now
                responseStarted = responseStart
                guard responseSteps
                        < Self.maximumAsynchronousFunctionResponseSteps,
                      now - responseStart < maximumResponseNanoseconds else {
                    break
                }
            }
            do {
                let wasCollectingCall = collectingFunctionCall
                let event = try advance(
                    audioEmbedding: functionSilenceEmbedding,
                    record: true,
                    recordEvent: false,
                    forceSilent: false,
                    audioMilliseconds: Double(
                        completedFrames * Self.frameMilliseconds),
                    perceptionLatencyMilliseconds: 0,
                    userTranscript: nil,
                    rnntIsBlank: true,
                    rnntHasLexicalToken: false)
                if wasCollectingCall {
                    callSteps += 1
                    asynchronousFunctionCallSteps += 1
                    functionCallModelMilliseconds +=
                        event.decisionLatencyMilliseconds
                    functionCallSpeechCacheMilliseconds +=
                        event.synthesisLatencyMilliseconds
                } else {
                    responseSteps += 1
                    asynchronousFunctionResponseSteps += 1
                }
                if let call = event.functionCall {
                    pendingCompletedFunctionCall = call
                }
            } catch {
                failed = true
                break
            }
            resumeFunctionFastPathProgressWaiters()
            await Task.yield()
        }

        if functionFastPathInterrupted, collectingFunctionCall {
            asynchronousFunctionCallInterruptions += 1
            abortFunctionFastPath()
        } else if hasFunctionFastPathWork {
            if collectingFunctionCall {
                asynchronousFunctionCallTimeouts += 1
            } else {
                asynchronousFunctionResponseTimeouts += 1
            }
            abortFunctionFastPath()
        }
        functionFastPathTask = nil
        resumeFunctionFastPathProgressWaiters()
        functionFastPathLastProgressNanoseconds = nil
    }

    private func resumeFunctionFastPathProgressWaiters() {
        functionFastPathLastProgressNanoseconds = DispatchTime.now()
            .uptimeNanoseconds
        guard !functionFastPathProgressWaiters.isEmpty else { return }
        let waiters = functionFastPathProgressWaiters
        functionFastPathProgressWaiters.removeAll(keepingCapacity: true)
        for waiter in waiters { waiter.resume() }
    }

    private func abortFunctionFastPath() {
        finishFunctionCallDecode(completed: false)
        finishFunctionResponseMetrics(completed: false)
        collectingFunctionCall = false
        functionCallTokens.removeAll(keepingCapacity: true)
        functionCallPrefixPending = false
        previousFunction = model.tokenizer.padID
        forcedFunctionTokens.removeAll(keepingCapacity: true)
        forcedFunctionTokenIndex = 0
        awaitingFunctionResponse = false
        injectingFunctionResponse = false
        functionStartGate.reset()
        awaitingPostFunctionAssistantTurn = true
        postFunctionAssistantTurnStarted = false
        functionFastPathInterrupted = false
        consecutiveFunctionInterruptionSpeechFrames = 0
        pendingCompletedFunctionCall = nil
        deferredFunctionStartHidden = nil
        // No runtime-authored fallback is injected. Reauthorize the assistant
        // side of the accepted turn so the model can recover conversationally
        // from the partial call on subsequent microphone frames.
        turnTakingState.resumeAfterFunctionResponse(
            deferredUserInput: !deferredMicrophoneFrames.isEmpty)
    }

    /// Closing the function-response channel returns control to the assistant
    /// side of the same user turn. This is required even when the coordinator
    /// supplied no scripted text and the checkpoint must answer—or call a
    /// second tool—from the injected result itself.
    private func completeFunctionResponseInjection() {
        finishFunctionResponseMetrics(completed: true)
        awaitingPostFunctionAssistantTurn = true
        postFunctionAssistantTurnStarted = false
        turnTakingState.resumeAfterFunctionResponse(
            deferredUserInput: !deferredMicrophoneFrames.isEmpty)
    }

    /// Keep the live microphone and RNN-T caption/interrupt signal moving while
    /// the function channel owns the language-model cache. No LLM or TTS work
    /// is charged to this real 80 ms input tick; playback remains silence.
    private func functionMicrophoneEvent(
        perceptionLatencyMilliseconds: Double,
        userTranscript: String?,
        rnntIsBlank: Bool?,
        rnntHasLexicalToken: Bool
    ) -> VoiceChatFrameEvent {
        microphoneFramesDuringAsynchronousFunctionWork += 1
        if collectingFunctionCall, let rnntIsBlank {
            if rnntIsBlank {
                consecutiveFunctionInterruptionSpeechFrames = 0
            } else {
                consecutiveFunctionInterruptionSpeechFrames += 1
                if consecutiveFunctionInterruptionSpeechFrames
                    >= turnTakingState.parameters.beginOfUtteranceFrames
                {
                    functionFastPathInterrupted = true
                }
            }
        } else if rnntHasLexicalToken {
            // Lexical activity is retained for causal replay below. It does
            // not mutate normal turn-taking state until the model actually
            // consumes the corresponding modality embedding.
            consecutiveFunctionInterruptionSpeechFrames = max(
                1, consecutiveFunctionInterruptionSpeechFrames)
        }
        let padID = model.tokenizer.padID
        let code = MLX.broadcast(
            model.speechDecoder.silenceCodes,
            to: previousSpeechCode?.shape ?? [1, 1, 8]).asType(.int32)
        let completedFunctionCall: String?
        if functionFastPathTask == nil,
           let pendingCompletedFunctionCall
        {
            completedFunctionCall = pendingCompletedFunctionCall
            self.pendingCompletedFunctionCall = nil
        } else {
            completedFunctionCall = nil
        }
        generatedCodes.append(code)
        emittedTokens.append(padID)
        let event = VoiceChatFrameEvent(
            index: recordedEvents.count,
            textToken: padID,
            functionToken: padID,
            functionCall: completedFunctionCall,
            text: model.tokenizer.decode([padID], skipSpecialTokens: false),
            userTranscript: userTranscript,
            rnntIsBlank: rnntIsBlank,
            turnTakingAction: .none,
            speaking: false,
            audioPositionMilliseconds: Double(
                completedFrames * Self.frameMilliseconds),
            perceptionLatencyMilliseconds: perceptionLatencyMilliseconds,
            decisionLatencyMilliseconds: 0,
            synthesisLatencyMilliseconds: 0,
            audio: [Float](
                repeating: 0, count: Self.outputSamplesPerFrame),
            playbackRequired: true)
        recordedEvents.append(event)
        return event
    }

    /// Save only meaningful microphone regions while the function channel owns
    /// the shared language cache. Perception and RNN-T have already advanced,
    /// so these evaluated modality embeddings—not raw PCM—are what must be
    /// replayed. This avoids double-updating the streaming encoder.
    private func deferMicrophoneFrame(_ frame: DeferredMicrophoneFrame) {
        let hasActivity = frame.rnntIsBlank == false
            || frame.rnntHasLexicalToken

        if hasActivity {
            if !deferredMicrophoneCaptureActive {
                for prefix in deferredMicrophonePreRoll {
                    appendDeferredMicrophoneFrame(prefix)
                }
                deferredMicrophonePreRoll.removeAll(keepingCapacity: true)
                deferredMicrophoneCaptureActive = true
            }
            deferredMicrophoneTrailingBlankCount = 0
            appendDeferredMicrophoneFrame(frame)
            return
        }

        // Sessions without RNN-T activity metadata cannot identify silence.
        // Preserve their bounded function-wait input rather than silently
        // discarding speech.
        if frame.rnntIsBlank == nil {
            if !deferredMicrophoneCaptureActive {
                for prefix in deferredMicrophonePreRoll {
                    appendDeferredMicrophoneFrame(prefix)
                }
                deferredMicrophonePreRoll.removeAll(keepingCapacity: true)
                deferredMicrophoneCaptureActive = true
            }
            appendDeferredMicrophoneFrame(frame)
            return
        }

        if deferredMicrophoneCaptureActive,
           deferredMicrophoneTrailingBlankCount
            < Self.deferredMicrophoneTrailingBlankFrames
        {
            deferredMicrophoneTrailingBlankCount += 1
            appendDeferredMicrophoneFrame(frame)
            return
        }

        deferredMicrophoneCaptureActive = false
        deferredMicrophoneTrailingBlankCount = 0
        deferredMicrophonePreRoll.append(frame)
        if deferredMicrophonePreRoll.count
            > Self.deferredMicrophonePreRollFrames
        {
            deferredMicrophonePreRoll.removeFirst(
                deferredMicrophonePreRoll.count
                    - Self.deferredMicrophonePreRollFrames)
        }
    }

    private func appendDeferredMicrophoneFrame(
        _ frame: DeferredMicrophoneFrame
    ) {
        guard deferredMicrophoneFrames.count
            < Self.maximumDeferredMicrophoneFrames else {
            deferredMicrophoneDroppedFrames += 1
            return
        }
        deferredMicrophoneFrames.append(frame)
    }

    /// Once every deferred language position is causal, advance the matching
    /// idle EAR-TTS positions in bounded batches between microphone callbacks.
    /// Flushing the complete accumulated tail in one operation can monopolize
    /// the actor for hundreds of milliseconds after a long interruption.
    private func synchronizeDeferredReplaySpeechCacheIfReady() throws -> Bool {
        guard !Self.usesPerceptionOnlyMicrophonePath(
                fastPathRunning: functionFastPathTask != nil,
                hasFastPathWork: hasFunctionFastPathWork,
                awaitingFunctionResponse: awaitingFunctionResponse),
              deferredMicrophoneFrames.isEmpty,
              deferredReplayPendingIdleSpeechFrames > 0,
              let speechState,
              let previousSpeechCode
        else { return false }

        try flushIdleSpeechFrames(
            state: speechState,
            previousCode: previousSpeechCode,
            maximumFrames: Self.idleSpeechBatchFrames)
        return true
    }

    /// Reinsert captured speech once the function call/result no longer owns
    /// the shared cache. Replay is deliberately paced to one model step per
    /// live callback. Newly captured frames continue through perception/RNN-T
    /// and remain causally queued until the older speech has caught up.
    /// Replayed idle frames advertise `playbackRequired == false`, because
    /// their wall-clock silence was already played while the tool was active.
    private func replayDeferredMicrophoneInputIfReady(
        recordEvent: Bool = true
    ) throws
        -> [VoiceChatFrameEvent]
    {
        guard !Self.usesPerceptionOnlyMicrophonePath(
                fastPathRunning: functionFastPathTask != nil,
                hasFastPathWork: hasFunctionFastPathWork,
                awaitingFunctionResponse: awaitingFunctionResponse)
        else { return [] }
        guard !deferredMicrophoneFrames.isEmpty else {
            deferredMicrophonePreRoll.removeAll(keepingCapacity: true)
            deferredMicrophoneCaptureActive = false
            deferredMicrophoneTrailingBlankCount = 0
            return []
        }

        var events: [VoiceChatFrameEvent] = []
        events.reserveCapacity(Self.maximumDeferredMicrophoneReplayFramesPerPush)
        for _ in 0 ..< Self.maximumDeferredMicrophoneReplayFramesPerPush {
            guard !deferredMicrophoneFrames.isEmpty else { break }
            let frame = deferredMicrophoneFrames.removeFirst()
            let event = try advance(
                audioEmbedding: frame.embedding,
                record: true,
                recordEvent: recordEvent,
                forceSilent: false,
                audioMilliseconds: frame.audioMilliseconds,
                perceptionLatencyMilliseconds: 0,
                userTranscript: frame.userTranscript,
                rnntIsBlank: frame.rnntIsBlank,
                rnntHasLexicalToken: frame.rnntHasLexicalToken,
                deferredInputReplay: true,
                deferIdleSpeechCache: true)
            events.append(event)
            replayedDeferredMicrophoneFrames += 1
            startFunctionFastPathIfNeeded()

            if Self.usesPerceptionOnlyMicrophonePath(
                fastPathRunning: functionFastPathTask != nil,
                hasFastPathWork: hasFunctionFastPathWork,
                awaitingFunctionResponse: awaitingFunctionResponse)
            {
                break
            }
        }
        return events
    }

    private var hasFunctionFastPathWork: Bool {
        collectingFunctionCall
            || forcedFunctionTokenIndex < forcedFunctionTokens.count
    }

    private func beginFunctionCallDecode() {
        functionCallDecodeStartedAtNanoseconds = DispatchTime.now()
            .uptimeNanoseconds
        functionCallDecodeSteps = 0
        functionCallModelMilliseconds = 0
        functionCallSpeechCacheMilliseconds = 0
    }

    private func beginFunctionResponseMetrics() {
        functionResponseStartedAtNanoseconds = DispatchTime.now()
            .uptimeNanoseconds
        functionResponseSteps = 0
        functionResponsePrefillBatches = 0
        functionResponseLanguageCacheMilliseconds = 0
        functionResponseSpeechCacheMilliseconds = 0
    }

    private func finishFunctionCallDecode(completed: Bool) {
        guard let started = functionCallDecodeStartedAtNanoseconds else {
            return
        }
        lastFunctionCallDecodeMetrics = VoiceChatFunctionCallDecodeMetrics(
            active: false,
            completed: completed,
            elapsedMilliseconds: Self.elapsedMilliseconds(since: started),
            tokenSteps: functionCallDecodeSteps,
            modelMilliseconds: functionCallModelMilliseconds,
            speechCacheMilliseconds: functionCallSpeechCacheMilliseconds)
        functionCallDecodeStartedAtNanoseconds = nil
        functionCallDecodeSteps = 0
        functionCallModelMilliseconds = 0
        functionCallSpeechCacheMilliseconds = 0
    }

    private func finishFunctionResponseMetrics(completed: Bool) {
        guard let started = functionResponseStartedAtNanoseconds else {
            return
        }
        lastFunctionResponseMetrics = VoiceChatFunctionResponseMetrics(
            active: false,
            completed: completed,
            elapsedMilliseconds: Self.elapsedMilliseconds(since: started),
            tokenSteps: functionResponseSteps,
            prefillBatches: functionResponsePrefillBatches,
            languageCacheMilliseconds:
                functionResponseLanguageCacheMilliseconds,
            speechCacheMilliseconds:
                functionResponseSpeechCacheMilliseconds)
        functionResponseStartedAtNanoseconds = nil
        functionResponseSteps = 0
        functionResponsePrefillBatches = 0
        functionResponseLanguageCacheMilliseconds = 0
        functionResponseSpeechCacheMilliseconds = 0
    }

    private static func elapsedMilliseconds(since started: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
    }

    /// System-prompt positions are conditioning inputs, not generated channel
    /// history. NeMo fills both generated channels with PAD over that prefix.
    /// Feeding their unsupervised predictions into the next prompt position can
    /// leave the function channel midway through a hallucinated call when live
    /// audio starts.
    static func channelFeedbackAfterStep(
        record: Bool,
        textToken: Int,
        functionToken: Int,
        padID: Int
    ) -> (text: Int, function: Int) {
        record
            ? (textToken, functionToken)
            : (padID, padID)
    }

    private func flushIdleSpeechFrames(
        state: VoiceChatSpeechDecoderState,
        previousCode: MLXArray,
        maximumFrames: Int? = nil
    ) throws {
        guard pendingIdleSpeechFrames > 0 else { return }
        let frameCount = min(
            pendingIdleSpeechFrames,
            maximumFrames ?? pendingIdleSpeechFrames)
        let code = model.speechDecoder.advanceIdleSilence(
            state: state,
            previousCode: previousCode,
            frames: frameCount,
            guidance: speechParameters.guidance)
        evaluateSpeech(code, state: state)
        self.previousSpeechCode = code
        pendingIdleSpeechFrames -= frameCount
        deferredReplayPendingIdleSpeechFrames = max(
            0, deferredReplayPendingIdleSpeechFrames - frameCount)
    }

    private func evaluateSpeech(
        _ code: MLXArray,
        state: VoiceChatSpeechDecoderState
    ) {
        let cacheState = state.attention.flatMap { cache in
            [cache.keys, cache.values].compactMap { $0 }
        }
        // Materialize every retained cache root with the generated code so
        // lazy concatenations cannot accumulate across a long conversation.
        MLX.eval([code] + cacheState)
    }

    private func sampleTextToken(_ inputLogits: MLXArray, greedy: Int) -> Int {
        let logits = inputLogits.asType(.float32)
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

extension VoiceChatSession {
    /// Test-only entry point for proving that the optimized phase-two replay
    /// preserves the same cache state as one-token decoding.
    func prefillFunctionResponseForTesting(
        audioEmbedding: MLXArray
    ) async throws -> Int {
        try await prefillFunctionResponse(audioEmbedding: audioEmbedding)
    }

    /// Disable the background task only in parity tests so the sequential and
    /// batched paths can be advanced deterministically from the same prefix.
    func suppressFunctionFastPathForTesting() async {
        functionFastPathSuppressedForTesting = true
        functionFastPathTask?.cancel()
        if let functionFastPathTask {
            await functionFastPathTask.value
        }
        self.functionFastPathTask = nil
    }

    /// Test-only boundary for exercising deferred-input capacity without
    /// relying on the checkpoint to select a particular native tool first.
    /// The real model still evaluates every microphone frame through its
    /// perception path before the embedding reaches the bounded queue.
    func beginDeferredMicrophoneCaptureForTesting() {
        precondition(functionCallingEnabled)
        awaitingFunctionResponse = true
    }

    /// Reproduce the former one-token response replay for cache-parity tests.
    func advanceFunctionResponseSequentiallyForTesting(
        audioEmbedding: MLXArray
    ) throws -> Int {
        var processed = 0
        while forcedFunctionTokenIndex < forcedFunctionTokens.count {
            _ = try advance(
                audioEmbedding: audioEmbedding,
                record: true,
                recordEvent: false,
                forceSilent: false,
                audioMilliseconds: 0,
                perceptionLatencyMilliseconds: 0,
                userTranscript: nil,
                rnntIsBlank: true,
                rnntHasLexicalToken: false)
            processed += 1
        }
        try flushIdleSpeechFrames(
            state: speechState!, previousCode: previousSpeechCode!)
        return processed
    }

    func languageCacheStateForTesting() -> [[MLXArray]] {
        languageCache.map(\.state)
    }

    func speechCacheStateForTesting() -> [[MLXArray]] {
        speechState?.attention.map {
            [$0.keys, $0.values].compactMap { $0 }
        } ?? []
    }

    func generatedCodeCountForTesting() -> Int {
        generatedCodes.count
    }

    func functionSilenceEmbeddingForTesting() -> MLXArray? {
        functionSilenceEmbedding
    }

    func turnTakingStateForTesting() -> VoiceChatRNNTTurnTakingState {
        turnTakingState
    }

    func deferredMicrophoneStatisticsForTesting()
        -> VoiceChatDeferredMicrophoneStatistics
    {
        VoiceChatDeferredMicrophoneStatistics(
            bufferedFrames: deferredMicrophoneFrames.count,
            replayedFrames: replayedDeferredMicrophoneFrames,
            droppedFrames: deferredMicrophoneDroppedFrames,
            pendingSpeechCacheFrames:
                deferredReplayPendingIdleSpeechFrames)
    }

}
