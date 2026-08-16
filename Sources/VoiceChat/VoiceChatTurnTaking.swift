/// RNN-T turn-taking thresholds used by a VoiceChat session.
///
/// NVIDIA's realtime wrapper treats the transcript head's first prediction for
/// each 80 ms encoder frame as a speech signal: blank frames accumulate toward
/// user end-of-utterance, while non-blank frames confirm speech and barge-in.
public struct VoiceChatTurnTakingParameters: Sendable, Equatable {
    public var enabled: Bool
    public var endOfUtteranceFrames: Int
    public var beginOfUtteranceFrames: Int
    /// Silence required to commit a model-native `<SPECIAL_20>` candidate.
    /// `nil` uses `endOfUtteranceFrames`. Tool-enabled live sessions use a
    /// shorter candidate-only threshold: the function head has already added
    /// semantic evidence, while ordinary conversation keeps the conservative
    /// NVIDIA fallback.
    public var functionCallEndOfUtteranceFrames: Int?
    public var minimumSpeechFrames: Int
    public var firstTurnMinimumSpeechFrames: Int
    public var noiseResetFrames: Int
    public var suppressUnpromptedTurns: Bool
    /// Use the RNN-T blank threshold as a fallback that forces agent BOS.
    public var forceAgentBeginOnEndOfUtterance: Bool
    /// Permit a model-native first BOS before user speech is confirmed.
    /// Normal microphone sessions keep this false; explicit greeting mode
    /// enables it so the prompt can intentionally open the conversation.
    public var allowInitialAgentTurn: Bool

    /// NVIDIA's deployed streaming configuration uses a 40-frame (3.2 s)
    /// RNN-T silence window. This is deliberately a safety fallback: the
    /// checkpoint's learned text head may emit BOS earlier when it decides the
    /// user has completed a turn.
    public static let nvidiaTurnTakingFallbackFrames = 40

    public init(
        enabled: Bool = true,
        endOfUtteranceFrames: Int = Self.nvidiaTurnTakingFallbackFrames,
        beginOfUtteranceFrames: Int = Self.nvidiaTurnTakingFallbackFrames,
        functionCallEndOfUtteranceFrames: Int? = nil,
        minimumSpeechFrames: Int = 3,
        firstTurnMinimumSpeechFrames: Int = 2,
        noiseResetFrames: Int = 10,
        suppressUnpromptedTurns: Bool = true,
        forceAgentBeginOnEndOfUtterance: Bool = true,
        allowInitialAgentTurn: Bool = false
    ) {
        self.enabled = enabled
        self.endOfUtteranceFrames = endOfUtteranceFrames
        self.beginOfUtteranceFrames = beginOfUtteranceFrames
        self.functionCallEndOfUtteranceFrames =
            functionCallEndOfUtteranceFrames
        self.minimumSpeechFrames = minimumSpeechFrames
        self.firstTurnMinimumSpeechFrames = firstTurnMinimumSpeechFrames
        self.noiseResetFrames = noiseResetFrames
        self.suppressUnpromptedTurns = suppressUnpromptedTurns
        self.forceAgentBeginOnEndOfUtterance =
            forceAgentBeginOnEndOfUtterance
        self.allowInitialAgentTurn = allowInitialAgentTurn
    }

    /// Preserve the model's learned BOS/EOS behavior without RNN-T overrides.
    public static let modelNative = VoiceChatTurnTakingParameters(enabled: false)

    /// NVIDIA realtime policy: learned BOS remains the normal response trigger,
    /// while 3.2 seconds of RNN-T silence is the last-resort forced-BOS window.
    /// NVIDIA's deployed profile uses the same 40-frame safety threshold for
    /// RNN-T barge-in; the checkpoint's learned EOS may still yield earlier.
    public static let nvidiaRealtime = VoiceChatTurnTakingParameters()

    /// Tool-enabled conversations preserve native function selection but can
    /// commit a currently-proposed function candidate after 640 ms of RNN-T
    /// silence. Ordinary responses and barge-in retain the 3.2-second safety
    /// window, so this does not globally shorten natural user pauses. The live
    /// session retains only the latest model candidate from one uninterrupted
    /// speech segment and discards it as soon as user speech resumes.
    public static let functionCallingRealtime = VoiceChatTurnTakingParameters(
        functionCallEndOfUtteranceFrames: 8)
}

public enum VoiceChatTurnTakingAction: Sendable, Equatable {
    case none
    case forcedAgentBegin
    case forcedAgentEnd
    case suppressedUnpromptedBegin
}

struct VoiceChatTurnTakingResult: Sendable, Equatable {
    let token: Int
    let action: VoiceChatTurnTakingAction
}

/// Tracks whether EAR-TTS is inside an agent turn. PAD is always canonical
/// silence while idle. Inside an open turn it keeps rendering the delayed
/// reply until NVIDIA's content-scaled acoustic budget is exhausted.
struct VoiceChatSpeechTurnState: Sendable, Equatable {
    /// NVIDIA's realtime wrapper allows three trailing PAD frames for every
    /// content token in the open turn. EAR-TTS consumes text faster than it
    /// renders audio, so this content-scaled budget preserves the delayed
    /// acoustic tail without running MaskGIT forever when EOS is missing.
    static let acousticPadFramesPerContentToken = 3
    /// Preserve the former 1.28-second grace period for very short replies.
    static let minimumAcousticTailFrames = 16

    static func acousticTailFrameBudget(contentFrames: Int) -> Int {
        max(
            minimumAcousticTailFrames,
            acousticPadFramesPerContentToken * contentFrames)
    }

    let compactExtendedPads: Bool
    private(set) var agentIdle = true
    private(set) var consecutivePadFrames = 0
    private(set) var contentFrames = 0

    init(compactExtendedPads: Bool = false) {
        self.compactExtendedPads = compactExtendedPads
    }

    mutating func shouldForceSilence(
        textToken: Int,
        padID: Int,
        bosID: Int,
        eosID: Int
    ) -> Bool {
        if textToken == bosID {
            agentIdle = false
            consecutivePadFrames = 0
            contentFrames = 0
            return false
        }
        if textToken == eosID {
            agentIdle = true
            consecutivePadFrames = 0
            contentFrames = 0
            return true
        }
        if textToken == padID {
            consecutivePadFrames += 1
            guard !agentIdle else { return true }
            guard compactExtendedPads, contentFrames > 0 else { return false }
            return consecutivePadFrames
                > Self.acousticTailFrameBudget(contentFrames: contentFrames)
        }
        consecutivePadFrames = 0
        if !agentIdle { contentFrames += 1 }
        return false
    }
}

/// Stateful port of the RNN-T blank/non-blank policy in NVIDIA's realtime
/// VoiceChat wrapper. The policy changes text-channel control tokens; it never
/// drops or compresses microphone audio.
struct VoiceChatRNNTTurnTakingState: Sendable {
    let parameters: VoiceChatTurnTakingParameters

    private(set) var consecutiveBlankFrames = 0
    private(set) var consecutiveSpeechFrames = 0
    private(set) var totalSpeechFrames = 0
    private(set) var speechConfirmed = false
    private(set) var agentSpeaking = false
    private(set) var firstAgentTurn = true
    private(set) var firstUserTurn = true
    private(set) var consecutiveAgentPadFrames = 0
    private(set) var agentContentFrames = 0

    init(parameters: VoiceChatTurnTakingParameters) {
        self.parameters = parameters
    }

    mutating func selectToken(
        proposedToken: Int,
        rnntIsBlank: Bool,
        padID: Int,
        bosID: Int,
        eosID: Int
    ) -> VoiceChatTurnTakingResult {
        guard parameters.enabled else {
            return .init(token: proposedToken, action: .none)
        }

        updateActivity(rnntIsBlank: rnntIsBlank)
        updateUserSpeechConfirmation()

        // NVIDIA suppresses a fresh model-native BOS after an agent turn until
        // the RNN-T head has confirmed new user speech. This prevents silence
        // from starting repeated self-play turns.
        if parameters.suppressUnpromptedTurns,
           proposedToken == bosID,
           !agentSpeaking,
           !speechConfirmed,
           (!firstAgentTurn || !parameters.allowInitialAgentTurn)
        {
            return .init(token: padID, action: .suppressedUnpromptedBegin)
        }

        if proposedToken == bosID {
            beginAgentTurn(userTurnCompleted: speechConfirmed)
        } else if proposedToken == eosID {
            endAgentTurn()
        } else if agentSpeaking {
            if proposedToken == padID, rnntIsBlank {
                consecutiveAgentPadFrames += 1
                if agentContentFrames > 0,
                   consecutiveAgentPadFrames
                    > VoiceChatSpeechTurnState.acousticTailFrameBudget(
                        contentFrames: agentContentFrames)
                {
                    endAgentTurn()
                    return .init(token: eosID, action: .none)
                }
            } else {
                consecutiveAgentPadFrames = 0
                if proposedToken != padID { agentContentFrames += 1 }
            }
        }

        // User EOU: confirmed speech followed by enough RNN-T blanks opens the
        // agent turn even if the language head has not emitted BOS itself.
        if !agentSpeaking,
           parameters.forceAgentBeginOnEndOfUtterance,
           speechConfirmed,
           consecutiveBlankFrames >= parameters.endOfUtteranceFrames,
           proposedToken != bosID
        {
            beginAgentTurn(userTurnCompleted: true)
            return .init(token: bosID, action: .forcedAgentBegin)
        }

        // User BOU while the agent is talking: sustained RNN-T non-blanks close
        // the agent turn, enabling low-latency barge-in without a separate VAD.
        if agentSpeaking,
           consecutiveSpeechFrames >= parameters.beginOfUtteranceFrames,
           proposedToken != eosID
        {
            endAgentTurn()
            consecutiveSpeechFrames = 0
            totalSpeechFrames = 0
            return .init(token: eosID, action: .forcedAgentEnd)
        }

        return .init(token: proposedToken, action: .none)
    }

    private mutating func updateActivity(rnntIsBlank: Bool) {
        if rnntIsBlank {
            consecutiveBlankFrames += 1
            consecutiveSpeechFrames = 0
        } else {
            consecutiveBlankFrames = 0
            consecutiveSpeechFrames += 1
            totalSpeechFrames += 1
        }
    }

    /// Update user-turn evidence before any token override can return. In
    /// particular, a model that repeatedly proposes BOS while idle must not
    /// prevent RNN-T non-blanks from arming the first user turn forever.
    private mutating func updateUserSpeechConfirmation() {
        guard !agentSpeaking else { return }
        if consecutiveBlankFrames >= parameters.noiseResetFrames,
           !speechConfirmed
        {
            totalSpeechFrames = 0
        }

        let minimumSpeech = firstUserTurn
            ? parameters.firstTurnMinimumSpeechFrames
            : parameters.minimumSpeechFrames
        if consecutiveSpeechFrames >= minimumSpeech
            || totalSpeechFrames >= minimumSpeech
        {
            speechConfirmed = true
        }
    }

    /// Clear only unfinished user-activity evidence after the live input queue
    /// drops stale audio. Agent/turn history remains intact, while partial
    /// speech before the discontinuity cannot force a response afterward.
    mutating func resynchronizeInput(
        preserveSpeechConfirmation: Bool = false
    ) {
        let hadConfirmedSpeech = speechConfirmed
        consecutiveBlankFrames = 0
        consecutiveSpeechFrames = 0
        totalSpeechFrames = 0
        speechConfirmed = preserveSpeechConfirmation && hadConfirmedSpeech
    }

    /// Start a function-channel turn after the user's request has been
    /// accepted. Function decoding runs independently of the microphone clock,
    /// so old end-of-utterance evidence must not be mistaken for a new barge-in.
    mutating func beginFunctionCall() {
        if agentSpeaking { endAgentTurn() }
        firstUserTurn = false
        speechConfirmed = false
        consecutiveBlankFrames = 0
        consecutiveSpeechFrames = 0
        totalSpeechFrames = 0
    }

    /// Resume the assistant side of the already-authorized user turn after a
    /// function response has entered the model cache. The function call
    /// deliberately consumes the old RNN-T activity so it cannot look like a
    /// barge-in; the response marker then restores only the authorization to
    /// continue. Priming the EOU count lets either the text head or a chained
    /// function call proceed on the next blank frame without another 3.2-second
    /// silence window.
    mutating func resumeAfterFunctionResponse(
        deferredUserInput: Bool = false
    ) {
        guard parameters.enabled else { return }
        if agentSpeaking { endAgentTurn() }
        firstUserTurn = false
        speechConfirmed = !deferredUserInput
        consecutiveBlankFrames = deferredUserInput
            ? 0
            : max(0, parameters.endOfUtteranceFrames - 1)
        consecutiveSpeechFrames = 0
        totalSpeechFrames = 0
    }

    /// Observe microphone activity while the language model is generating a
    /// function call on its asynchronous silence path. A sustained fresh
    /// RNN-T non-blank sequence is an interruption, matching NVIDIA's live
    /// background-perception worker rather than blocking capture behind the
    /// function projection.
    mutating func observeFunctionCallInput(rnntIsBlank: Bool) -> Bool {
        guard parameters.enabled else { return false }
        updateActivity(rnntIsBlank: rnntIsBlank)
        guard consecutiveSpeechFrames >= parameters.beginOfUtteranceFrames else {
            return false
        }
        speechConfirmed = true
        return true
    }

    /// Preserve user activity while old microphone embeddings catch up after
    /// tool work, without allowing the text channel to open an assistant turn
    /// before all already-captured speech has reached the shared timeline.
    mutating func observeDeferredInput(rnntIsBlank: Bool) {
        guard parameters.enabled else { return }
        updateActivity(rnntIsBlank: rnntIsBlank)
        updateUserSpeechConfirmation()
    }

    /// A real lexical RNN-T emission is stronger user-speech evidence than the
    /// first-prediction frame counter alone. This keeps short follow-up turns
    /// such as "yes" or "who are you" from waiting for more speech merely
    /// because their complete token sequence was emitted inside one frame.
    /// Barge-in remains governed by the sustained-frame threshold above.
    mutating func observeRecognizedSpeechToken() {
        guard parameters.enabled, !agentSpeaking else { return }
        speechConfirmed = true
    }

    private mutating func beginAgentTurn(userTurnCompleted: Bool) {
        agentSpeaking = true
        firstAgentTurn = false
        consecutiveAgentPadFrames = 0
        agentContentFrames = 0
        if userTurnCompleted {
            firstUserTurn = false
        }
        speechConfirmed = false
        totalSpeechFrames = 0
        // Speech that caused or preceded BOS is part of the completed user
        // turn. Barge-in must be armed only by fresh speech after the agent
        // starts; otherwise an early model-native BOS can immediately force
        // its own EOS and audibly truncate the response.
        consecutiveSpeechFrames = 0
        consecutiveBlankFrames = 0
    }

    private mutating func endAgentTurn() {
        agentSpeaking = false
        speechConfirmed = false
        totalSpeechFrames = 0
        consecutiveAgentPadFrames = 0
        agentContentFrames = 0
    }
}
