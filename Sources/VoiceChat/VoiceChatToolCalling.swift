import Foundation

enum VoiceChatFunctionHeadEvaluation: Sendable, Equatable {
    case none
    /// Compare only the PAD and `<SPECIAL_20>` rows. A full projection verifies
    /// the start token only if this exact necessary condition passes.
    case startProbe
    /// Generate an already-open tool call over the complete vocabulary.
    case fullVocabulary
}

/// Prevents a confirmation-required tool result from immediately starting a
/// second native call before the checkpoint has spoken to the user. This gate
/// observes only model protocol tokens; it never inspects transcript text.
struct VoiceChatFunctionStartGate: Sendable, Equatable {
    private(set) var isSuspended = false

    var allowsFunctionStarts: Bool { !isSuspended }

    mutating func requireAssistantTurn(_ required: Bool) {
        isSuspended = required
    }

    /// The first assistant BOS proves that the checkpoint has opened the
    /// confirmation turn. Normal assistant-turn and fresh-speech guards then
    /// protect every later function start.
    mutating func observeTextToken(_ token: Int, bosID: Int) {
        if isSuspended, token == bosID {
            isSuspended = false
        }
    }

    mutating func reset() {
        isSuspended = false
    }
}

struct VoiceChatFunctionHeadEvaluationStatistics: Sendable, Equatable {
    let probeFrames: Int
    let fullVerificationFrames: Int
    let openCallFrames: Int
    let asynchronousCallSteps: Int
    let asynchronousResponseSteps: Int
    /// Number of bounded causal-prefill calls used for known tool responses.
    /// This should be much smaller than `asynchronousResponseSteps`.
    let asynchronousResponsePrefillBatches: Int
    let asynchronousCallTimeouts: Int
    let asynchronousResponseTimeouts: Int
    let asynchronousCallInterruptions: Int
    let microphoneFramesDuringAsyncWork: Int
    let verificationWinners: [Int: Int]
}

/// Wall-clock cost of decoding a model-native function call. Audio RTF omits
/// this asynchronous work, so clients should display it separately when they
/// need to explain tool latency.
public struct VoiceChatFunctionCallDecodeMetrics: Sendable, Equatable {
    public let active: Bool
    public let completed: Bool
    public let elapsedMilliseconds: Double
    public let tokenSteps: Int
    /// Shared Nemotron-H plus native function-projection wall time accumulated
    /// across generated call steps.
    public let modelMilliseconds: Double
    /// Idle EAR-TTS cache maintenance charged to the same function timeline.
    public let speechCacheMilliseconds: Double

    public init(
        active: Bool,
        completed: Bool,
        elapsedMilliseconds: Double,
        tokenSteps: Int,
        modelMilliseconds: Double = 0,
        speechCacheMilliseconds: Double = 0
    ) {
        self.active = active
        self.completed = completed
        self.elapsedMilliseconds = elapsedMilliseconds
        self.tokenSteps = tokenSteps
        self.modelMilliseconds = modelMilliseconds
        self.speechCacheMilliseconds = speechCacheMilliseconds
    }

    public var tokensPerSecond: Double {
        guard elapsedMilliseconds > 0 else { return 0 }
        return Double(tokenSteps) * 1_000 / elapsedMilliseconds
    }

    /// Time outside the measured Nemotron-H/function projection and EAR-TTS
    /// cache updates. This includes actor yields that let live microphone work
    /// run, token bookkeeping, and scheduler/system contention.
    public var interleavingMilliseconds: Double {
        max(0, elapsedMilliseconds
            - modelMilliseconds
            - speechCacheMilliseconds)
    }
}

/// Wall-clock cost of replaying a completed external result through the
/// function channel. This is separate from both native call generation and
/// foreground audio RTF. Known result tokens are normally processed in causal
/// prefill batches rather than decoded one at a time.
public struct VoiceChatFunctionResponseMetrics: Sendable, Equatable {
    public let active: Bool
    public let completed: Bool
    public let elapsedMilliseconds: Double
    public let tokenSteps: Int
    public let prefillBatches: Int
    /// Nemotron-H causal-prefill wall time for the known result tokens.
    public let languageCacheMilliseconds: Double
    /// EAR-TTS idle-cache wall time needed to preserve the shared timeline.
    public let speechCacheMilliseconds: Double

    public init(
        active: Bool,
        completed: Bool,
        elapsedMilliseconds: Double,
        tokenSteps: Int,
        prefillBatches: Int,
        languageCacheMilliseconds: Double = 0,
        speechCacheMilliseconds: Double = 0
    ) {
        self.active = active
        self.completed = completed
        self.elapsedMilliseconds = elapsedMilliseconds
        self.tokenSteps = tokenSteps
        self.prefillBatches = prefillBatches
        self.languageCacheMilliseconds = languageCacheMilliseconds
        self.speechCacheMilliseconds = speechCacheMilliseconds
    }

    public var tokensPerSecond: Double {
        guard elapsedMilliseconds > 0 else { return 0 }
        return Double(tokenSteps) * 1_000 / elapsedMilliseconds
    }

    /// Time outside the measured language- and speech-cache evaluations. This
    /// includes bounded between-batch yields, bookkeeping, and scheduler/system
    /// contention while microphone work remains eligible to run.
    public var interleavingMilliseconds: Double {
        max(0, elapsedMilliseconds
            - languageCacheMilliseconds
            - speechCacheMilliseconds)
    }
}

/// One coherent snapshot of the model-native function channel. Live clients
/// should prefer this over several independent actor calls so diagnostics do
/// not repeatedly interleave with asynchronous 11B tool decoding.
package struct VoiceChatFunctionRuntimeStatus: Sendable, Equatable {
    package let generatingCall: Bool
    package let waitingForResponse: Bool
    package let callDecode: VoiceChatFunctionCallDecodeMetrics?
    package let responseSync: VoiceChatFunctionResponseMetrics?

    package init(
        generatingCall: Bool,
        waitingForResponse: Bool,
        callDecode: VoiceChatFunctionCallDecodeMetrics?,
        responseSync: VoiceChatFunctionResponseMetrics?
    ) {
        self.generatingCall = generatingCall
        self.waitingForResponse = waitingForResponse
        self.callDecode = callDecode
        self.responseSync = responseSync
    }
}

/// Selects the smallest function-head operation that can preserve exact tool
/// start detection. The 131k-row projection stays dormant until either the
/// two-row probe says `<SPECIAL_20>` may be the global argmax or a call is
/// already open.
struct VoiceChatFunctionHeadScheduler: Sendable {
    func evaluation(
        enabled: Bool,
        recording: Bool,
        forceSilent: Bool,
        collectingCall: Bool,
        outputInjectionActive: Bool,
        turnTakingEnabled: Bool,
        userSpeechConfirmed: Bool,
        assistantTurnActive: Bool
    ) -> VoiceChatFunctionHeadEvaluation {
        guard enabled, recording, !forceSilent else {
            return .none
        }
        if collectingCall { return .fullVocabulary }
        guard !outputInjectionActive, !assistantTurnActive else { return .none }
        guard !turnTakingEnabled || userSpeechConfirmed else { return .none }
        return .startProbe
    }
}

public extension VoiceChatSession {
    /// Build the function-calling prompt used by the released VoiceChat
    /// checkpoint. `availableToolsJSON` must be a compact JSON array containing
    /// at most five provider-neutral tool definitions.
    static func toolCallingSystemPrompt(
        basePrompt: String = personaSystemPrompt,
        availableToolsJSON: String,
        greet: Bool = false,
        requiresWriteConfirmation: Bool = false
    ) throws -> String {
        guard basePrompt.unicodeScalars.allSatisfy({ $0.isASCII }),
              availableToolsJSON.unicodeScalars.allSatisfy({ $0.isASCII }) else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "VoiceChat tool prompts must contain ASCII text only")
        }
        guard let data = availableToolsJSON.data(using: .utf8),
              let tools = try JSONSerialization.jsonObject(with: data)
                as? [[String: Any]],
              !tools.isEmpty,
              tools.count <= 5 else {
            throw VoiceChatGenerationError.invalidSpeechConfiguration(
                "VoiceChat requires between one and five tool definitions")
        }

        var prompt = basePrompt
        if greet {
            prompt += " Start the conversation by greeting the user."
        }
        prompt += "\n\nWhen you receive a request, follow this decision process:"
        prompt += "\n1. Does the request match one of your available tools below?"
        prompt += " If yes, you MUST call that tool - never answer it directly"
        prompt += " from your own knowledge."
        prompt += "\n2. Is it a general knowledge question? If yes, answer"
        prompt += " directly from your own knowledge - do not call any tool."
        prompt += "\n3. Does it require an external action or live data that none"
        prompt += " of your tools cover? If yes, say in one short sentence that"
        prompt += " the action is unavailable. Do not list unrelated limitations."
        prompt += "\n\nMatch requests to tools by meaning and description; users"
        prompt += " speak naturally and do not need to say a tool identifier."
        prompt += " Never answer from memory for live state that an available"
        prompt += " tool can read. Never invent tools. Tool-call arguments must"
        prompt += " be values the user spoke or exact identifiers returned by"
        prompt += " earlier tool results. If a required argument is missing,"
        prompt += " use an available read-only lookup tool to obtain valid"
        prompt += " choices, or ask the user; never guess. Tool providers may"
        prompt += " apply schema-documented defaults. Omit optional"
        prompt += " arguments that the user did not provide. For paired tools such"
        prompt += " as list and detail tools, use the broad list tool when the"
        prompt += " user does not name a specific item."
        prompt += " When asked what you can do, describe available tools"
        prompt += " positively and briefly. Ask only for the next required"
        prompt += " missing value, never a group of optional fields."
        prompt += "\n\nDo not claim an external action succeeded until its tool"
        prompt += " response says it succeeded."
        if requiresWriteConfirmation {
            prompt += " A confirmation_required response is not a failure: it"
            prompt += " means nothing was created or changed yet. Tell the user"
            prompt += " what would happen, ask for confirmation once, and wait."
            prompt += " After the user confirms, emit the exact same tool call"
            prompt += " again. If the user declines, answer without a tool call."
        } else {
            prompt += " When a write tool reports success, acknowledge the exact"
            prompt += " completed action briefly. Never ask for confirmation"
            prompt += " before or after a successful tool response."
        }
        prompt += "\n\nIf a tool call fails, is denied, or was already handled, do"
        prompt += " not retry it for the same request. Explain the result once."
        prompt += " If a result contains clarification_required, ask one short"
        prompt += " question for that field; it is missing input, not a provider"
        prompt += " failure. Retry only after the user supplies that value."
        prompt += "\n\nYou can use the following tools to assist the user if required:"
        prompt += "\n<AVAILABLE_TOOLS>\(availableToolsJSON)</AVAILABLE_TOOLS>"
        prompt += "\n\nIf you decide to call a tool, use exactly this format:"
        prompt += "\n<TOOLCALL>[{\"name\":\"tool_name\",\"arguments\":{}}]</TOOLCALL>"
        prompt += "\n\nThe user will return its result in this format:"
        prompt += "\n<TOOL_RESPONSE>[{\"ok\":true}]</TOOL_RESPONSE>"
        prompt += "\n\nUse the result to answer conversationally."
        return prompt
    }
}
