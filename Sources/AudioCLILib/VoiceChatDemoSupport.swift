import Foundation
import VoiceChat

/// Bounded producer/consumer buffer between Core Audio and model inference.
/// Core Audio never waits for MLX. When sustained inference falls behind, the
/// buffer discards only enough of the oldest queued audio to admit fresh
/// capture. A continuous overload is reported as one recovery episode. This
/// bounds interaction latency without turning several small overflows into the
/// loss of an entire utterance.
final class VoiceChatInputBuffer: @unchecked Sendable {
    struct Statistics: Sendable, Equatable {
        let bufferedFrames: Double
        let droppedSamples: Int
        let resynchronizations: Int
    }

    let frameSize: Int
    let maximumFrames: Int

    private let condition = NSCondition()
    private var samples: [Float] = []
    private var readIndex = 0
    private var closed = false
    private var droppedSamples = 0
    private var resynchronizations = 0
    private var overloadActive = false

    init(frameSize: Int, maximumFrames: Int) {
        precondition(frameSize > 0)
        precondition(maximumFrames > 0)
        self.frameSize = frameSize
        self.maximumFrames = maximumFrames
        samples.reserveCapacity(frameSize * maximumFrames)
    }

    @discardableResult
    func append(_ input: [Float]) -> Bool {
        guard !input.isEmpty else { return true }
        condition.lock()
        defer { condition.unlock() }
        guard !closed else { return false }

        let maximumSamples = frameSize * maximumFrames
        if input.count > maximumSamples {
            let available = samples.count - readIndex
            droppedSamples += available + input.count - maximumSamples
            samples.removeAll(keepingCapacity: true)
            readIndex = 0
            samples.append(contentsOf: input.suffix(maximumSamples))
            if !overloadActive {
                resynchronizations += 1
                overloadActive = true
            }
        } else {
            let available = samples.count - readIndex
            let overflow = max(0, available + input.count - maximumSamples)
            if overflow > 0 {
                // Drop only enough old audio to admit the newest callback.
                // Clearing the entire queue turns a small overload into a lost
                // utterance and repeatedly resets RNN-T turn evidence.
                readIndex += overflow
                droppedSamples += overflow
                if !overloadActive {
                    resynchronizations += 1
                    overloadActive = true
                }
                if readIndex >= frameSize * 4 {
                    samples.removeFirst(readIndex)
                    readIndex = 0
                }
            }
            samples.append(contentsOf: input)
        }
        condition.signal()
        return true
    }

    func nextFrame() -> [Float]? {
        condition.lock()
        defer { condition.unlock() }
        while samples.count - readIndex < frameSize,
              !closed {
            condition.wait()
        }
        guard samples.count - readIndex >= frameSize else { return nil }

        let end = readIndex + frameSize
        let frame = Array(samples[readIndex ..< end])
        readIndex = end
        if readIndex >= frameSize * 4 {
            samples.removeFirst(readIndex)
            readIndex = 0
        }
        if samples.count - readIndex <= frameSize * maximumFrames / 2 {
            overloadActive = false
        }
        return frame
    }

    func close() {
        condition.lock()
        closed = true
        condition.broadcast()
        condition.unlock()
    }

    func statistics() -> Statistics {
        condition.lock()
        defer { condition.unlock() }
        return Statistics(
            bufferedFrames: Double(samples.count - readIndex) / Double(frameSize),
            droppedSamples: droppedSamples,
            resynchronizations: resynchronizations)
    }
}

/// Distinguish a short queue overflow from a transcript-breaking gap.
///
/// Dropping one 80 ms frame is undesirable, but resetting the complete RNN-T
/// predictor at that point loses substantially more linguistic context than
/// the missing audio itself. The realtime governor still reacts immediately
/// to every overload episode. Decoder/turn state is reset only when cumulative
/// loss in that episode reaches the configured hard-discontinuity threshold.
struct VoiceChatInputDiscontinuityPolicy: Sendable, Equatable {
    struct Decision: Sendable, Equatable {
        let overloadStarted: Bool
        let requiresDecoderReset: Bool
    }

    let decoderResetSamples: Int

    private var observedEpisodes = 0
    private var lastObservedDroppedSamples = 0
    private var episodeBaselineDroppedSamples = 0
    private var resetIssuedForEpisode = false

    init(decoderResetSamples: Int) {
        precondition(decoderResetSamples > 0)
        self.decoderResetSamples = decoderResetSamples
    }

    mutating func observe(
        _ statistics: VoiceChatInputBuffer.Statistics
    ) -> Decision {
        let overloadStarted = statistics.resynchronizations > observedEpisodes
        if overloadStarted {
            observedEpisodes = statistics.resynchronizations
            episodeBaselineDroppedSamples = lastObservedDroppedSamples
            resetIssuedForEpisode = false
        }

        let droppedThisEpisode = max(
            0, statistics.droppedSamples - episodeBaselineDroppedSamples)
        let requiresDecoderReset = !resetIssuedForEpisode
            && droppedThisEpisode >= decoderResetSamples
        if requiresDecoderReset {
            resetIssuedForEpisode = true
        }
        lastObservedDroppedSamples = statistics.droppedSamples
        return Decision(
            overloadStarted: overloadStarted,
            requiresDecoderReset: requiresDecoderReset)
    }
}

/// Hysteretic quality governor for the live 80 ms model clock. It lowers only
/// the iterative EAR-TTS refinement work; perception, language, RNN-T, codec,
/// and every accepted microphone frame continue to run normally.
struct VoiceChatRealtimeGovernor: Sendable, Equatable {
    let preferredIterations: Int
    let fallbackIterations: Int
    let emergencyIterations: Int
    let activationBufferedFrames: Double
    let restorationFrames: Int
    let activationFrameMilliseconds: Double
    let emergencyFrameMilliseconds: Double

    private(set) var currentIterations: Int
    private(set) var stableFrames = 0

    init(
        preferredIterations: Int,
        fallbackIterations: Int,
        emergencyIterations: Int = 1,
        activationBufferedFrames: Double = 3,
        restorationFrames: Int = 100,
        activationFrameMilliseconds: Double = 88,
        emergencyFrameMilliseconds: Double = 120
    ) {
        precondition(preferredIterations > 0)
        precondition(fallbackIterations > 0)
        precondition(fallbackIterations <= preferredIterations)
        precondition(emergencyIterations > 0)
        precondition(emergencyIterations <= fallbackIterations)
        precondition(activationBufferedFrames > 0)
        precondition(restorationFrames > 0)
        precondition(activationFrameMilliseconds > 0)
        precondition(emergencyFrameMilliseconds >= activationFrameMilliseconds)
        self.preferredIterations = preferredIterations
        self.fallbackIterations = fallbackIterations
        self.emergencyIterations = emergencyIterations
        self.activationBufferedFrames = activationBufferedFrames
        self.restorationFrames = restorationFrames
        self.activationFrameMilliseconds = activationFrameMilliseconds
        self.emergencyFrameMilliseconds = emergencyFrameMilliseconds
        self.currentIterations = preferredIterations
    }

    /// Returns a new iteration count only when the session should be updated.
    mutating func observe(
        bufferedFrames: Double,
        frameComputeMilliseconds: Double?,
        didResynchronize: Bool = false
    ) -> Int? {
        let frameExceededActivation = frameComputeMilliseconds.map {
            $0 >= activationFrameMilliseconds
        } ?? false
        let frameExceededEmergency = frameComputeMilliseconds.map {
            $0 >= emergencyFrameMilliseconds
        } ?? false

        // A single severe callback already consumed more than one live audio
        // period. Go straight to the minimum refinement budget rather than
        // spending another callback at the intermediate setting. The same
        // applies after an input resynchronization, where preserving future
        // microphone audio is more important than acoustic refinement.
        if currentIterations > emergencyIterations,
           (bufferedFrames >= activationBufferedFrames * 2
            || frameExceededEmergency
            || didResynchronize)
        {
            currentIterations = emergencyIterations
            stableFrames = 0
            return currentIterations
        }
        if currentIterations == preferredIterations {
            guard fallbackIterations < preferredIterations,
                  (bufferedFrames >= activationBufferedFrames
                    || frameExceededActivation) else {
                return nil
            }
            currentIterations = fallbackIterations
            stableFrames = 0
            return currentIterations
        }

        // Restore quality only after eight seconds of model-time headroom at
        // the default 100-frame window. Hysteresis prevents quality oscillation.
        if bufferedFrames < 1,
           let frameComputeMilliseconds,
           frameComputeMilliseconds
               <= Double(VoiceChatSession.frameMilliseconds) * 0.8
        {
            stableFrames += 1
        } else {
            stableFrames = 0
        }
        guard stableFrames >= restorationFrames else { return nil }
        currentIterations = preferredIterations
        stableFrames = 0
        return currentIterations
    }

    var isProtectingRealtime: Bool {
        currentIterations < preferredIterations
    }
}

struct VoiceChatConversationLine: Sendable, Equatable {
    enum Role: String, Sendable, Equatable {
        case user = "you"
        case assistant = "ai"
        case tool
        case audio
    }

    let role: Role
    var text: String
    let timestampMilliseconds: Double?

    init(
        role: Role,
        text: String,
        timestampMilliseconds: Double? = nil
    ) {
        self.role = role
        self.text = text
        self.timestampMilliseconds = timestampMilliseconds
    }
}

struct VoiceChatDemoFrame: Sendable, Equatable {
    let index: Int
    let textToken: Int
    let text: String
    let userTranscript: String?
    let turnTakingAction: VoiceChatTurnTakingAction
    let speaking: Bool
    let perceptionLatencyMilliseconds: Double
    let decisionLatencyMilliseconds: Double
    let synthesisLatencyMilliseconds: Double
    let audioPositionMilliseconds: Double
    let audibleAudioEndMillisecondsWithinFrame: Double?

    init(_ event: VoiceChatFrameEvent) {
        index = event.index
        textToken = event.textToken
        text = event.text
        userTranscript = event.userTranscript
        turnTakingAction = event.turnTakingAction
        speaking = event.speaking
        perceptionLatencyMilliseconds = event.perceptionLatencyMilliseconds
        decisionLatencyMilliseconds = event.decisionLatencyMilliseconds
        synthesisLatencyMilliseconds = event.synthesisLatencyMilliseconds
        audioPositionMilliseconds = event.audioPositionMilliseconds
        audibleAudioEndMillisecondsWithinFrame = event.playbackRequired
            ? Self.lastAudibleEndMilliseconds(in: event.audio)
            : nil
    }

    init(
        index: Int,
        textToken: Int,
        text: String,
        userTranscript: String?,
        turnTakingAction: VoiceChatTurnTakingAction = .none,
        speaking: Bool,
        perceptionLatencyMilliseconds: Double = 10,
        decisionLatencyMilliseconds: Double = 35,
        synthesisLatencyMilliseconds: Double = 29,
        audioPositionMilliseconds: Double? = nil,
        audibleAudioEndMillisecondsWithinFrame: Double? = nil
    ) {
        self.index = index
        self.textToken = textToken
        self.text = text
        self.userTranscript = userTranscript
        self.turnTakingAction = turnTakingAction
        self.speaking = speaking
        self.perceptionLatencyMilliseconds = perceptionLatencyMilliseconds
        self.decisionLatencyMilliseconds = decisionLatencyMilliseconds
        self.synthesisLatencyMilliseconds = synthesisLatencyMilliseconds
        self.audioPositionMilliseconds = audioPositionMilliseconds
            ?? Double(index * VoiceChatSession.frameMilliseconds)
        self.audibleAudioEndMillisecondsWithinFrame =
            audibleAudioEndMillisecondsWithinFrame
    }

    /// Find the end of the final sustained 10 ms PCM window above -50 dBFS.
    /// This follows the benchmark's speech-audibility convention and ignores
    /// isolated low-level codec noise while retaining delayed EAR-TTS tails.
    static func lastAudibleEndMilliseconds(
        in audio: [Float]
    ) -> Double? {
        guard !audio.isEmpty else { return nil }
        let windowSamples = max(1, VoiceChatSession.outputSampleRate / 100)
        let threshold = pow(10.0, -50.0 / 20.0)
        var lastEndSample: Int?
        var start = 0
        while start < audio.count {
            let end = min(audio.count, start + windowSamples)
            var energy = 0.0
            for sample in audio[start ..< end] {
                let value = Double(sample)
                energy += value * value
            }
            let rms = sqrt(energy / Double(end - start))
            if rms >= threshold { lastEndSample = end }
            start = end
        }
        guard let lastEndSample else { return nil }
        return Double(lastEndSample) * 1_000
            / Double(VoiceChatSession.outputSampleRate)
    }
}

private struct VoiceChatFrameServiceSample: Sendable, Equatable {
    let milliseconds: Double
    let toolActive: Bool
}

struct VoiceChatDemoState: Sendable, Equatable {
    enum Status: String, Sendable, Equatable {
        case loading = "LOADING"
        case listening = "LISTENING"
        case preparingTool = "PREPARING TOOL"
        case usingTool = "USING TOOL"
        case speaking = "SPEAKING"
        case repeatNeeded = "PLEASE REPEAT"
        case stopped = "STOPPED"
        case error = "ERROR"
    }

    var title = "Soniqo VoiceChat (Nemotron 11B INT5)"
    var status: Status = .loading
    var microphone = "Default microphone"
    var turns = 0
    var lines: [VoiceChatConversationLine] = []
    var currentUserText = ""
    var currentAssistantText = ""
    var lastFrameMilliseconds: Double?
    var averageFrameMilliseconds: Double?
    var averageNormalFrameMilliseconds: Double?
    var averageToolFrameMilliseconds: Double?
    var p95FrameMilliseconds: Double?
    var turnGapMilliseconds: Double?
    var inputBufferedFrames: Double = 0
    var playbackUnderruns = 0
    var forcedTurnStarts = 0
    var forcedBargeIns = 0
    var inputResynchronizations = 0
    var droppedInputMilliseconds: Double = 0
    var preferredSpeechIterations = 8
    var currentSpeechIterations = 8
    var realtimeProtectionActive = false
    var repeatRequestActive = false
    var toolActivityFrames = 0
    var activeToolName: String?
    var lastToolActivity: VoiceChatMCPToolActivity?
    var functionCallDecodeMetrics: VoiceChatFunctionCallDecodeMetrics?
    var functionResponseMetrics: VoiceChatFunctionResponseMetrics?
    var debugTimelineEnabled = false
    private(set) var pendingDecodedToolName: String?

    private var fullUserTranscript = ""
    private var committedUserTranscript = ""
    private var assistantTurnOpen = false
    private var lastUserTextFrame: Int?
    private var frameLatencies: [VoiceChatFrameServiceSample] = []
    private var currentUserStartedAtMilliseconds: Double?
    private var currentAssistantStartedAtMilliseconds: Double?
    private var lastAssistantAudibleEndMilliseconds: Double?
    private var latestAudioPositionMilliseconds: Double = 0
    private var pendingToolActivityBaseline: VoiceChatMCPToolActivity?
    private var pendingFunctionResponseBaseline:
        VoiceChatFunctionResponseMetrics?
    private var lastTimelineToolActivityKey: String?
    private var observedNewFunctionResponse = false
    private var decodedToolCallPending = false

    mutating func ingest(
        _ event: VoiceChatFrameEvent,
        bosID: Int,
        eosID: Int,
        recordFrameLatency: Bool = true
    ) {
        ingest(
            VoiceChatDemoFrame(event),
            bosID: bosID,
            eosID: eosID,
            recordFrameLatency: recordFrameLatency)
    }

    mutating func ingest(
        _ event: VoiceChatDemoFrame,
        bosID: Int,
        eosID: Int,
        recordFrameLatency: Bool = true
    ) {
        latestAudioPositionMilliseconds = event.audioPositionMilliseconds
        if recordFrameLatency {
            observeMicrophoneFrameService(milliseconds:
                event.perceptionLatencyMilliseconds
                    + event.decisionLatencyMilliseconds
                    + event.synthesisLatencyMilliseconds)
        }

        switch event.turnTakingAction {
        case .forcedAgentBegin:
            forcedTurnStarts += 1
        case .forcedAgentEnd:
            forcedBargeIns += 1
        case .none, .suppressedUnpromptedBegin:
            break
        }

        if let transcript = event.userTranscript,
           transcript != fullUserTranscript {
            fullUserTranscript = transcript
            let suffix = transcriptSuffix(
                transcript, after: committedUserTranscript)
            let normalized = normalizedTurnBoundarySuffix(suffix)
            if normalized.isEmpty,
               currentUserText.isEmpty,
               !committedUserTranscript.isEmpty
            {
                // RNN-T can emit final punctuation a frame or two after the
                // assistant has opened its turn. Keep that punctuation on the
                // completed user line instead of showing the next utterance as
                // `? What ...` or `, please ...`.
                appendDelayedPunctuation(suffix)
                committedUserTranscript = transcript
            } else {
                if currentUserText.isEmpty, !normalized.isEmpty {
                    currentUserStartedAtMilliseconds =
                        event.audioPositionMilliseconds
                }
                currentUserText = normalized
                lastUserTextFrame = event.index
            }
        }

        if event.textToken == bosID {
            if assistantTurnOpen {
                // A completed tool result can become ready while the spoken
                // confirmation is still inside its acoustic PAD tail. BOS is
                // always a new assistant turn even if that older turn's EOS
                // has not reached the renderer yet.
                completeAssistantTurn(
                    fallbackEndMilliseconds: event.audioPositionMilliseconds)
            }
            // When the user confirms during that acoustic tail, preserve the
            // chronological order: old assistant, new user, new assistant.
            finalizeUserTurn()
            assistantTurnOpen = true
            currentAssistantText = ""
            currentAssistantStartedAtMilliseconds =
                event.audioPositionMilliseconds
            turns += 1
            status = .speaking
            observeAssistantAudio(event)
            return
        }

        if event.textToken == eosID {
            observeAssistantAudio(event)
            completeAssistantTurn(
                fallbackEndMilliseconds: event.audioPositionMilliseconds)
            assistantTurnOpen = false
            status = .listening
            return
        }

        if event.speaking {
            if !assistantTurnOpen {
                finalizeUserTurn()
                assistantTurnOpen = true
                currentAssistantStartedAtMilliseconds =
                    event.audioPositionMilliseconds
                turns += 1
            }
            if currentAssistantText.isEmpty,
               let lastUserTextFrame {
                turnGapMilliseconds = Double(
                    max(0, event.index - lastUserTextFrame)
                        * VoiceChatSession.frameMilliseconds)
            }
            currentAssistantText += event.text
            status = .speaking
        } else {
            status = assistantTurnOpen ? .speaking : .listening
        }
        observeAssistantAudio(event)
    }

    /// Record one foreground service interval for one captured 80 ms frame.
    /// Live tool/replay callbacks can emit multiple model events, but they get
    /// only one realtime denominator because only one new microphone frame was
    /// consumed.
    mutating func observeMicrophoneFrameService(
        milliseconds latency: Double,
        toolActive: Bool = false
    ) {
        lastFrameMilliseconds = latency
        frameLatencies.append(.init(
            milliseconds: latency,
            toolActive: toolActive))
        if frameLatencies.count > 120 {
            frameLatencies.removeFirst(frameLatencies.count - 120)
        }
        let all = frameLatencies.map(\.milliseconds)
        let normal = frameLatencies.filter { !$0.toolActive }
            .map(\.milliseconds)
        let tool = frameLatencies.filter(\.toolActive)
            .map(\.milliseconds)
        averageFrameMilliseconds = Self.average(all)
        averageNormalFrameMilliseconds = Self.average(normal)
        averageToolFrameMilliseconds = Self.average(tool)
        let sorted = all.sorted()
        p95FrameMilliseconds = sorted[
            min(sorted.count - 1, Int(Double(sorted.count) * 0.95))]
    }

    /// Attribute the fallback timer to the current tool phase. Native call
    /// decoding and provider waiting are distinct delays and must not share a
    /// cumulative counter in the terminal UI.
    mutating func transition(to nextStatus: Status) {
        if nextStatus == .preparingTool || nextStatus == .usingTool {
            toolActivityFrames = status == nextStatus
                ? toolActivityFrames + 1
                : 1
        } else {
            toolActivityFrames = 0
        }
        status = nextStatus
    }

    private static func average(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    mutating func finish() {
        finalizeUserTurn()
        completeAssistantTurn(
            fallbackEndMilliseconds: latestAudioPositionMilliseconds)
        assistantTurnOpen = false
        status = .stopped
    }

    mutating func noteInputResynchronization() {
        let text = currentUserText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            appendTimelineLine(.init(
                role: .user,
                text: text + " … [input dropped; please repeat]",
                timestampMilliseconds: debugTimelineEnabled
                    ? currentUserStartedAtMilliseconds : nil))
        }
        currentUserText = ""
        fullUserTranscript = ""
        committedUserTranscript = ""
        lastUserTextFrame = nil
        currentUserStartedAtMilliseconds = nil
        repeatRequestActive = true
    }

    /// Preserve the checkpoint's exact model-decoded tool decision in the
    /// optional debug timeline. This does not classify transcript text or
    /// modify the call before the coordinator validates and executes it.
    mutating func noteDecodedToolCall(
        _ rawCall: String,
        at timestampMilliseconds: Double
    ) {
        decodedToolCallPending = true
        observedNewFunctionResponse = false
        pendingToolActivityBaseline = lastToolActivity
        pendingFunctionResponseBaseline = functionResponseMetrics
        lastTimelineToolActivityKey = nil

        let summary: String
        if let call = try? VoiceChatFunctionCall.parse(rawCall) {
            pendingDecodedToolName = Self.sanitizedToolName(call.name)
            summary = "decoded \(pendingDecodedToolName ?? "unknown") "
                + Self.sanitizedTimelineText(call.argumentsJSON)
        } else {
            pendingDecodedToolName = nil
            summary = "decoded invalid tool call"
        }

        guard debugTimelineEnabled else { return }
        finalizeUserTurn()
        appendTimelineLine(.init(
            role: .tool,
            text: summary,
            timestampMilliseconds: timestampMilliseconds))
    }

    /// Add provider lifecycle transitions once. Polling updates the elapsed
    /// duration, so identity is based on tool name and state rather than the
    /// changing duration value.
    mutating func observeToolRuntimeStatus(
        _ status: VoiceChatMCPToolRuntimeStatus?,
        at timestampMilliseconds: Double
    ) {
        guard debugTimelineEnabled,
              decodedToolCallPending,
              let activity = status?.activity,
              activity != pendingToolActivityBaseline else { return }
        let key = activity.name + "\u{0}" + activity.state.rawValue
        guard key != lastTimelineToolActivityKey else { return }
        lastTimelineToolActivityKey = key

        let name = Self.sanitizedToolName(activity.name) ?? "unknown"
        let detail: String
        switch activity.state {
        case .running:
            detail = "MCP \(name) started"
        case .completed, .failed, .needsInput:
            detail = "MCP \(name) \(activity.state.rawValue) in "
                + Self.formattedDuration(activity.elapsedMilliseconds)
        }
        appendTimelineLine(.init(
            role: .tool,
            text: detail,
            timestampMilliseconds: timestampMilliseconds))
    }

    /// Record the result-cache synchronization that releases normal assistant
    /// generation. The baseline prevents the previous call's retained metric
    /// from being mislabeled as the current call.
    mutating func observeFunctionResponseMetrics(
        _ metric: VoiceChatFunctionResponseMetrics?,
        at timestampMilliseconds: Double
    ) {
        guard decodedToolCallPending, let metric else { return }
        if metric.active {
            if metric != pendingFunctionResponseBaseline {
                observedNewFunctionResponse = true
            }
            return
        }
        guard observedNewFunctionResponse
                || metric != pendingFunctionResponseBaseline else { return }

        if debugTimelineEnabled {
            let outcome = metric.completed ? "synchronized" : "incomplete"
            appendTimelineLine(.init(
                role: .tool,
                text: String(
                    format: "result %@ in %.1f s (%d tokens, %d batches)",
                    outcome,
                    metric.elapsedMilliseconds / 1_000,
                    metric.tokenSteps,
                    metric.prefillBatches),
                timestampMilliseconds: timestampMilliseconds))
        }
        decodedToolCallPending = false
        pendingDecodedToolName = nil
        pendingToolActivityBaseline = nil
        pendingFunctionResponseBaseline = nil
        observedNewFunctionResponse = false
    }

    var assistantActive: Bool { assistantTurnOpen }
    var currentUserTimestampMilliseconds: Double? {
        currentUserStartedAtMilliseconds
    }
    var currentAssistantTimestampMilliseconds: Double? {
        currentAssistantStartedAtMilliseconds
    }

    private mutating func finalizeUserTurn() {
        let text = currentUserText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            appendTimelineLine(.init(
                role: .user,
                text: text,
                timestampMilliseconds: debugTimelineEnabled
                    ? currentUserStartedAtMilliseconds : nil))
        }
        committedUserTranscript = fullUserTranscript
        currentUserText = ""
        currentUserStartedAtMilliseconds = nil
    }

    private mutating func finalizeAssistantTurn() {
        let text = currentAssistantText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.isEmpty {
            appendTimelineLine(.init(
                role: .assistant,
                text: text,
                timestampMilliseconds: debugTimelineEnabled
                    ? currentAssistantStartedAtMilliseconds : nil))
        }
        currentAssistantText = ""
        currentAssistantStartedAtMilliseconds = nil
    }

    /// Close the visible assistant turn and, in the optional debug timeline,
    /// mark the end of the generated audible PCM. This is model-timeline audio
    /// completion, not the later instant when Core Audio drains its queue.
    private mutating func completeAssistantTurn(
        fallbackEndMilliseconds: Double
    ) {
        let hadText = !currentAssistantText.trimmingCharacters(
            in: .whitespacesAndNewlines).isEmpty
        finalizeAssistantTurn()
        guard debugTimelineEnabled, hadText else {
            lastAssistantAudibleEndMilliseconds = nil
            return
        }

        let detectedEnd = lastAssistantAudibleEndMilliseconds
        appendTimelineLine(.init(
            role: .audio,
            text: detectedEnd == nil
                ? "pronunciation ended (no audible PCM detected)"
                : "pronunciation ended",
            timestampMilliseconds: detectedEnd ?? fallbackEndMilliseconds))
        lastAssistantAudibleEndMilliseconds = nil
    }

    private mutating func observeAssistantAudio(
        _ event: VoiceChatDemoFrame
    ) {
        guard assistantTurnOpen,
              let withinFrame = event.audibleAudioEndMillisecondsWithinFrame
        else { return }
        lastAssistantAudibleEndMilliseconds =
            event.audioPositionMilliseconds + withinFrame
    }

    private mutating func appendTimelineLine(
        _ line: VoiceChatConversationLine
    ) {
        lines.append(line)
        if lines.count > 80 {
            lines.removeFirst(lines.count - 80)
        }
    }

    private static func sanitizedToolName(_ name: String) -> String? {
        let safe = name.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0)
                || $0 == "_" || $0 == "-" || $0 == "."
        }
        let result = String(String.UnicodeScalarView(safe)).prefix(48)
        return result.isEmpty ? nil : String(result)
    }

    private static func sanitizedTimelineText(_ text: String) -> String {
        let safe = text.unicodeScalars.map { scalar -> Character in
            scalar.value >= 0x20 && scalar.value != 0x7f
                ? Character(String(scalar))
                : " "
        }
        return String(String(safe).prefix(512)).trimmingCharacters(
            in: .whitespacesAndNewlines)
    }

    private static func formattedDuration(_ milliseconds: Double) -> String {
        if milliseconds < 1_000 {
            return "\(Int(milliseconds.rounded())) ms"
        }
        return String(format: "%.1f s", milliseconds / 1_000)
    }

    private func transcriptSuffix(_ transcript: String, after prefix: String) -> String {
        guard transcript.hasPrefix(prefix) else { return transcript }
        return String(transcript.dropFirst(prefix.count))
    }

    private func normalizedTurnBoundarySuffix(_ suffix: String) -> String {
        guard currentUserText.isEmpty,
              !committedUserTranscript.isEmpty else { return suffix }
        return suffix.trimmingPrefixCharacters(in:
            CharacterSet.whitespacesAndNewlines
                .union(.punctuationCharacters))
    }

    private mutating func appendDelayedPunctuation(_ suffix: String) {
        let punctuation = suffix.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !punctuation.isEmpty,
              punctuation.unicodeScalars.allSatisfy({
                  CharacterSet.punctuationCharacters.contains($0)
              }),
              let index = lines.lastIndex(where: { $0.role == .user }),
              !lines[index].text.hasSuffix(punctuation)
        else { return }
        lines[index].text += punctuation
    }
}

private extension String {
    func trimmingPrefixCharacters(in set: CharacterSet) -> String {
        guard let first = unicodeScalars.firstIndex(where: {
            !set.contains($0)
        }) else { return "" }
        return String(unicodeScalars[first...])
    }
}

struct VoiceChatTerminalRenderer {
    var state: VoiceChatDemoState
    var width: Int

    init(state: VoiceChatDemoState, width: Int = 120) {
        self.state = state
        self.width = min(180, max(72, width))
    }

    func render(colorized: Bool = false) -> String {
        let inner = width - 2
        let heading = " \(state.title) "
        var top = "┌" + heading
            + String(repeating: "─", count: max(0, inner - heading.count)) + "┐"
        let status = "● \(state.status.rawValue)"
        let metrics = [
            delayMetric(),
            frameMetric(),
            "replies \(state.turns)",
        ].joined(separator: "   ")
        let mic = "mic: \(state.microphone)"
        var second = fit("  \(status)   \(mic)", right: metrics, width: inner)
        let realtime = fit(
            "  live-frame RTF",
            right: realTimeFactorMetrics(),
            width: inner)
        let bottom = "└" + String(repeating: "─", count: inner) + "┘"

        if colorized {
            top = top.replacingOccurrences(
                of: heading,
                with: "\u{001B}[1m\(heading)\u{001B}[0m")
            let statusColor: Int
            switch state.status {
            case .repeatNeeded, .preparingTool, .usingTool:
                statusColor = 214
            default:
                statusColor = 118
            }
            second = second.replacingOccurrences(
                of: status,
                with: "\u{001B}[38;5;\(statusColor)m\(status)\u{001B}[0m")
        }

        var output = [
            top,
            "│" + second + "│",
            "│" + realtime + "│",
            bottom,
            "",
        ]
        if state.repeatRequestActive {
            output.append(
                "Some microphone audio was skipped to stay live. "
                    + "Please repeat the interrupted sentence.")
            output.append("")
        }
        if state.realtimeProtectionActive,
           state.currentSpeechIterations < state.preferredSpeechIterations
        {
            output.append(
                "Voice detail is temporarily \(state.currentSpeechIterations)/"
                    + "\(state.preferredSpeechIterations) steps to keep up.")
            output.append("")
        }
        if state.status == .preparingTool || state.status == .usingTool {
            let fallbackSeconds = Double(state.toolActivityFrames)
                * Double(VoiceChatSession.frameMilliseconds) / 1_000
            let action: String
            let seconds: Double
            if state.status == .preparingTool {
                action = "Decoding the tool name and arguments"
                seconds = state.functionCallDecodeMetrics.map {
                    $0.elapsedMilliseconds / 1_000
                }
                    ?? fallbackSeconds
            } else if let metric = state.functionResponseMetrics,
                      metric.active {
                action = "Synchronizing the tool result with VoiceChat"
                seconds = metric.elapsedMilliseconds / 1_000
            } else if let name = sanitizedToolName(state.activeToolName) {
                action = "Waiting for \(name)"
                seconds = state.lastToolActivity.map {
                    $0.elapsedMilliseconds / 1_000
                }
                    ?? fallbackSeconds
            } else {
                action = "Waiting for the connected service"
                seconds = fallbackSeconds
            }
            output.append(String(format: "%@ (%.1f s).", action, seconds))
            output.append("")
        }
        let visibleLines = state.lines
            + (state.currentUserText.isEmpty ? [] : [
                VoiceChatConversationLine(
                    role: .user,
                    text: state.currentUserText,
                    timestampMilliseconds:
                        state.currentUserTimestampMilliseconds),
            ])
            + (state.currentAssistantText.isEmpty ? [] : [
                VoiceChatConversationLine(
                    role: .assistant,
                    text: state.currentAssistantText,
                    timestampMilliseconds:
                        state.currentAssistantTimestampMilliseconds),
            ])
        for line in visibleLines.suffix(24) {
            let wrapped = wrap(line: line, width: width)
            output.append(contentsOf: colorized
                ? colorize(wrapped, role: line.role)
                : wrapped)
        }
        if visibleLines.isEmpty {
            output.append("Speak naturally. Press Control-C to stop.")
        }
        if state.inputBufferedFrames >= 1
            || state.playbackUnderruns > 0
            || state.forcedTurnStarts > 0
            || state.forcedBargeIns > 0
            || state.inputResynchronizations > 0
            || state.realtimeProtectionActive
            || state.repeatRequestActive
        {
            output.append("")
            output.append(String(
                format: "queued microphone audio %.1f s   speaker gaps %d",
                state.inputBufferedFrames
                    * Double(VoiceChatSession.frameMilliseconds) / 1_000,
                state.playbackUnderruns))
            if state.forcedTurnStarts > 0 || state.forcedBargeIns > 0 {
                output.append(
                    "RNN-T forced starts \(state.forcedTurnStarts)   "
                        + "barge-ins \(state.forcedBargeIns)")
            }
            if state.inputResynchronizations > 0
                || state.droppedInputMilliseconds > 0
            {
                output.append(String(
                    format: "old-audio skips %d   microphone audio skipped %.1f s",
                    state.inputResynchronizations,
                    state.droppedInputMilliseconds / 1_000))
            }
        }
        return output.joined(separator: "\n")
    }

    private func delayMetric() -> String {
        let seconds = state.inputBufferedFrames
            * Double(VoiceChatSession.frameMilliseconds) / 1_000
        return String(format: "behind %.1f s", seconds)
    }

    private func realTimeFactorMetrics() -> String {
        "normal \(realTimeFactor(state.averageNormalFrameMilliseconds))"
            + "   tool \(realTimeFactor(state.averageToolFrameMilliseconds))"
            + "   avg \(realTimeFactor(state.averageFrameMilliseconds))"
    }

    private func realTimeFactor(_ averageMilliseconds: Double?) -> String {
        guard let averageMilliseconds,
              averageMilliseconds > 0 else { return "—" }
        let factor = averageMilliseconds
            / Double(VoiceChatSession.frameMilliseconds)
        return String(format: "%.2f×", factor)
    }

    private func frameMetric() -> String {
        guard let value = state.lastFrameMilliseconds else {
            return "last — for \(VoiceChatSession.frameMilliseconds) ms audio"
        }
        return "last \(Int(value.rounded())) ms for "
            + "\(VoiceChatSession.frameMilliseconds) ms audio"
    }

    private func sanitizedToolName(_ name: String?) -> String? {
        guard let name else { return nil }
        let safe = name.unicodeScalars.filter {
            CharacterSet.alphanumerics.contains($0)
                || $0 == "_" || $0 == "-" || $0 == "."
        }
        let result = String(String.UnicodeScalarView(safe)).prefix(48)
        return result.isEmpty ? nil : String(result)
    }

    private func fit(_ left: String, right: String, width: Int) -> String {
        let room = width - right.count
        let clipped = String(left.prefix(max(0, room - 1)))
        let spaces = max(1, width - clipped.count - right.count)
        return clipped + String(repeating: " ", count: spaces) + right
    }

    private func wrap(
        line: VoiceChatConversationLine,
        width: Int
    ) -> [String] {
        let prefix = timelineTimestamp(line.timestampMilliseconds)
            + rolePrefix(line.role)
        let continuation = String(repeating: " ", count: prefix.count)
        let available = max(16, width - prefix.count)
        let words = line.text.split(whereSeparator: \.isWhitespace)
        guard !words.isEmpty else { return [prefix] }

        var result: [String] = []
        var current = ""
        for wordSlice in words {
            let word = String(wordSlice)
            if current.isEmpty {
                current = word
            } else if current.count + word.count + 1 <= available {
                current += " " + word
            } else {
                result.append((result.isEmpty ? prefix : continuation) + current)
                current = word
            }
        }
        if !current.isEmpty {
            result.append((result.isEmpty ? prefix : continuation) + current)
        }
        return result
    }

    private func colorize(
        _ lines: [String],
        role: VoiceChatConversationLine.Role
    ) -> [String] {
        guard var first = lines.first else { return lines }
        let prefix = rolePrefix(role)
        let color: String
        switch role {
        case .user:
            color = "\u{001B}[38;5;190m"
        case .assistant:
            color = "\u{001B}[38;5;80m"
        case .tool:
            color = "\u{001B}[38;5;214m"
        case .audio:
            color = "\u{001B}[38;5;141m"
        }
        first = first.replacingOccurrences(
            of: prefix,
            with: "\(color)\(prefix)\u{001B}[0m",
            options: [],
            range: first.range(of: prefix))
        return [first] + lines.dropFirst()
    }

    private func rolePrefix(_ role: VoiceChatConversationLine.Role) -> String {
        switch role {
        case .user: "you     "
        case .assistant: "soniqo  "
        case .tool: "tool    "
        case .audio: "audio   "
        }
    }

    private func timelineTimestamp(_ milliseconds: Double?) -> String {
        guard state.debugTimelineEnabled, let milliseconds else { return "" }
        let totalMilliseconds = max(0, Int(milliseconds.rounded()))
        let hours = totalMilliseconds / 3_600_000
        let minutes = (totalMilliseconds / 60_000) % 60
        let seconds = (totalMilliseconds / 1_000) % 60
        let fraction = totalMilliseconds % 1_000
        if hours > 0 {
            return String(
                format: "[%02d:%02d:%02d.%03d] ",
                hours, minutes, seconds, fraction)
        }
        return String(
            format: "[%02d:%02d.%03d] ",
            minutes, seconds, fraction)
    }
}
