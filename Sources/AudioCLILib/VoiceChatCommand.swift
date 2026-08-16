import ArgumentParser
import AudioCommon
import Foundation
import VoiceChat

#if canImport(AVFoundation)
import AVFoundation
#endif

#if canImport(Darwin)
import Darwin
#endif

public struct VoiceChatCommand: ParsableCommand {
    public static let defaultINT5ModelID =
        "aufklarer/VoiceChat-11B-Perception-MLX-int5"

    public static let configuration = CommandConfiguration(
        commandName: "voice-chat",
        abstract: "Talk with Soniqo using the full-duplex Nemotron VoiceChat 11B model."
    )

    @Option(
        name: .shortAndLong,
        help: "Hugging Face model ID or local complete VoiceChat bundle.")
    public var model = Self.defaultINT5ModelID

    @Option(name: .long, help: "Hugging Face revision to download.")
    public var revision = "main"

    @Option(name: .long, help: "Override the VoiceChat system prompt.")
    public var systemPrompt: String?

    @Option(
        name: .long,
        help: "JSON configuration for provider-neutral MCP tool servers.")
    public var mcpConfig: String?

    @Option(
        name: .long,
        help: "Enable one named server from --mcp-config; repeat for more than one.")
    public var mcpServer: [String] = []

    @Option(
        name: .long,
        help: "Policy for MCP tools marked as writes: allow (default), confirm, or deny.")
    public var mcpWritePolicy: VoiceChatMCPWritePolicy = .allow

    @Option(
        name: .long,
        help: "MCP tool-call timeout in seconds; startup allows at least 60 seconds.")
    public var mcpTimeoutSeconds: Double = 15

    @Flag(name: .long, help: "Ask the model to greet before the user speaks.")
    public var greet = false

    @Flag(name: .long, help: "Disable Apple acoustic echo cancellation.")
    public var noAEC = false

    @Flag(name: .long, help: "Hide the bundle's streaming RNN-T user captions.")
    public var noTranscript = false

    @Flag(
        name: .long,
        help: "Disable NVIDIA RNN-T turn-taking safety fallbacks.")
    public var noRNNTTurnTaking = false

    @Option(
        name: .long,
        help: "Model-audio frames buffered before speaker playback (80 ms each).")
    public var prebufferFrames = 3

    @Option(
        name: .long,
        help: "Maximum queued 80 ms microphone frames before stale audio is dropped.")
    public var maxBufferedFrames = 8

    @Option(name: .long, help: "Stop live capture after this many seconds.")
    public var maxSeconds: Double?

    @Option(
        name: .long,
        help: "Use an audio file instead of the microphone, then exit.")
    public var input: String?

    @Option(
        name: [.short, .long],
        help: "Write exact full-context 22.05 kHz response audio to WAV.")
    public var output: String?

    @Option(name: .long, help: "Silent tail after finite-file input, in seconds.")
    public var tailSeconds: Double = 6

    @Flag(
        name: .long,
        help: "Force the model turn at file EOF (controlled regression only).")
    public var forceTurnAtEnd = false

    @Flag(name: .long, help: "Disable the updating terminal display.")
    public var plain = false

    @Flag(
        name: .long,
        help: "Show phrase/pronunciation timestamps and tool lifecycle events.")
    public var debugTimeline = false

    @Option(name: .long, help: "Terminal display width.")
    public var terminalWidth = 120

    @Option(name: .long, help: "Text sampling temperature (zero is greedy).")
    public var temperature: Float = 0

    @Option(name: .long, help: "Text nucleus-sampling probability.")
    public var textTopP: Float = 1

    @Option(name: .long, help: "EAR-TTS classifier-free guidance strength.")
    public var guidance: Float = 0.2

    @Option(name: .long, help: "EAR-TTS nucleus-sampling probability.")
    public var speechTopP: Float = 0.95

    @Option(name: .long, help: "EAR-TTS sampling noise.")
    public var speechNoise: Float = 0.001

    @Option(name: .long, help: "EAR-TTS MaskGIT iterations per frame.")
    public var speechIterations = 8

    @Option(
        name: .long,
        help: "Voice refinement steps used temporarily to protect realtime speed.")
    public var realtimeSpeechIterations = 2

    @Option(
        name: .long,
        help: "Recent EAR-TTS history retained in live mode, in seconds; zero keeps all history.")
    public var liveSpeechContextSeconds = 20.0

    public init() {}

    public mutating func validate() throws {
        guard !model.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("--model cannot be empty")
        }
        guard !revision.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ValidationError("--revision cannot be empty")
        }
        guard (1 ... 32).contains(prebufferFrames) else {
            throw ValidationError("--prebuffer-frames must be between 1 and 32")
        }
        guard (4 ... 300).contains(maxBufferedFrames) else {
            throw ValidationError("--max-buffered-frames must be between 4 and 300")
        }
        guard terminalWidth >= 72, terminalWidth <= 180 else {
            throw ValidationError("--terminal-width must be between 72 and 180")
        }
        if let maxSeconds {
            guard maxSeconds.isFinite, maxSeconds > 0 else {
                throw ValidationError("--max-seconds must be finite and positive")
            }
            let frameCount = maxSeconds * 1_000
                / Double(VoiceChatSession.frameMilliseconds)
            guard frameCount <= Double(Int.max) else {
                throw ValidationError("--max-seconds is too large")
            }
        }
        guard tailSeconds.isFinite,
              tailSeconds >= 0,
              tailSeconds <= VoiceChatSession.maximumSilenceSeconds else {
            throw ValidationError(
                "--tail-seconds must be between 0 and "
                    + "\(Int(VoiceChatSession.maximumSilenceSeconds))")
        }
        if forceTurnAtEnd, input == nil {
            throw ValidationError("--force-turn-at-end requires --input")
        }
        if greet, systemPrompt != nil {
            throw ValidationError("use either --greet or --system-prompt, not both")
        }
        if !mcpServer.isEmpty, mcpConfig == nil {
            throw ValidationError("--mcp-server requires --mcp-config")
        }
        guard mcpTimeoutSeconds.isFinite,
              mcpTimeoutSeconds > 0,
              mcpTimeoutSeconds <= 120 else {
            throw ValidationError(
                "--mcp-timeout-seconds must be between 0 and 120")
        }
        guard temperature.isFinite, temperature >= 0 else {
            throw ValidationError("--temperature must be finite and non-negative")
        }
        guard textTopP.isFinite, textTopP > 0, textTopP <= 1 else {
            throw ValidationError("--text-top-p must be finite and in (0, 1]")
        }
        guard guidance.isFinite, guidance >= 0 else {
            throw ValidationError("--guidance must be finite and non-negative")
        }
        guard speechTopP.isFinite, speechTopP > 0, speechTopP <= 1 else {
            throw ValidationError("--speech-top-p must be finite and in (0, 1]")
        }
        guard speechNoise.isFinite, speechNoise >= 0 else {
            throw ValidationError("--speech-noise must be finite and non-negative")
        }
        guard speechIterations > 0, speechIterations <= 64 else {
            throw ValidationError("--speech-iterations must be between 1 and 64")
        }
        guard realtimeSpeechIterations > 0,
              realtimeSpeechIterations <= speechIterations else {
            throw ValidationError(
                "--realtime-speech-iterations must be between 1 and --speech-iterations")
        }
        guard liveSpeechContextSeconds.isFinite,
              liveSpeechContextSeconds >= 0,
              liveSpeechContextSeconds <= 600 else {
            throw ValidationError(
                "--live-speech-context-seconds must be between 0 and 600")
        }
    }

    public func run() throws {
        try runAsync {
            #if canImport(Darwin)
            setvbuf(stdout, nil, _IONBF, 0)
            #endif
            let mcpRuntime: VoiceChatMCPRuntime?
            let toolCoordinator: VoiceChatMCPToolCoordinator?
            let prompt: String
            if let mcpConfig {
                print("Starting MCP tools from: \(mcpConfig)")
                let referenceDate = Date()
                let calendar = Calendar.current
                let runtime = try await VoiceChatMCPRuntime.start(
                    configurationURL: URL(fileURLWithPath: mcpConfig),
                    selectedServerNames: mcpServer,
                    timeoutSeconds: mcpTimeoutSeconds,
                    referenceDate: referenceDate,
                    calendar: calendar)
                do {
                    let toolsJSON = try await runtime.availableToolsJSON()
                    prompt = try VoiceChatSession.toolCallingSystemPrompt(
                        basePrompt: Self.toolCallingBasePrompt(
                            systemPrompt ?? VoiceChatSession.personaSystemPrompt,
                            referenceDate: referenceDate,
                            timeZone: calendar.timeZone),
                        availableToolsJSON: toolsJSON,
                        greet: greet,
                        requiresWriteConfirmation: mcpWritePolicy == .confirm)
                    if Self.mcpDebugEnabled {
                        print("MCP system prompt:\n\(prompt)")
                    }
                    mcpRuntime = runtime
                    toolCoordinator = VoiceChatMCPToolCoordinator(
                        executor: runtime,
                        writePolicy: mcpWritePolicy)
                    let names = await runtime.availableTools().map(\.name)
                    print("Enabled MCP tools: \(names.joined(separator: ", "))")
                } catch {
                    await runtime.shutdown()
                    throw error
                }
            } else {
                mcpRuntime = nil
                toolCoordinator = nil
                prompt = systemPrompt
                    ?? (greet
                        ? VoiceChatSession.greetingSystemPrompt
                        : VoiceChatSession.defaultSystemPrompt)
            }

            do {
                let voiceChat = try await loadModel()
                let session = try await voiceChat.startSession(
                    systemPrompt: prompt,
                    sampling: .init(temperature: temperature, topP: textTopP),
                    speech: .init(
                        guidance: guidance,
                        topP: speechTopP,
                        noise: speechNoise,
                        iterations: speechIterations,
                        recentContextFrames: liveSpeechContextFrames,
                        realtimeIdleOptimization: input == nil),
                    streamUserTranscript: !noTranscript,
                    turnTaking: turnTakingParameters,
                    functionCallingEnabled: mcpRuntime != nil)

                if let input {
                    try await runFile(
                        input: URL(fileURLWithPath: input),
                        model: voiceChat,
                        session: session,
                        toolCoordinator: toolCoordinator)
                } else {
                    try await runMicrophone(
                        model: voiceChat,
                        session: session,
                        toolCoordinator: toolCoordinator)
                }
            } catch {
                await mcpRuntime?.shutdown()
                throw error
            }
            await mcpRuntime?.shutdown()
        }
    }

    private var liveSpeechContextFrames: Int? {
        guard input == nil, liveSpeechContextSeconds > 0 else { return nil }
        return max(1, Int((
            liveSpeechContextSeconds * 1_000
                / Double(VoiceChatSession.frameMilliseconds)
        ).rounded()))
    }

    var turnTakingParameters: VoiceChatTurnTakingParameters {
        guard !noRNNTTurnTaking else { return .modelNative }
        var parameters = mcpConfig == nil
            ? VoiceChatTurnTakingParameters.nvidiaRealtime
            : VoiceChatTurnTakingParameters.functionCallingRealtime
        parameters.allowInitialAgentTurn = greet
        return parameters
    }

    private func loadModel() async throws -> VoiceChatModel {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(
            atPath: model, isDirectory: &isDirectory),
           isDirectory.boolValue {
            print("Loading VoiceChat bundle: \(model)")
            return try await VoiceChatModel.load(
                from: URL(fileURLWithPath: model),
                progressHandler: Self.modelLoadProgress)
        }

        print("Loading VoiceChat model: \(model) @ \(revision)")
        let progress = VoiceChatDownloadProgress()
        return try await VoiceChatModel.loadFromHub(
            model,
            revision: revision,
            progressHandler: { update in progress.report(update) },
            loadProgressHandler: Self.modelLoadProgress)
    }

    private static let modelLoadProgress: VoiceChatLoadProgressHandler = {
        progress, stage in
        print(String(
            format: "  [%3.0f%%] %@",
            max(0, min(1, progress)) * 100,
            stage))
    }

    static func toolCallingBasePrompt(
        _ basePrompt: String,
        referenceDate: Date,
        timeZone: TimeZone
    ) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = "MMMM d, yyyy 'at' h:mm a ZZZZZ"
        let localDate = formatter.string(from: referenceDate)
        return basePrompt
            + " The current local date and time is \(localDate)."
            + " Resolve relative dates from this value, but never invent a"
            + " missing date, time, reminder list, or other tool argument."
            + " Describe available help positively and briefly. Do not"
            + " volunteer a list of unavailable features. For reminders, ask"
            + " only for the single detail needed next; never demand optional"
            + " body text or priority."
    }

    private func runFile(
        input: URL,
        model: VoiceChatModel,
        session: VoiceChatSession,
        toolCoordinator: VoiceChatMCPToolCoordinator?
    ) async throws {
        let samples = try AudioFileLoader.load(
            url: input,
            targetSampleRate: VoiceChatSession.inputSampleRate)
        if forceTurnAtEnd {
            await session.forceTurn(
                atFrame: samples.count / VoiceChatSession.inputSamplesPerFrame)
        }

        var state = initialState(modelName: modelName(), microphone: "file input")
        state.status = .listening
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        for start in stride(from: 0, to: samples.count, by: frameSize) {
            let events = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)]))
            ingest(events, model: model, state: &state)
            for event in events {
                await toolCoordinator?.observeUserActivity(
                    rnntIsBlank: event.rnntIsBlank)
                if event.textToken == model.tokenizer.bosID {
                    await toolCoordinator?.observeModelTextTurnStart()
                } else if event.textToken == model.tokenizer.eosID {
                    await toolCoordinator?.observeModelTextTurnEnd()
                }
                await handleToolEvent(
                    event,
                    session: session,
                    coordinator: toolCoordinator)
            }
        }

        let tailSamples = Int(
            (tailSeconds * Double(VoiceChatSession.inputSampleRate)).rounded())
        let silence = [Float](repeating: 0, count: tailSamples)
        for start in stride(from: 0, to: silence.count, by: frameSize) {
            let events = try await session.pushAudio(
                Array(silence[start ..< min(silence.count, start + frameSize)]))
            ingest(events, model: model, state: &state)
            for event in events {
                await toolCoordinator?.observeUserActivity(
                    rnntIsBlank: event.rnntIsBlank)
                if event.textToken == model.tokenizer.bosID {
                    await toolCoordinator?.observeModelTextTurnStart()
                } else if event.textToken == model.tokenizer.eosID {
                    await toolCoordinator?.observeModelTextTurnEnd()
                }
                await handleToolEvent(
                    event,
                    session: session,
                    coordinator: toolCoordinator)
            }
        }

        if toolCoordinator != nil {
            let maximumDrainFrames = Int(
                30_000 / VoiceChatSession.frameMilliseconds)
            let silentFrame = [Float](
                repeating: 0,
                count: VoiceChatSession.inputSamplesPerFrame)
            var drainFrames = 0
            while await session.hasPendingFunctionOutput(),
                  drainFrames < maximumDrainFrames
            {
                let events = try await session.pushAudio(silentFrame)
                ingest(events, model: model, state: &state)
                for event in events {
                    await toolCoordinator?.observeUserActivity(
                        rnntIsBlank: event.rnntIsBlank)
                    if event.textToken == model.tokenizer.bosID {
                        await toolCoordinator?.observeModelTextTurnStart()
                    } else if event.textToken == model.tokenizer.eosID {
                        await toolCoordinator?.observeModelTextTurnEnd()
                    }
                    await handleToolEvent(
                        event,
                        session: session,
                        coordinator: toolCoordinator)
                }
                drainFrames += 1
            }
            if await session.hasPendingFunctionOutput() {
                print("Warning: MCP output did not finish within the 30-second file-mode drain limit.")
            }
        }

        state.activeToolName = await toolCoordinator?.executingToolName()
        state.lastToolActivity = await toolCoordinator?.toolActivity()
        state.functionCallDecodeMetrics = await session
            .functionCallDecodeMetrics()
        state.functionResponseMetrics = await session.functionResponseMetrics()
        state.finish()
        print(VoiceChatTerminalRenderer(
            state: state, width: terminalWidth).render())
        try await writeOutputIfRequested(session: session)
        await printMCPDebug(session: session, model: model)
        await printSummary(session: session)
    }

    private func runMicrophone(
        model: VoiceChatModel,
        session: VoiceChatSession,
        toolCoordinator: VoiceChatMCPToolCoordinator?
    ) async throws {
        #if os(macOS) && canImport(AVFoundation)
        guard await microphonePermissionGranted() else {
            throw ValidationError(
                "microphone permission was denied; enable it in System Settings > Privacy & Security > Microphone")
        }

        let inputBuffer = VoiceChatInputBuffer(
            frameSize: VoiceChatSession.inputSamplesPerFrame,
            maximumFrames: maxBufferedFrames)
        let audio = FullDuplexAudioIO(configuration: .init(
            inputSampleRate: VoiceChatSession.inputSampleRate,
            outputSampleRate: VoiceChatSession.outputSampleRate,
            playbackPrebufferFrames: prebufferFrames,
            enableAEC: !noAEC))
        var state = initialState(
            modelName: modelName(), microphone: audio.microphoneName)
        state.preferredSpeechIterations = speechIterations
        state.currentSpeechIterations = speechIterations
        var governor = VoiceChatRealtimeGovernor(
            preferredIterations: speechIterations,
            fallbackIterations: realtimeSpeechIterations)
        var toolTask: Task<Void, Never>?
        let console = VoiceChatConsole(
            interactive: !plain && isatty(STDOUT_FILENO) != 0,
            width: terminalWidth)
        let stopSignals = VoiceChatStopSignals { inputBuffer.close() }

        console.start(state)
        do {
            try audio.start { samples in
                _ = inputBuffer.append(samples)
            }
            state.status = .listening
            console.update(state, force: true)

            let maximumFrames = maxSeconds.map {
                Int(($0 * 1_000 / Double(VoiceChatSession.frameMilliseconds)).rounded(.up))
            }
            var capturedFrames = 0
            var discontinuityPolicy = VoiceChatInputDiscontinuityPolicy(
                decoderResetSamples:
                    VoiceChatSession.inputSamplesPerFrame * 2)
            var resynchronizationNoticeFrames = 0
            var functionRuntimeStatus = await session.functionRuntimeStatus()
            var toolRuntimeStatus = await toolCoordinator?.runtimeStatus()
            while true {
                guard let frame = inputBuffer.nextFrame() else { break }
                if let maximumFrames, capturedFrames >= maximumFrames { break }
                capturedFrames += 1
                let toolWorkWasActive =
                    functionRuntimeStatus.generatingCall
                    || functionRuntimeStatus.waitingForResponse
                    || functionRuntimeStatus.responseSync?.active == true
                    || toolRuntimeStatus?.executing == true
                let frameServiceStarted = DispatchTime.now().uptimeNanoseconds

                let events = try await session.pushAudio(frame)
                for event in events {
                    if event.playbackRequired {
                        audio.schedulePlayback(event.audio)
                    }
                    state.ingest(
                        event,
                        bosID: model.tokenizer.bosID,
                        eosID: model.tokenizer.eosID,
                        recordFrameLatency: false)
                    if let functionCall = event.functionCall {
                        state.noteDecodedToolCall(
                            functionCall,
                            at: event.audioPositionMilliseconds)
                    }
                    await toolCoordinator?.observeUserActivity(
                        rnntIsBlank: event.rnntIsBlank)
                    if event.textToken == model.tokenizer.bosID {
                        await toolCoordinator?.observeModelTextTurnStart()
                    } else if event.textToken == model.tokenizer.eosID {
                        await toolCoordinator?.observeModelTextTurnEnd()
                    }
                    if let task = scheduleToolEvent(
                        event,
                        session: session,
                        coordinator: toolCoordinator)
                    {
                        // A VoiceChat function cycle owns one shared channel,
                        // so only one external operation can be outstanding.
                        // Retaining just the current task avoids accumulating
                        // completed task handles over a long live session.
                        toolTask = task
                    }
                }
                let frameServiceMilliseconds = Double(
                    DispatchTime.now().uptimeNanoseconds
                        - frameServiceStarted)
                    / 1_000_000
                let audioStatistics = audio.statistics()
                let inputStatistics = inputBuffer.statistics()
                let discontinuity = discontinuityPolicy.observe(
                    inputStatistics)
                if discontinuity.requiresDecoderReset {
                    resynchronizationNoticeFrames = 25
                    await session.resynchronizeLiveInput()
                    state.noteInputResynchronization()
                } else if resynchronizationNoticeFrames > 0 {
                    resynchronizationNoticeFrames -= 1
                }
                if let iterations = governor.observe(
                    bufferedFrames: inputStatistics.bufferedFrames,
                    frameComputeMilliseconds: frameServiceMilliseconds,
                    didResynchronize: discontinuity.overloadStarted)
                {
                    try await session.setSpeechIterations(iterations)
                }
                state.inputBufferedFrames = inputStatistics.bufferedFrames
                state.playbackUnderruns = audioStatistics.underruns
                state.inputResynchronizations = inputStatistics.resynchronizations
                state.droppedInputMilliseconds = Double(
                    inputStatistics.droppedSamples) * 1_000
                    / Double(VoiceChatSession.inputSampleRate)
                state.currentSpeechIterations = governor.currentIterations
                state.realtimeProtectionActive = governor.isProtectingRealtime
                state.repeatRequestActive = resynchronizationNoticeFrames > 0
                // Terminal diagnostics may lag by at most one 80 ms frame.
                // One coherent snapshot prevents seven actor hops from
                // repeatedly interleaving with background function decoding.
                if capturedFrames.isMultiple(of: 2) {
                    functionRuntimeStatus = await session
                        .functionRuntimeStatus()
                    toolRuntimeStatus = await toolCoordinator?.runtimeStatus()
                }
                let generatingFunctionCall =
                    functionRuntimeStatus.generatingCall
                state.functionCallDecodeMetrics =
                    functionRuntimeStatus.callDecode
                state.functionResponseMetrics =
                    functionRuntimeStatus.responseSync
                let waitingForFunctionResponse =
                    functionRuntimeStatus.waitingForResponse
                let syncingFunctionResponse = state.functionResponseMetrics?
                    .active == true
                let executingTool = toolRuntimeStatus?.executing == true
                let toolWorkIsActive = generatingFunctionCall
                    || waitingForFunctionResponse
                    || syncingFunctionResponse
                    || executingTool
                state.observeMicrophoneFrameService(
                    milliseconds: frameServiceMilliseconds,
                    toolActive: toolWorkWasActive || toolWorkIsActive)
                state.observeToolRuntimeStatus(
                    toolRuntimeStatus,
                    at: Double(capturedFrames)
                        * Double(VoiceChatSession.frameMilliseconds))
                state.observeFunctionResponseMetrics(
                    functionRuntimeStatus.responseSync,
                    at: Double(capturedFrames)
                        * Double(VoiceChatSession.frameMilliseconds))
                state.activeToolName = toolRuntimeStatus?.name
                    ?? ((waitingForFunctionResponse || syncingFunctionResponse)
                        ? state.pendingDecodedToolName
                        : nil)
                state.lastToolActivity = toolRuntimeStatus?.activity
                let nextStatus: VoiceChatDemoState.Status
                if state.repeatRequestActive {
                    nextStatus = .repeatNeeded
                } else if generatingFunctionCall {
                    nextStatus = .preparingTool
                } else if waitingForFunctionResponse || executingTool
                    || syncingFunctionResponse {
                    nextStatus = .usingTool
                } else if state.assistantActive {
                    nextStatus = .speaking
                } else {
                    nextStatus = .listening
                }
                state.transition(to: nextStatus)
                console.update(
                    state,
                    force: discontinuity.overloadStarted
                        || discontinuity.requiresDecoderReset)
            }
        } catch {
            state.status = .error
            console.update(state, force: true)
            audio.stop()
            if let toolTask {
                toolTask.cancel()
                await toolTask.value
            }
            stopSignals.cancel()
            inputBuffer.close()
            console.finish(state)
            throw error
        }

        audio.stop()
        if let toolTask {
            toolTask.cancel()
            await toolTask.value
        }
        stopSignals.cancel()
        inputBuffer.close()
        state.finish()
        console.finish(state)
        try await writeOutputIfRequested(session: session)
        if plain { await printSummary(session: session) }
        #else
        throw ValidationError("live voice-chat requires macOS with AVFoundation")
        #endif
    }

    private func handleToolEvent(
        _ event: VoiceChatFrameEvent,
        session: VoiceChatSession,
        coordinator: VoiceChatMCPToolCoordinator?
    ) async {
        guard let coordinator else { return }
        if let functionCall = event.functionCall {
            let action = await coordinator.handleFunctionCall(functionCall)
            await injectToolAction(action, into: session)
        }
    }

    static let toolInjectionFallback = VoiceChatMCPAction(
        responseJSON: #"{"ok":false,"error":"tool response could not be processed"}"#,
        requireAssistantReplyBeforeNextFunctionCall: true)

    private func injectToolAction(
        _ action: VoiceChatMCPAction,
        into session: VoiceChatSession
    ) async {
        do {
            try await session.injectFunctionResponse(
                action.responseJSON,
                requireAssistantReplyBeforeNextFunctionCall:
                    action.requireAssistantReplyBeforeNextFunctionCall)
        } catch {
            let fallback = Self.toolInjectionFallback
            do {
                try await session.injectFunctionResponse(
                    fallback.responseJSON,
                    requireAssistantReplyBeforeNextFunctionCall:
                        fallback.requireAssistantReplyBeforeNextFunctionCall)
            } catch {
                await session.recoverFromFunctionResponseFailure()
                FileHandle.standardError.write(Data(
                    "VoiceChat could not recover from an MCP response injection error: \(error)\n"
                        .utf8))
            }
        }
    }

    private func scheduleToolEvent(
        _ event: VoiceChatFrameEvent,
        session: VoiceChatSession,
        coordinator: VoiceChatMCPToolCoordinator?
    ) -> Task<Void, Never>? {
        guard coordinator != nil,
              event.functionCall != nil else {
            return nil
        }
        return Task {
            await handleToolEvent(
                event,
                session: session,
                coordinator: coordinator)
        }
    }

    private func ingest(
        _ events: [VoiceChatFrameEvent],
        model: VoiceChatModel,
        state: inout VoiceChatDemoState
    ) {
        for event in events {
            state.ingest(
                event,
                bosID: model.tokenizer.bosID,
                eosID: model.tokenizer.eosID)
            if let functionCall = event.functionCall {
                state.noteDecodedToolCall(
                    functionCall,
                    at: event.audioPositionMilliseconds)
            }
        }
    }

    private func initialState(
        modelName: String,
        microphone: String
    ) -> VoiceChatDemoState {
        var state = VoiceChatDemoState()
        state.title = modelName
        state.microphone = microphone
        state.debugTimelineEnabled = debugTimeline
        return state
    }

    private func modelName() -> String {
        let lowered = model.lowercased()
        let precision = lowered.contains("int5")
            ? "INT5"
            : (lowered.contains("int8") ? "INT8" : "MLX")
        return "Soniqo VoiceChat (Nemotron 11B \(precision))"
    }

    private func writeOutputIfRequested(
        session: VoiceChatSession
    ) async throws {
        guard let output else { return }
        let waveform = await session.renderedAudio()
        let url = URL(fileURLWithPath: output)
        try WAVWriter.write(
            samples: waveform,
            sampleRate: VoiceChatSession.outputSampleRate,
            to: url)
        print("Wrote \(url.path)")
    }

    private func printSummary(session: VoiceChatSession) async {
        let transcript = await session.userTranscript()
        let reply = await session.reply()
        let summary = await session.summary()
        print("you  \(transcript)")
        print("soniqo  \(reply)")
        print(String(
            format: "frames %d, total p50 %.1f ms, p95 %.1f ms, real time %@",
            summary.frames,
            summary.totalP50Milliseconds,
            summary.totalP95Milliseconds,
            summary.realTime ? "yes" : "NO"))
    }

    private static var mcpDebugEnabled: Bool {
        ProcessInfo.processInfo.environment["SPEECH_VOICECHAT_MCP_DEBUG"] == "1"
    }

    private func printMCPDebug(
        session: VoiceChatSession,
        model: VoiceChatModel
    ) async {
        guard Self.mcpDebugEnabled else { return }
        let events = await session.events()
        let active = events.filter {
            $0.functionToken != model.tokenizer.padID
        }
        guard !active.isEmpty else {
            print("MCP function channel: all PAD")
            return
        }
        let preview = active.prefix(512).map {
            "\($0.index):\($0.functionToken):"
                + model.tokenizer.decode(
                    [$0.functionToken], skipSpecialTokens: false)
        }.joined(separator: " | ")
        print("MCP function channel: \(preview)")
        if active.count > 512 {
            print("MCP function channel: \(active.count - 512) more non-PAD tokens")
        }
    }

    #if os(macOS) && canImport(AVFoundation)
    private func microphonePermissionGranted() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { continuation in
                AVCaptureDevice.requestAccess(for: .audio) {
                    continuation.resume(returning: $0)
                }
            }
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }
    #endif
}

private final class VoiceChatDownloadProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var lastPercent = -1

    func report(_ progress: Progress) {
        let percent = max(0, min(100, Int(progress.fractionCompleted * 100)))
        lock.lock()
        guard percent != lastPercent else {
            lock.unlock()
            return
        }
        lastPercent = percent
        lock.unlock()
        print("  [\(percent)%] Downloading")
    }
}

final class VoiceChatConsole {
    private let interactive: Bool
    private let width: Int
    private let writeOutput: (String) -> Void
    private var lastUpdate: UInt64 = 0
    private var printedLines = 0

    init(
        interactive: Bool,
        width: Int,
        writeOutput: ((String) -> Void)? = nil
    ) {
        self.interactive = interactive
        self.width = width
        self.writeOutput = writeOutput ?? { text in
            FileHandle.standardOutput.write(Data(text.utf8))
        }
    }

    func start(_ state: VoiceChatDemoState) {
        if interactive {
            write("\u{001B}[?1049h\u{001B}[?25l")
            update(state, force: true)
        } else {
            print("\(state.title) — loading")
        }
    }

    func update(_ state: VoiceChatDemoState, force: Bool = false) {
        if interactive {
            let now = DispatchTime.now().uptimeNanoseconds
            guard force || now - lastUpdate >= 200_000_000 else { return }
            lastUpdate = now
            let screen = VoiceChatTerminalRenderer(
                state: state, width: width).render(colorized: true)
            write("\u{001B}[H\u{001B}[2J" + screen + "\n")
            return
        }

        guard state.lines.count > printedLines else { return }
        for line in state.lines[printedLines...] {
            print("\(line.role.rawValue)  \(line.text)")
        }
        printedLines = state.lines.count
    }

    func finish(_ state: VoiceChatDemoState) {
        if interactive {
            let screen = VoiceChatTerminalRenderer(
                state: state, width: width).render(colorized: true)
            write("\u{001B}[?25h\u{001B}[?1049l" + screen + "\n")
        } else {
            update(state, force: true)
        }
    }

    private func write(_ text: String) {
        writeOutput(text)
    }
}

#if os(macOS)
private final class VoiceChatStopSignals: @unchecked Sendable {
    private var sources: [DispatchSourceSignal] = []

    init(stop: @escaping @Sendable () -> Void) {
        for code in [SIGINT, SIGTERM] {
            signal(code, SIG_IGN)
            let source = DispatchSource.makeSignalSource(
                signal: code,
                queue: DispatchQueue.global(qos: .userInitiated))
            source.setEventHandler(handler: stop)
            source.resume()
            sources.append(source)
        }
    }

    func cancel() {
        for source in sources { source.cancel() }
        sources.removeAll()
        signal(SIGINT, SIG_DFL)
        signal(SIGTERM, SIG_DFL)
    }

    deinit { cancel() }
}
#endif
