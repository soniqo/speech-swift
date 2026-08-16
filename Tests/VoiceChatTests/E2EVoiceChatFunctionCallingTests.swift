import AVFoundation
import Foundation
import MLX
import XCTest

@testable import VoiceChat

/// Real-checkpoint coverage for the trained function head and model-native
/// result continuation. The isolated E2E runner gives this class
/// its own process because the complete VoiceChat bundle is multi-gigabyte.
final class E2EVoiceChatFunctionCallingTests: XCTestCase {
    private struct ToolPhraseProfileCase: Sendable {
        let id: String
        let phrase: String
        let expectedTool: String
        let availableToolsJSON: String
        let responseJSON: String
    }

    private struct SimulatedToolTiming: Sendable {
        let startedAtNanoseconds: UInt64
        let completedAtNanoseconds: UInt64
        let responseAcceptedAtNanoseconds: UInt64
        let elapsedMilliseconds: Double
    }

    private struct TimedMicrophonePush: Sendable {
        let startedAtNanoseconds: UInt64
        let completedAtNanoseconds: UInt64
        let deadlineNanoseconds: UInt64

        var elapsedMilliseconds: Double {
            Double(completedAtNanoseconds - startedAtNanoseconds) / 1_000_000
        }

        var budgetOverrunMilliseconds: Double {
            guard completedAtNanoseconds > deadlineNanoseconds else { return 0 }
            return Double(completedAtNanoseconds - deadlineNanoseconds)
                / 1_000_000
        }

        func overlaps(_ start: UInt64, _ end: UInt64) -> Bool {
            startedAtNanoseconds < end && completedAtNanoseconds > start
        }
    }

    func testBatchedToolResponsePreservesSequentialCacheState() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let sequential = try await model.startSession(
            systemPrompt: VoiceChatSession.baseSystemPrompt,
            speech: .init(realtimeIdleOptimization: true),
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let batched = try await model.startSession(
            systemPrompt: VoiceChatSession.baseSystemPrompt,
            speech: .init(realtimeIdleOptimization: true),
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        await sequential.suppressFunctionFastPathForTesting()
        await batched.suppressFunctionFastPathForTesting()

        let response = #"{"ok":true,"executed":true,"tool":"create_reminder"}"#
        try await sequential.injectFunctionResponse(response)
        try await batched.injectFunctionResponse(response)
        let optionalSilence = await batched
            .functionSilenceEmbeddingForTesting()
        let silence = try XCTUnwrap(optionalSilence)

        let sequentialSteps = try await sequential
            .advanceFunctionResponseSequentiallyForTesting(
                audioEmbedding: silence)
        let batchedSteps = try await batched
            .prefillFunctionResponseForTesting(audioEmbedding: silence)
        XCTAssertEqual(batchedSteps, sequentialSteps)
        let optionalSequentialResponseMetrics = await sequential
            .functionResponseMetrics()
        let optionalBatchedResponseMetrics = await batched
            .functionResponseMetrics()
        let sequentialResponseMetrics = try XCTUnwrap(
            optionalSequentialResponseMetrics)
        let batchedResponseMetrics = try XCTUnwrap(
            optionalBatchedResponseMetrics)
        XCTAssertTrue(sequentialResponseMetrics.completed)
        XCTAssertTrue(batchedResponseMetrics.completed)
        XCTAssertEqual(sequentialResponseMetrics.tokenSteps, sequentialSteps)
        XCTAssertEqual(batchedResponseMetrics.tokenSteps, batchedSteps)
        XCTAssertEqual(sequentialResponseMetrics.prefillBatches, sequentialSteps)
        XCTAssertLessThan(
            batchedResponseMetrics.prefillBatches,
            batchedResponseMetrics.tokenSteps)
        let batchedCodeCount = await batched.generatedCodeCountForTesting()
        let sequentialCodeCount = await sequential
            .generatedCodeCountForTesting()
        XCTAssertEqual(batchedCodeCount, sequentialCodeCount)

        let sequentialCache = await sequential.languageCacheStateForTesting()
        let batchedCache = await batched.languageCacheStateForTesting()
        XCTAssertEqual(sequentialCache.map(\.count), batchedCache.map(\.count))

        var minimumCosine: Float = 1
        var maximumAbsoluteDifference: Float = 0
        for (sequentialLayer, batchedLayer) in zip(
            sequentialCache, batchedCache
        ) {
            for (lhs, rhs) in zip(sequentialLayer, batchedLayer) {
                XCTAssertEqual(lhs.shape, rhs.shape)
                guard lhs.shape == rhs.shape else { continue }
                let a = lhs.asType(.float32).reshaped([-1])
                let b = rhs.asType(.float32).reshaped([-1])
                let cosine = (MLX.sum(a * b)
                    / (MLX.sqrt(MLX.sum(a.square()))
                        * MLX.sqrt(MLX.sum(b.square()))
                        + MLXArray(Float(1e-8))))
                    .item(Float.self)
                minimumCosine = min(minimumCosine, cosine)
                maximumAbsoluteDifference = max(
                    maximumAbsoluteDifference,
                    MLX.max(MLX.abs(a - b)).item(Float.self))
            }
        }
        print(
            "Tool-response cache parity: minimum cosine \(minimumCosine), "
                + "maximum absolute difference \(maximumAbsoluteDifference)")
        XCTAssertGreaterThan(minimumCosine, 0.999)
        // BF16 Mamba recurrence can accumulate elementwise rounding in a
        // different order while retaining nearly identical direction.
        XCTAssertLessThan(maximumAbsoluteDifference, 2)

        let sequentialSpeechCache = await sequential
            .speechCacheStateForTesting()
        let batchedSpeechCache = await batched.speechCacheStateForTesting()
        XCTAssertEqual(
            sequentialSpeechCache.map(\.count),
            batchedSpeechCache.map(\.count))
        var minimumSpeechCacheCosine: Float = 1
        for (sequentialLayer, batchedLayer) in zip(
            sequentialSpeechCache, batchedSpeechCache
        ) {
            for (lhs, rhs) in zip(sequentialLayer, batchedLayer) {
                XCTAssertEqual(lhs.shape, rhs.shape)
                guard lhs.shape == rhs.shape else { continue }
                let a = lhs.asType(.float32).reshaped([-1])
                let b = rhs.asType(.float32).reshaped([-1])
                let cosine = (MLX.sum(a * b)
                    / (MLX.sqrt(MLX.sum(a.square()))
                        * MLX.sqrt(MLX.sum(b.square()))
                        + MLXArray(Float(1e-8))))
                    .item(Float.self)
                minimumSpeechCacheCosine = min(
                    minimumSpeechCacheCosine, cosine)
            }
        }
        print(
            "Tool-response speech-cache parity: minimum cosine "
                + "\(minimumSpeechCacheCosine)")
        XCTAssertGreaterThan(minimumSpeechCacheCosine, 0.999)
    }

    func testToolResponseWithoutForcedTextReauthorizesNativeContinuation() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let session = try await model.startSession(
            systemPrompt: VoiceChatSession.baseSystemPrompt,
            speech: .init(realtimeIdleOptimization: true),
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        await session.suppressFunctionFastPathForTesting()

        try await session.injectFunctionResponse(
            #"{"ok":true,"tool":"list_lists","result":"[]"}"#)
        let optionalSilenceEmbedding = await session
            .functionSilenceEmbeddingForTesting()
        let silenceEmbedding = try XCTUnwrap(optionalSilenceEmbedding)
        _ = try await session.prefillFunctionResponseForTesting(
            audioEmbedding: silenceEmbedding)

        let turnState = await session.turnTakingStateForTesting()
        XCTAssertTrue(turnState.speechConfirmed)
        XCTAssertEqual(
            turnState.consecutiveBlankFrames,
            VoiceChatTurnTakingParameters.nvidiaTurnTakingFallbackFrames - 1)
        let hasPendingOutput = await session.hasPendingFunctionOutput()
        XCTAssertTrue(
            hasPendingOutput,
            "a finite-input caller must keep ticking until the result-conditioned assistant turn ends")
    }

    /// Regression for live captions that appeared during MCP work but were
    /// never delivered to the shared language timeline. Perception/RNN-T run
    /// once during the provider wait; their evaluated modality embeddings are
    /// then replayed after the result without replaying wall-clock silence.
    func testSpeechDuringToolWaitIsReplayedIntoTheModel() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete VoiceChat bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let tools = #"[{"description":"List active reminders","name":"list_reminders","parameters":{"type":"object","properties":{}}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools)
        let session = try await model.startSession(
            systemPrompt: prompt,
            speech: .init(
                iterations: 4,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)

        let requestURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-tool-wait-\(UUID().uuidString).aiff")
        let followUpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-tool-wait-followup-\(UUID().uuidString).aiff")
        defer {
            try? FileManager.default.removeItem(at: requestURL)
            try? FileManager.default.removeItem(at: followUpURL)
        }
        try synthesize("What reminders do I have?", to: requestURL)
        try synthesize("Who are you?", to: followUpURL)
        let request = trimTrailingSilence(try loadMono16k(requestURL))
        let followUp = trimTrailingSilence(try loadMono16k(followUpURL))
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let silence = [Float](repeating: 0, count: frameSize)

        var nativeCall: String?
        func captureCall(_ events: [VoiceChatFrameEvent]) {
            if nativeCall == nil {
                nativeCall = events.compactMap(\.functionCall).first
            }
        }
        for start in stride(from: 0, to: request.count, by: frameSize) {
            await session.assistFunctionFastPathIfStalled()
            captureCall(try await session.pushAudio(Array(
                request[start ..< min(request.count, start + frameSize)])))
        }
        for _ in 0 ..< 120 where nativeCall == nil {
            await session.assistFunctionFastPathIfStalled()
            captureCall(try await session.pushAudio(silence))
        }
        let completedCall = try XCTUnwrap(nativeCall)
        XCTAssertTrue(completedCall.contains("list_reminders"))
        let waitingForResponse = await session.isWaitingForFunctionResponse()
        XCTAssertTrue(waitingForResponse)

        // Simulate the user asking a follow-up while the external provider is
        // still running. These frames must remain out of the function result's
        // causal prefix, but they must not disappear.
        for start in stride(from: 0, to: followUp.count, by: frameSize) {
            _ = try await session.pushAudio(Array(
                followUp[start ..< min(followUp.count, start + frameSize)]))
        }
        for _ in 0 ..< 4 { _ = try await session.pushAudio(silence) }
        let waitingStatistics = await session
            .deferredMicrophoneStatisticsForTesting()
        XCTAssertGreaterThan(waitingStatistics.bufferedFrames, 0)
        XCTAssertEqual(waitingStatistics.replayedFrames, 0)

        try await session.injectFunctionResponse(
            #"{"ok":true,"tool":"list_reminders","result":[{"id":"r1","name":"Morning"}]}"#)
        var microphonePushMilliseconds: [Double] = []
        var responsePushMilliseconds: [Double] = []
        var replayPushMilliseconds: [Double] = []
        var speechCachePushMilliseconds: [Double] = []
        var speakingPushMilliseconds: [Double] = []
        var speakingDecisionMilliseconds: [Double] = []
        var speakingSynthesisMilliseconds: [Double] = []
        var reportedModelMilliseconds: [Double] = []
        var maximumReplayProgressBetweenPushes = 0
        var mutedReplayEvents = 0
        var nextFrameDeadline = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< 160 {
            let before = await session
                .deferredMicrophoneStatisticsForTesting()
            let responseWasActive = await session
                .functionResponseMetrics()?.active == true
            // Match the live CLI's RTF boundary: actor-isolated diagnostic
            // queries are not part of servicing the captured audio frame.
            let pushStarted = DispatchTime.now().uptimeNanoseconds
            let events = try await session.pushAudio(silence)
            let pushFinished = DispatchTime.now().uptimeNanoseconds
            microphonePushMilliseconds.append(Double(
                pushFinished - pushStarted) / 1_000_000)
            let measured = microphonePushMilliseconds.last!
            if responseWasActive {
                responsePushMilliseconds.append(measured)
            } else if before.bufferedFrames > 0 {
                replayPushMilliseconds.append(measured)
            } else if before.pendingSpeechCacheFrames > 0 {
                speechCachePushMilliseconds.append(measured)
            } else if events.contains(where: \.speaking) {
                speakingPushMilliseconds.append(measured)
                for event in events where event.speaking {
                    speakingDecisionMilliseconds.append(
                        event.decisionLatencyMilliseconds)
                    speakingSynthesisMilliseconds.append(
                        event.synthesisLatencyMilliseconds)
                }
            }
            reportedModelMilliseconds.append(events.reduce(0) {
                $0 + $1.perceptionLatencyMilliseconds
                    + $1.decisionLatencyMilliseconds
                    + $1.synthesisLatencyMilliseconds
            })
            mutedReplayEvents += events.filter { !$0.playbackRequired }.count
            let statistics = await session
                .deferredMicrophoneStatisticsForTesting()
            let replayedThisPush = statistics.replayedFrames
                - before.replayedFrames
            if replayedThisPush > 0 {
                XCTAssertEqual(
                    replayedThisPush,
                    1,
                    "a live callback must never drain multiple old semantic frames")
                let replayEvents = events.filter { !$0.playbackRequired }
                XCTAssertFalse(
                    replayEvents.isEmpty,
                    "every causal replay callback must suppress elapsed audio")
                XCTAssertTrue(replayEvents.allSatisfy {
                    $0.textToken == model.tokenizer.padID
                        && !$0.speaking
                        && $0.audio.allSatisfy { abs($0) <= 1e-8 }
                }, "replay must be a muted PAD timeline update")
            }
            maximumReplayProgressBetweenPushes = max(
                maximumReplayProgressBetweenPushes, replayedThisPush)
            let hasPendingOutput = await session.hasPendingFunctionOutput()
            if statistics.bufferedFrames == 0,
               statistics.replayedFrames > 0,
               statistics.pendingSpeechCacheFrames == 0,
               !hasPendingOutput {
                break
            }

            nextFrameDeadline += UInt64(
                VoiceChatSession.frameMilliseconds) * 1_000_000
            let now = DispatchTime.now().uptimeNanoseconds
            if nextFrameDeadline > now {
                try await Task.sleep(
                    nanoseconds: nextFrameDeadline - now)
            }
        }

        let finalStatistics = await session
            .deferredMicrophoneStatisticsForTesting()
        XCTAssertEqual(finalStatistics.bufferedFrames, 0)
        XCTAssertEqual(finalStatistics.pendingSpeechCacheFrames, 0)
        XCTAssertGreaterThanOrEqual(
            finalStatistics.replayedFrames,
            waitingStatistics.bufferedFrames)
        XCTAssertEqual(finalStatistics.droppedFrames, 0)
        XCTAssertGreaterThan(maximumReplayProgressBetweenPushes, 0)
        XCTAssertEqual(maximumReplayProgressBetweenPushes, 1)
        XCTAssertGreaterThan(
            mutedReplayEvents, 0,
            "causal replay must be visible but must not replay elapsed wall-clock audio")
        let sortedPushes = microphonePushMilliseconds.sorted()
        let pushP95 = sortedPushes[min(
            sortedPushes.count - 1,
            Int(Double(sortedPushes.count) * 0.95))]
        print(String(
            format: "Asynchronous deferred replay: %d microphone pushes, mean %.1f ms / p95 %.1f ms / max %.1f ms",
            microphonePushMilliseconds.count,
            microphonePushMilliseconds.reduce(0, +)
                / Double(microphonePushMilliseconds.count),
            pushP95,
            microphonePushMilliseconds.max() ?? 0))
        func phaseSummary(_ name: String, _ values: [Double]) -> String {
            guard !values.isEmpty else { return "\(name) 0 pushes" }
            return String(
                format: "%@ %d pushes, mean %.1f ms / max %.1f ms",
                name, values.count,
                values.reduce(0, +) / Double(values.count),
                values.max() ?? 0)
        }
        print(phaseSummary("response", responsePushMilliseconds))
        print(phaseSummary("replay", replayPushMilliseconds))
        print(phaseSummary("speech cache", speechCachePushMilliseconds))
        print(phaseSummary("assistant speech", speakingPushMilliseconds))
        print(phaseSummary(
            "assistant language", speakingDecisionMilliseconds))
        print(phaseSummary(
            "assistant synthesis", speakingSynthesisMilliseconds))
        print(String(
            format: "event-attributed model work mean %.1f ms / max %.1f ms",
            reportedModelMilliseconds.reduce(0, +)
                / Double(reportedModelMilliseconds.count),
            reportedModelMilliseconds.max() ?? 0))
        print(
            "Deferred replay function-head statistics: "
                + "\(await session.functionHeadEvaluationStatistics())")
        let transcript = await session.userTranscript().lowercased()
        XCTAssertTrue(transcript.contains("who"), transcript)
    }

    func testGeneralConversationKeepsFunctionHeadRealtime() async throws {
        guard ProcessInfo.processInfo.environment["VOICECHAT_PERFORMANCE_TEST"] == "1" else {
            throw XCTSkip("set VOICECHAT_PERFORMANCE_TEST=1 for the function-head timing gate")
        }
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-general-\(UUID().uuidString).aiff")
        defer { try? FileManager.default.removeItem(at: audioURL) }
        try synthesize(
            "Hey, how are you?",
            to: audioURL)

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let tools = #"[{"description":"Create a new reminder in a list","name":"create_reminder","parameters":{"type":"object","properties":{"name":{"type":"string"},"list":{"type":"string"}},"required":["name","list"]}},{"description":"List reminder lists","name":"list_lists","parameters":{"type":"object","properties":{}}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools)
        let session = try await model.startSession(
            systemPrompt: prompt,
            speech: .init(
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let samples = try loadMono16k(audioURL)
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let started = DispatchTime.now().uptimeNanoseconds
        for start in stride(from: 0, to: samples.count, by: frameSize) {
            _ = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)]))
        }
        let silence = [Float](repeating: 0, count: frameSize)
        for _ in 0 ..< 50 {
            _ = try await session.pushAudio(silence)
        }
        let elapsedMilliseconds = Double(
            DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
        let summary = await session.summary()
        let statistics = await session.functionHeadEvaluationStatistics()
        let rtf = elapsedMilliseconds
            / Double(summary.frames * VoiceChatSession.frameMilliseconds)

        print("Function-head statistics: \(statistics)")
        print(String(format: "Function-head live RTF %.2f", rtf))
        XCTAssertEqual(statistics.openCallFrames, 0)
        XCTAssertLessThan(rtf, 1.0)
    }

    func testWriteCallProducesModelGeneratedConfirmationBeforeExecution() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let audioURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-function-\(UUID().uuidString).aiff")
        defer { try? FileManager.default.removeItem(at: audioURL) }
        try synthesizeRequest(to: audioURL)

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let tools = #"[{"description":"Create a new reminder in a list","name":"create_reminder","parameters":{"type":"object","properties":{"name":{"type":"string"},"list":{"type":"string"}},"required":["name","list"]}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools,
            requiresWriteConfirmation: true)
        let session = try await model.startSession(
            systemPrompt: prompt,
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let samples = try loadMono16k(audioURL)
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        var call: String?
        var injected = false
        var sawAudibleConfirmation = false

        func process(_ events: [VoiceChatFrameEvent]) async throws {
            for event in events {
                if let functionCall = event.functionCall, !injected {
                    call = functionCall
                    injected = true
                    try await session.injectFunctionResponse(
                        #"{"confirmation_required":true,"executed":false,"tool":"create_reminder"}"#,
                        requireAssistantReplyBeforeNextFunctionCall: true)
                }
                if injected,
                   event.audio.contains(where: { abs($0) > 0.0001 })
                {
                    sawAudibleConfirmation = true
                }
            }
        }

        for start in stride(from: 0, to: samples.count, by: frameSize) {
            try await process(try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)])))
        }
        let silence = [Float](repeating: 0, count: frameSize)
        for _ in 0 ..< 250 {
            try await process(try await session.pushAudio(silence))
            if injected,
               !(await session.hasPendingFunctionOutput()),
               !(await session.reply())
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                break
            }
        }

        let statistics = await session.functionHeadEvaluationStatistics()
        print("Function-call statistics: \(statistics)")
        print("Function-call payload: \(call ?? "none")")
        guard statistics.asynchronousCallSteps > 0 else {
            throw XCTSkip(
                "the checkpoint chose ordinary text for this synthesized request; "
                    + "no native function call was available to measure")
        }
        let optionalDecodeMetrics = await session.functionCallDecodeMetrics()
        let decodeMetrics = try XCTUnwrap(optionalDecodeMetrics)
        print("Function-call decode metrics: \(decodeMetrics)")
        XCTAssertLessThan(
            statistics.openCallFrames,
            statistics.asynchronousCallSteps,
            "the fixed JSON prefix should use causal prefill instead of full-vocabulary projections")
        XCTAssertGreaterThan(statistics.openCallFrames, 0)
        XCTAssertGreaterThan(statistics.asynchronousResponseSteps, 0)
        XCTAssertGreaterThan(
            statistics.asynchronousResponsePrefillBatches, 0)
        XCTAssertLessThan(
            statistics.asynchronousResponsePrefillBatches,
            statistics.asynchronousResponseSteps)
        XCTAssertEqual(statistics.asynchronousCallTimeouts, 0)
        XCTAssertEqual(statistics.asynchronousResponseTimeouts, 0)
        XCTAssertEqual(statistics.asynchronousCallInterruptions, 0)
        XCTAssertGreaterThan(statistics.microphoneFramesDuringAsyncWork, 0)
        XCTAssertFalse(decodeMetrics.active)
        XCTAssertTrue(decodeMetrics.completed)
        XCTAssertGreaterThan(decodeMetrics.elapsedMilliseconds, 0)
        XCTAssertEqual(
            decodeMetrics.tokenSteps,
            statistics.asynchronousCallSteps)
        XCTAssertGreaterThan(decodeMetrics.tokensPerSecond, 0)
        XCTAssertTrue(call?.contains("create_reminder") == true, "function call: \(call ?? "none")")
        let reply = await session.reply()
        XCTAssertFalse(
            reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "the checkpoint should generate its own confirmation reply")
        XCTAssertTrue(sawAudibleConfirmation)
    }

    /// Profiles natural spoken requests across the complete model-native tool
    /// cycle. The fixed 200 ms provider delay is intentional: it makes the
    /// external service component visible while proving that microphone work
    /// continues on the session actor during the wait.
    func testNaturalToolPhraseLatencyProfile() async throws {
        guard ProcessInfo.processInfo.environment["VOICECHAT_PERFORMANCE_TEST"]
            == "1" else {
            throw XCTSkip("set VOICECHAT_PERFORMANCE_TEST=1")
        }
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete VoiceChat bundle")
        }

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let remindersTools = #"[{"description":"List active reminders across every reminder list","name":"list_reminders","parameters":{"type":"object","properties":{"search":{"type":"string"}}}},{"description":"Create a new reminder; omit list to use the system default","name":"create_reminder","parameters":{"type":"object","properties":{"name":{"type":"string"},"list":{"type":"string"},"due_date":{"type":"string"}},"required":["name"]}},{"description":"Update a reminder using an id returned by list_reminders","name":"update_reminder","parameters":{"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"},"due_date":{"type":"string"}},"required":["id"]}}]"#
        let createOnly = #"[{"description":"Create a new reminder in a list","name":"create_reminder","parameters":{"type":"object","properties":{"name":{"type":"string"},"list":{"type":"string"}},"required":["name","list"]}}]"#
        let allCases = [
            ToolPhraseProfileCase(
                id: "list-question",
                phrase: "What reminders do I have?",
                expectedTool: "list_reminders",
                availableToolsJSON: remindersTools,
                responseJSON: #"{"ok":true,"tool":"list_reminders","result":[{"id":"REM-123","name":"Phone John","due_date":"2026-08-14 20:00:00"}]}"#),
            ToolPhraseProfileCase(
                id: "list-command",
                phrase: "List my reminders.",
                expectedTool: "list_reminders",
                availableToolsJSON: remindersTools,
                responseJSON: #"{"ok":true,"tool":"list_reminders","result":[{"id":"REM-123","name":"Phone John","due_date":"2026-08-14 20:00:00"}]}"#),
            ToolPhraseProfileCase(
                id: "list-active",
                phrase: "Show my active reminders.",
                expectedTool: "list_reminders",
                availableToolsJSON: remindersTools,
                responseJSON: #"{"ok":true,"tool":"list_reminders","result":[{"id":"REM-123","name":"Phone John","due_date":"2026-08-14 20:00:00"}]}"#),
            ToolPhraseProfileCase(
                id: "create",
                phrase: "Create a reminder called Phone John in the Reminders list.",
                expectedTool: "create_reminder",
                availableToolsJSON: createOnly,
                responseJSON: #"{"ok":true,"tool":"create_reminder"}"#),
        ]
        let selectedID = ProcessInfo.processInfo.environment[
            "VOICECHAT_TOOL_PHRASE"]
        let providerDelayMilliseconds = min(
            10_000,
            max(
                0,
                Double(ProcessInfo.processInfo.environment[
                    "VOICECHAT_TOOL_PROVIDER_DELAY_MS"] ?? "") ?? 200))
        let cases = selectedID.map { id in
            allCases.filter { $0.id == id }
        } ?? allCases
        if let selectedID, cases.isEmpty {
            XCTFail(
                "unknown VOICECHAT_TOOL_PHRASE=\(selectedID); use "
                    + allCases.map(\.id).joined(separator: ", "))
            return
        }

        var completedProfiles = 0
        for profileCase in cases {
            let audioURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "voicechat-tool-profile-\(UUID().uuidString).aiff")
            defer { try? FileManager.default.removeItem(at: audioURL) }
            try synthesize(profileCase.phrase, to: audioURL)
            let synthesized = try loadMono16k(audioURL)
            let samples = trimTrailingSilence(synthesized)
            let prompt = try VoiceChatSession.toolCallingSystemPrompt(
                availableToolsJSON: profileCase.availableToolsJSON)
            let session = try await model.startSession(
                systemPrompt: prompt,
                speech: .init(
                    iterations: 4,
                    recentContextFrames: 250,
                    realtimeIdleOptimization: true),
                streamUserTranscript: true,
                turnTaking: .functionCallingRealtime,
                functionCallingEnabled: true)

            let frameSize = VoiceChatSession.inputSamplesPerFrame
            var nextDeadline = DispatchTime.now().uptimeNanoseconds
            var callCompleted: UInt64?
            var callPayload: String?
            var toolTask: Task<SimulatedToolTiming, Error>?
            var microphonePushes: [TimedMicrophonePush] = []
            var firstSpeech: UInt64?
            var ordinarySpeechWithoutTool: UInt64?

            func startToolIfPresent(
                _ events: [VoiceChatFrameEvent],
                observedAt: UInt64
            ) -> Task<SimulatedToolTiming, Error>? {
                guard callPayload == nil,
                      let payload = events.compactMap(\.functionCall).first else {
                    return nil
                }
                callPayload = payload
                callCompleted = observedAt
                return Task {
                    let started = DispatchTime.now().uptimeNanoseconds
                    try await Task.sleep(nanoseconds: UInt64(
                        providerDelayMilliseconds * 1_000_000))
                    let completed = DispatchTime.now().uptimeNanoseconds
                    try await session.injectFunctionResponse(
                        profileCase.responseJSON)
                    let responseAccepted = DispatchTime.now().uptimeNanoseconds
                    return SimulatedToolTiming(
                        startedAtNanoseconds: started,
                        completedAtNanoseconds: completed,
                        responseAcceptedAtNanoseconds: responseAccepted,
                        elapsedMilliseconds: Double(completed - started)
                            / 1_000_000)
                }
            }

            for start in stride(from: 0, to: samples.count, by: frameSize) {
                nextDeadline += UInt64(VoiceChatSession.frameMilliseconds)
                    * 1_000_000
                let frame = Array(
                    samples[start ..< min(samples.count, start + frameSize)])
                await session.assistFunctionFastPathIfStalled()
                let events = try await session.pushAudio(frame)
                let observedAt = DispatchTime.now().uptimeNanoseconds
                if toolTask == nil {
                    toolTask = startToolIfPresent(events, observedAt: observedAt)
                }
                let now = DispatchTime.now().uptimeNanoseconds
                if nextDeadline > now {
                    try await Task.sleep(nanoseconds: nextDeadline - now)
                }
            }
            let inputEnded = DispatchTime.now().uptimeNanoseconds

            let silence = [Float](repeating: 0, count: frameSize)
            let timeout = inputEnded + 20_000_000_000
            while DispatchTime.now().uptimeNanoseconds < timeout,
                  firstSpeech == nil {
                nextDeadline += UInt64(VoiceChatSession.frameMilliseconds)
                    * 1_000_000
                let pushStarted = DispatchTime.now().uptimeNanoseconds
                await session.assistFunctionFastPathIfStalled()
                let events = try await session.pushAudio(silence)
                let pushEnded = DispatchTime.now().uptimeNanoseconds
                microphonePushes.append(TimedMicrophonePush(
                    startedAtNanoseconds: pushStarted,
                    completedAtNanoseconds: pushEnded,
                    deadlineNanoseconds: nextDeadline))
                if toolTask == nil {
                    toolTask = startToolIfPresent(events, observedAt: pushEnded)
                }
                if callPayload != nil, events.contains(where: \.speaking) {
                    firstSpeech = pushEnded
                } else if callPayload == nil,
                          events.contains(where: \.speaking)
                {
                    // Text and function selection compete at the same model
                    // position. Once an ordinary spoken turn begins, this
                    // request cannot subsequently become a native tool call.
                    ordinarySpeechWithoutTool = pushEnded
                }
                let now = DispatchTime.now().uptimeNanoseconds
                if nextDeadline > now {
                    try await Task.sleep(nanoseconds: nextDeadline - now)
                }
                if ordinarySpeechWithoutTool != nil { break }
            }

            let optionalToolTiming = try await toolTask?.value
            let optionalCallMetrics = await session.functionCallDecodeMetrics()
            let optionalResponseMetrics = await session.functionResponseMetrics()
            let endToCallComplete = callCompleted.map {
                milliseconds(from: inputEnded, to: $0)
            }
            let endToCallStart = endToCallComplete.map {
                $0 - (optionalCallMetrics?.elapsedMilliseconds ?? 0)
            }
            let endToFirstSpeech = firstSpeech.map {
                milliseconds(from: inputEnded, to: $0)
            }
            let callStart = callCompleted.map {
                $0 - UInt64(max(
                    0,
                    optionalCallMetrics?.elapsedMilliseconds ?? 0) * 1_000_000)
            }
            let callMicrophone = zip(
                callStart.map { [$0] } ?? [],
                callCompleted.map { [$0] } ?? [])
                .flatMap { start, end in
                    microphonePushes.filter { $0.overlaps(start, end) }
                        .map(\.elapsedMilliseconds)
                }
            let providerMicrophone = optionalToolTiming.map { timing in
                microphonePushes.filter {
                    $0.overlaps(
                        timing.startedAtNanoseconds,
                        timing.completedAtNanoseconds)
                }.map(\.elapsedMilliseconds)
            } ?? []
            let resultMicrophone: [Double] = optionalToolTiming.flatMap { timing in
                optionalResponseMetrics.map { metrics in
                    let end = timing.responseAcceptedAtNanoseconds + UInt64(
                        max(0, metrics.elapsedMilliseconds) * 1_000_000)
                    return microphonePushes.filter {
                        $0.overlaps(timing.responseAcceptedAtNanoseconds, end)
                    }.map(\.elapsedMilliseconds)
                }
            } ?? []
            let allMicrophone = microphonePushes.map(\.elapsedMilliseconds)
            let microphoneBudgetOverruns = microphonePushes.map(
                \.budgetOverrunMilliseconds)
            let (callMicP95, callMicMaximum) = latencyDistribution(
                callMicrophone)
            let (providerMicP95, providerMicMaximum) = latencyDistribution(
                providerMicrophone)
            let (resultMicP95, resultMicMaximum) = latencyDistribution(
                resultMicrophone)
            let (micP95, micMaximum) = latencyDistribution(allMicrophone)
            let (budgetOverrunP95, budgetOverrunMaximum) = latencyDistribution(
                microphoneBudgetOverruns)

            print(String(
                format: "Tool phrase [%@: %@]: call start %.0f ms, call complete %.0f ms, native decode %.0f ms/%d steps/%.1f tok/s (model %.0f ms, voice cache %.0f ms, interleave %.0f ms), provider %.0f ms, result sync %.0f ms/%d tokens/%d batches (language %.0f ms, voice cache %.0f ms, interleave %.0f ms), first post-tool speech %.0f ms, mic service p95 %.1f ms max %.1f ms, 80 ms budget overrun p95 %.1f ms max %.1f ms; phase mic service call %.1f/%.1f ms, provider %.1f/%.1f ms, result %.1f/%.1f ms (p95/max)",
                profileCase.id,
                profileCase.phrase,
                endToCallStart ?? -1,
                endToCallComplete ?? -1,
                optionalCallMetrics?.elapsedMilliseconds ?? -1,
                optionalCallMetrics?.tokenSteps ?? -1,
                optionalCallMetrics?.tokensPerSecond ?? -1,
                optionalCallMetrics?.modelMilliseconds ?? -1,
                optionalCallMetrics?.speechCacheMilliseconds ?? -1,
                optionalCallMetrics?.interleavingMilliseconds ?? -1,
                optionalToolTiming?.elapsedMilliseconds ?? -1,
                optionalResponseMetrics?.elapsedMilliseconds ?? -1,
                optionalResponseMetrics?.tokenSteps ?? -1,
                optionalResponseMetrics?.prefillBatches ?? -1,
                optionalResponseMetrics?.languageCacheMilliseconds ?? -1,
                optionalResponseMetrics?.speechCacheMilliseconds ?? -1,
                optionalResponseMetrics?.interleavingMilliseconds ?? -1,
                endToFirstSpeech ?? -1,
                micP95,
                micMaximum,
                budgetOverrunP95,
                budgetOverrunMaximum,
                callMicP95,
                callMicMaximum,
                providerMicP95,
                providerMicMaximum,
                resultMicP95,
                resultMicMaximum))
            print("Tool phrase transcript: \(await session.userTranscript())")
            print("Tool phrase payload: \(callPayload ?? "none")")

            guard let callPayload, let firstSpeech else {
                if optionalCallMetrics?.active == true {
                    XCTFail(
                        "native function decode did not complete for "
                            + profileCase.phrase)
                    continue
                }
                print(
                    "Tool phrase outcome: checkpoint selected no native tool; "
                        + "this is a tool-selection quality observation")
                continue
            }
            XCTAssertTrue(
                callPayload.contains(profileCase.expectedTool),
                "unexpected call for \(profileCase.phrase): \(callPayload)")
            XCTAssertTrue(
                callPayload.hasPrefix("<TOOLCALL>"),
                "native payload must retain its trained opening marker: \(callPayload)")
            XCTAssertTrue(
                callPayload.hasSuffix("</TOOLCALL>"),
                "native payload must retain its trained closing marker: \(callPayload)")
            let callJSON = callPayload
                .replacingOccurrences(of: "<TOOLCALL>", with: "")
                .replacingOccurrences(of: "</TOOLCALL>", with: "")
            let callData = try XCTUnwrap(callJSON.data(using: .utf8))
            XCTAssertNotNil(
                try JSONSerialization.jsonObject(with: callData)
                    as? [[String: Any]],
                "native payload must remain valid JSON: \(callPayload)")
            XCTAssertLessThan(
                milliseconds(from: inputEnded, to: firstSpeech),
                8_000)
            XCTAssertLessThan(micMaximum, 600)
            XCTAssertTrue(optionalCallMetrics?.completed == true)
            XCTAssertTrue(optionalResponseMetrics?.completed == true)
            completedProfiles += 1
        }
        if selectedID == nil {
            XCTAssertGreaterThan(
                completedProfiles,
                0,
                "at least one natural phrase must complete a native tool cycle")
        }
    }

    /// Regression coverage for a list followed by an immediate write. The
    /// second call must complete on the same caches without inserting a
    /// confirmation turn or leaving a later native call on its start token.
    func testSequentialNativeToolCyclesKeepMakingProgress() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete VoiceChat bundle")
        }

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let tools = #"[{"description":"List active reminders across every reminder list","name":"list_reminders","parameters":{"type":"object","properties":{"search":{"type":"string"}}}},{"description":"Update a reminder using the id returned by list_reminders. Set completed true to remove it from active reminders","name":"update_reminder","parameters":{"type":"object","properties":{"id":{"type":"string"},"completed":{"type":"boolean"}},"required":["id"]}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools)
        let session = try await model.startSession(
            systemPrompt: prompt,
            speech: .init(
                iterations: 4,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let silence = [Float](repeating: 0, count: frameSize)
        let listResult = #"{"ok":true,"tool":"list_reminders","result":[{"id":"r1","name":"Morning"},{"id":"r2","name":"Come back home"}]}"#
        let updateSuccess = #"{"ok":true,"executed":true,"tool":"update_reminder"}"#

        func runCycle(_ phrase: String, cycle: Int) async throws -> Bool {
            let audioURL = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "voicechat-sequential-tool-\(cycle)-\(UUID().uuidString).aiff")
            defer { try? FileManager.default.removeItem(at: audioURL) }
            try synthesize(phrase, to: audioURL)
            let samples = trimTrailingSilence(try loadMono16k(audioURL))
            var call: String?
            var injected = false
            var sawPostToolSpeech = false
            var sawPostToolAudibleAudio = false
            var sawPostToolEnd = false
            var postToolContentTokens = 0
            var postToolPadTokens = 0
            var postToolBOSTokens = 0
            var postToolEOSTokens = 0
            var postToolLastTokens: [Int] = []

            func process(_ events: [VoiceChatFrameEvent]) async throws {
                for event in events {
                    if let payload = event.functionCall, !injected {
                        call = payload
                        injected = true
                        let response: String
                        switch cycle {
                        case 1: response = listResult
                        default: response = updateSuccess
                        }
                        try await session.injectFunctionResponse(response)
                    }
                    if injected {
                        if event.textToken == model.tokenizer.bosID {
                            postToolBOSTokens += 1
                        } else if event.textToken == model.tokenizer.eosID {
                            postToolEOSTokens += 1
                        } else if event.textToken == model.tokenizer.padID {
                            postToolPadTokens += 1
                        } else {
                            postToolContentTokens += 1
                        }
                        postToolLastTokens.append(event.textToken)
                        if postToolLastTokens.count > 24 {
                            postToolLastTokens.removeFirst()
                        }
                    }
                    if injected, event.speaking { sawPostToolSpeech = true }
                    if injected,
                       event.audio.contains(where: { abs($0) > 1e-6 }) {
                        sawPostToolAudibleAudio = true
                    }
                    if sawPostToolSpeech,
                       event.textToken == model.tokenizer.eosID {
                        sawPostToolEnd = true
                    }
                }
            }

            for start in stride(from: 0, to: samples.count, by: frameSize) {
                await session.assistFunctionFastPathIfStalled()
                try await process(try await session.pushAudio(Array(
                    samples[start ..< min(samples.count, start + frameSize)])))
            }
            var tailFrames = 0
            while tailFrames < 180, !sawPostToolEnd {
                await session.assistFunctionFastPathIfStalled()
                try await process(try await session.pushAudio(silence))
                tailFrames += 1
            }

            let metric = await session.functionCallDecodeMetrics()
            let turnState = await session.turnTakingStateForTesting()
            let statistics = await session.functionHeadEvaluationStatistics()
            print(
                "Sequential tool cycle \(cycle): call \(call ?? "none"), "
                    + "decode \(String(describing: metric)), "
                    + "post-tool end \(sawPostToolEnd), tokens "
                    + "audible \(sawPostToolAudibleAudio), "
                    + "BOS \(postToolBOSTokens) content \(postToolContentTokens) "
                    + "PAD \(postToolPadTokens) EOS \(postToolEOSTokens), "
                    + "last \(postToolLastTokens), transcript "
                    + "\(await session.userTranscript()), turn state "
                    + "\(turnState), function stats \(statistics)")
            if call == nil { return false }
            let callText = try XCTUnwrap(call)
            let callJSON = callText
                .replacingOccurrences(of: "<TOOLCALL>", with: "")
                .replacingOccurrences(of: "</TOOLCALL>", with: "")
            let callData = try XCTUnwrap(callJSON.data(using: .utf8))
            let calls = try XCTUnwrap(
                JSONSerialization.jsonObject(with: callData)
                    as? [[String: Any]])
            let nativeCall = try XCTUnwrap(calls.first)
            let expectedTool = cycle == 1
                ? "list_reminders"
                : "update_reminder"
            XCTAssertEqual(nativeCall["name"] as? String, expectedTool)
            if cycle > 1 {
                let arguments = try XCTUnwrap(
                    nativeCall["arguments"] as? [String: Any])
                XCTAssertEqual(
                    arguments["id"] as? String,
                    "r1",
                    "update must reuse the exact short ID: \(callText)")
                XCTAssertEqual(
                    arguments["completed"] as? Bool,
                    true,
                    "delete intent must map to completion: \(callText)")
            }
            XCTAssertTrue(metric?.completed == true)
            XCTAssertFalse(metric?.active == true)
            XCTAssertTrue(sawPostToolEnd)
            XCTAssertTrue(
                sawPostToolAudibleAudio,
                "post-tool text was generated without audible PCM")
            return true
        }

        let firstCycleCompleted = try await runCycle(
            "What reminders do I have?", cycle: 1)
        XCTAssertTrue(
            firstCycleCompleted,
            "the stable reminder-list fixture did not open list_reminders")
        guard firstCycleCompleted else { return }
        let listedReply = await session.reply().lowercased()
        XCTAssertTrue(
            listedReply.contains("morning"),
            "the spoken list omitted the first tool result: \(listedReply)")
        XCTAssertTrue(
            listedReply.contains("come back home"),
            "the spoken list omitted the second tool result: \(listedReply)")
        for _ in 0 ..< 8 {
            await session.assistFunctionFastPathIfStalled()
            _ = try await session.pushAudio(silence)
        }
        let secondCycleCompleted = try await runCycle(
            "Can you delete Morning, so how it goes?", cycle: 2)
        XCTAssertTrue(secondCycleCompleted)
        let updateReply = await session.reply().lowercased()
        XCTAssertFalse(updateReply.contains("confirm"), updateReply)
        XCTAssertFalse(updateReply.contains("would you like"), updateReply)
    }

    /// A failed external provider result must not strand the shared function
    /// channel. The model should explain the failure audibly, close that turn,
    /// and then answer an ordinary follow-up in the same long-lived session.
    func testFailedToolResultRecoversToAnAudibleFollowUpTurn() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete VoiceChat bundle")
        }

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let tools = #"[{"description":"List active reminders across every reminder list","name":"list_reminders","parameters":{"type":"object","properties":{"search":{"type":"string"}}}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools)
        let session = try await model.startSession(
            systemPrompt: prompt,
            speech: .init(
                iterations: 4,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let silence = [Float](repeating: 0, count: frameSize)

        let requestURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "voicechat-failed-tool-\(UUID().uuidString).aiff")
        let followUpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "voicechat-failed-tool-followup-\(UUID().uuidString).aiff")
        defer {
            try? FileManager.default.removeItem(at: requestURL)
            try? FileManager.default.removeItem(at: followUpURL)
        }
        try synthesize("What reminders do I have?", to: requestURL)
        try synthesize("Who are you?", to: followUpURL)
        let request = trimTrailingSilence(try loadMono16k(requestURL))
        let followUp = trimTrailingSilence(try loadMono16k(followUpURL))

        var callPayload: String?
        var injectedFailure = false
        var firstContent = false
        var firstAudibleAudio = false
        var firstTurnEnded = false

        func processFirst(_ events: [VoiceChatFrameEvent]) async throws {
            for event in events {
                if let call = event.functionCall, !injectedFailure {
                    callPayload = call
                    injectedFailure = true
                    try await session.injectFunctionResponse(
                        #"{"ok":false,"tool":"list_reminders","error":"provider failed"}"#,
                        requireAssistantReplyBeforeNextFunctionCall: true)
                }
                guard injectedFailure else { continue }
                if event.speaking { firstContent = true }
                if event.audio.contains(where: { abs($0) > 1e-6 }) {
                    firstAudibleAudio = true
                }
                if firstContent, event.textToken == model.tokenizer.eosID {
                    firstTurnEnded = true
                }
            }
        }

        for start in stride(from: 0, to: request.count, by: frameSize) {
            await session.assistFunctionFastPathIfStalled()
            try await processFirst(try await session.pushAudio(Array(
                request[start ..< min(request.count, start + frameSize)])))
        }
        var tailFrames = 0
        while tailFrames < 200, !firstTurnEnded {
            await session.assistFunctionFastPathIfStalled()
            try await processFirst(try await session.pushAudio(silence))
            tailFrames += 1
        }

        let completedCallPayload = try XCTUnwrap(
            callPayload,
            "the stable reminder-list fixture did not open list_reminders")
        XCTAssertTrue(
            completedCallPayload.contains("list_reminders"),
            completedCallPayload)
        XCTAssertTrue(injectedFailure)
        XCTAssertTrue(firstContent)
        XCTAssertTrue(firstAudibleAudio)
        XCTAssertTrue(firstTurnEnded)
        let hasPendingFunctionOutput = await session.hasPendingFunctionOutput()
        XCTAssertFalse(hasPendingFunctionOutput)
        let statistics = await session.functionHeadEvaluationStatistics()
        XCTAssertEqual(statistics.asynchronousCallTimeouts, 0)
        XCTAssertEqual(statistics.asynchronousResponseTimeouts, 0)

        // Leave a small idle boundary, then prove the failure did not poison
        // turn authorization, RNN-T state, the LLM cache, or the EAR-TTS cache.
        for _ in 0 ..< 8 {
            _ = try await session.pushAudio(silence)
        }
        let followUpStart = (await session.events()).count
        for start in stride(from: 0, to: followUp.count, by: frameSize) {
            _ = try await session.pushAudio(Array(
                followUp[start ..< min(followUp.count, start + frameSize)]))
        }
        var secondTurnEnded = false
        var secondTailFrames = 0
        while secondTailFrames < 200, !secondTurnEnded {
            let events = try await session.pushAudio(silence)
            let followUpEvents = (await session.events()).dropFirst(followUpStart)
            let sawContent = followUpEvents.contains(where: \.speaking)
            secondTurnEnded = sawContent && events.contains {
                $0.textToken == model.tokenizer.eosID
            }
            secondTailFrames += 1
        }

        let followUpEvents = (await session.events()).dropFirst(followUpStart)
        XCTAssertTrue(followUpEvents.contains {
            $0.textToken == model.tokenizer.bosID
        })
        XCTAssertTrue(followUpEvents.contains(where: \.speaking))
        XCTAssertTrue(followUpEvents.contains {
            $0.audio.contains(where: { abs($0) > 1e-6 })
        })
        XCTAssertTrue(followUpEvents.contains {
            $0.textToken == model.tokenizer.eosID
        })
        let transcript = await session.userTranscript().lowercased()
        XCTAssertTrue(transcript.contains("who"), transcript)
        let reply = await session.reply().lowercased()
        XCTAssertTrue(
            reply.contains("soniqo"),
            "the ordinary follow-up lost the configured assistant identity: \(reply)")
    }

    private func synthesizeRequest(to url: URL) throws {
        try synthesize(
            "Create a reminder called Phone John in the Reminders list.",
            to: url)
    }

    private func synthesize(_ text: String, to url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = [
            "-v", "Samantha",
            "-r", "170",
            "-o", url.path,
            text,
        ]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "VoiceChatFunctionCallingTests",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: "say failed"])
        }
    }

    private func loadMono16k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(VoiceChatSession.inputSampleRate),
            channels: 1,
            interleaved: false)!
        let converter = try XCTUnwrap(
            AVAudioConverter(from: file.processingFormat, to: format))
        let source = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: source)
        let ratio = format.sampleRate / file.processingFormat.sampleRate
        let output = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(
                Double(source.frameLength) * ratio) + 1_024)!
        var supplied = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if supplied {
                status.pointee = .endOfStream
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return source
        }
        if let conversionError { throw conversionError }
        let data = try XCTUnwrap(output.floatChannelData?[0])
        return Array(UnsafeBufferPointer(
            start: data, count: Int(output.frameLength)))
    }

    private func trimTrailingSilence(_ samples: [Float]) -> [Float] {
        guard let lastActive = samples.lastIndex(where: { abs($0) > 0.001 }) else {
            return samples
        }
        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let end = min(samples.count, lastActive + 1 + frameSize)
        let roundedEnd = min(
            samples.count,
            ((end + frameSize - 1) / frameSize) * frameSize)
        return Array(samples[..<roundedEnd])
    }

    private func milliseconds(from start: UInt64, to end: UInt64) -> Double {
        if end >= start {
            return Double(end - start) / 1_000_000
        }
        return -Double(start - end) / 1_000_000
    }

    private func latencyDistribution(
        _ values: [Double]
    ) -> (p95: Double, maximum: Double) {
        let sorted = values.sorted()
        guard !sorted.isEmpty else { return (0, 0) }
        return (
            sorted[min(sorted.count - 1, sorted.count * 95 / 100)],
            sorted.last ?? 0)
    }
}
