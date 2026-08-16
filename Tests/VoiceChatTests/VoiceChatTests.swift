import AVFoundation
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import VoiceChat

/// End-to-end tests for the VoiceChat perception and language bundles.
///
/// The weights are 10–19 GB and are not fetched automatically, so every test
/// that needs them skips unless `VOICECHAT_BUNDLE` points at an exported
/// directory containing `encoder/` and `llm/`. The tests that need no weights
/// run everywhere and cover the parts most likely to regress silently.
final class VoiceChatTests: XCTestCase {
    func testToolCallingPromptUsesCheckpointProtocolAndGroundsActions() throws {
        let tools = #"[{"description":"Create a reminder","name":"create_reminder","parameters":{"type":"object"}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools)

        XCTAssertTrue(prompt.contains("<AVAILABLE_TOOLS>\(tools)</AVAILABLE_TOOLS>"))
        XCTAssertTrue(prompt.contains("<TOOLCALL>"))
        XCTAssertTrue(prompt.contains("<TOOL_RESPONSE>"))
        XCTAssertTrue(prompt.contains(
            "Do not claim an external action succeeded until its tool response says it succeeded."))
        XCTAssertTrue(prompt.contains(
            "Never ask for confirmation before or after a successful tool response."))
        XCTAssertTrue(prompt.contains("you MUST call that tool"))
        XCTAssertTrue(prompt.contains(
            "Match requests to tools by meaning and description"))
        XCTAssertTrue(prompt.contains(
            "use the broad list tool when the user does not name a specific item"))
        XCTAssertTrue(prompt.contains(
            "describe available tools positively and briefly"))
        XCTAssertTrue(prompt.contains(
            "Ask only for the next required missing value"))
        XCTAssertTrue(prompt.contains(
            "exact identifiers returned by earlier tool results"))
        XCTAssertFalse(prompt.contains("confirmation_required"))
        XCTAssertFalse(prompt.contains("emit the exact same tool call again"))
        XCTAssertFalse(prompt.contains("cannot access apps"))
        XCTAssertTrue(prompt.unicodeScalars.allSatisfy(\.isASCII))
    }

    func testToolCallingPromptSupportsExplicitWriteConfirmation() throws {
        let tools = #"[{"description":"Create a reminder","name":"create_reminder","parameters":{"type":"object"}}]"#
        let prompt = try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: tools,
            requiresWriteConfirmation: true)

        XCTAssertTrue(prompt.contains(
            "A confirmation_required response is not a failure:"))
        XCTAssertTrue(prompt.contains("ask for confirmation once"))
        XCTAssertTrue(prompt.contains("emit the exact same tool call again"))
        XCTAssertFalse(prompt.contains(
            "Never ask for confirmation before or after a successful tool response."))
    }

    func testToolCallingPromptRejectsMoreThanFiveTools() {
        let tools: [[String: Any]] = (0 ..< 6).map {
            ["name": "tool_\($0)", "description": "test", "parameters": [:]]
        }
        let data = try! JSONSerialization.data(withJSONObject: tools)

        XCTAssertThrowsError(try VoiceChatSession.toolCallingSystemPrompt(
            availableToolsJSON: String(decoding: data, as: UTF8.self)))
    }

    func testSpeechSilencePolicyPreservesPadsInsideAgentTurn() {
        var state = VoiceChatSpeechTurnState(compactExtendedPads: true)

        XCTAssertTrue(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2))
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 1, padID: 0, bosID: 1, eosID: 2))
        XCTAssertFalse(state.agentIdle)
        for token in [42, 43] {
            XCTAssertFalse(state.shouldForceSilence(
                textToken: token, padID: 0, bosID: 1, eosID: 2))
        }
        XCTAssertEqual(state.contentFrames, 2)
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2),
            "PAD inside an open reply can still carry speech")
        let budget = VoiceChatSpeechTurnState.acousticTailFrameBudget(
            contentFrames: state.contentFrames)
        for _ in 1 ..< (budget - 1) {
            XCTAssertFalse(state.shouldForceSilence(
                textToken: 0, padID: 0, bosID: 1, eosID: 2))
        }
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2),
            "the full content-scaled acoustic budget must remain voiced")
        XCTAssertTrue(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2),
            "PAD beyond the content-scaled tail should become canonical silence")
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 42, padID: 0, bosID: 1, eosID: 2),
            "new speech text must reopen acoustic-tail rendering")
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2))
        XCTAssertTrue(state.shouldForceSilence(
            textToken: 2, padID: 0, bosID: 1, eosID: 2))
        XCTAssertTrue(state.agentIdle)
        XCTAssertTrue(state.shouldForceSilence(
            textToken: 0, padID: 0, bosID: 1, eosID: 2))
    }

    func testExactSpeechSilencePolicyNeverCompactsInTurnPads() {
        var state = VoiceChatSpeechTurnState()
        XCTAssertFalse(state.shouldForceSilence(
            textToken: 1, padID: 0, bosID: 1, eosID: 2))
        for _ in 0 ..< 100 {
            XCTAssertFalse(state.shouldForceSilence(
                textToken: 0, padID: 0, bosID: 1, eosID: 2))
        }
        XCTAssertFalse(state.agentIdle)
    }


    private var bundle: URL? {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else { return nil }
        return URL(fileURLWithPath: path)
    }

    private func requireBundle() throws -> URL {
        guard let bundle else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a directory containing encoder/ and llm/")
        }
        return bundle
    }

    // MARK: - No weights required

    func testVoiceChatFrontendUsesNeMoReflectionPadding() {
        let values = MLXArray([Float(0), 1, 2, 3])
        let actual = voiceChatReflectPad(values, padding: 2)
        eval(actual)
        XCTAssertEqual(actual.asArray(Float.self), [2, 1, 0, 1, 2, 3, 2, 1])

        let short = voiceChatReflectPad(MLXArray([Float(4), 9]), padding: 3)
        eval(short)
        XCTAssertEqual(short.asArray(Float.self), [9, 4, 9, 4, 9, 4, 9, 4])
    }

    /// The streaming mask is the single most dangerous thing to get wrong: an
    /// encoder with no mask attends to future audio, which a duplex model never
    /// has, and produces plausible-looking output with no error anywhere.
    func testChunkedLimitedMaskIsCausalAndBounded() {
        let mask = VoiceChatEncoder.chunkedLimitedMask(length: 8, leftContext: 3, rightContext: 0)
        XCTAssertEqual(mask.shape, [8, 8])

        // true means masked. Row i may see columns (i-3)...i and nothing else.
        let values = mask.asArray(Bool.self)
        for query in 0 ..< 8 {
            for key in 0 ..< 8 {
                let masked = values[query * 8 + key]
                let visible = key <= query && query - key <= 3
                XCTAssertEqual(masked, !visible,
                               "query \(query) key \(key): expected visible=\(visible)")
            }
        }
    }

    func testMaskNeverLetsAnyPositionSeeTheFuture() {
        let mask = VoiceChatEncoder.chunkedLimitedMask(length: 16, leftContext: 70, rightContext: 0)
        let values = mask.asArray(Bool.self)
        for query in 0 ..< 16 {
            for key in (query + 1) ..< 16 {
                XCTAssertTrue(values[query * 16 + key],
                              "position \(query) can see future position \(key)")
            }
        }
    }

    /// Each causal subsampling stage is floor(n/2)+1, so the encoder emits one
    /// frame more than a plain stride-2 stack would. Nominal rate is 80 ms.
    func testSubsamplingFrameCount() {
        XCTAssertEqual(CausalSubsampling.outputFrames(melFrames: 200), 26)
        XCTAssertEqual(CausalSubsampling.outputFrames(melFrames: 1600), 201)
        // Monotonic, and never more than mel/8 + 3.
        for melFrames in stride(from: 8, through: 4000, by: 137) {
            let out = CausalSubsampling.outputFrames(melFrames: melFrames)
            XCTAssertGreaterThan(out, 0)
            XCTAssertLessThanOrEqual(out, melFrames / 8 + 3)
        }
    }

    func testRNNTTurnTakingForcesEndOfUtteranceAndBargeIn() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 3,
            beginOfUtteranceFrames: 2,
            minimumSpeechFrames: 2,
            firstTurnMinimumSpeechFrames: 2,
            noiseResetFrames: 3)
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)
        let pad = 12
        let bos = 1
        let eos = 2

        for _ in 0 ..< 2 {
            let result = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(result.action, .none)
        }
        XCTAssertTrue(state.speechConfirmed)

        for _ in 0 ..< 2 {
            let result = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(result.action, .none)
        }
        let eou = state.selectToken(
            proposedToken: pad, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(eou, .init(token: bos, action: .forcedAgentBegin))
        XCTAssertTrue(state.agentSpeaking)

        let firstBargeIn = state.selectToken(
            proposedToken: 100, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(firstBargeIn.action, .none)
        let bargeIn = state.selectToken(
            proposedToken: 101, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(bargeIn, .init(token: eos, action: .forcedAgentEnd))
        XCTAssertFalse(state.agentSpeaking)
    }

    func testRNNTBargeInCountsOnlySpeechAfterAgentBegins() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 3,
            beginOfUtteranceFrames: 3,
            minimumSpeechFrames: 2,
            firstTurnMinimumSpeechFrames: 2,
            noiseResetFrames: 3)
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)
        let pad = 12
        let bos = 1
        let eos = 2

        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
        }
        let begin = state.selectToken(
            proposedToken: bos, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(begin, .init(token: bos, action: .none))
        XCTAssertTrue(state.agentSpeaking)
        XCTAssertEqual(state.consecutiveSpeechFrames, 0)

        for token in [100, 101] {
            let result = state.selectToken(
                proposedToken: token, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(result.action, .none)
        }
        let freshBargeIn = state.selectToken(
            proposedToken: 102, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(
            freshBargeIn,
            .init(token: eos, action: .forcedAgentEnd))
    }

    func testRealtimeDefaultsRequireSustainedFreshBargeInSpeech() {
        XCTAssertEqual(
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .endOfUtteranceFrames,
            40)
        XCTAssertEqual(
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .beginOfUtteranceFrames,
            40)
        XCTAssertFalse(
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .allowInitialAgentTurn)
        XCTAssertTrue(
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .forceAgentBeginOnEndOfUtterance)
        XCTAssertTrue(
            VoiceChatTurnTakingParameters.functionCallingRealtime
                .forceAgentBeginOnEndOfUtterance)
        XCTAssertEqual(
            VoiceChatTurnTakingParameters.functionCallingRealtime
                .endOfUtteranceFrames,
            40)
        XCTAssertEqual(
            VoiceChatTurnTakingParameters.functionCallingRealtime
                .beginOfUtteranceFrames,
            40)
        XCTAssertEqual(
            VoiceChatTurnTakingParameters.functionCallingRealtime
                .functionCallEndOfUtteranceFrames,
            8)
    }

    func testFunctionCallingTurnTakingForcesReplyAfterLongerSilence() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)
        let pad = 12
        let bos = 1
        let eos = 2

        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
        }
        for _ in 0 ..< 39 {
            let result = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(result, .init(token: pad, action: .none))
        }

        let forcedBegin = state.selectToken(
            proposedToken: pad, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(
            forcedBegin,
            .init(token: bos, action: .forcedAgentBegin))
        XCTAssertTrue(state.agentSpeaking)
    }

    func testLexicalTokenArmsShortFollowUpTurn() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)
        let pad = 12
        let bos = 1
        let eos = 2

        // Complete one normal user and assistant turn so the stricter
        // post-first-turn frame threshold is active.
        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
        }
        for _ in 0 ..< 40 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
        }
        _ = state.selectToken(
            proposedToken: eos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)

        // One recognized word can be emitted entirely inside one RNN-T frame.
        state.observeRecognizedSpeechToken()
        _ = state.selectToken(
            proposedToken: pad, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertTrue(state.speechConfirmed)
        for _ in 0 ..< 39 {
            let waiting = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(waiting, .init(token: pad, action: .none))
        }
        let reply = state.selectToken(
            proposedToken: pad, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(reply, .init(token: bos, action: .forcedAgentBegin))
    }

    func testLexicalTokenClassificationRejectsUnknownAndPunctuation() {
        XCTAssertFalse(RNNTHead.isLexicalVocabularyToken("<unk>"))
        XCTAssertFalse(RNNTHead.isLexicalVocabularyToken("?"))
        XCTAssertFalse(RNNTHead.isLexicalVocabularyToken("\u{2581}"))
        XCTAssertTrue(RNNTHead.isLexicalVocabularyToken("\u{2581}who"))
        XCTAssertTrue(RNNTHead.isLexicalVocabularyToken("8"))
    }

    func testFunctionResponseResumesAuthorizedTurnWithoutAnotherSilenceWindow() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)
        let pad = 12
        let bos = 1
        let eos = 2

        state.observeRecognizedSpeechToken()
        state.beginFunctionCall()
        XCTAssertFalse(state.speechConfirmed)
        XCTAssertEqual(state.consecutiveBlankFrames, 0)

        state.resumeAfterFunctionResponse()

        XCTAssertTrue(state.speechConfirmed)
        XCTAssertEqual(
            state.consecutiveBlankFrames,
            VoiceChatTurnTakingParameters.nvidiaTurnTakingFallbackFrames - 1)
        let continuation = state.selectToken(
            proposedToken: pad,
            rnntIsBlank: true,
            padID: pad,
            bosID: bos,
            eosID: eos)
        XCTAssertEqual(
            continuation,
            .init(token: bos, action: .forcedAgentBegin))
        XCTAssertTrue(state.agentSpeaking)
        XCTAssertFalse(state.speechConfirmed)

        _ = state.selectToken(
            proposedToken: eos,
            rnntIsBlank: true,
            padID: pad,
            bosID: bos,
            eosID: eos)
        let laterUnpromptedTurn = state.selectToken(
            proposedToken: bos,
            rnntIsBlank: true,
            padID: pad,
            bosID: bos,
            eosID: eos)
        XCTAssertEqual(
            laterUnpromptedTurn,
            .init(token: pad, action: .suppressedUnpromptedBegin))
    }

    func testDeferredUserSpeechPreventsPrematurePostToolReply() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)
        let pad = 12
        let bos = 1
        let eos = 2

        state.observeRecognizedSpeechToken()
        state.beginFunctionCall()
        state.resumeAfterFunctionResponse(deferredUserInput: true)

        XCTAssertFalse(state.speechConfirmed)
        XCTAssertEqual(state.consecutiveBlankFrames, 0)
        let preRoll = state.selectToken(
            proposedToken: bos,
            rnntIsBlank: true,
            padID: pad,
            bosID: bos,
            eosID: eos)
        XCTAssertEqual(
            preRoll,
            .init(token: pad, action: .suppressedUnpromptedBegin))

        state.observeRecognizedSpeechToken()
        let accepted = state.selectToken(
            proposedToken: bos,
            rnntIsBlank: false,
            padID: pad,
            bosID: bos,
            eosID: eos)
        XCTAssertEqual(accepted, .init(token: bos, action: .none))
    }

    func testDeferredInputTracksSpeechWithoutOpeningAssistantTurn() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)

        state.observeDeferredInput(rnntIsBlank: false)
        state.observeDeferredInput(rnntIsBlank: false)
        state.observeDeferredInput(rnntIsBlank: true)

        XCTAssertTrue(state.speechConfirmed)
        XCTAssertFalse(state.agentSpeaking)
        XCTAssertEqual(state.consecutiveBlankFrames, 1)
    }

    func testFunctionResponseCannotOverwriteAnOpenNativeCall() {
        XCTAssertFalse(VoiceChatSession.functionResponseInjectionAvailable(
            collectingCall: true,
            injectingResponse: false,
            forcedTokenIndex: 0,
            forcedTokenCount: 0))
        XCTAssertFalse(VoiceChatSession.functionResponseInjectionAvailable(
            collectingCall: false,
            injectingResponse: true,
            forcedTokenIndex: 0,
            forcedTokenCount: 0))
        XCTAssertFalse(VoiceChatSession.functionResponseInjectionAvailable(
            collectingCall: false,
            injectingResponse: false,
            forcedTokenIndex: 2,
            forcedTokenCount: 3))
        XCTAssertTrue(VoiceChatSession.functionResponseInjectionAvailable(
            collectingCall: false,
            injectingResponse: false,
            forcedTokenIndex: 3,
            forcedTokenCount: 3))
    }

    func testFunctionResponsePrefillUsesPreviousTokenFeedback() {
        XCTAssertEqual(
            VoiceChatSession.functionResponseFeedbackTokens(
                previousFunction: 7,
                responseTokens: [11, 12, 22]),
            [7, 11, 12])
        XCTAssertEqual(
            VoiceChatSession.functionResponseFeedbackTokens(
                previousFunction: 7,
                responseTokens: []),
            [])
    }

    func testNativeFunctionStartCommitsOnlyAtUserEndOfUtterance() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 10)

        XCTAssertFalse(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 8,
            rnntIsBlank: true))
        XCTAssertTrue(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 9,
            rnntIsBlank: true))
        XCTAssertFalse(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 9,
            rnntIsBlank: false))
        XCTAssertTrue(VoiceChatSession.functionStartCommitReady(
            parameters: .modelNative,
            speechConfirmed: false,
            consecutiveBlankFrames: 0,
            rnntIsBlank: nil))
    }

    func testNativeBOSCommitsFunctionCandidateBeforeSafetyFallback() {
        let parameters = VoiceChatTurnTakingParameters.functionCallingRealtime

        XCTAssertFalse(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 6,
            rnntIsBlank: true,
            proposedTextToken: 12,
            bosID: 1))
        XCTAssertTrue(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 6,
            rnntIsBlank: true,
            proposedTextToken: 1,
            bosID: 1))
        XCTAssertFalse(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: false,
            consecutiveBlankFrames: 6,
            rnntIsBlank: true,
            proposedTextToken: 1,
            bosID: 1))
    }

    func testFunctionCandidateUsesItsNarrowNativeEndpoint() {
        let parameters = VoiceChatTurnTakingParameters.functionCallingRealtime

        XCTAssertFalse(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 6,
            rnntIsBlank: true))
        XCTAssertTrue(VoiceChatSession.functionStartCommitReady(
            parameters: parameters,
            speechConfirmed: true,
            consecutiveBlankFrames: 7,
            rnntIsBlank: true))
        XCTAssertEqual(parameters.endOfUtteranceFrames, 40)
    }

    func testFunctionEndpointRequiresCurrentOrDeferredModelEvidence() {
        XCTAssertFalse(VoiceChatSession.functionStartVerificationReady(
            commitReady: true,
            currentProbeProposesStart: false,
            hasDeferredCandidate: false))
        XCTAssertFalse(VoiceChatSession.functionStartVerificationReady(
            commitReady: false,
            currentProbeProposesStart: true,
            hasDeferredCandidate: true))
        XCTAssertTrue(VoiceChatSession.functionStartVerificationReady(
            commitReady: true,
            currentProbeProposesStart: true,
            hasDeferredCandidate: false))
        XCTAssertTrue(VoiceChatSession.functionStartVerificationReady(
            commitReady: true,
            currentProbeProposesStart: false,
            hasDeferredCandidate: true))
    }

    func testDeferredFunctionCandidateIsInvalidatedByNewCausalActivity() {
        XCTAssertFalse(VoiceChatSession.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: false,
            agentSpeaking: false,
            rnntIsBlank: true))
        XCTAssertFalse(VoiceChatSession.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: false,
            agentSpeaking: false,
            rnntIsBlank: nil))
        XCTAssertTrue(VoiceChatSession.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: false,
            agentSpeaking: false,
            rnntIsBlank: false))
        XCTAssertTrue(VoiceChatSession.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: true,
            agentSpeaking: false,
            rnntIsBlank: true))
        XCTAssertTrue(VoiceChatSession.shouldDiscardDeferredFunctionCandidate(
            injectingOutput: false,
            agentSpeaking: true,
            rnntIsBlank: true))
    }

    func testReferenceFallbackDoesNotSplitSubsecondPause() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .functionCallingRealtime)
        let pad = 12
        let bos = 1
        let eos = 2

        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
        }
        for _ in 0 ..< 10 {
            let pause = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(pause, .init(token: pad, action: .none))
        }

        let resumedSpeech = state.selectToken(
            proposedToken: pad, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(resumedSpeech, .init(token: pad, action: .none))
        XCTAssertFalse(state.agentSpeaking)
        XCTAssertEqual(state.consecutiveBlankFrames, 0)
    }

    func testFunctionHeadUsesCheapProbeAfterSpeechIsConfirmed() {
        let scheduler = VoiceChatFunctionHeadScheduler()

        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: false,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: false,
            assistantTurnActive: false), .none)
        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: false,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: true,
            assistantTurnActive: false), .startProbe)
        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: false,
            outputInjectionActive: false,
            turnTakingEnabled: false,
            userSpeechConfirmed: false,
            assistantTurnActive: false), .startProbe)
        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: false,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: true,
            assistantTurnActive: true), .none,
            "the function channel must not start during an assistant turn")
    }

    func testConfirmationGateBlocksFunctionStartsUntilAssistantBOS() {
        var gate = VoiceChatFunctionStartGate()

        XCTAssertTrue(gate.allowsFunctionStarts)
        gate.requireAssistantTurn(true)
        XCTAssertFalse(gate.allowsFunctionStarts)

        gate.observeTextToken(0, bosID: 1)
        gate.observeTextToken(42, bosID: 1)
        gate.observeTextToken(2, bosID: 1)
        XCTAssertFalse(
            gate.allowsFunctionStarts,
            "PAD, content, and EOS cannot silently release confirmation")

        gate.observeTextToken(1, bosID: 1)
        XCTAssertTrue(gate.allowsFunctionStarts)

        gate.requireAssistantTurn(true)
        gate.reset()
        XCTAssertTrue(gate.allowsFunctionStarts)
    }

    func testFunctionHeadUsesFullVocabularyOnlyInsideToolCall() {
        let scheduler = VoiceChatFunctionHeadScheduler()

        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: false,
            forceSilent: true,
            collectingCall: false,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: true,
            assistantTurnActive: false), .none,
            "prompt prefill must not project function logits")
        XCTAssertEqual(scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: false,
            outputInjectionActive: true,
            turnTakingEnabled: true,
            userSpeechConfirmed: true,
            assistantTurnActive: false), .none,
            "injected tool output must not project natural function logits")
        for _ in 0 ..< 5 {
            XCTAssertEqual(scheduler.evaluation(
                enabled: true,
                recording: true,
                forceSilent: false,
                collectingCall: true,
                outputInjectionActive: false,
                turnTakingEnabled: true,
                userSpeechConfirmed: false,
                assistantTurnActive: true), .fullVocabulary)
        }
        XCTAssertEqual(scheduler.evaluation(
            enabled: false,
            recording: true,
            forceSilent: false,
            collectingCall: true,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: true,
            assistantTurnActive: false), .none)
    }

    func testOpenFunctionChannelDoesNotRequireTextProjection() {
        let scheduler = VoiceChatFunctionHeadScheduler()
        let evaluation = scheduler.evaluation(
            enabled: true,
            recording: true,
            forceSilent: false,
            collectingCall: true,
            outputInjectionActive: false,
            turnTakingEnabled: true,
            userSpeechConfirmed: false,
            assistantTurnActive: false)

        XCTAssertEqual(evaluation, .fullVocabulary)
    }

    func testFunctionOutputStatesDoNotNeedATextDecision() {
        XCTAssertTrue(VoiceChatSession.textDecisionIsSuppressed(
            functionEvaluation: .fullVocabulary,
            outputInjectionActive: false))
        XCTAssertTrue(VoiceChatSession.textDecisionIsSuppressed(
            functionEvaluation: .none,
            outputInjectionActive: true))
        XCTAssertFalse(VoiceChatSession.textDecisionIsSuppressed(
            functionEvaluation: .startProbe,
            outputInjectionActive: false))
    }

    func testExternalToolWaitKeepsLanguageTimelineFrozen() {
        XCTAssertTrue(VoiceChatSession.usesPerceptionOnlyMicrophonePath(
            fastPathRunning: true,
            hasFastPathWork: true,
            awaitingFunctionResponse: false))
        XCTAssertTrue(VoiceChatSession.usesPerceptionOnlyMicrophonePath(
            fastPathRunning: false,
            hasFastPathWork: false,
            awaitingFunctionResponse: true))
        XCTAssertFalse(VoiceChatSession.usesPerceptionOnlyMicrophonePath(
            fastPathRunning: false,
            hasFastPathWork: false,
            awaitingFunctionResponse: false))
        XCTAssertFalse(VoiceChatSession.usesPerceptionOnlyMicrophonePath(
            fastPathRunning: true,
            hasFastPathWork: false,
            awaitingFunctionResponse: false))
    }

    func testDeferredReplayKeepsFreshAudioOffTheSharedTimeline() {
        XCTAssertTrue(VoiceChatSession.shouldDeferLiveMicrophoneFrame(
            fastPathRunning: false,
            hasFastPathWork: false,
            awaitingFunctionResponse: false,
            deferredTimelineWorkThisPush: true,
            hasDeferredInput: false))
        XCTAssertTrue(VoiceChatSession.shouldDeferLiveMicrophoneFrame(
            fastPathRunning: false,
            hasFastPathWork: false,
            awaitingFunctionResponse: false,
            deferredTimelineWorkThisPush: false,
            hasDeferredInput: true))
        XCTAssertFalse(VoiceChatSession.shouldDeferLiveMicrophoneFrame(
            fastPathRunning: false,
            hasFastPathWork: false,
            awaitingFunctionResponse: false,
            deferredTimelineWorkThisPush: false,
            hasDeferredInput: false))
        XCTAssertEqual(
            VoiceChatSession.maximumDeferredMicrophoneReplayFramesPerPush,
            1)
    }

    func testFunctionFastPathAssistTriggersOnlyAfterAStall() {
        let last: UInt64 = 1_000_000_000
        XCTAssertFalse(VoiceChatSession.functionFastPathNeedsProgressAssist(
            fastPathRunning: true,
            hasFastPathWork: true,
            lastProgressNanoseconds: last,
            nowNanoseconds: last + 159_000_000,
            maximumStallMilliseconds: 160))
        XCTAssertTrue(VoiceChatSession.functionFastPathNeedsProgressAssist(
            fastPathRunning: true,
            hasFastPathWork: true,
            lastProgressNanoseconds: last,
            nowNanoseconds: last + 160_000_000,
            maximumStallMilliseconds: 160))
        XCTAssertFalse(VoiceChatSession.functionFastPathNeedsProgressAssist(
            fastPathRunning: false,
            hasFastPathWork: true,
            lastProgressNanoseconds: last,
            nowNanoseconds: last + 1_000_000_000,
            maximumStallMilliseconds: 160))
        XCTAssertFalse(VoiceChatSession.functionFastPathNeedsProgressAssist(
            fastPathRunning: true,
            hasFastPathWork: false,
            lastProgressNanoseconds: last,
            nowNanoseconds: last + 1_000_000_000,
            maximumStallMilliseconds: 160))
    }

    func testFunctionCallDecodeMetricsReportsThroughput() {
        let metric = VoiceChatFunctionCallDecodeMetrics(
            active: false,
            completed: true,
            elapsedMilliseconds: 2_000,
            tokenSteps: 40,
            modelMilliseconds: 1_200,
            speechCacheMilliseconds: 300)

        XCTAssertEqual(metric.tokensPerSecond, 20, accuracy: 0.0001)
        XCTAssertEqual(
            metric.interleavingMilliseconds, 500, accuracy: 0.0001)
    }

    func testFunctionResponseMetricsReportsBatchedThroughput() {
        let metric = VoiceChatFunctionResponseMetrics(
            active: false,
            completed: true,
            elapsedMilliseconds: 400,
            tokenSteps: 48,
            prefillBatches: 3,
            languageCacheMilliseconds: 250,
            speechCacheMilliseconds: 100)

        XCTAssertEqual(metric.tokensPerSecond, 120, accuracy: 0.0001)
        XCTAssertEqual(metric.prefillBatches, 3)
        XCTAssertEqual(
            metric.interleavingMilliseconds, 50, accuracy: 0.0001)
    }

    func testFunctionRuntimeStatusKeepsOneCoherentUISnapshot() {
        let call = VoiceChatFunctionCallDecodeMetrics(
            active: true,
            completed: false,
            elapsedMilliseconds: 120,
            tokenSteps: 3)
        let status = VoiceChatFunctionRuntimeStatus(
            generatingCall: true,
            waitingForResponse: false,
            callDecode: call,
            responseSync: nil)

        XCTAssertTrue(status.generatingCall)
        XCTAssertFalse(status.waitingForResponse)
        XCTAssertEqual(status.callDecode, call)
        XCTAssertNil(status.responseSync)
    }

    func testRealtimeTurnTakingSuppressesUnpromptedInitialSpeech() {
        var state = VoiceChatRNNTTurnTakingState(parameters: .nvidiaRealtime)
        let result = state.selectToken(
            proposedToken: 1, rnntIsBlank: true,
            padID: 12, bosID: 1, eosID: 2)

        XCTAssertEqual(
            result,
            .init(token: 12, action: .suppressedUnpromptedBegin))
        XCTAssertFalse(state.agentSpeaking)
        XCTAssertTrue(state.firstAgentTurn)
    }

    func testSuppressedInitialBOSCannotBlockUserSpeechConfirmation() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 3,
            beginOfUtteranceFrames: 2,
            minimumSpeechFrames: 2,
            firstTurnMinimumSpeechFrames: 2,
            noiseResetFrames: 3)
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)
        let pad = 12
        let bos = 1
        let eos = 2

        let initial = state.selectToken(
            proposedToken: bos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(
            initial,
            .init(token: pad, action: .suppressedUnpromptedBegin))

        let firstSpeech = state.selectToken(
            proposedToken: bos, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(
            firstSpeech,
            .init(token: pad, action: .suppressedUnpromptedBegin))
        XCTAssertFalse(state.speechConfirmed)

        let confirmedSpeech = state.selectToken(
            proposedToken: bos, rnntIsBlank: false,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(confirmedSpeech, .init(token: bos, action: .none))
        XCTAssertTrue(state.agentSpeaking)
        XCTAssertFalse(state.speechConfirmed)
    }

    func testExplicitGreetingAllowsInitialAgentSpeech() {
        var parameters = VoiceChatTurnTakingParameters.nvidiaRealtime
        parameters.allowInitialAgentTurn = true
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)

        let result = state.selectToken(
            proposedToken: 1, rnntIsBlank: true,
            padID: 12, bosID: 1, eosID: 2)

        XCTAssertEqual(result, .init(token: 1, action: .none))
        XCTAssertTrue(state.agentSpeaking)
    }

    func testRNNTAgentTurnClosesOnlyAfterContentScaledPadTail() {
        var parameters = VoiceChatTurnTakingParameters.nvidiaRealtime
        parameters.allowInitialAgentTurn = true
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)
        let pad = 12
        let bos = 1
        let eos = 2

        _ = state.selectToken(
            proposedToken: bos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        for token in 100 ..< 106 {
            _ = state.selectToken(
                proposedToken: token, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
        }
        let budget = VoiceChatSpeechTurnState.acousticTailFrameBudget(
            contentFrames: 6)
        XCTAssertEqual(budget, 18)

        for _ in 0 ..< budget {
            let tail = state.selectToken(
                proposedToken: pad, rnntIsBlank: true,
                padID: pad, bosID: bos, eosID: eos)
            XCTAssertEqual(tail, .init(token: pad, action: .none))
            XCTAssertTrue(state.agentSpeaking)
        }
        let completed = state.selectToken(
            proposedToken: pad, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)

        XCTAssertEqual(completed, .init(token: eos, action: .none))
        XCTAssertFalse(state.agentSpeaking)
    }

    func testInputResynchronizationClearsPartialSpeechButPreservesAgentTurn() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 3,
            beginOfUtteranceFrames: 3,
            minimumSpeechFrames: 2,
            firstTurnMinimumSpeechFrames: 2,
            noiseResetFrames: 3,
            allowInitialAgentTurn: true)
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)

        _ = state.selectToken(
            proposedToken: 1, rnntIsBlank: true,
            padID: 12, bosID: 1, eosID: 2)
        _ = state.selectToken(
            proposedToken: 100, rnntIsBlank: false,
            padID: 12, bosID: 1, eosID: 2)
        XCTAssertTrue(state.agentSpeaking)
        XCTAssertEqual(state.consecutiveSpeechFrames, 1)

        state.resynchronizeInput()

        XCTAssertTrue(state.agentSpeaking)
        XCTAssertFalse(state.speechConfirmed)
        XCTAssertEqual(state.consecutiveSpeechFrames, 0)
        XCTAssertEqual(state.consecutiveBlankFrames, 0)
        XCTAssertEqual(state.totalSpeechFrames, 0)
    }

    func testInputResynchronizationCanPreserveConfirmedUserTurn() {
        var state = VoiceChatRNNTTurnTakingState(
            parameters: .nvidiaRealtime)
        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: 12,
                rnntIsBlank: false,
                padID: 12,
                bosID: 1,
                eosID: 2)
        }
        XCTAssertTrue(state.speechConfirmed)

        state.resynchronizeInput(preserveSpeechConfirmation: true)

        XCTAssertTrue(state.speechConfirmed)
        XCTAssertEqual(state.consecutiveSpeechFrames, 0)
        XCTAssertEqual(state.consecutiveBlankFrames, 0)
        XCTAssertEqual(state.totalSpeechFrames, 0)
    }

    func testFunctionCallInterruptRequiresFreshSustainedSpeech() {
        var parameters = VoiceChatTurnTakingParameters.functionCallingRealtime
        parameters.beginOfUtteranceFrames = 3
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)

        // Old request activity is consumed when the function-channel turn
        // starts. Silence and short non-blank bursts must not abort the call.
        state.beginFunctionCall()
        XCTAssertFalse(state.observeFunctionCallInput(rnntIsBlank: true))
        XCTAssertFalse(state.observeFunctionCallInput(rnntIsBlank: false))
        XCTAssertFalse(state.observeFunctionCallInput(rnntIsBlank: true))
        XCTAssertFalse(state.observeFunctionCallInput(rnntIsBlank: false))
        XCTAssertFalse(state.observeFunctionCallInput(rnntIsBlank: false))
        XCTAssertTrue(state.observeFunctionCallInput(rnntIsBlank: false))
    }

    func testRNNTTurnTakingSuppressesSilenceSelfPlayUntilNewSpeech() {
        let parameters = VoiceChatTurnTakingParameters(
            endOfUtteranceFrames: 2,
            beginOfUtteranceFrames: 2,
            minimumSpeechFrames: 2,
            firstTurnMinimumSpeechFrames: 2,
            noiseResetFrames: 2,
            allowInitialAgentTurn: true)
        var state = VoiceChatRNNTTurnTakingState(parameters: parameters)
        let pad = 12
        let bos = 1
        let eos = 2

        // Let the model complete one native agent turn.
        _ = state.selectToken(
            proposedToken: bos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        _ = state.selectToken(
            proposedToken: eos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)

        let silenceBOS = state.selectToken(
            proposedToken: bos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(
            silenceBOS,
            .init(token: pad, action: .suppressedUnpromptedBegin))

        for _ in 0 ..< 2 {
            _ = state.selectToken(
                proposedToken: pad, rnntIsBlank: false,
                padID: pad, bosID: bos, eosID: eos)
        }
        XCTAssertTrue(state.speechConfirmed)
        let userPromptedBOS = state.selectToken(
            proposedToken: bos, rnntIsBlank: true,
            padID: pad, bosID: bos, eosID: eos)
        XCTAssertEqual(userPromptedBOS, .init(token: bos, action: .none))
    }

    func testDisabledRNNTTurnTakingDoesNotOverrideTokens() {
        var state = VoiceChatRNNTTurnTakingState(parameters: .modelNative)
        for isBlank in [false, false, true, true, true] {
            let result = state.selectToken(
                proposedToken: 77, rnntIsBlank: isBlank,
                padID: 12, bosID: 1, eosID: 2)
            XCTAssertEqual(result, .init(token: 77, action: .none))
        }
    }

    /// Stateful inference must be the same causal computation as the batch
    /// stack, not merely close enough to produce plausible speech.
    func testStreamingConformerMatchesBatchAndBoundsCaches() {
        let config = VoiceChatEncoderConfig(
            dModel: 16,
            nLayers: 2,
            nHeads: 4,
            featIn: 8,
            ffExpansionFactor: 2,
            convKernelSize: 3,
            subsamplingFactor: 8,
            subsamplingConvChannels: 4,
            preEncodeFreqOut: 1,
            causalConvIndices: [0, 2, 5],
            convNormType: "layer_norm",
            selfAttentionModel: "rel_pos",
            attContextSize: [4, 0],
            attContextStyle: "chunked_limited",
            posEmbMaxLen: 64,
            useBias: false,
            xscaling: false)
        let encoder = VoiceChatEncoder(config)
        MLXRandom.seed(7)
        let input = MLXRandom.normal([1, 12, config.dModel]).asType(.float32)

        let batch = encoder.conformerStack(input)
        let state = encoder.newStreamState()
        var frames: [MLXArray] = []
        for index in 0 ..< input.dim(1) {
            let frame = encoder.stream(
                input[0..., index..<(index + 1), 0...], state: state)
            MLX.eval([frame] + state.evaluatedArrays)
            frames.append(frame)
        }
        let streamed = MLX.concatenated(frames, axis: 1)
        eval(batch, streamed)

        let deviation = mean(abs(batch - streamed)).item(Float.self)
        let scale = mean(abs(batch)).item(Float.self)
        // MLX selects shape-dependent kernels for a 12-frame matrix and a
        // sequence of one-frame matrices. They are not bit-identical, but the
        // normalized drift must remain below one tenth of one percent.
        XCTAssertLessThan(deviation / max(scale, 1e-7), 1e-3)
        for cache in state.layers {
            XCTAssertLessThanOrEqual(cache.key?.dim(2) ?? 0, config.leftContext + 1)
            XCTAssertLessThanOrEqual(cache.value?.dim(2) ?? 0, config.leftContext + 1)
            XCTAssertEqual(cache.convolution?.dim(1), config.convKernelSize - 1)
        }
    }

    // MARK: - Weights required

    func testPerceptionEncodesIntoLanguageModelSpace() throws {
        let root = try requireBundle()
        let perception = try VoiceChatPerception.load(from: root.appendingPathComponent("encoder"))
        let config = perception.config

        // Three load-bearing config values; a stock Conformer gets all three
        // wrong and still loads.
        XCTAssertFalse(config.encoder.useBias)
        XCTAssertEqual(config.encoder.preEncodeFreqOut, 17)
        XCTAssertEqual(config.encoder.convNormType, "layer_norm")
        XCTAssertEqual(config.encoder.attContextSize, [70, 0])

        let melFrames = 200
        let mel = MLXArray.zeros([1, melFrames, config.encoder.featIn]) + 0.1
        let embeddings = perception(mel)
        eval(embeddings)

        XCTAssertEqual(embeddings.shape,
                       [1, CausalSubsampling.outputFrames(melFrames: melFrames),
                        config.modalityProj.outFeatures])
        XCTAssertTrue(all(isFinite(embeddings)).item(Bool.self), "encoder produced non-finite output")
    }

    func testLanguageModelProducesFiniteLogits() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))

        XCTAssertEqual(llm.configuration.numHiddenLayers, 56)
        XCTAssertEqual(llm.configuration.hiddenSize, 4480)
        let pattern = Array(llm.configuration.hybridOverridePattern)
        XCTAssertEqual(pattern.filter { $0 == "M" }.count, 27)
        XCTAssertEqual(pattern.filter { $0 == "-" }.count, 25)
        XCTAssertEqual(pattern.filter { $0 == "*" }.count, 4)

        let tokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let logits = llm(tokens)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 4, llm.configuration.vocabSize])
        XCTAssertTrue(all(isFinite(logits)).item(Bool.self), "language model produced non-finite logits")
    }

    /// Regression: `NemotronHBackbone` builds its causal mask from the cache and
    /// falls back to no mask at all when there isn't one, which silently makes
    /// attention bidirectional. `VoiceChatLanguageModel` therefore always
    /// supplies a cache. If that is ever removed, later positions change while
    /// early ones do not — so compare a prefix against the full sequence.
    func testAttentionIsCausalAcrossSequenceLength() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))

        let full = MLXArray([5, 9, 13, 21, 34]).reshaped(1, 5)
        let prefix = full[0..., ..<3]

        let logitsFull = llm(full)
        let logitsPrefix = llm(prefix)
        eval(logitsFull, logitsPrefix)

        // Under causal attention the first three positions cannot depend on
        // tokens 4 and 5, so their logits must be unchanged.
        let a = logitsFull[0..., ..<3, 0...].asType(.float32)
        let b = logitsPrefix.asType(.float32)
        let deviation = mean(abs(a - b)).item(Float.self)
        let scale = mean(abs(a)).item(Float.self)
        XCTAssertLessThan(deviation / scale, 0.01,
                          "prefix logits changed when later tokens were added — attention is not causal")
    }

    func testFunctionHeadIsCarriedButNotPartOfTheModel() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))
        // The tool-call channel is a separate head the stock model has no slot
        // for; it must survive the load rather than be dropped.
        let head = try XCTUnwrap(llm.functionHead, "function_head missing from the bundle")
        XCTAssertEqual(head.weight.shape.first, llm.configuration.vocabSize)

        // In a quantized bundle the head is stored packed, so the second
        // dimension is hiddenSize scaled by bits/32 rather than hiddenSize.
        // The int5 build deliberately holds this head at 8 bits, so assert the
        // packing is consistent with some supported width rather than
        // hard-coding one — that catches a corrupt head without pinning the
        // test to a single variant.
        let hidden = llm.configuration.hiddenSize
        let packed = try XCTUnwrap(head.weight.shape.last)
        if packed == hidden {
            return  // dense (fp16 bundle)
        }
        let bits = packed * 32 / hidden
        XCTAssertTrue([2, 3, 4, 5, 6, 8].contains(bits),
                      "function_head packing implies \(bits) bits, which MLX does not support")
        XCTAssertEqual(packed, hidden * bits / 32, "function_head packing is inconsistent")
    }

    /// Only 4 of 56 layers keep a growing KV cache; the 27 Mamba layers hold a
    /// fixed-size recurrent state instead. That split is why long conversations
    /// stay affordable, so assert the cache is built that way.
    func testCacheIsMostlyRecurrent() throws {
        let root = try requireBundle()
        let llm = try VoiceChatLanguageModel.load(from: root.appendingPathComponent("llm"))
        let cache = llm.newCache()
        XCTAssertEqual(cache.count, 31, "expected one cache per Mamba and attention layer")
        XCTAssertEqual(cache.filter { $0 is MambaCache }.count, 27)
    }

    // MARK: - End to end: audio in, transcript out

    /// The only test that exercises the whole speech-understanding path at once.
    ///
    /// Shape and finiteness checks pass happily on a model with a transposed
    /// filterbank, a mis-centred window or an off-by-one blank index. A
    /// transcript is the first thing that does not.
    func testTranscribesRealSpeech() throws {
        let root = try requireBundle()
        let encoderDir = root.appendingPathComponent("encoder")
        let perception = try VoiceChatPerception.load(from: encoderDir)
        let weights = try MLX.loadArrays(url: encoderDir.appendingPathComponent("model.safetensors"))

        let audioURL = try XCTUnwrap(Bundle.module.url(forResource: "fleurs_en", withExtension: "wav"))
        let samples = try loadMono16k(audioURL)
        XCTAssertGreaterThan(samples.count, VoiceChatTranscriber.sampleRate, "need at least a second of audio")

        let transcriber = try VoiceChatTranscriber(perception: perception, weights: weights)
        let transcript = transcriber.transcribe(samples).lowercased()
        print("transcript: \(transcript)")

        XCTAssertFalse(transcript.isEmpty, "produced no transcript at all")
        // Content words the Python reference produces on this clip across every
        // quantization variant. Asserting on words rather than an exact string
        // keeps the test meaningful without pinning it to one variant's output.
        for expected in ["also", "paid", "tribute", "luna"] {
            XCTAssertTrue(transcript.contains(expected),
                          "transcript is missing \(expected): \(transcript)")
        }
    }

    /// Read a wav as mono float at 16 kHz. Several fixtures are IEEE-float wavs.
    private func loadMono16k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: Double(VoiceChatTranscriber.sampleRate),
                                   channels: 1, interleaved: false)!
        let converter = try XCTUnwrap(AVAudioConverter(from: file.processingFormat, to: format))
        let source = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                      frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: source)

        let ratio = format.sampleRate / file.processingFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(source.frameLength) * ratio) + 1024
        let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity)!
        var supplied = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if supplied { status.pointee = .endOfStream; return nil }
            supplied = true
            status.pointee = .haveData
            return source
        }
        if let error { throw error }
        let pointer = try XCTUnwrap(output.floatChannelData?[0])
        return Array(UnsafeBufferPointer(start: pointer, count: Int(output.frameLength)))
    }
}
