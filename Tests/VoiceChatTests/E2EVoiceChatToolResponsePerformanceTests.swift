import Foundation
import XCTest

@testable import VoiceChat

/// Isolated latency coverage for the model-native phase-two tool result. This
/// intentionally avoids a phase-one model-generated call so timings measure
/// the optimization itself rather than function-head selection variability.
final class E2EVoiceChatToolResponsePerformanceTests: XCTestCase {
    func testLongContextToolSuccessStaysRealtime() async throws {
        guard ProcessInfo.processInfo.environment["VOICECHAT_PERFORMANCE_TEST"]
            == "1" else {
            throw XCTSkip("set VOICECHAT_PERFORMANCE_TEST=1")
        }
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE")
        }
        let contextFrames = 600
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let session = try await model.startSession(
            systemPrompt: VoiceChatSession.baseSystemPrompt,
            speech: .init(
                iterations: 4,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)
        let silence = [Float](
            repeating: 0,
            count: VoiceChatSession.inputSamplesPerFrame)
        for _ in 0 ..< contextFrames {
            _ = try await session.pushAudio(silence)
        }

        let started = DispatchTime.now().uptimeNanoseconds
        try await session.injectFunctionResponse(
            #"{"ok":true,"tool":"create_reminder"}"#)

        var nextDeadline = started
        var microphoneLatencies: [Double] = []
        var preSpeechMicrophoneLatencies: [Double] = []
        var firstSpeechMilliseconds: Double?
        var completed = false
        for _ in 0 ..< 160 {
            nextDeadline += UInt64(VoiceChatSession.frameMilliseconds)
                * 1_000_000
            let now = DispatchTime.now().uptimeNanoseconds
            if nextDeadline > now {
                try await Task.sleep(nanoseconds: nextDeadline - now)
            }
            let pushStarted = DispatchTime.now().uptimeNanoseconds
            let events = try await session.pushAudio(silence)
            let pushEnded = DispatchTime.now().uptimeNanoseconds
            microphoneLatencies.append(
                Double(pushEnded - pushStarted) / 1_000_000)
            let wasWaitingForSpeech = firstSpeechMilliseconds == nil
            if wasWaitingForSpeech {
                preSpeechMicrophoneLatencies.append(
                    Double(pushEnded - pushStarted) / 1_000_000)
            }
            if wasWaitingForSpeech, events.contains(where: \.speaking) {
                firstSpeechMilliseconds = Double(pushEnded - started)
                    / 1_000_000
            }
            if firstSpeechMilliseconds != nil,
               !(await session.hasPendingFunctionOutput())
            {
                completed = true
                break
            }
        }
        let ended = DispatchTime.now().uptimeNanoseconds
        let sorted = microphoneLatencies.sorted()
        let p95 = sorted[min(sorted.count - 1, sorted.count * 95 / 100)]
        let maximum = sorted.last ?? 0
        let preSpeechSorted = preSpeechMicrophoneLatencies.sorted()
        let preSpeechP95 = preSpeechSorted.isEmpty ? 0 : preSpeechSorted[
            min(preSpeechSorted.count - 1, preSpeechSorted.count * 95 / 100)]
        let preSpeechMaximum = preSpeechSorted.last ?? 0
        let statistics = await session.functionHeadEvaluationStatistics()
        let optionalResponseMetrics = await session.functionResponseMetrics()
        let responseMetrics = try XCTUnwrap(optionalResponseMetrics)
        print(String(format:
            "Long-context tool result: first speech %.0f ms, total %.0f ms, tool-phase mic p95 %.1f ms max %.1f ms, all mic p95 %.1f ms max %.1f ms, slow frames %d, response sync %.0f ms (language %.0f ms, voice cache %.0f ms, interleave %.0f ms), response steps %d in %d prefills",
            firstSpeechMilliseconds ?? -1,
            Double(ended - started) / 1_000_000,
            preSpeechP95, preSpeechMaximum, p95, maximum,
            microphoneLatencies.filter { $0 > 80 }.count,
            responseMetrics.elapsedMilliseconds,
            responseMetrics.languageCacheMilliseconds,
            responseMetrics.speechCacheMilliseconds,
            responseMetrics.interleavingMilliseconds,
            statistics.asynchronousResponseSteps,
            statistics.asynchronousResponsePrefillBatches))

        XCTAssertTrue(completed)
        XCTAssertLessThan(firstSpeechMilliseconds ?? .infinity, 1_500)
        XCTAssertLessThan(preSpeechMaximum, 600)
        XCTAssertGreaterThan(statistics.asynchronousResponsePrefillBatches, 0)
        XCTAssertLessThan(
            statistics.asynchronousResponsePrefillBatches,
            statistics.asynchronousResponseSteps)
        XCTAssertEqual(statistics.asynchronousResponseTimeouts, 0)
        XCTAssertFalse(responseMetrics.active)
        XCTAssertTrue(responseMetrics.completed)
        XCTAssertEqual(
            responseMetrics.tokenSteps,
            statistics.asynchronousResponseSteps)
        XCTAssertEqual(
            responseMetrics.prefillBatches,
            statistics.asynchronousResponsePrefillBatches)
        XCTAssertGreaterThan(responseMetrics.elapsedMilliseconds, 0)
    }
}
