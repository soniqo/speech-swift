import AVFoundation
import Foundation
import XCTest

@testable import VoiceChat

/// Real-checkpoint coverage for microphone information retained while the
/// native function channel owns the shared language timeline.
final class E2EVoiceChatDeferredInputTests: XCTestCase {
    /// A long provider wait may compact silence, but speech that begins after
    /// that silence must still reach RNN-T immediately and the language model
    /// causally after the tool result. Replaying it must never duplicate audio.
    func testLongToolWaitCompactsSilenceAndPreservesLaterSpeech() async throws {
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
                iterations: 2,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .functionCallingRealtime,
            functionCallingEnabled: true)

        let requestURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "voicechat-long-tool-wait-\(UUID().uuidString).aiff")
        let followUpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "voicechat-long-tool-wait-followup-\(UUID().uuidString).aiff")
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
        let call = try XCTUnwrap(nativeCall)
        XCTAssertTrue(call.contains("list_reminders"), call)
        let waitingForResponse = await session.isWaitingForFunctionResponse()
        XCTAssertTrue(waitingForResponse)

        // Nineteen seconds of model-time silence should remain only as the
        // bounded two-frame pre-roll, not as 240 old language positions.
        for _ in 0 ..< 240 {
            _ = try await session.pushAudio(silence)
        }
        let silenceStatistics = await session
            .deferredMicrophoneStatisticsForTesting()
        XCTAssertEqual(silenceStatistics.bufferedFrames, 0)
        XCTAssertEqual(silenceStatistics.droppedFrames, 0)

        for start in stride(from: 0, to: followUp.count, by: frameSize) {
            _ = try await session.pushAudio(Array(
                followUp[start ..< min(followUp.count, start + frameSize)]))
        }
        // Let the RNN-T activity region close. Only the bounded acoustic tail
        // is retained; later silence goes back to the two-frame pre-roll.
        for _ in 0 ..< 24 {
            _ = try await session.pushAudio(silence)
        }
        let waitingStatistics = await session
            .deferredMicrophoneStatisticsForTesting()
        let followUpFrames = (followUp.count + frameSize - 1) / frameSize
        XCTAssertGreaterThan(waitingStatistics.bufferedFrames, 0)
        XCTAssertLessThanOrEqual(
            waitingStatistics.bufferedFrames,
            followUpFrames + VoiceChatSession.deferredMicrophonePreRollFrames
                + VoiceChatSession.deferredMicrophoneTrailingBlankFrames)
        XCTAssertEqual(waitingStatistics.replayedFrames, 0)
        XCTAssertEqual(waitingStatistics.droppedFrames, 0)

        try await session.injectFunctionResponse(
            #"{"ok":true,"tool":"list_reminders","result":[{"id":"r1","name":"Morning"}]}"#)
        var mutedReplayFrames = 0
        var nextDeadline = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< 260 {
            let before = await session
                .deferredMicrophoneStatisticsForTesting()
            let events = try await session.pushAudio(silence)
            let after = await session
                .deferredMicrophoneStatisticsForTesting()
            let replayed = after.replayedFrames - before.replayedFrames
            XCTAssertLessThanOrEqual(replayed, 1)
            if replayed == 1 {
                let muted = events.filter { !$0.playbackRequired }
                XCTAssertEqual(muted.count, 1)
                XCTAssertEqual(muted[0].textToken, model.tokenizer.padID)
                XCTAssertFalse(muted[0].speaking)
                XCTAssertTrue(muted[0].audio.allSatisfy { abs($0) <= 1e-8 })
                mutedReplayFrames += 1
            }
            let pending = await session.hasPendingFunctionOutput()
            if after.bufferedFrames == 0,
               after.pendingSpeechCacheFrames == 0,
               !pending {
                break
            }
            nextDeadline += UInt64(VoiceChatSession.frameMilliseconds)
                * 1_000_000
            let now = DispatchTime.now().uptimeNanoseconds
            if nextDeadline > now {
                try await Task.sleep(nanoseconds: nextDeadline - now)
            }
        }

        let finalStatistics = await session
            .deferredMicrophoneStatisticsForTesting()
        XCTAssertEqual(finalStatistics.bufferedFrames, 0)
        XCTAssertEqual(finalStatistics.pendingSpeechCacheFrames, 0)
        XCTAssertEqual(finalStatistics.droppedFrames, 0)
        XCTAssertEqual(
            finalStatistics.replayedFrames,
            waitingStatistics.bufferedFrames)
        XCTAssertEqual(mutedReplayFrames, finalStatistics.replayedFrames)
        let transcript = await session.userTranscript().lowercased()
        XCTAssertTrue(transcript.contains("who"), transcript)
    }

    private func synthesize(_ text: String, to url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = [
            "-v", "Samantha", "-r", "170", "-o", url.path, text,
        ]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "VoiceChatDeferredInputTests",
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
}

/// The bounded queue is an intentional last-resort loss boundary. Exercise it
/// with the real perception model and prove overflow is counted rather than
/// silently overwriting retained embeddings.
final class E2EVoiceChatDeferredOverflowTests: XCTestCase {
    func testDeferredQueueOverflowIsExplicitlyAccounted() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete VoiceChat bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let session = try await model.startSession(
            systemPrompt: VoiceChatSession.baseSystemPrompt,
            speech: .init(iterations: 1, realtimeIdleOptimization: true),
            streamUserTranscript: false,
            turnTaking: .modelNative,
            functionCallingEnabled: true)
        await session.beginDeferredMicrophoneCaptureForTesting()

        let overflowFrames = 7
        let silence = [Float](
            repeating: 0,
            count: VoiceChatSession.inputSamplesPerFrame)
        for _ in 0 ..< VoiceChatSession.maximumDeferredMicrophoneFrames
            + overflowFrames
        {
            _ = try await session.pushAudio(silence)
        }

        let statistics = await session
            .deferredMicrophoneStatisticsForTesting()
        XCTAssertEqual(
            statistics.bufferedFrames,
            VoiceChatSession.maximumDeferredMicrophoneFrames)
        XCTAssertEqual(statistics.replayedFrames, 0)
        XCTAssertEqual(statistics.droppedFrames, overflowFrames)
        XCTAssertEqual(statistics.pendingSpeechCacheFrames, 0)
        let pending = await session.hasPendingFunctionOutput()
        XCTAssertTrue(pending)
    }
}
