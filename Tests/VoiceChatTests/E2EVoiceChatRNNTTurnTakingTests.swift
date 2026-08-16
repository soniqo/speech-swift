import AVFoundation
import Foundation
import XCTest

@testable import VoiceChat

/// Real-checkpoint coverage for the RNN-T blank stream and the turn policy.
/// The isolated E2E runner launches this class in its own process because the
/// complete VoiceChat bundle occupies multiple gigabytes.
final class E2EVoiceChatRNNTTurnTakingTests: XCTestCase {
    /// Regression coverage for two live failures reported by the CLI demo:
    /// silence must not make the assistant speak first, and a short natural
    /// pause inside an unfinished sentence must not split it into two turns.
    func testInitialSilenceAndSubsecondPauseStayInsideTheUserTurn() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let firstURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-paused-first-\(UUID().uuidString).aiff")
        let secondURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-paused-second-\(UUID().uuidString).aiff")
        defer {
            try? FileManager.default.removeItem(at: firstURL)
            try? FileManager.default.removeItem(at: secondURL)
        }
        try synthesize("I have a question about a city whose name is", to: firstURL)
        try synthesize("Paris. What country is it in?", to: secondURL)
        let first = trimTrailingSilence(try loadMono16k(firstURL))
        let second = trimTrailingSilence(try loadMono16k(secondURL))
        let session = try await model.startSession(
            speech: .init(
                iterations: 4,
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .nvidiaRealtime)

        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let silenceFrame = [Float](repeating: 0, count: frameSize)
        var prematureEvents: [VoiceChatFrameEvent] = []

        // One second of room silence previously allowed an unsolicited first
        // turn. The neutral session must remain idle until RNN-T sees speech.
        for _ in 0 ..< 13 {
            prematureEvents += try await session.pushAudio(silenceFrame)
        }
        XCTAssertFalse(prematureEvents.contains {
            $0.textToken == model.tokenizer.bosID || $0.speaking
        })

        for start in stride(from: 0, to: first.count, by: frameSize) {
            prematureEvents += try await session.pushAudio(Array(
                first[start ..< min(first.count, start + frameSize)]))
        }
        // 640 ms is a noticeable conversational pause but remains far below
        // NVIDIA's 3.2-second safety endpoint.
        for _ in 0 ..< 8 {
            prematureEvents += try await session.pushAudio(silenceFrame)
        }
        XCTAssertFalse(
            prematureEvents.contains {
                $0.textToken == model.tokenizer.bosID || $0.speaking
            },
            "the assistant started before the unfinished sentence resumed")

        let completionStart = (await session.events()).count
        for start in stride(from: 0, to: second.count, by: frameSize) {
            _ = try await session.pushAudio(Array(
                second[start ..< min(second.count, start + frameSize)]))
        }
        _ = try await session.pushSilence(seconds: 5)

        let events = await session.events()
        let completedTurn = events.dropFirst(completionStart)
        XCTAssertTrue(completedTurn.contains {
            $0.textToken == model.tokenizer.bosID
        })
        XCTAssertTrue(completedTurn.contains(where: \.speaking))
        XCTAssertTrue(completedTurn.contains {
            $0.audio.contains(where: { abs($0) > 1e-6 })
        })
        let transcript = await session.userTranscript().lowercased()
        XCTAssertTrue(transcript.contains("city"), transcript)
        XCTAssertTrue(transcript.contains("paris"), transcript)
        let reply = await session.reply()
        XCTAssertFalse(
            reply.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    func testNaturalRNNTEndOfUtteranceStartsResponse() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "fleurs_en", withExtension: "wav"))
        let samples = try loadMono16k(audioURL)
        let session = try await model.startSession(
            streamUserTranscript: true,
            turnTaking: .nvidiaRealtime)

        let frameSize = VoiceChatSession.inputSamplesPerFrame
        for start in stride(from: 0, to: samples.count, by: frameSize) {
            _ = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)]))
        }
        // No forceTurn: model-native turn-taking may start earlier, while the
        // RNN-T policy supplies the deterministic 3.2-second EOU fallback. Five
        // seconds covers that endpoint and leaves generated frames for speech.
        _ = try await session.pushSilence(seconds: 5)

        let summary = await session.summary()
        let transcript = await session.userTranscript().lowercased()
        let reply = await session.reply()
        let events = await session.events()
        let inputFrames = Int(Foundation.ceil(
            Double(samples.count) / Double(frameSize)))

        for expected in ["also", "paid", "tribute", "luna"] {
            XCTAssertTrue(
                transcript.contains(expected),
                "streaming transcript is missing \(expected): \(transcript)")
        }
        let nonblankSpeechFrames = events.prefix(inputFrames).filter {
            $0.rnntIsBlank == false
        }.count
        XCTAssertGreaterThanOrEqual(
            nonblankSpeechFrames,
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .firstTurnMinimumSpeechFrames)

        var currentBlankRun = 0
        var longestBlankRun = 0
        for event in events.dropFirst(inputFrames) {
            if event.rnntIsBlank == true {
                currentBlankRun += 1
                longestBlankRun = max(longestBlankRun, currentBlankRun)
            } else {
                currentBlankRun = 0
            }
        }
        XCTAssertGreaterThanOrEqual(
            longestBlankRun,
            VoiceChatTurnTakingParameters.nvidiaRealtime
                .endOfUtteranceFrames,
            "silent tail did not produce the RNN-T blank run required for EOU")
        XCTAssertFalse(reply.isEmpty, "RNN-T EOU produced no agent response")
        XCTAssertFalse(
            events.contains { $0.turnTakingAction == .forcedAgentEnd },
            "silent input must not force a barge-in during agent speech")
        let firstSpeechFrame = try XCTUnwrap(summary.firstSpeechFrame)
        print(
            "RNN-T turn-taking: input \(inputFrames) frames, first speech "
                + "\(firstSpeechFrame), EOF-relative "
                + "\((firstSpeechFrame - inputFrames) * VoiceChatSession.frameMilliseconds) ms")
        XCTAssertLessThanOrEqual(
            firstSpeechFrame,
            inputFrames
                + VoiceChatTurnTakingParameters.nvidiaRealtime
                    .endOfUtteranceFrames,
            "agent did not begin by the reference RNN-T safety endpoint")

        // Drain the first reply, then verify that a one-word follow-up can arm
        // a new turn even when its RNN-T labels fit inside very few frames.
        var latestEvents = await session.events()
        var drainFrames = 0
        while agentTurnIsOpen(
            latestEvents,
            bosID: model.tokenizer.bosID,
            eosID: model.tokenizer.eosID),
            drainFrames < 200
        {
            _ = try await session.pushAudio(
                [Float](repeating: 0, count: frameSize))
            drainFrames += 1
            latestEvents = await session.events()
        }
        XCTAssertFalse(agentTurnIsOpen(
            latestEvents,
            bosID: model.tokenizer.bosID,
            eosID: model.tokenizer.eosID))

        let followUpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("voicechat-short-followup-\(UUID().uuidString).aiff")
        defer { try? FileManager.default.removeItem(at: followUpURL) }
        try synthesize("Yes.", to: followUpURL)
        let followUp = try loadMono16k(followUpURL)
        let followUpStart = latestEvents.count
        for start in stride(from: 0, to: followUp.count, by: frameSize) {
            _ = try await session.pushAudio(Array(
                followUp[start ..< min(followUp.count, start + frameSize)]))
        }
        _ = try await session.pushSilence(seconds: 5)
        latestEvents = await session.events()
        let followUpEvents = latestEvents.dropFirst(followUpStart)
        XCTAssertTrue(
            followUpEvents.contains {
                $0.textToken == model.tokenizer.bosID
            },
            "recognized one-word follow-up did not start a new assistant turn")
        let completeTranscript = await session.userTranscript().lowercased()
        XCTAssertTrue(
            completeTranscript.contains("yes"),
            "RNN-T did not retain the one-word follow-up")
    }

    private func agentTurnIsOpen(
        _ events: [VoiceChatFrameEvent],
        bosID: Int,
        eosID: Int
    ) -> Bool {
        events.reversed().first {
            $0.textToken == bosID || $0.textToken == eosID
        }?.textToken == bosID
    }

    private func synthesize(_ text: String, to url: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-v", "Samantha", "-r", "170", "-o", url.path, text]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "VoiceChatRNNTTurnTakingTests",
                code: Int(process.terminationStatus),
                userInfo: [NSLocalizedDescriptionKey: "say failed"])
        }
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
}
