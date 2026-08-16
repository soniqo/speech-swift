import AVFoundation
import Foundation
import XCTest

@testable import VoiceChat

/// Real-checkpoint coverage for the live EAR-TTS tail. This class stays
/// separate so the isolated E2E runner releases the 11B bundle afterward.
final class E2EVoiceChatLiveTailTests: XCTestCase {
    func testContentScaledTailPreservesLateSpeechBeforeCompaction() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "fleurs_en", withExtension: "wav"))
        let samples = try loadMono16k(audioURL)
        let session = try await model.startSession(
            speech: .init(
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true,
            turnTaking: .nvidiaRealtime)

        let frameSize = VoiceChatSession.inputSamplesPerFrame
        for start in stride(from: 0, to: samples.count, by: frameSize) {
            _ = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)]))
        }
        _ = try await session.pushSilence(seconds: 20)

        let events = await session.events()
        let turnBegin = try XCTUnwrap(
            events.firstIndex { $0.textToken == model.tokenizer.bosID })
        let firstContent = try XCTUnwrap(
            events[(turnBegin + 1)...].firstIndex(where: \.speaking))
        let lastContent = try XCTUnwrap(events.lastIndex(where: \.speaking))
        let contentCount = events[turnBegin...lastContent]
            .filter(\.speaking).count
        let oldFixedCutoff = lastContent + 16
        let firstCompacted = lastContent
            + VoiceChatSpeechTurnState.acousticTailFrameBudget(
                contentFrames: contentCount)
            + 1

        XCTAssertGreaterThan(lastContent, firstContent)
        XCTAssertLessThan(firstCompacted, events.count)
        XCTAssertTrue(
            events[oldFixedCutoff].audio.contains { $0 != 0 },
            "the former fixed tail cutoff still contains generated speech")
        XCTAssertTrue(
            events[firstCompacted - 1].audio.contains { $0 != 0 },
            "the complete content-scaled tail must remain decoder-generated")
        XCTAssertTrue(
            events[firstCompacted].audio.allSatisfy { $0 == 0 },
            "the first frame beyond NVIDIA's content-scaled budget must compact")
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
