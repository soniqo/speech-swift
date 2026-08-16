import AVFoundation
import Foundation
import XCTest

@testable import VoiceChat

/// Opt-in performance regression for the live one-frame-at-a-time path.
/// Run in its own process because the complete INT5 bundle occupies multiple
/// gigabytes and timing is meaningful only when other model workloads are idle.
/// Use `swift test -c release`; debug builds are not a realtime performance gate.
final class E2EVoiceChatRealtimeCaptionTests: XCTestCase {
    func testStreamingCaptionsPreserveRealtimeThroughput() async throws {
        guard ProcessInfo.processInfo.environment["VOICECHAT_PERFORMANCE_TEST"] == "1" else {
            throw XCTSkip("set VOICECHAT_PERFORMANCE_TEST=1 for the live-caption timing gate")
        }
        let path = try XCTUnwrap(
            ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"],
            "set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")

        let model = try await VoiceChatModel.load(
            from: URL(fileURLWithPath: path))
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "fleurs_en", withExtension: "wav"))
        let samples = try loadMono16k(audioURL)
        let session = try await model.startSession(
            speech: .init(
                recentContextFrames: 250,
                realtimeIdleOptimization: true),
            streamUserTranscript: true)
        await session.forceTurn(
            atFrame: samples.count / VoiceChatSession.inputSamplesPerFrame)

        let frameSize = VoiceChatSession.inputSamplesPerFrame
        let started = DispatchTime.now().uptimeNanoseconds
        for start in stride(from: 0, to: samples.count, by: frameSize) {
            _ = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + frameSize)]))
        }
        // A short run does not expose retained lazy KV concatenations. Keep a
        // 30-second tail so this gate covers sustained-session throughput.
        let silence = [Float](
            repeating: 0,
            count: 30 * VoiceChatSession.inputSampleRate)
        for start in stride(from: 0, to: silence.count, by: frameSize) {
            _ = try await session.pushAudio(
                Array(silence[start ..< min(silence.count, start + frameSize)]))
        }
        let elapsedMilliseconds = Double(
            DispatchTime.now().uptimeNanoseconds - started) / 1_000_000

        let summary = await session.summary()
        let transcript = await session.userTranscript().lowercased()
        let timelineMilliseconds = Double(
            summary.frames * VoiceChatSession.frameMilliseconds)
        let rtf = elapsedMilliseconds / timelineMilliseconds
        // Run the comparison after the timed streaming region so the extra
        // offline encoder pass cannot contaminate the live RTF measurement.
        let offlineTranscript = model.transcriber.transcribe(samples)
            .lowercased()

        print(String(
            format: "VoiceChat captions: %.0f / %.0f ms, RTF %.2f, p50 %.1f ms",
            elapsedMilliseconds,
            timelineMilliseconds,
            rtf,
            summary.totalP50Milliseconds))
        print("VoiceChat offline caption: \(offlineTranscript)")
        print("VoiceChat streaming caption: \(transcript)")
        XCTAssertEqual(summary.frames, 420)
        XCTAssertLessThan(
            rtf, 1.0,
            "captioned inference cannot sustain the live frame clock")
        XCTAssertLessThan(
            summary.totalP50Milliseconds, 85,
            "median captioned inference no longer fits near the 80 ms frame budget")
        for expected in ["also", "paid", "tribute", "luna"] {
            XCTAssertTrue(
                transcript.contains(expected),
                "streaming transcript is missing \(expected): \(transcript)")
        }
        XCTAssertEqual(
            transcript,
            offlineTranscript,
            "one-frame streaming RNN-T must preserve offline greedy output")
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
