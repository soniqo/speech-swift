import AVFoundation
import Foundation
import MLX
import MLXRandom
import XCTest

@testable import VoiceChat

/// True audio-in/audio-out coverage. The isolated E2E runner launches this
/// class in its own process because the complete bundle is 8–22 GB.
final class E2EVoiceChatConversationTests: XCTestCase {
    private let paritySystemPrompt =
        "You are an AI voice assistant developed by NVIDIA. "
        + "Your name is NVIDIA Voice Chat. "
        + "Answer in a spoken, conversational style rather than a written one. "
        + "Do not repeat the same sentence over and over again."

    func testRealSpeechProducesTextAndModelAudio() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete encoder/llm/tts bundle")
        }
        let root = URL(fileURLWithPath: path)
        for component in ["encoder", "llm", "tts"] {
            guard FileManager.default.fileExists(
                atPath: root.appendingPathComponent(
                    "\(component)/model.safetensors").path) else {
                throw XCTSkip("VOICECHAT_BUNDLE has no \(component)/model.safetensors")
            }
        }

        let loadProgress = VoiceChatLoadProgressRecorder()
        let model = try await VoiceChatModel.load(
            from: root,
            progressHandler: { progress, stage in
                loadProgress.record(progress: progress, stage: stage)
            })
        XCTAssertEqual(loadProgress.stages, [
            "Loading tokenizer",
            "Loading perception encoder and RNN-T",
            "Loading 11B language model",
            "Loading EAR-TTS and audio codec",
            "Verifying and warming the audio codec",
            "VoiceChat model ready",
        ])
        XCTAssertEqual(loadProgress.lastProgress, 1, accuracy: 0.0001)
        let audioURL = try XCTUnwrap(
            Bundle.module.url(forResource: "fleurs_en", withExtension: "wav"))
        let samples = try loadMono16k(audioURL)
        XCTAssertFalse(model.transcriber.transcribe(samples).isEmpty)

        // Keep the published numerical fixture independent from the product
        // persona. Default-session coverage exercises the Soniqo prompt.
        MLXRandom.seed(0)
        let session = try await model.startSession(
            systemPrompt: paritySystemPrompt,
            streamUserTranscript: true)
        // Make completion deterministic while still exercising the learned
        // response tokens after the user audio.
        await session.forceTurn(atFrame: samples.count / VoiceChatSession.inputSamplesPerFrame)

        let chunk = VoiceChatSession.inputSamplesPerFrame * 4
        for start in stride(from: 0, to: samples.count, by: chunk) {
            _ = try await session.pushAudio(
                Array(samples[start ..< min(samples.count, start + chunk)]))
        }
        _ = try await session.pushSilence(seconds: 6)

        let reply = await session.reply()
        let userTranscript = await session.userTranscript().lowercased()
        let events = await session.events()
        let waveform = await session.renderedAudio()
        let summary = await session.summary()

        print("VoiceChat reply: \(reply)")
        print("VoiceChat streaming transcript: \(userTranscript)")
        print("VoiceChat summary: \(summary)")

        XCTAssertTrue(events.allSatisfy { $0.userTranscript != nil })
        for expected in ["also", "paid", "tribute", "luna"] {
            XCTAssertTrue(
                userTranscript.contains(expected),
                "streaming transcript is missing \(expected): \(userTranscript)")
        }
        XCTAssertFalse(reply.isEmpty, "duplex language channel emitted no response")
        XCTAssertGreaterThan(summary.speakingFrames, 0)
        XCTAssertNotNil(summary.firstSpeechFrame)
        XCTAssertGreaterThan(summary.perceptionP95Milliseconds, 0)
        XCTAssertGreaterThan(summary.totalP95Milliseconds, 0)
        XCTAssertEqual(
            summary.realTime,
            summary.totalP95Milliseconds < Double(VoiceChatSession.frameMilliseconds),
            "real-time classification must include perception, decision, and synthesis")
        let quantizationBits = try bundleQuantizationBits(
            root: root, component: "llm")
        if quantizationBits == 8 {
            XCTAssertEqual(summary.firstSpeechFrame, 42)
            XCTAssertEqual(
                reply,
                "Yes, Apollo eight, nine, and ten all made lunar orbits. Apollo eight was "
                    + "the first to orbit the moon in nineteen sixty eight. Apollo nine was "
                    + "in nineteen sixty nine, and Apollo ten was also in nineteen sixty "
                    + "nine. Apollo thirteen had an issue and did not reach the moon.")
        } else if quantizationBits == 5 {
            XCTAssertEqual(summary.firstSpeechFrame, 45)
            XCTAssertEqual(
                reply,
                "Yes, Apollo eight, nine, and ten all celebrated Christmas in lunar orbit. "
                    + "Apollo eight was the first to orbit the moon in nineteen sixty eight, "
                    + "and Apollo nine tested the lunar module in nineteen sixty nine. "
                    + "Apollo ten was the last to orbit the moon in nineteen seventy two. "
                    + "All three missions included the crew looping back to Earth to "
                    + "celebrate Christmas.")
        }
        XCTAssertEqual(
            waveform.count,
            events.count * VoiceChatSession.outputSamplesPerFrame)
        XCTAssertTrue(waveform.allSatisfy(\.isFinite))
        let rms = Foundation.sqrt(
            waveform.reduce(0) { $0 + Double($1 * $1) }
                / Double(max(1, waveform.count)))
        XCTAssertGreaterThan(rms, 1e-6, "model response decoded as silence")
        XCTAssertTrue(
            events.allSatisfy {
                $0.audio.count == VoiceChatSession.outputSamplesPerFrame
            })

        let liveWaveform = events.flatMap(\.audio)
        let dot = zip(liveWaveform, waveform).reduce(0.0) {
            $0 + Double($1.0 * $1.1)
        }
        let liveEnergy = liveWaveform.reduce(0.0) {
            $0 + Double($1 * $1)
        }
        let exactEnergy = waveform.reduce(0.0) {
            $0 + Double($1 * $1)
        }
        let cosine = dot / Foundation.sqrt(liveEnergy * exactEnergy)
        let streamingRMSE = Foundation.sqrt(
            zip(liveWaveform, waveform).reduce(0.0) {
                let difference = Double($1.0 - $1.1)
                return $0 + difference * difference
            } / Double(max(1, waveform.count)))
        print("Live codec vs full decode: cosine \(cosine), RMSE \(streamingRMSE)")
        // MLX convolution kernels can take shape-dependent GPU paths for the
        // bounded live window and the full offline decode. Preserve the tight
        // perceptual gate while allowing their sub-sample numeric drift.
        XCTAssertGreaterThan(cosine, 0.9999)
        XCTAssertLessThan(streamingRMSE, 2e-4)
    }

    private func bundleQuantizationBits(root: URL, component: String) throws -> Int? {
        let data = try Data(contentsOf: root.appendingPathComponent(
            "\(component)/config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return (json?["quantization"] as? [String: Any])?["bits"] as? Int
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
            frameCapacity: AVAudioFrameCount(Double(source.frameLength) * ratio) + 1_024)!
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

private final class VoiceChatLoadProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var updates: [(Double, String)] = []

    var stages: [String] {
        lock.lock()
        defer { lock.unlock() }
        return updates.map(\.1)
    }

    var lastProgress: Double {
        lock.lock()
        defer { lock.unlock() }
        return updates.last?.0 ?? 0
    }

    func record(progress: Double, stage: String) {
        lock.lock()
        updates.append((progress, stage))
        lock.unlock()
    }
}
