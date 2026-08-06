import Foundation
import MLX
import MLXRandom
import XCTest

@testable import VoiceChat

/// EAR-TTS and codec without the 11B language model, isolated from the complete
/// conversation test so peak memory from one cannot contaminate the other.
final class E2EVoiceChatSpeechGenerationTests: XCTestCase {
    func testTextScheduleGeneratesVaryingFiniteAudio() async throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a complete bundle")
        }
        let root = URL(fileURLWithPath: path)
        let speechDirectory = root.appendingPathComponent("tts")
        let weightsURL = speechDirectory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw XCTSkip("VOICECHAT_BUNDLE has no tts/model.safetensors")
        }

        let tokenizer = try await VoiceChatTokenizer.load(
            from: root.appendingPathComponent("llm"))
        let configuration = try VoiceChatSpeechConfiguration.load(from: speechDirectory)
        let weights = try MLX.loadArrays(url: weightsURL)
        let decoder = try VoiceChatSpeechDecoder(
            weights: weights, configuration: configuration, tokenizer: tokenizer)
        let codec = try VoiceChatCodec(weights: weights)

        let silence = codec.verifySilence()
        XCTAssertLessThan(silence.rms, VoiceChatCodec.silenceRMSLimit)

        let textIDs = tokenizer.encode(
            "Hello there, this is a VoiceChat speech test.")
        let catchup = max(
            1, Int((Double(textIDs.count) * 3.2).rounded()) - textIDs.count)
        let schedule = [tokenizer.padID, tokenizer.bosID]
            + textIDs
            + [Int](repeating: tokenizer.padID, count: catchup)
            + [tokenizer.eosID]
            + [Int](repeating: tokenizer.padID, count: 6)

        MLXRandom.seed(0)
        let warmup = decoder.warmup()
        let state = warmup.state
        var code = warmup.previousCode
        eval(code)
        var frames: [MLXArray] = []
        for token in schedule {
            code = try decoder.step(
                state: state, previousCode: code, textToken: token)
            eval(code)
            frames.append(code)
        }

        let codes = MLX.concatenated(frames, axis: 1)
        let waveform = codec.decode(latents: decoder.latents(for: codes))[0]
        eval(codes, waveform)

        XCTAssertEqual(codes.shape, [1, schedule.count, 31])
        XCTAssertGreaterThanOrEqual(MLX.min(codes).item(Int.self), 0)
        XCTAssertLessThan(MLX.max(codes).item(Int.self), 1_024)
        let primary = codes[0, 0..., 0].asArray(Int32.self)
        XCTAssertGreaterThan(Set(primary).count, 2, "codebook zero is stuck")
        if configuration.quantization?.bits == 8 {
            // Same seed/schedule through the Python MLX reference produces
            // these exact ids (all 1,364 codebook ids agree, not only this
            // diagnostic first stream).
            XCTAssertEqual(primary, [
                726, 726, 474, 807, 888, 852, 817, 757, 809, 105, 131,
                580, 580, 580, 580, 580, 954, 400, 840, 340, 639, 423,
                559, 668, 268, 575, 588, 231, 599, 197, 645, 826, 575,
                833, 668, 152, 726, 726, 726, 726, 726, 726, 726, 726,
            ])
        }

        XCTAssertEqual(waveform.dim(0), schedule.count * 1_764)
        XCTAssertTrue(MLX.all(MLX.isFinite(waveform)).item(Bool.self))
        let rms = MLX.sqrt(MLX.mean(waveform.square())).item(Float.self)
        print("EAR-TTS generated \(schedule.count) frames, RMS \(rms)")
        XCTAssertGreaterThan(rms, 1e-6)

        if let output = ProcessInfo.processInfo.environment["VOICECHAT_PARITY_OUTPUT"] {
            try MLX.save(
                arrays: ["codes": codes, "waveform": waveform],
                url: URL(fileURLWithPath: output))
        }
    }
}
