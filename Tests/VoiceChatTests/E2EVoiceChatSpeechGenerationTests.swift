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

        let forcedSilence = try decoder.step(
            state: state,
            previousCode: code,
            textToken: tokenizer.padID,
            forceSilence: true)
        eval(forcedSilence)
        XCTAssertEqual(
            forcedSilence.asArray(Int32.self),
            decoder.silenceCodes.asArray(Int32.self))

        let sequentialWarmup = decoder.warmup()
        var sequentialCode = sequentialWarmup.previousCode
        for _ in 0 ..< VoiceChatSession.idleSpeechBatchFrames {
            sequentialCode = try decoder.step(
                state: sequentialWarmup.state,
                previousCode: sequentialCode,
                textToken: tokenizer.padID,
                forceSilence: true)
            let cacheArrays = sequentialWarmup.state.attention.flatMap {
                [$0.keys, $0.values].compactMap { $0 }
            }
            MLX.eval([sequentialCode] + cacheArrays)
        }
        let batchedWarmup = decoder.warmup()
        let batchedCode = decoder.advanceIdleSilence(
            state: batchedWarmup.state,
            previousCode: batchedWarmup.previousCode,
            frames: VoiceChatSession.idleSpeechBatchFrames,
            guidance: 0.2)
        let batchedCacheArrays = batchedWarmup.state.attention.flatMap {
            [$0.keys, $0.values].compactMap { $0 }
        }
        MLX.eval([batchedCode] + batchedCacheArrays)
        XCTAssertEqual(
            batchedCode.asArray(Int32.self),
            decoder.silenceCodes.asArray(Int32.self))

        var minimumCacheCosine: Float = 1
        for (sequential, batched) in zip(
            sequentialWarmup.state.attention,
            batchedWarmup.state.attention
        ) {
            for (lhs, rhs) in [(sequential.keys!, batched.keys!),
                               (sequential.values!, batched.values!)] {
                let a = lhs.asType(.float32).reshaped([-1])
                let b = rhs.asType(.float32).reshaped([-1])
                let cosine = (MLX.sum(a * b)
                    / (MLX.sqrt(MLX.sum(a.square()))
                        * MLX.sqrt(MLX.sum(b.square())) + MLXArray(Float(1e-8))))
                    .item(Float.self)
                minimumCacheCosine = min(minimumCacheCosine, cosine)
            }
        }
        print("idle batch minimum cache cosine \(minimumCacheCosine)")
        XCTAssertGreaterThan(minimumCacheCosine, 0.999)

        // Use a fresh prompt state so the published schedule below remains an
        // exact parity check independent of the forced-silence cache advance.
        let scheduleWarmup = decoder.warmup()
        let scheduleState = scheduleWarmup.state
        code = scheduleWarmup.previousCode
        var frames: [MLXArray] = []
        for token in schedule {
            code = try decoder.step(
                state: scheduleState, previousCode: code, textToken: token)
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
        } else if configuration.quantization?.bits == 5 {
            // Preserve the protected-head INT5 output across performance
            // refactors. The pre-headroom and incremental-RVQ implementations
            // produce byte-identical full code and waveform tensors.
            XCTAssertEqual(primary, [
                726, 726, 474, 494, 888, 397, 891, 198, 109, 482, 580,
                580, 580, 580, 726, 580, 80, 400, 840, 345, 471, 324,
                452, 400, 771, 1001, 596, 345, 954, 146, 550, 231, 479,
                914, 345, 391, 580, 726, 726, 726, 726, 726, 726, 726,
            ])
        }

        XCTAssertEqual(waveform.dim(0), schedule.count * 1_764)
        XCTAssertTrue(MLX.all(MLX.isFinite(waveform)).item(Bool.self))
        let rms = MLX.sqrt(MLX.mean(waveform.square())).item(Float.self)
        print("EAR-TTS generated \(schedule.count) frames, RMS \(rms)")
        XCTAssertGreaterThan(rms, 1e-6)

        // Live sessions bound only generated TTS history. Advance beyond that
        // boundary, then require a later turn to remain valid and voiced. This
        // catches loss of the retained speaker prompt, RoPE reset, and cache
        // shape growth without changing the exact full-history parity above.
        let boundedWarmup = decoder.warmup(recentContextFrames: 250)
        let boundedState = boundedWarmup.state
        var boundedCode = boundedWarmup.previousCode
        for _ in 0 ..< 260 {
            boundedCode = try decoder.step(
                state: boundedState,
                previousCode: boundedCode,
                textToken: tokenizer.padID,
                forceSilence: true)
            let cacheArrays = boundedState.attention.flatMap {
                [$0.keys, $0.values].compactMap { $0 }
            }
            MLX.eval([boundedCode] + cacheArrays)
        }
        XCTAssertTrue(boundedState.attention.allSatisfy {
            $0.offset == configuration.promptFrames + 260
                && $0.keys?.dim(2) == configuration.promptFrames + 250
        })

        let laterText = tokenizer.encode("My name is Soniqo.")
        let laterSchedule = [tokenizer.bosID]
            + laterText
            + [Int](repeating: tokenizer.padID, count: max(4, laterText.count))
            + [tokenizer.eosID]
        MLXRandom.seed(0)
        var laterFrames: [MLXArray] = []
        for token in laterSchedule {
            boundedCode = try decoder.step(
                state: boundedState,
                previousCode: boundedCode,
                textToken: token)
            let cacheArrays = boundedState.attention.flatMap {
                [$0.keys, $0.values].compactMap { $0 }
            }
            MLX.eval([boundedCode] + cacheArrays)
            laterFrames.append(boundedCode)
        }
        let laterCodes = MLX.concatenated(laterFrames, axis: 1)
        let laterWaveform = codec.decode(
            latents: decoder.latents(for: laterCodes))[0]
        eval(laterCodes, laterWaveform)
        XCTAssertEqual(
            laterWaveform.dim(0), laterSchedule.count * VoiceChatCodec.samplesPerFrame)
        XCTAssertTrue(MLX.all(MLX.isFinite(laterWaveform)).item(Bool.self))
        XCTAssertGreaterThan(
            Set(laterCodes[0, 0..., 0].asArray(Int32.self)).count, 2,
            "bounded late-turn speech codes are stuck")
        XCTAssertGreaterThan(
            MLX.sqrt(MLX.mean(laterWaveform.square())).item(Float.self),
            1e-6)

        if let output = ProcessInfo.processInfo.environment["VOICECHAT_PARITY_OUTPUT"] {
            try MLX.save(
                arrays: ["codes": codes, "waveform": waveform],
                url: URL(fileURLWithPath: output))
        }
    }
}
