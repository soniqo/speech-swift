@testable import FishAudioTTS
import AudioCommon
import MLX
import Qwen3ASR
import XCTest

final class E2EFishAudioTTSTests: XCTestCase {
    func testLocalBundleLoadsAndGeneratesCodebooks() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio model-loading E2E")
        }

        let bundle = localFishAudioBundle()
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Fish Audio bundle not found at \(bundle.path)")
        }

        let tokenizer = try await FishAudioTokenizer.load(from: bundle)
        let config = try FishAudioConfig.load(from: bundle.appendingPathComponent("config.json"))
        let input = try FishAudioInputBuilder.build(
            text: "नमस्ते [excited]",
            tokenizer: tokenizer,
            config: config)

        let model = try FishAudioDualARModel.load(from: bundle)
        let generated = try model.generateCodebooks(
            from: input,
            sampling: FishAudioSamplingConfig(
                maxNewTokens: 1,
                temperature: 0,
                topK: 1,
                topP: 1,
                repetitionPenalty: 1))

        XCTAssertEqual(generated.codebookCount, config.audioDecoder.numCodebooks)
        XCTAssertLessThanOrEqual(generated.frameCount, 1)
    }

    func testLocalBundleTokenizerKeepsFishSpecialTokensAtomic() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio tokenizer E2E")
        }

        let bundle = localFishAudioBundle()
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("Fish Audio tokenizer not found at \(bundle.path)")
        }

        let tokenizer = try await FishAudioTokenizer.load(from: bundle)
        XCTAssertEqual(tokenizer.encode(FishAudioToken.imStart), [151_644])
        XCTAssertEqual(tokenizer.encode(FishAudioToken.imEnd), [151_645])
        XCTAssertEqual(tokenizer.encode(FishAudioToken.voiceModality), [151_673])
        XCTAssertEqual(tokenizer.encode(FishAudioToken.semantic(0)), [151_678])
    }

    func testLocalBundlePrefillTopSemanticMatchesPyTorchReference() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio prefill parity E2E")
        }

        let bundle = localFishAudioBundle()
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Fish Audio bundle not found at \(bundle.path)")
        }

        let tokenizer = try await FishAudioTokenizer.load(from: bundle)
        let config = try FishAudioConfig.load(from: bundle.appendingPathComponent("config.json"))
        let input = try FishAudioInputBuilder.build(
            text: "नमस्ते, आज मौसम अच्छा है। [excited]",
            tokenizer: tokenizer,
            config: config)
        let model = try FishAudioDualARModel.load(from: bundle)
        let result = model.forwardSlow(inputIds: input.asMLXArray(), state: model.initialSlowState()).0
        let lastIndex = result.logits.dim(1) - 1
        let logits = result.logits[0, lastIndex].asType(.float32).asArray(Float.self)
        let semanticTop = (config.semanticStartTokenId...config.semanticEndTokenId)
            .map { ($0, logits[$0]) }
            .sorted { $0.1 > $1.1 }
            .prefix(10)
            .map { ($0.0, $0.1, $0.0 - config.semanticStartTokenId) }

        XCTAssertEqual(semanticTop.first?.0, 154_636)
    }

    func testLocalBundleLoadsCodecAndDecodesOneFrame() throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio codec E2E")
        }

        let bundle = localFishAudioBundle()
        let codecURL = bundle.appendingPathComponent("codec.safetensors")
        guard FileManager.default.fileExists(atPath: codecURL.path) else {
            throw XCTSkip("Fish Audio codec not found at \(codecURL.path)")
        }

        let codec = try FishAudioCodec.load(from: bundle)
        let codes = MLXArray.zeros([
            1,
            FishAudioCodecDefaults.totalCodebooks,
            1,
        ], dtype: .int32)
        let waveform = try codec.decode(codes)
        eval(waveform)
        let samples = waveform.asArray(Float.self)

        XCTAssertEqual(samples.count, FishAudioCodecDefaults.samplesPerFrame)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite })
        XCTAssertLessThanOrEqual(samples.map(abs).max() ?? 0, 1.0001)
    }

    func testLocalBundleModelGeneratesAndDecodesWaveformSmoke() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio model waveform E2E")
        }

        let bundle = localFishAudioBundle()
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Fish Audio bundle not found at \(bundle.path)")
        }

        let model = try await FishAudioTTSModel.fromBundle(bundle)
        let codebooks = try model.generateCodebooks(
            text: "नमस्ते [excited]",
            sampling: FishAudioSamplingConfig(
                maxNewTokens: 1,
                temperature: 0,
                topK: 1,
                topP: 1,
                repetitionPenalty: 1))
        let samples = try model.decode(codebooks)

        XCTAssertEqual(samples.count, codebooks.frameCount * FishAudioCodecDefaults.samplesPerFrame)
        XCTAssertTrue(samples.allSatisfy { $0.isFinite })
        XCTAssertLessThanOrEqual(samples.map(abs).max() ?? 0, 1.0001)
    }

    func testLocalBundleCodecEncodesHindiReferenceAudio() throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio reference encode E2E")
        }

        let bundle = localFishAudioBundle()
        let referenceURL = URL(fileURLWithPath: "Tests/OmnilingualASRTests/Resources/fleurs_hi.wav")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Hindi reference fixture not found at \(referenceURL.path)")
        }

        let codec = try FishAudioCodec.load(from: bundle)
        let reference = try AudioFileLoader.load(
            url: referenceURL,
            targetSampleRate: FishAudioCodecDefaults.sampleRate)
        let codes = try codec.encode(
            audio: reference,
            sampleRate: FishAudioCodecDefaults.sampleRate)

        XCTAssertEqual(codes.codebookCount, FishAudioCodecDefaults.totalCodebooks)
        XCTAssertEqual(
            codes.frameCount,
            (reference.count + FishAudioCodecDefaults.samplesPerFrame - 1)
                / FishAudioCodecDefaults.samplesPerFrame)
        XCTAssertTrue(codes.codes.allSatisfy { $0.count == codes.frameCount })
        XCTAssertTrue(codes.codes[0].allSatisfy { (0..<FishAudioCodecDefaults.semanticCodebookSize).contains($0) })
        for row in codes.codes.dropFirst() {
            XCTAssertTrue(row.allSatisfy { (0..<FishAudioCodecDefaults.residualCodebookSize).contains($0) })
        }
    }

    func testLocalBundleVoiceCloningFromHindiReferenceSmoke() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_E2E=1 to run Fish Audio voice-cloning E2E")
        }

        let bundle = localFishAudioBundle()
        let referenceURL = URL(fileURLWithPath: "Tests/OmnilingualASRTests/Resources/fleurs_hi.wav")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Hindi reference fixture not found at \(referenceURL.path)")
        }

        let model = try await FishAudioTTSModel.fromBundle(bundle)
        let reference = try model.encodeReferencePrompt(
            audio: AudioFileLoader.load(
                url: referenceURL,
                targetSampleRate: FishAudioCodecDefaults.sampleRate),
            sampleRate: FishAudioCodecDefaults.sampleRate,
            text: "लूना को साथी पहलवानों ने भी श्रद्धांजलि दी।")
        let codebooks = try model.generateCodebooks(
            text: "नमस्ते [excited]",
            references: [reference],
            sampling: hindiCloneSampling(maxNewTokens: 32))
        let samples = try model.decode(codebooks)

        XCTAssertEqual(codebooks.codebookCount, FishAudioCodecDefaults.totalCodebooks)
        XCTAssertEqual(samples.count, codebooks.frameCount * FishAudioCodecDefaults.samplesPerFrame)
        assertUsableWaveform(samples, sampleRate: model.sampleRate)
    }

    func testVoiceCloningHindiRoundTripWithASR() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_ASR_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_ASR_E2E=1 to run Fish Audio ASR round-trip E2E")
        }

        let bundle = localFishAudioBundle()
        let referenceURL = URL(fileURLWithPath: "Tests/OmnilingualASRTests/Resources/fleurs_hi.wav")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Hindi reference fixture not found at \(referenceURL.path)")
        }

        let model = try await FishAudioTTSModel.fromBundle(bundle)
        let target = "नमस्ते, आज मौसम अच्छा है। [excited]"
        let waveform = try await model.generate(
            text: target,
            referenceAudioURL: referenceURL,
            referenceText: "लूना को साथी पहलवानों ने भी श्रद्धांजलि दी।",
            sampling: hindiCloneSampling(maxNewTokens: 96))
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)
        guard !waveform.isEmpty else { return }
        if let wavPath = ProcessInfo.processInfo.environment["FISH_AUDIO_ROUNDTRIP_WAV"] {
            try WAVWriter.write(
                samples: waveform,
                sampleRate: model.sampleRate,
                to: URL(fileURLWithPath: wavPath))
            print("Fish Audio Hindi voice-clone wav: \(wavPath)")
        }

        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(
            audio: waveform,
            sampleRate: model.sampleRate,
            language: "hindi")
        print("Fish Audio Hindi voice-clone roundtrip: \(target) -> \(transcription)")

        assertHindiConceptsRecovered(
            in: transcription,
            concepts: greetingWeatherConcepts,
            minimum: 2,
            message: "ASR should recover multiple Hindi concepts from Fish Audio cloned speech")
    }

    func testTextOnlyHindiRoundTripWithASR() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_ASR_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_ASR_E2E=1 to run Fish Audio ASR round-trip E2E")
        }

        let bundle = localFishAudioBundle()
        guard FileManager.default.fileExists(atPath: bundle.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Fish Audio bundle not found at \(bundle.path)")
        }

        let model = try await FishAudioTTSModel.fromBundle(bundle)
        let target = "नमस्ते, आज मौसम अच्छा है। [excited]"
        let waveform = try await model.generate(
            text: target,
            sampling: hindiCloneSampling(maxNewTokens: 96))
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)
        guard !waveform.isEmpty else { return }
        if let wavPath = ProcessInfo.processInfo.environment["FISH_AUDIO_TEXT_ONLY_WAV"] {
            try WAVWriter.write(
                samples: waveform,
                sampleRate: model.sampleRate,
                to: URL(fileURLWithPath: wavPath))
            print("Fish Audio Hindi text-only wav: \(wavPath)")
        }

        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(
            audio: waveform,
            sampleRate: model.sampleRate,
            language: "hindi")
        print("Fish Audio Hindi text-only roundtrip: \(target) -> \(transcription)")

        assertHindiConceptsRecovered(
            in: transcription,
            concepts: greetingWeatherConcepts,
            minimum: 2,
            message: "ASR should recover multiple Hindi concepts from Fish Audio speech")
    }

    func testCodecHindiReferenceRoundTripWithASR() async throws {
        guard ProcessInfo.processInfo.environment["FISH_AUDIO_ASR_E2E"] == "1" else {
            throw XCTSkip("Set FISH_AUDIO_ASR_E2E=1 to run Fish Audio codec ASR round-trip E2E")
        }

        let bundle = localFishAudioBundle()
        let referenceURL = URL(fileURLWithPath: "Tests/OmnilingualASRTests/Resources/fleurs_hi.wav")
        guard FileManager.default.fileExists(atPath: referenceURL.path) else {
            throw XCTSkip("Hindi reference fixture not found at \(referenceURL.path)")
        }

        let codec = try FishAudioCodec.load(from: bundle)
        let reference = try AudioFileLoader.load(
            url: referenceURL,
            targetSampleRate: FishAudioCodecDefaults.sampleRate)
        let codes = try codec.encode(
            audio: reference,
            sampleRate: FishAudioCodecDefaults.sampleRate)
        let reconstruction = try codec.decode(codes)
        assertUsableWaveform(reconstruction, sampleRate: FishAudioCodecDefaults.sampleRate)
        guard !reconstruction.isEmpty else { return }
        if let wavPath = ProcessInfo.processInfo.environment["FISH_AUDIO_CODEC_ROUNDTRIP_WAV"] {
            try WAVWriter.write(
                samples: reconstruction,
                sampleRate: FishAudioCodecDefaults.sampleRate,
                to: URL(fileURLWithPath: wavPath))
            print("Fish Audio codec round-trip wav: \(wavPath)")
        }

        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(
            audio: reconstruction,
            sampleRate: FishAudioCodecDefaults.sampleRate,
            language: "hindi")
        print("Fish Audio codec roundtrip: fleurs_hi.wav -> \(transcription)")

        assertHindiConceptsRecovered(
            in: transcription,
            concepts: [
                ["लुना", "लूना"],
                ["साथी"],
                ["पहलवानों", "पलवानों"],
            ],
            minimum: 2,
            message: "ASR should recover Hindi reference concepts after Fish codec encode/decode")
    }

    private var greetingWeatherConcepts: [[String]] {
        [
            ["नमस्ते"],
            ["मौसम", "मोसम", "मॉसम"],
            ["अच्छा", "अछा"],
        ]
    }

    private func localFishAudioBundle() -> URL {
        if let override = ProcessInfo.processInfo.environment["FISH_AUDIO_BUNDLE"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Caches/soniqo/hindi-emotion-tts-exports/Fish-Audio-S2-Pro-MLX-fp16",
                                    isDirectory: true)
    }

    private func assertHindiConceptsRecovered(
        in transcription: String,
        concepts: [[String]],
        minimum: Int,
        message: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let transcriptionScalars = Array(transcription.unicodeScalars.map(\.value))
        let recovered = concepts.filter { variants in
            variants.contains { containsScalarSequence($0, in: transcriptionScalars) }
        }
        XCTAssertGreaterThanOrEqual(
            recovered.count,
            minimum,
            "\(message): \(transcription)",
            file: file,
            line: line)
    }

    private func containsScalarSequence(_ needle: String, in haystack: [UInt32]) -> Bool {
        let target = Array(needle.unicodeScalars.map(\.value))
        guard !target.isEmpty, target.count <= haystack.count else { return false }
        for index in 0...(haystack.count - target.count) {
            if Array(haystack[index..<(index + target.count)]) == target {
                return true
            }
        }
        return false
    }

    private func assertUsableWaveform(
        _ waveform: [Float],
        sampleRate: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(waveform.isEmpty, "Waveform should not be empty", file: file, line: line)
        guard !waveform.isEmpty else { return }
        XCTAssertGreaterThan(Double(waveform.count) / Double(sampleRate), 0.05, file: file, line: line)
        XCTAssertTrue(waveform.allSatisfy(\.isFinite), "Waveform must not contain NaN/Inf", file: file, line: line)
        XCTAssertGreaterThan(waveform.map { abs($0) }.max() ?? 0, 0.0001, file: file, line: line)
        XCTAssertLessThanOrEqual(waveform.map { abs($0) }.max() ?? 0, 1.0001, file: file, line: line)
    }

    private func hindiCloneSampling(maxNewTokens: Int) -> FishAudioSamplingConfig {
        FishAudioSamplingConfig(
            maxNewTokens: maxNewTokens,
            temperature: 1.0,
            topK: 30,
            topP: 0.9,
            repetitionPenalty: 1.0,
            minNewTokens: min(48, maxNewTokens))
    }
}
