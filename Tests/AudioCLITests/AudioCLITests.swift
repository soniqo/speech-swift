import XCTest
import ArgumentParser
@testable import AudioCLILib

// MARK: - Root Command

final class AudioCLIRootTests: XCTestCase {

    func testHelpParsesSuccessfully() throws {
        // --help may throw CleanExit or return the command depending on AP version
        do {
            _ = try AudioCLI.parseAsRoot(["--help"])
        } catch {
            XCTAssertEqual(AudioCLI.exitCode(for: error), .success)
        }
    }

    func testHelpContainsAllSubcommands() {
        let help = AudioCLI.helpMessage()
        XCTAssertTrue(help.contains("transcribe"))
        XCTAssertTrue(help.contains("align"))
        XCTAssertTrue(help.contains("speak"))
        XCTAssertTrue(help.contains("respond"))
    }

    func testHelpContainsAbstract() {
        let help = AudioCLI.helpMessage()
        XCTAssertTrue(help.contains("AI speech models"))
    }

    func testUnknownSubcommandFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["synthesize"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testNoSubcommandParsesRootCommand() throws {
        // No arguments — returns root command (which prints help on run())
        do {
            let cmd = try AudioCLI.parseAsRoot([])
            XCTAssertTrue(cmd is AudioCLI)
        } catch {
            // Some AP versions throw; that's also acceptable
        }
    }
}

// MARK: - TranscribeCommand

final class TranscribeCommandTests: XCTestCase {

    func testParsesAudioFile() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "/path/to/audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.audioFile, "/path/to/audio.wav")
    }

    func testDefaultModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "0.6B")
    }

    func testDefaultLanguageIsNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertNil(transcribe.language)
    }

    func testParsesModelOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--model", "1.7B"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "1.7B")
    }

    func testParsesModelShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "-m", "1.7B"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "1.7B")
    }

    func testParsesLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--language", "zh"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.language, "zh")
    }

    func testDefaultContextIsNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertNil(transcribe.context)
    }

    func testParsesContext() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "transcribe", "audio.wav",
            "--context", "Project: Meander, participants: Will, Adam"
        ])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.context, "Project: Meander, participants: Will, Adam")
    }

    func testParsesFullModelId() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "-m", "org/my-custom-model"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.model, "org/my-custom-model")
    }

    func testMissingAudioFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["transcribe"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testHelpParsesSuccessfully() throws {
        do {
            _ = try AudioCLI.parseAsRoot(["transcribe", "--help"])
        } catch {
            XCTAssertEqual(AudioCLI.exitCode(for: error), .success)
        }
    }

    // MARK: --engine omnilingual

    func testParsesOmnilingualEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--engine", "omnilingual"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.engine, "omnilingual")
    }

    func testOmnilingualDefaultWindowIs10() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--engine", "omnilingual"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.window, 10)
    }

    func testOmnilingualParses5sWindow() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "transcribe", "audio.wav", "--engine", "omnilingual", "--window", "5"
        ])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.window, 5)
    }

    func testOmnilingualRejectsInvalidWindow() {
        XCTAssertThrowsError(try {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", "omnilingual", "--window", "7"
            ])
            try (cmd as? TranscribeCommand)?.validate()
        }())
    }

    func testRejectsUnknownEngine() {
        XCTAssertThrowsError(try {
            let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--engine", "bogus"])
            try (cmd as? TranscribeCommand)?.validate()
        }())
    }

    // MARK: --engine cohere / voxtral

    func testParsesCohereAndVoxtralEngines() throws {
        for engine in ["cohere", "voxtral"] {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", engine,
            ])
            let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
            XCTAssertEqual(transcribe.engine, engine)
            XCTAssertNoThrow(try transcribe.validate())
        }
    }

    func testCohereModelAliases() throws {
        XCTAssertEqual(
            try resolveCohereModelId("int5"),
            "aufklarer/Cohere-Transcribe-2B-MLX-5bit")
        XCTAssertEqual(
            try resolveCohereModelId("8bit"),
            "aufklarer/Cohere-Transcribe-2B-MLX-8bit")
        XCTAssertEqual(
            try resolveCohereModelId("fp16"),
            "aufklarer/Cohere-Transcribe-2B-MLX-FP16")
    }

    func testVoxtralModelAliases() throws {
        XCTAssertEqual(
            try resolveVoxtralModelId("default"),
            "aufklarer/Voxtral-Mini-3B-2507-MLX-5bit")
        XCTAssertEqual(
            try resolveVoxtralModelId("int8"),
            "aufklarer/Voxtral-Mini-3B-2507-MLX-8bit")
        XCTAssertEqual(
            try resolveVoxtralModelId("16bit"),
            "aufklarer/Voxtral-Mini-3B-2507-MLX-FP16")
    }

    func testCohereAndVoxtralAcceptModelIDsAndLocalPaths() throws {
        for resolver in [resolveCohereModelId, resolveVoxtralModelId] {
            XCTAssertEqual(
                try resolver("org/custom-model"),
                "org/custom-model")
            XCTAssertEqual(
                try resolver("./local-model"),
                "./local-model")
        }
    }

    func testCohereAndVoxtralRejectINT7() {
        XCTAssertThrowsError(try resolveCohereModelId("int7"))
        XCTAssertThrowsError(try resolveVoxtralModelId("7bit"))
    }

    func testCohereAndVoxtralRejectStreaming() {
        for engine in ["cohere", "voxtral"] {
            XCTAssertThrowsError(try {
                let cmd = try AudioCLI.parseAsRoot([
                    "transcribe", "audio.wav", "--engine", engine, "--stream",
                ])
                try (cmd as? TranscribeCommand)?.validate()
            }())
        }
    }

    // MARK: --engine whisper

    func testParsesWhisperEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--engine", "whisper"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.engine, "whisper")
        XCTAssertNoThrow(try transcribe.validate())
    }

    func testWhisperModelAliasesResolveDefaultCoreMLRepo() throws {
        for alias in ["default", "turbo", "whisper", "large-v3-turbo"] {
            XCTAssertEqual(
                try resolveWhisperModelId(alias),
                "aufklarer/Whisper-Large-v3-Turbo-CoreML",
                "\(alias) should resolve to the default Whisper CoreML repo")
        }
    }

    func testWhisperAcceptsFullModelId() throws {
        XCTAssertEqual(
            try resolveWhisperModelId("org/custom-whisper-coreml"),
            "org/custom-whisper-coreml")
    }

    func testWhisperRejectsUnknownShortModelName() {
        XCTAssertThrowsError(try resolveWhisperModelId("medium"))
    }

    // MARK: --engine omnilingual --backend mlx

    func testOmnilingualDefaultBackendIsCoreML() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe", "audio.wav", "--engine", "omnilingual"])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.backend, "coreml")
    }

    func testOmnilingualParsesMLXBackend() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "transcribe", "audio.wav", "--engine", "omnilingual", "--backend", "mlx"
        ])
        let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
        XCTAssertEqual(transcribe.backend, "mlx")
        XCTAssertEqual(transcribe.variant, "300M")
        XCTAssertEqual(transcribe.bits, 4)
    }

    func testOmnilingualMLXAcceptsAllVariants() throws {
        for v in ["300M", "1B", "3B", "7B"] {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", "omnilingual",
                "--backend", "mlx", "--variant", v
            ])
            let transcribe = try XCTUnwrap(cmd as? TranscribeCommand)
            XCTAssertNoThrow(try transcribe.validate(), "variant \(v) should validate")
        }
    }

    func testOmnilingualMLXRejectsBogusVariant() {
        XCTAssertThrowsError(try {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", "omnilingual",
                "--backend", "mlx", "--variant", "999B"
            ])
            try (cmd as? TranscribeCommand)?.validate()
        }())
    }

    func testOmnilingualMLXRejectsBogusBits() {
        XCTAssertThrowsError(try {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", "omnilingual",
                "--backend", "mlx", "--bits", "16"
            ])
            try (cmd as? TranscribeCommand)?.validate()
        }())
    }

    func testOmnilingualRejectsBogusBackend() {
        XCTAssertThrowsError(try {
            let cmd = try AudioCLI.parseAsRoot([
                "transcribe", "audio.wav", "--engine", "omnilingual", "--backend", "tflite"
            ])
            try (cmd as? TranscribeCommand)?.validate()
        }())
    }
}

// MARK: - TranscribeBatchCommand

final class TranscribeBatchCommandTests: XCTestCase {

    func testParsesInputDirectory() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe-batch", "/tmp/audio"])
        let batch = try XCTUnwrap(cmd as? TranscribeBatchCommand)
        XCTAssertEqual(batch.inputDir, "/tmp/audio")
    }

    func testDefaultBatchSizeIsOne() throws {
        let cmd = try AudioCLI.parseAsRoot(["transcribe-batch", "/tmp/audio"])
        let batch = try XCTUnwrap(cmd as? TranscribeBatchCommand)
        XCTAssertEqual(batch.batchSize, 1)
    }

    func testParsesBatchSize() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "transcribe-batch", "/tmp/audio", "--batch-size", "4"
        ])
        let batch = try XCTUnwrap(cmd as? TranscribeBatchCommand)
        XCTAssertEqual(batch.batchSize, 4)
    }

    func testParsesQwen3Options() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "transcribe-batch", "/tmp/audio",
            "--engine", "qwen3",
            "--model", "1.7B-8bit",
            "--language", "en",
            "--extensions", "wav,m4a",
            "--jsonl"
        ])
        let batch = try XCTUnwrap(cmd as? TranscribeBatchCommand)
        XCTAssertEqual(batch.engine, "qwen3")
        XCTAssertEqual(batch.model, "1.7B-8bit")
        XCTAssertEqual(batch.language, "en")
        XCTAssertEqual(batch.extensions, "wav,m4a")
        XCTAssertTrue(batch.jsonl)
    }
}

// MARK: - AlignCommand

final class AlignCommandTests: XCTestCase {

    func testParsesAudioFile() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "/path/audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.audioFile, "/path/audio.wav")
    }

    func testDefaultTextIsNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertNil(align.text)
    }

    func testDefaultModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.model, "0.6B")
    }

    func testDefaultAlignerModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        // The default is engine-dependent (mlx → 4-bit MLX, coreml → CoreML
        // FP16) and resolved inside `run()`, so the parsed value is nil.
        XCTAssertNil(align.alignerModel)
    }

    func testDefaultEngineIsMLX() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.engine, "mlx")
    }

    func testEngineCoremlParses() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--engine", "coreml"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.engine, "coreml")
        XCTAssertNil(align.alignerModel)
    }

    func testParsesTextOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--text", "Hello world"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.text, "Hello world")
    }

    func testParsesTextShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "-t", "Hello world"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.text, "Hello world")
    }

    func testParsesAlignerModel() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--aligner-model", "org/custom-aligner"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.alignerModel, "org/custom-aligner")
    }

    func testParsesLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["align", "audio.wav", "--language", "de"])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.language, "de")
    }

    func testMissingAudioFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["align"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testAllOptionsCombined() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "align", "audio.wav",
            "-t", "Test text",
            "-m", "1.7B",
            "--aligner-model", "org/aligner",
            "--language", "fr"
        ])
        let align = try XCTUnwrap(cmd as? AlignCommand)
        XCTAssertEqual(align.audioFile, "audio.wav")
        XCTAssertEqual(align.text, "Test text")
        XCTAssertEqual(align.model, "1.7B")
        XCTAssertEqual(align.alignerModel, "org/aligner")
        XCTAssertEqual(align.language, "fr")
    }
}

// MARK: - DiarizeCommand

final class DiarizeCommandTests: XCTestCase {

    func testCommunity1Defaults() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "diarize", "meeting.wav", "--engine", "community1",
        ])
        let diarize = try XCTUnwrap(cmd as? DiarizeCommand)
        XCTAssertEqual(diarize.audioFile, "meeting.wav")
        XCTAssertEqual(diarize.engine, "community1")
        XCTAssertEqual(diarize.community1ComputeUnits, "ane")
        XCTAssertNil(diarize.numSpeakers)
        XCTAssertEqual(diarize.minSpeakers, 1)
        XCTAssertNil(diarize.maxSpeakers)
    }

    func testCommunity1SpeakerBoundsAndComputeUnitsParse() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "diarize", "meeting.wav",
            "--engine", "community1",
            "--community1-compute-units", "cpu",
            "--num-speakers", "3",
            "--min-speakers", "2",
            "--max-speakers", "5",
        ])
        let diarize = try XCTUnwrap(cmd as? DiarizeCommand)
        XCTAssertEqual(diarize.community1ComputeUnits, "cpu")
        XCTAssertEqual(diarize.numSpeakers, 3)
        XCTAssertEqual(diarize.minSpeakers, 2)
        XCTAssertEqual(diarize.maxSpeakers, 5)
    }
}

// MARK: - EmbedSpeakerCommand

final class EmbedSpeakerCommandTests: XCTestCase {

    func testParsesReDimNet2Engine() throws {
        let command = try AudioCLI.parseAsRoot([
            "embed-speaker", "voice.wav", "--engine", "redimnet2",
        ])
        let embedSpeaker = try XCTUnwrap(command as? EmbedSpeakerCommand)

        XCTAssertEqual(embedSpeaker.audioFile, "voice.wav")
        XCTAssertEqual(embedSpeaker.engine, "redimnet2")
    }

    func testHelpDocumentsReDimNet2Engine() {
        XCTAssertTrue(EmbedSpeakerCommand.helpMessage().contains("redimnet2"))
    }
}

// MARK: - SpeakCommand

final class SpeakCommandTests: XCTestCase {

    // MARK: Defaults

    func testDefaultEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "qwen3")
    }

    func testDefaultOutput() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "output.wav")
    }

    func testDefaultLanguage() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertNil(speak.language)
    }

    func testDefaultSamplingParams() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.temperature, 0.3, accuracy: 0.001)
        XCTAssertEqual(speak.topK, 50)
        XCTAssertEqual(speak.maxTokens, 500)
    }

    func testDefaultFlags() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertFalse(speak.stream)
        XCTAssertFalse(speak.listSpeakers)
        XCTAssertFalse(speak.verbose)
    }

    func testDefaultQwen3Options() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "base")
        XCTAssertNil(speak.speaker)
        XCTAssertNil(speak.instruct)
        XCTAssertNil(speak.batchFile)
        XCTAssertEqual(speak.batchSize, 4)
        XCTAssertEqual(speak.firstChunkFrames, 3)
        XCTAssertEqual(speak.chunkFrames, 25)
    }

    func testDefaultCosyVoiceModelId() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        // --model-id is now opt-in; the runtime resolves the default through
        // --cosyvoice-variant (bf16 by default → aufklarer/CosyVoice3-0.5B-MLX-bf16).
        XCTAssertNil(speak.modelId)
        XCTAssertEqual(speak.cosyvoiceVariant, "bf16")
    }

    func testCosyVoiceVariantBf16() throws {
        let cmd = try AudioCLI.parseAsRoot(
            ["speak", "--engine", "cosyvoice", "--cosyvoice-variant", "bf16", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.cosyvoiceVariant, "bf16")
        XCTAssertNil(speak.modelId)
    }

    func testCosyVoiceVariant16BitAlias() throws {
        let cmd = try AudioCLI.parseAsRoot(
            ["speak", "--engine", "cosyvoice", "--cosyvoice-variant", "16bit", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.cosyvoiceVariant, "16bit")
        XCTAssertNil(speak.modelId)
    }

    func testCosyVoiceModelIdOverridesVariant() throws {
        let cmd = try AudioCLI.parseAsRoot(
            ["speak", "--engine", "cosyvoice", "--model-id", "custom/Model", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.modelId, "custom/Model")
    }

    // MARK: Engine selection

    func testCosyVoiceEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--engine", "cosyvoice", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "cosyvoice")
    }

    func testIndicMioEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--engine", "indic-mio", "नमस्ते <happy>"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "indic-mio")
        XCTAssertEqual(speak.indicMioModelId, "aufklarer/Indic-Mio-MLX-fp16")
        XCTAssertEqual(speak.indicMioTopP, 0.9, accuracy: 0.001)
        XCTAssertEqual(speak.indicMioRepetitionPenalty, 1.0, accuracy: 0.001)
    }

    func testIndexTTS2Engine() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "--engine", "indextts2", "Hello",
            "--voice-sample", "ref.wav",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "indextts2")
        XCTAssertEqual(speak.voiceSample, "ref.wav")
        XCTAssertEqual(speak.indextts2ModelId, "aufklarer/IndexTTS2-MLX-fp16")
    }

    func testInvalidEngineFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["speak", "--engine", "whisper", "hi"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    // MARK: Validation

    func testNoTextNoFlagsNoFileFails() {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(["speak"])) { error in
            XCTAssertEqual(AudioCLI.exitCode(for: error), .validationFailure)
        }
    }

    func testListSpeakersSatisfiesValidation() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--list-speakers"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.listSpeakers)
        XCTAssertNil(speak.text)
    }

    func testBatchFileSatisfiesValidation() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--batch-file", "texts.txt"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.batchFile, "texts.txt")
        XCTAssertNil(speak.text)
    }

    // MARK: Options parsing

    func testOutputOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--output", "out.wav"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "out.wav")
    }

    func testOutputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "-o", "short.wav"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.output, "short.wav")
    }

    func testStreamFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--stream"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.stream)
    }

    func testSamplingOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi",
            "--temperature", "0.7",
            "--top-k", "25",
            "--max-tokens", "200"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.temperature, 0.7, accuracy: 0.001)
        XCTAssertEqual(speak.topK, 25)
        XCTAssertEqual(speak.maxTokens, 200)
    }

    func testSpeakerOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--speaker", "vivian"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.speaker, "vivian")
    }

    func testInstructOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--instruct", "Speak cheerfully"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.instruct, "Speak cheerfully")
    }

    func testModelOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--model", "customVoice"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "customVoice")
    }

    func testModel8bitOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--model", "base-8bit"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "base-8bit")
    }

    func testModel17BOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--model", "1.7b"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "1.7b")
    }

    func testModel17B8bitOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--model", "1.7b-8bit"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.model, "1.7b-8bit")
    }

    func testChunkFramesOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi", "--stream",
            "--first-chunk-frames", "1",
            "--chunk-frames", "10"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.firstChunkFrames, 1)
        XCTAssertEqual(speak.chunkFrames, 10)
    }

    func testCosyVoiceModelIdOption() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "hi", "--engine", "cosyvoice",
            "--model-id", "org/my-cosyvoice"
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.modelId, "org/my-cosyvoice")
    }

    func testIndicMioOptions() throws {
        let embedding = Array(repeating: "0", count: 128).joined(separator: ",")
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "नमस्ते <happy>",
            "--engine", "indic-mio",
            "--indic-mio-model-id", "org/indic-mio",
            "--indic-mio-top-p", "0.8",
            "--indic-mio-repetition-penalty", "1.1",
            "--indic-mio-global-embedding", embedding,
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.indicMioModelId, "org/indic-mio")
        XCTAssertEqual(speak.indicMioTopP, 0.8, accuracy: 0.001)
        XCTAssertEqual(speak.indicMioRepetitionPenalty, 1.1, accuracy: 0.001)
        XCTAssertNotNil(speak.indicMioGlobalEmbedding)
    }

    func testIndexTTS2Options() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "Hello",
            "--engine", "indextts2",
            "--voice-sample", "ref.wav",
            "--indextts2-model-id", "org/IndexTTS2-MLX-fp16",
            "--indextts2-bundle-dir", "/tmp/IndexTTS2-MLX-fp16",
            "--indextts2-emotion", "eager",
            "--indextts2-emotion-weight", "0.75",
            "--indextts2-speaking-rate", "1.15",
            "--indextts2-max-pause", "0.18",
            "--indextts2-s2mel-steps", "15",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.indextts2ModelId, "org/IndexTTS2-MLX-fp16")
        XCTAssertEqual(speak.indextts2BundleDir, "/tmp/IndexTTS2-MLX-fp16")
        XCTAssertEqual(speak.indextts2Emotion, "eager")
        XCTAssertEqual(speak.indextts2EmotionWeight, 0.75, accuracy: 0.001)
        XCTAssertEqual(speak.indextts2SpeakingRate, 1.15, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(speak.indextts2MaxPause), 0.18, accuracy: 0.001)
        XCTAssertEqual(speak.indextts2S2MelSteps, 15)
    }

    func testIndexTTS2EmotionVectorOption() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "Hello",
            "--engine", "indextts2",
            "--voice-sample", "ref.wav",
            "--indextts2-emotion", "0.65,0,0,0,0,0,0.15,0",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.indextts2Emotion, "0.65,0,0,0,0,0,0.15,0")
    }

    func testIndexTTS2EmotionAudioOption() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "Hello",
            "--engine", "indextts2",
            "--voice-sample", "ref.wav",
            "--indextts2-emotion-audio", "style.wav",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.indextts2EmotionAudio, "style.wav")
    }

    func testIndicMioAcceptsInlineJSONEmbedding() throws {
        let embedding = "[\(Array(repeating: "0", count: 128).joined(separator: ","))]"
        XCTAssertNoThrow(try AudioCLI.parseAsRoot([
            "speak", "नमस्ते <happy>",
            "--engine", "indic-mio",
            "--indic-mio-global-embedding", embedding,
        ]))
    }

    func testVerboseFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "hi", "--verbose"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertTrue(speak.verbose)
    }

    func testLanguageOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "Hallo", "--language", "german"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.language, "german")
    }

    func testBatchSizeOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--batch-file", "f.txt", "--batch-size", "8"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.batchSize, 8)
    }

    // MARK: Magpie engine — validate that voice-cloning / qwen3-specific
    // flags are rejected with a helpful error instead of silently ignored.
    // Magpie has 5 baked speakers and no zero-shot conditioning in the
    // model, so passing `--voice-sample` / `--speaker` / `--instruct`
    // would otherwise let the user think cloning had worked.

    // `parseAsRoot` runs `validate()` during parsing, so the error
    // surfaces from the parse call rather than from a separate validate().
    private func expectSpeakReject(_ args: [String], contains needle: String,
                                   file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(args), file: file, line: line) { err in
            XCTAssertTrue("\(err)".contains(needle),
                          "expected error containing '\(needle)', got: \(err)",
                          file: file, line: line)
        }
    }

    private func expectMagpieReject(_ args: [String], contains needle: String,
                                      file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try AudioCLI.parseAsRoot(args), file: file, line: line) { err in
            XCTAssertTrue("\(err)".contains(needle),
                          "expected error containing '\(needle)', got: \(err)",
                          file: file, line: line)
        }
    }

    // MARK: - Indic-Mio engine

    func testIndexTTS2RejectsMissingVoiceSample() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "indextts2"],
            contains: "--voice-sample")
    }

    func testIndexTTS2RejectsUnsupportedControls() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "indextts2", "--voice-sample", "ref.wav", "--stream"],
            contains: "--stream")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "indextts2", "--voice-sample", "ref.wav", "--speaker", "someone"],
            contains: "--speaker")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "indextts2", "--voice-sample", "ref.wav", "--instruct", "friendly"],
            contains: "--instruct")
        expectSpeakReject(
            [
                "speak", "Hello",
                "--engine", "indextts2",
                "--voice-sample", "ref.wav",
                "--indextts2-emotion", "eager",
                "--indextts2-emotion-audio", "style.wav",
            ],
            contains: "mutually exclusive")
        expectSpeakReject(
            [
                "speak", "Hello",
                "--engine", "indextts2",
                "--voice-sample", "ref.wav",
                "--indextts2-speaking-rate", "2.0",
            ],
            contains: "speakingRate")
        expectSpeakReject(
            [
                "speak", "Hello",
                "--engine", "indextts2",
                "--voice-sample", "ref.wav",
                "--indextts2-max-pause", "0.01",
            ],
            contains: "maxInternalPauseDuration")
    }

    // MARK: - F5-TTS engine

    func testF5Engine() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "--engine", "f5", "Hello",
            "--voice-sample", "ref.wav",
            "--f5-reference-text", "Reference transcript.",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "f5")
        XCTAssertEqual(speak.voiceSample, "ref.wav")
        XCTAssertEqual(speak.f5ReferenceText, "Reference transcript.")
        XCTAssertEqual(speak.f5ModelId, "aufklarer/F5TTS-v1-Base-MLX-fp16")
        XCTAssertNil(speak.f5BundleDir)
        XCTAssertEqual(speak.f5Steps, 16)
        XCTAssertEqual(speak.f5CfgStrength, 2.0, accuracy: 0.001)
        XCTAssertEqual(speak.f5Sway, -1.0, accuracy: 0.001)
        XCTAssertEqual(speak.f5Speed, 1.0, accuracy: 0.001)
        XCTAssertEqual(speak.f5Seed, 0)
        XCTAssertEqual(speak.f5TargetRMS, 0.1, accuracy: 0.001)
    }

    func testF5Options() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "Hello",
            "--engine", "f5",
            "--voice-sample", "ref.wav",
            "--f5-reference-text", "Reference transcript.",
            "--f5-model-id", "org/F5TTS-v1-Base-MLX-fp16",
            "--f5-bundle-dir", "/tmp/F5TTS-v1-Base-MLX-fp16",
            "--f5-steps", "16",
            "--f5-cfg-strength", "1.5",
            "--f5-sway=-0.8",
            "--f5-speed", "1.25",
            "--f5-seed", "42",
            "--f5-target-rms", "0.12",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.f5ModelId, "org/F5TTS-v1-Base-MLX-fp16")
        XCTAssertEqual(speak.f5BundleDir, "/tmp/F5TTS-v1-Base-MLX-fp16")
        XCTAssertEqual(speak.f5Steps, 16)
        XCTAssertEqual(speak.f5CfgStrength, 1.5, accuracy: 0.001)
        XCTAssertEqual(speak.f5Sway, -0.8, accuracy: 0.001)
        XCTAssertEqual(speak.f5Speed, 1.25, accuracy: 0.001)
        XCTAssertEqual(speak.f5Seed, 42)
        XCTAssertEqual(speak.f5TargetRMS, 0.12, accuracy: 0.001)
    }

    func testF5RejectsMissingVoiceSample() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "f5", "--f5-reference-text", "hi"],
            contains: "--voice-sample")
    }

    func testF5RejectsMissingOrBlankReferenceText() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav"],
            contains: "--f5-reference-text")
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "   ",
            ],
            contains: "--f5-reference-text")
    }

    func testF5RejectsUnsupportedControls() {
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--stream",
            ],
            contains: "--stream")
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--speaker", "someone",
            ],
            contains: "--speaker")
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--instruct", "friendly",
            ],
            contains: "--instruct")
        expectSpeakReject(
            [
                "speak", "--engine", "f5", "--batch-file", "texts.txt",
                "--voice-sample", "ref.wav", "--f5-reference-text", "hi",
            ],
            contains: "single text")
    }

    func testF5RejectsInvalidSamplingValues() {
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--f5-steps", "0",
            ],
            contains: "steps")
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--f5-speed", "0",
            ],
            contains: "speed")
        expectSpeakReject(
            [
                "speak", "Hello", "--engine", "f5", "--voice-sample", "ref.wav",
                "--f5-reference-text", "hi", "--f5-cfg-strength=-1",
            ],
            contains: "cfgStrength")
    }

    // MARK: - Higgs engine

    func testHiggsEngine() throws {
        let cmd = try AudioCLI.parseAsRoot(["speak", "--engine", "higgs", "Hello"])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.engine, "higgs")
        XCTAssertEqual(speak.higgsModelId, "aufklarer/Higgs-TTS-3-4B-MLX-bf16")
        XCTAssertNil(speak.higgsBundleDir)
        XCTAssertNil(speak.higgsRefText)
        XCTAssertEqual(speak.higgsTemperature, 0.8, accuracy: 0.001)
        XCTAssertNil(speak.higgsTopP)
        XCTAssertNil(speak.higgsTopK)
        XCTAssertEqual(speak.higgsMaxNewTokens, 2048)
        XCTAssertEqual(speak.higgsSeed, 0)
    }

    func testHiggsOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "<|emotion:elation|>Hello!",
            "--engine", "higgs",
            "--voice-sample", "ref.wav",
            "--higgs-ref-text", "Reference transcript.",
            "--higgs-model-id", "org/Higgs-TTS-3-4B-MLX-bf16",
            "--higgs-bundle-dir", "/tmp/Higgs-TTS-3-4B-MLX-bf16",
            "--higgs-temperature", "1.0",
            "--higgs-top-p", "0.95",
            "--higgs-top-k", "50",
            "--higgs-max-new-tokens", "512",
            "--higgs-seed", "42",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.voiceSample, "ref.wav")
        XCTAssertEqual(speak.higgsRefText, "Reference transcript.")
        XCTAssertEqual(speak.higgsModelId, "org/Higgs-TTS-3-4B-MLX-bf16")
        XCTAssertEqual(speak.higgsBundleDir, "/tmp/Higgs-TTS-3-4B-MLX-bf16")
        XCTAssertEqual(speak.higgsTemperature, 1.0, accuracy: 0.001)
        XCTAssertEqual(try XCTUnwrap(speak.higgsTopP), 0.95, accuracy: 0.001)
        XCTAssertEqual(speak.higgsTopK, 50)
        XCTAssertEqual(speak.higgsMaxNewTokens, 512)
        XCTAssertEqual(speak.higgsSeed, 42)
    }

    func testHiggsAllowsPlainTTSWithoutReference() throws {
        XCTAssertNoThrow(try AudioCLI.parseAsRoot(["speak", "Hello", "--engine", "higgs"]))
    }

    func testHiggsRejectsRefTextWithoutVoiceSample() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--higgs-ref-text", "hi"],
            contains: "--voice-sample")
    }

    func testHiggsRejectsUnsupportedControls() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--stream"],
            contains: "--stream")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--speaker", "someone"],
            contains: "--speaker")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--instruct", "friendly"],
            contains: "--instruct")
        expectSpeakReject(
            ["speak", "--engine", "higgs", "--batch-file", "texts.txt"],
            contains: "single text")
    }

    func testHiggsRejectsInvalidSamplingValues() {
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--higgs-temperature=-1"],
            contains: "temperature")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--higgs-top-p", "1.5"],
            contains: "topP")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--higgs-top-k", "0"],
            contains: "topK")
        expectSpeakReject(
            ["speak", "Hello", "--engine", "higgs", "--higgs-max-new-tokens", "0"],
            contains: "maxNewTokens")
    }

    func testIndicMioAcceptsVoiceSampleReference() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "speak", "नमस्ते <happy>",
            "--engine", "indic-mio",
            "--voice-sample", "ref.wav",
        ])
        let speak = try XCTUnwrap(cmd as? SpeakCommand)
        XCTAssertEqual(speak.voiceSample, "ref.wav")
    }

    func testIndicMioRejectsVoiceSampleWithExplicitEmbedding() {
        let embedding = Array(repeating: "0", count: 128).joined(separator: ",")
        expectSpeakReject(
            [
                "speak", "नमस्ते <happy>",
                "--engine", "indic-mio",
                "--voice-sample", "ref.wav",
                "--indic-mio-global-embedding", embedding,
            ],
            contains: "either --voice-sample or --indic-mio-global-embedding")
    }

    func testIndicMioRejectsQwen3StyleControls() {
        expectSpeakReject(
            ["speak", "नमस्ते <happy>", "--engine", "indic-mio", "--speaker", "someone"],
            contains: "--speaker")
        expectSpeakReject(
            ["speak", "नमस्ते <happy>", "--engine", "indic-mio", "--instruct", "be sad"],
            contains: "--instruct")
    }

    func testIndicMioRejectsStreamingAndBatch() {
        expectSpeakReject(
            ["speak", "नमस्ते <happy>", "--engine", "indic-mio", "--stream"],
            contains: "--stream")
        expectSpeakReject(
            ["speak", "--engine", "indic-mio", "--batch-file", "texts.txt"],
            contains: "single text")
    }

    func testMagpieRejectsVoiceSample() {
        expectMagpieReject(
            ["speak", "hi", "--engine", "magpie", "--voice-sample", "ref.wav"],
            contains: "--voice-sample")
    }

    func testMagpieRejectsQwen3SpeakerFlag() {
        expectMagpieReject(
            ["speak", "hi", "--engine", "magpie", "--speaker", "someone"],
            contains: "--speaker")
    }

    func testMagpieRejectsInstruct() {
        expectMagpieReject(
            ["speak", "hi", "--engine", "magpie", "--instruct", "be friendly"],
            contains: "--instruct")
    }

    func testMagpieAcceptsBakedSpeakers() throws {
        for spk in ["sofia", "aria", "jason", "leo", "john"] {
            XCTAssertNoThrow(try AudioCLI.parseAsRoot(
                ["speak", "hi", "--engine", "magpie", "--magpie-speaker", spk]),
                             "speaker \(spk) should validate")
        }
    }

    func testMagpieRejectsUnknownSpeaker() {
        expectMagpieReject(
            ["speak", "hi", "--engine", "magpie", "--magpie-speaker", "elvis"],
            contains: "--magpie-speaker")
    }

    // MARK: - Magpie CoreML engine

    func testMagpieCoreMLAccepts() throws {
        XCTAssertNoThrow(try AudioCLI.parseAsRoot(
            ["speak", "hi", "--engine", "magpie-coreml", "--magpie-speaker", "aria"]))
    }

    func testMagpieCoreMLAcceptsStream() throws {
        // Streaming is supported via the dedicated 8-frame nanocodec
        // model. Validation must accept --stream now (used to be
        // rejected when only the 64-frame batch codec shipped).
        XCTAssertNoThrow(try AudioCLI.parseAsRoot(
            ["speak", "hi", "--engine", "magpie-coreml", "--stream"]))
    }

    func testMagpieCoreMLRejectsVoiceCloningFlags() {
        // Same five baked speakers as the MLX engine; reject the same set
        // of cross-engine flags with the same actionable error.
        for (flag, value) in [("--voice-sample", "ref.wav"),
                              ("--speaker",      "someone"),
                              ("--instruct",     "be friendly")] {
            expectMagpieReject(
                ["speak", "hi", "--engine", "magpie-coreml", flag, value],
                contains: flag)
        }
    }

    func testMagpieCoreMLAllSpeakers() throws {
        // The CoreML bundle uses a different speaker index ordering than the
        // MLX bundle (John=0 vs Sofia=0); the CLI name → enum lookup must
        // work for all five identities.
        for spk in ["sofia", "aria", "jason", "leo", "john"] {
            XCTAssertNoThrow(try AudioCLI.parseAsRoot(
                ["speak", "hi", "--engine", "magpie-coreml", "--magpie-speaker", spk]),
                             "coreml speaker \(spk) should validate")
        }
    }
}

// MARK: - RespondCommand

final class RespondCommandTests: XCTestCase {

    // MARK: Defaults

    func testDefaultValues() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--input", "user.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
        XCTAssertEqual(respond.output, "response.wav")
        XCTAssertEqual(respond.voice, "NATM0")
        XCTAssertEqual(respond.systemPrompt, "assistant")
        XCTAssertEqual(respond.maxSteps, 200)
        XCTAssertEqual(respond.modelId, "aufklarer/PersonaPlex-7B-MLX-4bit")
        XCTAssertFalse(respond.stream)
        XCTAssertEqual(respond.chunkFrames, 25)
        XCTAssertFalse(respond.compile)
        XCTAssertFalse(respond.listVoices)
        XCTAssertFalse(respond.listPrompts)
        XCTAssertFalse(respond.verbose)
    }

    // MARK: Input parsing

    func testInputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "user.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
    }

    func testOutputShortFlag() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "user.wav", "-o", "out.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.output, "out.wav")
    }

    // MARK: All options

    func testAllOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "respond",
            "--input", "user.wav",
            "--output", "out.wav",
            "--voice", "NATF1",
            "--system-prompt", "teacher",
            "--max-steps", "250",
            "--model-id", "my/custom-model",
            "--stream",
            "--chunk-frames", "50",
            "--compile",
            "--verbose"
        ])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.input, "user.wav")
        XCTAssertEqual(respond.output, "out.wav")
        XCTAssertEqual(respond.voice, "NATF1")
        XCTAssertEqual(respond.systemPrompt, "teacher")
        XCTAssertEqual(respond.maxSteps, 250)
        XCTAssertEqual(respond.modelId, "my/custom-model")
        XCTAssertTrue(respond.stream)
        XCTAssertEqual(respond.chunkFrames, 50)
        XCTAssertTrue(respond.compile)
        XCTAssertTrue(respond.verbose)
    }

    // MARK: List flags

    func testListVoices() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--list-voices"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertTrue(respond.listVoices)
    }

    func testListPrompts() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "--list-prompts"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertTrue(respond.listPrompts)
    }

    // MARK: Voice presets

    func testVoiceOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav", "--voice", "VARF0"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.voice, "VARF0")
    }

    func testSystemPromptOption() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav", "--system-prompt", "customer-service"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.systemPrompt, "customer-service")
    }

    // MARK: Sampling overrides

    func testSamplingOverrides() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "respond", "-i", "a.wav",
            "--audio-temp", "0.5",
            "--text-temp", "0.3",
            "--audio-top-k", "100",
            "--repetition-penalty", "1.5",
            "--text-repetition-penalty", "1.8",
            "--repetition-window", "20",
            "--silence-early-stop", "10"
        ])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.audioTemp, 0.5)
        XCTAssertEqual(respond.textTemp, 0.3)
        XCTAssertEqual(respond.audioTopK, 100)
        XCTAssertEqual(respond.repetitionPenalty, 1.5)
        XCTAssertEqual(respond.textRepetitionPenalty, 1.8)
        XCTAssertEqual(respond.repetitionWindow, 20)
        XCTAssertEqual(respond.silenceEarlyStop, 10)
    }

    func testEntropyOptions() throws {
        let cmd = try AudioCLI.parseAsRoot([
            "respond", "-i", "a.wav",
            "--entropy-threshold", "1.5",
            "--entropy-window", "5"
        ])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertEqual(respond.entropyThreshold, 1.5)
        XCTAssertEqual(respond.entropyWindow, 5)
    }

    func testEntropyOptionsDefaultNil() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertNil(respond.entropyThreshold)
        XCTAssertNil(respond.entropyWindow)
    }

    func testTranscriptAndJsonFlags() throws {
        let cmd = try AudioCLI.parseAsRoot(["respond", "-i", "a.wav", "--transcript", "--json"])
        let respond = try XCTUnwrap(cmd as? RespondCommand)
        XCTAssertTrue(respond.transcript)
        XCTAssertTrue(respond.json)
    }
}

// MARK: - Utility Functions

final class UtilityTests: XCTestCase {

    func testResolveASRModelId_06B() {
        XCTAssertTrue(resolveASRModelId("0.6b").contains("0.6B"))
        XCTAssertTrue(resolveASRModelId("0.6B").contains("0.6B"))
        XCTAssertTrue(resolveASRModelId("small").contains("0.6B"))
    }

    func testResolveASRModelId_17B() {
        XCTAssertTrue(resolveASRModelId("1.7b").contains("1.7B"))
        XCTAssertTrue(resolveASRModelId("1.7B").contains("1.7B"))
        XCTAssertTrue(resolveASRModelId("large").contains("1.7B"))
    }

    func testResolveASRModelId_8bit() {
        let small8 = resolveASRModelId("0.6B-8bit")
        XCTAssertTrue(small8.contains("0.6B"))
        XCTAssertTrue(small8.contains("8bit"))

        let small8alt = resolveASRModelId("small-8bit")
        XCTAssertTrue(small8alt.contains("0.6B"))
        XCTAssertTrue(small8alt.contains("8bit"))

        let large4 = resolveASRModelId("1.7B-4bit")
        XCTAssertTrue(large4.contains("1.7B"))
        XCTAssertTrue(large4.contains("4bit"))

        let large4alt = resolveASRModelId("large-4bit")
        XCTAssertTrue(large4alt.contains("1.7B"))
        XCTAssertTrue(large4alt.contains("4bit"))
    }

    func testResolveASRModelId_passthrough() {
        XCTAssertEqual(resolveASRModelId("org/custom-model"), "org/custom-model")
        XCTAssertEqual(resolveASRModelId("aufklarer/Qwen3-ASR-0.6B-MLX-4bit"), "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
    }

    func testFormatDuration() {
        XCTAssertEqual(formatDuration(24000), "1.00")
        XCTAssertEqual(formatDuration(48000), "2.00")
        XCTAssertEqual(formatDuration(12000), "0.50")
        XCTAssertEqual(formatDuration(0), "0.00")
    }

    func testFormatDurationCustomSampleRate() {
        XCTAssertEqual(formatDuration(16000, sampleRate: 16000), "1.00")
        XCTAssertEqual(formatDuration(44100, sampleRate: 44100), "1.00")
    }
}
