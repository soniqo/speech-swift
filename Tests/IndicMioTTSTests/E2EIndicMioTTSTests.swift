import XCTest
@testable import IndicMioTTS
import MLX
import Qwen3ASR

final class E2EIndicMioTTSTests: XCTestCase {
    override func tearDown() {
        super.tearDown()
        Memory.clearCache()
    }

    func testHindiEmotionTokenGenerationFromBundle() async throws {
        let model = try await loadModel()
        let text = "नमस्ते, आज हम खुश हैं। <happy>"
        let sampling = IndicMioSamplingConfig(
            maxNewTokens: 96,
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1)

        let tokens = try await model.generateSpeechTokens(
            text: text,
            language: "hindi",
            sampling: sampling)

        XCTAssertFalse(tokens.isEmpty, "Indic-Mio should emit MioCodec content tokens")
        XCTAssertLessThanOrEqual(tokens.count, sampling.maxNewTokens)
        XCTAssertTrue(tokens.allSatisfy { $0 >= 0 && $0 < MioCodecFSQ.codebookSize })

        let embeddings = try model.decodeContentEmbeddings(contentTokens: tokens)
        XCTAssertEqual(embeddings.dim(0), 1)
        XCTAssertEqual(embeddings.dim(1), tokens.count)
        XCTAssertEqual(embeddings.dim(2), MioCodecConfig.default.contentEmbeddingDim)

        let values = Array(embeddings.asType(.float32).asArray(Float.self).prefix(4096))
        XCTAssertFalse(values.isEmpty)
        XCTAssertTrue(values.allSatisfy(\.isFinite), "Content embeddings must not contain NaN/Inf")
        XCTAssertGreaterThan(values.map { abs($0) }.max() ?? 0, 0.0001)

        let waveform = try model.decodeWaveform(contentTokens: tokens)
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)

        let generated = try await model.generate(
            text: "यह एक छोटा परीक्षण है। <sad>",
            language: "hindi",
            sampling: sampling)
        assertUsableWaveform(generated, sampleRate: model.sampleRate)

    }

    func testRawReferenceAudioSynthesizesFromPublishedWavLMCompanion() async throws {
        let model = try await loadModel()
        let reference = sineReference(sampleRate: model.sampleRate, seconds: 1.0)
        let embedding = try await model.extractGlobalEmbedding(
            referenceAudio: reference,
            referenceSampleRate: model.sampleRate)

        XCTAssertEqual(embedding.count, MioCodecConfig.default.globalEmbeddingDim)
        XCTAssertTrue(embedding.allSatisfy(\.isFinite))
        XCTAssertGreaterThan(embedding.map { abs($0) }.max() ?? 0, 0.0001)

        let sampling = IndicMioSamplingConfig(
            maxNewTokens: 64,
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1)
        let waveform = try await model.generate(
            text: "नमस्ते, यह संदर्भ आवाज़ है। <happy>",
            language: "hindi",
            referenceAudio: reference,
            referenceSampleRate: model.sampleRate,
            sampling: sampling)
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)
    }

    func testHindiWaveformRoundTripWithASR() async throws {
        let model = try await loadModel()
        let sampling = IndicMioSamplingConfig(
            maxNewTokens: 128,
            temperature: 0,
            topK: 1,
            topP: 1,
            repetitionPenalty: 1)

        let target = "नमस्ते, आज मौसम अच्छा है। <happy>"
        let waveform = try await model.generate(
            text: target,
            language: "hindi",
            sampling: sampling)
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)

        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(audio: waveform, sampleRate: model.sampleRate)
        print("Indic-Mio Hindi roundtrip: \(target) -> \(transcription)")

        let expectedKeywords = ["नमस्ते", "मौसम", "अच्छा"]
        let recovered = expectedKeywords.filter { transcription.contains($0) }
        XCTAssertTrue(
            recovered.count >= 2,
            "ASR should recover multiple Hindi keywords from synthesized speech: \(transcription)")
    }

    private func assertUsableWaveform(
        _ waveform: [Float],
        sampleRate: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertFalse(waveform.isEmpty, "Waveform should not be empty", file: file, line: line)
        XCTAssertGreaterThan(Double(waveform.count) / Double(sampleRate), 0.05, file: file, line: line)
        XCTAssertTrue(waveform.allSatisfy(\.isFinite), "Waveform must not contain NaN/Inf", file: file, line: line)
        XCTAssertGreaterThan(waveform.map { abs($0) }.max() ?? 0, 0.0001, file: file, line: line)
    }

    private func sineReference(sampleRate: Int, seconds: Double) -> [Float] {
        let count = Int(Double(sampleRate) * seconds)
        return (0..<count).map { i in
            Float(0.08 * sin(2.0 * Double.pi * 220.0 * Double(i) / Double(sampleRate)))
        }
    }

    private func loadModel() async throws -> IndicMioTTSModel {
        let env = ProcessInfo.processInfo.environment
        if let bundle = env["INDIC_MIO_E2E_BUNDLE"], !bundle.isEmpty {
            return try await IndicMioTTSModel.fromBundle(URL(fileURLWithPath: bundle, isDirectory: true))
        }

        do {
            return try await IndicMioTTSModel.fromPretrained()
        } catch {
            throw XCTSkip("Indic-Mio E2E bundle unavailable: \(error)")
        }
    }
}
