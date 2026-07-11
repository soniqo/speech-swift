import AudioCommon
@testable import F5TTS
@testable import Qwen3ASR
import XCTest

final class E2EF5TTSTests: XCTestCase {
    func testLocalBundleSynthesizesEnglishCloneAndOptionalASRRoundTrip() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["F5TTS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set F5TTS_E2E_BUNDLE to run F5-TTS E2E")
        }
        guard let referencePath = env["F5TTS_REFERENCE_WAV"], !referencePath.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_WAV to run F5-TTS E2E")
        }
        guard let referenceText = env["F5TTS_REFERENCE_TEXT"], !referenceText.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_TEXT to run F5-TTS E2E")
        }

        let target = env["F5TTS_TARGET_TEXT"] ?? "This is a short local voice cloning test for speech swift."
        let model = try await F5TTSModel.fromBundle(URL(fileURLWithPath: bundlePath)) { progress, message in
            print("[F5-TTS] \(Int(progress * 100))% \(message)")
        }
        let started = Date()
        let waveform = try await model.generate(
            text: target,
            referenceAudio: URL(fileURLWithPath: referencePath),
            referenceText: referenceText,
            options: try F5TTSSynthesisOptions(speed: 1.0, seed: 0)) { progress, message in
                print("[F5-TTS] \(Int(progress * 100))% \(message)")
            }
        let elapsed = Date().timeIntervalSince(started)
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)

        let duration = Double(waveform.count) / Double(model.sampleRate)
        print(String(format: "[F5-TTS] duration %.2fs elapsed %.2fs RTF %.3f", duration, elapsed, elapsed / max(duration, 0.001)))

        if let wavPath = env["F5TTS_E2E_WAV"], !wavPath.isEmpty {
            try WAVWriter.write(samples: waveform, sampleRate: model.sampleRate, to: URL(fileURLWithPath: wavPath))
            print("[F5-TTS] saved \(wavPath)")
        }

        guard env["F5TTS_ASR_E2E"] == "1" else { return }
        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(audio: waveform, sampleRate: model.sampleRate, language: "english")
        print("[F5-TTS] ASR: \(transcription)")
        XCTAssertGreaterThan(
            lexicalOverlap(expected: target, actual: transcription),
            0.55,
            "ASR should recover the generated English content")
    }

    func testLocalBundleSynthesizesMandarinCloneAndOptionalASRRoundTrip() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["F5TTS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set F5TTS_E2E_BUNDLE to run F5-TTS E2E")
        }
        guard let referencePath = env["F5TTS_REFERENCE_WAV"], !referencePath.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_WAV to run F5-TTS E2E")
        }
        guard let referenceText = env["F5TTS_REFERENCE_TEXT"], !referenceText.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_TEXT to run F5-TTS E2E")
        }

        let target = env["F5TTS_ZH_TARGET_TEXT"] ?? "你好，这是一个在苹果芯片上本地运行的语音克隆测试。"
        let model = try await F5TTSModel.fromBundle(URL(fileURLWithPath: bundlePath)) { progress, message in
            print("[F5-TTS] \(Int(progress * 100))% \(message)")
        }
        guard model.tokenizer.pinyin != nil else {
            throw XCTSkip("Bundle has no pinyin_lexicon.tsv; Mandarin E2E needs the updated bundle")
        }

        let started = Date()
        let waveform = try await model.generate(
            text: target,
            referenceAudio: URL(fileURLWithPath: referencePath),
            referenceText: referenceText,
            options: try F5TTSSynthesisOptions(speed: 1.0, seed: 0)) { progress, message in
                print("[F5-TTS] \(Int(progress * 100))% \(message)")
            }
        let elapsed = Date().timeIntervalSince(started)
        assertUsableWaveform(waveform, sampleRate: model.sampleRate)

        let duration = Double(waveform.count) / Double(model.sampleRate)
        print(String(format: "[F5-TTS] zh duration %.2fs elapsed %.2fs RTF %.3f", duration, elapsed, elapsed / max(duration, 0.001)))

        if let wavPath = env["F5TTS_ZH_E2E_WAV"], !wavPath.isEmpty {
            try WAVWriter.write(samples: waveform, sampleRate: model.sampleRate, to: URL(fileURLWithPath: wavPath))
            print("[F5-TTS] saved \(wavPath)")
        }

        guard env["F5TTS_ASR_E2E"] == "1" else { return }
        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(audio: waveform, sampleRate: model.sampleRate, language: "zh")
        let cer = characterErrorRate(expected: target, actual: transcription)
        print("[F5-TTS] zh ASR: \(transcription) (CER \(String(format: "%.3f", cer)))")
        XCTAssertLessThanOrEqual(cer, 0.25, "ASR should recover the generated Mandarin content")
    }

    private func assertUsableWaveform(_ samples: [Float], sampleRate: Int) {
        XCTAssertGreaterThan(samples.count, sampleRate / 2)
        XCTAssertLessThan(samples.count, sampleRate * 30)
        XCTAssertFalse(samples.contains { !$0.isFinite })
        let peak = samples.map { abs($0) }.max() ?? 0
        XCTAssertGreaterThan(peak, 0.001)
        XCTAssertLessThanOrEqual(peak, 1.2)
    }

    private func lexicalOverlap(expected: String, actual: String) -> Double {
        let expectedWords = Set(words(expected))
        let actualWords = Set(words(actual))
        guard !expectedWords.isEmpty else { return 0 }
        return Double(expectedWords.intersection(actualWords).count) / Double(expectedWords.count)
    }

    private func words(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
    }

    private func characterErrorRate(expected: String, actual: String) -> Double {
        let ref = contentCharacters(expected)
        let hyp = contentCharacters(actual)
        guard !ref.isEmpty else { return hyp.isEmpty ? 0 : 1 }
        var previous = Array(0...hyp.count)
        var current = [Int](repeating: 0, count: hyp.count + 1)
        for i in 1...ref.count {
            current[0] = i
            for j in 1...hyp.count {
                let substitution = previous[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1)
                current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
            }
            swap(&previous, &current)
        }
        return Double(previous[hyp.count]) / Double(ref.count)
    }

    private func contentCharacters(_ text: String) -> [Character] {
        text.lowercased().filter { $0.isLetter || $0.isNumber }
    }
}
