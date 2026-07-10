import AudioCommon
@testable import HiggsTTS
@testable import Qwen3ASR
import XCTest

final class E2EHiggsTTSTests: XCTestCase {
    private struct ParityFixture: Decodable {
        let text: String
        let prompt_token_ids: [Int32]
        let delayed_rows: [[Int32]]
        let teacher_forced_logits: [[[Float]]]
        let audio_samples: Int
        let audio_rms: Float
        let audio_file: String
    }

    /// Voice cloning through the Swift LM + decoder using precomputed
    /// reference codes (the Swift encoder path lands separately).
    func testCloneFromPrecomputedReferenceCodes() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["HIGGS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set HIGGS_E2E_BUNDLE to run Higgs E2E")
        }
        guard let codesPath = env["HIGGS_REF_CODES"], !codesPath.isEmpty else {
            throw XCTSkip("Set HIGGS_REF_CODES to run the clone test")
        }
        guard let refText = env["HIGGS_REF_TEXT"], !refText.isEmpty else {
            throw XCTSkip("Set HIGGS_REF_TEXT to run the clone test")
        }
        guard let wavPath = env["HIGGS_CLONE_WAV"], !wavPath.isEmpty else {
            throw XCTSkip("Set HIGGS_CLONE_WAV to run the clone test")
        }
        struct Codes: Decodable { let delayed_codes: [[Int32]] }
        let codes = try JSONDecoder().decode(
            Codes.self, from: Data(contentsOf: URL(fileURLWithPath: codesPath)))

        let model = try await HiggsTTSModel.fromBundle(URL(fileURLWithPath: bundlePath))
        let text = env["HIGGS_CLONE_TEXT"]
            ?? "This is my cloned voice speaking through the native Swift Higgs port."
        let reference = HiggsTTSReference(delayedCodes: codes.delayed_codes, text: refText)
        let started = Date()
        let audio = try model.generate(
            text: text,
            references: [reference],
            options: try HiggsTTSSynthesisOptions(temperature: 0.8, seed: 0)) { progress, message in
                print("[Higgs] \(Int(progress * 100))% \(message)")
            }
        let elapsed = Date().timeIntervalSince(started)
        XCTAssertGreaterThan(audio.count, model.sampleRate / 2)
        let duration = Double(audio.count) / Double(model.sampleRate)
        print(String(format: "[Higgs] clone duration %.2fs elapsed %.2fs RTF %.3f",
                     duration, elapsed, elapsed / max(duration, 0.001)))
        try AudioCommon.WAVWriter.write(
            samples: audio, sampleRate: model.sampleRate, to: URL(fileURLWithPath: wavPath))
        print("[Higgs] saved \(wavPath)")
    }

    /// Fully native cloning: Swift codec encoder on the reference WAV, Swift
    /// LM + decoder for synthesis. Reports code agreement against the
    /// reference implementation's encoding when provided.
    func testCloneFromReferenceWav() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["HIGGS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set HIGGS_E2E_BUNDLE to run Higgs E2E")
        }
        guard let refWav = env["HIGGS_REF_WAV"], !refWav.isEmpty else {
            throw XCTSkip("Set HIGGS_REF_WAV to run the native clone test")
        }
        guard let refText = env["HIGGS_REF_TEXT"], !refText.isEmpty else {
            throw XCTSkip("Set HIGGS_REF_TEXT to run the native clone test")
        }
        let model = try await HiggsTTSModel.fromBundle(URL(fileURLWithPath: bundlePath))

        let started = Date()
        let reference = try model.encodeReference(
            audio: URL(fileURLWithPath: refWav), text: refText)
        print("[Higgs] encoded reference: \(reference.delayedCodes.count) delayed rows "
              + "in \(String(format: "%.2f", Date().timeIntervalSince(started)))s")

        if let codesPath = env["HIGGS_REF_CODES"], !codesPath.isEmpty {
            struct Codes: Decodable { let delayed_codes: [[Int32]] }
            let referenceCodes = try JSONDecoder().decode(
                Codes.self, from: Data(contentsOf: URL(fileURLWithPath: codesPath)))
            XCTAssertEqual(reference.delayedCodes.count, referenceCodes.delayed_codes.count,
                           "encoded frame count diverged")
            // Residual quantization cascades: one near-tie flip in an early
            // codebook decorrelates all later books for that frame, so only
            // codebook 0 is a meaningful numeric-agreement signal. The
            // functional gate is the ASR roundtrip below.
            var perBook = [Int](repeating: 0, count: model.config.audioNumCodebooks)
            var rows = 0
            for (swift, ref) in zip(reference.delayedCodes, referenceCodes.delayed_codes) {
                rows += 1
                for book in 0..<model.config.audioNumCodebooks where swift[book] == ref[book] {
                    perBook[book] += 1
                }
            }
            let percentages = perBook.map { String(format: "%.0f%%", Double($0) / Double(max(rows, 1)) * 100) }
            print("[Higgs] encoder agreement per codebook: \(percentages.joined(separator: " "))")
            XCTAssertGreaterThanOrEqual(
                Double(perBook[0]) / Double(max(rows, 1)), 0.75,
                "codebook 0 disagrees with the reference beyond near-tie tolerance")
        }

        let text = env["HIGGS_CLONE_TEXT"]
            ?? "The entire voice cloning pipeline now runs natively in Swift."
        let audio = try model.generate(
            text: text,
            references: [reference],
            options: try HiggsTTSSynthesisOptions(temperature: 0.8, seed: 0))
        XCTAssertGreaterThan(audio.count, model.sampleRate / 2)
        if let wavPath = env["HIGGS_CLONE_WAV"], !wavPath.isEmpty {
            try AudioCommon.WAVWriter.write(
                samples: audio, sampleRate: model.sampleRate, to: URL(fileURLWithPath: wavPath))
            print("[Higgs] saved \(wavPath)")
        }

        guard env["HIGGS_ASR_E2E"] == "1" else { return }
        let asr = try await Qwen3ASRModel.fromPretrained()
        let transcription = asr.transcribe(
            audio: audio, sampleRate: model.sampleRate, language: "english")
        print("[Higgs] clone ASR: \(transcription)")
        func words(_ text: String) -> Set<String> {
            Set(text.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty })
        }
        let expected = words(text)
        let overlap = Double(expected.intersection(words(transcription)).count)
            / Double(max(expected.count, 1))
        XCTAssertGreaterThan(overlap, 0.55, "ASR should recover the cloned English content")
    }

    /// Native cloned-voice roundtrips in English and Mandarin, mirroring the
    /// F5 gate: synthesize with the Swift pipeline, transcribe with Qwen3-ASR,
    /// and score lexical overlap / CER.
    func testNativeCloneRoundtripsEnglishAndMandarin() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["HIGGS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set HIGGS_E2E_BUNDLE to run Higgs E2E")
        }
        guard let refWav = env["HIGGS_REF_WAV"], !refWav.isEmpty else {
            throw XCTSkip("Set HIGGS_REF_WAV to run the multilingual clone gate")
        }
        guard env["HIGGS_ASR_E2E"] == "1" else {
            throw XCTSkip("Set HIGGS_ASR_E2E=1 to run the multilingual clone gate")
        }
        let refText = env["HIGGS_REF_TEXT"]

        let model = try await HiggsTTSModel.fromBundle(URL(fileURLWithPath: bundlePath))
        let reference = try model.encodeReference(
            audio: URL(fileURLWithPath: refWav), text: refText)
        let asr = try await Qwen3ASRModel.fromPretrained()

        let cases: [(name: String, language: String, cer: Bool, text: String)] = [
            ("en", "english", false,
             "This is a short Higgs voice cloning test running natively on Apple Silicon."),
            ("zh", "zh", true,
             "你好，这是一个在苹果芯片上本地运行的语音克隆测试。"),
        ]
        for testCase in cases {
            let started = Date()
            let audio = try model.generate(
                text: testCase.text,
                references: [reference],
                options: try HiggsTTSSynthesisOptions(temperature: 0.8, seed: 0))
            let elapsed = Date().timeIntervalSince(started)
            XCTAssertGreaterThan(audio.count, model.sampleRate / 2)
            let duration = Double(audio.count) / Double(model.sampleRate)

            let transcription = asr.transcribe(
                audio: audio, sampleRate: model.sampleRate, language: testCase.language)
            if testCase.cer {
                let cer = characterErrorRate(expected: testCase.text, actual: transcription)
                print(String(format: "[Higgs] %@ RTF %.2f CER %.3f ASR: %@",
                             testCase.name, elapsed / max(duration, 0.001), cer, transcription))
                XCTAssertLessThanOrEqual(cer, 0.25, "\(testCase.name) roundtrip failed")
            } else {
                let overlap = lexicalOverlap(expected: testCase.text, actual: transcription)
                print(String(format: "[Higgs] %@ RTF %.2f overlap %.2f ASR: %@",
                             testCase.name, elapsed / max(duration, 0.001), overlap, transcription))
                XCTAssertGreaterThan(overlap, 0.55, "\(testCase.name) roundtrip failed")
            }
            if let dir = env["HIGGS_GATE_WAV_DIR"], !dir.isEmpty {
                try? FileManager.default.createDirectory(
                    atPath: dir, withIntermediateDirectories: true)
                try AudioCommon.WAVWriter.write(
                    samples: audio, sampleRate: model.sampleRate,
                    to: URL(fileURLWithPath: dir).appendingPathComponent("higgs-\(testCase.name).wav"))
            }
        }
    }

    private func lexicalOverlap(expected: String, actual: String) -> Double {
        func words(_ text: String) -> Set<String> {
            Set(text.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { !$0.isEmpty })
        }
        let expectedWords = words(expected)
        guard !expectedWords.isEmpty else { return 0 }
        return Double(expectedWords.intersection(words(actual)).count) / Double(expectedWords.count)
    }

    private func characterErrorRate(expected: String, actual: String) -> Double {
        func content(_ text: String) -> [Character] {
            text.lowercased().filter { $0.isLetter || $0.isNumber }
        }
        let ref = content(expected)
        let hyp = content(actual)
        guard !ref.isEmpty else { return hyp.isEmpty ? 0 : 1 }
        var previous = Array(0...hyp.count)
        var current = [Int](repeating: 0, count: hyp.count + 1)
        for i in 1...ref.count {
            current[0] = i
            for j in 1...hyp.count {
                let substitution = previous[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1)
                current[j] = Swift.min(previous[j] + 1, current[j - 1] + 1, substitution)
            }
            swap(&previous, &current)
        }
        return Double(previous[hyp.count]) / Double(ref.count)
    }

    /// Compares tokenization, prompt layout, and greedy LM generation against
    /// a fixture dumped from the mlx-audio reference implementation. Greedy
    /// sampling makes the delayed rows deterministic, and a reference-free
    /// prompt keeps the codec out of the comparison on both sides.
    func testGreedyParityAgainstReferenceImplementation() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["HIGGS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set HIGGS_E2E_BUNDLE to run Higgs E2E")
        }
        guard let fixturePath = env["HIGGS_PARITY_FIXTURE"], !fixturePath.isEmpty else {
            throw XCTSkip("Set HIGGS_PARITY_FIXTURE to run Higgs LM parity")
        }
        let fixture = try JSONDecoder().decode(
            ParityFixture.self,
            from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))

        let model = try await HiggsTTSModel.fromBundle(URL(fileURLWithPath: bundlePath)) { progress, message in
            print("[Higgs] \(Int(progress * 100))% \(message)")
        }

        let builder = HiggsTTSPromptBuilder(specials: model.tokenizer.specials) { text in
            model.tokenizer.encode(text)
        }
        let prompt = builder.build(text: fixture.text)
        XCTAssertEqual(prompt.tokenIds, fixture.prompt_token_ids, "prompt tokenization diverged")

        // Teacher-forced numeric parity: identical inputs on both sides, so
        // the fused-head logits must agree to bf16 accumulation tolerance.
        let forcedRows = Array(fixture.delayed_rows.prefix(fixture.teacher_forced_logits.count))
        let swiftLogits = try model.teacherForcedLogits(text: fixture.text, forcedRows: forcedRows)
        XCTAssertEqual(swiftLogits.count, fixture.teacher_forced_logits.count)

        var maxAbsDiff: Float = 0
        var argmaxMatches = 0
        var argmaxTotal = 0
        let vocab = model.config.audioCodebookSize
        for (step, reference) in fixture.teacher_forced_logits.enumerated() {
            let flatReference = reference.flatMap { $0 }
            let swift = swiftLogits[step]
            XCTAssertEqual(swift.count, flatReference.count)
            for index in 0..<swift.count {
                maxAbsDiff = max(maxAbsDiff, abs(swift[index] - flatReference[index]))
            }
            for codebook in 0..<model.config.audioNumCodebooks {
                let range = (codebook * vocab)..<((codebook + 1) * vocab)
                let swiftArgmax = range.max { swift[$0] < swift[$1] }!
                let referenceArgmax = range.max { flatReference[$0] < flatReference[$1] }!
                argmaxTotal += 1
                if swiftArgmax == referenceArgmax { argmaxMatches += 1 }
            }
        }
        print("[Higgs] teacher-forced parity: maxAbsDiff \(maxAbsDiff), "
              + "argmax agreement \(argmaxMatches)/\(argmaxTotal)")
        XCTAssertLessThan(maxAbsDiff, 2.0, "logit divergence beyond bf16 tolerance")
        XCTAssertGreaterThanOrEqual(
            Double(argmaxMatches) / Double(argmaxTotal), 0.9,
            "greedy choices disagree beyond near-tie flips")

        // Free-running greedy generation as a smoke check: structural
        // validity plus similarity reporting against the reference rows.
        let started = Date()
        let rows = try model.generateDelayedCodes(
            text: fixture.text,
            options: try HiggsTTSSynthesisOptions(temperature: 0, maxNewTokens: 400)) { progress, message in
                print("[Higgs] \(Int(progress * 100))% \(message)")
            }
        let elapsed = Date().timeIntervalSince(started)
        print("[Higgs] generated \(rows.count) rows in \(String(format: "%.1f", elapsed))s "
              + "(\(String(format: "%.1f", Double(rows.count) / elapsed)) rows/s)")

        XCTAssertGreaterThan(rows.count, model.config.audioNumCodebooks)
        XCTAssertTrue(rows.allSatisfy { $0.count == model.config.audioNumCodebooks })
        XCTAssertTrue(rows.allSatisfy { row in
            row.allSatisfy { $0 >= 0 && $0 < Int32(model.config.audioCodebookSize) }
        })
        let limit = min(rows.count, fixture.delayed_rows.count)
        var firstDivergence = limit
        for index in 0..<limit where rows[index] != fixture.delayed_rows[index] {
            firstDivergence = index
            break
        }
        print("[Higgs] greedy rows: swift \(rows.count) vs reference \(fixture.delayed_rows.count), "
              + "first divergence at \(firstDivergence)")

        // Codec decode parity: identical codes in, so waveforms must agree to
        // float32 conv tolerance.
        let rawCodes = try HiggsTTSDelayPattern.reverse(
            fixture.delayed_rows, codebooks: model.config.audioNumCodebooks)
        let swiftAudio = try model.decodeCodes(rawCodes)
        XCTAssertEqual(swiftAudio.count, fixture.audio_samples, "decoded sample count diverged")

        let referenceData = try Data(contentsOf: URL(fileURLWithPath: fixture.audio_file))
        let referenceAudio = referenceData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        XCTAssertEqual(referenceAudio.count, fixture.audio_samples)

        var maxAudioDiff: Float = 0
        var sumSquares: Float = 0
        for index in 0..<min(swiftAudio.count, referenceAudio.count) {
            let diff = swiftAudio[index] - referenceAudio[index]
            maxAudioDiff = max(maxAudioDiff, abs(diff))
            sumSquares += diff * diff
        }
        let diffRMS = (sumSquares / Float(max(swiftAudio.count, 1))).squareRoot()
        let swiftRMS = (swiftAudio.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(swiftAudio.count, 1))).squareRoot()
        print("[Higgs] codec parity: maxAbsDiff \(maxAudioDiff), diffRMS \(diffRMS), "
              + "swiftRMS \(swiftRMS), referenceRMS \(fixture.audio_rms)")
        XCTAssertLessThan(diffRMS / max(fixture.audio_rms, 1e-6), 0.05,
                          "decoded waveform diverged beyond tolerance")

        if let wavPath = ProcessInfo.processInfo.environment["HIGGS_E2E_WAV"], !wavPath.isEmpty {
            let audio = try model.generate(
                text: fixture.text,
                options: try HiggsTTSSynthesisOptions(temperature: 0, maxNewTokens: 400))
            try AudioCommon.WAVWriter.write(
                samples: audio, sampleRate: model.sampleRate,
                to: URL(fileURLWithPath: wavPath))
            print("[Higgs] saved \(wavPath) (\(audio.count) samples)")
        }
    }
}
