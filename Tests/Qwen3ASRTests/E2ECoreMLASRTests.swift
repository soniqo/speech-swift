#if canImport(CoreML)
import XCTest
import Foundation
import AudioCommon
@testable import Qwen3ASR

/// End-to-end tests for the full CoreML ASR pipeline
/// (encoder + split decoder + tokenizer). Runs the real ANE-resident
/// bundle from ``aufklarer/Qwen3-ASR-CoreML``.
///
/// Regression coverage:
/// - Catches the "Cyrillic garbage" failure mode that previously only
///   reproduced through ``E2EMagpieCoreMLTests.testAsrTranscribeCapturedMagpieAudio``.
///   That coupling meant a broken CoreML decoder masqueraded as a Magpie
///   bug; this test exercises the CoreML ASR path directly against a
///   known-good English fixture so the next regression points straight
///   at Qwen3-ASR-CoreML.
/// - Verifies the **split-decoder** loader (``decoder_part1.mlmodelc``
///   + ``decoder_part2.mlmodelc`` chained via two ``MLState`` pools)
///   produces the expected text. Pre-split bundles fail to load because
///   ``findModel(named: "decoder_part1", ...)`` returns nil.
final class E2ECoreMLASRTests: XCTestCase {

    func testCoreMLTranscriptionEnglish() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav not found in Qwen3ASRTests resources")
        }

        let asr: CoreMLASRModel
        do {
            asr = try await CoreMLASRModel.fromPretrained { progress, status in
                if Int(progress * 100) % 25 == 0 {
                    print(String(format: "  loading: %.0f%% — %@", progress * 100, status))
                }
            }
        } catch {
            throw XCTSkip("CoreML ASR bundle unavailable: \(error)")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let targetSampleRate = 16000
        let audio: [Float]
        if sampleRate != targetSampleRate {
            audio = AudioFileLoader.resample(samples, from: sampleRate, to: targetSampleRate)
        } else {
            audio = samples
        }

        let start = CFAbsoluteTimeGetCurrent()
        let result = try asr.transcribe(audio: audio, sampleRate: targetSampleRate, language: "english")
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
        let audioMs = Double(audio.count) / Double(targetSampleRate) * 1000

        let normalised = result
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
        print("[COREML-ASR] raw=\"\(result)\"  normalised=\"\(normalised)\"")
        print(String(format: "[COREML-ASR-PERF] transcribe=%.0fms audio=%.0fms rtf=%.3f",
                     elapsedMs, audioMs, elapsedMs / audioMs))

        XCTAssertFalse(result.isEmpty, "Transcription should not be empty")
        // Fixture: "Can you guarantee that the replacement part will be shipped tomorrow?"
        for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
            XCTAssertTrue(normalised.contains(word),
                          "Missing expected word '\(word)' — raw=\"\(result)\"")
        }
    }
}
#endif
