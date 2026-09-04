#if canImport(CoreML)
import XCTest
import Foundation
import AudioCommon
import MLX
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

    // MARK: - MLX-free path

    /// Load a fixture at 16 kHz.
    private func loadFixture(_ name: String) throws -> [Float] {
        guard let url = Bundle.module.url(forResource: name, withExtension: "wav") else {
            throw XCTSkip("\(name).wav not found in Qwen3ASRTests resources")
        }
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: url)
        return sampleRate == 16000
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
    }

    private func loadModel() async throws -> CoreMLASRModel {
        do {
            return try await CoreMLASRModel.fromPretrained { _, _ in }
        } catch {
            throw XCTSkip("CoreML ASR bundle unavailable: \(error)")
        }
    }

    private func normalise(_ text: String) -> String {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .joined(separator: " ")
            .replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// The MLX-free path, which had no coverage at all until it shipped a
    /// SIGSEGV. It reads the encoder's Float16 `audio_embeddings` output;
    /// reading that as Float32 corrupted every embedding and ran off the
    /// end of the buffer past audio token 194 (~15 s).
    ///
    /// `transcribe()` and `transcribeWithoutMLX()` are two **independent**
    /// readings of the same encoder buffer — `multiArrayToMLXArray` and
    /// `copyRow`. Both widen Float16 exactly, feed the same CoreML models,
    /// and take the same greedy argmax, so they must agree character for
    /// character. Divergence means one of the two extraction routes has
    /// regressed, which is precisely the failure this bug was.
    func testMLXFreeTranscriptionMatchesMLXPath() async throws {
        let audio = try loadFixture("test_audio")
        let asr = try await loadModel()

        let viaMLX = try asr.transcribe(audio: audio, sampleRate: 16000, language: "english")
        let viaCoreMLOnly = try asr.transcribeWithoutMLX(
            audio: audio, sampleRate: 16000, language: "english")

        print("[COREML-ASR-NOMLX] mlx=\"\(viaMLX)\"  nomlx=\"\(viaCoreMLOnly)\"")

        XCTAssertFalse(viaCoreMLOnly.isEmpty, "MLX-free transcription should not be empty")
        XCTAssertEqual(viaCoreMLOnly, viaMLX,
                       "the two extraction routes read the same buffer and must agree")

        let normalised = normalise(viaCoreMLOnly)
        for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
            XCTAssertTrue(normalised.contains(word),
                          "Missing expected word '\(word)' — raw=\"\(viaCoreMLOnly)\"")
        }
    }

    /// `transcribeWithoutMLX` exists for one reason: to run where Metal
    /// eval would crash, i.e. iOS background execution. Nothing verified
    /// that it actually holds — and the doc comment on `transcribe()`
    /// claimed the same property while quietly round-tripping mel features
    /// and encoder output through MLXArray.
    ///
    /// This pins the invariant with MLX's own accounting: the MLX-free path
    /// must not move MLX's peak allocation at all, and `transcribe()` must,
    /// which is what makes the assertion meaningful rather than vacuous.
    func testOnlyTheMLXFreePathAvoidsMLXAllocation() async throws {
        let audio = try loadFixture("test_audio")
        let asr = try await loadModel()

        // Warm both paths first so one-off MLX setup isn't charged to the
        // measured run.
        _ = try asr.transcribe(audio: audio, sampleRate: 16000, language: "english")
        _ = try asr.transcribeWithoutMLX(audio: audio, sampleRate: 16000, language: "english")

        MLX.Memory.peakMemory = 0  // setter resets; newValue ignored
        let baseline = MLX.Memory.peakMemory
        _ = try asr.transcribeWithoutMLX(audio: audio, sampleRate: 16000, language: "english")
        let afterMLXFree = MLX.Memory.peakMemory

        MLX.Memory.peakMemory = 0
        _ = try asr.transcribe(audio: audio, sampleRate: 16000, language: "english")
        let afterMLXPath = MLX.Memory.peakMemory

        print("[COREML-ASR-MLXMEM] baseline=\(baseline) nomlx=\(afterMLXFree) mlx=\(afterMLXPath)")

        XCTAssertEqual(afterMLXFree, baseline,
                       "transcribeWithoutMLX allocated MLX memory — it is no longer MLX-free, "
                       + "and no longer safe in the background context it exists for")
        XCTAssertGreaterThan(afterMLXPath, baseline,
                             "transcribe() is expected to allocate through MLX; if it stopped, "
                             + "this test has lost its teeth and the comparison above proves nothing")
    }

    /// Drive the audio-token count to the very top of the encoder's fixed
    /// 390-token output so the extraction reads the last rows of the buffer.
    ///
    /// The 20 s fixture only reaches ~260 tokens. A ~29.5 s clip reaches
    /// ~384, leaving six rows of headroom — the strongest bounds assurance
    /// available against real weights, and far past the row-195 cliff where
    /// the old Float32 read left the allocation. A regression here faults or
    /// returns garbage rather than quietly passing.
    func testMLXFreePathAtEncoderTokenLimit() async throws {
        var audio = try loadFixture("test_audio")
        audio += try loadFixture("kokoro_continuous_stitched")

        // The encoder's fixed mel shape tops out at 30 s; stay just inside.
        let maxSamples = Int(29.5 * 16000)
        XCTAssertGreaterThan(audio.count, maxSamples,
                             "fixtures should concatenate to more than the cap so the trim is real")
        audio = Array(audio.prefix(maxSamples))

        let asr = try await loadModel()

        let viaMLX = try asr.transcribe(audio: audio, sampleRate: 16000, language: "english")
        let viaCoreMLOnly = try asr.transcribeWithoutMLX(
            audio: audio, sampleRate: 16000, language: "english")

        print("[COREML-ASR-LIMIT] nomlx=\"\(viaCoreMLOnly)\"")

        XCTAssertFalse(viaCoreMLOnly.isEmpty,
                       "a near-max-length clip should still transcribe")
        XCTAssertEqual(viaCoreMLOnly, viaMLX,
                       "the two extraction routes must agree at the top of the buffer too")

        // The first 20 s is the known fixture, so its content must survive
        // even though the tail is unrelated speech.
        let normalised = normalise(viaCoreMLOnly)
        for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
            XCTAssertTrue(normalised.contains(word),
                          "Missing expected word '\(word)' — raw=\"\(viaCoreMLOnly)\"")
        }
    }
}
#endif
