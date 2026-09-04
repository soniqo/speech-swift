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

    /// `test_audio.wav` is 20 s but the sentence occupies only seconds
    /// 5–9; the rest is digital silence. Slicing a prefix therefore yields
    /// silence, not short speech. This returns just the spoken region so a
    /// clip of any target length can be built around known content.
    private func speechRegion(of fixture: [Float]) -> [Float] {
        let start = Int(4.5 * 16000)
        let end = min(Int(9.5 * 16000), fixture.count)
        return Array(fixture[start..<end])
    }

    /// The spoken region padded with silence to `seconds`. Trailing silence
    /// is what the fixture itself contains and what the encoder pads with,
    /// so the expected transcript is identical at every length — only the
    /// audio-token count changes.
    private func clip(_ seconds: Double, from fixture: [Float]) -> [Float] {
        let speech = speechRegion(of: fixture)
        let target = Int(seconds * 16000)
        if speech.count >= target { return Array(speech.prefix(target)) }
        return speech + [Float](repeating: 0, count: target - speech.count)
    }

    /// Bracket the row-195 cliff with real weights.
    ///
    /// The defect was length-dependent in a way that hid it: below ~15 s the
    /// bad read stayed inside the allocation and returned quiet garbage;
    /// above it, it left the buffer. A single fixture length cannot see
    /// that. Every clip here carries the *same* sentence and differs only in
    /// audio-token count, so the expected transcript is constant across the
    /// boundary — the two extraction routes must agree with each other and
    /// both must still contain the sentence.
    ///
    /// Pre-fix this fails on both counts: below the cliff the routes
    /// diverge, above it the MLX-free path returns empty or faults.
    func testExtractionAgreesAcrossTheLengthCliff() async throws {
        let base = try loadFixture("test_audio")
        let asr = try await loadModel()

        // Two lengths well below the ~15 s cliff, two above it.
        for seconds in [5.0, 12.0, 18.0, 25.0] {
            let audio = clip(seconds, from: base)
            XCTAssertEqual(Double(audio.count) / 16000.0, seconds, accuracy: 0.01)

            let viaMLX = try asr.transcribe(
                audio: audio, sampleRate: 16000, language: "english")
            let viaCoreMLOnly = try asr.transcribeWithoutMLX(
                audio: audio, sampleRate: 16000, language: "english")

            print("[COREML-ASR-SWEEP] \(seconds)s nomlx=\"\(viaCoreMLOnly)\"")

            XCTAssertEqual(viaCoreMLOnly, viaMLX,
                           "extraction routes disagree at \(seconds)s")

            let normalised = normalise(viaCoreMLOnly)
            for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
                XCTAssertTrue(normalised.contains(word),
                              "\(seconds)s clip lost '\(word)' — raw=\"\(viaCoreMLOnly)\"")
            }
        }
    }

    /// The reporter's actual workload: one model instance, consecutive
    /// segments in a loop. That exercises `resetCache()` and the decoder's
    /// KV position reset between calls, which nothing else covers — every
    /// other test transcribes once per instance.
    ///
    /// Interleaving lengths is the point. If state bled from a long segment
    /// into the next call, the two identical long clips either side of a
    /// short one would not produce identical text.
    func testRepeatedSegmentsOnOneInstanceDoNotBleed() async throws {
        let base = try loadFixture("test_audio")
        let asr = try await loadModel()

        let long = clip(24.0, from: base)
        let short = clip(5.0, from: base)

        let first = try asr.transcribeWithoutMLX(
            audio: long, sampleRate: 16000, language: "english")
        let interposed = try asr.transcribeWithoutMLX(
            audio: short, sampleRate: 16000, language: "english")
        let second = try asr.transcribeWithoutMLX(
            audio: long, sampleRate: 16000, language: "english")
        let interposedAgain = try asr.transcribeWithoutMLX(
            audio: short, sampleRate: 16000, language: "english")

        print("[COREML-ASR-REUSE] long=\"\(first)\" short=\"\(interposed)\"")

        XCTAssertEqual(first, second,
                       "the same clip gave different text after an intervening segment — "
                       + "decoder state is bleeding across calls")
        XCTAssertEqual(interposed, interposedAgain,
                       "the short clip is not reproducible across repeated use of one instance")

        for (label, text) in [("long", first), ("short", interposed)] {
            let normalised = normalise(text)
            for word in ["guarantee", "replacement", "shipped", "tomorrow"] {
                XCTAssertTrue(normalised.contains(word),
                              "\(label) clip lost '\(word)' after repeated use — raw=\"\(text)\"")
            }
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
