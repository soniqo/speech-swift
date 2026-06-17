import CoreML
import XCTest
@testable import ParakeetASR
@testable import AudioCommon

/// E2E coverage for the multi-encoder layout shipped by
/// `aufklarer/Parakeet-TDT-v3-CoreML-INT8` after the M5-ANE fix
/// (speech-models commit "parakeet-asr: bump CoreML deployment target to
/// iOS18 …"; issue #313).
///
/// The legacy `-INT8` repo can't ship EnumeratedShapes anymore — iOS18's
/// MIL validator rejects the dynamic `tile` reps the FastConformer
/// pad-mask emits when the time dim is symbolic. So the repo now ships
/// three single-shape encoder variants in one directory:
///
///     encoder.mlmodelc       # 3000 frames (30s) — default, backwards-compat
///     encoder_5s.mlmodelc    # 500 frames
///     encoder_15s.mlmodelc   # 1500 frames
///
/// Decoder, joint, config, vocab are shared. The Swift loader picks the
/// encoder by `encoderVariant:` ("5s", "15s", or nil = default).
///
/// These tests exercise each variant end-to-end:
/// - the named encoder loads on ANE
/// - `supportedMelLengths` matches the expected shape
/// - a real transcription on the bundled audio returns sensible text
final class E2EParakeetMultiEncoderTests: XCTestCase {

    static let legacyModelId = "aufklarer/Parakeet-TDT-v3-CoreML-INT8"

    /// Resolve the cache dir for the legacy `-INT8` repo. Once the
    /// iOS18 + multi-encoder build is uploaded to HF, the tests can
    /// fall back on the normal download path; until then we expect the
    /// layout to already exist locally (staged by the converter) and
    /// fail clearly if it's missing.
    private static func cachedRepoOrSkip() throws -> URL {
        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: legacyModelId)
        let defaultEnc = cacheDir.appendingPathComponent("encoder.mlmodelc")
        let enc5 = cacheDir.appendingPathComponent("encoder_5s.mlmodelc")
        let enc15 = cacheDir.appendingPathComponent("encoder_15s.mlmodelc")
        guard FileManager.default.fileExists(atPath: defaultEnc.path),
              FileManager.default.fileExists(atPath: enc5.path),
              FileManager.default.fileExists(atPath: enc15.path) else {
            throw XCTSkip(
                "Multi-encoder layout not staged at \(cacheDir.path) — " +
                "stage encoder_5s.mlmodelc + encoder_15s.mlmodelc locally, " +
                "or wait until the iOS18 build is uploaded to HF.")
        }
        return cacheDir
    }

    /// Default encoder (no variant) is the 30s single-shape — same shape
    /// as the dedicated `-30s` repo.
    func testDefaultEncoderIsSingleShape3000() async throws {
        let cacheDir = try Self.cachedRepoOrSkip()
        let model = try await ParakeetASRModel.fromPretrained(
            modelId: Self.legacyModelId, cacheDir: cacheDir, offlineMode: true)
        XCTAssertEqual(
            model.supportedMelLengths, [3000],
            "Default encoder.mlmodelc in the multi-variant repo must be the 30s single-shape (3000 frames)"
        )
    }

    func test5sEncoderVariantIsSingleShape500() async throws {
        let cacheDir = try Self.cachedRepoOrSkip()
        let model = try await ParakeetASRModel.fromPretrained(
            modelId: Self.legacyModelId, cacheDir: cacheDir, offlineMode: true,
            encoderVariant: "5s")
        XCTAssertEqual(
            model.supportedMelLengths, [500],
            "encoderVariant=5s must load encoder_5s.mlmodelc with 500-frame shape"
        )
    }

    func test15sEncoderVariantIsSingleShape1500() async throws {
        let cacheDir = try Self.cachedRepoOrSkip()
        let model = try await ParakeetASRModel.fromPretrained(
            modelId: Self.legacyModelId, cacheDir: cacheDir, offlineMode: true,
            encoderVariant: "15s")
        XCTAssertEqual(
            model.supportedMelLengths, [1500],
            "encoderVariant=15s must load encoder_15s.mlmodelc with 1500-frame shape"
        )
    }

    /// Smallest variant must still transcribe the bundled clip
    /// correctly — proves the variant selection plumbing actually points
    /// at a working encoder, not just one that loads with the wrong shape.
    func test5sVariantTranscribesShortClip() async throws {
        let cacheDir = try Self.cachedRepoOrSkip()
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test audio not in bundle resources")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let resampled = sampleRate == 16000
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
        // Bundled clip: 20 s of "Can you guarantee that the replacement
        // part will be shipped tomorrow?" with ~3 s silence at the start.
        // The same 4 s slice the shape-adaptation test uses fits the 5 s
        // encoder (500 frames).
        let startSample = 5 * 16000
        let endSample = min(resampled.count, 9 * 16000)
        let slice = Array(resampled[startSample..<endSample])

        let model = try await ParakeetASRModel.fromPretrained(
            modelId: Self.legacyModelId, cacheDir: cacheDir, offlineMode: true,
            encoderVariant: "5s")
        let text = try model.transcribeAudio(slice, sampleRate: 16000)

        let lower = text.lowercased()
        XCTAssertTrue(
            lower.contains("guarantee") || lower.contains("replacement") || lower.contains("shipped"),
            "Expected the bundled phrase via the 5s encoder variant, got: '\(text)'"
        )
    }
}
