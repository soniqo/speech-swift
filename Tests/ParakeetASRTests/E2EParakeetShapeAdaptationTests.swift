import CoreML
import XCTest
@testable import ParakeetASR
@testable import AudioCommon

/// E2E coverage for encoder-input-shape adaptation: `supportedMelLengths`
/// comes from the encoder's CoreML constraint at load time, so every shipped
/// export works without per-variant Swift branching. These tests lock in:
///
/// - default model is a single fixed shape `[3000]` (30s, no EnumeratedShapes)
/// - iOS-5s variant produces the single fixed length `[500]`
/// - iOS-5s end-to-end transcribe of a real ≤5 s speech clip returns
///   sensible text
/// - a clip of exactly 5 s keeps its first mel frame
final class E2EParakeetShapeAdaptationTests: XCTestCase {

    static let defaultModelId = ParakeetASRModel.defaultModelId
    static let iOSModelId     = ParakeetASRModel.iosModelId

    func testDefaultModelDiscoversSingleShape3000() async throws {
        let model = try await ParakeetASRModel.fromPretrained(modelId: Self.defaultModelId)
        XCTAssertEqual(
            model.supportedMelLengths,
            [3000],
            "Default Parakeet encoder must be a single fixed shape [3000] (30s, no EnumeratedShapes)"
        )
    }

    func testIOSModelDiscoversSingleShape500() async throws {
        let model = try await ParakeetASRModel.fromPretrained(modelId: Self.iOSModelId)
        XCTAssertEqual(
            model.supportedMelLengths,
            [500],
            "iOS-5s Parakeet encoder must be detected as a single fixed shape [500]"
        )
    }

    /// Regression for the original symptom: with the iOS-5s single-shape
    /// model, transcribing a clip shorter than 5 s used to fail with a
    /// CoreML shape-mismatch error and silently return empty text. The
    /// fix should now produce real tokens.
    func testIOSModelTranscribesShortClip() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test audio not in bundle resources")
        }

        // Bundled clip is 20 s of "Can you guarantee that the replacement
        // part will be shipped tomorrow?" with ~3 s silence at the start.
        // We slice 4 s starting at +5 s so the chunk fits the iOS-5s
        // single-shape encoder *and* contains the spoken sentence.
        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let resampled = sampleRate == 16000
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
        let startSample = 5 * 16000
        let endSample   = min(resampled.count, 9 * 16000)
        XCTAssertGreaterThan(endSample, startSample, "Test audio shorter than 9 s — fixture changed?")
        let slice = Array(resampled[startSample..<endSample])

        let model = try await ParakeetASRModel.fromPretrained(modelId: Self.iOSModelId)
        let text = try model.transcribeAudio(slice, sampleRate: 16000)

        XCTAssertFalse(
            text.isEmpty,
            "iOS-5s model produced empty text — single-shape padding regressed"
        )
        let lower = text.lowercased()
        XCTAssertTrue(
            lower.contains("guarantee") || lower.contains("replacement") || lower.contains("shipped"),
            "Expected a recognisable phrase from the test sentence, got: '\(text)'"
        )
    }

    /// Regression: exactly 5 s of audio fills the iOS-5s window with 500
    /// valid mel frames, but mel extraction emits one extra zero frame.
    /// That frame used to push the window into the truncation path, which
    /// dropped the first real frame and moved word start times early. The
    /// words must match the same clip one sample shorter.
    func testIOSModelKeepsFirstFrameOfFullWindow() async throws {
        guard let wavURL = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("Test audio not in bundle resources")
        }

        let (samples, sampleRate) = try AudioFileLoader.loadWAV(url: wavURL)
        let resampled = sampleRate == 16000
            ? samples
            : AudioFileLoader.resample(samples, from: sampleRate, to: 16000)
        let startSample = 4 * 16000
        guard resampled.count >= startSample + 80000 else {
            return XCTFail("Test audio shorter than 9 s — fixture changed?")
        }

        let model = try await ParakeetASRModel.fromPretrained(modelId: Self.iOSModelId)
        func words(sampleCount: Int) throws -> [String] {
            let slice = Array(resampled[startSample..<(startSample + sampleCount)])
            _ = try model.transcribeAudio(slice, sampleRate: 16000, language: "en")
            return (model.lastWordConfidences ?? []).map { "\($0.word)@\($0.startTime ?? -1)" }
        }

        let fullWindow = try words(sampleCount: 80000)
        XCTAssertFalse(fullWindow.isEmpty, "iOS-5s model produced no words for a 5 s clip")
        XCTAssertEqual(
            fullWindow,
            try words(sampleCount: 79999),
            "A 5 s clip must keep its first mel frame, so its word starts match a clip one sample shorter"
        )
    }
}
