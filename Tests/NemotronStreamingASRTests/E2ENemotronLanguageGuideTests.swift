import XCTest
@testable import NemotronStreamingASR
import AudioCommon

/// Local-bundle integration tests. They do not download model artifacts.
final class E2ENemotronLanguageGuideTests: XCTestCase {
    private static var model: NemotronStreamingASRModel?

    override func setUp() async throws {
        try await super.setUp()
        if Self.model == nil {
            guard let path = ProcessInfo.processInfo.environment["NEMOTRON_35_LOCAL_BUNDLE"],
                !path.isEmpty, FileManager.default.fileExists(atPath: path)
            else { throw XCTSkip("Set NEMOTRON_35_LOCAL_BUNDLE to a local multilingual bundle") }
            Self.model = try await NemotronStreamingASRModel.fromLocal(bundleDir: URL(fileURLWithPath: path))
        }
    }

    private func audio() throws -> [Float] {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "test_audio", withExtension: "wav"))
        return try AudioFileLoader.load(url: url, targetSampleRate: 16000)
    }

    private func decode(
        _ samples: [Float], guide: LanguageGuideConfig? = nil
    ) throws -> NemotronStreamingASRModel.PartialTranscript {
        let model = try XCTUnwrap(Self.model)
        let session = try model.createSession(language: "auto", languageGuide: guide)
        var partials = try session.pushAudio(samples)
        partials.append(contentsOf: try session.finalize())
        return try XCTUnwrap(partials.last)
    }

    func testEmptyAndZeroGuidePreserveAutomaticOutputExactly() throws {
        let samples = try audio()
        let baseline = try decode(samples)
        XCTAssertFalse(baseline.text.isEmpty)
        for guide in [
            LanguageGuideConfig(expectedLanguages: []),
            LanguageGuideConfig(expectedLanguages: ["en-US"], expectedLanguageWeight: 0),
        ] {
            let candidate = try decode(samples, guide: guide)
            XCTAssertEqual(candidate.text, baseline.text)
            XCTAssertEqual(candidate.confidence, baseline.confidence)
            XCTAssertEqual(candidate.words, baseline.words)
            XCTAssertEqual(candidate.wordBoostingChangedDecisions, baseline.wordBoostingChangedDecisions)
        }
    }

    func testGuidedAndAutomaticSessionsOwnIndependentState() throws {
        let model = try XCTUnwrap(Self.model)
        guard model.supportedLanguageGuideLanguages.contains("en-US") else {
            throw XCTSkip("Local bundle has no en-US language-guide prompt")
        }
        let samples = try audio()
        let baseline = try decode(samples)
        let guide = LanguageGuideConfig(expectedLanguages: ["en-US"])
        let guidedBaseline = try decode(samples, guide: guide)
        let guided = try model.createSession(
            language: "auto", languageGuide: guide)
        let automatic = try model.createSession(language: "auto")
        var guidedPartials: [NemotronStreamingASRModel.PartialTranscript] = []
        var automaticPartials: [NemotronStreamingASRModel.PartialTranscript] = []
        for lower in stride(from: 0, to: samples.count, by: 5120) {
            let chunk = Array(samples[lower..<min(lower + 5120, samples.count)])
            guidedPartials.append(contentsOf: try guided.pushAudio(chunk))
            automaticPartials.append(contentsOf: try automatic.pushAudio(chunk))
        }
        guidedPartials.append(contentsOf: try guided.finalize())
        automaticPartials.append(contentsOf: try automatic.finalize())
        let guidedFinal = try XCTUnwrap(guidedPartials.last)
        XCTAssertEqual(guidedFinal.text, guidedBaseline.text)
        XCTAssertEqual(guidedFinal.confidence, guidedBaseline.confidence)
        XCTAssertEqual(guidedFinal.words, guidedBaseline.words)
        XCTAssertEqual(guidedFinal.wordBoostingChangedDecisions, guidedBaseline.wordBoostingChangedDecisions)
        let final = try XCTUnwrap(automaticPartials.last)
        XCTAssertEqual(final.text, baseline.text)
        XCTAssertEqual(final.confidence, baseline.confidence)
        XCTAssertEqual(final.words, baseline.words)
        let after = try decode(samples)
        XCTAssertEqual(after.text, baseline.text)
        XCTAssertEqual(after.confidence, baseline.confidence)
    }

    func testUnsupportedGuideAndFixedPromptFailAtAdmission() throws {
        let model = try XCTUnwrap(Self.model)
        XCTAssertThrowsError(try model.createSession(
            language: "auto", languageGuide: LanguageGuideConfig(expectedLanguages: ["invalid"])))
        XCTAssertThrowsError(try model.createSession(
            language: "en-US", languageGuide: LanguageGuideConfig(expectedLanguages: ["en-US"]))) {
                XCTAssertEqual($0 as? LanguageGuideError, .requiresAutomaticLanguage)
            }
    }
}
