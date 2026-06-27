import XCTest
@testable import AudioCommon

final class HindiEmotionTTSCatalogTests: XCTestCase {
    func testCatalogIncludesHindiEmotionModels() {
        let ids = Set(HindiEmotionTTSCatalog.all.map(\.id))
        XCTAssertTrue(ids.contains("indic-mio"))
        XCTAssertTrue(ids.contains("fish-audio-s2-pro"))
        XCTAssertTrue(ids.contains("svara-tts-v1"))
    }

    func testCommercialImplementationCandidatesExcludeResearchOnlyModels() {
        let candidates = HindiEmotionTTSCatalog.implementationCandidates
        let ids = Set(candidates.map(\.id))

        XCTAssertTrue(ids.contains("indic-mio"))
        XCTAssertTrue(ids.contains("svara-tts-v1"))
        XCTAssertTrue(ids.contains("indic-parler-tts"))
        XCTAssertFalse(ids.contains("fish-audio-s2-pro"))
        XCTAssertFalse(ids.contains("indicf5"))
        XCTAssertFalse(ids.contains("orpheus-tts-hi"))

        for candidate in candidates {
            XCTAssertEqual(candidate.usePolicy, .commercialSafe)
            XCTAssertTrue(candidate.supportsHindi)
            XCTAssertTrue(candidate.supportsExplicitEmotionMarkers)
            XCTAssertNotEqual(candidate.markerSyntax, .noneDocumented)
            XCTAssertFalse(candidate.supportedMarkers.isEmpty)
        }
    }

    func testFishAudioIsBenchmarkOnlyBecauseOfLicense() throws {
        let fish = try XCTUnwrap(HindiEmotionTTSCatalog.candidate(id: "fish-audio-s2-pro"))
        XCTAssertEqual(fish.usePolicy, .researchOnly)
        XCTAssertEqual(fish.readiness, .benchmarkOnly)
        XCTAssertEqual(fish.markerSyntax, .inlineBracketTag)
        XCTAssertTrue(fish.supportedMarkers.contains("[angry]"))
        XCTAssertTrue(fish.supportedMarkers.contains("[whisper]"))
    }

    func testIndicMioUsesIndianLanguageAngleTags() throws {
        let mio = try XCTUnwrap(HindiEmotionTTSCatalog.candidate(id: "SPRINGLab/Indic-Mio"))
        XCTAssertEqual(mio.id, "indic-mio")
        XCTAssertEqual(mio.markerSyntax, .suffixAngleTag)
        XCTAssertTrue(mio.supportedMarkers.contains("<happy>"))
        XCTAssertTrue(mio.supportedMarkers.contains("<fear>"))
        XCTAssertEqual(mio.voiceConditioning, .referenceOrEmbedding)
    }

    func testTrackedOnlyModelsAreNotEmotionMarkerCandidates() {
        for id in ["indicf5", "orpheus-tts-hi"] {
            let candidate = HindiEmotionTTSCatalog.candidate(id: id)
            XCTAssertEqual(candidate?.readiness, .trackedOnly)
            XCTAssertFalse(candidate?.supportsExplicitEmotionMarkers ?? true)
            XCTAssertTrue(candidate?.supportedMarkers.isEmpty ?? false)
        }
    }
}
