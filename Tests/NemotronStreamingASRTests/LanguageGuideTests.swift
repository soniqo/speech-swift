import XCTest
@testable import NemotronStreamingASR

final class LanguageGuideTests: XCTestCase {
    private let vocabulary = NemotronVocabulary(idToToken: [
        0: "<unk>", 1: "▁hello", 2: "<en-US>", 3: "<ru-RU>",
        4: "<fr-FR>", 5: "<en-GB>", 6: "<", 7: "<s>",
        8: "<auto>", 9: "<zz-ZZ>", 10: "<de-DE>", 11: "<hi-IN>",
        12: "▁привет", 13: "▁bonjour",
    ])
    private let languages = NemotronLanguages(promptDictionary: [
        "auto": 101, "en-US": 0, "en-GB": 1, "ru-RU": 12,
        "fr-FR": 7, "de-DE": 9, "en": 0,
    ])

    private func mask(
        _ expected: [String] = ["en-US"],
        weight: Float = 0.1,
        language: String? = nil,
        metadata: NemotronLanguages? = nil,
        hasLanguageMask: Bool = true,
        promptCount: Int = 128
    ) throws -> [Float]? {
        try LanguageGuideContext.makeMask(
            config: LanguageGuideConfig(expectedLanguages: expected, expectedLanguageWeight: weight),
            vocabulary: vocabulary,
            languages: metadata ?? languages,
            language: language,
            hasLanguageMask: hasLanguageMask,
            promptCount: promptCount)
    }

    func testEmptyAndZeroGuideUseOriginalOneHotPath() throws {
        XCTAssertNil(try mask([]))
        XCTAssertNil(try mask(weight: 0))
        XCTAssertNil(try mask([], language: "ru-RU"))
        XCTAssertNil(try LanguageGuideContext.makeMask(
            config: nil, vocabulary: vocabulary,
            languages: NemotronLanguages(promptDictionary: [:]), language: "en-US",
            hasLanguageMask: false, promptCount: 0))
    }

    func testDefaultMixtureRetainsNinetyPercentAuto() throws {
        let config = LanguageGuideConfig(expectedLanguages: ["en-US", "en-GB", "ru-RU"])
        XCTAssertEqual(config.expectedLanguageWeight, 0.1)
        let mixed = try XCTUnwrap(LanguageGuideContext.makeMask(
            config: config, vocabulary: vocabulary, languages: languages,
            language: "auto", hasLanguageMask: true, promptCount: 128))
        XCTAssertEqual(mixed.count, 128)
        XCTAssertEqual(mixed[101], 0.9, accuracy: 0.000001)
        for slot in [0, 1, 12] { XCTAssertEqual(mixed[slot], 0.1 / 3, accuracy: 0.000001) }
        XCTAssertEqual(mixed.reduce(0, +), 1, accuracy: 0.000001)
        XCTAssertEqual(mixed.filter { $0 > 0 }.count, 4)
        XCTAssertGreaterThanOrEqual(mixed[101], Float(0.9))
    }

    func testLocaleAndPromptAliasesDoNotMultiplyWeight() throws {
        let metadata = NemotronLanguages(promptDictionary: [
            "auto": 101, "en-US": 0, "en-GB": 0, "ru-RU": 12,
        ])
        let mixed = try XCTUnwrap(mask(
            ["en-US", "en-GB", "en-US", "ru-RU"], metadata: metadata))
        XCTAssertEqual(mixed[0], 0.05)
        XCTAssertEqual(mixed[12], 0.05)
        XCTAssertEqual(mixed[101], 0.9)
        XCTAssertEqual(mixed.reduce(0, +), 1, accuracy: 0.000001)
    }

    func testInvalidWeightsFailAtAdmission() {
        for weight in [Float.nan, .infinity, -.infinity, -0.001, Float(0.1).nextUp, 0.25, 1] {
            XCTAssertThrowsError(try mask(weight: weight)) {
                XCTAssertEqual($0 as? LanguageGuideError, .invalidWeight)
            }
        }
        XCTAssertThrowsError(try mask([], weight: .nan))
    }

    func testExactLocalesAndLoadedSlotBoundsAreRequired() {
        for locale in ["", "auto", "en", "EN-US", "<en-US>", "zz-ZZ", "hi-IN"] {
            XCTAssertThrowsError(try mask([locale])) {
                XCTAssertEqual($0 as? LanguageGuideError, .unsupportedLanguage(locale))
            }
        }
        for slot in [-1, 128, 101] {
            let metadata = NemotronLanguages(promptDictionary: ["auto": 101, "en-US": slot])
            XCTAssertThrowsError(try mask(metadata: metadata)) {
                XCTAssertEqual($0 as? LanguageGuideError, .unsupportedLanguage("en-US"))
            }
        }
        // Zero bypasses mixture allocation, not explicit-input validation.
        XCTAssertThrowsError(try mask(["en"], weight: 0))
    }

    func testGuideRequiresValidAutomaticPromptAndMaskInput() {
        for language in ["en-US", "ru-RU", "unknown"] {
            XCTAssertThrowsError(try mask(language: language)) {
                XCTAssertEqual($0 as? LanguageGuideError, .requiresAutomaticLanguage)
            }
        }
        XCTAssertThrowsError(try mask(hasLanguageMask: false)) {
            XCTAssertEqual($0 as? LanguageGuideError, .automaticLanguageUnavailable)
        }
        for autoSlot in [-1, 128] {
            XCTAssertThrowsError(try mask(metadata: NemotronLanguages(
                promptDictionary: ["auto": autoSlot, "en-US": 0]))) {
                XCTAssertEqual($0 as? LanguageGuideError, .automaticLanguageUnavailable)
            }
        }
        XCTAssertThrowsError(try mask(promptCount: 0))
        XCTAssertThrowsError(try mask(metadata: NemotronLanguages(promptDictionary: ["en-US": 0])))
    }

    func testCapabilitiesMatchAdmissionInsteadOfSparePromptSlots() {
        let metadata = NemotronLanguages(promptDictionary: [
            "auto": 101, "en-US": 0, "en-GB": 0, "ru-RU": 12,
            "fr-FR": 101, "de-DE": -1, "hi-IN": 128, "en": 0,
        ])
        XCTAssertEqual(LanguageGuideContext.supportedLanguages(
            vocabulary: vocabulary, languages: metadata, promptCount: 128, hasLanguageMask: true),
            ["en-GB", "en-US", "ru-RU"])
        XCTAssertTrue(LanguageGuideContext.supportedLanguages(
            vocabulary: vocabulary, languages: metadata, promptCount: 128, hasLanguageMask: false).isEmpty)
        XCTAssertTrue(LanguageGuideContext.supportedLanguages(
            vocabulary: vocabulary, languages: metadata, promptCount: 0, hasLanguageMask: true).isEmpty)
        XCTAssertEqual(vocabulary.languageTags[6], nil)
        XCTAssertEqual(vocabulary.languageTags[8], nil)
    }

    func testConfigAndMaskRemainSourceLocalSnapshots() throws {
        var expected = ["en-US"]
        let config = LanguageGuideConfig(expectedLanguages: expected)
        expected.append("ru-RU")
        XCTAssertEqual(config.expectedLanguages, ["en-US"])
        let system = try XCTUnwrap(mask(["en-US"]))
        var microphone = try XCTUnwrap(mask(["ru-RU"]))
        microphone[12] = 0
        XCTAssertEqual(system[0], 0.1)
        XCTAssertEqual(system[12], 0)
        XCTAssertEqual(try XCTUnwrap(mask(["ru-RU"]))[12], 0.1)
    }

    func testPromptGuideDoesNotFilterOutsiderVocabulary() throws {
        let before = vocabulary.decode([12, 13])
        _ = try mask(["en-US"])
        XCTAssertEqual(vocabulary.decode([12, 13]), before)
        XCTAssertEqual(before, "привет bonjour")
        XCTAssertEqual(vocabulary.count, 14)
        // This verifies vocabulary preservation only, not acoustic accuracy
        // or the probability of recognizing an unlisted language.
    }
}
