import Foundation

/// Experimental fractional language-prompt conditioning. This is a package
/// experiment; the model's documented language input is a one-hot vector.
/// Expected languages never remove tokens from the decoder vocabulary.
public struct LanguageGuideConfig: Sendable, Equatable {
    public let expectedLanguages: [String]
    public let expectedLanguageWeight: Float

    public init(
        expectedLanguages: [String],
        expectedLanguageWeight: Float = 0.1
    ) {
        self.expectedLanguages = expectedLanguages
        self.expectedLanguageWeight = expectedLanguageWeight
    }
}

public enum LanguageGuideError: Error, LocalizedError, Equatable {
    case invalidWeight
    case unsupportedLanguage(String)
    case requiresAutomaticLanguage
    case automaticLanguageUnavailable

    public var errorDescription: String? {
        switch self {
        case .invalidWeight:
            return "Experimental language-guide weight must be finite and between 0 and 0.1."
        case .unsupportedLanguage(let language):
            return "Language guide is unavailable for the exact locale '\(language)'."
        case .requiresAutomaticLanguage:
            return "A language guide requires automatic-language recognition."
        case .automaticLanguageUnavailable:
            return "This model does not support automatic-language guidance."
        }
    }
}

/// Pure setup-time validation and mask construction. Each session receives
/// its own immutable mask. This does not directly bias decoder scores or mask
/// vocabulary; encoder and decoder caches remain session-local.
enum LanguageGuideContext {
    static func supportedLanguages(
        vocabulary: NemotronVocabulary,
        languages: NemotronLanguages,
        promptCount: Int,
        hasLanguageMask: Bool
    ) -> [String] {
        guard hasLanguageMask, promptCount > 0,
            let autoSlot = languages.promptDictionary["auto"],
            (0..<promptCount).contains(autoSlot)
        else { return [] }
        return Array(Set(vocabulary.languageTags.values.filter { locale in
            guard let slot = languages.promptDictionary[locale] else { return false }
            return (0..<promptCount).contains(slot) && slot != autoSlot
        })).sorted()
    }

    static func makeMask(
        config: LanguageGuideConfig?,
        vocabulary: NemotronVocabulary,
        languages: NemotronLanguages,
        language: String?,
        hasLanguageMask: Bool,
        promptCount: Int
    ) throws -> [Float]? {
        guard let config else { return nil }
        guard config.expectedLanguageWeight.isFinite,
            (0...Float(0.1)).contains(config.expectedLanguageWeight)
        else { throw LanguageGuideError.invalidWeight }
        guard !config.expectedLanguages.isEmpty else { return nil }
        guard language == nil || language == "auto" else {
            throw LanguageGuideError.requiresAutomaticLanguage
        }
        guard hasLanguageMask, promptCount > 0,
            let autoSlot = languages.promptDictionary["auto"],
            (0..<promptCount).contains(autoSlot)
        else { throw LanguageGuideError.automaticLanguageUnavailable }

        let supported = Set(supportedLanguages(
            vocabulary: vocabulary,
            languages: languages,
            promptCount: promptCount,
            hasLanguageMask: hasLanguageMask))
        var selectedSlots = Set<Int>()
        for locale in config.expectedLanguages {
            guard supported.contains(locale), let slot = languages.promptDictionary[locale] else {
                throw LanguageGuideError.unsupportedLanguage(locale)
            }
            selectedSlots.insert(slot)
        }
        // Zero takes the original one-hot allocation path exactly, after
        // explicit inputs have been validated at session admission.
        guard config.expectedLanguageWeight > 0 else { return nil }
        var mask = [Float](repeating: 0, count: promptCount)
        mask[autoSlot] = 1 - config.expectedLanguageWeight
        let selectedWeight = config.expectedLanguageWeight / Float(selectedSlots.count)
        for slot in selectedSlots { mask[slot] = selectedWeight }
        return mask
    }
}
