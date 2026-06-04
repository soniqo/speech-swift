import Foundation
import AudioCommon

/// Language tag → prompt slot index mapping shipped with multilingual Nemotron
/// bundles as `languages.json`. The model uses this to look up the one-hot
/// position in the `language_mask` input (shape `[1, numPrompts]`).
///
/// File format (CoreML bundle):
/// ```json
/// {"promptDictionary": {"en-US": 0, "en": 0, "en-GB": 1, ... "auto": 101}}
/// ```
public struct NemotronLanguages: Sendable, Codable {
    public let promptDictionary: [String: Int]

    public init(promptDictionary: [String: Int]) {
        self.promptDictionary = promptDictionary
    }

    public static func load(from url: URL) throws -> NemotronLanguages {
        let data = try Data(contentsOf: url)
        if let wrapped = try? JSONDecoder().decode(NemotronLanguages.self, from: data) {
            return wrapped
        }
        // Fall back to a bare dictionary (the MLX bundle ships lang2slot.json
        // as a flat map without the promptDictionary wrapper).
        let flat = try JSONDecoder().decode([String: Int].self, from: data)
        return NemotronLanguages(promptDictionary: flat)
    }

    /// Find the prompt slot for a language tag. Tries the full tag first
    /// (`en-US`), falls back to the language prefix (`en`), then to `"auto"`.
    public func slot(for language: String?) -> Int {
        guard let language = language, !language.isEmpty else {
            return promptDictionary["auto"] ?? 0
        }
        if let s = promptDictionary[language] { return s }
        // Normalize: "en-us" → "en-US"; "english" → "en" (best-effort).
        let normalized = language.replacingOccurrences(of: "_", with: "-")
        if let s = promptDictionary[normalized] { return s }
        let prefix = String(normalized.split(separator: "-").first ?? "")
        if !prefix.isEmpty, let s = promptDictionary[prefix] { return s }
        if !prefix.isEmpty, let s = promptDictionary[prefix.lowercased()] { return s }
        return promptDictionary["auto"] ?? 0
    }

    public var count: Int { promptDictionary.count }
}
