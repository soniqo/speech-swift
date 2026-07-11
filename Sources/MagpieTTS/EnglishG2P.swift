import Foundation

/// Dict-based IPA grapheme-to-phoneme converter.
///
/// Mirrors NeMo's `IpaG2p` + `IPATokenizer` pipeline by loading one of the
/// CMU-style IPA dictionaries shipped in the Magpie `.nemo` bundle:
///
/// - English: `cmudict_ipa_en.txt` (125 k words, NeMo's `cdd41953...`)
/// - Spanish: `cmudict_ipa_es.txt` (NeMo's `9a6b090b...`)
/// - German:  `cmudict_ipa_de.txt` (NeMo's `bafa5b4c...`)
///
/// Pipeline:
///   1. Lazily load the dict into a `[uppercased word → IPA]` map.
///   2. Tokenise input by Unicode letter-or-apostrophe runs vs. everything
///      else (punctuation stays verbatim).
///   3. Look up each word case-insensitively. OOV words fall back to the
///      lower-cased grapheme stream (`use_chars=True` semantics).
///   4. Concatenate phonemised words separated by spaces — NeMo's
///      `' '.join(phonemized_words)` behaviour.
///
/// The result is an IPA + punctuation string ready for char-level lookup
/// against the Magpie shared vocab.
public final class MagpieDictG2P {

    public static let english = MagpieDictG2P(resource: "cmudict_ipa_en")
    public static let spanish = MagpieDictG2P(resource: "cmudict_ipa_es")
    public static let german  = MagpieDictG2P(resource: "cmudict_ipa_de")

    private let resource: String
    private var dict: [String: String] = [:]
    private var loaded = false
    private let loadLock = NSLock()

    private init(resource: String) {
        self.resource = resource
    }

    public func ensureLoaded() throws {
        loadLock.lock()
        defer { loadLock.unlock() }
        if loaded { return }
        guard let url = Bundle.module.url(forResource: resource, withExtension: "txt") else {
            throw MagpieTTSError.missingFile("\(resource).txt (resource bundle)")
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        dict.reserveCapacity(150_000)
        let whitespace = CharacterSet.whitespaces
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: true) {
            let line = rawLine.trimmingCharacters(in: .whitespacesAndNewlines)
            if line.isEmpty || line.hasPrefix(";;;") { continue }
            // Format depends on the dict:
            //   English (CMU IPA):  `HELLO  həˈloʊ`        (uppercased, double space)
            //   Spanish (es_ES):    `HOLA  ˈola`            (uppercased, double space)
            //   German  (de):       `Hallo\thˈaloː`        (mixed-case, TAB sep)
            // We split on any whitespace and lowercase the key so a single
            // lookup path handles all three. Alternate-pron entries
            // (`WORD(1)`, `WORD(2)`) are skipped to match NeMo's
            // `phoneme_probability=1.0` lookup.
            let parts = line.unicodeScalars.split { whitespace.contains($0) }
            if parts.count < 2 { continue }
            let head = String(String.UnicodeScalarView(parts[0]))
            if head.contains("(") { continue }
            let ipa = parts[1...].map { String(String.UnicodeScalarView($0)) }.joined()
            dict[head.lowercased()] = ipa
        }
        loaded = true
    }

    /// Phonemise raw text. The caller is expected to have applied any
    /// per-language preprocessing (lowercasing, NFC normalisation) before
    /// passing the string in.
    public func phonemize(_ text: String) -> String {
        do { try ensureLoaded() } catch { return text }
        var phonemes: [String] = []
        var current = ""
        for ch in text {
            if ch.isLetter || ch == "'" {
                current.append(ch)
            } else {
                if !current.isEmpty {
                    phonemes.append(lookup(current))
                    current = ""
                }
                if !ch.isWhitespace {
                    phonemes.append(String(ch))
                }
            }
        }
        if !current.isEmpty {
            phonemes.append(lookup(current))
        }
        return phonemes.joined(separator: " ")
    }

    private func lookup(_ word: String) -> String {
        if let hit = dict[word.lowercased()] { return hit }
        return word.lowercased()
    }
}

/// Legacy alias — earlier patches referenced this name directly. The new
/// shared implementation lives on `MagpieDictG2P`.
public typealias MagpieEnglishG2P = MagpieDictG2P
extension MagpieDictG2P {
    public static let shared = MagpieDictG2P.english
}
