import Foundation

/// Pragmatic Swift port of `whisper.normalizers.EnglishTextNormalizer.__call__`
/// for bench parity with Python. Covers everything except the spelled-number
/// → digit state machine (skipped: low impact on FLEURS where numbers are
/// already spelled out; not worth ~300 lines of Swift state machine).
///
/// Faithful to Whisper's English normalizer for:
///   - filler removal (`hmm|mm|mhm|mmm|uh|um`)
///   - bracketed content stripping (`[...]`, `<...>`, `(...)`)
///   - contraction expansion (`won't → will not`, generic `n't → not`, etc.)
///   - title abbreviations (`mr → mister`, `dr → doctor`, etc.)
///   - perfect-tense markers (`'d been → had been`, `'s gone → has gone`)
///   - digit-comma removal, period handling
///   - symbol+diacritic removal (M*/S*/P* except keep `.%$¢€£`)
///   - prefix/suffix symbol cleanup after numbers
///   - British→American spelling normalization (english.json)
///   - whitespace collapse
struct WhisperEnglishNormalizer {

    private let ignorePattern: NSRegularExpression
    private let bracketPattern: NSRegularExpression
    private let parenPattern: NSRegularExpression
    private let spaceApostrophe: NSRegularExpression
    private let digitComma: NSRegularExpression
    private let periodNotDigit: NSRegularExpression
    private let prefixSymbol: NSRegularExpression
    private let percentSuffix: NSRegularExpression
    private let multiSpace: NSRegularExpression

    private let replacers: [(NSRegularExpression, String)]
    private let keepSymbols: Set<Unicode.Scalar>
    private let spellingMap: [String: String]
    private let numberNormalizer = EnglishNumberNormalizer()

    init() {
        ignorePattern = try! NSRegularExpression(pattern: "\\b(hmm|mm|mhm|mmm|uh|um)\\b")
        bracketPattern = try! NSRegularExpression(pattern: "[\\[<][^\\]>]*[\\]>]")
        parenPattern = try! NSRegularExpression(pattern: "\\(([^)]+?)\\)")
        spaceApostrophe = try! NSRegularExpression(pattern: "\\s+'")
        digitComma = try! NSRegularExpression(pattern: "(\\d),(\\d)")
        periodNotDigit = try! NSRegularExpression(pattern: "\\.([^0-9]|$)")
        prefixSymbol = try! NSRegularExpression(pattern: "[.$¢€£]([^0-9])")
        percentSuffix = try! NSRegularExpression(pattern: "([^0-9])%")
        multiSpace = try! NSRegularExpression(pattern: "\\s+")

        // Load british→american spelling map from bundled english.json
        // (same file Whisper ships in `whisper.normalizers.english_json`).
        if let url = Bundle.module.url(forResource: "english", withExtension: "json"),
           let data = try? Data(contentsOf: url),
           let dict = try? JSONDecoder().decode([String: String].self, from: data) {
            spellingMap = dict
        } else {
            spellingMap = [:]
        }

        // Exact replacement table from Whisper's EnglishTextNormalizer.
        let patterns: [(String, String)] = [
            // Common contractions
            ("\\bwon't\\b", "will not"),
            ("\\bcan't\\b", "can not"),
            ("\\blet's\\b", "let us"),
            ("\\bain't\\b", "aint"),
            ("\\by'all\\b", "you all"),
            ("\\bwanna\\b", "want to"),
            ("\\bgotta\\b", "got to"),
            ("\\bgonna\\b", "going to"),
            ("\\bi'ma\\b", "i am going to"),
            ("\\bimma\\b", "i am going to"),
            ("\\bwoulda\\b", "would have"),
            ("\\bcoulda\\b", "could have"),
            ("\\bshoulda\\b", "should have"),
            ("\\bma'am\\b", "madam"),
            // Title abbreviations
            ("\\bmr\\b", "mister "),
            ("\\bmrs\\b", "missus "),
            ("\\bst\\b", "saint "),
            ("\\bdr\\b", "doctor "),
            ("\\bprof\\b", "professor "),
            ("\\bcapt\\b", "captain "),
            ("\\bgov\\b", "governor "),
            ("\\bald\\b", "alderman "),
            ("\\bgen\\b", "general "),
            ("\\bsen\\b", "senator "),
            ("\\brep\\b", "representative "),
            ("\\bpres\\b", "president "),
            ("\\brev\\b", "reverend "),
            ("\\bhon\\b", "honorable "),
            ("\\basst\\b", "assistant "),
            ("\\bassoc\\b", "associate "),
            ("\\blt\\b", "lieutenant "),
            ("\\bcol\\b", "colonel "),
            ("\\bjr\\b", "junior "),
            ("\\bsr\\b", "senior "),
            ("\\besq\\b", "esquire "),
            // Perfect tenses
            ("'d been\\b", " had been"),
            ("'s been\\b", " has been"),
            ("'d gone\\b", " had gone"),
            ("'s gone\\b", " has gone"),
            ("'d done\\b", " had done"),
            ("'s got\\b", " has got"),
            // General contractions
            ("n't\\b", " not"),
            ("'re\\b", " are"),
            ("'s\\b", " is"),
            ("'d\\b", " would"),
            ("'ll\\b", " will"),
            ("'t\\b", " not"),
            ("'ve\\b", " have"),
            ("'m\\b", " am"),
        ]
        replacers = patterns.map { (try! NSRegularExpression(pattern: $0.0), $0.1) }
        keepSymbols = Set(".%$¢€£".unicodeScalars)
    }

    func callAsFunction(_ input: String) -> String {
        var s = input.lowercased()

        s = applyRegex(bracketPattern, on: s, with: "")
        s = applyRegex(parenPattern, on: s, with: "")
        s = applyRegex(ignorePattern, on: s, with: "")
        s = applyRegex(spaceApostrophe, on: s, with: "'")

        for (pat, repl) in replacers {
            s = applyRegex(pat, on: s, with: repl)
        }

        s = applyRegex(digitComma, on: s, with: "$1$2")
        s = applyRegex(periodNotDigit, on: s, with: " $1")
        s = removeSymbolsAndDiacritics(s)

        s = numberNormalizer(s)
        s = standardizeSpellings(s)

        // Strip prefix currency/period not followed by digit, and % not preceded
        // by digit. Matches Whisper's final cleanup pair.
        s = applyRegex(prefixSymbol, on: s, with: " $1")
        s = applyRegex(percentSuffix, on: s, with: "$1 ")

        s = applyRegex(multiSpace, on: s, with: " ")
        return s.trimmingCharacters(in: .whitespaces)
    }

    private func standardizeSpellings(_ s: String) -> String {
        if spellingMap.isEmpty { return s }
        let words = s.split(separator: " ", omittingEmptySubsequences: false)
        return words.map { spellingMap[String($0)] ?? String($0) }.joined(separator: " ")
    }

    // MARK: - Helpers

    private func applyRegex(_ re: NSRegularExpression, on s: String, with template: String) -> String {
        let r = NSRange(s.startIndex..., in: s)
        return re.stringByReplacingMatches(in: s, range: r, withTemplate: template)
    }

    /// Mirrors Whisper's `remove_symbols_and_diacritics(s, keep=".%$¢€£")`:
    /// NFD-normalize, strip non-spacing marks (Mn), replace other M/S/P
    /// scalars with space, keep ASCII + numeric currency symbols.
    private func removeSymbolsAndDiacritics(_ s: String) -> String {
        let nfd = s.decomposedStringWithCanonicalMapping
        var out = ""
        out.reserveCapacity(nfd.unicodeScalars.count)
        for scalar in nfd.unicodeScalars {
            if keepSymbols.contains(scalar) {
                out.unicodeScalars.append(scalar)
                continue
            }
            let cat = scalar.properties.generalCategory
            switch cat {
            case .nonspacingMark:
                continue  // drop diacritics entirely
            case .enclosingMark, .spacingMark,
                 .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
                 .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
                 .initialPunctuation, .finalPunctuation, .otherPunctuation:
                out.unicodeScalars.append(Unicode.Scalar(0x20)!)
            default:
                out.unicodeScalars.append(scalar)
            }
        }
        return out
    }
}
