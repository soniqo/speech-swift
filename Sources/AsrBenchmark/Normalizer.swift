import Foundation

/// Text normalizer for cross-engine WER comparison.
///
/// Different ASR engines output different conventions (punctuation, casing,
/// digit-vs-spelled numbers, possessives). Normalizing both reference and
/// hypothesis to a common form makes WERs comparable. This is a simplified
/// take on Whisper's `BasicTextNormalizer`: lowercase, strip punctuation,
/// expand a handful of common contractions, collapse whitespace.
public enum Normalizer {
    public static func normalize(_ s: String) -> String {
        var t = s

        // Strip ChatML / Qwen-style special tokens (`<|im_end|>`, `<|asr_text|>`,
        // `<|audio_end|>`, etc.) before lowercasing. If we let them through
        // they get split into bogus words ("im end") by the punctuation pass.
        t = t.replacingOccurrences(
            of: #"<\|[^|>]*\|>"#,
            with: " ",
            options: .regularExpression
        )

        t = t.lowercased()

        // Replace a few unicode quotes/dashes that survive case folding.
        let unicodeMap: [(String, String)] = [
            ("\u{2018}", "'"), ("\u{2019}", "'"),  // ‘ ’
            ("\u{201C}", "\""), ("\u{201D}", "\""), // “ ”
            ("\u{2013}", "-"), ("\u{2014}", "-"),  // – —
            ("\u{00A0}", " ")                       // nbsp
        ]
        for (a, b) in unicodeMap { t = t.replacingOccurrences(of: a, with: b) }

        // Expand a small fixed set of contractions. Whisper's normalizer has
        // a much bigger table — we keep the high-frequency ones that move WER
        // most across engines (Qwen, Whisper, Parakeet often disagree here).
        let contractions: [(String, String)] = [
            ("won't", "will not"), ("can't", "can not"), ("n't", " not"),
            ("'re", " are"), ("'ve", " have"), ("'ll", " will"),
            ("'d", " would"), ("'m", " am"), ("let's", "let us"),
            ("it's", "it is"), ("'s", " is")
        ]
        for (a, b) in contractions {
            t = t.replacingOccurrences(of: a, with: b)
        }

        // Drop all punctuation; keep letters, digits, spaces, apostrophes
        // (apostrophes remain because the contraction pass may leave 's
        // tokens we *want* to compare as-is for already-expanded forms).
        var out = ""
        out.reserveCapacity(t.count)
        for ch in t.unicodeScalars {
            if CharacterSet.letters.contains(ch) ||
               CharacterSet.decimalDigits.contains(ch) ||
               ch == " " || ch == "\t" || ch == "\n" {
                out.unicodeScalars.append(ch)
            } else {
                out.unicodeScalars.append(" ")
            }
        }

        // Collapse whitespace.
        let collapsed = out.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
