import Foundation

/// Per-character script-weight model used to estimate speech duration from text.
///
/// Direct port of `RuleDurationEstimator` in `omnivoice/utils/duration.py`. Each
/// character contributes a weight relative to a Latin letter (baseline 1.0); the
/// sum approximates relative speaking time across 600+ scripts. `OmniVoice`'s
/// `generate()` uses this to pick the output token count when no duration is
/// supplied.
public final class RuleDurationEstimator: @unchecked Sendable {
    public static let shared = RuleDurationEstimator()

    // Phonetic weights (relative speaking time vs one Latin character).
    private let weights: [String: Double] = [
        "cjk": 3.0,
        "hangul": 2.5,
        "kana": 2.2,
        "ethiopic": 3.0,
        "yi": 3.0,
        "indic": 1.8,
        "thai_lao": 1.5,
        "khmer_myanmar": 1.8,
        "arabic": 1.5,
        "hebrew": 1.5,
        "latin": 1.0,
        "cyrillic": 1.0,
        "greek": 1.0,
        "armenian": 1.0,
        "georgian": 1.0,
        "punctuation": 0.5,
        "space": 0.2,
        "digit": 3.5,
        "mark": 0.0,
        "default": 1.0,
    ]

    // (end_codepoint, script_key), ascending — binary searched by `bisect_left`.
    private let ranges: [(UInt32, String)] = [
        (0x02AF, "latin"), (0x03FF, "greek"), (0x052F, "cyrillic"), (0x058F, "armenian"),
        (0x05FF, "hebrew"), (0x077F, "arabic"), (0x089F, "arabic"), (0x08FF, "arabic"),
        (0x097F, "indic"), (0x09FF, "indic"), (0x0A7F, "indic"), (0x0AFF, "indic"),
        (0x0B7F, "indic"), (0x0BFF, "indic"), (0x0C7F, "indic"), (0x0CFF, "indic"),
        (0x0D7F, "indic"), (0x0DFF, "indic"), (0x0EFF, "thai_lao"), (0x0FFF, "indic"),
        (0x109F, "khmer_myanmar"), (0x10FF, "georgian"), (0x11FF, "hangul"), (0x137F, "ethiopic"),
        (0x139F, "ethiopic"), (0x13FF, "default"), (0x167F, "default"), (0x169F, "default"),
        (0x16FF, "default"), (0x171F, "default"), (0x173F, "default"), (0x175F, "default"),
        (0x177F, "default"), (0x17FF, "khmer_myanmar"), (0x18AF, "default"), (0x18FF, "default"),
        (0x194F, "indic"), (0x19DF, "indic"), (0x19FF, "khmer_myanmar"), (0x1A1F, "indic"),
        (0x1AAF, "indic"), (0x1B7F, "indic"), (0x1BBF, "indic"), (0x1BFF, "indic"),
        (0x1C4F, "indic"), (0x1C7F, "indic"), (0x1C8F, "cyrillic"), (0x1CBF, "georgian"),
        (0x1CCF, "indic"), (0x1CFF, "indic"), (0x1D7F, "latin"), (0x1DBF, "latin"),
        (0x1DFF, "default"), (0x1EFF, "latin"), (0x309F, "kana"), (0x30FF, "kana"),
        (0x312F, "cjk"), (0x318F, "hangul"), (0x9FFF, "cjk"), (0xA4CF, "yi"),
        (0xA4FF, "default"), (0xA63F, "default"), (0xA69F, "cyrillic"), (0xA6FF, "default"),
        (0xA7FF, "latin"), (0xA82F, "indic"), (0xA87F, "default"), (0xA8DF, "indic"),
        (0xA8FF, "indic"), (0xA92F, "indic"), (0xA95F, "indic"), (0xA97F, "hangul"),
        (0xA9DF, "indic"), (0xA9FF, "khmer_myanmar"), (0xAA5F, "indic"), (0xAA7F, "khmer_myanmar"),
        (0xAADF, "indic"), (0xAAFF, "indic"), (0xAB2F, "ethiopic"), (0xAB6F, "latin"),
        (0xABBF, "default"), (0xABFF, "indic"), (0xD7AF, "hangul"), (0xFAFF, "cjk"),
        (0xFDFF, "arabic"), (0xFE6F, "default"), (0xFEFF, "arabic"), (0xFFEF, "latin"),
    ]
    private let breakpoints: [UInt32]

    public init() {
        breakpoints = ranges.map { $0.0 }
    }

    /// `bisect_left(breakpoints, code)` — leftmost index where `breakpoints[i] >= code`.
    private func bisectLeft(_ code: UInt32) -> Int {
        var lo = 0, hi = breakpoints.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if breakpoints[mid] < code { lo = mid + 1 } else { hi = mid }
        }
        return lo
    }

    /// Weight of a single Unicode scalar (port of `_get_char_weight`).
    func charWeight(_ scalar: Unicode.Scalar) -> Double {
        let code = scalar.value
        if (65 ... 90).contains(code) || (97 ... 122).contains(code) { return weights["latin"]! }
        if code == 32 { return weights["space"]! }
        if code == 0x0640 { return weights["mark"]! }  // Arabic Tatweel

        let cat = Self.generalCategoryPrefix(scalar)
        switch cat {
        case "M": return weights["mark"]!
        case "P", "S": return weights["punctuation"]!
        case "Z": return weights["space"]!
        case "N": return weights["digit"]!
        default: break
        }

        let idx = bisectLeft(code)
        if idx < ranges.count {
            return weights[ranges[idx].1] ?? weights["default"]!
        }
        if code > 0x20000 { return weights["cjk"]! }
        return weights["default"]!
    }

    /// Sum of per-character weights for `text` (port of `calculate_total_weight`).
    public func totalWeight(_ text: String) -> Double {
        var sum = 0.0
        for scalar in text.unicodeScalars { sum += charWeight(scalar) }
        return sum
    }

    /// Estimate the target audio **token** count, matching OmniVoice's
    /// `_estimate_target_tokens` -> `estimate_duration`. The reference's *token
    /// count* (not its wall-clock seconds) sets the speaker's pace: a target whose
    /// text weight is k times the reference's gets roughly k times the reference's
    /// tokens. When the reference text/length is absent, the reference falls back
    /// to a fixed prior ("Nice to meet you." / 25 tokens). Short estimates get a
    /// power-curve boost.
    public func estimateTargetTokens(
        targetText: String, refText: String?, numRefAudioTokens: Int?,
        lowThreshold: Double = 50, boostStrength: Double = 3
    ) -> Int {
        let effectiveRefText: String
        let effectiveRefTokens: Int
        if let refText, !refText.isEmpty, let numRefAudioTokens {
            effectiveRefText = refText
            effectiveRefTokens = numRefAudioTokens
        } else {
            effectiveRefText = "Nice to meet you."
            effectiveRefTokens = 25
        }
        let refDuration = Double(effectiveRefTokens)
        let refWeight = totalWeight(effectiveRefText)
        guard refDuration > 0, refWeight > 0 else { return 1 }
        let speedFactor = refWeight / refDuration
        var est = totalWeight(targetText) / speedFactor
        if est < lowThreshold {
            est = lowThreshold * pow(est / lowThreshold, 1.0 / boostStrength)
        }
        return max(1, Int(est))
    }

    /// First letter of the Unicode general category (`L`/`M`/`N`/`P`/`S`/`Z`/…),
    /// matching Python `unicodedata.category(c)[0]` for the cases the estimator
    /// branches on.
    private static func generalCategoryPrefix(_ scalar: Unicode.Scalar) -> Character {
        let p = scalar.properties
        if p.isWhitespace && scalar.value != 0x09 && scalar.value != 0x0A
            && scalar.value != 0x0D && scalar.value != 0x0B && scalar.value != 0x0C {
            // Zs/Zl/Zp — spacing separators (control whitespace falls through to "C").
            return "Z"
        }
        switch p.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
             .modifierLetter, .otherLetter:
            return "L"
        case .nonspacingMark, .spacingMark, .enclosingMark:
            return "M"
        case .decimalNumber, .letterNumber, .otherNumber:
            return "N"
        case .connectorPunctuation, .dashPunctuation, .openPunctuation,
             .closePunctuation, .initialPunctuation, .finalPunctuation,
             .otherPunctuation:
            return "P"
        case .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol:
            return "S"
        case .spaceSeparator, .lineSeparator, .paragraphSeparator:
            return "Z"
        default:
            return "C"
        }
    }
}
