import Foundation

/// Swift port of `whisper.normalizers.EnglishNumberNormalizer`.
/// Converts spelled-out English numbers to digits with handling for:
///   - cardinals + ordinals + plurals
///   - signs (minus/plus/negative/positive → -/+)
///   - currency (`dollars → $`, `cents → ¢`, `pounds → £`, `euros → €`)
///   - percent (`per cent → %`, `percent → %`)
///   - decimals via "point"
///   - "double X" / "triple X" repetition
///   - "X and a half" → "X point five"
///   - currency cents postprocessing (`$2 and ¢7 → $2.07`)
///   - `1(s)` → `one(s)` after conversion (Whisper readability hack)
///
/// Uses Int64 throughout — supports up to quintillion (10^18). Multipliers
/// sextillion+ from the Python list are kept but values truncate; FLEURS
/// test text doesn't reach them.
struct EnglishNumberNormalizer {

    // MARK: - Value sum type (int or partial string)

    private enum Value {
        case int(Int64)
        case str(String)

        var asString: String {
            switch self {
            case .int(let n): return String(n)
            case .str(let s): return s
            }
        }
    }

    // MARK: - Tables

    private let zeros: Set<String>
    private let ones: [String: Int64]
    private let onesSuffixed: [String: (Int64, String)]
    private let tens: [String: Int64]
    private let tensSuffixed: [String: (Int64, String)]
    private let multipliers: [String: Int64]
    private let multipliersSuffixed: [String: (Int64, String)]
    private let precedingPrefixers: [String: String]
    private let followingPrefixers: [String: String]
    private let prefixChars: Set<Character>
    private let specials: Set<String> = ["and", "double", "triple", "point"]
    private let words: Set<String>
    private let decimals: Set<String>

    private let onesList = [
        "one","two","three","four","five","six","seven","eight","nine",
        "ten","eleven","twelve","thirteen","fourteen","fifteen",
        "sixteen","seventeen","eighteen","nineteen",
    ]
    private let tensList = [
        "twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety",
    ]

    init() {
        // zeros
        zeros = ["o", "oh", "zero"]
        // ones
        var onesM: [String: Int64] = [:]
        for (i, name) in onesList.enumerated() { onesM[name] = Int64(i + 1) }
        ones = onesM
        // ones plural + ordinal -> suffixed
        var onesSuf: [String: (Int64, String)] = [:]
        for (name, value) in onesM {
            let plural = (name == "six") ? "sixes" : name + "s"
            onesSuf[plural] = (value, "s")
        }
        let onesOrdSpecial: [String: (Int64, String)] = [
            "zeroth": (0, "th"),
            "first":  (1, "st"),
            "second": (2, "nd"),
            "third":  (3, "rd"),
            "fifth":  (5, "th"),
            "twelfth":(12,"th"),
        ]
        for (k, v) in onesOrdSpecial { onesSuf[k] = v }
        for (name, value) in onesM {
            if value > 3 && value != 5 && value != 12 {
                let key = name.hasSuffix("t") ? name + "h" : name + "th"
                onesSuf[key] = (value, "th")
            }
        }
        onesSuffixed = onesSuf

        // tens
        var tensM: [String: Int64] = [:]
        for (i, name) in tensList.enumerated() { tensM[name] = Int64((i + 2) * 10) }
        tens = tensM
        // tens plural + ordinal -> suffixed
        var tensSuf: [String: (Int64, String)] = [:]
        for (name, value) in tensM {
            // "twenty" → "twenties" (plural) and "twentieth" (ordinal)
            let plural = name.replacingOccurrences(of: "y", with: "ies")
            tensSuf[plural] = (value, "s")
            let ordinal = name.replacingOccurrences(of: "y", with: "ieth")
            tensSuf[ordinal] = (value, "th")
        }
        tensSuffixed = tensSuf

        // multipliers
        let multM: [String: Int64] = [
            "hundred":     100,
            "thousand":    1_000,
            "million":     1_000_000,
            "billion":     1_000_000_000,
            "trillion":    1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            // Beyond Int64 — keep names recognized but value truncated to max.
            "sextillion":  Int64.max,
            "septillion":  Int64.max,
            "octillion":   Int64.max,
            "nonillion":   Int64.max,
            "decillion":   Int64.max,
        ]
        multipliers = multM
        var multSuf: [String: (Int64, String)] = [:]
        for (name, value) in multM {
            multSuf[name + "s"] = (value, "s")
            multSuf[name + "th"] = (value, "th")
        }
        multipliersSuffixed = multSuf

        // prefixers
        precedingPrefixers = [
            "minus": "-", "negative": "-", "plus": "+", "positive": "+",
        ]
        followingPrefixers = [
            "pound": "£", "pounds": "£",
            "euro": "€",  "euros": "€",
            "dollar": "$","dollars": "$",
            "cent": "¢",  "cents": "¢",
        ]
        var pfx: Set<Character> = []
        for v in precedingPrefixers.values { if let c = v.first { pfx.insert(c) } }
        for v in followingPrefixers.values  { if let c = v.first { pfx.insert(c) } }
        prefixChars = pfx

        // words set (anything we recognize as numeric vocabulary)
        var allWords: Set<String> = []
        allWords.formUnion(zeros)
        allWords.formUnion(onesM.keys)
        allWords.formUnion(onesSuf.keys)
        allWords.formUnion(tensM.keys)
        allWords.formUnion(tensSuf.keys)
        allWords.formUnion(multM.keys)
        allWords.formUnion(multSuf.keys)
        allWords.formUnion(precedingPrefixers.keys)
        allWords.formUnion(followingPrefixers.keys)
        allWords.insert("per"); allWords.insert("cent"); allWords.insert("percent")
        allWords.formUnion(specials)
        words = allWords

        // decimals set (for "point N" handling)
        var dec: Set<String> = []
        dec.formUnion(onesM.keys)
        dec.formUnion(tensM.keys)
        dec.formUnion(zeros)
        decimals = dec
    }

    // MARK: - Public entry point

    func callAsFunction(_ s: String) -> String {
        let pre = preprocess(s)
        let tokens = pre.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        let processed = processWords(tokens)
        return postprocess(processed.joined(separator: " "))
    }

    // MARK: - Preprocess

    private static let andAHalfRE = try! NSRegularExpression(pattern: "\\band\\s+a\\s+half\\b")
    private static let letterDigitRE = try! NSRegularExpression(pattern: "([a-z])([0-9])")
    private static let digitLetterRE = try! NSRegularExpression(pattern: "([0-9])([a-z])")
    private static let suffixCloseRE = try! NSRegularExpression(pattern: "([0-9])\\s+(st|nd|rd|th|s)\\b")

    private func preprocess(_ s: String) -> String {
        // Split by "and a half"; for each segment except the last, if the
        // preceding segment ends with a number-like word, append "point five",
        // otherwise put back "and a half".
        let range = NSRange(s.startIndex..., in: s)
        let matches = Self.andAHalfRE.matches(in: s, range: range)
        var segments: [String] = []
        var cursor = s.startIndex
        for m in matches {
            if let r = Range(m.range, in: s) {
                segments.append(String(s[cursor..<r.lowerBound]))
                cursor = r.upperBound
            }
        }
        segments.append(String(s[cursor...]))

        var results: [String] = []
        for (i, seg) in segments.enumerated() {
            let trimmed = seg.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            results.append(seg)
            if i != segments.count - 1 {
                let lastWord = seg.split(separator: " ").last.map(String.init) ?? ""
                if decimals.contains(lastWord) || multipliers.keys.contains(lastWord) {
                    results.append("point five")
                } else {
                    results.append("and a half")
                }
            }
        }
        var out = results.joined(separator: " ")

        // Space between letter and digit (both directions)
        out = applyRegex(Self.letterDigitRE, on: out, with: "$1 $2")
        out = applyRegex(Self.digitLetterRE, on: out, with: "$1 $2")
        // Remove space inside numeric suffix (`5 th` → `5th`)
        out = applyRegex(Self.suffixCloseRE, on: out, with: "$1$2")
        return out
    }

    // MARK: - Postprocess

    private static let combineCentsRE = try! NSRegularExpression(
        pattern: "([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\\b")
    private static let extractCentsRE = try! NSRegularExpression(
        pattern: "[€£$]0\\.([0-9]{1,2})\\b")
    private static let oneReplaceRE = try! NSRegularExpression(pattern: "\\b1(s?)\\b")

    private func postprocess(_ s: String) -> String {
        var out = s
        // "$2 and ¢7" → "$2.07"
        out = replaceMatches(in: out, using: Self.combineCentsRE) { ns, m in
            guard let r1 = Range(m.range(at: 1), in: ns),
                  let r2 = Range(m.range(at: 2), in: ns),
                  let r3 = Range(m.range(at: 3), in: ns),
                  let cents = Int(ns[r3]) else { return String(ns[Range(m.range, in: ns)!]) }
            return "\(ns[r1])\(ns[r2]).\(String(format: "%02d", cents))"
        }
        // "$0.07" → "¢7"
        out = replaceMatches(in: out, using: Self.extractCentsRE) { ns, m in
            guard let r1 = Range(m.range(at: 1), in: ns),
                  let v = Int(ns[r1]) else { return String(ns[Range(m.range, in: ns)!]) }
            return "¢\(v)"
        }
        // "1(s)" → "one(s)"
        out = applyRegex(Self.oneReplaceRE, on: out, with: "one$1")
        return out
    }

    // MARK: - Process words (state machine)

    private static let numericRE = try! NSRegularExpression(pattern: "^\\d+(\\.\\d+)?$")

    private func isNumeric(_ s: String) -> Bool {
        let r = NSRange(s.startIndex..., in: s)
        return Self.numericRE.firstMatch(in: s, range: r) != nil
    }

    private func processWords(_ tokens: [String]) -> [String] {
        var out: [String] = []
        var prefix: String? = nil
        var value: Value? = nil
        var skip = false

        // Windowed iteration with None bookends.
        let padded: [String?] = [nil] + tokens.map(Optional.some) + [nil]

        func outputResult(_ result: String) -> String {
            var r = result
            if let p = prefix { r = p + r }
            prefix = nil
            value = nil
            return r
        }
        func emit(_ s: String) { out.append(outputResult(s)) }

        for i in 1..<(padded.count - 1) {
            if skip { skip = false; continue }
            let prev = padded[i - 1]
            guard let current = padded[i] else { continue }
            let nextOpt = padded[i + 1]
            let nextIsNumeric = nextOpt.map(isNumeric) ?? false

            let hasPrefix = current.first.map { prefixChars.contains($0) } ?? false
            let currentNoPrefix = hasPrefix ? String(current.dropFirst()) : current

            if isNumeric(currentNoPrefix) {
                // Arabic numeric token (possibly with sign and fraction)
                if let v = value {
                    if case .str(let s) = v, s.hasSuffix(".") {
                        value = .str(s + current)
                        continue
                    } else {
                        emit(v.asString)
                    }
                }
                if hasPrefix { prefix = String(current.first!) }
                if let n = Int64(currentNoPrefix) {
                    value = .int(n)
                } else {
                    value = .str(currentNoPrefix)
                }
            } else if !words.contains(current) {
                // Non-numeric word — flush any pending value, then emit token
                if let v = value { emit(v.asString) }
                emit(current)
            } else if zeros.contains(current) {
                let base = value.map { $0.asString } ?? ""
                value = .str(base + "0")
            } else if let onesV = ones[current] {
                if value == nil {
                    value = .int(onesV)
                } else if case .str(_) = value! {
                    if let p = prev, tens.keys.contains(p), onesV < 10 {
                        // replace last zero with digit (twenty + four → 24)
                        var s = value!.asString
                        precondition(s.hasSuffix("0"))
                        s.removeLast()
                        value = .str(s + String(onesV))
                    } else {
                        value = .str(value!.asString + String(onesV))
                    }
                } else if let p = prev, ones.keys.contains(p) {
                    if let pp = prev, tens.keys.contains(pp), onesV < 10 {
                        var s = value!.asString
                        precondition(s.hasSuffix("0"))
                        s.removeLast()
                        value = .str(s + String(onesV))
                    } else {
                        value = .str(value!.asString + String(onesV))
                    }
                } else if onesV < 10, case .int(let n) = value! {
                    if n % 10 == 0 { value = .int(n + onesV) }
                    else { value = .str(value!.asString + String(onesV)) }
                } else if case .int(let n) = value! {
                    // eleven to nineteen
                    if n % 100 == 0 { value = .int(n + onesV) }
                    else { value = .str(value!.asString + String(onesV)) }
                }
            } else if let (onesV, suffix) = onesSuffixed[current] {
                if value == nil {
                    emit(String(onesV) + suffix)
                } else if case .str(_) = value! {
                    if let p = prev, tens.keys.contains(p), onesV < 10 {
                        var s = value!.asString
                        precondition(s.hasSuffix("0"))
                        s.removeLast()
                        emit(s + String(onesV) + suffix)
                    } else {
                        emit(value!.asString + String(onesV) + suffix)
                    }
                } else if let p = prev, ones.keys.contains(p) {
                    emit(value!.asString + String(onesV) + suffix)
                } else if onesV < 10, case .int(let n) = value! {
                    if n % 10 == 0 { emit(String(n + onesV) + suffix) }
                    else { emit(value!.asString + String(onesV) + suffix) }
                } else if case .int(let n) = value! {
                    if n % 100 == 0 { emit(String(n + onesV) + suffix) }
                    else { emit(value!.asString + String(onesV) + suffix) }
                }
                value = nil
            } else if let tensV = tens[current] {
                if value == nil {
                    value = .int(tensV)
                } else if case .str(_) = value! {
                    value = .str(value!.asString + String(tensV))
                } else if case .int(let n) = value! {
                    if n % 100 == 0 { value = .int(n + tensV) }
                    else { value = .str(value!.asString + String(tensV)) }
                }
            } else if let (tensV, suffix) = tensSuffixed[current] {
                if value == nil {
                    emit(String(tensV) + suffix)
                } else if case .str(_) = value! {
                    emit(value!.asString + String(tensV) + suffix)
                } else if case .int(let n) = value! {
                    if n % 100 == 0 { emit(String(n + tensV) + suffix) }
                    else { emit(value!.asString + String(tensV) + suffix) }
                }
            } else if let mult = multipliers[current] {
                if value == nil {
                    value = .int(mult)
                } else if case .str(let s) = value! {
                    if let frac = Double(s) {
                        let p = frac * Double(mult)
                        if p == p.rounded() {
                            value = .int(Int64(p))
                        } else {
                            emit(s)
                            value = .int(mult)
                        }
                    } else {
                        emit(s)
                        value = .int(mult)
                    }
                } else if case .int(let n) = value! {
                    if n == 0 {
                        // (str-or-zero branch — never reached with int=0 in normal flow)
                        emit(String(n))
                        value = .int(mult)
                    } else {
                        let before = (n / 1000) * 1000
                        let residual = n % 1000
                        value = .int(before + residual * mult)
                    }
                }
            } else if let (mult, suffix) = multipliersSuffixed[current] {
                if value == nil {
                    emit(String(mult) + suffix)
                } else if case .str(let s) = value! {
                    if let frac = Double(s) {
                        let p = frac * Double(mult)
                        if p == p.rounded() {
                            emit(String(Int64(p)) + suffix)
                        } else {
                            emit(s)
                            emit(String(mult) + suffix)
                        }
                    } else {
                        emit(s)
                        emit(String(mult) + suffix)
                    }
                } else if case .int(let n) = value! {
                    let before = (n / 1000) * 1000
                    let residual = n % 1000
                    let combined = before + residual * mult
                    value = .int(combined)
                    emit(String(combined) + suffix)
                }
                value = nil
            } else if let sign = precedingPrefixers[current] {
                if let v = value { emit(v.asString) }
                if let n = nextOpt, words.contains(n) || nextIsNumeric {
                    prefix = sign
                } else {
                    emit(current)
                }
            } else if let cur = followingPrefixers[current] {
                if let v = value {
                    prefix = cur
                    emit(v.asString)
                } else {
                    emit(current)
                }
            } else if current == "percent" {
                if let v = value {
                    emit(v.asString + "%")
                } else {
                    emit(current)
                }
            } else if current == "per" {
                if let v = value {
                    if let n = nextOpt, n == "cent" {
                        emit(v.asString + "%")
                        skip = true
                    } else {
                        emit(v.asString)
                        emit(current)
                    }
                } else {
                    emit(current)
                }
            } else if specials.contains(current) {
                let n = nextOpt
                let nextIsWord = n.map { words.contains($0) } ?? false
                if !nextIsWord && !nextIsNumeric {
                    if let v = value { emit(v.asString) }
                    emit(current)
                } else if current == "and" {
                    // ignore "and" after hundreds/thousands/etc
                    if !(prev.map { multipliers.keys.contains($0) } ?? false) {
                        if let v = value { emit(v.asString) }
                        emit(current)
                    }
                } else if current == "double" || current == "triple" {
                    if let n = nextOpt, (ones.keys.contains(n) || zeros.contains(n)) {
                        let repeats = current == "double" ? 2 : 3
                        let onesV = ones[n] ?? 0
                        let base = value.map { $0.asString } ?? ""
                        value = .str(base + String(repeating: String(onesV), count: repeats))
                        skip = true
                    } else {
                        if let v = value { emit(v.asString) }
                        emit(current)
                    }
                } else if current == "point" {
                    if let n = nextOpt, decimals.contains(n) || nextIsNumeric {
                        let base = value.map { $0.asString } ?? ""
                        value = .str(base + ".")
                    }
                }
            }
        }
        if let v = value { emit(v.asString) }
        return out
    }

    // MARK: - Regex helpers

    private func applyRegex(_ re: NSRegularExpression, on s: String, with template: String) -> String {
        let r = NSRange(s.startIndex..., in: s)
        return re.stringByReplacingMatches(in: s, range: r, withTemplate: template)
    }

    private func replaceMatches(
        in s: String,
        using re: NSRegularExpression,
        with transform: (String, NSTextCheckingResult) -> String
    ) -> String {
        let ns = s as NSString
        let matches = re.matches(in: s, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty { return s }
        var out = ""
        var cursor = 0
        for m in matches {
            let r = m.range
            if r.location > cursor {
                out += ns.substring(with: NSRange(location: cursor, length: r.location - cursor))
            }
            out += transform(s, m)
            cursor = r.location + r.length
        }
        if cursor < ns.length {
            out += ns.substring(with: NSRange(location: cursor, length: ns.length - cursor))
        }
        return out
    }
}
