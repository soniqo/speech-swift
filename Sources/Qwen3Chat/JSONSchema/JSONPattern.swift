import Foundation

/// A JSON Schema `pattern`, compiled to a small NFA over Unicode scalars.
///
/// The subset: literal characters and escapes (`\\`, `\/`, `\.`, `\t`, `\n`, `\r`, `\uXXXX`, …),
/// `.`, the classes `\d \D \w \W \s \S`, bracket classes with ranges and negation, groups `(…)`
/// and `(?:…)`, alternation `|`, quantifiers `* + ? {n} {n,} {n,m}`, and `^` / `$` at the very
/// start and end. Like JSON Schema, an unanchored pattern matches anywhere in the string.
/// Backreferences, lookaround, word boundaries, lazy quantifiers and Unicode property escapes are
/// rejected. The NFA may have at most 63 character positions.
///
/// Simulation state is one `UInt64`: bit `i` means position `i` may consume the next character,
/// and ``acceptBit`` means the characters so far match.
struct JSONPattern: Hashable {
    static let acceptBit: UInt64 = 1 << 63

    /// Character predicate of each position.
    let classes: [CharClass]
    /// Positions (and possibly ``acceptBit``) enabled after position `i` consumes a character.
    let follow: [UInt64]
    let start: UInt64
    /// Fewest characters from after position `i` to a match (`Int.max` if none).
    let remainingAfter: [Int]

    /// Fewest further characters that complete a match from `set`, or `Int.max`.
    func minimumToAccept(_ set: UInt64) -> Int {
        if set & Self.acceptBit != 0 { return 0 }
        var best = Int.max
        var bits = set
        while bits != 0 {
            let i = bits.trailingZeroBitCount
            bits &= bits - 1
            if remainingAfter[i] != .max { best = min(best, 1 + remainingAfter[i]) }
        }
        return best
    }

    func step(_ set: UInt64, _ scalar: UInt32) -> UInt64 {
        var result: UInt64 = 0
        var bits = set & ~Self.acceptBit
        while bits != 0 {
            let i = bits.trailingZeroBitCount
            bits &= bits - 1
            if classes[i].contains(scalar) { result |= follow[i] }
        }
        return result
    }

    /// Whether some character in `range` can be consumed from `set`. A string that has begun an
    /// escape or a multi-byte character only continues while the character it can still become
    /// might match; otherwise the matcher would admit bytes that lead nowhere.
    func canConsume(_ set: UInt64, in range: ClosedRange<UInt32>) -> Bool {
        var bits = set & ~Self.acceptBit
        while bits != 0 {
            let i = bits.trailingZeroBitCount
            bits &= bits - 1
            if classes[i].intersects(range) { return true }
        }
        return false
    }

    static func compile(_ source: String, path: String) throws -> JSONPattern {
        var parser = RegexParser(scalars: Array(source.unicodeScalars), path: path)
        var anchoredStart = false, anchoredEnd = false
        if parser.scalars.first == "^" { anchoredStart = true; parser.index = 1 }
        if parser.scalars.count > parser.index, parser.scalars.last == "$",
           !(parser.scalars.count >= 2 && parser.scalars[parser.scalars.count - 2] == "\\") {
            anchoredEnd = true
            parser.end = parser.scalars.count - 1
        }
        var ast = try parser.alternation()
        guard parser.index == parser.end else { throw parser.fail("unexpected ')'") }
        let any = RegexNode.star(.char(CharClass.anyScalar))
        if !anchoredStart { ast = .concat([any, ast]) }
        if !anchoredEnd { ast = .concat([ast, any]) }

        var builder = Glushkov()
        let info = try builder.build(ast, path: path)
        var follow = builder.follow
        var start = info.first
        if info.nullable { start |= acceptBit }
        for i in 0 ..< follow.count where info.last & (1 << UInt64(i)) != 0 { follow[i] |= acceptBit }
        // Shortest distances to acceptance, by relaxation over at most 63 positions.
        var remaining = follow.map { $0 & acceptBit != 0 ? 0 : Int.max }
        var changed = true
        while changed {
            changed = false
            for i in 0 ..< follow.count {
                var bits = follow[i] & ~acceptBit
                while bits != 0 {
                    let j = bits.trailingZeroBitCount
                    bits &= bits - 1
                    if remaining[j] != .max, remaining[j] + 1 < remaining[i] {
                        remaining[i] = remaining[j] + 1
                        changed = true
                    }
                }
            }
        }
        return JSONPattern(classes: builder.classes, follow: follow, start: start,
                           remainingAfter: remaining)
    }
}

/// A set of Unicode scalars: sorted, disjoint, inclusive ranges.
struct CharClass: Hashable {
    var ranges: [ClosedRange<UInt32>]

    static let maxScalar: UInt32 = 0x10FFFF
    static let anyScalar = CharClass(ranges: [0 ... maxScalar])
    static let digit = CharClass(ranges: [0x30 ... 0x39])
    static let word = CharClass(ranges: [0x30 ... 0x39, 0x41 ... 0x5A, 0x5F ... 0x5F, 0x61 ... 0x7A])
    /// ECMA-262 `\s`: WhiteSpace plus LineTerminator.
    static let space = CharClass(ranges: [
        0x09 ... 0x0D, 0x20 ... 0x20, 0xA0 ... 0xA0, 0x1680 ... 0x1680, 0x2000 ... 0x200A,
        0x2028 ... 0x2029, 0x202F ... 0x202F, 0x205F ... 0x205F, 0x3000 ... 0x3000, 0xFEFF ... 0xFEFF,
    ])
    /// ECMA-262 `.`: everything except line terminators.
    static let dot = CharClass(ranges: [0x0A ... 0x0A, 0x0D ... 0x0D, 0x2028 ... 0x2029]).inverted

    func contains(_ s: UInt32) -> Bool {
        var lo = 0, hi = ranges.count - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if s < ranges[mid].lowerBound { hi = mid - 1 }
            else if s > ranges[mid].upperBound { lo = mid + 1 }
            else { return true }
        }
        return false
    }

    func intersects(_ r: ClosedRange<UInt32>) -> Bool {
        var lo = 0, hi = ranges.count - 1
        while lo <= hi {
            let mid = (lo + hi) / 2
            if ranges[mid].upperBound < r.lowerBound { lo = mid + 1 }
            else if ranges[mid].lowerBound > r.upperBound { hi = mid - 1 }
            else { return true }
        }
        return false
    }

    static func union(_ parts: [CharClass]) -> CharClass {
        let all = parts.flatMap(\.ranges).sorted { $0.lowerBound < $1.lowerBound }
        var merged: [ClosedRange<UInt32>] = []
        for r in all {
            if let last = merged.last, r.lowerBound <= last.upperBound &+ 1 {
                merged[merged.count - 1] = last.lowerBound ... max(last.upperBound, r.upperBound)
            } else {
                merged.append(r)
            }
        }
        return CharClass(ranges: merged)
    }

    var inverted: CharClass {
        var out: [ClosedRange<UInt32>] = []
        var next: UInt32 = 0
        for r in ranges {
            if r.lowerBound > next { out.append(next ... r.lowerBound - 1) }
            next = r.upperBound &+ 1
            if r.upperBound >= Self.maxScalar { next = Self.maxScalar + 1 }
        }
        if next <= Self.maxScalar { out.append(next ... Self.maxScalar) }
        return CharClass(ranges: out)
    }
}

indirect enum RegexNode {
    case empty
    case char(CharClass)
    case concat([RegexNode])
    case alt([RegexNode])
    case star(RegexNode)
    case optional(RegexNode)
}

private struct RegexParser {
    let scalars: [Unicode.Scalar]
    let path: String
    var index = 0
    var end: Int

    init(scalars: [Unicode.Scalar], path: String) {
        self.scalars = scalars
        self.path = path
        self.end = scalars.count
    }

    func fail(_ why: String) -> ChatResponseFormatError {
        .unsupported(keyword: "pattern (\(why))", path: path)
    }

    func peek() -> Unicode.Scalar? { index < end ? scalars[index] : nil }

    mutating func alternation() throws -> RegexNode {
        var branches = [try concatenation()]
        while peek() == "|" {
            index += 1
            branches.append(try concatenation())
        }
        return branches.count == 1 ? branches[0] : .alt(branches)
    }

    mutating func concatenation() throws -> RegexNode {
        var items: [RegexNode] = []
        while let c = peek(), c != "|", c != ")" {
            var atom = try self.atom()
            atom = try quantified(atom)
            items.append(atom)
        }
        return items.isEmpty ? .empty : (items.count == 1 ? items[0] : .concat(items))
    }

    mutating func quantified(_ atom: RegexNode) throws -> RegexNode {
        guard let c = peek() else { return atom }
        var result: RegexNode
        switch c {
        case "*": index += 1; result = .star(atom)
        case "+": index += 1; result = .concat([atom, .star(atom)])
        case "?": index += 1; result = .optional(atom)
        case "{":
            guard let (lo, hi) = try bounds() else { return atom }
            var parts = Array(repeating: atom, count: lo)
            if let hi {
                guard hi >= lo else { throw fail("{n,m} with m < n") }
                parts += Array(repeating: RegexNode.optional(atom), count: hi - lo)
            } else {
                parts.append(.star(atom))
            }
            result = parts.isEmpty ? .empty : .concat(parts)
        default: return atom
        }
        if peek() == "?" { throw fail("lazy quantifier") }
        if let c = peek(), "*+?{".unicodeScalars.contains(c) {
            if c != "{" { throw fail("stacked quantifier") }
        }
        return result
    }

    /// Parses `{n}`, `{n,}` or `{n,m}` at `index`; nil (index unchanged) if it is a literal `{`.
    mutating func bounds() throws -> (Int, Int?)? {
        let save = index
        index += 1
        func number() -> Int? {
            var v: Int?
            while let c = peek(), let d = Int(String(c)), c.isASCII {
                v = (v ?? 0) * 10 + d
                index += 1
                if v! > 1000 { return v }
            }
            return v
        }
        guard let lo = number() else { index = save; return nil }
        var hi: Int? = lo
        if peek() == "," {
            index += 1
            hi = number()
        }
        guard peek() == "}" else { index = save; return nil }
        index += 1
        guard lo <= 64, (hi ?? 0) <= 64 else { throw fail("repetition bound above 64") }
        return (lo, hi)
    }

    mutating func atom() throws -> RegexNode {
        let c = scalars[index]
        index += 1
        switch c {
        case ".": return .char(CharClass.dot)
        case "(":
            if peek() == "?" {
                index += 1
                guard peek() == ":" else { throw fail("lookaround or named group") }
                index += 1
            }
            let inner = try alternation()
            guard peek() == ")" else { throw fail("unclosed group") }
            index += 1
            return inner
        case "[": return .char(try bracket())
        case "\\": return .char(try escape(inClass: false))
        case "^", "$": throw fail("anchor inside the pattern")
        case "*", "+", "?": throw fail("quantifier without operand")
        default: return .char(CharClass(ranges: [c.value ... c.value]))
        }
    }

    mutating func escape(inClass: Bool) throws -> CharClass {
        guard let c = peek() else { throw fail("trailing backslash") }
        index += 1
        func single(_ v: UInt32) -> CharClass { CharClass(ranges: [v ... v]) }
        switch c {
        case "d": return .digit
        case "D": return CharClass.digit.inverted
        case "w": return .word
        case "W": return CharClass.word.inverted
        case "s": return .space
        case "S": return CharClass.space.inverted
        case "t": return single(0x09)
        case "n": return single(0x0A)
        case "r": return single(0x0D)
        case "f": return single(0x0C)
        case "v": return single(0x0B)
        case "0": return single(0)
        case "b" where inClass: return single(0x08)
        case "u":
            guard index + 4 <= end,
                  let v = UInt32(String(String.UnicodeScalarView(scalars[index ..< index + 4])), radix: 16)
            else { throw fail("invalid \\u escape") }
            index += 4
            return single(v)
        case "x":
            guard index + 2 <= end,
                  let v = UInt32(String(String.UnicodeScalarView(scalars[index ..< index + 2])), radix: 16)
            else { throw fail("invalid \\x escape") }
            index += 2
            return single(v)
        default:
            if c.properties.isAlphabetic || c.properties.numericType != nil {
                throw fail("escape \\\(c)")
            }
            return single(c.value)
        }
    }

    mutating func bracket() throws -> CharClass {
        var negated = false
        if peek() == "^" { negated = true; index += 1 }
        var parts: [CharClass] = []
        var first = true
        while true {
            guard let c = peek() else { throw fail("unclosed character class") }
            if c == "]" && !first { index += 1; break }
            first = false
            let lower: CharClass
            if c == "\\" {
                index += 1
                lower = try escape(inClass: true)
            } else {
                index += 1
                lower = CharClass(ranges: [c.value ... c.value])
            }
            // A range `a-z`: only between two single characters.
            if peek() == "-", index + 1 < end, scalars[index + 1] != "]",
               lower.ranges.count == 1, lower.ranges[0].count == 1 {
                index += 1
                let hiScalar = scalars[index]
                let upper: CharClass
                if hiScalar == "\\" {
                    index += 1
                    upper = try escape(inClass: true)
                } else {
                    index += 1
                    upper = CharClass(ranges: [hiScalar.value ... hiScalar.value])
                }
                guard upper.ranges.count == 1, upper.ranges[0].count == 1 else {
                    throw fail("range to a class")
                }
                let lo = lower.ranges[0].lowerBound, hi = upper.ranges[0].lowerBound
                guard lo <= hi else { throw fail("reversed range") }
                parts.append(CharClass(ranges: [lo ... hi]))
            } else {
                parts.append(lower)
            }
        }
        let set = CharClass.union(parts)
        return negated ? set.inverted : set
    }
}

/// Glushkov construction: one NFA position per character leaf, no epsilon transitions.
private struct Glushkov {
    var classes: [CharClass] = []
    var follow: [UInt64] = []

    struct Info { var nullable: Bool; var first: UInt64; var last: UInt64 }

    mutating func build(_ node: RegexNode, path: String) throws -> Info {
        switch node {
        case .empty: return Info(nullable: true, first: 0, last: 0)
        case .char(let set):
            guard classes.count < 63 else {
                throw ChatResponseFormatError.unsupported(
                    keyword: "pattern (more than 63 character positions)", path: path)
            }
            let bit: UInt64 = 1 << UInt64(classes.count)
            classes.append(set)
            follow.append(0)
            return Info(nullable: false, first: bit, last: bit)
        case .concat(let items):
            var acc = Info(nullable: true, first: 0, last: 0)
            for item in items {
                let info = try build(item, path: path)
                link(acc.last, info.first)
                acc = Info(
                    nullable: acc.nullable && info.nullable,
                    first: acc.nullable ? acc.first | info.first : acc.first,
                    last: info.nullable ? acc.last | info.last : info.last)
            }
            return acc
        case .alt(let branches):
            var acc = Info(nullable: false, first: 0, last: 0)
            for b in branches {
                let info = try build(b, path: path)
                acc = Info(nullable: acc.nullable || info.nullable,
                           first: acc.first | info.first, last: acc.last | info.last)
            }
            return acc
        case .star(let inner):
            let info = try build(inner, path: path)
            link(info.last, info.first)
            return Info(nullable: true, first: info.first, last: info.last)
        case .optional(let inner):
            var info = try build(inner, path: path)
            info.nullable = true
            return info
        }
    }

    private mutating func link(_ from: UInt64, _ to: UInt64) {
        var bits = from
        while bits != 0 {
            let i = bits.trailingZeroBitCount
            bits &= bits - 1
            follow[i] |= to
        }
    }
}
