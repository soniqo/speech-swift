import Foundation

/// Pushdown recogniser for "a byte prefix of a JSON document that validates against a grammar".
///
/// A matcher state is a small set of stacks, because `anyOf`, `type` lists and literal sets
/// that share a prefix (`1` and `10`) are ambiguous until more bytes arrive. Each stack is the
/// chain of open values; its top frame consumes the next byte. A stack that has popped its root
/// is a complete document. The matcher works on bytes, not characters, because tokens are byte
/// strings and a byte-fallback token can end in the middle of a UTF-8 character.
struct JSONSchemaMatcher {
    typealias Stack = [JSONFrame]

    let grammar: JSONSchemaGrammar

    /// Longest whitespace run: one space, or one newline followed by up to this many spaces/tabs.
    static let maxIndent: UInt8 = 20
    /// Integer digits accepted (keeps every prefix exact in `Int64`).
    static let maxIntegerDigits: UInt32 = 18
    /// Bytes accepted in an unbounded `number`.
    static let maxNumberBytes: UInt32 = 32

    init(grammar: JSONSchemaGrammar) { self.grammar = grammar }

    var initial: [Stack] { [[JSONFrame(kind: .value, node: grammar.root)]] }

    // MARK: - Queries

    /// True when the stacks may stop here: some stack holds a complete document.
    func canEnd(_ states: [Stack]) -> Bool { states.contains(where: canEnd) }

    /// True when every stack is a complete document, so nothing but the end can follow.
    func isDone(_ states: [Stack]) -> Bool { !states.isEmpty && states.allSatisfy(\.isEmpty) }

    func canEnd(_ stack: Stack) -> Bool {
        if stack.isEmpty { return true }
        return stack.count == 1 && completable(stack[0])
    }

    /// A top frame that is a complete value but could still be extended (`12` → `123`).
    private func completable(_ f: JSONFrame) -> Bool {
        switch f.kind {
        case .literal:
            let lits = literals(f.node)
            return bits(f.c).contains { lits[$0].count == Int(f.a) }
        case .integer:
            return f.phase == 2 && integerValid(f)
        case .number:
            return [2, 3, 5, 8].contains(f.phase)
        default:
            return false
        }
    }

    // MARK: - Advancing

    func advance(_ states: [Stack], byte: UInt8) -> [Stack] {
        var out: [Stack] = []
        for s in states { feed(s, byte, &out) }
        if out.count > 1 { out = Array(Set(out)) }
        return out
    }

    func advance(_ states: [Stack], bytes: [UInt8]) -> [Stack] {
        var current = states
        for b in bytes {
            current = advance(current, byte: b)
            if current.isEmpty { break }
        }
        return current
    }

    /// Every stack reachable from `stack` by consuming `b`, appended to `out`.
    func feed(_ stack: Stack, _ b: UInt8, _ out: inout [Stack]) {
        guard var top = stack.last else { return }   // a complete document takes no more bytes
        var s = stack
        switch top.kind {
        case .value:
            s.removeLast()
            for frame in starts(top.node) {
                var t = s
                t.append(frame)
                feed(t, b, &out)
            }

        case .literal:
            let lits = literals(top.node)
            let pos = Int(top.a)
            var alive: UInt64 = 0
            var complete = false
            for i in bits(top.c) {
                if lits[i].count == pos { complete = true }
                else if lits[i][pos] == b { alive |= 1 << UInt64(i) }
            }
            if alive != 0 {
                top.c = alive
                top.a += 1
                if bits(alive).allSatisfy({ lits[$0].count == pos + 1 }) {
                    s.removeLast()
                } else {
                    s[s.count - 1] = top
                }
                out.append(s)
            }
            if complete { feed(Array(stack.dropLast()), b, &out) }

        case .string:
            feedString(top, stack: s, b, &out)

        case .integer:
            feedInteger(top, stack: s, b, &out)

        case .number:
            feedNumber(top, stack: s, b, &out)

        case .array:
            guard case .array(let items, let minItems, let maxItems) = grammar.nodes[Int(top.node)] else { return }
            if top.phase == 255 {
                guard b == 0x5B else { return }
                top.phase = 0
                s[s.count - 1] = top
                out.append(s)
                return
            }
            if Self.isWhitespace(b) {
                guard let w = Self.nextWhitespace(top.ws, b) else { return }
                top.ws = w
                s[s.count - 1] = top
                out.append(s)
                return
            }
            switch top.phase {
            case 0, 2:
                if top.phase == 0, b == 0x5D {
                    guard minItems == 0 else { return }
                    s.removeLast()
                    out.append(s)
                    return
                }
                if let maxItems, Int(top.a) >= maxItems { return }
                top.a += 1
                top.phase = 1
                top.ws = 0
                s[s.count - 1] = top
                s.append(JSONFrame(kind: .value, node: items))
                feed(s, b, &out)
            case 1:
                if b == 0x2C {
                    if let maxItems, Int(top.a) >= maxItems { return }
                    top.phase = 2
                    top.ws = 0
                    s[s.count - 1] = top
                    out.append(s)
                } else if b == 0x5D {
                    guard Int(top.a) >= minItems else { return }
                    s.removeLast()
                    out.append(s)
                }
            default: return
            }

        case .object:
            feedObject(top, stack: s, b, &out)

        case .key:
            guard case .object(let spec) = grammar.nodes[Int(top.node)] else { return }
            let pos = Int(top.a)
            if b == 0x22 {
                guard let i = bits(top.c).first(where: { spec.properties[$0].key.count == pos }) else { return }
                s[s.count - 1] = JSONFrame(kind: .object, node: top.node, phase: 1, a: UInt32(i + 1), b: top.b + 1)
                out.append(s)
                return
            }
            var alive: UInt64 = 0
            for i in bits(top.c) {
                let key = spec.properties[i].key
                if key.count > pos, key[pos] == b { alive |= 1 << UInt64(i) }
            }
            guard alive != 0 else { return }
            top.c = alive
            top.a += 1
            s[s.count - 1] = top
            out.append(s)
        }
    }

    // MARK: - Objects

    private func feedObject(_ frame: JSONFrame, stack: Stack, _ b: UInt8, _ out: inout [Stack]) {
        guard case .object(let spec) = grammar.nodes[Int(frame.node)] else { return }
        var top = frame
        var s = stack
        func emit() { s[s.count - 1] = top; out.append(s) }
        if top.phase == 255 {
            guard b == 0x7B else { return }
            top.phase = 0
            emit()
            return
        }
        if Self.isWhitespace(b) {
            guard let w = Self.nextWhitespace(top.ws, b) else { return }
            top.ws = w
            emit()
            return
        }
        let next = Int(top.a), count = Int(top.b)
        switch top.phase {
        case 0, 4:
            if b == 0x22 {
                if spec.freeValues != nil {
                    if let maxP = spec.maxProperties, count >= maxP { return }
                    top.phase = 1
                    top.ws = 0
                    top.b += 1
                    s[s.count - 1] = top
                    s.append(JSONFrame(kind: .string, node: grammar.freeString, phase: 255))
                    feed(s, b, &out)
                } else {
                    let candidates = keyCandidates(spec, next: next, count: count)
                    guard candidates != 0 else { return }
                    s[s.count - 1] = JSONFrame(kind: .key, node: top.node, a: 0, b: top.b, c: candidates)
                    out.append(s)
                }
            } else if b == 0x7D, top.phase == 0 {
                guard canClose(spec, next: next, count: count) else { return }
                s.removeLast()
                out.append(s)
            }
        case 1:
            guard b == 0x3A else { return }
            top.phase = 2
            top.ws = 0
            emit()
        case 2:
            let valueNode = spec.freeValues ?? spec.properties[next - 1].node
            top.phase = 3
            top.ws = 0
            s[s.count - 1] = top
            s.append(JSONFrame(kind: .value, node: valueNode))
            feed(s, b, &out)
        case 3:
            if b == 0x2C {
                if spec.freeValues != nil {
                    if let maxP = spec.maxProperties, count >= maxP { return }
                } else if keyCandidates(spec, next: next, count: count) == 0 {
                    return
                }
                top.phase = 4
                top.ws = 0
                emit()
            } else if b == 0x7D {
                guard canClose(spec, next: next, count: count) else { return }
                s.removeLast()
                out.append(s)
            }
        default: return
        }
    }

    /// Declared properties that may be written next: those from `next` up to and including the
    /// first required one, as long as the property counts can still be met.
    private func keyCandidates(_ spec: JSONSchemaGrammar.ObjectSpec, next: Int, count: Int) -> UInt64 {
        if let maxP = spec.maxProperties, count >= maxP { return 0 }
        var mask: UInt64 = 0
        var p = next
        while p < spec.properties.count {
            let after = count + 1
            let fitsMax = spec.maxProperties.map { after + spec.requiredFrom[p + 1] <= $0 } ?? true
            let fitsMin = after + (spec.properties.count - p - 1) >= spec.minProperties
            if fitsMax && fitsMin { mask |= 1 << UInt64(p) }
            if spec.properties[p].required { break }
            p += 1
        }
        return mask
    }

    private func canClose(_ spec: JSONSchemaGrammar.ObjectSpec, next: Int, count: Int) -> Bool {
        count >= spec.minProperties && (spec.freeValues != nil || spec.requiredFrom[next] == 0)
    }

    // MARK: - Strings

    private func feedString(_ frame: JSONFrame, stack: Stack, _ b: UInt8, _ out: inout [Stack]) {
        guard case .string(let spec) = grammar.nodes[Int(frame.node)] else { return }
        var top = frame
        var s = stack
        func emit() { s[s.count - 1] = top; out.append(s) }
        func room() -> Bool { spec.maxLength.map { Int(top.a) < $0 } ?? true }
        /// A partly written character can still become one the pattern accepts.
        func viable(_ range: ClosedRange<UInt32>) -> Bool {
            spec.pattern.map { $0.canConsume(top.c, in: range) } ?? true
        }
        /// A match is still reachable within `maxLength`.
        func patternFits(_ pattern: JSONPattern) -> Bool {
            guard let maxLength = spec.maxLength else { return true }
            let need = pattern.minimumToAccept(top.c)
            return need != .max && Int(top.a) + need <= maxLength
        }
        /// One more character of content, `scalar`.
        func character(_ scalar: UInt32) {
            guard room() else { return }
            top.a += 1
            if let pattern = spec.pattern {
                top.c = pattern.step(top.c, scalar)
                guard top.c != 0, patternFits(pattern) else { return }
            }
            top.phase = 0
            emit()
        }

        switch top.phase {
        case 255:
            guard b == 0x22 else { return }
            top.phase = 0
            top.c = spec.pattern?.start ?? 0
            if let pattern = spec.pattern, !patternFits(pattern) { return }
            emit()
        case 1:
            let escaped: UInt32
            switch b {
            case 0x22: escaped = 0x22
            case 0x5C: escaped = 0x5C
            case 0x2F: escaped = 0x2F
            case 0x62: escaped = 0x08
            case 0x66: escaped = 0x0C
            case 0x6E: escaped = 0x0A
            case 0x72: escaped = 0x0D
            case 0x74: escaped = 0x09
            case 0x75:
                guard viable(0 ... 0xFFFF) else { return }
                top.phase = 2
                top.d = 0
                emit()
                return
            default: return
            }
            character(escaped)
        case 2 ... 5:
            guard let h = Self.hexValue(b) else { return }
            top.d = top.d << 4 | h
            if top.phase < 5 {
                // The escape must still be able to name a non-surrogate the pattern accepts.
                let unknown = UInt32(4 * (5 - top.phase))
                let lo = top.d << unknown, hi = ((top.d + 1) << unknown) - 1
                let below = lo < 0xD800 ? lo ... min(hi, 0xD7FF) : nil
                let above = hi > 0xDFFF ? max(lo, 0xE000) ... hi : nil
                guard [below, above].contains(where: { $0.map(viable) ?? false }) else { return }
                top.phase += 1
                emit()
            } else {
                // Surrogate halves are refused: astral characters are written as raw UTF-8.
                guard !(0xD800 ... 0xDFFF).contains(top.d) else { return }
                character(top.d)
            }
        default:
            let remaining = top.b & 0xF
            if remaining > 0 {
                let lo = UInt8((top.b >> 8) & 0xFF), hi = UInt8((top.b >> 16) & 0xFF)
                guard b >= lo, b <= hi else { return }
                top.d = top.d << 6 | UInt32(b & 0x3F)
                if remaining == 1 {
                    top.b = 0
                    character(top.d)
                } else {
                    let unknown = 6 * (remaining - 1)
                    guard viable(top.d << unknown ... ((top.d + 1) << unknown) - 1) else { return }
                    top.b = (remaining - 1) | 0x80 << 8 | 0xBF << 16
                    emit()
                }
                return
            }
            switch b {
            case 0x22:
                guard Int(top.a) >= spec.minLength else { return }
                if spec.pattern != nil, top.c & JSONPattern.acceptBit == 0 { return }
                s.removeLast()
                out.append(s)
            case 0x5C:
                // Every escape denotes a character in 0…U+FFFF.
                guard room(), viable(0 ... 0xFFFF) else { return }
                top.phase = 1
                emit()
            case 0 ..< 0x20:
                return
            case 0x20 ..< 0x80:
                character(UInt32(b))
            default:
                // A multi-byte UTF-8 lead: remember how many continuation bytes follow and the
                // range the first of them must fall in (no overlongs, no surrogates, ≤ U+10FFFF).
                let need: UInt32, lo: UInt32, hi: UInt32, bitsValue: UInt32
                switch b {
                case 0xC2 ... 0xDF: need = 1; lo = 0x80; hi = 0xBF; bitsValue = UInt32(b & 0x1F)
                case 0xE0: need = 2; lo = 0xA0; hi = 0xBF; bitsValue = UInt32(b & 0x0F)
                case 0xED: need = 2; lo = 0x80; hi = 0x9F; bitsValue = UInt32(b & 0x0F)
                case 0xE1 ... 0xEF: need = 2; lo = 0x80; hi = 0xBF; bitsValue = UInt32(b & 0x0F)
                case 0xF0: need = 3; lo = 0x90; hi = 0xBF; bitsValue = UInt32(b & 0x07)
                case 0xF1 ... 0xF3: need = 3; lo = 0x80; hi = 0xBF; bitsValue = UInt32(b & 0x07)
                case 0xF4: need = 3; lo = 0x80; hi = 0x8F; bitsValue = UInt32(b & 0x07)
                default: return
                }
                // The characters this lead byte can still become.
                let unknown = 6 * need
                let first = [UInt32(0x80), 0x800, 0x10000][Int(need) - 1]
                var span = max(bitsValue << unknown, first) ... ((bitsValue + 1) << unknown) - 1
                if b == 0xED { span = span.lowerBound ... 0xD7FF }
                if b == 0xF4 { span = span.lowerBound ... 0x10FFFF }
                guard room(), viable(span) else { return }
                top.b = need | lo << 8 | hi << 16
                top.d = bitsValue
                emit()
            }
        }
    }

    // MARK: - Numbers

    private func feedInteger(_ frame: JSONFrame, stack: Stack, _ b: UInt8, _ out: inout [Stack]) {
        var top = frame
        var s = stack
        let isDigit = (0x30 ... 0x39).contains(b)
        switch top.phase {
        case 255, 1:
            if top.phase == 255, b == 0x2D {
                top.b = 1
                top.phase = 1
                guard integerFeasible(top, extending: true) else { return }
                s[s.count - 1] = top
                out.append(s)
                return
            }
            guard isDigit else { return }
            top.c = UInt64(b - 0x30)
            top.a = 1
            top.phase = 2
        default:
            if !isDigit {
                guard integerValid(top) else { return }
                feed(Array(stack.dropLast()), b, &out)
                return
            }
            guard !(top.a == 1 && top.c == 0), top.a < Self.maxIntegerDigits else { return }
            top.c = top.c * 10 + UInt64(b - 0x30)
            top.a += 1
        }
        guard integerFeasible(top, extending: false) || integerFeasible(top, extending: true) else { return }
        if integerValid(top), !integerFeasible(top, extending: true) {
            s.removeLast()           // nothing can follow this digit: the integer is finished
        } else {
            s[s.count - 1] = top
        }
        out.append(s)
    }

    private func integerBounds(_ node: Int32) -> (Int64?, Int64?) {
        guard case .integer(let lo, let hi) = grammar.nodes[Int(node)] else { return (nil, nil) }
        return (lo, hi)
    }

    private func integerValid(_ f: JSONFrame) -> Bool {
        guard f.a >= 1 else { return false }
        let (lo, hi) = integerBounds(f.node)
        let v = f.b & 1 == 1 ? -Int64(f.c) : Int64(f.c)
        return (lo.map { v >= $0 } ?? true) && (hi.map { v <= $0 } ?? true)
    }

    /// Whether some continuation of the digits so far (`extending`: at least one more digit;
    /// otherwise exactly these) lies in range.
    private func integerFeasible(_ f: JSONFrame, extending: Bool) -> Bool {
        let (lo, hi) = integerBounds(f.node)
        let negative = f.b & 1 == 1
        let digits = Int(f.a)
        if !extending { return integerValid(f) }
        if digits >= 1 && f.c == 0 { return false }        // no digits after a leading zero
        var scale: Int64 = 1
        for k in 1 ... max(1, Int(Self.maxIntegerDigits) - digits) {
            guard digits + k <= Int(Self.maxIntegerDigits) else { break }
            scale *= 10
            let magLo = digits == 0 ? (k == 1 ? 0 : scale / 10) : Int64(f.c) * scale
            let magHi = digits == 0 ? scale - 1 : Int64(f.c) * scale + scale - 1
            let vLo = negative ? -magHi : magLo
            let vHi = negative ? -magLo : magHi
            if (hi.map { vLo <= $0 } ?? true) && (lo.map { vHi >= $0 } ?? true) { return true }
        }
        return false
    }

    private func feedNumber(_ frame: JSONFrame, stack: Stack, _ b: UInt8, _ out: inout [Stack]) {
        var top = frame
        var s = stack
        let isDigit = (0x30 ... 0x39).contains(b)
        let next: UInt8?
        switch (top.phase, b) {
        case (255, 0x2D): next = 1
        case (255, 0x30), (1, 0x30): next = 2
        case (255, _) where isDigit, (1, _) where isDigit: next = 3
        case (3, _) where isDigit: next = 3
        case (2, 0x2E), (3, 0x2E): next = 4
        case (4, _) where isDigit, (5, _) where isDigit: next = 5
        case (2, 0x65), (2, 0x45), (3, 0x65), (3, 0x45), (5, 0x65), (5, 0x45): next = 6
        case (6, 0x2B), (6, 0x2D): next = 7
        case (6, _) where isDigit, (7, _) where isDigit, (8, _) where isDigit: next = 8
        default: next = nil
        }
        if let next, top.a < Self.maxNumberBytes {
            top.phase = next
            top.a += 1
            s[s.count - 1] = top
            out.append(s)
        } else if [2, 3, 5, 8].contains(top.phase) {
            feed(Array(stack.dropLast()), b, &out)
        }
    }

    // MARK: - Helpers

    /// The frames a value of `node` may begin as.
    private func starts(_ node: Int32) -> [JSONFrame] {
        switch grammar.nodes[Int(node)] {
        case .literals(let lits):
            return [JSONFrame(kind: .literal, node: node, a: 0, c: lits.count == 64 ? ~0 : (1 << UInt64(lits.count)) - 1)]
        case .string: return [JSONFrame(kind: .string, node: node, phase: 255)]
        case .integer: return [JSONFrame(kind: .integer, node: node, phase: 255)]
        case .number: return [JSONFrame(kind: .number, node: node, phase: 255)]
        case .array: return [JSONFrame(kind: .array, node: node, phase: 255)]
        case .object: return [JSONFrame(kind: .object, node: node, phase: 255)]
        case .union(let members): return members.flatMap(starts)
        }
    }

    private func literals(_ node: Int32) -> [[UInt8]] {
        guard case .literals(let lits) = grammar.nodes[Int(node)] else { return [] }
        return lits
    }

    static func isWhitespace(_ b: UInt8) -> Bool { b == 0x20 || b == 0x0A || b == 0x09 || b == 0x0D }

    /// Whitespace state after `b`: 0 none, 1 one space (closed), 2 + n a newline and n indent.
    static func nextWhitespace(_ w: UInt8, _ b: UInt8) -> UInt8? {
        switch b {
        case 0x20:
            if w == 0 { return 1 }
            return (w >= 2 && w < 2 + maxIndent) ? w + 1 : nil
        case 0x09: return (w >= 2 && w < 2 + maxIndent) ? w + 1 : nil
        case 0x0A: return w == 0 ? 2 : nil
        default: return nil
        }
    }

    static func hexValue(_ b: UInt8) -> UInt32? {
        switch b {
        case 0x30 ... 0x39: UInt32(b - 0x30)
        case 0x41 ... 0x46: UInt32(b - 0x41 + 10)
        case 0x61 ... 0x66: UInt32(b - 0x61 + 10)
        default: nil
        }
    }
}

/// Indices of the set bits of `mask`, lowest first.
@inline(__always)
func bits(_ mask: UInt64) -> [Int] {
    var out: [Int] = []
    var m = mask
    while m != 0 {
        out.append(m.trailingZeroBitCount)
        m &= m - 1
    }
    return out
}

/// One open value on a matcher stack. Fields are reused per kind:
///
/// | kind    | phase                       | a            | b                    | c              | d        |
/// |---------|-----------------------------|--------------|----------------------|----------------|----------|
/// | literal | –                           | position     | –                    | live literals  | –        |
/// | string  | 255 unopened, 0, 1 `\`, 2–5 `\u` | characters | UTF-8 continuation state | pattern state | code point |
/// | integer | 255, 1 after `-`, 2 digits  | digits       | 1 if negative        | magnitude      | –        |
/// | number  | 255, grammar state 1–8      | bytes        | –                    | –              | –        |
/// | array   | 255, 0 open, 1 item, 2 comma| items        | –                    | –              | –        |
/// | object  | 255, 0 open, 1 key, 2 colon, 3 value, 4 comma | next property | properties | – | –   |
/// | key     | –                           | position     | properties           | live keys      | –        |
struct JSONFrame: Hashable {
    enum Kind: UInt8 { case value, literal, string, integer, number, array, object, key }

    var kind: Kind
    var node: Int32
    var phase: UInt8 = 0
    var ws: UInt8 = 0
    var a: UInt32 = 0
    var b: UInt32 = 0
    var c: UInt64 = 0
    var d: UInt32 = 0
}
