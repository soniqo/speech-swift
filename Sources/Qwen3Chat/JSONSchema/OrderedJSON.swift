import Foundation

/// A JSON value that keeps object members in the order they were written.
///
/// `JSONSerialization` returns dictionaries, which lose member order; the constraint needs that
/// order to write a schema's optional properties deterministically in the author's order.
indirect enum OrderedJSON {
    case null
    case bool(Bool)
    /// The number's literal text as written, and its value.
    case number(String, Double)
    case string(String)
    case array([OrderedJSON])
    case object([(key: String, value: OrderedJSON)])

    subscript(key: String) -> OrderedJSON? {
        guard case .object(let members) = self else { return nil }
        return members.first { $0.key == key }?.value
    }

    var typeName: String {
        switch self {
        case .null: "null"
        case .bool: "boolean"
        case .number(_, let v): (v.rounded() == v && abs(v) < 1e18) ? "integer" : "number"
        case .string: "string"
        case .array: "array"
        case .object: "object"
        }
    }

    /// Compact JSON text: strings escaped as `JSONEncoding.escape` does, numbers as written.
    var canonicalBytes: [UInt8] {
        var out: [UInt8] = []
        write(into: &out)
        return out
    }

    private func write(into out: inout [UInt8]) {
        switch self {
        case .null: out += Array("null".utf8)
        case .bool(let b): out += Array((b ? "true" : "false").utf8)
        case .number(let text, _): out += Array(text.utf8)
        case .string(let s):
            out.append(0x22); out += JSONEncoding.escape(s); out.append(0x22)
        case .array(let items):
            out.append(0x5B)
            for (i, item) in items.enumerated() {
                if i > 0 { out.append(0x2C) }
                item.write(into: &out)
            }
            out.append(0x5D)
        case .object(let members):
            out.append(0x7B)
            for (i, m) in members.enumerated() {
                if i > 0 { out.append(0x2C) }
                out.append(0x22); out += JSONEncoding.escape(m.key); out.append(0x22)
                out.append(0x3A)
                m.value.write(into: &out)
            }
            out.append(0x7D)
        }
    }

    static func parse(_ text: String) throws -> OrderedJSON {
        var parser = Parser(bytes: Array(text.utf8))
        parser.skipWhitespace()
        let value = try parser.value(depth: 0)
        parser.skipWhitespace()
        guard parser.index == parser.bytes.count else { throw parser.error("trailing characters") }
        return value
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        func error(_ what: String) -> ChatResponseFormatError {
            .invalidSchema("\(what) at byte \(index)")
        }

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        mutating func expect(_ literal: String) throws {
            let l = Array(literal.utf8)
            guard index + l.count <= bytes.count, Array(bytes[index ..< index + l.count]) == l else {
                throw error("expected \(literal)")
            }
            index += l.count
        }

        mutating func value(depth: Int) throws -> OrderedJSON {
            guard depth < 128 else { throw error("nesting too deep") }
            guard index < bytes.count else { throw error("unexpected end") }
            switch bytes[index] {
            case 0x7B: // {
                index += 1
                var members: [(key: String, value: OrderedJSON)] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == 0x7D { index += 1; return .object(members) }
                while true {
                    skipWhitespace()
                    guard index < bytes.count, bytes[index] == 0x22 else { throw error("expected key") }
                    let key = try string()
                    skipWhitespace()
                    try expect(":")
                    skipWhitespace()
                    members.append((key, try value(depth: depth + 1)))
                    skipWhitespace()
                    guard index < bytes.count else { throw error("unterminated object") }
                    if bytes[index] == 0x2C { index += 1; continue }
                    if bytes[index] == 0x7D { index += 1; return .object(members) }
                    throw error("expected , or }")
                }
            case 0x5B: // [
                index += 1
                var items: [OrderedJSON] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == 0x5D { index += 1; return .array(items) }
                while true {
                    skipWhitespace()
                    items.append(try value(depth: depth + 1))
                    skipWhitespace()
                    guard index < bytes.count else { throw error("unterminated array") }
                    if bytes[index] == 0x2C { index += 1; continue }
                    if bytes[index] == 0x5D { index += 1; return .array(items) }
                    throw error("expected , or ]")
                }
            case 0x22: return .string(try string())
            case 0x74: try expect("true"); return .bool(true)
            case 0x66: try expect("false"); return .bool(false)
            case 0x6E: try expect("null"); return .null
            default: return try number()
            }
        }

        mutating func number() throws -> OrderedJSON {
            let start = index
            if index < bytes.count, bytes[index] == 0x2D { index += 1 }
            func digits() -> Int {
                let s = index
                while index < bytes.count, (0x30 ... 0x39).contains(bytes[index]) { index += 1 }
                return index - s
            }
            guard digits() > 0 else { throw error("invalid value") }
            if index < bytes.count, bytes[index] == 0x2E {
                index += 1
                guard digits() > 0 else { throw error("invalid number") }
            }
            if index < bytes.count, bytes[index] == 0x65 || bytes[index] == 0x45 {
                index += 1
                if index < bytes.count, bytes[index] == 0x2B || bytes[index] == 0x2D { index += 1 }
                guard digits() > 0 else { throw error("invalid number") }
            }
            let text = String(decoding: bytes[start ..< index], as: UTF8.self)
            guard let v = Double(text) else { throw error("invalid number") }
            return .number(text, v)
        }

        mutating func string() throws -> String {
            index += 1 // opening quote
            var out: [UInt8] = []
            while index < bytes.count {
                let b = bytes[index]
                index += 1
                switch b {
                case 0x22: return String(decoding: out, as: UTF8.self)
                case 0x5C:
                    guard index < bytes.count else { throw error("unterminated escape") }
                    let e = bytes[index]
                    index += 1
                    switch e {
                    case 0x22: out.append(0x22)
                    case 0x5C: out.append(0x5C)
                    case 0x2F: out.append(0x2F)
                    case 0x62: out.append(0x08)
                    case 0x66: out.append(0x0C)
                    case 0x6E: out.append(0x0A)
                    case 0x72: out.append(0x0D)
                    case 0x74: out.append(0x09)
                    case 0x75:
                        var cp = try hex4()
                        if (0xD800 ... 0xDBFF).contains(cp), index + 1 < bytes.count,
                           bytes[index] == 0x5C, bytes[index + 1] == 0x75 {
                            index += 2
                            let low = try hex4()
                            if (0xDC00 ... 0xDFFF).contains(low) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
                            }
                        }
                        out += Array(String(Character(Unicode.Scalar(cp) ?? "\u{FFFD}")).utf8)
                    default: throw error("invalid escape")
                    }
                default:
                    guard b >= 0x20 else { throw error("control character in string") }
                    out.append(b)
                }
            }
            throw error("unterminated string")
        }

        mutating func hex4() throws -> UInt32 {
            guard index + 4 <= bytes.count,
                  let v = UInt32(String(decoding: bytes[index ..< index + 4], as: UTF8.self), radix: 16)
            else { throw error("invalid \\u escape") }
            index += 4
            return v
        }
    }
}

enum JSONEncoding {
    /// JSON string-content bytes for `s`: `"` and `\` escaped, control characters as their short
    /// escape or `\u00XX`, everything else as raw UTF-8.
    static func escape(_ s: String) -> [UInt8] {
        var out: [UInt8] = []
        for b in s.utf8 {
            switch b {
            case 0x22: out += [0x5C, 0x22]
            case 0x5C: out += [0x5C, 0x5C]
            case 0x0A: out += [0x5C, 0x6E]
            case 0x0D: out += [0x5C, 0x72]
            case 0x09: out += [0x5C, 0x74]
            case 0x08: out += [0x5C, 0x62]
            case 0x0C: out += [0x5C, 0x66]
            case 0 ..< 0x20: out += Array(String(format: "\\u%04x", b).utf8)
            default: out.append(b)
            }
        }
        return out
    }
}
