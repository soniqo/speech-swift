import Foundation

/// Parses FunctionGemma's `<start_function_call>...<end_function_call>` blocks
/// into ``FunctionCall`` values. Mirrors `function_calls.parse_function_calls`
/// from the Python export repo.
///
/// Grammar reminder:
/// ```
/// <start_function_call>call:NAME{KEY:VALUE,...}<end_function_call>
/// ```
/// with strings escaped as `<escape>...<escape>` and ints / doubles / bools
/// emitted literally.
public enum FunctionGemmaParser {

    public static func parseFunctionCalls(_ text: String) -> [FunctionCall] {
        let start = FunctionGemmaPrompt.functionCallStart
        let end = FunctionGemmaPrompt.functionCallEnd
        var calls: [FunctionCall] = []
        var cursor = text.startIndex
        while cursor < text.endIndex,
              let startRange = text.range(of: start, range: cursor..<text.endIndex),
              let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex) {
            let body = text[startRange.upperBound..<endRange.lowerBound]
            if let call = parseCallBody(String(body)) {
                calls.append(call)
            }
            cursor = endRange.upperBound
        }
        return calls
    }

    private static func parseCallBody(_ body: String) -> FunctionCall? {
        // Expected shape: "call:NAME{...}"
        guard body.hasPrefix("call:") else { return nil }
        let afterCall = body.dropFirst("call:".count)
        guard let bracePos = afterCall.firstIndex(of: "{") else { return nil }
        let name = String(afterCall[afterCall.startIndex..<bracePos])
        let rest = String(afterCall[bracePos...])  // includes leading "{"
        var parser = Parser(input: rest)
        guard case .object(let args) = parser.parseValue() else { return nil }
        return FunctionCall(name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                            arguments: args)
    }

    // MARK: - Tiny recursive-descent parser for the FunctionGemma grammar

    private struct Parser {
        let chars: [Character]
        var pos: Int = 0

        init(input: String) {
            self.chars = Array(input)
        }

        mutating func parseValue() -> ArgumentValue {
            skipSpace()
            guard pos < chars.count else { return .null }
            let c = chars[pos]
            switch c {
            case "{":
                return .object(parseObject())
            case "[":
                return .array(parseArray())
            case "<":
                if matchLiteral(FunctionGemmaPrompt.escape) {
                    return .string(parseEscapedString())
                }
                return .null
            case "t", "f":
                if matchLiteral("true")  { return .bool(true) }
                if matchLiteral("false") { return .bool(false) }
                return .null
            case "n":
                if matchLiteral("null") { return .null }
                return .null
            default:
                return parseNumber()
            }
        }

        mutating func parseObject() -> [String: ArgumentValue] {
            var result: [String: ArgumentValue] = [:]
            guard pos < chars.count, chars[pos] == "{" else { return result }
            pos += 1
            skipSpace()
            if pos < chars.count, chars[pos] == "}" { pos += 1; return result }
            while pos < chars.count {
                skipSpace()
                let key = parseKey()
                skipSpace()
                if pos < chars.count, chars[pos] == ":" { pos += 1 }
                let value = parseValue()
                result[key] = value
                skipSpace()
                if pos < chars.count, chars[pos] == "," {
                    pos += 1
                    continue
                }
                if pos < chars.count, chars[pos] == "}" { pos += 1; break }
                break
            }
            return result
        }

        mutating func parseArray() -> [ArgumentValue] {
            var result: [ArgumentValue] = []
            guard pos < chars.count, chars[pos] == "[" else { return result }
            pos += 1
            skipSpace()
            if pos < chars.count, chars[pos] == "]" { pos += 1; return result }
            while pos < chars.count {
                result.append(parseValue())
                skipSpace()
                if pos < chars.count, chars[pos] == "," { pos += 1; continue }
                if pos < chars.count, chars[pos] == "]" { pos += 1; break }
                break
            }
            return result
        }

        mutating func parseKey() -> String {
            var out = ""
            while pos < chars.count {
                let c = chars[pos]
                if c == ":" || c == "," || c == "}" { break }
                out.append(c)
                pos += 1
            }
            return out.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        mutating func parseEscapedString() -> String {
            // We've already consumed the opening <escape>; consume until the
            // closing <escape>.
            let escape = FunctionGemmaPrompt.escape
            var out = ""
            while pos < chars.count {
                if matchLiteral(escape) { return out }
                out.append(chars[pos])
                pos += 1
            }
            return out
        }

        mutating func parseNumber() -> ArgumentValue {
            var raw = ""
            var sawDot = false
            while pos < chars.count {
                let c = chars[pos]
                if c.isWholeNumber || c == "-" || c == "+" { raw.append(c); pos += 1 }
                else if c == "." { sawDot = true; raw.append(c); pos += 1 }
                else if c == "e" || c == "E" { sawDot = true; raw.append(c); pos += 1 }
                else { break }
            }
            if raw.isEmpty { return .null }
            if sawDot, let d = Double(raw) { return .double(d) }
            if let i = Int64(raw) { return .int(i) }
            if let d = Double(raw) { return .double(d) }
            return .string(raw)
        }

        mutating func skipSpace() {
            while pos < chars.count, chars[pos].isWhitespace { pos += 1 }
        }

        mutating func matchLiteral(_ literal: String) -> Bool {
            let arr = Array(literal)
            guard pos + arr.count <= chars.count else { return false }
            for i in 0..<arr.count where chars[pos + i] != arr[i] { return false }
            pos += arr.count
            return true
        }
    }
}
