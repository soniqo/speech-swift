import Foundation

/// Mirrors `function_calls.format_tool_call_prompt` from the Python export
/// repo so Swift-side prompts hit the same grammar FunctionGemma was trained
/// on. Output is plain text — wrap it with the model's chat template before
/// tokenizing.
enum FunctionGemmaPrompt {

    static let functionDeclarationsStart = "<start_function_declarations>"
    static let functionDeclarationsEnd   = "<end_function_declarations>"
    static let functionDeclarationStart  = "<start_function_declaration>"
    static let functionDeclarationEnd    = "<end_function_declaration>"
    static let functionCallStart         = "<start_function_call>"
    static let functionCallEnd           = "<end_function_call>"
    static let functionResponseStart     = "<start_function_response>"
    static let functionResponseEnd       = "<end_function_response>"
    static let escape                    = "<escape>"

    /// Serialise the tool list in the FunctionGemma grammar, e.g.:
    /// ```
    /// <start_function_declarations>
    /// <start_function_declaration>
    /// name:get_weather,description:<escape>Get current weather<escape>,parameters:{type:<escape>object<escape>,properties:{location:{type:<escape>string<escape>}}}
    /// <end_function_declaration>
    /// <end_function_declarations>
    /// ```
    static func formatDeclarations(_ tools: [FunctionDeclaration]) -> String {
        var out = "\(functionDeclarationsStart)\n"
        for tool in tools {
            out += "\(functionDeclarationStart)\n"
            out += "name:\(tool.name),description:\(escape)\(tool.description)\(escape),parameters:"
            out += formatValue(tool.parameters)
            out += "\n\(functionDeclarationEnd)\n"
        }
        out += functionDeclarationsEnd
        return out
    }

    /// Format a tool-call prompt as a plain user message — the chat template
    /// wraps it with `<start_of_turn>user … <end_of_turn><start_of_turn>model`.
    static func formatUserTurn(tools: [FunctionDeclaration], userText: String) -> String {
        let declarations = formatDeclarations(tools)
        return "\(declarations)\n\(userText)"
    }

    // MARK: - JSON-like serialiser

    private static func formatValue(_ value: Any) -> String {
        if let dict = value as? [String: Any] {
            return formatObject(dict)
        }
        if let arr = value as? [Any] {
            let items = arr.map(formatValue).joined(separator: ",")
            return "[\(items)]"
        }
        if let s = value as? String {
            return "\(escape)\(s)\(escape)"
        }
        if let b = value as? Bool {
            return b ? "true" : "false"
        }
        if let n = value as? NSNumber {
            // `NSNumber` covers Int, Double, etc. The serialiser distinguishes
            // them based on objCType: 'i'/'q' = integer, 'd'/'f' = float.
            let type = String(cString: n.objCType)
            if type == "i" || type == "q" || type == "l" || type == "s" || type == "c" {
                return n.stringValue
            }
            return n.stringValue
        }
        return "null"
    }

    private static func formatObject(_ dict: [String: Any]) -> String {
        // Preserve "type" first then "properties" if both exist — that's the
        // order the Python tokenisation script emits, which matters because
        // FunctionGemma was trained on that exact field ordering.
        let preferred = ["type", "description", "properties", "required", "items", "enum"]
        var keys = Array(dict.keys)
        keys.sort { a, b in
            let ai = preferred.firstIndex(of: a) ?? Int.max
            let bi = preferred.firstIndex(of: b) ?? Int.max
            if ai != bi { return ai < bi }
            return a < b
        }
        let pairs = keys.map { key -> String in
            "\(key):\(formatValue(dict[key] as Any))"
        }
        return "{\(pairs.joined(separator: ","))}"
    }

    /// Encode the result of executing a tool back into the FunctionGemma
    /// response format for the second forward pass.
    static func formatResponse(name: String, response: [String: Any]) -> String {
        let body = formatObject(response)
        return "\(functionResponseStart)response:\(name)\(body)\(functionResponseEnd)"
    }
}
