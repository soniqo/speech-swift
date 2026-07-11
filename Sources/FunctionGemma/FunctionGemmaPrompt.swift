import Foundation

/// Mirrors `function_calls.format_tool_call_prompt` from the Python export
/// repo so Swift-side prompts hit the same grammar FunctionGemma was trained
/// on. Output is the fully-formed prompt — feed it straight to the tokenizer
/// without any additional chat-template wrapping.
enum FunctionGemmaPrompt {

    static let startOfTurn               = "<start_of_turn>"
    static let endOfTurn                 = "<end_of_turn>"
    static let developerPrompt           = "You are a model that can do function calling with the following functions"
    static let functionDeclarationStart  = "<start_function_declaration>"
    static let functionDeclarationEnd    = "<end_function_declaration>"
    static let functionCallStart         = "<start_function_call>"
    static let functionCallEnd           = "<end_function_call>"
    static let functionResponseStart     = "<start_function_response>"
    static let functionResponseEnd       = "<end_function_response>"
    static let escape                    = "<escape>"

    /// Serialise a single tool declaration in the FunctionGemma grammar, e.g.:
    /// ```
    /// <start_function_declaration>declaration:get_weather{description:<escape>Get current weather<escape>,parameters:{type:<escape>object<escape>,properties:{location:{type:<escape>string<escape>}}}}<end_function_declaration>
    /// ```
    static func formatFunctionDeclaration(_ tool: FunctionDeclaration) -> String {
        // Mirror Python: body = {"description": ..., "parameters": ...}
        let body = "{description:\(escape)\(tool.description)\(escape),parameters:\(formatStructuredValue(tool.parameters))}"
        return "\(functionDeclarationStart)declaration:\(tool.name)\(body)\(functionDeclarationEnd)"
    }

    /// Format the developer turn that introduces the tool list:
    /// `<start_of_turn>developer\n{prompt}{declarations}<end_of_turn>`.
    static func formatDeveloperTurn(_ tools: [FunctionDeclaration]) -> String {
        let declarations = tools.map(formatFunctionDeclaration).joined()
        return "\(startOfTurn)developer\n\(developerPrompt)\(declarations)\(endOfTurn)"
    }

    /// Format a user turn: `<start_of_turn>user\nTEXT<end_of_turn>`.
    static func formatUserTurn(_ text: String) -> String {
        return "\(startOfTurn)user\n\(text)\(endOfTurn)"
    }

    /// Format the trailing model-turn prefix used before generation:
    /// `<start_of_turn>model\n`.
    static func formatModelTurnPrefix() -> String {
        return "\(startOfTurn)model\n"
    }

    /// Build the full first-pass tool-call prompt:
    /// `developer_turn\nuser_turn\nmodel_prefix`. The result is byte-identical
    /// to `function_calls.format_tool_call_prompt` in the Python export, so
    /// the model sees the same grammar it was trained on.
    static func formatToolCallPrompt(tools: [FunctionDeclaration], userText: String) -> String {
        return "\(formatDeveloperTurn(tools))\n\(formatUserTurn(userText))\n\(formatModelTurnPrefix())"
    }

    /// Encode the result of executing a tool back into the FunctionGemma
    /// response format for the second forward pass.
    static func formatResponse(name: String, response: [String: Any]) -> String {
        return "\(functionResponseStart)response:\(name)\(formatStructuredValue(response))\(functionResponseEnd)"
    }

    // MARK: - JSON-like serialiser

    /// Recursive serialiser that mirrors Python's ``_format_structured_value``.
    /// Strings are wrapped in `<escape>...<escape>`, bools/null are lowercase,
    /// dicts become `{k:v,...}`, lists become `[v,...]`. Dict keys are emitted
    /// in a stable order — Python relies on dict insertion order, which Swift
    /// `[String: Any]` does not preserve, so we apply a preferred ordering
    /// that matches the canonical layout (`type, description, properties,
    /// required, items, enum`) and fall back to alphabetical for unknown keys.
    private static func formatStructuredValue(_ value: Any) -> String {
        if let dict = value as? [String: Any] {
            return formatObject(dict)
        }
        if let arr = value as? [Any] {
            let items = arr.map(formatStructuredValue).joined(separator: ",")
            return "[\(items)]"
        }
        if let s = value as? String {
            return "\(escape)\(s)\(escape)"
        }
        if let b = value as? Bool {
            return b ? "true" : "false"
        }
        if let n = value as? NSNumber {
            return n.stringValue
        }
        return "null"
    }

    private static func formatObject(_ dict: [String: Any]) -> String {
        let preferred = ["type", "description", "properties", "required", "items", "enum"]
        var keys = Array(dict.keys)
        keys.sort { a, b in
            let ai = preferred.firstIndex(of: a) ?? Int.max
            let bi = preferred.firstIndex(of: b) ?? Int.max
            if ai != bi { return ai < bi }
            return a < b
        }
        let pairs = keys.map { key -> String in
            "\(key):\(formatStructuredValue(dict[key] as Any))"
        }
        return "{\(pairs.joined(separator: ","))}"
    }
}
