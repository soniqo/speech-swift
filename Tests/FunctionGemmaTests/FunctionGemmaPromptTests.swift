import XCTest
@testable import FunctionGemma

final class FunctionGemmaPromptTests: XCTestCase {

    /// The rendered first-pass prompt must contain the full developer block
    /// with `<start_function_declaration>` entries — that's the grammar
    /// FunctionGemma was trained on, and the previous simplified template
    /// omitted it, causing garbage `<start_function_call>` repetition at
    /// inference time.
    func testFormatToolCallPromptIncludesDeveloperAndDeclarationBlocks() {
        let tool = FunctionDeclaration(
            name: "get_exchange_rate",
            description: "Get the exchange rate between two currencies.",
            parameters: [
                "type": "object",
                "properties": [
                    "amount": ["type": "number"],
                    "from_currency": ["type": "string"],
                    "to_currency": ["type": "string"],
                ],
                "required": ["amount", "from_currency", "to_currency"],
            ]
        )
        let userText = "Convert 23 USD to EUR"

        let prompt = FunctionGemmaPrompt.formatToolCallPrompt(
            tools: [tool],
            userText: userText
        )

        // Developer turn (the bug fix: this block was missing before).
        XCTAssertTrue(prompt.contains("<start_of_turn>developer"),
                      "missing developer turn in prompt:\n\(prompt)")
        XCTAssertTrue(prompt.contains("<start_function_declaration>"),
                      "missing <start_function_declaration> in prompt:\n\(prompt)")
        XCTAssertTrue(prompt.contains("get_exchange_rate"),
                      "missing tool name in prompt:\n\(prompt)")
        XCTAssertTrue(prompt.contains("<end_function_declaration>"),
                      "missing <end_function_declaration> in prompt:\n\(prompt)")
        XCTAssertTrue(prompt.contains("<end_of_turn>"),
                      "missing <end_of_turn> in prompt:\n\(prompt)")

        // User turn.
        XCTAssertTrue(prompt.contains("<start_of_turn>user"),
                      "missing user turn in prompt:\n\(prompt)")
        XCTAssertTrue(prompt.contains(userText),
                      "missing user text in prompt:\n\(prompt)")

        // Model-turn prefix (asserted as the contiguous end-of-user →
        // start-of-model transition that wraps the generation slot).
        XCTAssertTrue(prompt.contains("<end_of_turn>\n<start_of_turn>model"),
                      "missing user→model transition in prompt:\n\(prompt)")
    }

    /// The declaration body must use `declaration:NAME{description:...,parameters:...}`
    /// with `<escape>`-wrapped strings — mirrors the Python reference exactly.
    func testFormatFunctionDeclarationMatchesPythonGrammar() {
        let tool = FunctionDeclaration(
            name: "set_timer",
            description: "Start a countdown timer.",
            parameters: [
                "type": "object",
                "properties": ["seconds": ["type": "integer"]],
            ]
        )

        let rendered = FunctionGemmaPrompt.formatFunctionDeclaration(tool)

        XCTAssertTrue(rendered.hasPrefix("<start_function_declaration>declaration:set_timer{"),
                      "unexpected declaration prefix: \(rendered)")
        XCTAssertTrue(rendered.hasSuffix("<end_function_declaration>"),
                      "unexpected declaration suffix: \(rendered)")
        XCTAssertTrue(rendered.contains("description:<escape>Start a countdown timer.<escape>"),
                      "description not wrapped in <escape>: \(rendered)")
        XCTAssertTrue(rendered.contains("parameters:{type:<escape>object<escape>"),
                      "parameters block missing or mis-ordered: \(rendered)")
    }

    /// Multiple tools must concatenate without a separator and both end up in
    /// the developer turn body.
    func testFormatDeveloperTurnConcatenatesTools() {
        let a = FunctionDeclaration(name: "turn_on", description: "On",
                                    parameters: ["type": "object"])
        let b = FunctionDeclaration(name: "turn_off", description: "Off",
                                    parameters: ["type": "object"])

        let turn = FunctionGemmaPrompt.formatDeveloperTurn([a, b])

        XCTAssertTrue(turn.hasPrefix("<start_of_turn>developer\n"))
        XCTAssertTrue(turn.hasSuffix("<end_of_turn>"))
        XCTAssertTrue(turn.contains("declaration:turn_on"))
        XCTAssertTrue(turn.contains("declaration:turn_off"))
        // Back-to-back: end of one declaration must touch the start of the next.
        XCTAssertTrue(turn.contains("<end_function_declaration><start_function_declaration>"),
                      "declarations should concatenate without separator: \(turn)")
    }
}
