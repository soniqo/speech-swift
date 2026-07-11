import XCTest
@testable import FunctionGemma

final class FunctionGemmaParserTests: XCTestCase {

    func testParseSingleCallWithEscapedString() {
        let text = "\(FunctionGemmaPrompt.functionCallStart)call:get_weather{location:<escape>Tokyo<escape>}\(FunctionGemmaPrompt.functionCallEnd)"
        let calls = FunctionGemmaParser.parseFunctionCalls(text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "get_weather")
        XCTAssertEqual(calls[0].arguments["location"], .string("Tokyo"))
    }

    func testParseScalarArguments() {
        let text = "\(FunctionGemmaPrompt.functionCallStart)call:set_timer{seconds:300,label:<escape>tea<escape>,enabled:true}\(FunctionGemmaPrompt.functionCallEnd)"
        let calls = FunctionGemmaParser.parseFunctionCalls(text)
        XCTAssertEqual(calls[0].name, "set_timer")
        XCTAssertEqual(calls[0].arguments["seconds"], .int(300))
        XCTAssertEqual(calls[0].arguments["label"], .string("tea"))
        XCTAssertEqual(calls[0].arguments["enabled"], .bool(true))
    }

    func testParseExchangeRateCanonicalPrompt() {
        // The byte-identical output from our published palettize-8 export.
        let text = "<start_function_call>call:get_exchange_rate{amount:23,from_currency:<escape>USD<escape>,to_currency:<escape>EUR<escape>}<end_function_call>"
        let calls = FunctionGemmaParser.parseFunctionCalls(text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "get_exchange_rate")
        XCTAssertEqual(calls[0].arguments["amount"], .int(23))
        XCTAssertEqual(calls[0].arguments["from_currency"], .string("USD"))
        XCTAssertEqual(calls[0].arguments["to_currency"], .string("EUR"))
    }

    func testParseDoubleAndNull() {
        let text = "<start_function_call>call:fn{x:0.5,y:null}<end_function_call>"
        let calls = FunctionGemmaParser.parseFunctionCalls(text)
        XCTAssertEqual(calls[0].arguments["x"], .double(0.5))
        XCTAssertEqual(calls[0].arguments["y"], .null)
    }

    func testParseParallelCalls() {
        let text = """
            <start_function_call>call:turn_on{device:<escape>flashlight<escape>}<end_function_call>\
            <start_function_call>call:create_note{title:<escape>Groceries<escape>}<end_function_call>
            """
        let calls = FunctionGemmaParser.parseFunctionCalls(text)
        XCTAssertEqual(calls.map { $0.name }, ["turn_on", "create_note"])
        XCTAssertEqual(calls[1].arguments["title"], .string("Groceries"))
    }

    func testNoCallReturnsEmpty() {
        XCTAssertTrue(FunctionGemmaParser.parseFunctionCalls("hello world").isEmpty)
    }
}
