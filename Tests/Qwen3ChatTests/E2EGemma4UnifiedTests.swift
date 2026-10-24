import AudioCommon
import Foundation
import MLX
import XCTest
@testable import Qwen3Chat

final class E2EGemma4UnifiedTests: XCTestCase {
    func testLongPromptCacheOnlyPrefillMatchesReferenceGreedyTokens() throws {
        guard let folder = ProcessInfo.processInfo.environment["GEMMA4_12B_MODEL_DIR"] else {
            throw XCTSkip("Set GEMMA4_12B_MODEL_DIR to the staged Gemma 4 12B INT4 checkpoint")
        }
        final class Capture: @unchecked Sendable {
            private let lock = NSLock()
            private var value: Gemma4GenerationTrace?
            func store(_ trace: Gemma4GenerationTrace) { lock.withLock { value = trace } }
            var trace: Gemma4GenerationTrace? { lock.withLock { value } }
        }
        Memory.cacheLimit = 128 * 1024 * 1024
        Memory.memoryLimit = 20 * 1024 * 1024 * 1024
        let chat = try Gemma4UnifiedChat.fromDirectory(URL(fileURLWithPath: folder))
        guard chat.prefixCacheBudgetBytes > 0 else {
            throw XCTSkip("Enable the Gemma system prefix cache for this parity check")
        }
        let tools = [Gemma4Tool(name: "record_note", description: "Record the exact text of a note.",
                               parameters: .object(["type": .string("object"),
                                                    "properties": .object(["text": .object(["type": .string("string")])]),
                                                    "required": .array([.string("text")])]))]
        let reference = String(repeating: "Reference notes describe routine scheduling, inventory, and account policies. ", count: 700)
        let messages: [Gemma4ConversationMessage] = [
            .init(role: .system, content: reference + "\nUse record_note to save the user's exact note. Do not include the reference notes in the saved text."),
            .init(role: .user, content: "Save a note with the exact text: launch review")]
        let sampling = ChatSamplingConfig(temperature: 0, maxTokens: 128, repetitionPenalty: 1)
        var traces: [Gemma4GenerationTrace] = []
        var responses: [Gemma4Response] = []
        // Compare a cold optimized request and a warm prefix hit against the
        // former implementation on both sides of the pair.
        for strategy in [Gemma4PrefillStrategy.fullLogits, .cacheOnly, .cacheOnly, .fullLogits] {
            Memory.clearCache()
            let capture = Capture()
            let started = Date()
            let response = try chat.generateResponse(messages: messages, tools: tools, sampling: sampling,
                                                       cancellation: nil, onGeneration: { capture.store($0) },
                                                       prefillStrategy: strategy)
            let trace = try XCTUnwrap(capture.trace)
            XCTAssertGreaterThan(trace.promptTokenIDs.count, 4096)
            print("Gemma4 prefill \(strategy): \(trace.promptTokenIDs.count) prompt tokens, \(chat.lastCachedPrefixTokens) cached tokens, \(chat.prefixCacheCandidateBytes) prefix bytes / \(chat.prefixCacheBudgetBytes) budget, \(trace.generatedTokenIDs.count) generated tokens, \(Date().timeIntervalSince(started)) seconds")
            XCTAssertEqual(response.toolCalls.count, 1)
            let call = try XCTUnwrap(response.toolCalls.first)
            XCTAssertEqual(call.name, "record_note")
            XCTAssertEqual(call.arguments, .object(["text": .string("launch review")]))
            traces.append(trace)
            responses.append(response)
        }
        for index in traces.indices.dropFirst() {
            XCTAssertEqual(traces[index].promptTokenIDs, traces[0].promptTokenIDs)
            XCTAssertEqual(traces[index].generatedTokenIDs, traces[0].generatedTokenIDs)
            XCTAssertEqual(traces[index].rawOutput, traces[0].rawOutput)
            XCTAssertEqual(responses[index].content, responses[0].content)
        }
        XCTAssertGreaterThan(chat.cachedPrefixTokenCount, 4096)
        XCTAssertEqual(chat.prefixCacheHits, 1)

        let otherMessages = [messages[0], Gemma4ConversationMessage(role: .user,
            content: "Background context, excluded from the saved note: " +
                String(repeating: "Routine scheduling information is provided for context only. ", count: 150) +
                "\nSave a note with the exact text: second review")]
        let reusedCapture = Capture()
        let reused = try chat.generateResponse(messages: otherMessages, tools: tools, sampling: sampling,
                                               cancellation: nil, onGeneration: { reusedCapture.store($0) })
        XCTAssertGreaterThan(chat.lastCachedPrefixTokens, 4096)
        XCTAssertEqual(reused.toolCalls.first?.arguments, ChatJSONValue.object(["text": .string("second review")]))
        let uncachedCapture = Capture()
        let uncached = try chat.generateResponse(messages: otherMessages, tools: tools, sampling: sampling,
                                                cancellation: nil, onGeneration: { uncachedCapture.store($0) }, prefillStrategy: .fullLogits)
        XCTAssertEqual(reused.content, uncached.content)
        XCTAssertEqual(reused.toolCalls.map(\.arguments), uncached.toolCalls.map(\.arguments))
        XCTAssertEqual(try XCTUnwrap(reusedCapture.trace).generatedTokenIDs, try XCTUnwrap(uncachedCapture.trace).generatedTokenIDs)
        let repeatedCapture = Capture()
        let repeated = try chat.generateResponse(messages: messages, tools: tools, sampling: sampling,
                                                 cancellation: nil, onGeneration: { repeatedCapture.store($0) })
        XCTAssertGreaterThan(chat.lastCachedPrefixTokens, 4096)
        XCTAssertEqual(repeated.toolCalls.map(\.arguments), responses[0].toolCalls.map(\.arguments))
        XCTAssertEqual(try XCTUnwrap(repeatedCapture.trace).generatedTokenIDs, traces[0].generatedTokenIDs)
        chat.clearPrefixCache()
        XCTAssertEqual(chat.cachedPrefixTokenCount, 0)
    }

    func testCompletionTraceSurvivesMalformedToolOutput() throws {
        guard let folder = ProcessInfo.processInfo.environment["GEMMA4_12B_MODEL_DIR"] else {
            throw XCTSkip("Set GEMMA4_12B_MODEL_DIR to the staged Gemma 4 12B INT4 checkpoint")
        }
        final class Capture: @unchecked Sendable {
            private let lock = NSLock()
            private var value: Gemma4GenerationTrace?
            func store(_ trace: Gemma4GenerationTrace) { lock.withLock { value = trace } }
            var trace: Gemma4GenerationTrace? { lock.withLock { value } }
        }
        let capture = Capture()
        let chat = try Gemma4UnifiedChat.fromDirectory(URL(fileURLWithPath: folder))
        let tools = [Gemma4Tool(name: "record_note", description: "Record the exact text of a note.",
                               parameters: .object(["type": .string("object"),
                                                    "properties": .object(["text": .object(["type": .string("string")])]),
                                                    "required": .array([.string("text")])]))]
        XCTAssertThrowsError(try chat.generateResponse(messages: [
            .init(role: .system, content: "Use record_note to save notes."),
            .init(role: .user, content: "Save a note with the exact text: launch review")],
            tools: tools, sampling: .init(temperature: 0, maxTokens: 1, repetitionPenalty: 1),
            onGeneration: { capture.store($0) }))
        let trace = try XCTUnwrap(capture.trace)
        XCTAssertEqual(trace.finishReason, "length")
        XCTAssertNil(trace.stopTokenID)
        XCTAssertEqual(trace.generatedTokenIDs.count, 1)
        XCTAssertFalse(trace.rawOutput.isEmpty)
        XCTAssertTrue(trace.prompt.contains("record_note"))
        XCTAssertGreaterThan(trace.promptTokenIDs.count, 1)
    }

    func testNativeToolCallResultHistoryAndCancellation() throws {
        guard let folder = ProcessInfo.processInfo.environment["GEMMA4_12B_MODEL_DIR"] else {
            throw XCTSkip("Set GEMMA4_12B_MODEL_DIR to the staged Gemma 4 12B INT4 checkpoint")
        }
        let chat = try Gemma4UnifiedChat.fromDirectory(URL(fileURLWithPath: folder))
        let tools = [Gemma4Tool(name: "record_note", description: "Record the exact text of a note.",
                               parameters: .object(["type": .string("object"),
                                                    "properties": .object(["text": .object(["type": .string("string")])]),
                                                    "required": .array([.string("text")])]))]
        let messages: [Gemma4ConversationMessage] = [
            .init(role: .system, content: "Use record_note to save notes. After a successful result, respond: Note recorded."),
            .init(role: .user, content: "Save a note with the exact text: launch review")]
        let sampling = ChatSamplingConfig(temperature: 0, maxTokens: 512, repetitionPenalty: 1)
        let generate: ([Gemma4ConversationMessage], [Gemma4Tool], ChatSamplingConfig, InferenceCancellationToken?) throws -> Gemma4Response = chat.generateResponse
        let reply = try generate(messages, tools, sampling, nil)
        XCTAssertEqual(reply.toolCalls.count, 1)
        let call = try XCTUnwrap(reply.toolCalls.first)
        XCTAssertEqual(call.name, "record_note")
        XCTAssertEqual(call.arguments, .object(["text": .string("launch review")]))
        let result = try chat.generateResponse(messages: messages + [
            .init(role: .assistant, toolCalls: reply.toolCalls),
            .init(role: .tool, content: "{\"status\":\"recorded\"}", toolCallID: call.id)],
            tools: tools, sampling: sampling)
        XCTAssertTrue(result.toolCalls.isEmpty)
        XCTAssertTrue(result.content.lowercased().contains("record"), result.content)
        for response in [reply, result] {
            XCTAssertFalse(response.content.contains("<channel|>"), response.content)
            XCTAssertFalse(response.content.contains("<|channel>"), response.content)
        }
        let token = InferenceCancellationToken()
        token.cancel()
        XCTAssertThrowsError(try chat.generateResponse(messages: messages, cancellation: token))
    }
}
