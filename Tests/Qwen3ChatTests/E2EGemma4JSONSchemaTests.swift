import XCTest
import Foundation
import AudioCommon
@testable import Qwen3Chat

/// JSON-schema constrained decoding against the real Gemma 4 vocabulary and, where the weights
/// are present, the real model.
///
/// Set `GEMMA4_E4B_MODEL_DIR` to a local `gemma-4-E4B-it-MLX-4bit` directory; otherwise the
/// HuggingFace cache location is used and the tests skip when it is empty.
final class E2EGemma4JSONSchemaTests: XCTestCase {
    private static let modelDir: URL = {
        if let p = ProcessInfo.processInfo.environment["GEMMA4_E4B_MODEL_DIR"] {
            return URL(fileURLWithPath: p)
        }
        return (try? HuggingFaceDownloader.getCacheDirectory(for: "aufklarer/gemma-4-E4B-it-MLX-4bit"))
            ?? URL(fileURLWithPath: "/nonexistent")
    }()

    private func requireFile(_ name: String) throws {
        guard FileManager.default.fileExists(atPath: Self.modelDir.appendingPathComponent(name).path) else {
            throw XCTSkip("Gemma 4 E4B \(name) unavailable in \(Self.modelDir.path)")
        }
    }

    /// Fact extraction over a short invented exchange, the shape the schema was written for.
    static let system = "Read one finalized transcript packet. Return up to eight distinct source-linked facts worth retaining for later retrieval: concrete amounts, counts, dates, durations, decisions, commitments, proposals, acceptance, refusal and corrections. Each fact has candidate plus exactly six fields: subject, property, value, unit, status, timeframe. Candidate copies the smallest literal span from one turn that carries the fact's value. Each field has text and turn_ids, selecting only T IDs from the packet. Use empty text and empty turn_ids for an unstated field. Reply with one JSON object and nothing else."
    static let user = #"{"turns":[{"id":"T1","speaker":"Anna","text":"So what budget do we have for the pilot?"},{"id":"T2","speaker":"Ben","text":"I'd propose forty thousand euros, spread over three months."},{"id":"T3","speaker":"Anna","text":"Forty is fine. Let's start on the first of March."},{"id":"T4","speaker":"Ben","text":"Great, and we keep two engineers on it full time."}]}"#

    private func messages() -> [ChatMessage] {
        [ChatMessage(role: .system, content: Self.system), ChatMessage(role: .user, content: Self.user)]
    }

    // MARK: - Vocabulary only (tokenizer.json, no weights)

    /// The tokenizer's own encoding of valid documents must be admissible at every step — the
    /// trie, the clean-token shortcut and the special-token exclusion all agree with it.
    func testTokenizerEncodingOfValidDocumentsIsAdmitted() throws {
        try requireFile("tokenizer.json")
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.modelDir)
        let config = try Gemma4DenseConfig.load(from: Self.modelDir.appendingPathComponent("config.json"))

        let built = Date()
        let vocabulary = JSONTokenVocabulary(gemma: tokenizer, size: config.vocabSize)
        let buildSeconds = Date().timeIntervalSince(built)
        let cleanCount = vocabulary.cleanCharsHost.filter { $0 != .max }.count
        print("[json-schema] vocabulary: \(config.vocabSize) ids, \(cleanCount) clean, built in \(String(format: "%.2f", buildSeconds)) s")

        let grammar = try JSONSchemaGrammar(schema: JSONSchemaMatcherTests.factSchema)
        let documents = [
            JSONSchemaMatcherTests.validFactDocument,
            "{\n  \"facts\": [\n    {\n      \"candidate\": \"сорок тысяч евро\", \"subject\": {\"text\": \"пилот\", \"turn_ids\": [\"T2\"]}, \"property\": {\"text\": \"бюджет\", \"turn_ids\": [\"T2\"]}, \"value\": {\"text\": \"сорок тысяч евро\", \"turn_ids\": [\"T2\"]}, \"unit\": {\"text\": \"\", \"turn_ids\": []}, \"status\": {\"text\": \"предложение \\\"в силе\\\"\", \"turn_ids\": [\"T2\"]}, \"timeframe\": {\"text\": \"三个月\", \"turn_ids\": [\"T2\"]}\n    }\n  ]\n}",
        ]
        var stepTimes: [Double] = []
        for document in documents {
            var c = JSONTokenConstraint(grammar: grammar, vocabulary: vocabulary,
                                        endTokens: tokenizer.eosTokenIds.sorted())
            for id in tokenizer.encode(document) {
                let t0 = Date()
                let allowance = c.allowance()
                stepTimes.append(Date().timeIntervalSince(t0) * 1000)
                let admitted = allowance.ids.contains(Int32(id))
                    || allowance.cleanUpTo.map { Int(vocabulary.cleanCharsHost[id]) <= $0 } ?? false
                XCTAssertTrue(admitted, "token \(id) \(String(decoding: tokenizer.tokenBytes(id), as: UTF8.self).debugDescription) refused")
                XCTAssertTrue(c.accept(id))
            }
            XCTAssertTrue(c.isDone, "document should be complete")
        }
        let sorted = stepTimes.sorted()
        print(String(format: "[json-schema] mask computation per step: mean %.3f ms, p50 %.3f ms, p95 %.3f ms, max %.3f ms over %d steps",
                     stepTimes.reduce(0, +) / Double(stepTimes.count), sorted[sorted.count / 2],
                     sorted[sorted.count * 95 / 100], sorted.last!, sorted.count))
    }

    /// The shortcut over clean tokens admits exactly what a token-by-token check admits, over the
    /// whole real vocabulary, inside a free string and inside a length-limited one.
    func testCleanTokenShortcutMatchesTokenByTokenCheck() throws {
        try requireFile("tokenizer.json")
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.modelDir)
        let vocabulary = JSONTokenVocabulary(gemma: tokenizer, size: 262_144)
        let schema = #"{"type":"object","additionalProperties":false,"required":["a","b"],"properties":{"a":{"type":"string"},"b":{"type":"string","minLength":2,"maxLength":5}}}"#
        let grammar = try JSONSchemaGrammar(schema: schema)
        for prefix in [#"{"a":"x"#, #"{"a":"xy","b":""#, #"{"a":"","b":"abc"#] {
            var c = JSONTokenConstraint(grammar: grammar, vocabulary: vocabulary, endTokens: [1])
            c.replaceStates(c.matcher.advance(c.matcher.initial, bytes: Array(prefix.utf8)))
            XCTAssertNotNil(c.allowance().cleanUpTo, "shortcut should apply at \(prefix)")
            var reference = Set<Int>()
            for (id, bytes) in vocabulary.tokenBytes.enumerated() where !bytes.isEmpty {
                if !c.matcher.advance(c.states, bytes: bytes).isEmpty { reference.insert(id) }
            }
            XCTAssertEqual(c.allowedIDs(), reference, "at \(prefix)")
        }
    }

    // MARK: - Real model

    private func loadModel() throws -> Gemma4Chat {
        try requireFile("model.safetensors")
        do {
            return try Gemma4Chat.fromDirectory(Self.modelDir)
        } catch {
            throw XCTSkip("model load failed (weights/metallib): \(error)")
        }
    }

    private struct Run {
        var text = ""
        var tokens = 0
        var firstTokenMs = 0.0
        var totalMs = 0.0
        var decodeTokensPerSecond: Double {
            tokens > 1 ? Double(tokens - 1) / ((totalMs - firstTokenMs) / 1000) : 0
        }
    }

    private func run(
        _ chat: Gemma4Chat, _ sampling: ChatSamplingConfig,
        engine: JSONConstraintEngine = .current
    ) throws -> Run {
        let constraint = try chat.makeConstraint(sampling.responseFormat, engine: engine)
        let prompt = Gemma4ChatTemplate.encode(messages: messages(), tokenizer: chat.gemmaTokenizer)
        var result = Run()
        let start = Date()
        let failure = chat.decode(
            promptTokens: prompt, sampling: sampling, constraint: constraint,
            onToken: { _ in
                if result.tokens == 0 { result.firstTokenMs = Date().timeIntervalSince(start) * 1000 }
                result.tokens += 1
            },
            onText: { result.text += $0 })
        result.totalMs = Date().timeIntervalSince(start) * 1000
        XCTAssertNil(failure)
        return result
    }

    func testConstrainedFactExtractionParsesAndValidates() throws {
        let chat = try loadModel()
        let format = ChatResponseFormat.jsonSchema(JSONSchemaMatcherTests.factSchema)
        let greedy = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 1500,
                                        repetitionPenalty: 1.1, responseFormat: format)
        let reply = try chat.generate(messages: messages(), sampling: greedy)
        print("[json-schema] constrained reply: \(reply)")
        try assertValidFactDocument(reply)

        // Sampled decoding respects the same mask.
        var sampled = greedy
        sampled.temperature = 0.7
        sampled.topK = 40
        sampled.topP = 0.9
        for _ in 0 ..< 2 {
            let text = try chat.generate(messages: messages(), sampling: sampled)
            print("[json-schema] sampled reply: \(text)")
            try assertValidFactDocument(text)
        }
    }

    private func assertValidFactDocument(_ text: String, file: StaticString = #filePath, line: UInt = #line) throws {
        let matcher = JSONSchemaMatcher(grammar: try JSONSchemaGrammar(schema: JSONSchemaMatcherTests.factSchema))
        // Whitespace between tokens is layout, not content: XGrammar admits any JSON whitespace
        // up to its cap while the Swift matcher admits a narrower layout. Judge the document.
        let states = matcher.advance(matcher.initial, bytes: Self.minified(text))
        XCTAssertTrue(matcher.isDone(states), "reply must be one complete schema-valid document:\n\(text)",
                      file: file, line: line)
        let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
        let facts = try XCTUnwrap(object?["facts"] as? [[String: Any]], file: file, line: line)
        for fact in facts {
            XCTAssertEqual(Set(fact.keys), ["candidate", "subject", "property", "value", "unit", "status", "timeframe"],
                           file: file, line: line)
            XCTAssertFalse((fact["candidate"] as? String ?? "").isEmpty, file: file, line: line)
            for role in ["subject", "property", "value", "unit", "status", "timeframe"] {
                let field = try XCTUnwrap(fact[role] as? [String: Any], role, file: file, line: line)
                XCTAssertNotNil(field["text"] as? String, file: file, line: line)
                let ids = try XCTUnwrap(field["turn_ids"] as? [String], file: file, line: line)
                XCTAssertLessThanOrEqual(ids.count, 4, file: file, line: line)
                for id in ids { XCTAssertNotNil(id.range(of: "^T[1-9][0-9]*$", options: .regularExpression), id) }
            }
        }
    }

    /// Greedy decode of the same prompt, unconstrained and constrained. Prints first-token
    /// latency (prefill + first mask) and decode throughput; asserts nothing about speed.
    func testConstrainedDecodingThroughput() throws {
        let chat = try loadModel()
        let factFormat = ChatResponseFormat.jsonSchema(JSONSchemaMatcherTests.factSchema)
        let plain = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 400, repetitionPenalty: 1.1)
        var constrained = plain
        constrained.responseFormat = factFormat

        _ = try run(chat, ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 8))   // warm up kernels
        let vocabularyStart = Date()
        _ = chat.constraintVocabulary()
        print(String(format: "[json-schema] vocabulary index built in %.0f ms", Date().timeIntervalSince(vocabularyStart) * 1000))

        for round in 1 ... 2 {
            for (label, sampling) in [("unconstrained", plain), ("fact schema", constrained)] {
                let r = try run(chat, sampling)
                print(String(format: "[json-schema] round %d %@: first token %.0f ms, %d tokens, %.1f tok/s decode, total %.0f ms",
                             round, label, r.firstTokenMs, r.tokens, r.decodeTokensPerSecond, r.totalMs))
                if round == 2 { print("[json-schema] \(label) output: \(r.text.prefix(600))") }
            }
        }
    }

    // MARK: - XGrammar

    /// XGrammar's mask admits the tokenizer's own encoding of valid documents at every step.
    func testXGrammarAdmitsTokenizerEncodingOfValidDocuments() throws {
        try requireFile("tokenizer.json")
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.modelDir)
        let config = try Gemma4DenseConfig.load(from: Self.modelDir.appendingPathComponent("config.json"))

        let built = Date()
        let vocabulary = try XGrammarVocabulary(gemma: tokenizer, size: config.vocabSize)
        print(String(format: "[xgrammar] vocabulary compiler built in %.2f s", Date().timeIntervalSince(built)))
        let compiled = Date()
        _ = try vocabulary.constraint(schema: JSONSchemaMatcherTests.factSchema)
        print(String(format: "[xgrammar] fact schema compiled in %.0f ms (cold)", Date().timeIntervalSince(compiled) * 1000))

        let documents = [
            JSONSchemaMatcherTests.validFactDocument,
            "{\"facts\":[{\"candidate\":\"сорок тысяч евро\",\"subject\":{\"text\":\"пилот\",\"turn_ids\":[\"T2\"]},\"property\":{\"text\":\"бюджет\",\"turn_ids\":[\"T2\"]},\"value\":{\"text\":\"三个月\",\"turn_ids\":[\"T2\"]},\"unit\":{\"text\":\"\",\"turn_ids\":[]},\"status\":{\"text\":\"предложение \\\"в силе\\\"\",\"turn_ids\":[\"T2\"]},\"timeframe\":{\"text\":\"\",\"turn_ids\":[]}}]}",
        ]
        var stepTimes: [Double] = []
        for document in documents {
            let c = try vocabulary.constraint(schema: JSONSchemaMatcherTests.factSchema)
            for id in tokenizer.encode(document) {
                let t0 = Date()
                let mask = c.nextMask()
                stepTimes.append(Date().timeIntervalSince(t0) * 1000)
                guard case .packed(let packed) = mask else { return XCTFail("expected a packed mask") }
                let set = packed.words[id / 32] & Int32(bitPattern: 1 << UInt32(id % 32)) != 0
                XCTAssertTrue(set, "token \(id) \(String(decoding: tokenizer.tokenBytes(id), as: UTF8.self).debugDescription) refused")
                XCTAssertTrue(c.accept(id))
            }
            XCTAssertTrue(c.isDone, "document should be complete")
        }
        let sorted = stepTimes.sorted()
        print(String(format: "[xgrammar] mask computation per step: mean %.3f ms, p50 %.3f ms, p95 %.3f ms, max %.3f ms over %d steps",
                     stepTimes.reduce(0, +) / Double(stepTimes.count), sorted[sorted.count / 2],
                     sorted[sorted.count * 95 / 100], sorted.last!, sorted.count))
    }

    /// Both engines over one prompt, interleaved: first-token latency, decode rate and a
    /// schema-valid document from each. Asserts validity, not speed.
    func testEngineComparison() throws {
        let chat = try loadModel()
        let format = ChatResponseFormat.jsonSchema(JSONSchemaMatcherTests.factSchema)
        let plain = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 1500, repetitionPenalty: 1.1)
        var constrained = plain
        constrained.responseFormat = format
        _ = try run(chat, ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 8))
        let swiftVocabulary = Date()
        _ = chat.constraintVocabulary()
        let xgrammarVocabulary = Date()
        _ = try chat.xgrammarVocabulary()
        print(String(format: "[engines] vocabulary: swift %.0f ms, xgrammar %.0f ms",
                     xgrammarVocabulary.timeIntervalSince(swiftVocabulary) * 1000,
                     Date().timeIntervalSince(xgrammarVocabulary) * 1000))
        for round in 1 ... 3 {
            for (label, sampling, engine) in [
                ("unconstrained", plain, JSONConstraintEngine.xgrammar),
                ("swift", constrained, .swift),
                ("xgrammar", constrained, .xgrammar),
            ] {
                let r = try run(chat, sampling, engine: engine)
                print(String(format: "[engines] round %d %@: first token %.0f ms, %d tokens, %.1f tok/s decode, total %.0f ms",
                             round, label, r.firstTokenMs, r.tokens, r.decodeTokensPerSecond, r.totalMs))
                if sampling.responseFormat != nil {
                    try assertValidFactDocument(r.text)
                    if round == 1 { print("[engines] \(label) output: \(r.text.prefix(400))") }
                }
            }
        }
    }

    /// `text` with whitespace outside strings removed.
    static func minified(_ text: String) -> [UInt8] {
        var out: [UInt8] = []
        var inString = false
        var escaped = false
        for byte in text.utf8 {
            if inString {
                out.append(byte)
                if escaped { escaped = false } else if byte == 0x5C { escaped = true } else if byte == 0x22 { inString = false }
            } else if byte == 0x22 {
                inString = true
                out.append(byte)
            } else if ![0x20, 0x0A, 0x0D, 0x09].contains(byte) {
                out.append(byte)
            }
        }
        return out
    }

    /// Every schema in `JSON_SCHEMA_CORPUS` (a JSON array of `{id, gate, schema}`): does the
    /// keyword gate accept it, and what does a cold compile cost each engine? Skipped without it.
    func testSchemaCorpusGateAndCompileCost() throws {
        guard let path = ProcessInfo.processInfo.environment["JSON_SCHEMA_CORPUS"] else {
            throw XCTSkip("set JSON_SCHEMA_CORPUS to a schema corpus")
        }
        try requireFile("tokenizer.json")
        let tokenizer = Gemma4Tokenizer()
        try tokenizer.load(from: Self.modelDir)
        let config = try Gemma4DenseConfig.load(from: Self.modelDir.appendingPathComponent("config.json"))
        let vocabulary = try XGrammarVocabulary(gemma: tokenizer, size: config.vocabSize)
        let entries = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: path))) as? [[String: String]])
        var swiftMs: [Double] = [], xgrammarMs: [Double] = []
        var rejected: [String] = []
        for entry in entries {
            let schema = entry["schema"] ?? ""
            let t0 = Date()
            do {
                _ = try JSONSchemaGrammar(schema: schema)
            } catch {
                rejected.append("\(entry["gate"] ?? "?") \(entry["id"] ?? ""): \(error.localizedDescription)")
                continue
            }
            swiftMs.append(Date().timeIntervalSince(t0) * 1000)
            let t1 = Date()
            _ = try vocabulary.constraint(schema: schema)
            xgrammarMs.append(Date().timeIntervalSince(t1) * 1000)
            print(String(format: "[corpus] %@ %@: swift %.1f ms, xgrammar %.0f ms",
                         entry["gate"] ?? "?", entry["id"] ?? "", swiftMs.last!, xgrammarMs.last!))
        }
        func summary(_ values: [Double]) -> String {
            let sorted = values.sorted()
            guard !sorted.isEmpty else { return "none" }
            return String(format: "mean %.1f ms, p50 %.1f ms, p95 %.1f ms, max %.1f ms",
                          values.reduce(0, +) / Double(values.count), sorted[sorted.count / 2],
                          sorted[min(sorted.count - 1, sorted.count * 95 / 100)], sorted.last!)
        }
        print("[corpus] \(entries.count) schemas, \(rejected.count) rejected by the keyword gate")
        rejected.forEach { print("[corpus] rejected \($0)") }
        print("[corpus] swift compile: \(summary(swiftMs))")
        print("[corpus] xgrammar cold compile: \(summary(xgrammarMs))")
    }
}
