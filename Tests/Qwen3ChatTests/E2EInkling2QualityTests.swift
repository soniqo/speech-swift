import Foundation
import CoreML
import MLX
import XCTest
@testable import Qwen3Chat

/// A small, explicit acceptance corpus, not a general model benchmark.
final class E2EInkling2QualityTests: XCTestCase {
    func testTextAndImageAcceptanceCorpus() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let directory = env["INKLING2_MODEL_DIR"] else {
            throw XCTSkip("Set INKLING2_MODEL_DIR to run the quality corpus")
        }
        let backend = env["INKLING2_QUALITY_BACKEND"] ?? "mlx"
        let generate: ([ChatMessage], [Inkling2Image], ChatSamplingConfig) -> AsyncThrowingStream<String, Error>
        let encode: (String) -> [Int]
        switch backend {
        case "mlx":
            MLX.Memory.cacheLimit = 128 * 1024 * 1024
            let model = try Inkling2Chat.fromDirectory(URL(fileURLWithPath: directory))
            generate = { messages, images, sampling in
                model.generateStream(messages: messages, reasoningEffort: 0, sampling: sampling, images: images)
            }
            encode = model.encode
        case "coreml":
            let units: [String: MLComputeUnits] = ["cpu": .cpuOnly, "gpu": .cpuAndGPU, "ane": .cpuAndNeuralEngine, "all": .all]
            guard let compute = units[env["INKLING2_COREML_COMPUTE_UNITS"] ?? "cpu"] else {
                throw Inkling2Error.invalid("invalid Core ML compute units for quality test")
            }
            let model = try Inkling2CoreMLChat.fromDirectory(URL(fileURLWithPath: directory), computeUnits: compute)
            generate = { messages, images, sampling in
                model.generateStream(messages: messages, reasoningEffort: 0, sampling: sampling, images: images)
            }
            encode = model.encode
        default: throw Inkling2Error.invalid("quality backend must be mlx or coreml")
        }
        let fixtures = Bundle.module.resourceURL!.appendingPathComponent("Inkling2")
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtures.appendingPathComponent("quality-cases.json"))) as! [[String: Any]]
        let output = URL(fileURLWithPath: env["INKLING2_QUALITY_OUTPUT"] ?? "/tmp/inkling-quality/results.json")
        var results: [[String: Any]] = []
        let filter = env["INKLING2_QUALITY_FILTER"]
        for item in cases where filter == nil || (item["id"] as! String).contains(filter!) {
            let messages = (item["messages"] as! [[String: String]]).map {
                ChatMessage(role: ChatMessage.Role(rawValue: $0["role"]!)!, content: $0["content"]!)
            }
            let images = try (item["image"] as? String).map { [try Inkling2Image.load(fixtures.appendingPathComponent($0))] } ?? []
            let sampling = ChatSamplingConfig(temperature: 0, topK: 0, topP: 1, maxTokens: 512, repetitionPenalty: 1)
            var text = "", firstText: Double?
            let start = Date()
            for try await part in generate(messages, images, sampling) {
                if firstText == nil { firstText = Date().timeIntervalSince(start) }
                text += part
            }
            var passed = !text.isEmpty && !text.contains("<|") && !text.contains("\u{FFFD}")
            if let pattern = item["pattern"] as? String {
                passed = passed && text.range(of: pattern, options: .regularExpression) != nil
            }
            if let expected = item["json_expected"] as? [String: Any] {
                let parsed = try? JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any]
                passed = passed && parsed.map { NSDictionary(dictionary: $0).isEqual(to: expected) } == true
            }
            if let limit = item["max_words"] as? Int { passed = passed && text.split(whereSeparator: \.isWhitespace).count <= limit }
            for forbidden in item["forbidden"] as? [String] ?? [] { passed = passed && !text.localizedCaseInsensitiveContains(forbidden) }
            var result = item
            result["response"] = text; result["passed"] = passed
            result["seconds"] = Date().timeIntervalSince(start)
            result["first_visible_text_seconds"] = firstText
            result["visible_output_tokens"] = encode(text).count
            results.append(result)
            let report: [String: Any] = ["scope": "25 fixed acceptance cases; deterministic greedy decoding; not a broad quality benchmark",
                "hardware": env["INKLING2_QUALITY_HARDWARE"] ?? "Apple Silicon", "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "model": URL(fileURLWithPath: directory).lastPathComponent, "backend": backend,
                "temperature": 0, "max_tokens": 512,
                "reasoning_effort": 0, "results": results]
            try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]).write(to: output, options: .atomic)
            print("[inkling-quality] \(item["id"]!) passed=\(passed) response=\(text)")
            XCTAssertTrue(passed, "\(item["id"]!): \(text)")
        }
    }
}
