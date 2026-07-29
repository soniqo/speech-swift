import XCTest
import AudioCommon
import MLXCommon
@testable import Qwen3Chat

/// Loads the real Qwen3.5-0.8B MLX checkpoint to prove two things about quantization:
/// the INT4 checkpoint still loads exactly as it always did, and a config that declares
/// a bit width the tensors do not have now fails with an error instead of quietly
/// building a mis-shaped model.
///
/// The second case is the one that mattered: every quantized layer used to be built with
/// literal `bits: 4`, and mlx-swift's `update(parameters:)` verifies nothing, so a packed
/// row of the wrong width was installed without a word.
final class E2EQwen35QuantizationLoadTests: XCTestCase {

    /// The cached (or downloadable) INT4 variant directory.
    private func int4Directory() throws -> URL {
        let cache = try HuggingFaceDownloader.getCacheDirectory(
            for: Qwen35MLXChat.defaultModelId)
        let dir = cache.appendingPathComponent("int4")
        guard FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("Qwen3.5-0.8B INT4 weights are not cached at \(dir.path)")
        }
        return dir
    }

    /// A directory that links the real weights but declares a different quantization.
    private func directoryDeclaring(_ quantizationFields: [String: Any]) throws -> URL {
        let source = try int4Directory()
        let staged = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen35-quant-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: staged, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: staged) }

        // Link rather than copy — the weights are ~400 MB.
        for file in ["model.safetensors", "tokenizer.json", "tokenizer_config.json"] {
            let original = source.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: original.path) else { continue }
            try FileManager.default.createSymbolicLink(
                at: staged.appendingPathComponent(file), withDestinationURL: original)
        }

        let configURL = source.appendingPathComponent("config.json")
        var config = try JSONSerialization.jsonObject(
            with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        config.merge(quantizationFields) { _, new in new }
        try JSONSerialization.data(withJSONObject: config)
            .write(to: staged.appendingPathComponent("config.json"))

        return staged
    }

    func testInt4CheckpointStillLoadsAndGenerates() async throws {
        let directory = try int4Directory()
        let model = try await Qwen35MLXChat.fromLocal(directory: directory)

        XCTAssertEqual(model.config.quantBits, 4)
        XCTAssertEqual(model.config.quantGroupSize, 64)

        let response = try model.generate(
            messages: [
                ChatMessage(role: .user, content: "What is 2+2? Reply with just the number.")
            ],
            sampling: ChatSamplingConfig(temperature: 0.1, topK: 10, maxTokens: 20))
        let trimmed = response.trimmingCharacters(in: .whitespacesAndNewlines)
        print("[quantization-e2e] INT4 reply: '\(trimmed)'")
        XCTAssertTrue(trimmed.contains("4"), "INT4 load regressed; got '\(trimmed)'")
    }

    func testDeclaringInt8OverInt4TensorsFailsClosed() async throws {
        let directory = try directoryDeclaring([
            "quantization": "int8",
            "quantization_bits": 8,
            "quantization_group_size": 64,
        ])

        do {
            _ = try await Qwen35MLXChat.fromLocal(directory: directory)
            XCTFail("an INT8 declaration over INT4 tensors must not load")
        } catch let error as QuantizedWeightMismatch {
            guard case .bits(let tensor, let declared, let implied) = error else {
                return XCTFail("expected a bit-width mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 8)
            XCTAssertEqual(implied, 4)
            print("[quantization-e2e] rejected at \(tensor): \(error.localizedDescription)")
        }
    }

    func testDeclaringTheWrongGroupSizeFailsClosed() async throws {
        let directory = try directoryDeclaring([
            "quantization_bits": 4,
            "quantization_group_size": 32,
        ])

        do {
            _ = try await Qwen35MLXChat.fromLocal(directory: directory)
            XCTFail("a group size the scales contradict must not load")
        } catch let error as QuantizedWeightMismatch {
            guard case .groupSize(_, let declared, let implied) = error else {
                return XCTFail("expected a group-size mismatch, got \(error)")
            }
            XCTAssertEqual(declared, 32)
            XCTAssertEqual(implied, 64)
        }
    }
}
