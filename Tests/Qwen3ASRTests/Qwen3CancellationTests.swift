import XCTest
import MLX
@testable import Qwen3ASR

final class Qwen3CancellationTests: XCTestCase {
    private static func decode(slow: Bool, maxTokens: Int) -> [Int32] {
        var config = TextDecoderConfig()
        config.vocabSize = 64
        config.hiddenSize = 64
        config.numLayers = 0
        config.intermediateSize = 64
        let decoder = QuantizedTextModel(config: config)
        let logits = MLXArray(Array(repeating: Float(0), count: 64), [1, 1, 64])
        if slow {
            return Qwen3ASRModel.generateSlow(
                textDecoder: decoder, initialLogits: logits, cache: [],
                maxTokens: maxTokens, options: Qwen3DecodingOptions(noRepeatNgramSize: 3)
            )
        }
        return Qwen3ASRModel.generateGreedyAsyncEval(
            textDecoder: decoder, initialLogits: logits, cache: [], maxTokens: maxTokens
        )
    }

    func testCancelledTaskDoesNotStartEitherDecoder() async {
        for slow in [false, true] {
            let tokens = await Task.detached {
                withUnsafeCurrentTask { $0?.cancel() }
                return Self.decode(slow: slow, maxTokens: 2)
            }.value
            XCTAssertTrue(tokens.isEmpty)
        }
    }

    func testCancellationStopsBothDecoderLoops() async throws {
        for slow in [false, true] {
            // The tiny vocabulary cannot emit Qwen's EOS token, so only
            // cancellation can stop decoding before the token limit.
            let task = Task.detached { Self.decode(slow: slow, maxTokens: 10_000) }
            try await Task.sleep(for: .milliseconds(100))
            task.cancel()
            let tokens = await task.value
            XCTAssertLessThan(tokens.count, 10_000)
        }
    }
}
