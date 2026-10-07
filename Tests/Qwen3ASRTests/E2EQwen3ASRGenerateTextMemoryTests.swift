import XCTest
import MLX
@testable import Qwen3ASR

/// Single-input decoder exits must release reusable MLX buffers, including
/// successful generation and cancellation before prefill or during decode.
///
/// Uses a weight-free, zero-layer decoder with the complete prompt
/// vocabulary. Decoder evaluation touches the GPU, so this runs with
/// the E2E suites without downloading model weights.
final class E2EQwen3ASRGenerateTextMemoryTests: XCTestCase {
    private func makeModelAndDecoder(numAudioTokens: Int) -> (
        model: Qwen3ASRModel, decoder: QuantizedTextModel, audioEmbeds: MLXArray
    ) {
        var config = TextDecoderConfig()
        config.vocabSize = TextDecoderConfig.small.vocabSize
        config.hiddenSize = 64
        config.numLayers = 0
        config.intermediateSize = 64
        let decoder = QuantizedTextModel(config: config)
        let model = Qwen3ASRModel(
            audioConfig: ASRModelSize.small.audioConfig,
            textConfig: config)
        let audioEmbeds = MLXArray.zeros([1, numAudioTokens, config.hiddenSize])
        return (model, decoder, audioEmbeds)
    }

    /// Fills the MLX cache with an unrelated buffer so a no-op fix can't
    /// pass the assertion below by accident (the cache already being
    /// empty before `generateText` ever runs).
    private func evaluateFiller() {
        let filler = MLXArray.zeros([1024, 1024])
        eval(filler)
    }

    private func inflateCache() {
        evaluateFiller()
        // Wait after the filler's Swift reference leaves scope: Metal's
        // completion handler can retain its buffer beyond eval's return.
        StreamOrDevice.default.stream.synchronize()
    }

    func testGenerateTextClearsMLXCacheOnSuccessfulReturn() {
        let priorCache = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.cacheLimit = priorCache
            MLX.Memory.clearCache()
        }
        MLX.Memory.cacheLimit = 256 * 1024 * 1024

        inflateCache()
        XCTAssertGreaterThan(
            MLX.Memory.snapshot().cacheMemory, 0,
            "precondition: the filler buffer should be sitting in the cache")

        let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
        _ = model.generateText(
            audioEmbeds: audioEmbeds,
            textDecoder: decoder,
            language: nil,
            maxTokens: 3,
            checkCancellation: {})

        XCTAssertEqual(
            MLX.Memory.snapshot().cacheMemory, 0,
            "generateText must clear the MLX cache on every successful return, or a " +
                "long-running process (e.g. speech-server) accumulates it across requests")
    }

    func testGenerateTextClearsMLXCacheOnCancellation() {
        let priorCache = MLX.Memory.cacheLimit
        defer {
            MLX.Memory.cacheLimit = priorCache
            MLX.Memory.clearCache()
        }
        MLX.Memory.cacheLimit = 256 * 1024 * 1024

        inflateCache()
        XCTAssertGreaterThan(
            MLX.Memory.snapshot().cacheMemory, 0,
            "precondition: the filler buffer should be sitting in the cache")

        let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
        XCTAssertThrowsError(
            try model.generateText(
                audioEmbeds: audioEmbeds,
                textDecoder: decoder,
                language: nil,
                maxTokens: 100,
                checkCancellation: { throw CancellationError() }))

        XCTAssertEqual(
            MLX.Memory.snapshot().cacheMemory, 0,
            "generateText must clear the MLX cache even when the request is cancelled")
    }

    func testGenerateTextClearsMLXCacheOnSlowPathReturn() {
        let priorCache = Memory.cacheLimit
        defer {
            Memory.cacheLimit = priorCache
            Memory.clearCache()
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        inflateCache()
        XCTAssertGreaterThan(Memory.snapshot().cacheMemory, 0)
        let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
        let text = model.generateText(
            audioEmbeds: audioEmbeds, textDecoder: decoder,
            language: nil, maxTokens: 3,
            decodingOptions: Qwen3DecodingOptions(repetitionPenalty: 1.15),
            checkCancellation: {})
        XCTAssertFalse(text.isEmpty)
        XCTAssertEqual(Memory.snapshot().cacheMemory, 0)
    }

    func testGenerateTextClearsMLXCacheAfterDecodeHasStarted() {
        for slow in [false, true] {
            let priorCache = Memory.cacheLimit
            defer {
                Memory.cacheLimit = priorCache
                Memory.clearCache()
            }
            Memory.cacheLimit = 256 * 1024 * 1024
            inflateCache()
            XCTAssertGreaterThan(Memory.snapshot().cacheMemory, 0)
            let (model, decoder, audioEmbeds) = makeModelAndDecoder(numAudioTokens: 4)
            var checkpoints = 0
            XCTAssertThrowsError(
                try model.generateText(
                    audioEmbeds: audioEmbeds, textDecoder: decoder,
                    language: nil, maxTokens: 100,
                    decodingOptions: slow
                        ? Qwen3DecodingOptions(repetitionPenalty: 1.15)
                        : Qwen3DecodingOptions(),
                    checkCancellation: {
                        checkpoints += 1
                        if checkpoints == 4 { throw CancellationError() }
                    })) { error in
                        XCTAssertTrue(error is CancellationError)
                    }
            XCTAssertEqual(checkpoints, 4)
            XCTAssertEqual(Memory.snapshot().cacheMemory, 0, "slow=\(slow)")
        }
    }

    func testCancellationCleanupUsesTheScopedStream() {
        Stream.withNewDefaultStream {
            testGenerateTextClearsMLXCacheAfterDecodeHasStarted()
        }
    }
}
