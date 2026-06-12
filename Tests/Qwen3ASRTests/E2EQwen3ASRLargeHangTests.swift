import XCTest
@testable import Qwen3ASR
@testable import AudioCommon

/// Regression test for the reported 1.7B inference hang. The 1.7B variant
/// uses the same `generateGreedyAsyncEval` loop as 0.6B but with a 2048
/// hidden dim / 28 layers / 8-bit quantized decoder — there have been
/// in-the-wild reports of `transcribe()` never returning.
///
/// Strategy: wrap the synchronous `transcribe` call in a background queue
/// and race it against `XCTestExpectation.wait(timeout:)`. If the timeout
/// fires we fail the test and leak the worker thread (acceptable for a
/// regression marker since the process exits when XCTest tears down).
///
/// All method names are prefixed `testLargeModel*` so the test lands in the
/// existing `qwen3-asr-17b-mlx` nightly shard (`.github/workflows/nightly-e2e.yml`
/// filters on that prefix). The class is prefixed `E2E` so the default
/// `--skip E2E` CI filter excludes it from the standard build.
final class E2EQwen3ASRLargeHangTests: XCTestCase {

    static let modelId = ASRModelSize.large.defaultModelId  // 1.7B-8bit

    private static var _model: Qwen3ASRModel?

    override func setUp() async throws {
        try await super.setUp()
        if Self._model == nil {
            Self._model = try await Qwen3ASRModel.fromPretrained(modelId: Self.modelId)
        }
    }

    private func wavAudio() throws -> (samples: [Float], sampleRate: Int) {
        let wavURL = try XCTUnwrap(
            Bundle.module.url(forResource: "test_audio", withExtension: "wav"),
            "test_audio.wav missing"
        )
        let (samples, sr) = try AudioFileLoader.loadWAV(url: wavURL)
        let target = 24000
        let audio: [Float] = (sr == target)
            ? samples
            : AudioFileLoader.resample(samples, from: sr, to: target)
        return (audio, target)
    }

    /// Runs `transcribe` on a background dispatch queue, races against a
    /// wall-clock budget. Fails (and leaks the worker) on timeout.
    private func transcribeWithTimeout(
        _ model: Qwen3ASRModel,
        audio: [Float],
        sampleRate: Int,
        maxTokens: Int,
        timeout: TimeInterval,
        label: String
    ) -> String? {
        let exp = expectation(description: "transcribe-\(label)")
        var captured: String?
        let start = Date()
        // Use a dedicated queue + DispatchWorkItem so cancellation is at least
        // visible to the dispatch system, even though MLX inside doesn't
        // honour cancellation.
        let queue = DispatchQueue(label: "qwen3-1.7b-hang-test-\(label)", qos: .userInitiated)
        queue.async {
            let result = model.transcribe(audio: audio, sampleRate: sampleRate, maxTokens: maxTokens)
            captured = result
            exp.fulfill()
        }
        let outcome = XCTWaiter().wait(for: [exp], timeout: timeout)
        let elapsed = Date().timeIntervalSince(start)
        if outcome == .completed {
            print("1.7B [\(label)] completed in \(elapsed)s: \"\(captured ?? "")\"")
            return captured
        } else {
            XCTFail(
                "1.7B transcribe[\(label)] did not return within \(timeout)s (outcome=\(outcome.rawValue), elapsed=\(elapsed)s) — hang regression. Worker thread is leaked; XCTest exit will surface this."
            )
            return nil
        }
    }

    /// First-token / short-horizon hang detector. If the 1.7B variant hangs
    /// even at `maxTokens=32` the bug is at the very first step (weight load
    /// stall, KV-cache init, position-encoder kernel — not inside the loop).
    /// Generous 90 s budget covers a cold 8-bit decoder warm-up.
    func testLargeModelDoesNotHangOnFirstToken() throws {
        guard let model = Self._model else { throw XCTSkip("1.7B model not loaded") }
        let (audio, sr) = try wavAudio()
        let result = transcribeWithTimeout(
            model, audio: audio, sampleRate: sr,
            maxTokens: 32, timeout: 90.0, label: "maxTokens=32"
        )
        if let result = result {
            XCTAssertFalse(result.isEmpty, "result should not be empty even at maxTokens=32")
        }
    }

    /// Full-horizon hang detector at the default `maxTokens=448`. A hang here
    /// but a pass at maxTokens=32 localizes the bug to the loop body (e.g.
    /// MLX async-eval starvation, quantized matmul drift, repeat penalty
    /// state growth). 180 s wall-clock budget is generous enough for cold
    /// CoreML/MLX kernel JIT.
    func testLargeModelDoesNotHangOnShortAudio() throws {
        guard let model = Self._model else { throw XCTSkip("1.7B model not loaded") }
        let (audio, sr) = try wavAudio()
        let result = transcribeWithTimeout(
            model, audio: audio, sampleRate: sr,
            maxTokens: 448, timeout: 180.0, label: "full-horizon"
        )
        if let result = result {
            XCTAssertFalse(result.isEmpty, "result should not be empty")
            // Quality sanity: at least one expected keyword should land. We
            // don't gate on a strict count because the focus here is
            // liveness, not quality.
            let lower = result.lowercased()
            let expected = ["guarantee", "replacement", "shipped", "tomorrow"]
            let found = expected.filter { lower.contains($0) }
            XCTAssertFalse(
                found.isEmpty,
                "1.7B produced no expected keywords: \"\(result)\""
            )
        }
    }
}
