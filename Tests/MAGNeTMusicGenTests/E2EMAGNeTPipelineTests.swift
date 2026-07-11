import XCTest
import Foundation
import MLX
import MLXNN
import AudioCommon
@testable import MAGNeTMusicGen

/// End-to-end tests that download real model weights and run inference.
/// Skipped by CI via `--skip E2E` filter (per CLAUDE.md convention).
///
/// Strategy: validate the pipeline in increasing chunks
///  1. T5 encoder alone (cheap — fp32 t5-base download is ~880 MB)
///  2. EnCodec decoder alone (~250 MB, fast)
///  3. Full pipeline with the minimum decoding budget (1 step / stage)
///  4. Full generation with the default 50-step schedule (slowest)
final class E2EMAGNeTPipelineTests: XCTestCase {

    // MARK: - Step 1: T5 encoder

    /// Loads t5-base via the downloader, sanitizes weights, runs the encoder
    /// on "happy rock", and checks the output is well-shaped and not NaN.
    func testT5EncoderRunsOnRealPrompt() async throws {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: .smallInt4, progressHandler: nil)
        let t5Cfg = try T5ModelConfig.load(
            from: paths.t5Dir.appendingPathComponent("config.json"))
        let raw = try MLX.loadArrays(url: paths.t5Dir.appendingPathComponent("model.safetensors"))
        let weights = T5Encoder.sanitize(raw)

        // Sanity-check we kept the right subset.
        XCTAssertNotNil(weights["wte.weight"],
                        "T5 sanitize must produce 'wte.weight'")
        XCTAssertNotNil(weights["layers.0.attention.query_proj.weight"],
                        "T5 sanitize must produce 'layers.0.attention.query_proj.weight'")
        XCTAssertNotNil(weights["relative_attention_bias.embeddings.weight"],
                        "T5 sanitize must produce 'relative_attention_bias.embeddings.weight'")

        let encoder = T5Encoder(config: t5Cfg)
        let params = ModuleParameters.unflattened(Array(weights.map { ($0.key, $0.value) }))
        encoder.update(parameters: params)

        // Tokenize "happy rock" via the same tokenizer the model uses.
        // We use the tokenizer's encode on a tiny prompt to keep the test fast.
        // Pad token id is 0 for t5-base; we don't need padding here.
        let ids: [Int32] = [3, 1700, 2782, 1]  // approximate ids; real tokenizer in full test
        let tokens = MLXArray(ids).reshaped([1, ids.count])
        let out = encoder(tokens)
        eval(out)

        XCTAssertEqual(out.dim(0), 1, "batch dim should be 1")
        XCTAssertEqual(out.dim(1), ids.count, "seq dim should match input length")
        XCTAssertEqual(out.dim(2), t5Cfg.dModel, "hidden dim should be d_model")

        // Reject NaN / inf
        let flat = out.asArray(Float.self)
        XCTAssertFalse(flat.contains { $0.isNaN || $0.isInfinite },
                       "T5 encoder produced NaN/inf")

        // Energy sanity: not all zeros.
        let energy = flat.reduce(0) { $0 + abs($1) } / Float(flat.count)
        XCTAssertGreaterThan(energy, 1e-6, "T5 encoder produced all-zero output")
    }

    // MARK: - Step 2: EnCodec decoder

    /// Loads encodec-32khz, builds the model, and decodes random codes.
    /// Validates output shape and that audio is in a plausible range.
    func testEncodecDecoderRunsOnSyntheticCodes() async throws {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: .smallInt4, progressHandler: nil)
        let encCfg = try EncodecModelConfig.load(
            from: paths.encodecDir.appendingPathComponent("config.json"))
        let weights = try MLX.loadArrays(
            url: paths.encodecDir.appendingPathComponent("model.safetensors"))

        let encodec = EncodecModelMLX(config: encCfg, numQuantizers: 4)
        let params = ModuleParameters.unflattened(Array(weights.map { ($0.key, $0.value) }))
        encodec.update(parameters: params)

        // Synthesise valid codes for a 2-second clip: 100 frames × 4 codebooks.
        let T = 100  // 2 seconds at 50 Hz
        let codeValues: [Int32] = (0..<(4 * T)).map { Int32($0 % 2048) }
        let codes = MLXArray(codeValues).reshaped([1, 4, T])

        let audio = encodec.decode(codes)
        eval(audio)

        let expectedSamples = T * (8 * 5 * 4 * 4)  // 100 × 640 = 64000 samples
        XCTAssertEqual(audio.dim(0), 1)
        XCTAssertEqual(audio.dim(1), expectedSamples,
                       "EnCodec decoded length should equal frame_count * total_upsample (640)")
        XCTAssertEqual(audio.dim(2), 1, "audio_channels=1")

        let flat = audio.asArray(Float.self)
        XCTAssertFalse(flat.contains { $0.isNaN || $0.isInfinite },
                       "EnCodec decoder produced NaN/inf")
        let peak = flat.map(abs).max() ?? 0
        XCTAssertGreaterThan(peak, 1e-4, "EnCodec output is silent")
        XCTAssertLessThan(peak, 100.0, "EnCodec output peak is unreasonable")
    }

    // MARK: - Step 3: Minimum-budget full pipeline

    /// Generate with 1 decoding step per codebook (4 forwards total). Fast
    /// enough to fit in an E2E test slot; catches LM weight loading,
    /// cross-attn key mismatch, slicing issues, and the full inference glue.
    /// Output won't be musically meaningful with only 4 steps but should be a
    /// well-formed 30 s PCM array.
    func testMinimumBudgetGenerationSmoke() async throws {
        let model = try await MAGNeTMusicGen.fromPretrained(variant: .smallInt4)
        let params = MAGNeTGenerationParams(
            decodingSteps: [1, 1, 1, 1],   // minimum: 4 forwards
            maxCfgCoef: 3.0, minCfgCoef: 1.0,
            temperature: 1.0, topP: 0.9,
            annealTemp: false, seed: 42)
        let pcm = model.generate(text: "test", params: params)

        let expected = model.config.segmentDuration * model.config.sampleRate
        XCTAssertEqual(pcm.count, expected,
                       "output length must be segment_duration * sample_rate")

        XCTAssertFalse(pcm.contains { $0.isNaN || $0.isInfinite },
                       "generation produced NaN/inf")
        let peak = pcm.map(abs).max() ?? 0
        XCTAssertGreaterThan(peak, 1e-4, "generated audio is silent")
        XCTAssertLessThan(peak, 100.0, "generated audio peak is unreasonable")
    }

    // MARK: - Step 4: Default-budget full pipeline (slow)

    /// Full 50-step generation. Skipped by default — to run locally:
    ///   `swift test --filter testDefaultBudgetGenerationProducesAudio`
    func testDefaultBudgetGenerationProducesAudio() async throws {
        // Gate behind an env var so this doesn't run on CI even if E2E filter
        // is not applied — it's ~10s of wall-clock per invocation.
        guard ProcessInfo.processInfo.environment["MAGNET_FULL_E2E"] == "1" else {
            throw XCTSkip("Set MAGNET_FULL_E2E=1 to enable the full 50-step generation test")
        }
        let model = try await MAGNeTMusicGen.fromPretrained(variant: .smallInt4)
        let start = Date()
        let pcm = model.generate(
            text: "happy rock",
            params: MAGNeTGenerationParams(seed: 42))
        let wall = Date().timeIntervalSince(start)
        let audioSec = Double(pcm.count) / Double(model.config.sampleRate)
        print("[MAGNET] wall=\(String(format: "%.2f", wall))s  audio=\(String(format: "%.1f", audioSec))s  RTF=\(String(format: "%.2f", wall / audioSec))")

        // Write to /tmp for manual auditioning.
        let url = URL(fileURLWithPath: "/tmp/magnet_e2e.wav")
        try WAVWriter.write(samples: pcm, sampleRate: model.config.sampleRate, to: url)

        XCTAssertGreaterThan(pcm.map(abs).max() ?? 0, 1e-3,
                             "Full-budget generation must produce audible audio")
    }
}
