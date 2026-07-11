import XCTest
import MLX
@testable import CosyVoiceTTS

/// Mel-extractor invariants for the two extractors we ship in the cloning
/// path. Neither test loads any model weights — we synthesise input audio
/// (silence, pure tones, white noise) so the tests stay fast and don't need
/// a network round-trip. The invariants are deliberately loose: they catch
/// catastrophic regressions (wrong shape, dead silence, NaNs, mel filter
/// pointing at the wrong frequency) without locking us into an exact
/// upstream byte-comparison that vDSP's power-of-2 FFT can't deliver.
final class MelExtractorTests: XCTestCase {

    // MARK: - Test signals

    /// `seconds` of silence at `sampleRate`.
    private func silence(seconds: Float, sampleRate: Int) -> [Float] {
        [Float](repeating: 0, count: Int(seconds * Float(sampleRate)))
    }

    /// A pure sine wave at `freq` Hz, amplitude 0.5.
    private func sine(freq: Float, seconds: Float, sampleRate: Int) -> [Float] {
        let n = Int(seconds * Float(sampleRate))
        var out = [Float](repeating: 0, count: n)
        let step = 2 * Float.pi * freq / Float(sampleRate)
        for i in 0..<n {
            out[i] = 0.5 * sin(Float(i) * step)
        }
        return out
    }

    private func mlxToFloats(_ array: MLXArray) -> [Float] {
        array.asType(.float32).reshaped(-1).asArray(Float.self)
    }

    // MARK: - WhisperMelExtractor (128-mel @ 16 kHz, for the speech tokenizer)

    func testWhisperMelOutputShape() {
        let ex = WhisperMelExtractor()
        let audio = sine(freq: 440, seconds: 1.0, sampleRate: 16_000)
        let mel = ex.extract(audio)

        // [1, n_mels=128, T] — T derived from torch.stft center=True (reflect
        // pad n_fft/2 = 200 each side) then drop last frame, hop_length=160.
        XCTAssertEqual(mel.shape[0], 1, "Batch dim should be 1")
        XCTAssertEqual(mel.shape[1], 128, "Should produce 128 mel bins")
        // 1 s of audio is ~100 frames at 100 Hz; we accept ±2 for boundary handling.
        XCTAssertGreaterThan(mel.shape[2], 90)
        XCTAssertLessThan(mel.shape[2], 110)
    }

    func testWhisperMelSilenceIsBoundedLow() {
        let ex = WhisperMelExtractor()
        let mel = mlxToFloats(ex.extract(silence(seconds: 0.5, sampleRate: 16_000)))

        XCTAssertFalse(mel.isEmpty)
        XCTAssertFalse(mel.contains(where: { !$0.isFinite }), "No NaN/inf")

        // s3tokenizer normalisation `(x + 4) / 4` maps log10 floor 1e-10 → (-10+4)/4 = -1.5.
        // Silence should sit at that floor across every bin.
        let maxVal = mel.max() ?? .infinity
        XCTAssertLessThan(maxVal, -1.2, "Silence mel should sit near the -1.5 normalisation floor")
    }

    func testWhisperMelSineHasHigherEnergyThanSilence() {
        let ex = WhisperMelExtractor()
        let silenceMel = mlxToFloats(ex.extract(silence(seconds: 0.5, sampleRate: 16_000)))
        let sineMel = mlxToFloats(ex.extract(sine(freq: 1_000, seconds: 0.5, sampleRate: 16_000)))

        XCTAssertGreaterThan(sineMel.max() ?? -.infinity,
                             silenceMel.max() ?? -.infinity,
                             "A pure tone must produce higher-energy mel bins than silence")
    }

    func testWhisperMelIsDeterministic() {
        let ex = WhisperMelExtractor()
        let audio = sine(freq: 880, seconds: 0.3, sampleRate: 16_000)
        let a = mlxToFloats(ex.extract(audio))
        let b = mlxToFloats(ex.extract(audio))
        XCTAssertEqual(a.count, b.count)
        for i in 0..<min(a.count, b.count) {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-6, "Mel should be deterministic")
        }
    }

    // MARK: - FlowMelExtractor (80-mel @ 24 kHz, Matcha-TTS spec, for `prompt_feat`)

    func testFlowMelOutputShape() {
        let ex = FlowMelExtractor()
        let audio = sine(freq: 440, seconds: 1.0, sampleRate: 24_000)
        let mel = ex.extract(audio)

        // [1, n_mels=80, T] — T derived from manual reflect pad
        // (n_fft - hop)/2 = 720 each side, then center=False STFT.
        // For 1 s at 24 kHz: ~50 frames (50 Hz frame rate).
        XCTAssertEqual(mel.shape[0], 1)
        XCTAssertEqual(mel.shape[1], 80, "Should produce 80 mel bins")
        XCTAssertGreaterThan(mel.shape[2], 45)
        XCTAssertLessThan(mel.shape[2], 55)
    }

    func testFlowMelNaturalLogRange() {
        // Matcha's compression is `log(clamp(mel, min=1e-5))` — natural log,
        // floor at log(1e-5) ≈ -11.5, with values typically below ~2 for
        // normal-volume speech. The vDSP 2× bin-scaling fix lives in this
        // extractor; the test pins the value range so a regression in either
        // direction (squared values, missing log) is caught.
        let ex = FlowMelExtractor()
        let mel = mlxToFloats(ex.extract(sine(freq: 1_000, seconds: 0.5, sampleRate: 24_000)))

        let minVal = mel.min() ?? 0
        let maxVal = mel.max() ?? 0

        XCTAssertGreaterThanOrEqual(minVal, -11.6, "Min should not go below log(1e-5)")
        // A 0.5-amplitude sine should never push log-mel above ~3.
        XCTAssertLessThan(maxVal, 3.0, "Max log-mel for a pure tone should stay bounded")
        XCTAssertGreaterThan(maxVal, -5.0, "Sine peak should be well above the floor")
    }

    func testFlowMelSilenceSitsAtClampFloor() {
        let ex = FlowMelExtractor()
        let mel = mlxToFloats(ex.extract(silence(seconds: 0.3, sampleRate: 24_000)))
        XCTAssertFalse(mel.isEmpty)
        XCTAssertFalse(mel.contains(where: { !$0.isFinite }), "No NaN/inf in silence")
        let maxVal = mel.max() ?? .infinity
        XCTAssertLessThan(maxVal, -10.0,
                          "Silence should sit at log(1e-5) ≈ -11.5 (clamp floor)")
    }

    func testFlowMelIsDeterministic() {
        let ex = FlowMelExtractor()
        let audio = sine(freq: 880, seconds: 0.4, sampleRate: 24_000)
        let a = mlxToFloats(ex.extract(audio))
        let b = mlxToFloats(ex.extract(audio))
        XCTAssertEqual(a.count, b.count)
        for i in 0..<min(a.count, b.count) {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-6)
        }
    }

    // MARK: - Cross-check: the two extractors produce DIFFERENT scales

    func testWhisperAndFlowProduceDifferentNormalisations() {
        // WhisperMelExtractor applies `(x + 4)/4`, FlowMelExtractor does not.
        // Silence under Whisper sits near -1.5; under Flow it sits at log(1e-5) ≈ -11.5.
        // If anyone accidentally cross-wires the normalisations, this test catches it.
        let wMel = mlxToFloats(WhisperMelExtractor().extract(silence(seconds: 0.3, sampleRate: 16_000)))
        let fMel = mlxToFloats(FlowMelExtractor().extract(silence(seconds: 0.3, sampleRate: 24_000)))
        XCTAssertNotEqual(wMel.max() ?? 0, fMel.max() ?? 0, accuracy: 0.5,
                          "Whisper (s3tokenizer norm) and Flow (matcha log) must not collapse to the same value range")
    }
}
