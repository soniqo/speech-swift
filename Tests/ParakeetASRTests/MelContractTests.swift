import CoreML
import XCTest
@testable import ParakeetASR

/// Holds the mel front end to NeMo's `AudioToMelSpectrogramPreprocessor`, which
/// is what the Parakeet and Canary checkpoints were trained with.
///
/// The reference in `Resources/nemo_mel_reference.json` was produced by running
/// that preprocessor directly on a deterministic signal — a linear congruential
/// generator, reproduced bit-for-bit below, so no audio fixture is involved and
/// nothing resamples in between.
///
/// Worth a dedicated test because the failure is quiet. Reflect padding instead
/// of constant, or a periodic Hann window instead of symmetric, still produces
/// a plausible-looking spectrogram and a fluent transcript — it just costs
/// accuracy. Measured on the same contract mismatch elsewhere: +0.3 WER for
/// English, +0.6 for German and +2.0 for French on FLEURS.
final class MelContractTests: XCTestCase {

    private struct Reference: Decodable {
        let frames: Int
        let bins: Int
        let validFrames: Int
        let melFrames: Int
        let mel: [Float]   // frame-major: melFrames × bins
    }

    /// The exact samples the reference was generated from.
    private func referenceSignal(count: Int) -> [Float] {
        var state: UInt32 = 12345
        var samples = [Float](repeating: 0, count: count)
        for i in 0..<count {
            state = state &* 1664525 &+ 1013904223
            samples[i] = (Float(state) / 2147483648.0 - 1.0) * 0.3
        }
        return samples
    }

    private func loadReference() throws -> Reference {
        guard let url = Bundle.module.url(forResource: "nemo_mel_reference", withExtension: "json") else {
            throw XCTSkip("nemo_mel_reference.json not in test resources")
        }
        return try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
    }

    func testMelMatchesTheTrainingPreprocessor() throws {
        let reference = try loadReference()
        let preprocessor = MelPreprocessor(config: .default)
        let audio = referenceSignal(count: 16000 * 2)

        let (mel, melLength) = try preprocessor.extract(audio)
        XCTAssertEqual(melLength, reference.validFrames,
                       "frame count should follow NeMo's floor(samples / hop)")

        let bins = reference.bins
        let frames = reference.melFrames
        // The preprocessor emits float16 — reading it as Float32 walks off the
        // end of the buffer and produces garbage rather than a small deviation.
        XCTAssertEqual(mel.dataType, .float16)
        let pointer = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        let stride = mel.shape[2].intValue

        var worst: Float = 0
        var worstAt = (bin: 0, frame: 0)
        var sumSquared: Double = 0
        for t in 0..<frames {
            for b in 0..<bins {
                let expected = reference.mel[t * bins + b]
                let actual = Float(pointer[b * stride + t])
                let delta = abs(expected - actual)
                sumSquared += Double(delta * delta)
                if delta > worst {
                    worst = delta
                    worstAt = (b, t)
                }
            }
        }
        let rms = sqrt(sumSquared / Double(frames * bins))
        print("mel vs NeMo — worst \(worst), RMS \(rms)")

        // Float32 signal processing against a float32 reference: differences
        // should sit in rounding, not in the third decimal place.
        XCTAssertLessThan(worst, 0.05,
            "worst mel deviation \(worst) at bin \(worstAt.bin), frame \(worstAt.frame); RMS \(rms). "
            + "A gap this size means the front end is off the training contract — check the "
            + "STFT padding mode and the Hann window symmetry before anything else.")
        XCTAssertLessThan(rms, 0.01, "RMS mel deviation \(rms)")
    }
}
