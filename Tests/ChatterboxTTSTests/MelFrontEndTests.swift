import MLX
import MLXCommon
import XCTest

/// Numeric gate for the VoiceEncoder mel front-end (shared `SlaneyMel`), validated
/// against a golden dumped from mlx-audio's `melspectrogram` on the SAME prepared
/// 16 kHz input (`/tmp/cbx_ve_input16k.f32`), so the resample/trim steps are
/// isolated out and only the STFT + slaney filterbank + power are under test.
final class MelFrontEndTests: XCTestCase {
    private let inputPath = "/tmp/cbx_ve_input16k.f32"

    private var veMelConfig: SlaneyMelConfig {
        // VoiceEncConfig: 40-mel slaney, n_fft=400, hop=160, power=2.0, amp (no log).
        SlaneyMelConfig(
            sampleRate: 16000, nFft: 400, hop: 160, win: 400, nMels: 40,
            fmin: 0, fmax: 8000, power: 2.0, logMel: false, centerPad: true)
    }

    func testMelMatchesOracle() throws {
        guard FileManager.default.fileExists(atPath: inputPath) else {
            throw XCTSkip("golden input \(inputPath) not present; run /tmp/cbx_ve_golden.py")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: inputPath))
        let samples: [Float] = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        XCTAssertEqual(samples.count, 51200)

        let mel = SlaneyMel.melSpec(samples: samples, config: veMelConfig)
        XCTAssertEqual(mel.shape, [321, 40])

        let sum = MLX.sum(mel).item(Float.self)
        let mean = MLX.mean(mel).item(Float.self)
        // Golden: sum=1188.4485, mean=0.0925583.
        XCTAssertEqual(sum, 1188.4485, accuracy: 1.2)       // ~0.1%
        XCTAssertEqual(mean, 0.0925583, accuracy: 1e-4)
    }
}
