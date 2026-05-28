import XCTest
import MLX
import MLXRandom
@testable import SourceSeparation

/// Dev parity/smoke tests for the HTDemucs (Demucs v4) port. Uses the weights
/// exported locally by speech-models. Skips if they aren't present.
final class E2EHTDemucsTests: XCTestCase {

    static var weightsDir: URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("repos/speech-models-htdemucs/models/htdemucs/export/htdemucs-ft")
    }

    private func skipIfMissing() throws {
        let st = Self.weightsDir.appendingPathComponent("htdemucs_ft.safetensors")
        if !FileManager.default.fileExists(atPath: st.path) {
            throw XCTSkip("htdemucs_ft weights not exported at \(st.path)")
        }
    }

    /// The make-or-break structural check: every Swift module key must match the
    /// exported safetensors (verify: .all in fromLocal throws otherwise).
    func testLoadAllSubModels() throws {
        try skipIfMissing()
        let sep = try HTDemucsSeparator.fromLocal(directory: Self.weightsDir)
        XCTAssertEqual(sep.models.count, 4)
        XCTAssertEqual(sep.config.sources, ["drums", "bass", "other", "vocals"])
        XCTAssertEqual(sep.config.arch.bottomChannels, 512)
    }

    static var fp32Dir: URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("repos/speech-models-htdemucs/models/htdemucs/export/htdemucs-ft-fp32")
    }
    static var parityFile: URL {
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("repos/speech-models-htdemucs/models/htdemucs/export/parity_m0.safetensors")
    }

    /// THE correctness gate: run sub-model 0 on the same input as the PyTorch
    /// reference and compare. High SNR ⇒ the architecture is numerically faithful.
    func testNumericalParityModel0() throws {
        guard FileManager.default.fileExists(atPath: Self.fp32Dir.appendingPathComponent("htdemucs_ft.safetensors").path),
              FileManager.default.fileExists(atPath: Self.parityFile.path) else {
            throw XCTSkip("fp32 weights or parity_m0.safetensors not present")
        }
        let sep = try HTDemucsSeparator.fromLocal(directory: Self.fp32Dir)
        let dump = try MLX.loadArrays(url: Self.parityFile)
        let input = dump["input"]!          // [1, 2, N]
        let expected = dump["expected"]!    // [1, 4, 2, N]

        let out = sep.models[0](input)
        eval(out)
        XCTAssertEqual(out.shape, expected.shape)

        let diff = out - expected
        let sigPow = (expected * expected).sum().item(Float.self)
        let errPow = (diff * diff).sum().item(Float.self)
        let snr = 10 * log10(sigPow / max(errPow, 1e-20))
        let maxAbs = MLX.abs(diff).max().item(Float.self)
        print("PARITY model_0: SNR = \(snr) dB, maxAbsDiff = \(maxAbs)")
        XCTAssertGreaterThan(snr, 20.0, "Swift output should match the PyTorch reference (SNR \(snr) dB)")
    }

    /// Forward runs end-to-end and produces the right shape with finite values.
    func testForwardShape() throws {
        try skipIfMissing()
        let sep = try HTDemucsSeparator.fromLocal(directory: Self.weightsDir)
        let n = sep.config.trainingLength
        let mix = MLXRandom.normal([1, 2, n]) * 0.1
        let out = sep.models[0](mix)
        eval(out)
        XCTAssertEqual(out.shape, [1, 4, 2, n])
        let flat = out.asArray(Float.self)
        XCTAssertFalse(flat.contains { $0.isNaN || $0.isInfinite }, "output must be finite")
    }
}
