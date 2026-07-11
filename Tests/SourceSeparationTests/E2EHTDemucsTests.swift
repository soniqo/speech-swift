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

    /// int8 quantization quality: how much does quantizing the transformer
    /// Linear layers cost vs fp32? Informs the publish decision.
    func testInt8QuantizationParity() throws {
        guard FileManager.default.fileExists(atPath: Self.fp32Dir.appendingPathComponent("htdemucs_ft.safetensors").path),
              FileManager.default.fileExists(atPath: Self.parityFile.path) else {
            throw XCTSkip("fp32 weights or parity_m0.safetensors not present")
        }
        func snr(_ ref: MLXArray, _ x: MLXArray) -> Float {
            let d = x - ref
            let s = (ref * ref).sum().item(Float.self)
            let e = (d * d).sum().item(Float.self)
            return 10 * log10(s / max(e, 1e-20))
        }
        let dump = try MLX.loadArrays(url: Self.parityFile)
        let input = dump["input"]!, expected = dump["expected"]!

        let fp32 = try HTDemucsSeparator.fromLocal(directory: Self.fp32Dir)
        let outFp32 = fp32.models[0](input); eval(outFp32)

        let int8 = try HTDemucsSeparator.fromLocal(directory: Self.fp32Dir, quantizeBits: 8)
        let outInt8 = int8.models[0](input); eval(outInt8)

        let snrTorch = snr(expected, outInt8)
        let snrFp32 = snr(outFp32, outInt8)
        print("INT8 PARITY: SNR(int8 vs torch) = \(snrTorch) dB, SNR(int8 vs swift-fp32) = \(snrFp32) dB")
        XCTAssertGreaterThan(snrTorch, 20.0, "int8 should still track the reference (\(snrTorch) dB)")
    }

    /// Export a pre-quantized int8 bundle, reload it via `fromLocal`, and confirm
    /// it tracks both the in-memory int8 path and the PyTorch reference. Validates
    /// the publishable int8 round-trip (quantize → save → quantize-structure → load).
    func testInt8BundleRoundTrip() throws {
        guard FileManager.default.fileExists(atPath: Self.fp32Dir.appendingPathComponent("htdemucs_ft.safetensors").path),
              FileManager.default.fileExists(atPath: Self.parityFile.path) else {
            throw XCTSkip("fp32 weights or parity_m0.safetensors not present")
        }
        func snr(_ ref: MLXArray, _ x: MLXArray) -> Float {
            let d = x - ref
            let s = (ref * ref).sum().item(Float.self)
            let e = (d * d).sum().item(Float.self)
            return 10 * log10(s / max(e, 1e-20))
        }
        let dump = try MLX.loadArrays(url: Self.parityFile)
        let input = dump["input"]!, expected = dump["expected"]!

        let inMem = try HTDemucsSeparator.fromLocal(directory: Self.fp32Dir, quantizeBits: 8)
        let outMem = inMem.models[0](input); eval(outMem)

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("htd_int8_\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try HTDemucsSeparator.exportQuantizedBundle(fromDirectory: Self.fp32Dir, toDirectory: tmp)

        let reloaded = try HTDemucsSeparator.fromLocal(directory: tmp, modelName: "htdemucs_ft_int8")
        XCTAssertEqual(reloaded.config.quantization?.bits, 8)
        XCTAssertEqual(reloaded.config.quantization?.groupSize, 64)
        let outReloaded = reloaded.models[0](input); eval(outReloaded)

        XCTAssertEqual(outReloaded.shape, expected.shape)
        let snrVsMem = snr(outMem, outReloaded)
        let snrVsTorch = snr(expected, outReloaded)
        print("INT8 BUNDLE: SNR(reloaded vs in-mem int8) = \(snrVsMem) dB, SNR(reloaded vs torch) = \(snrVsTorch) dB")
        XCTAssertGreaterThan(snrVsMem, 35.0, "bundle reload should match in-memory int8 (\(snrVsMem) dB)")
        XCTAssertGreaterThan(snrVsTorch, 20.0, "int8 bundle should track the reference (\(snrVsTorch) dB)")
    }

    /// One-shot artifact export: set `HTDEMUCS_INT8_OUT=<dir>` to write the
    /// publishable int8 bundle (skipped otherwise, so CI/dev runs stay clean).
    func testExportInt8Artifact() throws {
        guard let outPath = ProcessInfo.processInfo.environment["HTDEMUCS_INT8_OUT"] else {
            throw XCTSkip("set HTDEMUCS_INT8_OUT=<dir> to export the int8 bundle")
        }
        guard FileManager.default.fileExists(atPath: Self.fp32Dir.appendingPathComponent("htdemucs_ft.safetensors").path) else {
            throw XCTSkip("fp32 weights not present")
        }
        let out = URL(fileURLWithPath: outPath)
        try HTDemucsSeparator.exportQuantizedBundle(fromDirectory: Self.fp32Dir, toDirectory: out)
        print("Wrote int8 bundle to \(out.path)")
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
