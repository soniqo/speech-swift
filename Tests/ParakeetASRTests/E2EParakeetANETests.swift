import CoreML
import XCTest
@testable import ParakeetASR
@testable import AudioCommon

/// E2E coverage for the encoder loading on `.cpuAndNeuralEngine`.
///
/// `ParakeetASRModel.fromPretrained` pins macOS to `.cpuAndGPU` because the
/// INT8-palettized encoder has historically SIGSEGV'd during BNNS graph
/// compile on Apple Silicon ANE — first on early M-series, then on M5
/// (issue #313, reported on M5 Max running macOS 26.5 Tahoe). That pin
/// keeps the app safe but hides whether the shipped `.mlmodelc` itself
/// can ANE-compile, which is what embedders who load with CPU+ANE need.
///
/// This test bypasses the macOS pin: it loads `encoder.mlmodelc` with
/// `.cpuAndNeuralEngine` directly and runs one prediction so the BNNS
/// graph compile is forced.
///
/// Empirically the crash is process-shape sensitive — a bare `swift`
/// JIT process reproduces the BNNS SIGSEGV stack from issue #313 on
/// M5 Pro / macOS 26.5, but the `xctest` host (arm64e, macOS14
/// deployment target) succeeds with full ANE compile. So a green run
/// here does NOT prove the model is fixed for the M5 Max bug report,
/// only that the xctest-shaped load path works. A red/crashed run is
/// a real regression. After the export-side fix lands (either ship
/// `.mlpackage` for on-device AOT, or re-export the `.mlmodelc` with
/// a deployment target the M5 BNNS compiler accepts) this test should
/// continue to pass.
final class E2EParakeetANETests: XCTestCase {

    /// Force `encoder.mlmodelc` load on `.cpuAndNeuralEngine`, then run one
    /// prediction with a zero mel so the ANE graph compile is exercised.
    func testEncoderLoadsAndPredictsOnANE() async throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("ANE not available in the iOS simulator")
        #else
        // Make sure the model is on disk. fromPretrained uses .cpuAndGPU on
        // macOS so this download/load itself can never trip the ANE bug.
        let modelId = ParakeetASRModel.defaultModelId
        _ = try await ParakeetASRModel.fromPretrained(modelId: modelId)

        let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: modelId)
        let encoderURL = cacheDir.appendingPathComponent("encoder.mlmodelc", isDirectory: true)
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            XCTFail("encoder.mlmodelc missing at \(encoderURL.path) after fromPretrained")
            return
        }

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine

        // Loading the precompiled .mlmodelc triggers BNNS graph compile
        // inside MLE5ProgramLibrary's lazy init. On a known-bad export
        // (issue #313) this is where the process SIGSEGVs on M5 ANE —
        // the crash itself is the regression signal.
        let encoder = try MLModel(contentsOf: encoderURL, configuration: cfg)

        // Build a zero mel input shaped for the model's accepted length.
        // For range/enumerated exports any supported length works; we
        // grab the first one the loader code already discovers.
        let supportedLengths: [Int] = {
            guard let melDesc = encoder.modelDescription.inputDescriptionsByName["mel"],
                  let arr = melDesc.multiArrayConstraint else { return [3000] }
            switch arr.shapeConstraint.type {
            case .enumerated:
                let frames = arr.shapeConstraint.enumeratedShapes
                    .compactMap { $0.count >= 3 ? $0[2].intValue : nil }
                    .sorted()
                return frames.isEmpty ? [3000] : frames
            case .unspecified:
                let canonical = arr.shape
                return canonical.count >= 3 ? [canonical[2].intValue] : [3000]
            case .range:
                let perDim = arr.shapeConstraint.sizeRangeForDimension
                if perDim.count >= 3 {
                    let r = perDim[2].rangeValue
                    return [r.location + r.length]
                }
                return [3000]
            @unknown default:
                return [3000]
            }
        }()
        let melFrames = supportedLengths.first ?? 3000

        let mel = try MLMultiArray(
            shape: [1, 128, melFrames as NSNumber], dataType: .float16)
        let raw = mel.dataPointer.assumingMemoryBound(to: Float16.self)
        for i in 0..<mel.count { raw[i] = Float16(0) }

        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: Int32(melFrames))

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: mel),
            "length": MLFeatureValue(multiArray: length),
        ])

        // First prediction forces the ANE pipeline to actually dispatch —
        // the second class of M5 failure (load OK, predict crashes) shows
        // up here.
        let out = try await encoder.prediction(from: input)
        XCTAssertTrue(
            out.featureNames.contains("encoded"),
            "Encoder must expose an 'encoded' output (got \(out.featureNames))"
        )
        let encoded = out.featureValue(for: "encoded")?.multiArrayValue
        XCTAssertNotNil(encoded, "'encoded' must be an MLMultiArray")
        XCTAssertEqual(
            encoded?.shape.count, 3,
            "Encoder output must be a rank-3 multiarray [B, T, C]"
        )
        #endif
    }
}
