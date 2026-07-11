import XCTest
import Foundation
import MLX
import MLXNN
@testable import MAGNeTMusicGen

/// Decodes the same deterministic synthetic codes the Python reference uses
/// (codes[k, t] = (k*100 + t) % 2048, K=4, T=100) and prints stats + the
/// first/last 10 samples so we can diff numerically against Python.
final class E2EEncodecComparisonTests: XCTestCase {
    func testDecodeMatchesPythonOnSyntheticCodes() async throws {
        let paths = try await MAGNeTDownloader.ensureDownloaded(
            variant: .smallInt4, progressHandler: nil)
        let encCfg = try EncodecModelConfig.load(
            from: paths.encodecDir.appendingPathComponent("config.json"))
        let weights = try MLX.loadArrays(
            url: paths.encodecDir.appendingPathComponent("model.safetensors"))
        let encodec = EncodecModelMLX(config: encCfg, numQuantizers: 4)
        let params = ModuleParameters.unflattened(Array(weights.map { ($0.key, $0.value) }))
        encodec.update(parameters: params)

        let T = 100
        var values: [Int32] = []
        values.reserveCapacity(4 * T)
        for k in 0..<4 {
            for t in 0..<T {
                values.append(Int32((k * T + t) % 2048))
            }
        }
        let codes = MLXArray(values).reshaped([1, 4, T])

        let audio = encodec.decode(codes)
        eval(audio)
        let flat = audio.asArray(Float.self)
        let peak = flat.map(abs).max()!
        let rms = (flat.reduce(0) { $0 + $1 * $1 } / Float(flat.count)).squareRoot()
        print("[Swift Encodec] shape=\(audio.shape) peak=\(peak) rms=\(rms)")
        print("[Swift Encodec] first 10: \(Array(flat.prefix(10)))")
        print("[Swift Encodec] last 10:  \(Array(flat.suffix(10)))")
        // Python reference (mlx-community/encodec-32khz-float32) gave:
        //   peak=1.3045  rms=0.3102
        //   first10[0..2]=[0.1132, 0.1364]
        // If we match to ~1e-4 we're functionally identical.
        XCTAssertEqual(peak, 1.3045, accuracy: 0.01,
                       "EnCodec peak should match Python reference within 1%")
        XCTAssertEqual(rms, 0.3102, accuracy: 0.005,
                       "EnCodec rms should match Python reference within 1.5%")
        XCTAssertEqual(flat[0], 0.11319, accuracy: 0.0005,
                       "First sample should match Python reference")
        XCTAssertEqual(flat[1], 0.13642, accuracy: 0.0005,
                       "Second sample should match Python reference")
    }
}
