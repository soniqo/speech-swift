import XCTest
import MLX
import MLXRandom
@testable import MagpieTTS

final class MagpieSamplingTests: XCTestCase {

    func testTopKFilterKeepsTop2() {
        let logits = MLXArray([Float(0.1), 5.0, 2.0, 3.0, 0.5, 4.0])
        let filtered = topKFilter(logits, k: 2)
        // Keeps 5.0 (index 1) and 4.0 (index 5); everything else becomes -1e30.
        let vals = filtered.asArray(Float.self)
        XCTAssertEqual(vals[1], 5.0, accuracy: 1e-3)
        XCTAssertEqual(vals[5], 4.0, accuracy: 1e-3)
        XCTAssertLessThan(vals[0], -1e29)
        XCTAssertLessThan(vals[2], -1e29)
        XCTAssertLessThan(vals[3], -1e29)
        XCTAssertLessThan(vals[4], -1e29)
    }

    func testForbidIdsZeroesOutSpecificIndices() {
        let logits = MLXArray([Float(1.0), 2.0, 3.0, 4.0])
        let result = forbidIds(logits, ids: [1, 3])
        let vals = result.asArray(Float.self)
        XCTAssertEqual(vals[0], 1.0, accuracy: 1e-3)
        XCTAssertLessThan(vals[1], -1e29)
        XCTAssertEqual(vals[2], 3.0, accuracy: 1e-3)
        XCTAssertLessThan(vals[3], -1e29)
    }

    func testGreedySampleSelectsArgmax() {
        let logits = MLXArray([Float(0.1), 0.2, 5.0, 0.4])
        let tok = sampleTopK(logits, temperature: 0.0, k: 1)
        XCTAssertEqual(tok.asArray(Int32.self)[0], 2)
    }

    func testSampleWithSeedIsReproducible() {
        let logits = MLXArray([Float(1.0), 2.0, 1.5, 0.5, 2.5, 1.0])
        MLXRandom.seed(42)
        let a = sampleTopK(logits, temperature: 0.8, k: 3).asArray(Int32.self)[0]
        MLXRandom.seed(42)
        let b = sampleTopK(logits, temperature: 0.8, k: 3).asArray(Int32.self)[0]
        XCTAssertEqual(a, b)
    }
}
