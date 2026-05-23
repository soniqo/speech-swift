import XCTest
@testable import MagpieTTSCoreML
@testable import MagpieTTS

final class MagpieCoreMLConfigTests: XCTestCase {

    func testSpeakerNameLookup() {
        XCTAssertEqual(MagpieCoreMLSpeaker(named: "john"), .john)
        XCTAssertEqual(MagpieCoreMLSpeaker(named: "John Van Stan"), .john)
        XCTAssertEqual(MagpieCoreMLSpeaker(named: "Sofia"), .sofia)
        XCTAssertEqual(MagpieCoreMLSpeaker(named: "ARIA"), .aria)
        XCTAssertNil(MagpieCoreMLSpeaker(named: "nobody"))
    }

    func testSpeakerOrderingMatchesCoreMLBundle() {
        // FluidInference's `constants/speaker_info.json` ships
        //   {"0": "John", "1": "Sofia", "2": "Aria", "3": "Jason", "4": "Leo"}
        // The enum's raw values must match exactly so we index the right
        // `speaker_*.npy` file on disk.
        XCTAssertEqual(MagpieCoreMLSpeaker.john.rawValue,  0)
        XCTAssertEqual(MagpieCoreMLSpeaker.sofia.rawValue, 1)
        XCTAssertEqual(MagpieCoreMLSpeaker.aria.rawValue,  2)
        XCTAssertEqual(MagpieCoreMLSpeaker.jason.rawValue, 3)
        XCTAssertEqual(MagpieCoreMLSpeaker.leo.rawValue,   4)
    }

    func testSpeakerMlxBridge() {
        // CoreML index → MLX speaker enum. Used by the CLI when --language ja
        // is requested with --engine magpie-coreml: the CoreML bundle has no
        // JA support so the call routes to the MLX backend, and we need the
        // speaker identity to survive that handoff.
        XCTAssertEqual(MagpieCoreMLSpeaker.john.mlxSpeaker,  .johnVanStan)
        XCTAssertEqual(MagpieCoreMLSpeaker.sofia.mlxSpeaker, .sofia)
        XCTAssertEqual(MagpieCoreMLSpeaker.aria.mlxSpeaker,  .aria)
        XCTAssertEqual(MagpieCoreMLSpeaker.jason.mlxSpeaker, .jason)
        XCTAssertEqual(MagpieCoreMLSpeaker.leo.mlxSpeaker,   .leo)
    }

    func testLanguageRoundTripExcludesJapanese() {
        // Round-trip every CoreML language through the MLX enum and back.
        for coreLang in MagpieCoreMLLanguage.allCases {
            let mlx = coreLang.mlx
            XCTAssertEqual(MagpieCoreMLLanguage(mlx: mlx), coreLang)
        }
        // Japanese is intentionally unmappable.
        XCTAssertNil(MagpieCoreMLLanguage(mlx: .japanese))
    }

    func testParamsClampMaxStepsToARCap() {
        // Asking for more than the AR cap (500) is silently clamped. The
        // codec chunks longer sequences internally so this isn't strictly
        // required by the codec — but it matches the MLX backend's hard
        // cap so behaviour is consistent across engines.
        let p = MagpieCoreMLParams(maxSteps: 600)
        XCTAssertEqual(p.maxSteps, MagpieCoreMLConstants.maxARSteps)
        XCTAssertEqual(p.maxSteps, 500)
    }

    func testParamsDefaultsMatchMlx() {
        // Same defaults as the MLX backend so callers can swap engines
        // without seeing different temperature/topK/min-frames behaviour.
        let p = MagpieCoreMLParams()
        XCTAssertEqual(p.temperature, 0.6, accuracy: 1e-6)
        XCTAssertEqual(p.topK, 80)
        XCTAssertEqual(p.minFrames, 4)
    }

    func testFsqInverseBosCodeDecodesToZero() {
        // BOS id = 2016. FSQ inverse formula:
        //   d_j = (i // base[j]) % level[j]
        // base = [1, 8, 56, 336], levels = [8, 7, 6, 6]
        // For i = 2016:
        //   d_0 = 2016 % 8       = 0   → (0 - 4) / 4 = -1
        //   d_1 = (2016/8) % 7   = 252 % 7 = 0 → (0 - 3) / 3 = -1
        //   d_2 = (2016/56) % 6  = 36  % 6 = 0 → (0 - 3) / 3 = -1
        //   d_3 = (2016/336) % 6 = 6   % 6 = 0 → (0 - 3) / 3 = -1
        // So all 4 dequants for BOS are -1.0 — sanity-check the inverse
        // implementation against this trivially-derivable value.
        let codes: [Int32] = Array(repeating: 2016, count: 8)
        let latent = MagpieCoreMLFSQ.decodeFrame(codes: codes)
        XCTAssertEqual(latent.count, 32)
        for v in latent {
            XCTAssertEqual(v, -1.0, accuracy: 1e-6, "FSQ inverse on BOS should be -1")
        }
    }
}
