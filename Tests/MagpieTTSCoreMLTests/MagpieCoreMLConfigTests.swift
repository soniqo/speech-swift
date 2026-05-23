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

    func testParamsClampMaxStepsToNanocodecLimit() {
        // Caller asks for 500 frames (MLX default); the CoreML codec is
        // window-limited to 256, so the params struct must clamp to avoid
        // an out-of-range Nanocodec call.
        let p = MagpieCoreMLParams(maxSteps: 500)
        XCTAssertEqual(p.maxSteps, MagpieCoreMLConstants.maxNanocodecFrames)
        XCTAssertEqual(p.maxSteps, 256)
    }

    func testParamsDefaultsMatchBundle() {
        // The reference bundle's `constants/constants.json` ships
        //   temperature=0.6 topk=80 cfg_scale=2.5 max_decoder_steps=500
        //   min_generated_frames=4
        // We default cfg_scale to 1.0 in our params (single decoder_step call
        // per frame — half the wall time) and clamp max_decoder_steps to the
        // codec's hard cap. Other knobs match.
        let p = MagpieCoreMLParams()
        XCTAssertEqual(p.temperature, 0.6, accuracy: 1e-6)
        XCTAssertEqual(p.topK, 80)
        XCTAssertEqual(p.minFrames, 4)
        XCTAssertEqual(p.cfgScale, 1.0, accuracy: 1e-6)
    }
}
