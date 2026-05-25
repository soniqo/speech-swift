import XCTest
@testable import StableAudio3MusicGen

final class ConfigurationTests: XCTestCase {
    func testVariantRepoIds() {
        XCTAssertEqual(StableAudio3Variant.mediumInt8.huggingFaceRepoId,
                       "aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit")
        XCTAssertEqual(StableAudio3Variant.smallMusicInt4.huggingFaceRepoId,
                       "aufklarer/Stable-Audio-3-DiT-Small-Music-MLX-4bit")
    }

    func testVariantBits() {
        for v in StableAudio3Variant.allCases {
            XCTAssertTrue(v.bits == 4 || v.bits == 8)
        }
        XCTAssertEqual(StableAudio3Variant.mediumInt8.bits, 8)
        XCTAssertEqual(StableAudio3Variant.mediumInt4.bits, 4)
    }

    func testVariantFamily() {
        XCTAssertEqual(StableAudio3Variant.mediumInt8.family, .medium)
        XCTAssertEqual(StableAudio3Variant.smallMusicInt4.family, .smallMusic)
        XCTAssertEqual(StableAudio3Variant.smallSFXInt8.family, .smallSFX)
    }

    func testComponentRouting() {
        XCTAssertEqual(SA3Components.dit(for: .mediumInt8), SA3Components.ditMedium)
        XCTAssertEqual(SA3Components.dit(for: .smallMusicInt8), SA3Components.ditSmallMusic)
        XCTAssertEqual(SA3Components.dit(for: .smallSFXInt8), SA3Components.ditSmallSFX)
        XCTAssertEqual(SA3Components.sameEncoder(for: .mediumInt8), SA3Components.sameLEncoder)
        XCTAssertEqual(SA3Components.sameDecoder(for: .smallMusicInt8), SA3Components.sameSDecoder)
    }

    func testComputeTLat() {
        // 30s at 44100/4096 = ~323 latents
        XCTAssertEqual(StableAudio3MusicGen.computeTLat(seconds: 30.0), 323)
        XCTAssertEqual(StableAudio3MusicGen.computeTLat(seconds: 1.0), 11)
        XCTAssertGreaterThan(StableAudio3MusicGen.computeTLat(seconds: 0.001), 0)
    }
}
