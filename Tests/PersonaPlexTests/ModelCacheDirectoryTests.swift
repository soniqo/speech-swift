import XCTest
@testable import PersonaPlex
@testable import AudioCommon

/// Unit-level regression coverage for [#300](https://github.com/soniqo/speech-swift/pull/300).
///
/// PR #300 fixed a hardcoded `"aufklarer/PersonaPlex-7B-MLX-4bit"` modelId
/// at `PersonaPlex.swift:1015` inside `respondRealtime` that silently broke
/// the 8-bit variant — the voice prompt directory was looked up in the
/// wrong cache, the prompt failed to load, and the model generated
/// incoherent non-English audio.
///
/// The follow-up centralized the cache-dir lookup behind
/// `PersonaPlexModel.modelCacheDirectory()`, eliminating the copy-paste
/// pattern that produced the bug. These tests pin that helper so a future
/// regression where someone re-hardcodes the modelId would fail in CI
/// without needing the 8-bit weights or any model download.
final class ModelCacheDirectoryTests: XCTestCase {

    /// `modelCacheDirectory()` must reflect the configured `modelId`. If the
    /// helper ever drifts back to a hardcoded string, this test catches it
    /// because two distinct modelIds would resolve to the same directory.
    func testModelCacheDirectoryReflectsModelId() throws {
        let m4bit = PersonaPlexModel(cfg: .default)
        m4bit.modelId = "aufklarer/PersonaPlex-7B-MLX-4bit"
        let m8bit = PersonaPlexModel(cfg: .default)
        m8bit.modelId = "aufklarer/PersonaPlex-7B-MLX-8bit"

        let dir4 = try m4bit.modelCacheDirectory()
        let dir8 = try m8bit.modelCacheDirectory()

        XCTAssertNotEqual(
            dir4.path, dir8.path,
            "modelCacheDirectory must differ when modelId differs — this is the bug class PR #300 fixed (a hardcoded modelId would silently make these equal)")
    }

    /// `modelCacheDirectory()` must encode the modelId into the resolved
    /// path. Defensive guard: even if the cache implementation changes how
    /// it normalizes slashes, the modelId's repo segment must appear in the
    /// path so users (and `brew info`-style introspection) can see which
    /// model the cache belongs to.
    func testModelCacheDirectoryEncodesRepoIdentifier() throws {
        let model = PersonaPlexModel(cfg: .default)
        let fakeId = "fake-org/fake-personaplex-test-\(Int.random(in: 0..<999_999))"
        model.modelId = fakeId
        let dir = try model.modelCacheDirectory()
        // Cache impls typically replace `/` with `_` or `-` for filesystem
        // safety; accept any path that contains the unique repo segment.
        let segment = fakeId.split(separator: "/").last!
        XCTAssertTrue(
            dir.path.contains(String(segment)),
            "modelCacheDirectory path should contain the modelId's repo segment '\(segment)' — got: \(dir.path)")
    }

    /// Default-initialized models resolve to the default modelId path.
    /// Belt-and-suspenders guard against the modelId default ever drifting.
    func testModelCacheDirectoryOnDefaultMatchesDefaultModelId() throws {
        let model = PersonaPlexModel(cfg: .default)
        XCTAssertEqual(model.modelId, PersonaPlexModel.defaultModelId,
                       "Default-initialized model should carry defaultModelId")
        // The helper must produce *some* valid URL — we don't assert the
        // exact path because that depends on the cache implementation;
        // we just assert it doesn't throw.
        XCTAssertNoThrow(try model.modelCacheDirectory())
    }
}
