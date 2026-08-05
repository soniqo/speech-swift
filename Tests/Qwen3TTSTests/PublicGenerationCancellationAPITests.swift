import Qwen3TTS
import XCTest

final class PublicGenerationCancellationAPITests: XCTestCase {
    func testExactCancellableSynthesisIsPublic() {
        let publicMethod = Qwen3TTSModel.synthesizeCancellable
        let callerIsolatedUse = Self.synthesizeOnCallerExecutor
        XCTAssertFalse(
            String(reflecting: type(of: publicMethod)).isEmpty)
        XCTAssertFalse(
            String(reflecting: type(of: callerIsolatedUse)).isEmpty)
    }

    private static func synthesizeOnCallerExecutor(
        model: Qwen3TTSModel,
        isolation: isolated (any Actor)
    ) async throws -> [Float] {
        try await model.synthesizeCancellable(
            text: "Public API compile contract",
            isolation: isolation)
    }
}
