import Qwen3TTS
import XCTest

final class PublicGenerationCancellationAPITests: XCTestCase {
    func testExactCancellableSynthesisIsPublic() {
        let publicMethod = Qwen3TTSModel.synthesizeCancellable
        XCTAssertFalse(
            String(reflecting: type(of: publicMethod)).isEmpty)
    }
}
