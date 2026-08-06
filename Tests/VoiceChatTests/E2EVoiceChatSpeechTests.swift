import Foundation
import XCTest

@testable import VoiceChat

/// Model-backed speech tests run in their own process under the isolated E2E
/// runner. They never download the 8–22 GB bundle implicitly.
final class E2EVoiceChatSpeechTests: XCTestCase {
    func testCanonicalSilenceDecodesBelowRegressionLimit() throws {
        guard let path = ProcessInfo.processInfo.environment["VOICECHAT_BUNDLE"] else {
            throw XCTSkip("set VOICECHAT_BUNDLE to a full bundle containing tts/")
        }
        let root = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(
            atPath: root.appendingPathComponent("tts/model.safetensors").path)
        else { throw XCTSkip("VOICECHAT_BUNDLE has no exported tts/ component") }

        let codec = try VoiceChatCodec.load(from: root)
        let metrics = codec.verifySilence()
        XCTAssertLessThan(
            metrics.rms, VoiceChatCodec.silenceRMSLimit,
            "canonical silence decoded to RMS \(metrics.rms), peak \(metrics.peak)")
    }
}
