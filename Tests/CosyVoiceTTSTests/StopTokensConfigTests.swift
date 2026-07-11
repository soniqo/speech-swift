import XCTest
@testable import CosyVoiceTTS

/// The upstream Qwen2LM (which CosyVoice3LM inherits) recognises three stop
/// tokens, not one. Our previous loader only handled `eosToken`, so the LLM
/// kept generating to fill maxTokens when it wanted to stop via either of
/// the other two — causing per-segment repetitions in long-form synthesis.
/// These tests pin the three-token contract so a future config change doesn't
/// silently regress it.
final class StopTokensConfigTests: XCTestCase {

    func testStopTokensIncludesAllThreeUpstreamStops() {
        let cfg = CosyVoiceLLMConfig()
        let stops = cfg.stopTokens
        XCTAssertEqual(stops.count, 3, "Upstream defines stop_token_ids = speech_token_size + 0/1/2")

        // The exact values: speech_token_size = 6561 ⇒ stops = [6561, 6562, 6563].
        XCTAssertEqual(stops, [6561, 6562, 6563])

        // And they must match the named special-token getters so downstream
        // code that uses eosToken/taskIdToken stays consistent.
        XCTAssertTrue(stops.contains(cfg.sosToken),
                      "sosToken \(cfg.sosToken) should be a stop signal")
        XCTAssertTrue(stops.contains(cfg.eosToken),
                      "eosToken \(cfg.eosToken) should be a stop signal")
        XCTAssertTrue(stops.contains(cfg.taskIdToken),
                      "taskIdToken \(cfg.taskIdToken) should be a stop signal")
    }

    func testStopTokensScalesWithSpeechTokenSize() {
        // If someone bumps speech_token_size for a future variant, the three
        // stop tokens should follow the upstream convention `size + 0/1/2`.
        var cfg = CosyVoiceLLMConfig()
        cfg.speechTokenSize = 8192
        XCTAssertEqual(cfg.stopTokens, [8192, 8193, 8194])
    }

    func testTotalSpeechVocabIsLargerThanStops() {
        // The decoder output (totalSpeechVocabSize logits) must include the
        // stop tokens; otherwise sampling can't ever select them and
        // generation never terminates.
        let cfg = CosyVoiceLLMConfig()
        for stop in cfg.stopTokens {
            XCTAssertLessThan(stop, cfg.totalSpeechVocabSize,
                              "stop token \(stop) must be inside the decoder vocab")
        }
    }
}
