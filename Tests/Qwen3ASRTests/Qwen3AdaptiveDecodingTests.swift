import XCTest
import Foundation
@testable import Qwen3ASR

/// Unit tests for ``Qwen3DecodingOptions/adaptedFor(audioDurationSeconds:)``,
/// the length-gated auto-escalation helper added by Bug 3a. The helper
/// returns a modified copy IFF all three predicates hold:
///   1. `audioDurationSeconds > longInputThresholdSeconds`
///   2. `noRepeatNgramSize == 0` (caller default)
///   3. `longInputNoRepeatNgramSize > 0` (escalation not disabled)
/// Otherwise it returns `self` unchanged. These tests are pure-Swift —
/// no MLX, no Metal, no model download.
final class Qwen3AdaptiveDecodingTests: XCTestCase {

    // MARK: - Threshold behaviour

    /// Audio shorter than the 15 s threshold with all-default options must
    /// return an unchanged copy (no escalation triggered).
    func testShortAudioWithDefaultsReturnsIdentity() {
        let opts = Qwen3DecodingOptions()
        let result = opts.adaptedFor(audioDurationSeconds: 10.0)

        XCTAssertEqual(result.noRepeatNgramSize, 0,
            "Short audio must not escalate noRepeatNgramSize")
        assertAllFieldsEqual(result, opts)
    }

    /// Audio longer than the 15 s threshold with all-default options must
    /// return a copy whose `noRepeatNgramSize` was bumped to the configured
    /// `longInputNoRepeatNgramSize` (default 3).
    func testLongAudioWithDefaultsEscalates() {
        let opts = Qwen3DecodingOptions()
        let result = opts.adaptedFor(audioDurationSeconds: 20.0)

        XCTAssertEqual(result.noRepeatNgramSize, 3,
            "Long audio with default options should escalate to longInputNoRepeatNgramSize")
        // All other fields preserved.
        XCTAssertEqual(result.maxTokens, opts.maxTokens)
        XCTAssertEqual(result.language, opts.language)
        XCTAssertEqual(result.context, opts.context)
        XCTAssertEqual(result.repetitionPenalty, opts.repetitionPenalty)
        XCTAssertEqual(result.temperature, opts.temperature)
        XCTAssertEqual(result.longInputThresholdSeconds, opts.longInputThresholdSeconds)
        XCTAssertEqual(result.longInputNoRepeatNgramSize, opts.longInputNoRepeatNgramSize)
    }

    /// Audio over the threshold but the caller already tuned
    /// `noRepeatNgramSize` to a non-zero value. The caller's value must
    /// win — escalation is suppressed because the caller is no longer on
    /// the default greedy path.
    func testCallerSetNgramSizeIsHonouredOverThreshold() {
        var opts = Qwen3DecodingOptions()
        opts.noRepeatNgramSize = 5
        let result = opts.adaptedFor(audioDurationSeconds: 20.0)

        XCTAssertEqual(result.noRepeatNgramSize, 5,
            "Caller-set noRepeatNgramSize must not be overwritten by the helper")
        assertAllFieldsEqual(result, opts)
    }

    /// Documenting the known indistinguishable case: the helper cannot tell
    /// whether `noRepeatNgramSize == 0` came from the default or from an
    /// explicit caller assignment of `0`. Either way it escalates when the
    /// other predicates pass.
    ///
    /// If a future caller needs "explicitly disable n-gram masking even on
    /// long audio", they must also set `longInputNoRepeatNgramSize = 0`
    /// (see ``testEscalationDisabledByZeroLongInputNgram``).
    func testCallerExplicitlySetZeroNgramIsIndistinguishableFromDefault() {
        var opts = Qwen3DecodingOptions()
        opts.noRepeatNgramSize = 0  // explicit — but observationally identical to default
        let result = opts.adaptedFor(audioDurationSeconds: 20.0)

        XCTAssertEqual(result.noRepeatNgramSize, 3,
            "Explicit 0 is indistinguishable from default 0; helper escalates either way")
    }

    /// Audio exactly at the threshold (15.0 s) must NOT escalate — the
    /// guard uses strict `>`, not `>=`.
    func testAudioAtExactThresholdDoesNotEscalate() {
        let opts = Qwen3DecodingOptions()
        let result = opts.adaptedFor(audioDurationSeconds: 15.0)

        XCTAssertEqual(result.noRepeatNgramSize, 0,
            "Audio exactly at threshold must not trigger escalation (strict >)")
        assertAllFieldsEqual(result, opts)
    }

    // MARK: - Disable knobs

    /// Setting `longInputThresholdSeconds = .infinity` is the documented way
    /// to globally disable adaptive escalation. No finite audio duration
    /// can exceed it, so the helper always returns `self`.
    func testInfiniteThresholdDisablesEscalation() {
        var opts = Qwen3DecodingOptions()
        opts.longInputThresholdSeconds = .infinity
        let result = opts.adaptedFor(audioDurationSeconds: 3600.0)

        XCTAssertEqual(result.noRepeatNgramSize, 0,
            "Infinite threshold must never escalate")
        XCTAssertEqual(result.longInputThresholdSeconds, .infinity)
        assertAllFieldsEqual(result, opts)
    }

    /// Setting `longInputNoRepeatNgramSize = 0` disables escalation
    /// directly — the third predicate fails, so the helper returns `self`
    /// regardless of audio length.
    func testEscalationDisabledByZeroLongInputNgram() {
        var opts = Qwen3DecodingOptions()
        opts.longInputNoRepeatNgramSize = 0
        let result = opts.adaptedFor(audioDurationSeconds: 60.0)

        XCTAssertEqual(result.noRepeatNgramSize, 0,
            "longInputNoRepeatNgramSize == 0 must disable escalation")
        assertAllFieldsEqual(result, opts)
    }

    // MARK: - Field preservation

    /// When the helper returns `self` (no escalation), ALL fields — not
    /// just `noRepeatNgramSize` — must be preserved bit-for-bit. Use a
    /// fully-customized options struct to detect any accidental mutation.
    func testIdentityReturnPreservesAllFields() {
        let opts = Qwen3DecodingOptions(
            maxTokens: 256,
            language: "zh",
            context: "weather report",
            repetitionPenalty: 1.25,
            noRepeatNgramSize: 4,                  // already non-zero → no escalation
            temperature: 0.6,
            longInputThresholdSeconds: 30.0,
            longInputNoRepeatNgramSize: 5
        )
        let result = opts.adaptedFor(audioDurationSeconds: 600.0)

        XCTAssertEqual(result.maxTokens, 256)
        XCTAssertEqual(result.language, "zh")
        XCTAssertEqual(result.context, "weather report")
        XCTAssertEqual(result.repetitionPenalty, 1.25)
        XCTAssertEqual(result.noRepeatNgramSize, 4,
            "Caller's noRepeatNgramSize must not be overwritten")
        XCTAssertEqual(result.temperature, 0.6)
        XCTAssertEqual(result.longInputThresholdSeconds, 30.0)
        XCTAssertEqual(result.longInputNoRepeatNgramSize, 5)
    }

    // MARK: - Helpers

    /// Assert that two `Qwen3DecodingOptions` values are equal field by
    /// field. The struct does not conform to `Equatable`, so we compare
    /// each public field individually.
    private func assertAllFieldsEqual(
        _ a: Qwen3DecodingOptions,
        _ b: Qwen3DecodingOptions,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(a.maxTokens, b.maxTokens, "maxTokens", file: file, line: line)
        XCTAssertEqual(a.language, b.language, "language", file: file, line: line)
        XCTAssertEqual(a.context, b.context, "context", file: file, line: line)
        XCTAssertEqual(a.repetitionPenalty, b.repetitionPenalty, "repetitionPenalty", file: file, line: line)
        XCTAssertEqual(a.noRepeatNgramSize, b.noRepeatNgramSize, "noRepeatNgramSize", file: file, line: line)
        XCTAssertEqual(a.temperature, b.temperature, "temperature", file: file, line: line)
        XCTAssertEqual(
            a.longInputThresholdSeconds, b.longInputThresholdSeconds,
            "longInputThresholdSeconds", file: file, line: line)
        XCTAssertEqual(
            a.longInputNoRepeatNgramSize, b.longInputNoRepeatNgramSize,
            "longInputNoRepeatNgramSize", file: file, line: line)
    }
}
