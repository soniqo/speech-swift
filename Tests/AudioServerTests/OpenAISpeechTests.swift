import XCTest
@testable import AudioServer

final class OpenAISpeechTests: XCTestCase {
    private func jsonData(_ value: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: value)
    }

    func testParsesRequiredFieldsAndDefaults() throws {
        let request = try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "input": "Hello world",
            "voice": "alloy",
        ]))

        XCTAssertEqual(request.model, "tts-1")
        XCTAssertEqual(request.input, "Hello world")
        XCTAssertEqual(request.voice, "alloy")
        XCTAssertEqual(request.responseFormat, .wav)
        XCTAssertEqual(request.speed, 1.0)
        XCTAssertEqual(request.language, "english")
    }

    func testParsesPCMAndOptionalFields() throws {
        let request = try OpenAISpeechRequest.parse(jsonData([
            "model": "kokoro",
            "input": "Bonjour",
            "voice": "ff_siwis",
            "response_format": "pcm",
            "speed": 1.25,
            "language": "fr",
        ]))

        XCTAssertEqual(request.responseFormat, .pcm)
        XCTAssertEqual(request.speed, 1.25)
        XCTAssertEqual(request.language, "fr")
    }

    func testRejectsMissingRequiredFields() throws {
        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "input": "Hello",
            "voice": "alloy",
        ]))) { error in
            XCTAssertEqual(error as? OpenAISpeechRequestError, .missingField("model"))
        }

        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "voice": "alloy",
        ]))) { error in
            XCTAssertEqual(error as? OpenAISpeechRequestError, .missingField("input"))
        }

        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "input": "Hello",
        ]))) { error in
            XCTAssertEqual(error as? OpenAISpeechRequestError, .missingField("voice"))
        }
    }

    func testRejectsUnsupportedFormatAndSpeed() throws {
        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
            "response_format": "mp3",
        ]))) { error in
            XCTAssertEqual(
                error as? OpenAISpeechRequestError,
                .unsupportedResponseFormat("mp3"))
        }

        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "input": "Hello",
            "voice": "alloy",
            "speed": 4.1,
        ]))) { error in
            XCTAssertEqual(error as? OpenAISpeechRequestError, .speedOutOfRange(4.1))
        }
    }

    func testRejectsOversizedInput() throws {
        let input = String(repeating: "a", count: OpenAISpeechRequest.maximumInputCharacters + 1)
        XCTAssertThrowsError(try OpenAISpeechRequest.parse(jsonData([
            "model": "tts-1",
            "input": input,
            "voice": "alloy",
        ]))) { error in
            XCTAssertEqual(
                error as? OpenAISpeechRequestError,
                .inputTooLong(OpenAISpeechRequest.maximumInputCharacters))
        }
    }

    func testOpenAIModelAliasesResolveToKokoro() {
        XCTAssertEqual(resolveModelToTTSVariant("tts-1")?.engine, "kokoro")
        XCTAssertEqual(resolveModelToTTSVariant("tts-1-hd")?.engine, "kokoro")
        XCTAssertEqual(resolveModelToTTSVariant("gpt-4o-mini-tts")?.engine, "kokoro")
        XCTAssertEqual(
            resolveModelToTTSVariant("gpt-4o-mini-tts-2025-12-15")?.engine,
            "kokoro")
    }

    func testGenericAndNativeVoiceResolution() {
        XCTAssertEqual(resolvedLocalVoice("alloy", engine: "kokoro"), "af_heart")
        XCTAssertEqual(resolvedLocalVoice("juniper", engine: "kokoro"), "af_heart")
        XCTAssertNil(resolvedLocalVoice("alloy", engine: "qwen3-tts"))
        XCTAssertEqual(resolvedLocalVoice("af_heart", engine: "kokoro"), "af_heart")
        XCTAssertEqual(resolvedLocalVoice("vivian", engine: "qwen3-tts"), "vivian")
        XCTAssertNil(resolvedLocalVoice("af_heart", engine: "cosyvoice"))
    }
}
