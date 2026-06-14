import XCTest
@testable import AudioServer

final class RealtimeAPITests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19384

    override class func setUp() {
        super.setUp()
        serverTask = Task {
            let server = AudioServer(host: "127.0.0.1", port: port)
            try await server.run()
        }
        Thread.sleep(forTimeInterval: 1.5)
    }

    override class func tearDown() {
        serverTask?.cancel()
        Thread.sleep(forTimeInterval: 0.5)
        super.tearDown()
    }

    // MARK: - Helpers

    private func connect() async throws -> URLSessionWebSocketTask {
        let url = URL(string: "ws://127.0.0.1:\(Self.port)/v1/realtime")!
        let ws = URLSession.shared.webSocketTask(with: url)
        ws.resume()
        return ws
    }

    private func receiveJSON(_ ws: URLSessionWebSocketTask) async throws -> [String: Any] {
        let msg = try await ws.receive()
        guard case .string(let text) = msg,
              let data = text.data(using: .utf8),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            XCTFail("Expected JSON text message")
            return [:]
        }
        return json
    }

    private func sendJSON(_ ws: URLSessionWebSocketTask, _ dict: [String: Any]) async throws {
        let data = try JSONSerialization.data(withJSONObject: dict)
        try await ws.send(.string(String(data: data, encoding: .utf8)!))
    }

    // MARK: - Tests

    func testSessionCreated() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.created")
        let session = msg["session"] as? [String: Any]
        XCTAssertNotNil(session?["id"])
        // Defaults: ASR=parakeet, TTS=kokoro. The canonical model name on
        // session.created reflects the TTS-side default since that's what
        // the user hears.
        XCTAssertEqual(session?["model"] as? String, "kokoro")
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
        // Legacy `engine` field aliases tts_engine for backwards compat.
        XCTAssertEqual(session?["engine"] as? String, "kokoro")
        XCTAssertEqual(session?["language"] as? String, "english")
        XCTAssertEqual(session?["input_audio_format"] as? String, "pcm16")
        XCTAssertEqual(session?["output_audio_format"] as? String, "pcm16")
        let modalities = session?["modalities"] as? [String]
        XCTAssertEqual(modalities, ["audio", "text"])
    }

    func testSessionUpdate() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["engine": "qwen3", "language": "chinese"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.updated")
        let session = msg["session"] as? [String: Any]
        // Legacy `engine` writes to the TTS slot; the ASR slot is unaffected.
        XCTAssertEqual(session?["engine"] as? String, "qwen3")
        XCTAssertEqual(session?["tts_engine"] as? String, "qwen3")
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
        XCTAssertEqual(session?["language"] as? String, "chinese")
        XCTAssertEqual(session?["input_audio_format"] as? String, "pcm16")
        XCTAssertEqual(session?["output_audio_format"] as? String, "pcm16")
    }

    func testInputAudioBufferClear() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, ["type": "input_audio_buffer.clear"])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "input_audio_buffer.cleared")
    }

    func testInputAudioBufferCommitEmpty() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "error")
        let error = msg["error"] as? [String: Any]
        XCTAssertEqual(error?["type"] as? String, "invalid_request_error")
        XCTAssertTrue((error?["message"] as? String)?.contains("empty") ?? false)
    }

    func testResponseCreateNoText() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "response.create",
            "response": ["modalities": ["audio"]]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "error")
        let error = msg["error"] as? [String: Any]
        XCTAssertTrue((error?["message"] as? String)?.contains("text") ?? false)
    }

    func testUnknownEventType() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, ["type": "foo.bar.baz"])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "error")
        let error = msg["error"] as? [String: Any]
        XCTAssertTrue((error?["message"] as? String)?.contains("foo.bar.baz") ?? false)
    }

    func testAppendThenClearThenCommitErrors() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        // Append some dummy audio
        let dummyAudio = Data(repeating: 0, count: 200)
        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": dummyAudio.base64EncodedString()
        ])

        // Clear the buffer
        try await sendJSON(ws, ["type": "input_audio_buffer.clear"])
        let cleared = try await receiveJSON(ws)
        XCTAssertEqual(cleared["type"] as? String, "input_audio_buffer.cleared")

        // Commit should fail — buffer was cleared
        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])
        let error = try await receiveJSON(ws)
        XCTAssertEqual(error["type"] as? String, "error")
        XCTAssertTrue((error["error"] as? [String: Any])?["message"] as? String == "Audio buffer is empty")
    }

    func testAppendInvalidBase64() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": "not-valid-base64!!!"
        ])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "error")
    }

    func testMultipleSessionUpdates() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        // First update
        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["engine": "qwen3"]
        ] as [String: Any])
        let msg1 = try await receiveJSON(ws)
        XCTAssertEqual((msg1["session"] as? [String: Any])?["engine"] as? String, "qwen3")
        // language should remain default
        XCTAssertEqual((msg1["session"] as? [String: Any])?["language"] as? String, "english")

        // Second update — only changes language, engine should persist
        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["language": "german"]
        ] as [String: Any])
        let msg2 = try await receiveJSON(ws)
        XCTAssertEqual((msg2["session"] as? [String: Any])?["engine"] as? String, "qwen3")
        XCTAssertEqual((msg2["session"] as? [String: Any])?["language"] as? String, "german")
    }

    func testSessionCreatedEchoesDefaultModel() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.created")
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["model"] as? String, "kokoro")
    }

    func testSessionUpdateWithASRModelUpdatesASRSlot() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "parakeet"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.updated")
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["model"] as? String, "parakeet")
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
        // TTS untouched — model field routed to ASR-only.
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
    }

    func testSessionUpdateWithTTSModelUpdatesTTSSlot() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "voxcpm2"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.updated")
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["model"] as? String, "voxcpm2")
        XCTAssertEqual(session?["tts_engine"] as? String, "voxcpm2")
        // ASR untouched — model field routed to TTS-only.
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
    }

    func testInputAudioTranscriptionModelUpdatesASRSlotOnly() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": [
                "input_audio_transcription": ["model": "qwen3"]
            ]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["asr_engine"] as? String, "qwen3")
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
    }

    func testSessionUpdateUnknownModelEchoedEnginesUnchanged() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "future-model-v9"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.updated")
        let session = msg["session"] as? [String: Any]
        // Unknown model name echoed back, engines unchanged from defaults.
        XCTAssertEqual(session?["model"] as? String, "future-model-v9")
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
    }

    func testEmptyModelStringIsIgnored() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": ""]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        // Empty string does not overwrite the canonical model.
        XCTAssertEqual(session?["model"] as? String, "kokoro")
    }

    func testQwen3SpeechRoutesToTTS() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "qwen3-speech"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["tts_engine"] as? String, "qwen3")
        // ASR untouched: qwen3-speech is the TTS-side variant.
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
    }
}

// MARK: - Unit tests for the model→engine resolvers (no server required)

final class ResolveModelToEngineTests: XCTestCase {

    func testASREngineResolver() {
        XCTAssertEqual(resolveModelToASREngine("qwen3"), "qwen3")
        XCTAssertEqual(resolveModelToASREngine("qwen3-asr"), "qwen3")
        XCTAssertEqual(resolveModelToASREngine("qwen3-asr-3b"), "qwen3")
        XCTAssertEqual(resolveModelToASREngine("parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToASREngine("parakeet-tdt-0.6b"), "parakeet")
        // TTS-only and unknown names return nil.
        XCTAssertNil(resolveModelToASREngine("kokoro"))
        XCTAssertNil(resolveModelToASREngine("voxcpm2"))
        XCTAssertNil(resolveModelToASREngine("qwen3-speech"))
        XCTAssertNil(resolveModelToASREngine("future-model-v9"))
    }

    func testTTSEngineResolver() {
        XCTAssertEqual(resolveModelToTTSEngine("kokoro"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("kokoro-82m"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("cosyvoice"), "cosyvoice")
        XCTAssertEqual(resolveModelToTTSEngine("cosyvoice-v2"), "cosyvoice")
        XCTAssertEqual(resolveModelToTTSEngine("voxcpm2"), "voxcpm2")
        XCTAssertEqual(resolveModelToTTSEngine("qwen3-speech"), "qwen3")
        XCTAssertEqual(resolveModelToTTSEngine("qwen3-tts"), "qwen3")
        XCTAssertEqual(resolveModelToTTSEngine("qwen3-speech-0.6b"), "qwen3")
        // ASR-only and unknown names return nil.
        XCTAssertNil(resolveModelToTTSEngine("parakeet"))
        XCTAssertNil(resolveModelToTTSEngine("qwen3"))
        XCTAssertNil(resolveModelToTTSEngine("future-model-v9"))
    }

    func testLegacyResolverFallsThroughASRThenTTS() {
        // Names that exist on only one side route there.
        XCTAssertEqual(resolveModelToEngine("parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToEngine("kokoro"), "kokoro")
        XCTAssertEqual(resolveModelToEngine("voxcpm2"), "voxcpm2")
        XCTAssertEqual(resolveModelToEngine("cosyvoice"), "cosyvoice")
        // Ambiguous bare "qwen3" resolves to ASR first (the side the resolver
        // checks before TTS).
        XCTAssertEqual(resolveModelToEngine("qwen3"), "qwen3")
        XCTAssertEqual(resolveModelToEngine("qwen3-speech"), "qwen3")
    }

    func testCaseInsensitive() {
        XCTAssertEqual(resolveModelToASREngine("Parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToTTSEngine("KOKORO"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("VoxCPM2"), "voxcpm2")
    }

    func testUnknownModelReturnsNil() {
        XCTAssertNil(resolveModelToEngine("future-model-v9"))
        XCTAssertNil(resolveModelToEngine("gpt-4o-realtime"))
        XCTAssertNil(resolveModelToEngine(""))
        // Engines that were advertised by the previous resolver but aren't
        // wired into dispatch yet — keep them out of the resolver so we don't
        // hand clients an engine name the server can't actually launch.
        XCTAssertNil(resolveModelToEngine("nemotron"))
        XCTAssertNil(resolveModelToEngine("omnilingual"))
        XCTAssertNil(resolveModelToEngine("magpie"))
    }

    func testKokoroLanguageCodeMapping() {
        XCTAssertEqual(mapToKokoroLanguageCode("english"), "en")
        XCTAssertEqual(mapToKokoroLanguageCode("French"), "fr")
        XCTAssertEqual(mapToKokoroLanguageCode("zh"), "zh")
        XCTAssertEqual(mapToKokoroLanguageCode("chinese"), "zh")
        // Unknown languages fall back to English.
        XCTAssertEqual(mapToKokoroLanguageCode("klingon"), "en")
        XCTAssertEqual(mapToKokoroLanguageCode(""), "en")
    }
}
