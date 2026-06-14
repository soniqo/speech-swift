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
        // Defaults: ASR=parakeet (default variant), TTS=kokoro (default variant).
        // `model` echoes the canonical TTS variant name — that's what the
        // user hears, matching OpenAI's convention.
        XCTAssertEqual(session?["model"] as? String, "kokoro-82m-coreml")
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
        XCTAssertEqual(session?["asr_model"] as? String, "parakeet-tdt-v3-coreml-int8-30s")
        XCTAssertEqual(session?["tts_model"] as? String, "kokoro-82m-coreml")
        XCTAssertNil(session?["s2s_engine"] as? String)
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
        // Legacy `engine` field round-trips the raw user string AND updates
        // the TTS slot. ASR slot stays at the default.
        XCTAssertEqual(session?["engine"] as? String, "qwen3")
        XCTAssertEqual(session?["tts_engine"] as? String, "qwen3-tts")
        XCTAssertEqual(session?["tts_model"] as? String, "qwen3-tts-0.6b-mlx-int4")
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
        XCTAssertEqual(session?["model"] as? String, "kokoro-82m-coreml")
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
        XCTAssertEqual(session?["asr_model"] as? String, "parakeet-tdt-v3-coreml-int8-30s")
        // TTS untouched — `parakeet` is an ASR-only alias.
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
        XCTAssertEqual(session?["tts_model"] as? String, "voxcpm2-mlx-bf16")
        // ASR untouched — `voxcpm2` is a TTS-only alias.
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
    }

    func testSessionUpdateWithSpecificVariantSelectsExactBundle() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        // Issue #274: clients want to pick a specific variant, not just an
        // engine family. `voxcpm2-int8` should land on the int8 bundle, not
        // the bf16 default.
        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "voxcpm2-int8"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["tts_model"] as? String, "voxcpm2-mlx-int8")
    }

    func testSessionUpdateWithQwen31_7BSelectsLargeVariant() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "qwen3-1.7b"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["asr_model"] as? String, "qwen3-asr-1.7b-mlx-int8")
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
        XCTAssertEqual(session?["asr_engine"] as? String, "qwen3-asr")
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
        XCTAssertEqual(session?["model"] as? String, "kokoro-82m-coreml")
    }

    func testBareQwen3PairsBothASRAndTTS() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "qwen3"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        // Bare "qwen3" flips both slots — that's the qwen3-speech pair.
        XCTAssertEqual(session?["asr_engine"] as? String, "qwen3-asr")
        XCTAssertEqual(session?["tts_engine"] as? String, "qwen3-tts")
    }

    func testQwen3SpeechRoutesToTTSOnly() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "qwen3-speech"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["tts_engine"] as? String, "qwen3-tts")
        // ASR untouched: qwen3-speech is a TTS-only alias.
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
    }

    func testS2SModelTurnsOnS2SAndPicksingASROrTTSTurnsItOff() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        // Engage S2S.
        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "personaplex"]
        ] as [String: Any])
        let msgS2S = try await receiveJSON(ws)
        var session = msgS2S["session"] as? [String: Any]
        XCTAssertEqual(session?["s2s_engine"] as? String, "personaplex")
        XCTAssertEqual(session?["s2s_model"] as? String, "personaplex-7b-mlx-4bit")

        // Pick an ASR-only model — S2S must turn off so the compose path
        // comes back.
        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "parakeet"]
        ] as [String: Any])
        let msgASR = try await receiveJSON(ws)
        session = msgASR["session"] as? [String: Any]
        XCTAssertNil(session?["s2s_engine"] as? String)
        XCTAssertEqual(session?["asr_engine"] as? String, "parakeet")
    }

    func testHibikiVariantSelectsCorrectModelId() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "hibiki-8bit"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["s2s_engine"] as? String, "hibiki")
        XCTAssertEqual(session?["s2s_model"] as? String, "hibiki-zero-3b-mlx-8bit")
    }

    // MARK: - HTTP surface

    /// `GET /v1/realtime/models` dumps the registry. Clients use this to
    /// discover what names `session.update` accepts.
    func testRealtimeModelsEndpoint() async throws {
        let url = URL(string: "http://127.0.0.1:\(Self.port)/v1/realtime/models")!
        let (data, response) = try await URLSession.shared.data(from: url)
        let http = response as! HTTPURLResponse
        XCTAssertEqual(http.statusCode, 200)

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["object"] as? String, "list")
        let models = json?["data"] as? [[String: Any]]
        XCTAssertNotNil(models)
        // At least every must-be-reachable engine has at least one row.
        let engines = Set((models ?? []).compactMap { $0["engine"] as? String })
        XCTAssertTrue(engines.contains("parakeet"))
        XCTAssertTrue(engines.contains("kokoro"))
        XCTAssertTrue(engines.contains("voxcpm2"))
        XCTAssertTrue(engines.contains("hibiki"))
        XCTAssertTrue(engines.contains("personaplex"))
        XCTAssertTrue(engines.contains("vibevoice"))
        // Every row has the contract fields set.
        for row in models ?? [] {
            XCTAssertEqual(row["object"] as? String, "model")
            XCTAssertNotNil(row["id"])
            XCTAssertNotNil(row["model_id"])
            XCTAssertNotNil(row["kind"])
        }
    }

    /// HTTP `/transcribe` returns a 400 for unknown model names rather than
    /// silently falling back to the default — clients should learn fast.
    func testTranscribeRejectsUnknownModel() async throws {
        let url = URL(string: "http://127.0.0.1:\(Self.port)/transcribe")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // No audio_base64 because the request should fail before audio load.
        let body: [String: Any] = ["model": "definitely-not-a-real-model-9000"]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        let http = response as! HTTPURLResponse
        XCTAssertEqual(http.statusCode, 400)
    }

    func testSpeakRejectsUnknownModel() async throws {
        let url = URL(string: "http://127.0.0.1:\(Self.port)/speak")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "text": "hello",
            "model": "totally-not-a-tts-engine"
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        let http = response as! HTTPURLResponse
        XCTAssertEqual(http.statusCode, 400)
    }

    func testRespondRejectsUnknownModel() async throws {
        let url = URL(string: "http://127.0.0.1:\(Self.port)/respond")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        // Even with an audio payload the unknown model should error first.
        let body: [String: Any] = [
            "audio_base64": "",
            "model": "not-an-s2s-model"
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (_, response) = try await URLSession.shared.data(for: req)
        let http = response as! HTTPURLResponse
        XCTAssertEqual(http.statusCode, 400)
    }
}

// MARK: - Unit tests for the model→engine resolvers (no server required)

final class ResolveModelToEngineTests: XCTestCase {

    func testASREngineResolver() {
        // Registry engine slots are kind-qualified ("qwen3-asr" vs "qwen3-tts")
        // so the same family name routes unambiguously per side.
        XCTAssertEqual(resolveModelToASREngine("qwen3"), "qwen3-asr")
        XCTAssertEqual(resolveModelToASREngine("qwen3-asr"), "qwen3-asr")
        XCTAssertEqual(resolveModelToASREngine("qwen3-0.6b"), "qwen3-asr")
        XCTAssertEqual(resolveModelToASREngine("qwen3-1.7b"), "qwen3-asr")
        XCTAssertEqual(resolveModelToASREngine("qwen3-asr-coreml"), "qwen3-asr")
        XCTAssertEqual(resolveModelToASREngine("parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToASREngine("parakeet-ios"), "parakeet")
        XCTAssertEqual(resolveModelToASREngine("nemotron"), "nemotron")
        XCTAssertEqual(resolveModelToASREngine("omnilingual"), "omnilingual")
        // TTS-only and unknown names return nil.
        XCTAssertNil(resolveModelToASREngine("kokoro"))
        XCTAssertNil(resolveModelToASREngine("voxcpm2"))
        XCTAssertNil(resolveModelToASREngine("magpie"))
        XCTAssertNil(resolveModelToASREngine("qwen3-speech"))
        XCTAssertNil(resolveModelToASREngine("future-model-v9"))
    }

    func testTTSEngineResolver() {
        XCTAssertEqual(resolveModelToTTSEngine("kokoro"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("kokoro-82m"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("cosyvoice"), "cosyvoice")
        XCTAssertEqual(resolveModelToTTSEngine("cosyvoice-3"), "cosyvoice")
        XCTAssertEqual(resolveModelToTTSEngine("voxcpm2"), "voxcpm2")
        XCTAssertEqual(resolveModelToTTSEngine("voxcpm2-int8"), "voxcpm2")
        XCTAssertEqual(resolveModelToTTSEngine("magpie"), "magpie")
        XCTAssertEqual(resolveModelToTTSEngine("magpie-tts"), "magpie")
        XCTAssertEqual(resolveModelToTTSEngine("qwen3-speech"), "qwen3-tts")
        XCTAssertEqual(resolveModelToTTSEngine("qwen3-tts"), "qwen3-tts")
        // Bare "qwen3" also resolves on the TTS side now — it's the
        // pairing alias so a single update can flip both slots.
        XCTAssertEqual(resolveModelToTTSEngine("qwen3"), "qwen3-tts")
        // ASR-only and unknown names return nil.
        XCTAssertNil(resolveModelToTTSEngine("parakeet"))
        XCTAssertNil(resolveModelToTTSEngine("nemotron"))
        XCTAssertNil(resolveModelToTTSEngine("omnilingual"))
        XCTAssertNil(resolveModelToTTSEngine("future-model-v9"))
    }

    func testS2SEngineResolver() {
        // Speech-to-speech models live on their own kind.
        XCTAssertEqual(resolveModelToS2SVariant("personaplex")?.engine, "personaplex")
        XCTAssertEqual(resolveModelToS2SVariant("personaplex-7b-mlx-4bit")?.engine, "personaplex")
        XCTAssertEqual(resolveModelToS2SVariant("personaplex-8bit")?.engine, "personaplex")
        XCTAssertEqual(resolveModelToS2SVariant("hibiki")?.engine, "hibiki")
        XCTAssertEqual(resolveModelToS2SVariant("hibiki-8bit")?.engine, "hibiki")
        // ASR/TTS names don't resolve as S2S.
        XCTAssertNil(resolveModelToS2SVariant("parakeet"))
        XCTAssertNil(resolveModelToS2SVariant("kokoro"))
    }

    func testLegacyResolverFallsThroughASRThenTTS() {
        XCTAssertEqual(resolveModelToEngine("parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToEngine("nemotron"), "nemotron")
        XCTAssertEqual(resolveModelToEngine("omnilingual"), "omnilingual")
        XCTAssertEqual(resolveModelToEngine("kokoro"), "kokoro")
        XCTAssertEqual(resolveModelToEngine("voxcpm2"), "voxcpm2")
        XCTAssertEqual(resolveModelToEngine("magpie"), "magpie")
        XCTAssertEqual(resolveModelToEngine("cosyvoice"), "cosyvoice")
        // Bare "qwen3" hits both sides; ASR is checked first.
        XCTAssertEqual(resolveModelToEngine("qwen3"), "qwen3-asr")
        // The TTS-side variant routes there directly.
        XCTAssertEqual(resolveModelToEngine("qwen3-speech"), "qwen3-tts")
    }

    func testResolveAllVariantsPairsBothSidesForQwen3() {
        // Bare "qwen3" matches both an ASR variant and a TTS variant via
        // the shared alias. The session.update handler relies on this to
        // pair both slots in one message.
        let hits = resolveAllVariants("qwen3")
        XCTAssertEqual(Set(hits.map { $0.engine }), ["qwen3-asr", "qwen3-tts"])
    }

    func testCanonicalNameLookup() {
        // Exact canonical names short-circuit the alias scan.
        XCTAssertEqual(resolveModelVariant("kokoro-82m-coreml")?.name, "kokoro-82m-coreml")
        XCTAssertEqual(resolveModelVariant("voxcpm2-mlx-int8")?.engine, "voxcpm2")
        XCTAssertEqual(resolveModelVariant("voxcpm2-mlx-int8")?.modelId, "aufklarer/VoxCPM2-MLX-int8")
        XCTAssertEqual(resolveModelVariant("parakeet-tdt-v3-coreml-int8-ios-5s")?.engine, "parakeet")
        // Hibiki int8 vs int4 — distinct variants, both selectable.
        XCTAssertEqual(resolveModelVariant("hibiki-zero-3b-mlx-4bit")?.modelId, "aufklarer/Hibiki-Zero-3B-MLX-4bit")
        XCTAssertEqual(resolveModelVariant("hibiki-zero-3b-mlx-8bit")?.modelId, "aufklarer/Hibiki-Zero-3B-MLX-8bit")
    }

    func testAliasResolvesToVariantWithExplicitModelId() {
        // Aliases must always come back to a fully-specified ModelVariant —
        // the protocol surface should never hand the loader a nil modelId.
        XCTAssertEqual(resolveModelVariant("kokoro")?.modelId, "aufklarer/Kokoro-82M-CoreML")
        XCTAssertEqual(resolveModelVariant("parakeet")?.modelId,
                       "aufklarer/Parakeet-TDT-v3-CoreML-INT8-30s")
        XCTAssertEqual(resolveModelVariant("qwen3-0.6b")?.modelId,
                       "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
        XCTAssertEqual(resolveModelVariant("qwen3-1.7b")?.modelId,
                       "aufklarer/Qwen3-ASR-1.7B-MLX-8bit")
    }

    func testStreamingAndCoreMLVariantsRegistered() {
        // Variants added in the registry-expansion pass.
        XCTAssertEqual(resolveModelVariant("parakeet-streaming")?.engine, "parakeet-streaming")
        XCTAssertEqual(resolveModelVariant("parakeet-eou")?.modelId,
                       "aufklarer/Parakeet-EOU-120M-CoreML-INT8")
        XCTAssertEqual(resolveModelVariant("vibevoice")?.engine, "vibevoice")
        XCTAssertEqual(resolveModelVariant("vibevoice-1.5b")?.engine, "vibevoice-1.5b")
        XCTAssertEqual(resolveModelVariant("qwen3-tts-coreml")?.engine, "qwen3-tts-coreml")
        XCTAssertEqual(resolveModelVariant("magpie-coreml")?.engine, "magpie-coreml")
    }

    func testRegistryHasAtLeastOneVariantPerWiredEngine() {
        // Every engine the WS dispatch can launch must be reachable via at
        // least one registered variant. Catches the case where a dispatch
        // arm gets added but the registry row is forgotten.
        let registeredEngines = Set(MODEL_REGISTRY.map { $0.engine })
        let mustBeReachable: Set<String> = [
            "qwen3-asr", "parakeet", "parakeet-streaming", "nemotron", "omnilingual",
            "kokoro", "qwen3-tts", "qwen3-tts-coreml", "cosyvoice", "voxcpm2",
            "magpie", "magpie-coreml", "vibevoice", "vibevoice-1.5b",
            "personaplex", "hibiki",
        ]
        for engine in mustBeReachable {
            XCTAssertTrue(registeredEngines.contains(engine),
                          "Engine '\(engine)' has a dispatch arm but no registry row")
        }
    }

    func testCaseInsensitive() {
        XCTAssertEqual(resolveModelToASREngine("Parakeet"), "parakeet")
        XCTAssertEqual(resolveModelToTTSEngine("KOKORO"), "kokoro")
        XCTAssertEqual(resolveModelToTTSEngine("VoxCPM2"), "voxcpm2")
        XCTAssertEqual(resolveModelToS2SVariant("Hibiki")?.engine, "hibiki")
    }

    func testUnknownModelReturnsNil() {
        XCTAssertNil(resolveModelToEngine("future-model-v9"))
        XCTAssertNil(resolveModelToEngine("gpt-4o-realtime"))
        XCTAssertNil(resolveModelToEngine(""))
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
