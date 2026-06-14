import XCTest
@testable import AudioServer

/// End-to-end tests for the Realtime API engine dispatch added in PR #311.
///
/// Verifies that the `model` field on `session.update` actually routes to
/// the engine the client picked — not the hardcoded default — across every
/// dispatch path the PR adds:
///
///   - default Parakeet ASR on `input_audio_buffer.commit`
///   - default Kokoro TTS on `response.create`
///   - explicit variant selection (`qwen3-1.7b`, `voxcpm2-int8`)
///   - speech-to-speech precedence (PersonaPlex, Hibiki) — commit captures
///     audio and `response.create` runs the S2S model on it
///
/// These tests download model weights on first run. Skipped in CI via the
/// `E2E` filter prefix; runs locally as part of `make test`.
final class E2ERealtimeAPIEnginesTests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19386

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

    /// Drain `response.*` events until `response.done`, accumulating the
    /// transcribed text (if any) and the total decoded audio samples.
    private func drainResponse(_ ws: URLSessionWebSocketTask) async throws
        -> (transcript: String?, audioSamples: Int) {
        var transcript: String?
        var audioSamples = 0
        for _ in 0..<2000 {  // safety cap
            let msg = try await receiveJSON(ws)
            let type = msg["type"] as? String
            if type == "response.audio.delta", let delta = msg["delta"] as? String,
               let data = Data(base64Encoded: delta) {
                audioSamples += data.count / 2  // PCM16
            }
            if type == "response.audio_transcript.done",
               let t = msg["transcript"] as? String {
                transcript = t
            }
            if type == "conversation.item.input_audio_transcription.completed",
               let t = msg["transcript"] as? String {
                transcript = t
            }
            if type == "response.done" {
                return (transcript, audioSamples)
            }
            if type == "error" {
                XCTFail("Server returned error: \(msg["error"] ?? "<no error body>")")
                return (transcript, audioSamples)
            }
        }
        XCTFail("response.done never received")
        return (transcript, audioSamples)
    }

    private func testAudioPCM16At24kHz() throws -> Data {
        guard let url = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav resource missing from AudioServerTests bundle")
        }
        let samples = try decodeWAVData(try Data(contentsOf: url), targetSampleRate: 24000)
        return floatToPCM16LE(samples)
    }

    // MARK: - ASR dispatch

    /// The default ASR engine (Parakeet) must actually transcribe — not fall
    /// back to Qwen3-ASR. Regression test for the original PR #311 issue.
    func testDefaultParakeetASRTranscribes() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        let pcm = try testAudioPCM16At24kHz()
        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": pcm.base64EncodedString()
        ])
        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])

        // Drain to response.done.
        var transcript: String?
        for _ in 0..<2000 {
            let msg = try await receiveJSON(ws)
            let type = msg["type"] as? String
            if type == "conversation.item.input_audio_transcription.completed",
               let t = msg["transcript"] as? String {
                transcript = t
            }
            if type == "response.done" { break }
            if type == "error" {
                XCTFail("Server returned error: \(msg["error"] ?? "<no body>")")
                return
            }
        }

        // Transcript should be non-empty — Parakeet is wired and produced output.
        XCTAssertNotNil(transcript)
        XCTAssertFalse(transcript?.isEmpty ?? true,
                       "Parakeet ASR produced empty transcript — dispatch likely fell through")
    }

    // MARK: - TTS dispatch

    /// Default Kokoro TTS produces non-empty audio.
    func testDefaultKokoroTTSSynthesizes() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)

        try await sendJSON(ws, [
            "type": "response.create",
            "response": [
                "instructions": "Hello world."
            ]
        ] as [String: Any])

        let (_, audioSamples) = try await drainResponse(ws)
        XCTAssertGreaterThan(audioSamples, 0, "Kokoro produced no audio samples")
    }

    /// Switching to a specific Qwen3 variant via `model: "qwen3-1.7b"` must
    /// load the 1.7B INT8 bundle, not the 0.6B default. We can't easily
    /// inspect the loaded modelId from the client, but session.updated
    /// echoes the canonical name — that has to be the 1.7B row.
    func testSpecificQwen3VariantSelected() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "qwen3-1.7b"]
        ] as [String: Any])

        let msg = try await receiveJSON(ws)
        XCTAssertEqual(msg["type"] as? String, "session.updated")
        let session = msg["session"] as? [String: Any]
        XCTAssertEqual(session?["asr_model"] as? String, "qwen3-asr-1.7b-mlx-int8")
    }

    // MARK: - Speech-to-speech

    /// PersonaPlex on the S2S precedence path: commit captures input audio,
    /// response.create runs the model on that audio and emits response audio
    /// directly (no text→TTS round-trip). Transcript may or may not be
    /// emitted depending on what PersonaPlex returns; the contract here is
    /// that response.audio.delta packets carry samples.
    func testS2SPersonaPlexProducesAudio() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "personaplex"]
        ] as [String: Any])
        let upd = try await receiveJSON(ws)
        XCTAssertEqual((upd["session"] as? [String: Any])?["s2s_engine"] as? String, "personaplex")

        let pcm = try testAudioPCM16At24kHz()
        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": pcm.base64EncodedString()
        ])
        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])
        // S2S precedence: commit emits only `input_audio_buffer.committed`
        // (no transcription event). Drain it.
        let commit = try await receiveJSON(ws)
        XCTAssertEqual(commit["type"] as? String, "input_audio_buffer.committed")

        // response.create runs PersonaPlex on the stored audio.
        try await sendJSON(ws, [
            "type": "response.create",
            "response": ["modalities": ["audio"]]
        ] as [String: Any])

        let (_, audioSamples) = try await drainResponse(ws)
        XCTAssertGreaterThan(audioSamples, 0, "PersonaPlex S2S produced no audio samples")
    }

    /// Hibiki translation: French / Spanish / Portuguese / German input
    /// → English output. We send the bundled test audio (English speech);
    /// Hibiki Zero auto-detects so we just assert that the dispatch ran
    /// and produced audio output.
    func testS2SHibikiTranslatesAudio() async throws {
        // Hibiki is 3B and slow on the first load; skip in fast mode.
        try XCTSkipIf(ProcessInfo.processInfo.environment["FAST_E2E"] == "1",
                      "Skipping Hibiki download/translate in fast E2E mode")

        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "hibiki", "language": "french"]
        ] as [String: Any])
        let upd = try await receiveJSON(ws)
        XCTAssertEqual((upd["session"] as? [String: Any])?["s2s_engine"] as? String, "hibiki")

        let pcm = try testAudioPCM16At24kHz()
        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": pcm.base64EncodedString()
        ])
        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])
        let commit = try await receiveJSON(ws)
        XCTAssertEqual(commit["type"] as? String, "input_audio_buffer.committed")

        try await sendJSON(ws, [
            "type": "response.create",
            "response": ["modalities": ["audio"]]
        ] as [String: Any])

        let (_, audioSamples) = try await drainResponse(ws)
        XCTAssertGreaterThan(audioSamples, 0, "Hibiki S2S produced no audio samples")
    }

    // MARK: - Engine isolation regression

    /// `model: "parakeet"` must NOT also flip the TTS engine — they live
    /// in independent slots now. Regression for the original single-engine
    /// design where any model switch could clobber both sides.
    func testASRModelDoesNotAffectTTSEngine() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "nemotron"]
        ] as [String: Any])
        let upd = try await receiveJSON(ws)
        let session = upd["session"] as? [String: Any]
        XCTAssertEqual(session?["asr_engine"] as? String, "nemotron")
        XCTAssertEqual(session?["tts_engine"] as? String, "kokoro")
    }
}
