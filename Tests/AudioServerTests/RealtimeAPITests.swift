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
        XCTAssertEqual(session?["model"] as? String, "qwen3-speech")
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
        XCTAssertEqual(session?["engine"] as? String, "qwen3")
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
}
