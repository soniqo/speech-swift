import XCTest
@testable import AudioServer

final class RealtimeAPIErrorHandlingTests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19387

    override class func setUp() {
        super.setUp()
        serverTask = Task {
            let server = AudioServer(
                host: "127.0.0.1",
                port: port,
                realtimeState: FailingRealtimeModelLoading())
            try await server.run()
        }
        Thread.sleep(forTimeInterval: 1.5)
    }

    override class func tearDown() {
        serverTask?.cancel()
        Thread.sleep(forTimeInterval: 0.5)
        super.tearDown()
    }

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

    func testModelFailureReturnsErrorAndKeepsConnectionOpen() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws) // session.created

        try await sendJSON(ws, [
            "type": "session.update",
            "session": ["model": "personaplex"]
        ] as [String: Any])
        let updated = try await receiveJSON(ws)
        XCTAssertEqual(updated["type"] as? String, "session.updated")
        XCTAssertEqual((updated["session"] as? [String: Any])?["s2s_engine"] as? String, "personaplex")

        let dummyAudio = Data(repeating: 0, count: 4800)
        try await sendJSON(ws, [
            "type": "input_audio_buffer.append",
            "audio": dummyAudio.base64EncodedString()
        ])
        try await sendJSON(ws, ["type": "input_audio_buffer.commit"])
        let committed = try await receiveJSON(ws)
        XCTAssertEqual(committed["type"] as? String, "input_audio_buffer.committed")

        try await sendJSON(ws, [
            "type": "response.create",
            "response": ["modalities": ["audio"]]
        ] as [String: Any])

        let created = try await receiveJSON(ws)
        XCTAssertEqual(created["type"] as? String, "response.created")

        let failure = try await receiveJSON(ws)
        XCTAssertEqual(failure["type"] as? String, "error")
        let error = failure["error"] as? [String: Any]
        XCTAssertEqual(error?["type"] as? String, "server_error")
        XCTAssertEqual(error?["event_type"] as? String, "response.create")
        XCTAssertTrue((error?["message"] as? String)?.contains("forced realtime model failure") ?? false)

        try await sendJSON(ws, ["type": "input_audio_buffer.clear"])
        let cleared = try await receiveJSON(ws)
        XCTAssertEqual(cleared["type"] as? String, "input_audio_buffer.cleared")
    }
}

final class RealtimeAPIKeepaliveTests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19388

    override class func setUp() {
        super.setUp()
        serverTask = Task {
            let server = AudioServer(
                host: "127.0.0.1",
                port: port,
                realtimeState: FailingRealtimeModelLoading(
                    beforeFailure: { Thread.sleep(forTimeInterval: 61) }))
            try await server.run()
        }
        Thread.sleep(forTimeInterval: 1.5)
    }

    override class func tearDown() {
        serverTask?.cancel()
        Thread.sleep(forTimeInterval: 0.5)
        super.tearDown()
    }

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

    func testBlockingModelLoadEmitsKeepaliveBeforeFailure() async throws {
        let ws = try await connect()
        defer { ws.cancel(with: .normalClosure, reason: nil) }

        _ = try await receiveJSON(ws)
        try await sendJSON(ws, [
            "type": "response.create",
            "response": ["instructions": "Hello"]
        ] as [String: Any])

        let created = try await receiveJSON(ws)
        XCTAssertEqual(created["type"] as? String, "response.created")

        var keepaliveCount = 0
        while true {
            let message = try await receiveJSON(ws)
            switch message["type"] as? String {
            case "realtime.keepalive":
                keepaliveCount += 1
            case "error":
                XCTAssertGreaterThanOrEqual(keepaliveCount, 4)
                return
            default:
                XCTFail("Unexpected realtime message: \(message)")
                return
            }
        }
    }
}
