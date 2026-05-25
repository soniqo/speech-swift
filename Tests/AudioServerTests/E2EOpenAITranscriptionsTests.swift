import XCTest
@testable import AudioServer

/// End-to-end tests for POST /v1/audio/transcriptions.
///
/// Requires Qwen3-ASR weights to be downloadable from HuggingFace. Skipped in
/// CI via the `--skip E2E` test filter; runs locally as part of `make test`.
final class E2EOpenAITranscriptionsTests: XCTestCase {
    static var serverTask: Task<Void, Error>?
    static let port = 19385

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

    private func testAudioData() throws -> Data {
        guard let url = Bundle.module.url(forResource: "test_audio", withExtension: "wav") else {
            throw XCTSkip("test_audio.wav resource missing from AudioServerTests bundle")
        }
        return try Data(contentsOf: url)
    }

    private func multipartBody(
        boundary: String,
        file: Data,
        filename: String = "audio.wav",
        fields: [String: String]
    ) -> Data {
        var body = Data()
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data(
            "Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(file)
        body.append(Data("\r\n".utf8))
        for (key, value) in fields {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(key)\"\r\n\r\n".utf8))
            body.append(Data(value.utf8))
            body.append(Data("\r\n".utf8))
        }
        body.append(Data("--\(boundary)--\r\n".utf8))
        return body
    }

    private func post(
        path: String,
        body: Data,
        contentType: String
    ) async throws -> (Int, Data, [String: String]) {
        let url = URL(string: "http://127.0.0.1:\(Self.port)\(path)")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(contentType, forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        req.timeoutInterval = 120
        let (data, response) = try await URLSession.shared.data(for: req)
        let http = response as! HTTPURLResponse
        var headers: [String: String] = [:]
        for (k, v) in http.allHeaderFields {
            if let key = k as? String, let value = v as? String {
                headers[key.lowercased()] = value
            }
        }
        return (http.statusCode, data, headers)
    }

    // MARK: - Tests

    func testMultipartJSON() async throws {
        let audio = try testAudioData()
        let boundary = "----TestBoundary\(UUID().uuidString)"
        let body = multipartBody(
            boundary: boundary,
            file: audio,
            fields: ["model": "whisper-1"])
        let (status, data, _) = try await post(
            path: "/v1/audio/transcriptions",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)")
        XCTAssertEqual(status, 200, "Body: \(String(data: data, encoding: .utf8) ?? "<binary>")")

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json?["text"] as? String)
        XCTAssertFalse((json?["text"] as? String ?? "").isEmpty,
            "Transcription text should not be empty")
        XCTAssertNotNil(json?["duration"] as? Double)
    }

    func testMultipartVerboseJSON() async throws {
        let audio = try testAudioData()
        let boundary = "----TestBoundary\(UUID().uuidString)"
        let body = multipartBody(
            boundary: boundary,
            file: audio,
            fields: ["model": "whisper-1", "response_format": "verbose_json"])
        let (status, data, _) = try await post(
            path: "/v1/audio/transcriptions",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)")
        XCTAssertEqual(status, 200, "Body: \(String(data: data, encoding: .utf8) ?? "<binary>")")

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["task"] as? String, "transcribe")
        XCTAssertNotNil(json?["text"] as? String)
        XCTAssertNotNil(json?["duration"] as? Double)
        XCTAssertNotNil(json?["language"] as? String)
        let segments = json?["segments"] as? [[String: Any]]
        XCTAssertNotNil(segments)
        XCTAssertGreaterThanOrEqual(segments?.count ?? 0, 1)
    }

    func testMultipartTextResponse() async throws {
        let audio = try testAudioData()
        let boundary = "----TestBoundary\(UUID().uuidString)"
        let body = multipartBody(
            boundary: boundary,
            file: audio,
            fields: ["model": "whisper-1", "response_format": "text"])
        let (status, data, headers) = try await post(
            path: "/v1/audio/transcriptions",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)")
        XCTAssertEqual(status, 200)
        XCTAssertTrue((headers["content-type"] ?? "").contains("text/plain"))
        let text = String(data: data, encoding: .utf8) ?? ""
        XCTAssertFalse(text.isEmpty, "Plain-text response should contain transcription")
    }

    func testMissingFileReturnsBadRequest() async throws {
        let boundary = "----TestBoundary\(UUID().uuidString)"
        // Send a multipart body that has only a `model` field, no `file`.
        var body = Data()
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"model\"\r\n\r\n".utf8))
        body.append(Data("whisper-1".utf8))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))

        let (status, _, _) = try await post(
            path: "/v1/audio/transcriptions",
            body: body,
            contentType: "multipart/form-data; boundary=\(boundary)")
        XCTAssertEqual(status, 400)
    }
}
