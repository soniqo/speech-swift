import XCTest
@testable import AudioServer

/// End-to-end coverage for POST /v1/audio/speech.
///
/// Requires Kokoro weights and CoreML inference. The `E2E` class prefix keeps
/// it out of the CI-safe unit phase; run it with the isolated E2E test runner.
final class E2EOpenAISpeechTests: XCTestCase {
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

    func testSynthesizesOpenAICompatibleWAV() async throws {
        let url = URL(string: "http://127.0.0.1:\(Self.port)/v1/audio/speech")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: [
            "model": "tts-1",
            "input": "Hello from Speech Swift.",
            "voice": "alloy",
            "response_format": "wav",
        ])
        request.timeoutInterval = 180

        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        XCTAssertTrue((http.value(forHTTPHeaderField: "Content-Type") ?? "").contains("audio/wav"))
        XCTAssertGreaterThan(data.count, 44)
        XCTAssertEqual(String(data: data.prefix(4), encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: data.dropFirst(8).prefix(4), encoding: .ascii), "WAVE")
    }
}
