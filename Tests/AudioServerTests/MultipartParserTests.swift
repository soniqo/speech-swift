import XCTest
@testable import AudioServer

final class MultipartParserTests: XCTestCase {

    private func buildMultipart(
        boundary: String,
        parts: [(name: String, filename: String?, contentType: String?, body: Data)]
    ) -> Data {
        var out = Data()
        for p in parts {
            out.append(Data("--\(boundary)\r\n".utf8))
            var disp = "Content-Disposition: form-data; name=\"\(p.name)\""
            if let f = p.filename { disp += "; filename=\"\(f)\"" }
            out.append(Data("\(disp)\r\n".utf8))
            if let ct = p.contentType {
                out.append(Data("Content-Type: \(ct)\r\n".utf8))
            }
            out.append(Data("\r\n".utf8))
            out.append(p.body)
            out.append(Data("\r\n".utf8))
        }
        out.append(Data("--\(boundary)--\r\n".utf8))
        return out
    }

    func testSingleTextField() {
        let boundary = "BOUND-1"
        let data = buildMultipart(
            boundary: boundary,
            parts: [
                (name: "model", filename: nil, contentType: nil, body: Data("whisper-1".utf8))
            ])
        let parts = MultipartParser.parse(data, boundary: boundary)
        XCTAssertEqual(parts.count, 1)
        XCTAssertEqual(parts[0].name, "model")
        XCTAssertEqual(parts[0].stringValue, "whisper-1")
        XCTAssertNil(parts[0].filename)
    }

    func testFileAndFields() {
        let boundary = "----WebKitFormBoundaryABC"
        let fileBytes = Data((0..<256).map { UInt8($0 & 0xFF) })
        let data = buildMultipart(
            boundary: boundary,
            parts: [
                (name: "file", filename: "audio.wav", contentType: "audio/wav", body: fileBytes),
                (name: "model", filename: nil, contentType: nil, body: Data("whisper-1".utf8)),
                (name: "language", filename: nil, contentType: nil, body: Data("english".utf8)),
                (name: "response_format", filename: nil, contentType: nil, body: Data("verbose_json".utf8))
            ])
        let parts = MultipartParser.parse(data, boundary: boundary)
        XCTAssertEqual(parts.count, 4)

        let byName = Dictionary(uniqueKeysWithValues: parts.compactMap { p -> (String, MultipartPart)? in
            p.name.map { ($0, p) }
        })

        XCTAssertEqual(byName["file"]?.filename, "audio.wav")
        XCTAssertEqual(byName["file"]?.body, fileBytes)
        XCTAssertEqual(byName["model"]?.stringValue, "whisper-1")
        XCTAssertEqual(byName["language"]?.stringValue, "english")
        XCTAssertEqual(byName["response_format"]?.stringValue, "verbose_json")
    }

    func testBinaryBodyWithCRLFInside() {
        // Binary payload containing CRLFs and partial boundary lookalikes must
        // round-trip exactly.
        let boundary = "B"
        var fileBytes = Data()
        for _ in 0..<10 {
            fileBytes.append(Data([0x0D, 0x0A, 0x2D, 0x2D, 0x42, 0x58]))  // \r\n--BX (not a boundary)
        }
        fileBytes.append(Data([0xFF, 0xFE, 0x00, 0x01, 0x02]))
        let data = buildMultipart(
            boundary: boundary,
            parts: [(name: "file", filename: "x.bin", contentType: nil, body: fileBytes)])
        let parts = MultipartParser.parse(data, boundary: boundary)
        XCTAssertEqual(parts.count, 1)
        XCTAssertEqual(parts[0].body, fileBytes)
    }

    func testNoBoundaryReturnsEmpty() {
        let data = Data("not multipart at all".utf8)
        let parts = MultipartParser.parse(data, boundary: "WHATEVER")
        XCTAssertEqual(parts.count, 0)
    }
}
