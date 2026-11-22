import Foundation
import CoreGraphics
import ImageIO
import XCTest

final class E2EInklingCLITests: XCTestCase {
    private func run(prompt: String, image: URL? = nil) throws -> String {
        let env = ProcessInfo.processInfo.environment
        guard let binary = env["SPEECH_CLI"], let model = env["INKLING2_MODEL_DIR"] else {
            throw XCTSkip("Set SPEECH_CLI and INKLING2_MODEL_DIR for the full-weight CLI checks")
        }
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: binary)
        process.arguments = ["inkling", prompt, "--model-dir", model, "--temperature", "0", "--max-tokens", "512"]
        if let image { process.arguments! += ["--image", image.path] }
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let output = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, output)
        XCTAssertFalse(output.contains("<|"), output)
        return output
    }

    func testLocalExportAnswersThroughCLI() throws {
        XCTAssertEqual(try run(prompt: "Reply with only the result of 2 + 2.").trimmingCharacters(in: .whitespacesAndNewlines), "4")
    }

    func testImageFileDecodesAndGeneratesThroughCLI() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at: url) }
        let pixel: [UInt8] = [255, 0, 0]
        let rgb = Array(repeating: pixel, count: 40 * 40).flatMap { $0 }
        let provider = try XCTUnwrap(CGDataProvider(data: Data(rgb) as CFData))
        let image = try XCTUnwrap(CGImage(width: 40, height: 40, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: 120, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: [], provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let output = try run(prompt: "What is the main color of this image? Reply with just one color.", image: url)
        XCTAssertTrue(output.lowercased().contains("red"), output)
    }
}
