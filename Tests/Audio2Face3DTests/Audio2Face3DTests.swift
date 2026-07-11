import XCTest
@testable import Audio2Face3D

final class Audio2Face3DTests: XCTestCase {
    func testDefaultConfigurationMatchesNvidiaV23Window() {
        let config = Audio2Face3DConfiguration()
        XCTAssertEqual(config.inputSampleRate, 16_000)
        XCTAssertEqual(config.bufferLength, 8_320)
        XCTAssertEqual(config.hopLength, 4_160)
        XCTAssertEqual(config.emotionVectorLength, 26)
        XCTAssertEqual(config.outputCoefficientCount, 301)
        XCTAssertEqual(config.frameSampleCount, 533)
    }

    func testDefaultCoefficientLayoutMatchesNvidiaV23Mark() {
        let layout = Audio2Face3DCoefficientLayout.nvidiaV23Mark
        XCTAssertEqual(layout.skinCount, 272)
        XCTAssertEqual(layout.tongueCount, 10)
        XCTAssertEqual(layout.jawCount, 15)
        XCTAssertEqual(layout.eyeCount, 4)
        XCTAssertEqual(layout.coefficientCount, 301)
        XCTAssertEqual(layout.skinRange, 0 ..< 272)
        XCTAssertEqual(layout.tongueRange, 272 ..< 282)
        XCTAssertEqual(layout.jawRange, 282 ..< 297)
        XCTAssertEqual(layout.eyeRange, 297 ..< 301)
    }

    func testFrameKeepsFullCoefficientVector() {
        let coefficients = (0..<301).map { Float($0) / 300 }
        let frame = Audio2Face3DFrame(timeSeconds: 0.25, coefficients: coefficients)

        XCTAssertEqual(frame.timeSeconds, 0.25)
        XCTAssertEqual(frame.coefficients.count, 301)
        XCTAssertEqual(frame.layout, .nvidiaV23Mark)
        XCTAssertEqual(frame.coefficients[282], coefficients[282])
    }

    func testConfigurationCanReadJamesCoefficientLayout() throws {
        let dir = temporaryDirectory()
        try """
        {
          "params": {
            "implicit_emotion_len": 16,
            "explicit_emotions": [
              "amazement", "anger", "cheekiness", "disgust", "fear",
              "grief", "joy", "outofbreath", "pain", "sadness"
            ],
            "num_shapes_skin": 140,
            "num_shapes_tongue": 10,
            "result_jaw_size": 15,
            "result_eyes_size": 4
          },
          "audio_params": {
            "buffer_len": 8320,
            "buffer_ofs": 4160,
            "samplerate": 16000
          }
        }
        """.write(
            to: dir.appendingPathComponent("network_info.json"),
            atomically: true,
            encoding: .utf8)
        try """
        { "config": { "input_strength": 1.0 } }
        """.write(
            to: dir.appendingPathComponent("model_config.json"),
            atomically: true,
            encoding: .utf8)

        let config = try Audio2Face3DDownloader.configuration(from: dir)

        XCTAssertEqual(config.coefficientLayout.skinCount, 140)
        XCTAssertEqual(config.coefficientLayout.tongueCount, 10)
        XCTAssertEqual(config.coefficientLayout.jawCount, 15)
        XCTAssertEqual(config.coefficientLayout.eyeCount, 4)
        XCTAssertEqual(config.outputCoefficientCount, 169)
    }

    func testJamesMLXRuntimeLoadsWhenBundleIsAvailable() throws {
        let env = ProcessInfo.processInfo.environment["AUDIO2FACE3D_JAMES_FIXTURE_DIR"]
        let dir = env.map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
                .appendingPathComponent("out/audio2face3d-mlx-james", isDirectory: true)
        let weights = dir.appendingPathComponent("audio2face3d.safetensors")
        guard FileManager.default.fileExists(atPath: weights.path) else {
            throw XCTSkip("set AUDIO2FACE3D_JAMES_FIXTURE_DIR to an exported James Audio2Face3D MLX bundle")
        }

        let model = try Audio2Face3DModel.fromLocal(directory: dir)
        let audio = [Float](repeating: 0, count: model.configuration.bufferLength)
        let frames = try model.frames(for: audio, sampleRate: model.configuration.inputSampleRate)

        XCTAssertEqual(model.configuration.coefficientLayout.skinCount, 140)
        XCTAssertEqual(model.configuration.outputCoefficientCount, 169)
        XCTAssertEqual(frames.first?.coefficients.count, 169)
    }

    func testLocalModelRequiresExportedMLXWeights() {
        let dir = temporaryDirectory()

        XCTAssertThrowsError(try Audio2Face3DModel.fromLocal(directory: dir)) { error in
            XCTAssertEqual(
                error as? Audio2Face3DError,
                .missingExportedWeights(dir.appendingPathComponent("audio2face3d.safetensors").path))
        }
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("Audio2Face3DTests-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
