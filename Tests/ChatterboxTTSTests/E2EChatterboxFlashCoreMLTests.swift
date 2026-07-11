#if canImport(CoreML)
import CoreML
import Foundation
import XCTest

@testable import ChatterboxTTS

final class E2EChatterboxFlashCoreMLTests: XCTestCase {
    func testBundleLoadsAllGraphs() throws {
        let model = try ChatterboxFlashCoreMLModel(directory: try localBundleURL(), computeUnits: .cpuOnly)

        XCTAssertEqual(model.sampleRate, 24_000)
        XCTAssertEqual(model.t3.config.speechLen, 1024)
        XCTAssertEqual(model.t3.config.speechVocabSize, 8194)
        XCTAssertGreaterThanOrEqual(model.audio.config.tokenLen, 192)
        XCTAssertEqual(model.audio.config.melLen, model.audio.config.tokenLen * model.audio.config.tokenMelRatio)
    }

    func testExportedTokenizerMatchesReferenceIds() throws {
        let tokenizer = try ChatterboxFlashTokenizer(directory: try localBundleURL())

        XCTAssertEqual(
            tokenizer.encode("Core ML speech test."),
            [255, 279, 28, 46, 2, 289, 288, 2, 32, 124, 18, 71, 2, 33, 218, 9, 0]
        )
    }

    func testT3SpeechTokenSmoke() throws {
        let bundle = try localBundleURL()
        let model = try ChatterboxFlashCoreMLModel(directory: bundle, computeUnits: .all)
        let conditioning = ChatterboxFlashT3Conditioning(
            speakerEmbedding: deterministicValues(count: 256, scale: 0.01),
            promptSpeechTokens: Array(repeating: Int32(0), count: model.t3.config.promptSpeechLen),
            emotionAdv: 0.5
        )
        let tokens = try model.generateSpeechTokens(
            text: "Core ML speech test.",
            conditioning: conditioning,
            options: ChatterboxFlashGenerationOptions(
                maxSpeechTokens: 16,
                numSteps: 2,
                temperature: 0,
                positionTemperature: 0,
                seed: 1234
            )
        )

        XCTAssertLessThanOrEqual(tokens.count, 16)
        XCTAssertTrue(tokens.allSatisfy { $0 >= 0 && $0 < model.t3.config.startSpeechToken })
    }

    func testT3SpeechTokenSmokeWithCFGWhenNullPrefillExists() throws {
        let bundle = try localBundleURL()
        guard FileManager.default.fileExists(
            atPath: bundle.appendingPathComponent("t3/NullTextPrefill.mlmodelc").path)
        else {
            throw XCTSkip("Core ML bundle does not include t3/NullTextPrefill.mlmodelc")
        }

        let model = try ChatterboxFlashCoreMLModel(directory: bundle, computeUnits: .all)
        let conditioning = ChatterboxFlashT3Conditioning(
            speakerEmbedding: deterministicValues(count: 256, scale: 0.01),
            promptSpeechTokens: Array(repeating: Int32(0), count: model.t3.config.promptSpeechLen),
            emotionAdv: 0.5
        )
        let tokens = try model.generateSpeechTokens(
            text: "Core ML speech test.",
            conditioning: conditioning,
            options: ChatterboxFlashGenerationOptions(
                maxSpeechTokens: 16,
                numSteps: 2,
                temperature: 0,
                cfgScale: 1.0,
                positionTemperature: 0,
                seed: 1234
            )
        )

        XCTAssertLessThanOrEqual(tokens.count, 16)
        XCTAssertTrue(tokens.allSatisfy { $0 >= 0 && $0 < model.t3.config.startSpeechToken })
    }

    func testAudioBackHalfSmoke() throws {
        let bundle = try localBundleURL()
        let audio = try ChatterboxFlashAudioGraphs(
            directory: bundle,
            config: ChatterboxFlashAudioConfig.load(from: bundle),
            computeUnits: .cpuOnly
        )

        let reference = ChatterboxFlashS3GenReference(
            embedding: deterministicValues(count: 192, scale: 0.01),
            promptToken: Array(repeating: Int32(1), count: 8),
            promptFeature: Array(repeating: 0, count: 16 * 80),
            promptFeatureFrames: 16
        )

        let waveform = try audio.synthesize(
            speechTokens: Array(repeating: 2, count: 8),
            reference: reference,
            seed: 1234
        )

        XCTAssertEqual(waveform.count, 16 * 480)
        XCTAssertTrue(waveform.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(waveform.map { abs($0) }.max() ?? 0, 0)
    }

    func testReferenceCloneThroughFlashCoreML() throws {
        let fm = FileManager.default
        let mlxBundle = ProcessInfo.processInfo.environment["CHATTERBOX_MLX_PATH"] ?? "/tmp/cbx-fp16"
        let refPath = ProcessInfo.processInfo.environment["CHATTERBOX_REFERENCE_WAV"] ?? "/tmp/clone_reference.wav"
        for path in [mlxBundle + "/model.safetensors", refPath] where !fm.fileExists(atPath: path) {
            throw XCTSkip("missing \(path); set CHATTERBOX_MLX_PATH and CHATTERBOX_REFERENCE_WAV")
        }
        guard let s3Tokenizer = Self.cachedSnapshotFile(
            repo: "models--mlx-community--S3TokenizerV2",
            file: "model.safetensors"
        ) else {
            throw XCTSkip("S3TokenizerV2 weights not cached")
        }

        let conditioningModel = try ChatterboxTTSModel.fromPretrained(
            localDir: URL(fileURLWithPath: mlxBundle, isDirectory: true),
            s3TokenizerWeights: s3Tokenizer
        )
        let (samples, sampleRate) = try Self.loadWav(refPath)
        let conditioning = try conditioningModel.prepareFlashConditioning(
            referenceSamples: samples,
            sampleRate: sampleRate
        )

        let flash = try ChatterboxFlashCoreMLModel(directory: try localBundleURL(), computeUnits: .all)
        let waveform = try flash.generate(
            text: "Hello there, this is a Chatterbox Flash voice clone.",
            conditioning: conditioning,
            options: ChatterboxFlashGenerationOptions(
                maxSpeechTokens: 96,
                numSteps: 4,
                temperature: 0,
                positionTemperature: 0,
                seed: 1234
            )
        )

        XCTAssertFalse(waveform.isEmpty, "synthesized audio is empty")
        XCTAssertTrue(waveform.allSatisfy { $0.isFinite })
        let rms = (waveform.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(waveform.count)).squareRoot()
        XCTAssertGreaterThan(rms, 1e-3, "synthesized audio is silent (rms=\(rms))")
        try Self.writeWav(waveform, sampleRate: flash.sampleRate, to: "/tmp/cbx_flash_clone.wav")
        print("[FlashClone] wrote /tmp/cbx_flash_clone.wav: \(waveform.count) samples, rms=\(rms)")
    }

    private func localBundleURL() throws -> URL {
        if let override = ProcessInfo.processInfo.environment["CHATTERBOX_FLASH_COREML_PATH"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }

        let fallback = URL(fileURLWithPath: "/tmp/chatterbox-flash-coreml-upload", isDirectory: true)
        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("Set CHATTERBOX_FLASH_COREML_PATH to a Chatterbox Flash Core ML export bundle")
        }
        return fallback
    }

    private func deterministicValues(count: Int, scale: Float) -> [Float] {
        (0..<count).map { index in
            sin(Float(index) * 0.17) * scale
        }
    }

    private static func cachedSnapshotFile(repo: String, file: String) -> URL? {
        let base = ("~/.cache/huggingface/hub/\(repo)/snapshots" as NSString).expandingTildeInPath
        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: base) else { return nil }
        for snapshot in snapshots {
            let path = URL(fileURLWithPath: base)
                .appendingPathComponent(snapshot)
                .appendingPathComponent(file)
            if fm.fileExists(atPath: path.path) {
                return path
            }
        }
        return nil
    }

    private static func loadWav(_ path: String) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        return data.withUnsafeBytes { raw -> (samples: [Float], sampleRate: Int) in
            let bytes = raw.bindMemory(to: UInt8.self)
            func u32(_ offset: Int) -> UInt32 {
                UInt32(bytes[offset])
                    | UInt32(bytes[offset + 1]) << 8
                    | UInt32(bytes[offset + 2]) << 16
                    | UInt32(bytes[offset + 3]) << 24
            }
            func u16(_ offset: Int) -> UInt16 {
                UInt16(bytes[offset]) | UInt16(bytes[offset + 1]) << 8
            }

            var position = 12
            var sampleRate = 24_000
            var audioFormat: UInt16 = 1
            var bitsPerSample: UInt16 = 16
            var samples: [Float] = []
            while position + 8 <= bytes.count {
                let id = String(bytes: (0..<4).map { bytes[position + $0] }, encoding: .ascii) ?? ""
                let size = Int(u32(position + 4))
                let body = position + 8
                if id == "fmt " {
                    audioFormat = u16(body)
                    sampleRate = Int(u32(body + 4))
                    bitsPerSample = u16(body + 14)
                } else if id == "data" {
                    let end = min(body + size, bytes.count)
                    if audioFormat == 3 || bitsPerSample == 32 {
                        var offset = body
                        while offset + 4 <= end {
                            samples.append(Float(bitPattern: u32(offset)))
                            offset += 4
                        }
                    } else {
                        var offset = body
                        while offset + 2 <= end {
                            samples.append(Float(Int16(bitPattern: u16(offset))) / 32768.0)
                            offset += 2
                        }
                    }
                }
                position = body + size + (size & 1)
            }
            return (samples, sampleRate)
        }
    }

    private static func writeWav(_ samples: [Float], sampleRate: Int, to path: String) throws {
        var data = Data()
        func u32(_ value: UInt32) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        func u16(_ value: UInt16) {
            var v = value.littleEndian
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }

        let byteCount = samples.count * 2
        data.append(contentsOf: Array("RIFF".utf8))
        u32(UInt32(36 + byteCount))
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        u32(16)
        u16(1)
        u16(1)
        u32(UInt32(sampleRate))
        u32(UInt32(sampleRate * 2))
        u16(2)
        u16(16)
        data.append(contentsOf: Array("data".utf8))
        u32(UInt32(byteCount))
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            u16(UInt16(bitPattern: Int16((clamped * 32767.0).rounded())))
        }
        try data.write(to: URL(fileURLWithPath: path))
    }
}
#endif
