import AudioCommon
@testable import F5TTS
import Foundation
import MLX
import MLXCommon
import XCTest

final class E2EF5TTSFlowParityTests: XCTestCase {
    func testSamplesMelAndOptionalDump() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["F5TTS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set F5TTS_E2E_BUNDLE to run F5-TTS flow parity")
        }
        guard let referencePath = env["F5TTS_REFERENCE_WAV"], !referencePath.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_WAV to run F5-TTS flow parity")
        }
        guard let referenceText = env["F5TTS_REFERENCE_TEXT"], !referenceText.isEmpty else {
            throw XCTSkip("Set F5TTS_REFERENCE_TEXT to run F5-TTS flow parity")
        }

        let target = env["F5TTS_TARGET_TEXT"] ?? "This is a short local voice cloning test for speech swift."
        let seed = UInt64(env["F5TTS_SEED"] ?? "") ?? 0
        let options = try F5TTSSynthesisOptions(seed: seed)
        let bundleURL = URL(fileURLWithPath: bundlePath)
        let info = try F5TTSBundleLoader.load(from: bundleURL)
        let tokenizer = try F5TTSTokenizer(vocabURL: bundleURL.appendingPathComponent(info.config.files.vocab))
        let modelWeights = try CommonWeightLoader.loadSafetensors(
            url: bundleURL.appendingPathComponent(info.config.files.model))
        try F5TTSFlow.validate(modelWeights, config: info.config)
        let flow = F5TTSFlow(weights: modelWeights, config: info.config)
        let referenceSamples = try AudioFileLoader.load(
            url: URL(fileURLWithPath: referencePath),
            targetSampleRate: info.sampleRate)

        let reference: F5TTSPreparedReference
        if let melPath = env["F5TTS_REFERENCE_MEL_F32"], !melPath.isEmpty {
            let data = try Data(contentsOf: URL(fileURLWithPath: melPath))
            let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let frames = values.count / info.config.melSpec.nMelChannels
            XCTAssertEqual(values.count, frames * info.config.melSpec.nMelChannels)
            let mel = MLXArray(values, [1, frames, info.config.melSpec.nMelChannels])
            reference = F5TTSPreparedReference(
                samples: referenceSamples,
                rms: 0,
                mel: mel,
                rawFrameCount: Int(env["F5TTS_REFERENCE_RAW_FRAMES"] ?? "") ?? max(0, frames - 1))
        } else {
            reference = flow.prepareReference(samples: referenceSamples, options: options)
        }
        print("[F5-TTS Flow] reference mel shape \(reference.mel.shape), raw frames \(reference.rawFrameCount)")
        if let rawPath = env["F5TTS_REF_MEL_F32"], !rawPath.isEmpty {
            let values = reference.mel.transposed(0, 2, 1).asType(.float32).asArray(Float.self)
            let data = values.withUnsafeBufferPointer { buffer in
                Data(buffer: buffer)
            }
            try data.write(to: URL(fileURLWithPath: rawPath))
            print("[F5-TTS Flow] saved reference mel \(rawPath)")
        }
        let started = Date()
        let initialNoise: MLXArray?
        if let noisePath = env["F5TTS_INITIAL_X_F32"], !noisePath.isEmpty {
            let data = try Data(contentsOf: URL(fileURLWithPath: noisePath))
            let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            let frames = values.count / info.config.melSpec.nMelChannels
            XCTAssertEqual(values.count, frames * info.config.melSpec.nMelChannels)
            initialNoise = MLXArray(values, [1, frames, info.config.melSpec.nMelChannels])
        } else {
            initialNoise = nil
        }
        let mel = try flow.sampleMel(
            reference: reference,
            referenceText: referenceText,
            targetText: target,
            tokenizer: tokenizer,
            options: options,
            initialNoise: initialNoise,
            stepDumpDirectory: env["F5TTS_STEP_DUMP_DIR"].map { URL(fileURLWithPath: $0) })
            .asType(.float32)
        eval(mel)
        let elapsed = Date().timeIntervalSince(started)
        print("[F5-TTS Flow] generated mel shape \(mel.shape), elapsed \(String(format: "%.2f", elapsed))s")
        let values = mel.asArray(Float.self)
        XCTAssertFalse(values.contains { !$0.isFinite })
        XCTAssertGreaterThan(values.count, 100)

        if let rawPath = env["F5TTS_FLOW_MEL_F32"], !rawPath.isEmpty {
            let data = values.withUnsafeBufferPointer { buffer in
                Data(buffer: buffer)
            }
            try data.write(to: URL(fileURLWithPath: rawPath))
            print("[F5-TTS Flow] saved \(rawPath)")
        }
    }
}
