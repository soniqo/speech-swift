@testable import F5TTS
import Foundation
import MLX
import MLXCommon
import XCTest

final class E2EF5TTSTransformerParityTests: XCTestCase {
    func testFixedInputVelocityDump() throws {
        let env = ProcessInfo.processInfo.environment
        guard let bundlePath = env["F5TTS_E2E_BUNDLE"], !bundlePath.isEmpty else {
            throw XCTSkip("Set F5TTS_E2E_BUNDLE to run F5-TTS transformer parity")
        }
        guard let xPath = env["F5TTS_PARITY_X_F32"],
              let condPath = env["F5TTS_PARITY_COND_F32"],
              let tokenPath = env["F5TTS_PARITY_TOKENS_I32"],
              let seqText = env["F5TTS_PARITY_SEQ"],
              let seqLen = Int(seqText),
              !xPath.isEmpty, !condPath.isEmpty, !tokenPath.isEmpty, seqLen > 0 else {
            throw XCTSkip("Set F5TTS_PARITY_X_F32/COND_F32/TOKENS_I32/SEQ to run F5-TTS transformer parity")
        }

        let bundleURL = URL(fileURLWithPath: bundlePath)
        let info = try F5TTSBundleLoader.load(from: bundleURL)
        let modelWeights = try CommonWeightLoader.loadSafetensors(
            url: bundleURL.appendingPathComponent(info.config.files.model))
        try F5TTSFlow.validate(modelWeights, config: info.config)
        let flow = F5TTSFlow(weights: modelWeights, config: info.config)

        let dtype: DType = env["F5TTS_PARITY_DTYPE"] == "fp16" ? .float16 : .float32
        let x = MLXArray(try loadFloat32(path: xPath, expectedCount: seqLen * 100), [1, seqLen, 100])
            .asType(dtype)
        let cond = MLXArray(try loadFloat32(path: condPath, expectedCount: seqLen * 100), [1, seqLen, 100])
            .asType(dtype)
        let tokenIds = try loadInt32(path: tokenPath)
        let time = Float(env["F5TTS_PARITY_TIME"] ?? "") ?? 0
        let cfg = Float(env["F5TTS_PARITY_CFG"] ?? "") ?? 2

        let predicted = flow.predictVelocityForTesting(
            x: x,
            cond: cond,
            tokenIds: tokenIds,
            time: time,
            cfgStrength: cfg)
            .asType(.float32)
        eval(predicted)
        let values = predicted.asArray(Float.self)
        XCTAssertEqual(values.count, seqLen * 100)

        if let outPath = env["F5TTS_PARITY_OUT_F32"], !outPath.isEmpty {
            let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
            try data.write(to: URL(fileURLWithPath: outPath))
            print("[F5-TTS Transformer] saved \(outPath)")
        }

        if let traceDir = env["F5TTS_TRACE_DIR"], !traceDir.isEmpty {
            let directory = URL(fileURLWithPath: traceDir)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            for (name, array) in flow.traceConditionedBranchForTesting(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                time: time) {
                let traceValues = array.asType(.float32).asArray(Float.self)
                let data = traceValues.withUnsafeBufferPointer { Data(buffer: $0) }
                try data.write(to: directory.appendingPathComponent("\(name).f32"))
                print("[F5-TTS Transformer] saved trace \(name) \(array.shape)")
            }
        }

        if let traceDir = env["F5TTS_TRACE_UNCOND_DIR"], !traceDir.isEmpty {
            let directory = URL(fileURLWithPath: traceDir)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            for (name, array) in flow.traceUnconditionedBranchForTesting(
                x: x,
                cond: cond,
                tokenIds: tokenIds,
                time: time) {
                let traceValues = array.asType(.float32).asArray(Float.self)
                let data = traceValues.withUnsafeBufferPointer { Data(buffer: $0) }
                try data.write(to: directory.appendingPathComponent("\(name).f32"))
                print("[F5-TTS Transformer] saved uncond trace \(name) \(array.shape)")
            }
        }
    }

    private func loadFloat32(path: String, expectedCount: Int) throws -> [Float] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        XCTAssertEqual(data.count, expectedCount * MemoryLayout<Float>.stride)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func loadInt32(path: String) throws -> [Int32] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        XCTAssertEqual(data.count % MemoryLayout<Int32>.stride, 0)
        return data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
    }
}
