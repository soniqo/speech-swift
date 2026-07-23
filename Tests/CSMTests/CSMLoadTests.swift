import XCTest
import Foundation
import MLX
import PersonaPlex
@testable import CSM

final class CSMLoadTests: XCTestCase {
    /// Loads OUR exported CSM weights into the Swift model and verifies a clean
    /// key bijection (a missing/unused key throws inside CSMWeightLoader.load).
    /// Set CSM_MODEL_DIR to a converted export dir (model.safetensors + config.json).
    func testLoadOurExport() throws {
        guard let dir = ProcessInfo.processInfo.environment["CSM_MODEL_DIR"] else {
            throw XCTSkip("set CSM_MODEL_DIR to a converted CSM export directory")
        }
        let url = URL(fileURLWithPath: dir)
        let cfg = try CSMConfig.load(from: url)
        XCTAssertEqual(cfg.audioNumCodebooks, 32)
        XCTAssertEqual(cfg.backbone.numLayers, 16)
        XCTAssertEqual(cfg.decoder.numLayers, 4)

        let model = CSMModel(cfg)
        try CSMWeightLoader.load(model: model, from: url)   // throws on any key mismatch

        // audio_head must have loaded to its real [31, decoderDim, audioVocab] shape
        XCTAssertEqual(model.audioHead.shape,
                       [cfg.audioNumCodebooks - 1, cfg.decoder.dim, cfg.audioVocabSize])
    }

    /// Runs the generation core (embed → backbone → codebook0 → 32-step decoder)
    /// once on the real weights and checks it emits a well-formed 32-codebook frame.
    func testGenerateFrameRuns() throws {
        guard let dir = ProcessInfo.processInfo.environment["CSM_MODEL_DIR"] else {
            throw XCTSkip("set CSM_MODEL_DIR to a converted CSM export directory")
        }
        let url = URL(fileURLWithPath: dir)
        let cfg = try CSMConfig.load(from: url)
        let model = CSMModel(cfg)
        try CSMWeightLoader.load(model: model, from: url)
        model.backbone.resetCache()

        let c = cfg.audioNumCodebooks
        let tokens = MLXArray.zeros([1, 1, c + 1], type: Int32.self)   // valid ids (0)
        let mask = MLXArray.ones([1, 1, c + 1], type: Float.self)
        let frame = model.generateFrame(
            tokens: tokens, tokensMask: mask, backboneOffset: 0, sampler: argmaxSampler)
        eval(frame)

        XCTAssertEqual(frame.shape, [1, c])                            // 32 codebook tokens
        XCTAssertTrue(all(frame .>= MLXArray(Int32(0))).item())        // valid token ids
        XCTAssertTrue(all(frame .< MLXArray(Int32(cfg.audioVocabSize))).item())
    }

    /// Runs the full autoregressive frame loop on a real prompt (dumped from the
    /// reference tokenizer) and checks it emits a well-formed [1, C, N] token block.
    func testGenerateFrameLoop() throws {
        guard let dir = ProcessInfo.processInfo.environment["CSM_MODEL_DIR"],
              let promptPath = ProcessInfo.processInfo.environment["CSM_PROMPT"] else {
            throw XCTSkip("set CSM_MODEL_DIR and CSM_PROMPT")
        }
        let cfg = try CSMConfig.load(from: URL(fileURLWithPath: dir))
        let model = CSMModel(cfg)
        try CSMWeightLoader.load(model: model, from: URL(fileURLWithPath: dir))

        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: promptPath))
        let tokens = arrays["tokens"]!.asType(.int32)
        let mask = arrays["mask"]!.asType(.float32)

        let frames = model.generate(
            promptTokens: tokens, promptMask: mask, maxFrames: 16, sampler: argmaxSampler)
        eval(frames)

        XCTAssertEqual(frames.shape[0], 1)
        XCTAssertEqual(frames.shape[1], cfg.audioNumCodebooks)
        XCTAssertLessThanOrEqual(frames.shape[2], 16)
        print("CSM Swift generated \(frames.shape[2]) frames of \(cfg.audioNumCodebooks) codebooks")
    }

    /// End-to-end audio: generate frames on a real prompt, then Mimi-decode them
    /// to a 24 kHz waveform and verify it is non-silent speech-range audio.
    func testGenerateAudio() throws {
        guard let dir = ProcessInfo.processInfo.environment["CSM_MODEL_DIR"],
              let promptPath = ProcessInfo.processInfo.environment["CSM_PROMPT"] else {
            throw XCTSkip("set CSM_MODEL_DIR and CSM_PROMPT")
        }
        let url = URL(fileURLWithPath: dir)
        let cfg = try CSMConfig.load(from: url)
        let model = CSMModel(cfg)
        try CSMWeightLoader.load(model: model, from: url)

        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: promptPath))
        let frames = model.generate(
            promptTokens: arrays["tokens"]!.asType(.int32),
            promptMask: arrays["mask"]!.asType(.float32),
            maxFrames: 32, sampler: argmaxSampler)

        let mimi = try CSMMimi.load(from: url.appendingPathComponent("mimi.safetensors"),
                                    numCodebooks: cfg.audioNumCodebooks)
        let audio = MimiStreamingDecoder(mimi).decodeFrames(frames).squeezed()  // [samples]
        eval(audio)

        let n = audio.shape.last ?? 0
        XCTAssertGreaterThan(n, 0)
        let rms = sqrt((audio.asType(.float32) * audio.asType(.float32)).mean()).item(Float.self)
        print("CSM Swift audio: \(n) samples (\(Double(n) / 24000.0)s), RMS \(rms)")
        XCTAssertGreaterThan(rms, 0.005)   // non-silent

        // write a wav for listening
        if let out = ProcessInfo.processInfo.environment["CSM_OUT"] {
            try writeWav(audio.asType(.float32), to: URL(fileURLWithPath: out), sampleRate: 24000)
        }
    }
}

/// Minimal 16-bit PCM WAV writer for test output.
func writeWav(_ samples: MLXArray, to url: URL, sampleRate: Int) throws {
    let floats = samples.asArray(Float.self)
    var data = Data()
    func u32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    func u16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    let n = floats.count
    data.append("RIFF".data(using: .ascii)!); data.append(u32(UInt32(36 + n * 2)))
    data.append("WAVE".data(using: .ascii)!); data.append("fmt ".data(using: .ascii)!)
    data.append(u32(16)); data.append(u16(1)); data.append(u16(1))
    data.append(u32(UInt32(sampleRate))); data.append(u32(UInt32(sampleRate * 2)))
    data.append(u16(2)); data.append(u16(16))
    data.append("data".data(using: .ascii)!); data.append(u32(UInt32(n * 2)))
    for f in floats {
        let s = Int16(max(-1, min(1, f)) * 32767)
        data.append(u16(UInt16(bitPattern: s)))
    }
    try data.write(to: url)
}
