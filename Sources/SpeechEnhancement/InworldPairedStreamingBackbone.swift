import AudioCommon
import CoreML
import Foundation
import Metal

/// Run both overlap lanes over a shared Metal attention cache.
final class InworldPairedStreamingBackbone: InworldStreamingInference {
    private static let laneCount = InworldStreamingLayout.pairedLanes
    private static let bands = InworldVoiceIsolator.bands
    private static let width = InworldVoiceIsolator.hiddenWidth
    private static let historyWidth = 3

    final class Lane {
        var frames = 0
        let slot: Int
        init(slot: Int) { self.slot = slot }
    }

    private struct State {
        let history: MTLBuffer
        let scan: MTLBuffer
    }

    let blockFrames: Int
    private let stageFiles: InworldCoreMLStages
    private let encoder: MLModel
    private let pairedStages: [MLModel]?
    private let singleLaneStages: [MLModel]?
    private let serialOverlap: Bool
    private let bandDecoder: InworldBandDecoder?
    private let scan: InworldMambaScan
    private let scanWorkspace: InworldMambaScan.Workspace
    private let attention: InworldStreamingAttention
    private let caches: [InworldStreamingAttention.Cache]
    private let states: [State]
    private let lookahead: MLMultiArray
    private let partialSpectrum: MLMultiArray
    private let validFrameInput: String
    private let validCounts: [MLMultiArray]

    init(modelDirectory: URL, blockFrames: Int, lookaheadIndex: Int, configuration: MLModelConfiguration,
         useSingleLaneStages: Bool, useMetalDecoder: Bool, keyEncoding: InworldStreamingAttention.CacheEncoding,
         valueEncoding: InworldStreamingAttention.CacheEncoding) throws {
        self.blockFrames = blockFrames
        // The wider graph's working memory is also reused during crossfades.
        let serialOverlap = useMetalDecoder && blockFrames == 8
        let graphLanes = serialOverlap ? 1 : Self.laneCount
        self.serialOverlap = serialOverlap
        self.bandDecoder = useMetalDecoder
            ? try InworldBandDecoder(url: modelDirectory.appendingPathComponent("decoder.bin"),
                                     blockFrames: blockFrames, laneCapacity: graphLanes) : nil
        configuration.functionName = "lanes_2_frames_\(blockFrames)"
        let stageFiles = try InworldCoreMLStages(directory: modelDirectory, count: InworldMambaScan.stageCount + 2)
        self.stageFiles = stageFiles
        let encoder = try CoreMLLoader.load(url: stageFiles.urls[0], configuration: configuration, name: "inworld-streaming-encoder")
        self.encoder = encoder
        func loadStages(lanes: Int) throws -> [MLModel] {
            let config = MLModelConfiguration()
            config.computeUnits = configuration.computeUnits
            config.functionName = "lanes_\(lanes)_frames_\(blockFrames)"
            return try stageFiles.urls.enumerated().map { index, url in
                if index == 0 { return encoder }
                return try CoreMLLoader.load(url: url, configuration: config, name: "inworld-\(lanes)-lane-stage_\(index)")
            }
        }
        self.pairedStages = serialOverlap ? nil : try loadStages(lanes: Self.laneCount)
        self.singleLaneStages = useSingleLaneStages ? try loadStages(lanes: 1) : nil
        let coefficients = try Data(contentsOf: modelDirectory.appendingPathComponent("scan_coefficients.bin"))
        let scan = try InworldMambaScan(coefficients: coefficients)
        self.scan = scan
        self.scanWorkspace = try scan.workspace(tokens: graphLanes * Self.bands * blockFrames)
        let attention = try InworldStreamingAttention(keyEncoding: keyEncoding, valueEncoding: valueEncoding, blockFrames: blockFrames,
                                                      serialOverlap: serialOverlap)
        self.attention = attention
        self.caches = try (0..<4).map { _ in try attention.makeCache() }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw InworldIsolationError.predictionFailed("could not create the streaming state device")
        }
        self.states = try (0..<3).map { _ in
            let bytes = Self.laneCount * Self.bands * InworldMambaScan.channels * Self.historyWidth * MemoryLayout<Float>.stride
            guard let history = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw InworldIsolationError.predictionFailed("could not allocate convolution history")
            }
            history.contents().initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
            return State(history: history, scan: try scan.makeCarry(rows: Self.laneCount * Self.bands))
        }
        self.lookahead = try MLMultiArray(shape: [1], dataType: .int32)
        self.lookahead[0] = NSNumber(value: lookaheadIndex)
        self.partialSpectrum = try MLMultiArray(
            shape: [1, NSNumber(value: blockFrames), NSNumber(value: InworldIsolationDSP.bins), 2], dataType: .float32
        )
        let usesSecondFrameFlag = blockFrames == InworldStreamingLayout.pairedBlockFrames
        self.validFrameInput = usesSecondFrameFlag ? "valid_second" : "valid_frames"
        self.validCounts = try (1...blockFrames).map { count in
            let value = try MLMultiArray(shape: [1], dataType: usesSecondFrameFlag ? .float32 : .int32)
            value[0] = NSNumber(value: usesSecondFrameFlag ? count - 1 : count)
            return value
        }
    }

    func makeLane() -> Lane { Lane(slot: 0) }

    func beginLane(copying active: Lane, reusing spare: Lane?) -> Lane {
        let lane = spare ?? Lane(slot: 1 - active.slot)
        lane.frames = 0
        for state in states {
            for buffer in [state.history, state.scan] {
                let bytes = buffer.length / 2
                (buffer.contents() + lane.slot * bytes)
                    .copyMemory(from: buffer.contents() + active.slot * bytes, byteCount: bytes)
            }
        }
        return lane
    }

    func retireLane(_ lane: Lane) {
        attention.retire(lane: lane.slot, frames: lane.frames)
        lane.frames = 0
    }

    func prepare(_ spectrum: MLMultiArray, lane: Lane) throws -> Lane {
        _ = try predict(spectrum, frames: blockFrames, lanes: [lane])
        retireLane(lane)
        for state in states {
            for buffer in [state.history, state.scan] {
                let bytes = buffer.length / Self.laneCount
                (buffer.contents() + lane.slot * bytes)
                    .initializeMemory(as: UInt8.self, repeating: 0, count: bytes)
            }
        }
        return lane
    }

    func predict(_ spectrum: MLMultiArray, frames validFrames: Int, lanes requested: [Lane]) throws -> [MLMultiArray] {
        let frames = blockFrames
        let inputSpectrum: MLMultiArray
        if validFrames == frames {
            inputSpectrum = spectrum
        } else {
            padSpectrum(spectrum, frames: validFrames)
            inputSpectrum = partialSpectrum
        }
        let shared = try attention.projectedArray(lanes: 1, frames: frames, encoderProjection: true)
        let encoded = try prediction(encoder, ["spectrum": inputSpectrum, "lookahead_index": lookahead],
                                     backings: ["qkv": shared])
        if serialOverlap && requested.count > 1 {
            return try requested.map { lane in
                try autoreleasepool {
                    try predict(encoded, shared: shared, frames: validFrames, lanes: [lane])[0]
                }
            }
        }
        return try predict(encoded, shared: shared, frames: validFrames, lanes: requested)
    }

    private func predict(_ encoded: MLFeatureProvider, shared: MLMultiArray, frames validFrames: Int,
                         lanes requested: [Lane]) throws -> [MLMultiArray] {
        let lanes = requested.sorted { $0.slot < $1.slot }
        let count = singleLaneStages == nil ? Self.laneCount : lanes.count
        guard let models = count == 1 ? singleLaneStages : pairedStages else {
            throw InworldIsolationError.predictionFailed("missing Core ML stages for \(count) streaming lanes")
        }
        let firstLane = count == 1 ? lanes[0].slot : 0
        let frames = blockFrames
        let qkv = try attention.projectedArray(lanes: count, frames: frames)
        var projected = encoded
        let slots = lanes.map(\.slot)
        let positions = lanes.map(\.frames)
        try attention.reserve(lanes: slots, positions: positions, frames: validFrames)
        let featuresShape = [count, Self.bands, frames, Self.width]
        for block in 0..<4 {
            let backing = block == 0 ? shared : qkv
            let hiddenShape = block == 0 ? [1, Self.bands, frames, Self.width] : featuresShape
            let hidden = try InworldVoiceIsolator.array(projected, named: "hidden", shape: hiddenShape)
            let project = try InworldVoiceIsolator.array(projected, named: "qkv", shape: backing.shape.map(\.intValue))
            let attended = try attention.run(project, frames: frames, validFrames: validFrames, slots: slots, positions: positions,
                                             cache: caches[block], backing: backing, modelLanes: count,
                                             encoderProjection: block == 0)
            var inputs = ["hidden": hidden, "attended": attended]
            let stage = block * 2 + 1
            if block == 3 {
                let raw: MLMultiArray
                if let bandDecoder {
                    let backing = try bandDecoder.input(lanes: count)
                    let output = try prediction(models[stage], inputs, backings: ["decoder_features": backing])
                    let features = try InworldVoiceIsolator.array(output, named: "decoder_features",
                        shape: featuresShape)
                    if features !== backing { features.transfer(to: backing) }
                    raw = try bandDecoder.run(lanes: count)
                } else {
                    let output = try prediction(models[stage], inputs)
                    raw = try InworldVoiceIsolator.array(output, named: "mask_residual",
                        shape: [count, frames, InworldIsolationDSP.bins, 4])
                }
                for lane in lanes { lane.frames += validFrames }
                return try requested.map { try copyRawLane(raw, index: $0.slot - firstLane, frames: validFrames) }
            }

            let state = states[block]
            let history = try stateArray(state.history, firstLane: firstLane, lanes: count, width: Self.historyWidth)
            let carry = try stateArray(state.scan, firstLane: firstLane, lanes: count, width: InworldMambaScan.states)
            inputs["convolution_history"] = history
            inputs["carry"] = carry
            inputs[validFrameInput] = validCounts[validFrames - 1]
            let rows = count * frames
            let scanBacking = try scanWorkspace.inputArray(rows: rows, steps: Self.bands, lanes: count)
            let output = try prediction(models[stage], inputs, backings: ["scan_input": scanBacking])
            let packed = try InworldVoiceIsolator.array(output, named: "scan_input", shape: scanBacking.shape.map(\.intValue))
            if packed !== scanBacking { packed.transfer(to: scanBacking) }
            let nextHistory = try InworldVoiceIsolator.array(output, named: "next_history", shape: history.shape.map(\.intValue))
            let nextCarry = try InworldVoiceIsolator.array(output, named: "next_carry", shape: carry.shape.map(\.intValue))
            nextHistory.transfer(to: history)
            nextCarry.transfer(to: carry)
            let scanned = try scan.run(stage: stage, rows: rows, steps: Self.bands, workspace: scanWorkspace, lanes: count)
            let features = try InworldVoiceIsolator.array(output, named: "scan_features",
                shape: [count, frames, Self.bands, Self.width])
            projected = try prediction(models[stage + 1], ["features": features, "scan_output": scanned],
                                       backings: ["qkv": qkv])
        }
        throw InworldIsolationError.predictionFailed("no paired streaming decoder output")
    }

    /// Unused frames are masked in attention and excluded from the saved recurrence state.
    private func padSpectrum(_ spectrum: MLMultiArray, frames: Int) {
        let inputStrides = spectrum.strides.map(\.intValue)
        spectrum.withUnsafeBufferPointer(ofType: Float.self) { input in
            partialSpectrum.withUnsafeMutableBufferPointer(ofType: Float.self) { output, strides in
                output.initialize(repeating: 0)
                for frame in 0..<frames {
                    for bin in 0..<InworldIsolationDSP.bins {
                        for component in 0..<2 {
                            let target = frame * strides[1] + bin * strides[2] + component * strides[3]
                            output[target] = input[frame * inputStrides[1] + bin * inputStrides[2] + component * inputStrides[3]]
                        }
                    }
                }
            }
        }
    }

    private func prediction(
        _ model: MLModel, _ inputs: [String: MLMultiArray], backings: [String: MLMultiArray] = [:]
    ) throws -> MLFeatureProvider {
        let options = MLPredictionOptions()
        options.outputBackings = backings
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
        return try model.prediction(from: provider, options: options)
    }

    private func stateArray(_ buffer: MTLBuffer, firstLane: Int, lanes: Int, width: Int) throws -> MLMultiArray {
        let channels = InworldMambaScan.channels
        let offset = firstLane * Self.bands * channels * width * MemoryLayout<Float>.stride
        return try MLMultiArray(
            dataPointer: buffer.contents() + offset,
            shape: [lanes, Self.bands, channels, width].map { NSNumber(value: $0) }, dataType: .float32,
            strides: [Self.bands * channels * width, channels * width, width, 1].map { NSNumber(value: $0) },
            deallocator: { [buffer] _ in withExtendedLifetime(buffer) {} })
    }

    private func copyRawLane(_ input: MLMultiArray, index: Int, frames: Int) throws -> MLMultiArray {
        let bins = InworldIsolationDSP.bins
        let output = try MLMultiArray(shape: [1, frames, bins, 4].map { NSNumber(value: $0) }, dataType: .float32)
        let inputStrides = input.strides.map(\.intValue)
        input.withUnsafeBufferPointer(ofType: Float.self) { from in
            output.withUnsafeMutableBufferPointer(ofType: Float.self) { to, strides in
                if inputStrides[1] == bins * 4, inputStrides[2] == 4, inputStrides[3] == 1,
                   strides[1] == bins * 4, strides[2] == 4, strides[3] == 1,
                   let source = from.baseAddress, let destination = to.baseAddress {
                    destination.update(from: source + index * inputStrides[0], count: frames * bins * 4)
                    return
                }
                for frame in 0..<frames {
                    for bin in 0..<bins {
                        for component in 0..<4 {
                            to[frame * strides[1] + bin * strides[2] + component * strides[3]] =
                                from[index * inputStrides[0] + frame * inputStrides[1] + bin * inputStrides[2] + component * inputStrides[3]]
                        }
                    }
                }
            }
        }
        return output
    }
}
