import AudioCommon
import CoreML
import Foundation
import Metal

/// Keep one set of Core ML models and Metal scan buffers resident for the session.
final class InworldStreamingBackbone: InworldStreamingInference {
    private static let bands = InworldVoiceIsolator.bands
    private static let width = InworldVoiceIsolator.hiddenWidth
    private static let history = 3
    private static let halfHeadSize = 32
    private static let prefixBucket = 64

    final class Lane {
        var frames = 0
        let attention: [MLState]
        var convolution: [MLMultiArray]
        let carry: [MTLBuffer]

        init(attention: [MLState], convolution: [MLMultiArray], carry: [MTLBuffer]) {
            self.attention = attention
            self.convolution = convolution
            self.carry = carry
        }
    }

    let blockFrames: Int
    private let cacheFrames: Int
    private let stages: [MLModel]
    private let stageFiles: InworldCoreMLStages
    private let scan: InworldMambaScan
    private let workspace: InworldMambaScan.Workspace
    private let zeroHistory: MLMultiArray
    private let lookahead: MLMultiArray
    private let rotaryCosines: [Float]
    private let rotarySines: [Float]

    init(modelDirectory: URL, blockFrames: Int, lookaheadIndex: Int, configuration: MLModelConfiguration) throws {
        let coefficients = try Data(contentsOf: modelDirectory.appendingPathComponent("scan_coefficients.bin"))
        let scan = try InworldMambaScan(coefficients: coefficients)
        let workspace = try scan.workspace(tokens: Self.bands * blockFrames)
        let history = try Self.array([Self.bands, InworldMambaScan.channels, Self.history])
        history.withUnsafeMutableBufferPointer(ofType: Float.self) { values, _ in
            values.baseAddress!.update(repeating: 0, count: values.count)
        }
        let lookahead = try MLMultiArray(shape: [1], dataType: .int32)
        lookahead[0] = NSNumber(value: lookaheadIndex)

        let stageFiles = try InworldCoreMLStages(directory: modelDirectory, count: InworldMambaScan.stageCount + 1)
        self.stageFiles = stageFiles
        self.stages = try stageFiles.urls.enumerated().map { index, location in
            try CoreMLLoader.load(url: location, configuration: configuration, name: "inworld-streaming-stage_\(index)")
        }
        self.blockFrames = blockFrames
        self.cacheFrames = InworldStreamingLayout.laneCacheFrames
        self.scan = scan
        self.workspace = workspace
        self.zeroHistory = history
        self.lookahead = lookahead
        let rates = (0..<Self.halfHeadSize).map { pow(10_000, -Float($0) / Float(Self.halfHeadSize)) }
        let angles = (0..<InworldStreamingLayout.laneCacheFrames).flatMap { frame in rates.map { Float(frame) * $0 } }
        self.rotaryCosines = angles.map(cos)
        self.rotarySines = angles.map(sin)
    }

    func makeLane() throws -> Lane {
        let attention = stride(from: 0, to: stages.count, by: 2).map { stages[$0].makeState() }
        let carry = try (0..<3).map { _ in try scan.makeCarry(rows: Self.bands) }
        return Lane(attention: attention, convolution: Array(repeating: zeroHistory, count: 3), carry: carry)
    }

    func beginLane(copying active: Lane, reusing spare: Lane?) throws -> Lane {
        let lane = try spare ?? makeLane()
        lane.frames = 0
        lane.convolution = active.convolution
        for (destination, source) in zip(lane.carry, active.carry) {
            destination.contents().copyMemory(from: source.contents(), byteCount: destination.length)
        }
        return lane
    }

    func retireLane(_ lane: Lane) {}

    func predict(_ spectrum: MLMultiArray, frames: Int, lanes: [Lane]) throws -> [MLMultiArray] {
        try lanes.map { try predict(spectrum, frames: frames, lane: $0) }
    }

    private func predict(_ spectrum: MLMultiArray, frames: Int, lane: Lane) throws -> MLMultiArray {
        let attention = try attentionInputs(first: lane.frames, frames: frames)
        var inputs = ["spectrum": spectrum, "lookahead_index": lookahead]
        for (index, model) in stages.enumerated() {
            let timeAxis = index.isMultiple(of: 2)
            let block = index / 2
            let last = index == InworldMambaScan.stageCount
            if timeAxis {
                inputs.merge(attention) { _, next in next }
                if !last { inputs["convolution_history"] = lane.convolution[block] }
            }
            let rows = timeAxis ? Self.bands : frames
            let steps = timeAxis ? frames : Self.bands
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let output = try timeAxis
                ? model.prediction(from: provider, using: lane.attention[block])
                : model.prediction(from: provider)
            if last {
                let raw = try InworldVoiceIsolator.array(output, named: "mask_residual",
                    shape: [1, frames, InworldIsolationDSP.bins, 4])
                lane.frames += frames
                return raw
            }
            let packed = try InworldVoiceIsolator.array(output, named: "scan_input",
                shape: [timeAxis ? 1 : 2, rows, steps, InworldMambaScan.packedWidth])
            let scanInput = try workspace.inputArray(rows: rows, steps: steps, directions: timeAxis ? 1 : 2)
            packed.transfer(to: scanInput)
            let scanned = try scan.run(stage: index, rows: rows, steps: steps, workspace: workspace,
                                       carry: timeAxis ? lane.carry[block] : nil)
            if timeAxis {
                lane.convolution[block] = try InworldVoiceIsolator.array(output, named: "next_history",
                    shape: [Self.bands, InworldMambaScan.channels, Self.history])
            }
            let features = try InworldVoiceIsolator.array(output, named: "scan_features", shape: [rows, steps, Self.width])
            inputs = ["features": features, "scan_output": scanned]
        }
        throw InworldIsolationError.predictionFailed("the streaming bundle did not produce a decoder result")
    }

    private func attentionInputs(first: Int, frames: Int) throws -> [String: MLMultiArray] {
        let end = first + frames
        guard end <= cacheFrames else {
            throw InworldIsolationError.predictionFailed("the streaming lane exceeded its attention cache")
        }
        let bucket = Self.prefixBucket
        let prefix = min((end + bucket - 1) / bucket * bucket, cacheFrames)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        position[0] = NSNumber(value: first)
        let cosines = try rotaryArray(rotaryCosines, first: first, frames: frames)
        let sines = try rotaryArray(rotarySines, first: first, frames: frames)
        let mask = try Self.array([frames, prefix])
        mask.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
            for frame in 0..<frames {
                for key in 0..<prefix {
                    values[frame * strides[0] + key * strides[1]] = key <= first + frame ? 0 : -1e9
                }
            }
        }
        return ["cache_position": position, "rope_cos": cosines, "rope_sin": sines, "attention_mask": mask]
    }

    private func rotaryArray(_ table: [Float], first: Int, frames: Int) throws -> MLMultiArray {
        let width = Self.halfHeadSize
        let result = try Self.array([frames, width])
        result.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
            for frame in 0..<frames {
                for i in 0..<width {
                    values[frame * strides[0] + i * strides[1]] = table[(first + frame) * width + i]
                }
            }
        }
        return result
    }

    private static func array(_ shape: [Int]) throws -> MLMultiArray {
        try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
    }

}
