import AudioCommon
import CoreML
import Foundation

/// Run distinct time and band axes in groups of independent rows.
struct InworldTiledBackbone {
    static let format = "speech-swift-inworld-voice-isolator-tiled-v1"
    static let stageCount = 22

    let timeRows: Int
    let bandRows: Int
    let supportsOutputBackings: Bool
    let supportsStaticTiles: Bool
    let urls: [URL]
    let computeUnits: MLComputeUnits
    let scan: InworldMambaScan

    /// Process one or two windows of the same length with a single load per stage.
    func predict(spectra: [MLMultiArray], frames: Int) throws -> [MLMultiArray] {
        let hiddenShape = [InworldVoiceIsolator.bands, frames, InworldVoiceIsolator.hiddenWidth]
        var windows = try autoreleasepool {
            let encoder = try stage(0)
            return try spectra.map { spectrum in
                try autoreleasepool {
                    let result = try prediction(encoder, inputs: ["spectrum": spectrum])
                    return try InworldVoiceIsolator.array(result, named: "hidden", shape: hiddenShape)
                }
            }
        }

        var stageIndex = 1
        var scanIndex = 0
        for block in 0..<7 {
            let usesScan = !block.isMultiple(of: 2)
            for acrossBands in [false, true] {
                try autoreleasepool {
                    try axis(&windows, stageIndex: stageIndex, scanIndex: usesScan ? scanIndex : nil,
                             acrossBands: acrossBands, frames: frames)
                }
                stageIndex += usesScan ? 2 : 1
                if usesScan { scanIndex += 1 }
            }
        }

        return try autoreleasepool {
            let decoder = try stage(stageIndex)
            return try windows.map { hidden in
                try autoreleasepool {
                    let result = try prediction(decoder, inputs: ["features": hidden])
                    return try InworldVoiceIsolator.array(
                        result, named: "mask_residual", shape: [1, frames, InworldIsolationDSP.bins, 4])
                }
            }
        }
    }

    private func axis(
        _ windows: inout [MLMultiArray], stageIndex: Int, scanIndex: Int?, acrossBands: Bool, frames: Int
    ) throws {
        let bands = InworldVoiceIsolator.bands
        let width = InworldVoiceIsolator.hiddenWidth
        let rows = acrossBands ? frames : bands
        let steps = acrossBands ? bands : frames
        let rowLimit = min(rows, acrossBands ? bandRows : timeRows)
        let fullExtent = acrossBands ? rows >= bandRows : frames == InworldVoiceIsolator.fullFrames
        let fixedProjection = fullExtent && supportsOutputBackings && scanIndex != nil
        let fixedTile = fullExtent && supportsStaticTiles
        let useFixedFunction = fixedTile || fixedProjection
        let model = try stage(stageIndex, fullTile: useFixedFunction)
        let merge = scanIndex == nil ? nil : try stage(stageIndex + 1, fullTile: fixedTile)
        let workspace = scanIndex == nil ? nil : try scan.workspace(tokens: rowLimit * steps)
        for i in windows.indices {
            windows[i] = try autoreleasepool {
                let hidden = windows[i]
                let output = try MLMultiArray(shape: hidden.shape, dataType: .float32)

                for start in stride(from: 0, to: rows, by: rowLimit) {
                    try autoreleasepool {
                        let count = min(rowLimit, rows - start)
                        let modeledRows = acrossBands && !useFixedFunction ? count : rowLimit
                        let tileShape = [modeledRows, steps, width]
                        let tile = try MLMultiArray(shape: tileShape.map { NSNumber(value: $0) }, dataType: .float32)
                        if modeledRows > count {
                            tile.withUnsafeMutableBufferPointer(ofType: Float.self) { values, _ in
                                values.baseAddress?.update(repeating: 0, count: values.count)
                            }
                        }
                        Self.copy(hidden, to: tile, start: start, rows: count, steps: steps,
                                  acrossBands: acrossBands, sourceIsTile: false)

                        let destination = try workspace?.inputArray(rows: modeledRows, steps: steps)
                        let result = try prediction(model, inputs: ["features": tile],
                                                    scanBacking: fixedProjection ? destination : nil)
                        let finished: MLFeatureProvider
                        if let scanIndex, let merge, let workspace, let destination {
                            let packed = try InworldVoiceIsolator.array(result, named: "scan_input",
                                shape: [2, modeledRows, steps, InworldMambaScan.packedWidth])
                            if packed !== destination { packed.transfer(to: destination) }
                            let scanned = try scan.run(stage: scanIndex, rows: modeledRows, steps: steps, workspace: workspace)
                            finished = try prediction(merge, inputs: ["features": tile, "scan_output": scanned])
                        } else {
                            finished = result
                        }
                        let processed = try InworldVoiceIsolator.array(finished, named: "hidden", shape: tileShape)
                        Self.copy(processed, to: output, start: start, rows: count, steps: steps,
                                  acrossBands: acrossBands, sourceIsTile: true)
                    }
                }
                return output
            }
        }
    }

    private func stage(_ index: Int, fullTile: Bool = false) throws -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        if fullTile { configuration.functionName = "full_tile" }
        return try CoreMLLoader.load(url: urls[index], configuration: configuration,
                                     name: "inworld-tiled-stage_\(index)")
    }

    private func prediction(
        _ model: MLModel, inputs: [String: MLMultiArray], scanBacking: MLMultiArray? = nil
    ) throws -> MLFeatureProvider {
        let options = MLPredictionOptions()
        if let scanBacking { options.outputBackings = ["scan_input": scanBacking] }
        return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs), options: options)
    }

    /// Move a group of rows into or out of the canonical [band, frame, width] layout.
    private static func copy(
        _ source: MLMultiArray, to destination: MLMultiArray, start: Int, rows: Int, steps: Int,
        acrossBands: Bool, sourceIsTile: Bool
    ) {
        let sourceStrides = source.strides.map(\.intValue)
        let width = InworldVoiceIsolator.hiddenWidth
        source.withUnsafeBufferPointer(ofType: Float.self) { input in
            destination.withUnsafeMutableBufferPointer(ofType: Float.self) { output, destinationStrides in
                let full = sourceIsTile ? destinationStrides : sourceStrides
                let tile = sourceIsTile ? sourceStrides : destinationStrides
                for row in 0..<rows {
                    for step in 0..<steps {
                        let band = acrossBands ? step : start + row
                        let frame = acrossBands ? start + row : step
                        let fullOffset = band * full[0] + frame * full[1]
                        let tileOffset = row * tile[0] + step * tile[1]
                        let from = sourceIsTile ? tileOffset : fullOffset
                        let to = sourceIsTile ? fullOffset : tileOffset
                        if sourceStrides[2] == 1 && destinationStrides[2] == 1 {
                            output.baseAddress!.advanced(by: to).update(from: input.baseAddress! + from, count: width)
                        } else {
                            for channel in 0..<width {
                                output[to + channel * destinationStrides[2]] = input[from + channel * sourceStrides[2]]
                            }
                        }
                    }
                }
            }
        }
    }
}
