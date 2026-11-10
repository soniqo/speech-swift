// Standalone diagnostic for the manually selected CPU compatibility job.
// Compile with swiftc; no package dependencies or model downloads are needed.
import CoreML
import Foundation

func inspect(_ array: MLMultiArray) throws -> [String: Any] {
    let shape = array.shape.map(\.intValue)
    let strides = array.strides.map(\.intValue)
    guard shape.count == 3, shape[0] == 1 else {
        throw NSError(domain: "CPUCompatibilityProbe", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "Unexpected logits shape: \(shape)"])
    }
    var finite = 0
    var nonzero = 0
    var minimum = Float.infinity
    var maximum = -Float.infinity
    func visit(_ value: Float) {
        if value.isFinite {
            finite += 1
            if value != 0 { nonzero += 1 }
            minimum = min(minimum, value)
            maximum = max(maximum, value)
        }
    }
    for row in 0..<shape[1] {
        for column in 0..<shape[2] {
            let offset = row * strides[1] + column * strides[2]
            switch array.dataType {
            case .float16:
                visit(Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[offset]))
            case .float32:
                visit(array.dataPointer.assumingMemoryBound(to: Float.self)[offset])
            default:
                throw NSError(domain: "CPUCompatibilityProbe", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Unexpected logits type"])
            }
        }
    }
    return ["shape": shape, "strides": strides, "dtype": array.dataType.rawValue,
            "count": array.count, "finite": finite, "finite_nonzero": nonzero,
            "minimum": finite == 0 ? NSNull() : minimum as Any,
            "maximum": finite == 0 ? NSNull() : maximum as Any]
}

guard CommandLine.arguments.count == 2 else {
    fatalError("Usage: probe-coreml-aligner-cpu <text_decoder.mlmodelc>")
}
let modelURL = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
for backing in [false, true] {
    var report: [String: Any] = ["output_backing": backing,
        "model": modelURL.path, "os": ProcessInfo.processInfo.operatingSystemVersionString]
    autoreleasepool {
        do {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuOnly
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            guard let constraint = model.modelDescription.inputDescriptionsByName["inputs_embeds"]?.multiArrayConstraint,
                  let outputConstraint = model.modelDescription.outputDescriptionsByName["logits"]?.multiArrayConstraint else {
                throw NSError(domain: "CPUCompatibilityProbe", code: 3,
                              userInfo: [NSLocalizedDescriptionKey: "Missing decoder interface"])
            }
            let input = try MLMultiArray(shape: constraint.shape, dataType: .float32)
            let pointer = input.dataPointer.assumingMemoryBound(to: Float.self)
            let hidden = constraint.shape.last!.intValue
            for index in 0..<input.count {
                pointer[index] = index < 400 * hidden ? sin(Float(index) * 0.013) * 0.02 : 0
            }
            let options = MLPredictionOptions()
            if backing {
                let output = try MLMultiArray(shape: outputConstraint.shape, dataType: outputConstraint.dataType)
                options.outputBackings = ["logits": output]
            }
            let features = try MLDictionaryFeatureProvider(dictionary: ["inputs_embeds": input])
            let prediction = try model.prediction(from: features, options: options)
            guard let logits = prediction.featureValue(for: "logits")?.multiArrayValue else {
                throw NSError(domain: "CPUCompatibilityProbe", code: 4,
                              userInfo: [NSLocalizedDescriptionKey: "Missing logits"])
            }
            report.merge(try inspect(logits)) { _, new in new }
        } catch {
            report["error"] = String(describing: error)
        }
    }
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    print("[COREML-CPU-NUMERICS] \(String(decoding: data, as: UTF8.self))")
}
