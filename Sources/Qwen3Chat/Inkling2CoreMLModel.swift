// Copyright 2026 the HuggingFace Team. All rights reserved.
// Adapted from Apache-2.0 Inkling 2 reference equations for Core ML partition
// orchestration and explicit visit-local caches. See ThirdPartyNotices/Inkling2Mini.txt.
import Foundation
import CoreML
import AudioCommon

struct Inkling2CoreMLManifest {
    static let format = "speech-swift-inkling2-coreml-v1"
    struct Part {
        let package: String
        let compiled: String?
    }
    let contextLimit, prefillSize, imagePatchLimit, groupSize: Int
    let activationType: MLMultiArrayDataType
    let parts: [String: Part]

    init(data: Data, architecture: Inkling2Config) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              root["format"] as? String == Self.format,
              root["status"] as? String == "converted",
              root["weight_bits"] as? Int == 8,
              root["convolution_history_dtype"] as? String == "float32",
              let context = root["context_limit"] as? Int, (1...architecture.maxContext).contains(context),
              let prefill = root["prefill_size"] as? Int, (1...min(128, context)).contains(prefill),
              let images = root["image_patch_limit"] as? Int, (0...1024).contains(images),
              root["physical_layers"] as? Int == architecture.numLayers,
              let rawParts = root["parts"] as? [String: [String: Any]] else {
            throw Inkling2Error.invalid("expected a complete Core ML INT8 export with FP32 convolution histories")
        }
        switch root["activation_dtype"] as? String {
        case "float16": activationType = .float16
        case "float32": activationType = .float32
        default: throw Inkling2Error.invalid("unsupported Core ML activation type")
        }
        if let visits = root["layer_visits"] as? [Int], visits != architecture.layerVisits {
            throw Inkling2Error.invalid("Core ML logical layer visits do not match the architecture")
        }
        let names = ["embedding", "head"] + (0..<architecture.numLayers).map { "layer-\($0)" }
            + (images > 0 ? ["vision"] : [])
        guard Set(rawParts.keys) == Set(names) else {
            throw Inkling2Error.invalid("Core ML export must contain one partition per physical layer")
        }
        var parsed: [String: Part] = [:]
        for name in names {
            let raw = rawParts[name]!
            guard raw["package"] as? String == "\(name).mlpackage" else {
                throw Inkling2Error.invalid("invalid Core ML package path for \(name)")
            }
            let compiled = raw["compiled"] as? String
            guard compiled == nil || compiled == "compiled/\(name).mlmodelc" else {
                throw Inkling2Error.invalid("invalid compiled Core ML path for \(name)")
            }
            parsed[name] = Part(package: "\(name).mlpackage", compiled: compiled)
        }
        contextLimit = context; prefillSize = prefill; imagePatchLimit = images
        groupSize = root["expert_group_size"] as? Int ?? architecture.defaultQuantization.groupSize
        guard [32, 64, 128].contains(groupSize) else {
            throw Inkling2Error.invalid("unsupported Core ML INT8 group size")
        }
        parts = parsed
    }
}

enum InklingCoreMLTensor {
    static func shape(_ array: MLMultiArray) -> [Int] { array.shape.map(\.intValue) }

    private static func count(_ shape: [Int]) throws -> Int {
        guard !shape.isEmpty else { throw Inkling2Error.invalid("Core ML tensors require dimensions") }
        var count = 1
        for dimension in shape {
            let result = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !result.overflow, result.partialValue <= 50_000_000 else {
                throw Inkling2Error.invalid("invalid Core ML tensor dimensions")
            }
            count = result.partialValue
        }
        return count
    }

    static func make(_ shape: [Int], type: MLMultiArrayDataType = .float32,
                     values: [Float]? = nil) throws -> MLMultiArray {
        let count = try Self.count(shape)
        guard values == nil || values!.count == count else {
            throw Inkling2Error.invalid("Core ML tensor data does not match its shape")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
        switch type {
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            for index in 0..<count { pointer[index] = values?[index] ?? 0 }
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for index in 0..<count { pointer[index] = Float16(values?[index] ?? 0) }
        default: throw Inkling2Error.invalid("expected a floating-point Core ML tensor")
        }
        return array
    }

    static func integers(_ values: [Int], shape: [Int]) throws -> MLMultiArray {
        guard try count(shape) == values.count, values.allSatisfy({ Int32(exactly: $0) != nil }) else {
            throw Inkling2Error.invalid("invalid Core ML integer tensor")
        }
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        let pointer = array.dataPointer.assumingMemoryBound(to: Int32.self)
        for (index, value) in values.enumerated() { pointer[index] = Int32(value) }
        return array
    }

    static func offset(_ flat: Int, shape: [Int], strides: [Int]) -> Int {
        var rest = flat, result = 0
        for axis in shape.indices.reversed() {
            result += (rest % shape[axis]) * strides[axis]
            rest /= shape[axis]
        }
        return result
    }

    static func floats(_ array: MLMultiArray) throws -> [Float] {
        let dimensions = shape(array), strides = array.strides.map(\.intValue)
        var expected = 1, contiguous = true
        for axis in dimensions.indices.reversed() {
            if dimensions[axis] > 1 && strides[axis] != expected { contiguous = false }
            expected *= dimensions[axis]
        }
        let index: (Int) -> Int = { contiguous ? $0 : offset($0, shape: dimensions, strides: strides) }
        switch array.dataType {
        case .float32:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
            return (0..<array.count).map { pointer[index($0)] }
        case .float16:
            let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return (0..<array.count).map { Float(pointer[index($0)]) }
        default: throw Inkling2Error.invalid("unexpected Core ML output precision")
        }
    }

    static func slice(_ array: MLMultiArray, axis: Int, range: Range<Int>) throws -> MLMultiArray {
        let original = shape(array)
        guard original.indices.contains(axis), !range.isEmpty, range.lowerBound >= 0,
              range.upperBound <= original[axis] else {
            throw Inkling2Error.invalid("invalid Core ML tensor slice")
        }
        var target = original; target[axis] = range.count
        let source = try floats(array)
        let inner = original.dropFirst(axis + 1).reduce(1, *)
        let outer = original.prefix(axis).reduce(1, *)
        var result: [Float] = []; result.reserveCapacity(target.reduce(1, *))
        for group in 0..<outer {
            let start = (group * original[axis] + range.lowerBound) * inner
            result.append(contentsOf: source[start..<(start + range.count * inner)])
        }
        return try make(target, type: array.dataType, values: result)
    }

    static func bank(_ arrays: [MLMultiArray], tokens: Int, width: Int) throws -> MLMultiArray {
        var parts: [[Float]] = [], counts: [Int] = []
        for array in arrays {
            let dimensions = shape(array)
            guard dimensions.count == 4, dimensions[0] == 1, dimensions[1] == tokens,
                  dimensions[3] == width else { throw Inkling2Error.invalid("invalid residual bank shape") }
            parts.append(try floats(array)); counts.append(dimensions[2])
        }
        var result: [Float] = []
        for token in 0..<tokens {
            for (part, count) in zip(parts, counts) {
                let offset = token * count * width
                result.append(contentsOf: part[offset..<(offset + count * width)])
            }
        }
        return try make([1, tokens, counts.reduce(0, +), width], values: result)
    }
}

final class Inkling2CoreMLModel {
    struct LayerState {
        var keys, values: MLMultiArray
        var histories: [MLMultiArray]
    }
    struct State {
        var position = 0
        var layers: [LayerState?]
    }
    let config: Inkling2Config
    let manifest: Inkling2CoreMLManifest
    let contextLimit: Int
    private var models: [String: MLModel]
    private let vision: [String: Any]?
    let temporaryCompilationDirectory: URL?

    deinit {
        models.removeAll()
        if let temporaryCompilationDirectory {
            try? FileManager.default.removeItem(at: temporaryCompilationDirectory)
        }
    }

    init(directory: URL, contextLimit: Int, computeUnits: MLComputeUnits,
         progressHandler: ((Double, String) -> Void)? = nil) throws {
        guard contextLimit > 0 else { throw Inkling2Error.invalid("context limit must be positive") }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let architecture = try Inkling2Config(data: configData)
        let manifest = try Inkling2CoreMLManifest(data: Data(contentsOf:
            directory.appendingPathComponent("coreml-manifest.json")), architecture: architecture)
        guard contextLimit <= manifest.contextLimit else {
            throw Inkling2Error.invalid("context limit exceeds the exported Core ML capacity")
        }
        self.config = architecture; self.manifest = manifest; self.contextLimit = contextLimit
        let root = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
        vision = root?["vision_config"] as? [String: Any]
        var loaded: [String: MLModel] = [:]
        var temporary: URL?, completed = false
        defer {
            if !completed {
                loaded.removeAll()
                if let temporary { try? FileManager.default.removeItem(at: temporary) }
            }
        }
        let order = ["embedding"] + (0..<architecture.numLayers).map { "layer-\($0)" } + ["head"]
            + (manifest.imagePatchLimit > 0 ? ["vision"] : [])
        for (index, name) in order.enumerated() {
            let part = manifest.parts[name]!
            let compiled = part.compiled.map { directory.appendingPathComponent($0) }
            let url: URL
            if let compiled, FileManager.default.fileExists(atPath: compiled.path) { url = compiled }
            else {
                let package = directory.appendingPathComponent(part.package)
                guard FileManager.default.fileExists(atPath: package.path) else {
                    throw Inkling2Error.invalid("missing Core ML partition \(name)")
                }
                let compiledTemporary = try MLModel.compileModel(at: package)
                defer { try? FileManager.default.removeItem(at: compiledTemporary) }
                if temporary == nil {
                    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
                        "speech-swift-inkling2-coreml-\(UUID().uuidString)", isDirectory: true)
                    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false,
                        attributes: [.posixPermissions: 0o700])
                    temporary = directory
                }
                url = temporary!.appendingPathComponent("\(name).mlmodelc", isDirectory: true)
                try FileManager.default.moveItem(at: compiledTemporary, to: url)
            }
            loaded[name] = try CoreMLLoader.load(url: url, computeUnits: computeUnits, name: "inkling2-\(name)")
            progressHandler?(Double(index + 1) / Double(order.count), "Loaded Core ML \(name)")
        }
        models = loaded
        temporaryCompilationDirectory = temporary
        completed = true
    }

    func initialState() -> State { State(layers: Array(repeating: nil, count: config.layerVisits.count)) }

    private func predict(_ name: String, _ arrays: [String: MLMultiArray]) throws -> [String: MLMultiArray] {
        guard let model = models[name] else { throw Inkling2Error.invalid("missing Core ML partition \(name)") }
        let required = model.modelDescription.inputDescriptionsByName
        var features: [String: MLFeatureValue] = [:]
        for key in required.keys {
            guard let array = arrays[key] else { throw Inkling2Error.invalid("missing Core ML input \(key)") }
            features[key] = MLFeatureValue(multiArray: array)
        }
        let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        var outputs: [String: MLMultiArray] = [:]
        for key in result.featureNames {
            if let array = result.featureValue(for: key)?.multiArrayValue { outputs[key] = array }
        }
        return outputs
    }

    private func output(_ outputs: [String: MLMultiArray], _ name: String,
                        type: MLMultiArrayDataType) throws -> MLMultiArray {
        guard let value = outputs[name], value.dataType == type else {
            throw Inkling2Error.invalid("missing or incorrectly typed Core ML output \(name)")
        }
        return value
    }

    func embeddings(_ ids: [Int]) throws -> MLMultiArray {
        let result = try predict("embedding", ["input_ids": InklingCoreMLTensor.integers(ids, shape: [1, ids.count])])
        return try output(result, "embeddings", type: manifest.activationType)
    }

    func forward(_ ids: [Int], state: inout State, embeddings: MLMultiArray? = nil,
                 produceLogits: Bool = true, checkCancellation: (() throws -> Void)? = nil) throws -> [Float] {
        let c = config, count = ids.count, width = c.hiddenSize, dtype = manifest.activationType
        guard count > 0, count <= manifest.prefillSize, state.position >= 0,
              state.position <= contextLimit - count, state.layers.count == c.layerVisits.count,
              ids.allSatisfy({ (0..<c.vocabSize).contains($0) }) else {
            throw Inkling2Error.invalid("invalid Core ML tokens, visit state, or context budget")
        }
        var x = try embeddings ?? self.embeddings(ids)
        guard InklingCoreMLTensor.shape(x) == [1, count, width], x.dataType == dtype else {
            throw Inkling2Error.invalid("Core ML embeddings do not match the tokens")
        }
        var blocks = try InklingCoreMLTensor.make([1, count, 1, width])
        var bank: MLMultiArray?, prelude: MLMultiArray?, preludeBlocks: MLMultiArray?
        for (visit, layer) in c.layerVisits.enumerated() {
            try checkCancellation?()
            if layer == c.coreStart {
                if visit == layer { prelude = x; preludeBlocks = blocks }
                else {
                    let start = InklingCoreMLTensor.shape(preludeBlocks!)[2] + 1
                    let total = InklingCoreMLTensor.shape(blocks)[2]
                    var sources: [MLMultiArray] = []
                    if total > start { sources.append(try InklingCoreMLTensor.slice(blocks, axis: 2, range: start..<total)) }
                    sources.append(try InklingCoreMLTensor.make([1, count, 1, width], values: InklingCoreMLTensor.floats(x)))
                    bank = try InklingCoreMLTensor.bank(sources, tokens: count, width: width)
                    x = prelude!; blocks = preludeBlocks!
                }
            }
            let local = c.localLayers.contains(layer)
            let kv = local ? c.slidingKVHeads : c.numKVHeads, dim = local ? c.slidingHeadDim : c.headDim
            let initial = state.layers[visit] == nil
            let cache: LayerState
            if let previous = state.layers[visit] { cache = previous }
            else {
                cache = try LayerState(keys: InklingCoreMLTensor.make([1, kv, 1, dim], type: dtype),
                    values: InklingCoreMLTensor.make([1, kv, 1, dim], type: dtype),
                    histories: [kv * dim, kv * dim, width, width].map {
                        try InklingCoreMLTensor.make([1, c.convolutionSize - 1, $0])
                    })
            }
            let validCount = InklingCoreMLTensor.shape(cache.keys)[2]
            let arrays: [String: MLMultiArray] = try ["x": x, "blocks": blocks,
                "bank": bank ?? InklingCoreMLTensor.make([1, count, 1, width]),
                "enabled": InklingCoreMLTensor.make([1], values: [bank != nil && layer < c.coreEnd ? 1 : 0]),
                "keys": cache.keys, "values": cache.values,
                "valid": InklingCoreMLTensor.make([validCount], values: Array(repeating: initial ? 0 : 1, count: validCount)),
                "history_k": cache.histories[0], "history_v": cache.histories[1],
                "history_a": cache.histories[2], "history_m": cache.histories[3],
                "position": InklingCoreMLTensor.integers([state.position], shape: [1])]
            let result = try predict("layer-\(layer)", arrays)
            x = try output(result, "x_out", type: dtype)
            blocks = try output(result, "blocks_out", type: .float32)
            var keys = try output(result, "keys_out", type: dtype)
            var values = try output(result, "values_out", type: dtype)
            let expectedCache = local ? min(validCount + count, c.slidingWindow) : validCount + count
            guard InklingCoreMLTensor.shape(x) == [1, count, width],
                  InklingCoreMLTensor.shape(keys) == [1, kv, expectedCache, dim],
                  InklingCoreMLTensor.shape(values) == [1, kv, expectedCache, dim] else {
                throw Inkling2Error.invalid("Core ML layer \(layer) returned an unexpected activation or cache shape")
            }
            if initial && (!local || count < c.slidingWindow) {
                let length = InklingCoreMLTensor.shape(keys)[2]
                keys = try InklingCoreMLTensor.slice(keys, axis: 2, range: 1..<length)
                values = try InklingCoreMLTensor.slice(values, axis: 2, range: 1..<length)
            }
            state.layers[visit] = try LayerState(keys: keys, values: values,
                histories: ["history_k_out", "history_v_out", "history_a_out", "history_m_out"].map {
                    try output(result, $0, type: .float32)
                })
        }
        state.position += count
        guard produceLogits else { return [] }
        let lastX = try InklingCoreMLTensor.slice(x, axis: 1, range: (count - 1)..<count)
        let lastBlocks = try InklingCoreMLTensor.slice(blocks, axis: 1, range: (count - 1)..<count)
        let result = try predict("head", ["x": lastX, "blocks": lastBlocks])
        return try InklingCoreMLTensor.floats(output(result, "logits", type: .float32))
    }

    func imageEmbeddings(_ image: Inkling2Image) throws -> MLMultiArray {
        guard manifest.imagePatchLimit > 0, let vision,
              let hidden = vision["hidden_size"] as? Int,
              let heads = vision["num_attention_heads"] as? Int, heads > 0, hidden % heads == 0,
              hidden / heads % 4 == 0,
              vision["native_patch_size"] as? Int == 20,
              vision["adapter_merge_size"] as? Int == 2,
              let grid = vision["posemb_grid_size"] as? Int, grid > 0, grid <= 256 else {
            throw Inkling2Error.invalid("bundle does not contain the supported Core ML vision encoder")
        }
        let pixels = try image.patchPixels(maxTokens: manifest.imagePatchLimit)
        let patches = image.rows * image.columns, dim = hidden / heads
        var indices: [Int] = [], coefficients: [Float] = [], coordinates: [(Int, Int)] = [(0, 0)]
        for row in 0..<image.rows {
            for column in 0..<image.columns {
                for dy in 0..<2 { for dx in 0..<2 {
                    let y = row * 2 + dy, x = column * 2 + dx
                    let yf = max(0, (Float(y) + 0.5) * Float(grid) / Float(image.rows * 2) - 0.5)
                    let xf = max(0, (Float(x) + 0.5) * Float(grid) / Float(image.columns * 2) - 0.5)
                    let y0 = min(Int(yf), grid - 1), x0 = min(Int(xf), grid - 1)
                    let y1 = min(y0 + 1, grid - 1), x1 = min(x0 + 1, grid - 1)
                    let wy = yf - Float(y0), wx = xf - Float(x0)
                    indices += [y0 * grid + x0, y0 * grid + x1, y1 * grid + x0, y1 * grid + x1]
                    coefficients += [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]
                    coordinates.append((x + 1, y + 1))
                } }
            }
        }
        let inverse = stride(from: 0, to: dim / 2, by: 2).map { powf(10000, -Float($0) / Float(dim / 2)) }
        var cosine: [Float] = [], sine: [Float] = []
        for (x, y) in coordinates {
            for coordinate in [x, y] {
                for frequency in inverse {
                    let angle = Float(coordinate) * frequency
                    cosine += [cosf(angle), cosf(angle)]; sine += [sinf(angle), sinf(angle)]
                }
            }
        }
        let input: [String: MLMultiArray] = try [
            "pixels": InklingCoreMLTensor.make([patches, 40, 40, 3], values: pixels),
            "position_indices": InklingCoreMLTensor.integers(indices, shape: [patches * 4, 4]),
            "position_coefficients": InklingCoreMLTensor.make([patches * 4, 4], values: coefficients),
            "cosine": InklingCoreMLTensor.make([patches * 4 + 1, dim], values: cosine),
            "sine": InklingCoreMLTensor.make([patches * 4 + 1, dim], values: sine)]
        let result = try predict("vision", input)
        let features = try output(result, "image_embeddings", type: .float32)
        guard InklingCoreMLTensor.shape(features) == [patches, config.hiddenSize] else {
            throw Inkling2Error.invalid("Core ML image features have an unexpected shape")
        }
        return features
    }
}
