import CoreML
import Foundation
import Metal

final class InworldMambaScan {
    static let stageCount = 6
    static let channels = 1_024
    static let states = 16
    static let packedWidth = 3 * channels + 2 * states
    private static let coefficientBytes = 2 * channels * (states + 1) * MemoryLayout<Float>.stride

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let coefficients: MTLBuffer

    init(coefficients data: Data) throws {
        guard data.count == Self.stageCount * Self.coefficientBytes else {
            throw InworldIsolationError.invalidBundle("incorrect scan_coefficients.bin size")
        }
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw InworldIsolationError.predictionFailed("a Metal device is required")
        }
        let buffer = data.withUnsafeBytes { bytes in
            device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
        }
        guard let buffer else {
            throw InworldIsolationError.predictionFailed("could not allocate scan coefficients")
        }
        let library = try device.makeLibrary(source: Self.source, options: nil)
        guard let function = library.makeFunction(name: "inworld_scan") else {
            throw InworldIsolationError.predictionFailed("could not load the Mamba scan kernel")
        }
        self.device = device
        self.queue = queue
        self.pipeline = try device.makeComputePipelineState(function: function)
        self.coefficients = buffer
    }

    func workspace(tokens: Int) throws -> Workspace {
        let inputBytes = 2 * tokens * Self.packedWidth * MemoryLayout<Float>.stride
        let outputBytes = 2 * tokens * Self.channels * MemoryLayout<Float>.stride
        guard let input = device.makeBuffer(length: inputBytes, options: .storageModeShared),
              let output = device.makeBuffer(length: outputBytes, options: .storageModeShared) else {
            throw InworldIsolationError.predictionFailed("could not allocate scan buffers for \(tokens) tokens")
        }
        return Workspace(input: input, output: output)
    }

    func makeCarry(rows: Int) throws -> MTLBuffer {
        let size = rows * Self.channels * Self.states * MemoryLayout<Float>.stride
        guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
            throw InworldIsolationError.predictionFailed("could not allocate the streaming scan state")
        }
        buffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: size)
        return buffer
    }

    func run(stage: Int, rows: Int, steps: Int, workspace: Workspace, carry: MTLBuffer? = nil,
             lanes: Int? = nil) throws -> MLMultiArray {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw InworldIsolationError.predictionFailed("could not create a Metal command for Mamba")
        }
        let directions = carry == nil ? 2 : 1
        var dimensions = SIMD4<UInt32>(UInt32(rows), UInt32(steps), UInt32(directions), carry == nil ? 0 : 1)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(workspace.input, offset: 0, index: 0)
        encoder.setBuffer(coefficients, offset: stage * Self.coefficientBytes, index: 1)
        encoder.setBuffer(workspace.output, offset: 0, index: 2)
        encoder.setBytes(&dimensions, length: MemoryLayout.size(ofValue: dimensions), index: 3)
        encoder.setBuffer(carry ?? workspace.output, offset: 0, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: directions * rows * Self.channels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw InworldIsolationError.predictionFailed(command.error?.localizedDescription ?? "Metal scan failed")
        }
        return try workspace.outputArray(rows: rows, steps: steps, directions: directions, lanes: lanes)
    }

    struct Workspace {
        fileprivate let input: MTLBuffer
        fileprivate let output: MTLBuffer

        func inputArray(rows: Int, steps: Int, directions: Int = 2, lanes: Int? = nil) throws -> MLMultiArray {
            try Self.array(backedBy: input, rows: rows, steps: steps, directions: directions,
                           width: InworldMambaScan.packedWidth, lanes: lanes)
        }

        fileprivate func outputArray(rows: Int, steps: Int, directions: Int, lanes: Int?) throws -> MLMultiArray {
            try Self.array(backedBy: output, rows: rows, steps: steps, directions: directions,
                           width: InworldMambaScan.channels, lanes: lanes)
        }

        private static func array(
            backedBy buffer: MTLBuffer, rows: Int, steps: Int, directions: Int, width: Int, lanes: Int?
        ) throws -> MLMultiArray {
            let shape = lanes.map { [directions, $0, rows / $0, steps, width] } ?? [directions, rows, steps, width]
            let strides = lanes.map { [rows * steps * width, rows / $0 * steps * width, steps * width, width, 1] }
                ?? [rows * steps * width, steps * width, width, 1]
            return try MLMultiArray(
                dataPointer: buffer.contents(),
                shape: shape.map { NSNumber(value: $0) },
                dataType: .float32,
                strides: strides.map { NSNumber(value: $0) },
                deallocator: { [buffer] _ in withExtendedLifetime(buffer) {} })
        }
    }

    // Recurrence adapted from Inworld Voice Isolator (Copyright 2026 Inworld AI,
    // Apache-2.0) for the contiguous Core ML projection layout. Each thread
    // keeps one channel's state in registers across the sequence.
    private static let source = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void inworld_scan(
        device const float *packed [[buffer(0)]],
        device const float *coefficients [[buffer(1)]],
        device float *output [[buffer(2)]],
        constant uint4 &dimensions [[buffer(3)]],
        device float *carried [[buffer(4)]],
        uint index [[thread_position_in_grid]]) {
        constexpr uint channels = \(channels);
        constexpr uint states = \(states);
        constexpr uint packedWidth = \(packedWidth);
        const uint rows = dimensions.x;
        const uint steps = dimensions.y;
        if (index >= dimensions.z * rows * channels) return;

        const uint row = index / channels;
        const uint channel = index % channels;
        const uint direction = row / rows;
        device const float *rate = coefficients + (direction * channels + channel) * (states + 1);
        float state[states] = {};
        if (dimensions.w) {
            for (uint n = 0; n < states; ++n) state[n] = carried[index * states + n];
        }
        for (uint step = 0; step < steps; ++step) {
            const uint token = row * steps + step;
            device const float *values = packed + token * packedWidth;
            const float u = values[channel];
            const float delta = values[channels + channel];
            const float z = values[2 * channels + channel];
            const float drive = delta * u;
            float y = 0.0f;
            for (uint n = 0; n < states; ++n) {
                state[n] = precise::exp(delta * rate[n]) * state[n] + drive * values[3 * channels + n];
                y += state[n] * values[3 * channels + states + n];
            }
            y += u * rate[states];
            output[token * channels + channel] = y * precise::divide(z, 1.0f + precise::exp(-z));
        }
        if (dimensions.w) {
            for (uint n = 0; n < states; ++n) carried[index * states + n] = state[n];
        }
    }
    """
}
