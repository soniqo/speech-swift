import CoreML
import Foundation
import Metal

/// Run the 46 band-specific output heads in one Metal command.
final class InworldBandDecoder {
    private static let bands = InworldVoiceIsolator.bands
    private static let width = InworldVoiceIsolator.hiddenWidth
    private static let channels = InworldIsolationDSP.bins * 4
    private static let featureCount = bands * width
    private static let weightCount = featureCount * width
    private static let finalWeightCount = channels * width
    private static let biasOffset = weightCount * 2
    private static let finalOffset = biasOffset + featureCount * 2
    private static let finalBiasOffset = finalOffset + finalWeightCount * 4
    private static let scaleOffset = finalBiasOffset + channels * 4
    private static let shiftOffset = scaleOffset + width * 4
    private static let gainOffset = shiftOffset + width * 4
    private static let normBiasOffset = gainOffset + featureCount * 4
    private static let bandsOffset = normBiasOffset + featureCount * 4

    private let weights: MTLBuffer
    private let features: MTLBuffer
    private let normRows: MTLBuffer
    private let activations: MTLBuffer
    private let output: MTLBuffer
    private let queue: MTLCommandQueue
    private let normalize: MTLComputePipelineState
    private let expand: MTLComputePipelineState
    private let project: MTLComputePipelineState
    private let bandsByChannel: MTLBuffer
    private let frames: Int

    init(url: URL, blockFrames: Int, laneCapacity: Int) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let requiredBytes = Self.bandsOffset + (Self.bands + 1) * 4
        guard data.count == requiredBytes else {
            throw InworldIsolationError.invalidBundle("invalid streaming decoder weights")
        }
        let bandIndices = try Self.channelBands(in: data.suffix((Self.bands + 1) * 4))
        let featureElements = Self.featureCount * blockFrames * laneCapacity
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let features = device.makeBuffer(length: featureElements * 4, options: .storageModeShared),
              let normRows = device.makeBuffer(length: featureElements * 2, options: .storageModePrivate),
              let activations = device.makeBuffer(length: featureElements * 4, options: .storageModePrivate),
              let output = device.makeBuffer(length: Self.channels * blockFrames * laneCapacity * 4, options: .storageModeShared),
              let bandsByChannel = bandIndices.withUnsafeBytes({ bytes in
                  device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
              }),
              let weights = data.withUnsafeBytes({ bytes in
                  device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
              }) else {
            throw InworldIsolationError.predictionFailed("could not allocate the streaming decoder")
        }
        let library = try device.makeLibrary(source: Self.source(blockFrames: blockFrames, laneCapacity: laneCapacity), options: nil)
        func pipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw InworldIsolationError.predictionFailed("missing streaming decoder kernel")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.normalize = try pipeline("normalize_bands")
        self.expand = try pipeline("expand_bands")
        self.project = try pipeline("project_bands")
        self.weights = weights
        self.bandsByChannel = bandsByChannel
        self.features = features
        self.normRows = normRows
        self.activations = activations
        self.output = output
        self.queue = queue
        self.frames = blockFrames
    }

    /// Decode the little-endian output channel boundaries written by the exporter.
    static func channelBands(in data: Data) throws -> [UInt8] {
        let stride = MemoryLayout<UInt32>.stride
        guard data.count == (bands + 1) * stride else {
            throw InworldIsolationError.invalidBundle("invalid streaming decoder band count")
        }
        let boundaries = data.withUnsafeBytes { bytes in
            (0...bands).map { Int(UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * stride, as: UInt32.self))) }
        }
        guard boundaries.first == 0, boundaries.last == channels else {
            throw InworldIsolationError.invalidBundle("invalid streaming decoder channel range")
        }
        var result: [UInt8] = []
        result.reserveCapacity(channels)
        for (band, (start, end)) in zip(boundaries, boundaries.dropFirst()).enumerated() {
            guard end > start, end <= channels, end.isMultiple(of: 4) else {
                throw InworldIsolationError.invalidBundle("invalid streaming decoder band boundary")
            }
            result.append(contentsOf: repeatElement(UInt8(band), count: end - start))
        }
        return result
    }

    func input(lanes: Int) throws -> MLMultiArray {
        try Self.array(features, shape: [lanes, Self.bands, frames, Self.width],
                       strides: [Self.bands * frames * Self.width, frames * Self.width, Self.width, 1])
    }

    func run(lanes: Int) throws -> MLMultiArray {
        guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else {
            throw InworldIsolationError.predictionFailed("could not create a decoder command")
        }
        var rowCount = UInt32(lanes * frames)

        encoder.setComputePipelineState(normalize)
        encoder.setBuffer(features, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: Self.scaleOffset, index: 1)
        encoder.setBuffer(weights, offset: Self.shiftOffset, index: 2)
        encoder.setBuffer(weights, offset: Self.gainOffset, index: 3)
        encoder.setBuffer(weights, offset: Self.normBiasOffset, index: 4)
        encoder.setBuffer(normRows, offset: 0, index: 5)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: Self.bands, height: Int(rowCount), depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)

        let groups = 4
        let threads = MTLSize(width: groups * 32, height: 1, depth: 1)
        encoder.setComputePipelineState(expand)
        encoder.setBuffer(normRows, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: Self.biasOffset, index: 2)
        encoder.setBuffer(activations, offset: 0, index: 3)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: Self.width / groups, height: Self.bands, depth: 1),
                                     threadsPerThreadgroup: threads)
        encoder.memoryBarrier(scope: .buffers)

        encoder.setComputePipelineState(project)
        encoder.setBuffer(activations, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: Self.finalOffset, index: 1)
        encoder.setBuffer(weights, offset: Self.finalBiasOffset, index: 2)
        encoder.setBuffer(bandsByChannel, offset: 0, index: 3)
        encoder.setBuffer(output, offset: 0, index: 4)
        encoder.setBytes(&rowCount, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: (Self.channels + groups - 1) / groups, height: 1, depth: 1),
                                     threadsPerThreadgroup: threads)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw InworldIsolationError.predictionFailed(command.error?.localizedDescription ?? "Metal decoder failed")
        }
        return try Self.array(output, shape: [lanes, frames, InworldIsolationDSP.bins, 4],
                              strides: [frames * Self.channels, Self.channels, 4, 1])
    }

    private static func array(_ buffer: MTLBuffer, shape: [Int], strides: [Int]) throws -> MLMultiArray {
        try MLMultiArray(dataPointer: buffer.contents(), shape: shape.map { NSNumber(value: $0) }, dataType: .float32,
                         strides: strides.map { NSNumber(value: $0) },
                         deallocator: { [buffer] _ in withExtendedLifetime(buffer) {} })
    }

    private static func source(blockFrames: Int, laneCapacity: Int) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        constant uint bands = 46;
        constant uint vectors = 128;
        constant uint features = 512;
        constant uint channels = 4100;
        constant uint frames = \(blockFrames);
        constant uint groups = 4;
        #define DECODER_ROWS (\(blockFrames) * \(laneCapacity))

        kernel void normalize_bands(device const float4 *input [[buffer(0)]],
                                    device const float4 *scale [[buffer(1)]],
                                    device const float4 *shift [[buffer(2)]],
                                    device const float4 *gain [[buffer(3)]],
                                    device const float4 *bias [[buffer(4)]],
                                    device half4 *output [[buffer(5)]],
                                    constant uint &rows [[buffer(6)]],
                                    uint2 group [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
            uint band = group.x;
            uint row = group.y;
            uint inputStart = ((row / frames) * bands * frames + band * frames + (row % frames)) * vectors;
            uint gainStart = band * vectors;
            uint outputStart = (band * rows + row) * vectors;
            float4 values[4];
            float total = 0.0f;
            for (uint column = 0; column < 4; ++column) {
                uint index = lane + 32 * column;
                values[column] = fma(input[inputStart + index], scale[index], shift[index]);
                total += dot(values[column], float4(1.0f));
            }
            float mean = simd_sum(total) / float(features);
            float sq = 0.0f;
            for (uint column = 0; column < 4; ++column) {
                values[column] -= mean;
                sq += dot(values[column], values[column]);
            }
            float rs = rsqrt(simd_sum(sq) / float(features) + 1e-5f);
            for (uint column = 0; column < 4; ++column) {
                uint index = lane + 32 * column;
                output[outputStart + index] = half4(fma(values[column] * rs, gain[gainStart + index], bias[gainStart + index]));
            }
        }

        float gelu(float x) {
            // Approximate the complementary error function to within 1.5e-7.
            float z = abs(x) * 0.7071067811865475f;
            float t = 1.0f / (1.0f + 0.3275911f * z);
            float q = fma(fma(fma(fma(1.061405429f, t, -1.453152027f), t, 1.421413741f),
                              t, -0.284496736f), t, 0.254829592f) * t * exp(-z * z);
            return x * (x < 0 ? 0.5f * q : 1.0f - 0.5f * q);
        }

        #if DECODER_ROWS == 4
        template <typename V>
        float4 product(device const V *input, device const V *weights, uint rows, uint lane) {
            float4 accum = 0;
            for (uint column = lane; column < vectors; column += 32) {
                float4 weight = float4(weights[column]);
                accum.x += dot(float4(input[column]), weight);
                accum.y += dot(float4(input[vectors + column]), weight);
                if (rows == 4) {
                    accum.z += dot(float4(input[2 * vectors + column]), weight);
                    accum.w += dot(float4(input[3 * vectors + column]), weight);
                }
            }
            return simd_sum(accum);
        }
        #else
        template <typename V>
        inline void product(device const V *input, device const V *weights, uint rows, uint lane,
                            thread float *accum) {
            #pragma unroll
            for (uint row = 0; row < DECODER_ROWS; ++row) accum[row] = 0.0f;
            for (uint column = lane; column < vectors; column += 32) {
                float4 weight = float4(weights[column]);
                #pragma unroll
                for (uint row = 0; row < DECODER_ROWS; ++row) {
                    if (row < rows) accum[row] += dot(float4(input[row * vectors + column]), weight);
                }
            }
            #pragma unroll
            for (uint row = 0; row < DECODER_ROWS; ++row) {
                if (row < rows) accum[row] = simd_sum(accum[row]);
            }
        }
        #endif

        kernel void expand_bands(device const half4 *input [[buffer(0)]],
                                 device const half4 *weights [[buffer(1)]],
                                 device const half *bias [[buffer(2)]],
                                 device float *output [[buffer(3)]],
                                 constant uint &rows [[buffer(4)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
            uint band = group.y;
            uint column = group.x * groups + simd;
            uint weightRow = band * features + column;
            #if DECODER_ROWS == 4
            float4 value = product(input + band * rows * vectors, weights + weightRow * vectors, rows, lane);
            #else
            float value[DECODER_ROWS];
            product(input + band * rows * vectors, weights + weightRow * vectors, rows, lane, value);
            #endif
            if (lane == 0) {
                for (uint row = 0; row < rows; ++row) {
                    output[(band * rows + row) * features + column] = gelu(float(half(value[row] + float(bias[weightRow]))));
                }
            }
        }

        kernel void project_bands(device const float4 *input [[buffer(0)]],
                                  device const float4 *weights [[buffer(1)]],
                                  device const float *bias [[buffer(2)]],
                                  device const uchar *bandIndices [[buffer(3)]],
                                  device float *output [[buffer(4)]],
                                  constant uint &rows [[buffer(5)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
            uint column = group * groups + simd;
            if (column >= channels) {
                return;
            }
            uint band = bandIndices[column];
            #if DECODER_ROWS == 4
            float4 value = product(input + band * rows * vectors, weights + column * vectors, rows, lane);
            #else
            float value[DECODER_ROWS];
            product(input + band * rows * vectors, weights + column * vectors, rows, lane, value);
            #endif
            if (lane == 0) {
                for (uint row = 0; row < rows; ++row) {
                    output[row * channels + column] = value[row] + bias[column];
                }
            }
        }
        """
    }
}
