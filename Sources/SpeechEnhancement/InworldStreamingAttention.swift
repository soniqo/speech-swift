import CoreML
import Foundation
import Metal

/// Causal attention over a shared cache of Float16 or packed keys and values.
final class InworldStreamingAttention {
    private static let bands = InworldVoiceIsolator.bands
    private static let heads = 8
    private static let width = InworldVoiceIsolator.hiddenWidth
    private static let capacity = InworldStreamingLayout.laneCacheFrames
    // The retiring lane can contain 800 frames while its successor holds 200.
    private static let poolCapacity = capacity + InworldStreamingLayout.crossfadeFrames
    private static let cachePageFrames = 125
    private static let cachePageShift = 7
    private static let maxLanes = InworldStreamingLayout.pairedLanes
    private static let groups = 8

    enum CacheEncoding: Int, Decodable {
        case float16 = 16
        case packed12 = 12
        case packed10 = 10

        // Each head contains 64 components. The packed layouts store Float32 scales.
        var bytesPerHead: Int {
            switch self {
            case .float16: 128
            case .packed12: 100
            case .packed10: 88
            }
        }
    }

    enum Cache {
        case contiguous(keys: MTLBuffer, values: MTLBuffer)
        case paged(CachePages)
    }

    final class CachePages {
        let addresses: MTLBuffer
        var buffers: [MTLResource] = []

        init(addresses: MTLBuffer) { self.addresses = addresses }
    }

    private let device: MTLDevice
    private let keyEncoding: CacheEncoding
    private let valueEncoding: CacheEncoding
    private let queue: MTLCommandQueue
    private let attend: MTLComputePipelineState
    private let rotary: MTLBuffer
    private let qkv: MTLBuffer
    private let encoderQKV: MTLBuffer?
    private let attended: MTLBuffer
    private let cacheSlots: MTLBuffer
    private let slots: UnsafeMutablePointer<UInt16>
    private let usesPagedCache: Bool
    private var requiredPageCount = 0
    private var freeSlots: [UInt16]

    init(keyEncoding: CacheEncoding, valueEncoding: CacheEncoding, blockFrames: Int, serialOverlap: Bool) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw InworldIsolationError.predictionFailed("a Metal device is required for streaming attention")
        }
        let usesPagedCache = serialOverlap && device.argumentBuffersSupport == .tier2
        self.usesPagedCache = usesPagedCache
        let slots = (0..<Self.poolCapacity).reversed()
        self.freeSlots = usesPagedCache ? slots.map(Self.pagedSlot(for:)) : slots.map(UInt16.init)
        let library = try device.makeLibrary(
            source: Self.source(keyEncoding: keyEncoding, valueEncoding: valueEncoding, blockFrames: blockFrames,
                                pagedCache: usesPagedCache),
            options: nil
        )
        let graphLanes = serialOverlap ? 1 : Self.maxLanes
        let bytes = graphLanes * Self.bands * blockFrames * Self.width * MemoryLayout<Float>.stride
        guard let attend = library.makeFunction(name: "attend_block"),
              let qkv = device.makeBuffer(length: 3 * bytes, options: .storageModeShared),
              let attended = device.makeBuffer(length: bytes, options: .storageModeShared),
              let cacheSlots = device.makeBuffer(length: Self.maxLanes * Self.capacity * MemoryLayout<UInt16>.stride,
                                                options: .storageModeShared) else {
            throw InworldIsolationError.predictionFailed("could not initialize streaming attention")
        }
        let rates = (0..<32).map { pow(10_000, -Float($0) / 32) }
        let table = (0..<Self.capacity).flatMap { frame in
            let angles = rates.map { Float(frame) * $0 }
            return angles.map(cos) + angles.map(sin)
        }
        guard let rotary = table.withUnsafeBytes({ raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        }) else {
            throw InworldIsolationError.predictionFailed("could not allocate rotary tables")
        }
        self.device = device
        self.keyEncoding = keyEncoding
        self.valueEncoding = valueEncoding
        self.queue = queue
        self.attend = try device.makeComputePipelineState(function: attend)
        self.rotary = rotary
        self.qkv = qkv
        if serialOverlap {
            // Keep the encoder projection live while both lanes use the same workspaces.
            guard let encoderQKV = device.makeBuffer(length: 3 * bytes, options: .storageModeShared) else {
                throw InworldIsolationError.predictionFailed("could not allocate the streaming encoder projection")
            }
            self.encoderQKV = encoderQKV
        } else {
            self.encoderQKV = nil
        }
        self.attended = attended
        self.cacheSlots = cacheSlots
        self.slots = cacheSlots.contents().bindMemory(to: UInt16.self, capacity: Self.maxLanes * Self.capacity)
    }

    func makeCache() throws -> Cache {
        if usesPagedCache {
            // The Metal argument buffer has two GPU addresses per page.
            let addressCount = 2 * Self.cachePageCount(for: Self.poolCapacity)
            guard let addresses = device.makeBuffer(length: addressCount * MemoryLayout<UInt64>.stride,
                                                    options: .storageModeShared) else {
                throw InworldIsolationError.predictionFailed("could not allocate the streaming KV page table")
            }
            return .paged(CachePages(addresses: addresses))
        }
        let buffers = try allocateCacheBuffers(frames: Self.poolCapacity)
        return .contiguous(keys: buffers.keys, values: buffers.values)
    }

    static func cachePageCount(for slotCount: Int) -> Int {
        (slotCount + cachePageFrames - 1) / cachePageFrames
    }

    // Encode the page and its offset once so Metal can use a shift and a mask per lookup.
    static func pagedSlot(for slot: Int) -> UInt16 {
        UInt16(((slot / cachePageFrames) << cachePageShift) | (slot % cachePageFrames))
    }

    private func allocateCacheBuffers(frames: Int) throws -> (keys: MTLBuffer, values: MTLBuffer) {
        let vectors = Self.bands * Self.heads * frames
        guard let keys = device.makeBuffer(length: vectors * keyEncoding.bytesPerHead, options: .storageModePrivate),
              let values = device.makeBuffer(length: vectors * valueEncoding.bytesPerHead, options: .storageModePrivate) else {
            throw InworldIsolationError.predictionFailed("could not allocate the streaming KV cache")
        }
        return (keys, values)
    }

    private func allocatePages(_ cache: CachePages) throws {
        let allocated = cache.buffers.count / 2
        guard allocated < requiredPageCount else { return }
        let addresses = cache.addresses.contents().bindMemory(to: UInt64.self,
            capacity: 2 * Self.cachePageCount(for: Self.poolCapacity))
        for page in allocated..<requiredPageCount {
            let buffers = try allocateCacheBuffers(frames: Self.cachePageFrames)
            addresses[2 * page] = buffers.keys.gpuAddress
            addresses[2 * page + 1] = buffers.values.gpuAddress
            cache.buffers.append(buffers.keys)
            cache.buffers.append(buffers.values)
        }
    }

    /// All attention layers share this map from live lane positions to physical cache slots.
    func reserve(lanes: [Int], positions: [Int], frames: Int) throws {
        guard positions.allSatisfy({ $0 + frames <= Self.capacity }),
              freeSlots.count >= lanes.count * frames else {
            throw InworldIsolationError.predictionFailed("the streaming attention cache is full")
        }
        for (lane, first) in zip(lanes, positions) {
            let map = slots + lane * Self.capacity
            for position in first..<(first + frames) {
                let slot = freeSlots.removeLast()
                map[position] = slot
                if usesPagedCache {
                    requiredPageCount = max(requiredPageCount, (Int(slot) >> Self.cachePageShift) + 1)
                }
            }
        }
    }

    func retire(lane: Int, frames: Int) {
        let released = UnsafeBufferPointer(start: slots + lane * Self.capacity, count: frames)
        freeSlots.append(contentsOf: released.reversed())
    }

    func projectedArray(lanes: Int, frames: Int, encoderProjection: Bool = false) throws -> MLMultiArray {
        try array(projectionBuffer(encoderProjection), lanes: lanes, frames: frames, width: 3 * Self.width)
    }

    func run(_ projected: MLMultiArray, frames: Int, validFrames: Int, slots: [Int], positions: [Int],
             cache: Cache, backing: MLMultiArray, modelLanes: Int, encoderProjection: Bool) throws -> MLMultiArray {
        if case .paged(let pages) = cache { try allocatePages(pages) }
        guard positions.allSatisfy({ $0 + validFrames <= Self.capacity }), let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            throw InworldIsolationError.predictionFailed("could not prepare streaming attention")
        }
        if projected !== backing { projected.transfer(to: backing) }
        let rows = Self.bands * Self.heads
        var positionBySlot = [0, 0]
        var activeMask = 0
        for (slot, position) in zip(slots, positions) {
            activeMask |= 1 << slot
            positionBySlot[slot] = position
        }
        let options = [frames, activeMask, backing.shape[0].intValue, positionBySlot[0],
                       positionBySlot[1], validFrames, modelLanes == 1 ? slots[0] : 0].map(UInt32.init)
        encoder.setComputePipelineState(attend)
        encoder.setBuffer(projectionBuffer(encoderProjection), offset: 0, index: 0)
        switch cache {
        case .contiguous(let keys, let values):
            encoder.setBuffer(keys, offset: 0, index: 1)
            encoder.setBuffer(values, offset: 0, index: 2)
        case .paged(let pages):
            encoder.setBuffer(pages.addresses, offset: 0, index: 1)
            encoder.useResources(pages.buffers, usage: [.read, .write])
        }
        encoder.setBuffer(rotary, offset: 0, index: 3)
        options.withUnsafeBytes { encoder.setBytes($0.baseAddress!, length: $0.count, index: 4) }
        encoder.setBuffer(attended, offset: 0, index: 5)
        encoder.setBuffer(cacheSlots, offset: 0, index: 6)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: modelLanes),
                                    threadsPerThreadgroup: MTLSize(width: Self.groups * 32, height: 1, depth: 1))
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
            throw InworldIsolationError.predictionFailed(command.error?.localizedDescription ?? "Metal attention failed")
        }
        return try array(attended, lanes: modelLanes, frames: frames, width: Self.width)
    }

    private func projectionBuffer(_ encoderProjection: Bool) -> MTLBuffer {
        encoderProjection ? encoderQKV ?? qkv : qkv
    }

    private func array(_ buffer: MTLBuffer, lanes: Int, frames: Int, width: Int) throws -> MLMultiArray {
        try MLMultiArray(
            dataPointer: buffer.contents(),
            shape: [lanes, Self.bands, frames, width].map { NSNumber(value: $0) },
            dataType: .float32,
            strides: [Self.bands * frames * width, frames * width, width, 1].map { NSNumber(value: $0) },
            deallocator: { [buffer] _ in withExtendedLifetime(buffer) {} })
    }

    private static func source(keyEncoding: CacheEncoding, valueEncoding: CacheEncoding, blockFrames: Int,
                               pagedCache: Bool) -> String {
        """
        #include <metal_stdlib>
        using namespace metal;

        constant uint heads = \(heads);
        constant uint width = \(width);
        constant uint capacity = \(capacity);
        constant uint physical_capacity = \(pagedCache ? cachePageFrames : poolCapacity);
        #define KV_GROUPS \(groups)
        #define PAGED_CACHE \(pagedCache ? 1 : 0)
        #define PAGE_SHIFT \(cachePageShift)
        #define CACHE_BITS \(valueEncoding.rawValue)
        #define KEY_BITS \(keyEncoding.rawValue)
        #define BLOCK_FRAMES \(blockFrames)
        constant uint cache_rows = \(bands * heads);

        #if KEY_BITS == 10
        struct PackedKeys {
            ushort low[32];
            uchar high[16];
            float scale[2];
        };
        static_assert(sizeof(PackedKeys) == \(keyEncoding.bytesPerHead), "incorrect packed key stride");

        inline void put(device PackedKeys *buffer, uint vector, ushort lane, float2 value) {
            // Each half of the SIMD group quantizes 32 components together.
            float maximum = max(abs(value.x), abs(value.y));
            for (ushort mask = 1; mask < 16; mask *= 2) {
                maximum = max(maximum, simd_shuffle_xor(maximum, mask));
            }
            const float scale = max(maximum, 1.0e-9f) / 511.0f;
            const uint2 bits = uint2(int2(clamp(rint(value / scale), -511.0f, 511.0f)) + 512);
            buffer[vector].low[lane] = ushort((bits.x & 255u) | ((bits.y & 255u) << 8));
            const uint upper = (bits.x >> 8) | ((bits.y >> 8) << 2);
            const uint neighbor = simd_shuffle_xor(upper, 1);
            if ((lane & 1) == 0) buffer[vector].high[lane >> 1] = uchar(upper | (neighbor << 4));
            if ((lane & 15) == 0) buffer[vector].scale[lane >> 4] = scale;
        }

        inline float2 get(device const PackedKeys *buffer, uint vector, ushort lane) {
            const uint low = buffer[vector].low[lane];
            const uint upper = uint(buffer[vector].high[lane >> 1]) >> ((lane & 1) * 4);
            const uint first = (low & 255u) | ((upper & 3u) << 8);
            const uint second = (low >> 8) | ((upper & 12u) << 6);
            return float2(int(first) - 512, int(second) - 512) * buffer[vector].scale[lane >> 4];
        }
        #endif

        #if CACHE_BITS == 12
        struct PackedPair {
            uchar low0;
            uchar low1;
            uchar high;
        };

        typedef PackedPair CacheElement;

        inline void put(device CacheElement *buffer, uint vector, ushort lane, float2 value) {
            const float scale = max(simd_max(max(abs(value.x), abs(value.y))), 1.0e-9f) / 2047.0f;
            const uint2 bits = uint2(int2(clamp(rint(value / scale), -2047.0f, 2047.0f)) + 2048);
            buffer[vector * 32 + lane] = PackedPair{uchar(bits.x), uchar(bits.y),
                                                   uchar((bits.x >> 8) | ((bits.y >> 8) << 4))};
            if (lane == 0) {
                reinterpret_cast<device float *>(buffer + cache_rows * physical_capacity * 32)[vector] = scale;
            }
        }

        inline float2 get(device const CacheElement *buffer, uint vector, ushort lane) {
            const PackedPair packed = buffer[vector * 32 + lane];
            const int first = int(packed.low0) | (int(packed.high & 15) << 8);
            const int second = int(packed.low1) | (int(packed.high >> 4) << 8);
            const float scale = reinterpret_cast<device const float *>(buffer + cache_rows * physical_capacity * 32)[vector];
            return float2(first - 2048, second - 2048) * scale;
        }
        #else
        typedef half CacheElement;

        inline void put(device CacheElement *buffer, uint vector, ushort lane, float2 value) {
            buffer[vector * 64 + lane] = half(value.x);
            buffer[vector * 64 + lane + 32] = half(value.y);
        }

        inline float2 get(device const CacheElement *buffer, uint vector, ushort lane) {
            return float2(buffer[vector * 64 + lane], buffer[vector * 64 + lane + 32]);
        }
        #endif

        #if KEY_BITS == 10
        typedef PackedKeys KeyElement;
        #else
        typedef CacheElement KeyElement;
        #endif

        struct CachePage {
            device KeyElement *keys;
            device CacheElement *values;
        };

        #if PAGED_CACHE
        #define KV_PAGE(slot) pages[(slot) >> PAGE_SHIFT]
        #define KV_VECTOR(slot) (cache + ((slot) & ((1u << PAGE_SHIFT) - 1u)))
        #else
        #define KV_PAGE(slot) contiguous
        #define KV_VECTOR(slot) (cache + (slot))
        #endif

        inline float2 rotate(float2 value, float2 angle) {
            return float2(value.x * angle.x - value.y * angle.y, value.x * angle.y + value.y * angle.x);
        }

        struct OnlineSoftmax {
            float peak;
            float denominator;
            float2 numerator;
        };

        inline void add_value(thread OnlineSoftmax &state, float2 query, float2 key, float2 value) {
            const float score = 0.125f * simd_sum(query.x * key.x + query.y * key.y);
            if (score > state.peak) {
                const float scale = state.denominator == 0.0f ? 0.0f : fast::exp(state.peak - score);
                state.denominator = state.denominator * scale + 1.0f;
                state.numerator = state.numerator * scale + value;
                state.peak = score;
            } else {
                const float weight = fast::exp(score - state.peak);
                state.denominator += weight;
                state.numerator += weight * value;
            }
        }

        kernel void attend_block(
            device const float *qkv [[buffer(0)]],
            #if PAGED_CACHE
            constant CachePage *pages [[buffer(1)]],
            #else
            device KeyElement *keys [[buffer(1)]],
            device CacheElement *values [[buffer(2)]],
            #endif
            device const float *rotary [[buffer(3)]],
            constant uint *options [[buffer(4)]],
            device float *output [[buffer(5)]],
            constant ushort *slots [[buffer(6)]],
            uint3 group [[threadgroup_position_in_grid]],
            ushort lane [[thread_index_in_simdgroup]], ushort part [[simdgroup_index_in_threadgroup]]) {
            const uint row = group.x, logical = group.z + options[6], frames = options[0];
            const uint column = row % heads * 64 + lane;
            const uint target = ((group.z * cache_rows + row) / heads * frames) * width + column;
            const uint visible = options[1] & (1u << logical) ? options[5] : 0;
            if (visible == 0) {
                if (part == 0) {
                    for (uint frame = 0; frame < frames; ++frame) {
                        output[target + frame * width] = 0.0f;
                        output[target + frame * width + 32] = 0.0f;
                    }
                }
                return;
            }

            const uint first = options[3 + logical];
            const uint projected = (options[2] == 1 ? 0 : group.z * cache_rows) + row;
            const uint token = projected / heads * frames * 3 * width + column;
            constant ushort *indices = slots + logical * capacity;
            const uint cache = row * physical_capacity;
            #if !PAGED_CACHE
            const CachePage contiguous = {keys, values};
            #endif
            float2 queries[BLOCK_FRAMES];
            float2 current[BLOCK_FRAMES];
            for (uint frame = 0; frame < visible; ++frame) {
                const uint offset = token + frame * 3 * width;
                const uint pos = first + frame;
                const float2 angle = float2(rotary[pos * 64 + lane], rotary[pos * 64 + 32 + lane]);
                queries[frame] = rotate(float2(qkv[offset], qkv[offset + 32]), angle);
                current[frame] = float2(qkv[offset + 2 * width], qkv[offset + 2 * width + 32]);
                if (part == (BLOCK_FRAMES == 2 ? 0 : frame)) {
                    const float2 key = rotate(float2(qkv[offset + width], qkv[offset + width + 32]), angle);
                    const uint slot = uint(indices[pos]);
                    const uint vector = KV_VECTOR(slot);
                    const CachePage page = KV_PAGE(slot);
                    put(page.keys, vector, lane, key);
                    put(page.values, vector, lane, current[frame]);
                }
            }

            // This group is the sole writer of this head and lane's new cache rows.
            threadgroup_barrier(mem_flags::mem_device);

            threadgroup float peaks[BLOCK_FRAMES][KV_GROUPS], weights[BLOCK_FRAMES][KV_GROUPS];
            threadgroup float2 sums[BLOCK_FRAMES][KV_GROUPS][32];

            #if BLOCK_FRAMES == 2
            OnlineSoftmax left = {-INFINITY, 0.0f, float2(0.0f)};
            OnlineSoftmax right = {-INFINITY, 0.0f, float2(0.0f)};
            for (uint i = part; i <= first; i += KV_GROUPS) {
                const uint slot = uint(indices[i]);
                const uint vector = KV_VECTOR(slot);
                const CachePage page = KV_PAGE(slot);
                const float2 key = get(page.keys, vector, lane);
                const float2 value = get(page.values, vector, lane);
                add_value(left, queries[0], key, value);
                if (visible == 2) add_value(right, queries[1], key, value);
            }
            if (visible == 2 && (first + 1) % KV_GROUPS == part) {
                const uint slot = uint(indices[first + 1]);
                const uint vector = KV_VECTOR(slot);
                const CachePage page = KV_PAGE(slot);
                add_value(right, queries[1], get(page.keys, vector, lane), get(page.values, vector, lane));
            }

            if (lane == 0) {
                peaks[0][part] = left.peak;
                peaks[1][part] = right.peak;
                weights[0][part] = left.denominator;
                weights[1][part] = right.denominator;
            }
            sums[0][part][lane] = left.numerator;
            sums[1][part][lane] = right.numerator;
            #else
            OnlineSoftmax states[BLOCK_FRAMES];
            #pragma unroll
            for (uint frame = 0; frame < BLOCK_FRAMES; ++frame) {
                states[frame] = OnlineSoftmax{-INFINITY, 0.0f, float2(0.0f)};
            }
            for (uint i = part; i < first + visible; i += KV_GROUPS) {
                const uint slot = uint(indices[i]);
                const uint vector = KV_VECTOR(slot);
                const CachePage page = KV_PAGE(slot);
                const float2 key = get(page.keys, vector, lane);
                const float2 value = get(page.values, vector, lane);
                #pragma unroll
                for (uint frame = 0; frame < BLOCK_FRAMES; ++frame) {
                    if (frame < visible && i <= first + frame) {
                        add_value(states[frame], queries[frame], key, value);
                    }
                }
            }
            for (uint frame = 0; frame < visible; ++frame) {
                if (lane == 0) {
                    peaks[frame][part] = states[frame].peak;
                    weights[frame][part] = states[frame].denominator;
                }
                sums[frame][part][lane] = states[frame].numerator;
            }
            #endif
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (BLOCK_FRAMES != 2 || part == 0) {
                const uint firstFrame = BLOCK_FRAMES == 2 ? 0 : part;
                const uint stride = BLOCK_FRAMES == 2 ? 1 : KV_GROUPS;
                for (uint frame = firstFrame; frame < visible; frame += stride) {
                    float maximum = -INFINITY;
                    for (uint i = 0; i < KV_GROUPS; ++i) maximum = max(maximum, peaks[frame][i]);
                    float2 combined = float2(0.0f);
                    float denominator = 0.0f;
                    for (uint i = 0; i < KV_GROUPS; ++i) {
                        const float scale = weights[frame][i] == 0.0f ? 0.0f : fast::exp(peaks[frame][i] - maximum);
                        denominator += weights[frame][i] * scale;
                        combined += sums[frame][i][lane] * scale;
                    }
                    combined /= denominator;
                    const float2 value = current[frame];
                    const float coefficient = simd_sum(combined.x * value.x + combined.y * value.y)
                        / max(1.0e-6f, simd_sum(value.x * value.x + value.y * value.y));
                    const float2 result = combined - coefficient * value;
                    output[target + frame * width] = result.x;
                    output[target + frame * width + 32] = result.y;
                }
                #if BLOCK_FRAMES == 2
                for (uint frame = visible; frame < frames; ++frame) {
                    output[target + frame * width] = 0.0f;
                    output[target + frame * width + 32] = 0.0f;
                }
                #else
                if (part >= visible && part < frames) {
                    output[target + part * width] = 0.0f;
                    output[target + part * width + 32] = 0.0f;
                }
                #endif
            }
        }
        """
    }
}
