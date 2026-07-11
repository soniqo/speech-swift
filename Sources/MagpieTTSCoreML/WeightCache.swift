import Foundation

/// Disk cache for the extracted LT + audio_embedding weights.
///
/// The first time ``MagpieTTSCoreML.fromPretrained()`` runs, we load the
/// MLX MagpieTTS module (~250 MB safetensors → MLX arrays → ~3–4 s of
/// init), snapshot the weights we need, and serialize them to a flat
/// `.bin` file. Every subsequent process load reads the `.bin` directly
/// — no MLX module load, no per-tensor MLX-to-Swift copies.
///
/// File layout: one big binary blob holding all the tensors back-to-back,
/// preceded by an 8-byte little-endian magic + version + a JSON header
/// describing each tensor's offset and shape. Self-describing so we can
/// extend it (e.g. add LT-norm bias when NeMo adds one) without breaking
/// existing caches.
enum MagpieCoreMLWeightCache {

    private static let magic: UInt64 = 0x4D41_4750_4945_5743  // "MAGPIEWC"
    private static let version: UInt32 = 1

    struct Manifest: Codable {
        struct Entry: Codable {
            let name: String
            let shape: [Int]
            let offset: Int        // byte offset from start of payload
            let elementCount: Int  // Float32 elements; bytes = count * 4
        }
        var entries: [Entry]
    }

    /// Load cached weights. Returns `nil` if the cache file is missing or
    /// has an unexpected layout. Callers should fall back to MLX
    /// extraction in that case.
    static func load(from url: URL) -> (lt: MagpieCoreMLLocalTransformerWeights,
                                          audioEmbeds: [[Float]])? {
        guard FileManager.default.fileExists(atPath: url.path),
              let data = try? Data(contentsOf: url, options: [.mappedIfSafe]) else {
            return nil
        }
        do {
            return try parse(data: data)
        } catch {
            // Cache is corrupt or from an incompatible version. Caller
            // will re-extract from MLX.
            return nil
        }
    }

    /// Write the extracted weights + audio embedding tables to the cache
    /// file. Atomic write via temp + rename so a crashed process doesn't
    /// leave a half-written file.
    static func save(to url: URL,
                      lt: MagpieCoreMLLocalTransformerWeights,
                      audioEmbeds: [[Float]]) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let blob = try encode(lt: lt, audioEmbeds: audioEmbeds)
        let tmp = url.appendingPathExtension("tmp-\(UUID().uuidString)")
        try blob.write(to: tmp, options: [.atomic])
        try FileManager.default.moveItem(at: tmp, to: url)
    }

    // MARK: - Codec

    private static func encode(
        lt: MagpieCoreMLLocalTransformerWeights,
        audioEmbeds: [[Float]]
    ) throws -> Data {
        let K = audioEmbeds.count
        var entries: [Manifest.Entry] = []
        var payload = Data()

        func push(_ name: String, _ floats: [Float], shape: [Int]) {
            let byteCount = floats.count * MemoryLayout<Float>.size
            let entry = Manifest.Entry(
                name: name, shape: shape,
                offset: payload.count,
                elementCount: floats.count)
            entries.append(entry)
            // `Array.withUnsafeBytes` gives a raw buffer covering ALL the
            // Float bytes — `Data(_:)` copies them in one shot. The earlier
            // `withMemoryRebound(to: UInt8.self)` path had a count bug
            // (used Float count instead of byte count) and wrote 1/4 of
            // the data, leading to short cache files + SIGSEGV on read.
            floats.withUnsafeBytes { raw in
                payload.append(Data(raw))
            }
            assert(payload.count == entry.offset + byteCount,
                   "\(name): expected \(byteCount) bytes appended, got \(payload.count - entry.offset)")
        }

        push("lt.in_proj.weight", lt.inProjWeight, shape: [lt.localDim, lt.dModel])
        push("lt.in_proj.bias",   lt.inProjBias,   shape: [lt.localDim])
        push("lt.pos_embedding",  lt.posEmbedding, shape: [lt.maxPositions, lt.localDim])
        push("lt.norm1.weight",   lt.norm1Weight,  shape: [lt.localDim])
        push("lt.norm2.weight",   lt.norm2Weight,  shape: [lt.localDim])
        push("lt.sa.qkv.weight",  lt.saQkvWeight,  shape: [3 * lt.localDim, lt.localDim])
        push("lt.sa.o.weight",    lt.saOWeight,    shape: [lt.localDim, lt.localDim])
        push("lt.ffn.conv1.weight", lt.ffnConv1Weight, shape: [lt.ffnDim, lt.localDim])
        push("lt.ffn.conv2.weight", lt.ffnConv2Weight, shape: [lt.localDim, lt.ffnDim])
        for cb in 0..<lt.numCodebooks {
            push("lt.out_proj.\(cb).weight",
                 lt.outProjWeights[cb],
                 shape: [lt.numCodesPerCodebook, lt.localDim])
            push("lt.out_proj.\(cb).bias",
                 lt.outProjBiases[cb],
                 shape: [lt.numCodesPerCodebook])
        }
        for k in 0..<K {
            push("audio_embedding.\(k)",
                 audioEmbeds[k],
                 shape: [MagpieCoreMLConstants.numCodesPerCodebook,
                         MagpieCoreMLConstants.dModel])
        }

        let manifest = Manifest(entries: entries)
        let manifestJSON = try JSONEncoder().encode(manifest)

        // File: magic(8) + version(4) + headerLen(4) + headerJSON + payload
        var out = Data()
        var magic = Self.magic.littleEndian
        var version = Self.version.littleEndian
        var headerLen = UInt32(manifestJSON.count).littleEndian
        out.append(Data(bytes: &magic, count: 8))
        out.append(Data(bytes: &version, count: 4))
        out.append(Data(bytes: &headerLen, count: 4))
        out.append(manifestJSON)
        out.append(payload)
        return out
    }

    private static func parse(data: Data) throws
        -> (lt: MagpieCoreMLLocalTransformerWeights, audioEmbeds: [[Float]])
    {
        guard data.count >= 16 else { throw Err.shortFile }
        let magicRead = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        guard UInt64(littleEndian: magicRead) == Self.magic else { throw Err.badMagic }
        let versionRead = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self) }
        guard UInt32(littleEndian: versionRead) == Self.version else { throw Err.badVersion }
        let headerLen = Int(UInt32(littleEndian: data.withUnsafeBytes {
            $0.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
        }))
        let headerStart = 16
        let headerEnd = headerStart + headerLen
        guard headerEnd <= data.count else { throw Err.shortHeader }
        let manifestJSON = data.subdata(in: headerStart..<headerEnd)
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestJSON)
        let payload = data.subdata(in: headerEnd..<data.count)

        var byName: [String: [Float]] = [:]
        for entry in manifest.entries {
            let byteCount = entry.elementCount * MemoryLayout<Float>.size
            let end = entry.offset + byteCount
            guard end <= payload.count else { throw Err.shortPayload(entry.name) }
            let slice = payload.subdata(in: entry.offset..<end)
            let floats: [Float] = slice.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float.self)
                return Array(p)
            }
            byName[entry.name] = floats
        }

        // Decode LT.
        let localDim = MagpieCoreMLConstants.localTransformerDim
        let dModel = MagpieCoreMLConstants.dModel
        let ffnDim = MagpieCoreMLConstants.localTransformerFfnDim
        let maxPos = MagpieCoreMLConstants.localTransformerMaxPositions
        let K = MagpieCoreMLConstants.numCodebooks
        let V = MagpieCoreMLConstants.numCodesPerCodebook

        func required(_ name: String) throws -> [Float] {
            guard let v = byName[name] else { throw Err.missing(name) }
            return v
        }

        var outProjW: [[Float]] = []
        var outProjB: [[Float]] = []
        for cb in 0..<K {
            outProjW.append(try required("lt.out_proj.\(cb).weight"))
            outProjB.append(try required("lt.out_proj.\(cb).bias"))
        }

        let lt = MagpieCoreMLLocalTransformerWeights(
            inProjWeight: try required("lt.in_proj.weight"),
            inProjBias: try required("lt.in_proj.bias"),
            posEmbedding: try required("lt.pos_embedding"),
            norm1Weight: try required("lt.norm1.weight"),
            norm2Weight: try required("lt.norm2.weight"),
            saQkvWeight: try required("lt.sa.qkv.weight"),
            saOWeight: try required("lt.sa.o.weight"),
            ffnConv1Weight: try required("lt.ffn.conv1.weight"),
            ffnConv2Weight: try required("lt.ffn.conv2.weight"),
            outProjWeights: outProjW, outProjBiases: outProjB,
            localDim: localDim, dModel: dModel, ffnDim: ffnDim,
            maxPositions: maxPos, numCodebooks: K, numCodesPerCodebook: V)

        var embeds: [[Float]] = []
        for k in 0..<K {
            embeds.append(try required("audio_embedding.\(k)"))
        }
        return (lt, embeds)
    }

    enum Err: Error {
        case shortFile, badMagic, badVersion, shortHeader
        case shortPayload(String)
        case missing(String)
    }
}
