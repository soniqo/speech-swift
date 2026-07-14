import Foundation
import MLXCommon
import AudioCommon

/// Weight loading for DeepFilterNet3 Core ML model.
///
/// Loads the pre-compiled ``.mlmodelc`` model and auxiliary signal processing
/// data (ERB filterbank, Vorbis window, normalization states) from an npz file.
enum DeepFilterNet3WeightLoader {

    /// Load Core ML model and auxiliary data.
    ///
    /// Only the pre-compiled ``DeepFilterNet3.mlmodelc`` is supported — on-device
    /// ``MLModel.compileModel`` from ``.mlpackage`` is known to drift per
    /// runtime (Mac vs simulator vs iPhone). Users upgrading with a stale
    /// ``.mlpackage``-only cache trigger a transparent re-download of the
    /// compiled bundle because ``.mlmodelc`` is missing and the Hub fetch
    /// glob targets it.
    ///
    /// - Parameters:
    ///   - directory: Directory containing DeepFilterNet3.mlmodelc and
    ///     auxiliary.npz
    /// - Returns: `(network, auxiliaryData)`
    static func load(from directory: URL) throws -> (DeepFilterNet3Network, AuxiliaryData) {
        let modelURL = directory.appendingPathComponent("DeepFilterNet3.mlmodelc")
        let auxURL = directory.appendingPathComponent("auxiliary.npz")

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw WeightLoadingError.noWeightsFound(directory)
        }
        guard FileManager.default.fileExists(atPath: auxURL.path) else {
            throw WeightLoadingError.missingRequiredWeight(
                "auxiliary.npz not found in \(directory.path)")
        }

        let network = try DeepFilterNet3Network(modelURL: modelURL)
        let auxData = try loadAuxiliaryData(from: auxURL)

        return (network, auxData)
    }

    /// Parse the auxiliary.npz file containing signal processing constants.
    private static func loadAuxiliaryData(from url: URL) throws -> AuxiliaryData {
        let arrays = try NpzReader.read(url: url)

        guard let erbFb = arrays["erb_fb"],
              let erbInvFb = arrays["erb_inv_fb"],
              let window = arrays["window"],
              let meanNormState = arrays["mean_norm_state"],
              let unitNormState = arrays["unit_norm_state"] else {
            throw WeightLoadingError.missingRequiredWeight(
                "Missing keys in auxiliary.npz")
        }

        return AuxiliaryData(
            erbFb: erbFb,
            erbInvFb: erbInvFb,
            window: window,
            meanNormState: meanNormState,
            unitNormState: unitNormState
        )
    }

    /// Auxiliary data loaded alongside Core ML model.
    struct AuxiliaryData {
        /// Forward ERB filterbank [freqBins, erbBands]
        let erbFb: [Float]
        /// Inverse ERB filterbank [erbBands, freqBins]
        let erbInvFb: [Float]
        /// Vorbis analysis/synthesis window [fftSize]
        let window: [Float]
        /// Initial mean normalization state [erbBands]
        let meanNormState: [Float]
        /// Initial unit normalization state [dfBins]
        let unitNormState: [Float]
    }
}

// MARK: - NPZ Reader

/// Minimal reader for NumPy .npz files (uncompressed, float32 only).
enum NpzReader {

    static func read(url: URL) throws -> [String: [Float]] {
        let data = try Data(contentsOf: url)
        var result = [String: [Float]]()

        // NPZ is a ZIP archive of .npy files
        var offset = 0
        while offset + 30 <= data.count {
            // ZIP local file header signature
            let b0 = data[offset], b1 = data[offset+1], b2 = data[offset+2], b3 = data[offset+3]
            guard b0 == 0x50 && b1 == 0x4b && b2 == 0x03 && b3 == 0x04 else { break }

            var compressedSize = Int(readUInt32(data, at: offset + 18))
            var uncompressedSize = Int(readUInt32(data, at: offset + 22))
            let nameLen = Int(readUInt16(data, at: offset + 26))
            let extraLen = Int(readUInt16(data, at: offset + 28))

            let nameStart = offset + 30
            guard nameStart + nameLen <= data.count else { break }
            let nameData = data.subdata(in: nameStart..<nameStart + nameLen)
            var name = String(data: nameData, encoding: .utf8) ?? ""

            // Strip .npy extension
            if name.hasSuffix(".npy") {
                name = String(name.dropLast(4))
            }

            // Handle ZIP64 extra field (sizes stored as UInt64 when 0xFFFFFFFF)
            if compressedSize == 0xFFFFFFFF || uncompressedSize == 0xFFFFFFFF {
                let extraStart = nameStart + nameLen
                if extraLen >= 4 {
                    let tag = readUInt16(data, at: extraStart)
                    if tag == 0x0001 {  // ZIP64 extended information
                        var extraOffset = extraStart + 4
                        if uncompressedSize == 0xFFFFFFFF {
                            uncompressedSize = Int(readUInt64(data, at: extraOffset))
                            extraOffset += 8
                        }
                        if compressedSize == 0xFFFFFFFF {
                            compressedSize = Int(readUInt64(data, at: extraOffset))
                        }
                    }
                }
            }

            let dataStart = nameStart + nameLen + extraLen
            let dataSize = max(compressedSize, uncompressedSize)
            guard dataStart + dataSize <= data.count else { break }

            if let floats = parseNpy(data, npyOffset: dataStart, npySize: uncompressedSize) {
                result[name] = floats
            }

            offset = dataStart + dataSize
        }

        return result
    }

    /// Parse a .npy record at the given offset in the data buffer.
    private static func parseNpy(_ data: Data, npyOffset: Int, npySize: Int) -> [Float]? {
        // Magic: \x93NUMPY
        guard npyOffset >= 0,
              npySize >= 10,
              npyOffset + npySize <= data.count,
              data[npyOffset] == 0x93,
              data[npyOffset + 1] == 0x4E else { return nil }

        let majorVersion = data[npyOffset + 6]
        let headerLen: Int
        switch majorVersion {
        case 1:
            headerLen = Int(readUInt16(data, at: npyOffset + 8))
        case 2, 3:
            guard npySize >= 12 else { return nil }
            headerLen = Int(readUInt32(data, at: npyOffset + 8))
        default:
            return nil
        }

        let headerSize = (majorVersion == 1) ? 10 : 12
        let headerStart = npyOffset + headerSize
        let floatStart = headerStart + headerLen
        let numBytes = npySize - headerSize - headerLen

        guard headerLen > 0,
              floatStart <= npyOffset + npySize,
              numBytes > 0,
              numBytes.isMultiple(of: MemoryLayout<Float>.size),
              floatStart + numBytes <= data.count,
              let header = String(
                data: data[headerStart..<floatStart], encoding: .ascii)
        else { return nil }
        let numFloats = numBytes / 4

        var raw = [Float](repeating: 0, count: numFloats)
        _ = raw.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: floatStart..<floatStart + numFloats * 4)
        }

        // NumPy stores transposed/non-contiguous arrays in column-major order
        // and records that layout in the .npy header. The published DFN3
        // auxiliary bundle does this for erb_inv_fb [32, 481]. Returning its
        // bytes unchanged makes callers interpret columns as rows and expands
        // the predicted ERB mask onto the wrong FFT bins.
        let fortranOrder = header.contains("'fortran_order': True")
            || header.contains("\"fortran_order\": True")
        guard fortranOrder else { return raw }
        guard let shape = parseShape(fromNpyHeader: header),
              shape.count > 1,
              elementCount(of: shape, limit: numFloats) == numFloats
        else { return nil }

        return fortranToRowMajor(raw, shape: shape)
    }

    /// Validate the shape while avoiding integer overflow on malformed input.
    private static func elementCount(of shape: [Int], limit: Int) -> Int? {
        var count = 1
        for dimension in shape {
            guard dimension > 0, count <= limit / dimension else { return nil }
            count *= dimension
        }
        return count
    }

    /// Extract the tuple following `shape` from a NumPy header dictionary.
    private static func parseShape(fromNpyHeader header: String) -> [Int]? {
        guard let shapeKey = header.range(of: "shape"),
              let open = header[shapeKey.upperBound...].firstIndex(of: "("),
              let close = header[open...].firstIndex(of: ")")
        else { return nil }

        let contents = header[header.index(after: open)..<close]
        var dimensions = [Int]()
        for part in contents.split(separator: ",") {
            let value = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let dimension = Int(value) else { return nil }
            dimensions.append(dimension)
        }
        return dimensions.isEmpty ? nil : dimensions
    }

    /// Convert a NumPy Fortran-order flat buffer to conventional row-major
    /// order while preserving the declared N-dimensional shape.
    private static func fortranToRowMajor(_ input: [Float], shape: [Int]) -> [Float] {
        var rowMajorStrides = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 {
            for dimension in stride(from: shape.count - 2, through: 0, by: -1) {
                rowMajorStrides[dimension] = rowMajorStrides[dimension + 1]
                    * shape[dimension + 1]
            }
        }

        var output = [Float](repeating: 0, count: input.count)
        for rowMajorIndex in output.indices {
            var remainder = rowMajorIndex
            var fortranIndex = 0
            var fortranStride = 1
            for dimension in shape.indices {
                let coordinate = remainder / rowMajorStrides[dimension]
                remainder %= rowMajorStrides[dimension]
                fortranIndex += coordinate * fortranStride
                fortranStride *= shape[dimension]
            }
            output[rowMajorIndex] = input[fortranIndex]
        }
        return output
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt16.self)
        }
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt32.self)
        }
    }

    private static func readUInt64(_ data: Data, at offset: Int) -> UInt64 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt64.self)
        }
    }
}
