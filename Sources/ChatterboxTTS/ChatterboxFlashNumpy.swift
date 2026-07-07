#if canImport(CoreML)
import Foundation

enum ChatterboxFlashNumpy {
    static func loadFloat32Vector(from url: URL) throws -> [Float] {
        try parseFloat32Vector(try Data(contentsOf: url), label: url.lastPathComponent)
    }

    static func parseFloat32Vector(_ data: Data, label: String = "npy") throws -> [Float] {
        let bytes = [UInt8](data)
        guard bytes.count >= 16,
              bytes[0] == 0x93,
              String(bytes: bytes[1...5], encoding: .ascii) == "NUMPY"
        else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) is not a NumPy .npy file")
        }

        let major = bytes[6]
        let headerLengthOffset = 8
        let headerLength: Int
        let headerStart: Int
        if major == 1 {
            headerLength = Int(bytes[headerLengthOffset])
                | (Int(bytes[headerLengthOffset + 1]) << 8)
            headerStart = 10
        } else if major == 2 || major == 3 {
            headerLength = Int(bytes[headerLengthOffset])
                | (Int(bytes[headerLengthOffset + 1]) << 8)
                | (Int(bytes[headerLengthOffset + 2]) << 16)
                | (Int(bytes[headerLengthOffset + 3]) << 24)
            headerStart = 12
        } else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration(
                "\(label) uses unsupported NumPy version \(major)"
            )
        }

        let headerEnd = headerStart + headerLength
        guard headerEnd <= bytes.count,
              let header = String(bytes: bytes[headerStart..<headerEnd], encoding: .ascii)
        else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) has an invalid NumPy header")
        }
        guard header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"")
                || header.contains("'descr': '|f4'") || header.contains("\"descr\": \"|f4\"")
        else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) is not a float32 vector")
        }
        guard header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false") else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) uses unsupported Fortran order")
        }
        guard let count = parseVectorCount(from: header) else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) has unsupported shape")
        }

        let payloadBytes = count * 4
        guard headerEnd + payloadBytes <= bytes.count else {
            throw ChatterboxFlashCoreMLError.unsupportedConfiguration("\(label) payload is truncated")
        }

        var values = [Float]()
        values.reserveCapacity(count)
        for index in 0..<count {
            let offset = headerEnd + index * 4
            let bits = UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
            values.append(Float(bitPattern: bits))
        }
        return values
    }

    private static func parseVectorCount(from header: String) -> Int? {
        guard let shapeRange = header.range(of: "'shape':")
                ?? header.range(of: "\"shape\":")
        else { return nil }
        guard let open = header[shapeRange.upperBound...].firstIndex(of: "("),
              let close = header[open...].firstIndex(of: ")")
        else { return nil }
        let inside = header[header.index(after: open)..<close]
        let parts = inside.split(separator: ",", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard parts.count == 1, let count = Int(parts[0]), count >= 0 else {
            return nil
        }
        return count
    }
}
#endif
