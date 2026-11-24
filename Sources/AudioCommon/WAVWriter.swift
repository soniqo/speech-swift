import AVFoundation
import AudioToolbox
import Foundation

/// Write float audio samples to WAV file
public enum WAVWriter {

    /// Write mono IEEE Float32 WAV without quantizing or clamping sample levels.
    public static func writeFloat32(samples: [Float], sampleRate: Int, to url: URL) throws {
        guard sampleRate > 0,
              let frameCount = AVAudioFrameCount(exactly: samples.count),
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate),
                                         channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: max(1, frameCount)),
              let channel = buffer.floatChannelData?[0] else {
            throw AudioLoadError.bufferCreationFailed
        }
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                channel.update(from: base, count: source.count)
            }
        }
        buffer.frameLength = frameCount
        var settings = format.settings
        settings[AVAudioFileTypeKey] = kAudioFileWAVEType
        let file = try AVAudioFile(forWriting: url, settings: settings,
                                   commonFormat: .pcmFormatFloat32, interleaved: false)
        defer { file.close() }
        try file.write(from: buffer)
    }

    /// Write mono float samples to a 16-bit PCM WAV file
    /// - Parameters:
    ///   - samples: Float audio samples in [-1.0, 1.0] range
    ///   - sampleRate: Sample rate in Hz (default 24000)
    ///   - url: Output file URL
    public static func write(samples: [Float], sampleRate: Int = 24000, to url: URL) throws {
        let numChannels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample) / 8
        let dataSize = samples.count * bytesPerSample
        let fileSize = 36 + dataSize

        var data = Data(capacity: fileSize + 8)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(&data, UInt32(fileSize))
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(&data, 16)                         // chunk size
        appendUInt16(&data, 1)                          // PCM format
        appendUInt16(&data, numChannels)
        appendUInt32(&data, UInt32(sampleRate))
        appendUInt32(&data, UInt32(sampleRate * Int(numChannels) * bytesPerSample))  // byte rate
        appendUInt16(&data, numChannels * UInt16(bytesPerSample))  // block align
        appendUInt16(&data, bitsPerSample)

        // data chunk
        data.append(contentsOf: "data".utf8)
        appendUInt32(&data, UInt32(dataSize))

        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let int16Value = Int16(clamped * 32767.0)
            appendInt16(&data, int16Value)
        }

        try data.write(to: url)
    }

    /// Write stereo float samples to a 16-bit PCM WAV file.
    /// - Parameters:
    ///   - left: Left channel float samples in [-1.0, 1.0]
    ///   - right: Right channel float samples in [-1.0, 1.0]
    ///   - sampleRate: Sample rate in Hz
    ///   - url: Output file URL
    public static func writeStereo(left: [Float], right: [Float], sampleRate: Int = 44100, to url: URL) throws {
        let numChannels: UInt16 = 2
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample) / 8
        let frameCount = min(left.count, right.count)
        let dataSize = frameCount * Int(numChannels) * bytesPerSample
        let fileSize = 36 + dataSize

        var data = Data(capacity: fileSize + 8)

        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(&data, UInt32(fileSize))
        data.append(contentsOf: "WAVE".utf8)

        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(&data, 16)
        appendUInt16(&data, 1)  // PCM
        appendUInt16(&data, numChannels)
        appendUInt32(&data, UInt32(sampleRate))
        appendUInt32(&data, UInt32(sampleRate * Int(numChannels) * bytesPerSample))
        appendUInt16(&data, numChannels * UInt16(bytesPerSample))
        appendUInt16(&data, bitsPerSample)

        data.append(contentsOf: "data".utf8)
        appendUInt32(&data, UInt32(dataSize))

        for i in 0..<frameCount {
            let l = max(-1.0, min(1.0, left[i]))
            let r = max(-1.0, min(1.0, right[i]))
            appendInt16(&data, Int16(l * 32767.0))
            appendInt16(&data, Int16(r * 32767.0))
        }

        try data.write(to: url)
    }

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 4))
    }

    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }

    private static func appendInt16(_ data: inout Data, _ value: Int16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }
}
