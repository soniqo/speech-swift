import Foundation
import AVFoundation

/// Sample-rate-conversion quality. Both options fully drain the converter and
/// produce exact-length output; they differ only in the SRC filter.
public enum ResampleQuality {
    /// Framework-default band-limited SRC (`Normal` algorithm). Anti-aliases
    /// steep downsamples and retains high frequencies well below Nyquist;
    /// rolls off slightly more near Nyquist than `.mastering`. The right
    /// default for speech/voice, which is band-limited and usually
    /// downsampled (e.g. 44.1k→16k for ASR), where mastering-grade filtering
    /// is wasted cost.
    case standard
    /// Mastering algorithm at maximum quality — fullest high-frequency
    /// retention right up to Nyquist, at higher cost. Use for music (source
    /// separation) and upsampling/super-resolution, where full-band fidelity
    /// matters.
    case mastering
}

/// Loads audio files and converts to float samples
public enum AudioFileLoader {
    /// Load audio file and return samples at target sample rate.
    /// `quality` selects the SRC filter when resampling (default `.standard`;
    /// pass `.mastering` for music/upsampling).
    public static func load(url: URL, targetSampleRate: Int = 24000, quality: ResampleQuality = .standard) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioLoadError.bufferCreationFailed
        }

        try audioFile.read(into: buffer)

        guard let floatData = buffer.floatChannelData else {
            throw AudioLoadError.noFloatData
        }

        // Get mono samples (use first channel)
        let samples = Array(UnsafeBufferPointer(start: floatData[0], count: Int(buffer.frameLength)))

        // Resample if needed
        let inputSampleRate = Int(format.sampleRate)
        if inputSampleRate != targetSampleRate {
            return resample(samples, from: inputSampleRate, to: targetSampleRate, quality: quality)
        }

        return samples
    }

    /// Load audio file and return stereo channels at target sample rate.
    /// Returns `[left, right]` — mono files are duplicated to stereo.
    /// `quality` selects the SRC filter when resampling (default `.standard`;
    /// pass `.mastering` for music).
    public static func loadStereo(url: URL, targetSampleRate: Int = 44100, quality: ResampleQuality = .standard) throws -> [[Float]] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioLoadError.bufferCreationFailed
        }

        try audioFile.read(into: buffer)

        guard let floatData = buffer.floatChannelData else {
            throw AudioLoadError.noFloatData
        }

        let count = Int(buffer.frameLength)
        let left = Array(UnsafeBufferPointer(start: floatData[0], count: count))
        let right: [Float]
        if format.channelCount >= 2 {
            right = Array(UnsafeBufferPointer(start: floatData[1], count: count))
        } else {
            right = left  // Mono → duplicate
        }

        let inputSampleRate = Int(format.sampleRate)
        if inputSampleRate != targetSampleRate {
            // Resample both channels in one converter pass so L/R stay
            // phase-aligned (two independent converters can drift).
            return resampleStereo([left, right], from: inputSampleRate, to: targetSampleRate, quality: quality)
        }

        return [left, right]
    }

    /// Load WAV file directly (for 16-bit PCM)
    public static func loadWAV(url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let data = try Data(contentsOf: url)

        // Parse WAV header
        guard data.count > 44 else {
            throw AudioLoadError.invalidWAVFile
        }

        // Check RIFF header
        let riff = String(data: data[0..<4], encoding: .ascii)
        guard riff == "RIFF" else {
            throw AudioLoadError.invalidWAVFile
        }

        // Check WAVE format
        let wave = String(data: data[8..<12], encoding: .ascii)
        guard wave == "WAVE" else {
            throw AudioLoadError.invalidWAVFile
        }

        // Parse format chunk (handle unaligned reads)
        let audioFormat = data[20..<22].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
        let numChannels = data[22..<24].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }
        let sampleRate = data[24..<28].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        let bitsPerSample = data[34..<36].withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) }

        guard audioFormat == 1 else { // PCM
            throw AudioLoadError.unsupportedFormat("Not PCM format")
        }

        guard numChannels > 0 else {
            throw AudioLoadError.invalidWAVFile
        }

        guard bitsPerSample == 16 else {
            throw AudioLoadError.unsupportedFormat("Not 16-bit")
        }

        // Find data chunk
        var dataOffset = 36
        var dataChunkSize: UInt32? = nil
        while dataOffset < data.count - 8 {
            let chunkId = String(data: data[dataOffset..<(dataOffset+4)], encoding: .ascii)
            let chunkSize = data[(dataOffset+4)..<(dataOffset+8)].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }

            if chunkId == "data" {
                dataOffset += 8
                dataChunkSize = chunkSize
                break
            }

            // Validate chunk advance to avoid out-of-bounds.
            let nextOffset = dataOffset + 8 + Int(chunkSize)
            guard nextOffset >= dataOffset, nextOffset <= data.count else {
                throw AudioLoadError.invalidWAVFile
            }
            dataOffset = nextOffset
        }

        // Read samples
        guard let chunkSize = dataChunkSize else {
            throw AudioLoadError.invalidWAVFile
        }
        let chunkSizeInt = Int(chunkSize)
        guard dataOffset >= 0, dataOffset <= data.count, dataOffset + chunkSizeInt <= data.count else {
            throw AudioLoadError.invalidWAVFile
        }

        let sampleData = data[dataOffset..<(dataOffset + chunkSizeInt)]
        let channels = Int(numChannels)
        let bytesPerSample = 2
        let frameSize = bytesPerSample * channels
        let sampleCount = sampleData.count / frameSize

        var samples = [Float](repeating: 0, count: sampleCount)
        sampleData.withUnsafeBytes { ptr in
            let int16Ptr = ptr.bindMemory(to: Int16.self)
            for i in 0..<sampleCount {
                // Take first channel only
                let sampleIndex = i * channels
                if sampleIndex < int16Ptr.count {
                    samples[i] = Float(int16Ptr[sampleIndex]) / 32768.0
                }
            }
        }

        return (samples, Int(sampleRate))
    }

    /// Resample mono audio using `AVAudioConverter`.
    ///
    /// Fully drains the converter via `.endOfStream` so the filter tail isn't
    /// truncated, and normalizes the result to the exact expected frame count.
    /// `quality` selects the SRC filter: `.standard` (default) for speech,
    /// `.mastering` for music/upsampling — see ``ResampleQuality``.
    ///
    /// - Parameters:
    ///   - samples: mono PCM Float32 audio
    ///   - inputRate: source sample rate in Hz
    ///   - outputRate: target sample rate in Hz
    ///   - quality: SRC filter quality (default `.standard`)
    /// - Returns: resampled audio at `outputRate`. On converter-setup or
    ///   conversion failure, returns the original `samples` unchanged (callers
    ///   never receive partial/truncated output).
    public static func resample(_ samples: [Float], from inputRate: Int, to outputRate: Int, quality: ResampleQuality = .standard) -> [Float] {
        guard inputRate != outputRate, !samples.isEmpty else { return samples }

        guard let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(inputRate),
                channels: 1, interleaved: false),
              let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(outputRate),
                channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
              let sourceBuffer = AVAudioPCMBuffer(
                pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else {
            return samples
        }

        configureSRC(converter, quality: quality)
        sourceBuffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            sourceBuffer.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }

        let ratio = Double(outputRate) / Double(inputRate)
        guard let out = convertDrained(
            converter: converter, source: sourceBuffer, targetFormat: targetFormat,
            inputFrames: samples.count, ratio: ratio, channels: 1)
        else {
            return samples
        }
        return out[0]
    }

    /// Resample a stereo signal in a single converter pass so the two channels
    /// stay phase-aligned. `channels[0]` = left, `channels[1]` = right; both
    /// must have equal length. `quality` selects the SRC filter (default
    /// `.standard`; pass `.mastering` for music). Falls back to per-channel
    /// mono resampling for non-stereo input or on converter-setup failure.
    public static func resampleStereo(_ channels: [[Float]], from inputRate: Int, to outputRate: Int, quality: ResampleQuality = .standard) -> [[Float]] {
        guard channels.count == 2 else {
            return channels.map { resample($0, from: inputRate, to: outputRate, quality: quality) }
        }
        let n = channels[0].count
        guard inputRate != outputRate, n > 0, channels[1].count == n else {
            return channels
        }

        guard let sourceFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(inputRate),
                channels: 2, interleaved: false),
              let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: Double(outputRate),
                channels: 2, interleaved: false),
              let converter = AVAudioConverter(from: sourceFormat, to: targetFormat),
              let sourceBuffer = AVAudioPCMBuffer(
                pcmFormat: sourceFormat, frameCapacity: AVAudioFrameCount(n))
        else {
            return channels.map { resample($0, from: inputRate, to: outputRate, quality: quality) }
        }

        configureSRC(converter, quality: quality)
        sourceBuffer.frameLength = AVAudioFrameCount(n)
        channels[0].withUnsafeBufferPointer {
            sourceBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: n)
        }
        channels[1].withUnsafeBufferPointer {
            sourceBuffer.floatChannelData![1].update(from: $0.baseAddress!, count: n)
        }

        let ratio = Double(outputRate) / Double(inputRate)
        guard let out = convertDrained(
            converter: converter, source: sourceBuffer, targetFormat: targetFormat,
            inputFrames: n, ratio: ratio, channels: 2)
        else {
            return channels.map { resample($0, from: inputRate, to: outputRate, quality: quality) }
        }
        return out
    }

    /// Configure the converter's SRC filter. Must be set before the first
    /// `convert`. `.standard` leaves the framework default (`Normal`); only
    /// `.mastering` opts into the slower, full-band Mastering algorithm.
    private static func configureSRC(_ converter: AVAudioConverter, quality: ResampleQuality) {
        switch quality {
        case .standard:
            break  // framework default Normal SRC — already drains + exact length
        case .mastering:
            converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
            converter.sampleRateConverterQuality = .max
        }
    }

    /// Run the converter to completion, draining its internal tail via
    /// `.endOfStream`, and return one Float array per channel normalized to the
    /// exact expected frame count.
    ///
    /// Returns `nil` unless the converter reaches `.endOfStream` cleanly. Only
    /// `.endOfStream` is success: `.error` (or a thrown `NSError`) is a hard
    /// failure, and a no-progress step that isn't end-of-stream means the
    /// converter is stuck. In every non-success case the partial output is
    /// discarded rather than returned, so callers can fall back instead of
    /// silently propagating a truncated buffer (which would desync downstream
    /// audio/video).
    private static func convertDrained(
        converter: AVAudioConverter,
        source: AVAudioPCMBuffer,
        targetFormat: AVAudioFormat,
        inputFrames: Int,
        ratio: Double,
        channels: Int
    ) -> [[Float]]? {
        // ceil + headroom for the sinc filter's priming/tail latency.
        let capacity = AVAudioFrameCount(ceil(Double(inputFrames) * ratio)) + 4096
        guard let target = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
            return nil
        }

        var out = [[Float]](repeating: [], count: channels)
        for c in 0..<channels { out[c].reserveCapacity(Int(capacity)) }

        var fed = false
        while true {
            target.frameLength = 0
            var error: NSError?
            let status = converter.convert(to: target, error: &error) { _, outStatus in
                if fed {
                    // Signal end-of-stream so the converter flushes its sinc
                    // tail instead of stopping short.
                    outStatus.pointee = .endOfStream
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return source
            }

            let produced = Int(target.frameLength)
            if produced > 0, let chans = target.floatChannelData {
                for c in 0..<channels {
                    out[c].append(contentsOf: UnsafeBufferPointer(start: chans[c], count: produced))
                }
            }

            // Order matters: a clean `.endOfStream` (success) usually arrives on
            // a call that produces 0 frames, so check it before the no-progress
            // guard. `.inputRanDry` is not terminal — keep pulling until the
            // input block signals `.endOfStream`.
            if status == .error || error != nil { return nil }
            if status == .endOfStream { break }
            if produced == 0 { return nil }
        }

        // Normalize to the exact expected length so output is a strict function
        // of input length and ratio (downstream stereo alignment and A/V sync
        // rely on this). Mastering SRC compensates filter latency, so this is a
        // 0–1 sample trim/pad in practice.
        let expected = Int((Double(inputFrames) * ratio).rounded())
        guard expected > 0, !out[0].isEmpty else { return nil }
        for c in 0..<channels {
            if out[c].count > expected {
                out[c].removeLast(out[c].count - expected)
            } else if out[c].count < expected {
                out[c].append(contentsOf: repeatElement(0, count: expected - out[c].count))
            }
        }
        return out
    }
}

public enum AudioLoadError: Error, LocalizedError {
    case bufferCreationFailed
    case noFloatData
    case invalidWAVFile
    case unsupportedFormat(String)

    public var errorDescription: String? {
        switch self {
        case .bufferCreationFailed:
            return "Failed to create audio buffer"
        case .noFloatData:
            return "No float channel data available"
        case .invalidWAVFile:
            return "Invalid WAV file format"
        case .unsupportedFormat(let reason):
            return "Unsupported audio format: \(reason)"
        }
    }
}
