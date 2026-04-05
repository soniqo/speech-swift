import AVFoundation
import Observation

/// Captures mic audio and streams 16kHz Float32 chunks to a callback.
@Observable
final class StreamingRecorder {
    private(set) var isRecording = false
    private(set) var audioLevel: Float = 0

    private var audioEngine: AVAudioEngine?
    private var onChunk: (([Float]) -> Void)?

    /// Start recording and call `onChunk` with 16kHz mono Float32 samples.
    func start(onChunk: @escaping ([Float]) -> Void) {
        self.onChunk = onChunk
        audioLevel = 0

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        ) else { return }

        guard let converter = AVAudioConverter(from: hwFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }

            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * 16000.0 / hwFormat.sampleRate
            )
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: frameCount
            ) else { return }

            var error: NSError?
            converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if error != nil { return }

            guard let channelData = convertedBuffer.floatChannelData else { return }
            let count = Int(convertedBuffer.frameLength)
            let data = Array(UnsafeBufferPointer(start: channelData[0], count: count))

            let rms = sqrt(data.reduce(0) { $0 + $1 * $1 } / max(Float(count), 1))
            DispatchQueue.main.async { self.audioLevel = rms }

            self.onChunk?(data)
        }

        do {
            try engine.start()
            audioEngine = engine
            isRecording = true
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }

    func stop() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isRecording = false
        audioLevel = 0
        onChunk = nil
    }
}
