#if canImport(AVFoundation)
import AVFoundation
import os

/// Reusable audio I/O manager — handles mic capture, resampling, and playback.
///
/// Eliminates AVAudioEngine boilerplate that every demo app reimplements.
///
/// ```swift
/// let audio = AudioIO()
/// try audio.startMicrophone(targetSampleRate: 16000) { samples in
///     pipeline.pushAudio(samples)
/// }
/// audio.player.scheduleChunk(ttsOutput)
/// audio.stopMicrophone()
/// ```
public final class AudioIO {
    /// Microphone state.
    public enum MicrophoneState: Sendable {
        case stopped, running, error(String)
    }

    /// Audio player for TTS output. Attached when microphone capture starts
    /// only if `enablePlayback` is true.
    public let player = StreamingAudioPlayer()

    /// Current microphone state.
    public private(set) var microphoneState: MicrophoneState = .stopped

    /// RMS audio level (0.0–1.0) for UI meters. Updated on each mic buffer.
    public private(set) var audioLevel: Float = 0

    /// Whether to enable Apple's Voice Processing I/O for acoustic echo
    /// cancellation before microphone samples reach the callback.
    public let enableAEC: Bool

    /// Whether to attach the streaming player to the capture engine.
    /// Listen-only clients can disable this to keep the I/O graph capture-only.
    public let enablePlayback: Bool

    /// Playback sample rate (for TTS output).
    public let playbackSampleRate: Double

    private var engine: AVAudioEngine?
    private static let log = Logger(subsystem: "audio.soniqo", category: "AudioIO")

    public init(
        enableAEC: Bool = false,
        playbackSampleRate: Double = 24000,
        enablePlayback: Bool = true
    ) {
        self.enableAEC = enableAEC
        self.playbackSampleRate = playbackSampleRate
        self.enablePlayback = enablePlayback
    }

    /// Start microphone capture, resampled to targetSampleRate.
    ///
    /// When `enablePlayback` is true, also attaches the player to the engine
    /// for simultaneous playback. Call `player.scheduleChunk()` to play audio
    /// while recording.
    ///
    /// - Parameters:
    ///   - targetSampleRate: Output sample rate for onSamples (default 16kHz for VAD/ASR)
    ///   - onSamples: Callback with resampled mono Float32 samples (called on audio thread)
    public func startMicrophone(
        targetSampleRate: Int = 16000,
        onSamples: @escaping ([Float]) -> Void
    ) throws {
        try startMicrophoneTimestamped(targetSampleRate: targetSampleRate) { chunk in
            onSamples(chunk.samples)
        }
    }

    /// Start microphone capture with the host timestamp of each input buffer.
    ///
    /// The timestamp belongs to the first hardware input frame represented by
    /// the delivered chunk. It remains tied to that frame when capture-side
    /// resampling changes the number of samples.
    public func startMicrophoneTimestamped(
        targetSampleRate: Int = 16000,
        onChunk: @escaping (CapturedAudioChunk) -> Void
    ) throws {
        stopMicrophone()

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        if enableAEC {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
        } else {
            try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        }
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        // Voice Processing changes both I/O nodes and the input format, so it
        // must be enabled while the engine is stopped and before reading that
        // format or installing a tap. Channel 0 is the processed microphone
        // signal; any additional channels are Voice Processing metadata.
        do {
            try Self.enableVoiceProcessingIfRequested(enableAEC) { enabled in
                try inputNode.setVoiceProcessingEnabled(enabled)
            }
        } catch {
            microphoneState = .error("Cannot enable acoustic echo cancellation: \(error.localizedDescription)")
            throw error
        }

        if enableAEC {
            // Stenography and ASR clients must not noticeably turn down the
            // audio they are observing. Voice Processing still gets the
            // device playback reference used for echo cancellation, while
            // applying the minimum available ducking to other audio.
            #if os(macOS) || os(iOS)
            if #available(macOS 14.0, iOS 17.0, *) {
                inputNode.voiceProcessingOtherAudioDuckingConfiguration =
                    AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                        enableAdvancedDucking: false,
                        duckingLevel: .min)
            }
            #endif
        }
        let hwFormat = inputNode.outputFormat(forBus: 0)

        // Mono intermediate at hardware rate
        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: hwFormat.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            microphoneState = .error("Cannot create mono format")
            return
        }

        // Target format for VAD/ASR
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        ) else {
            microphoneState = .error("Cannot create target format")
            return
        }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else {
            microphoneState = .error("Cannot create resampler")
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, when in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            // Extract channel 0 into mono buffer
            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            // Resample
            let outFrameCount = AVAudioFrameCount(Double(frameLen) * Double(targetSampleRate) / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outFrameCount) else { return }

            var error: NSError?
            resampler.convert(to: outBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return monoBuffer
            }
            if error != nil { return }

            guard let outData = outBuffer.floatChannelData else { return }
            let count = Int(outBuffer.frameLength)
            guard count > 0 else { return }
            let samples = Array(UnsafeBufferPointer(start: outData[0], count: count))

            // RMS for audio level
            var sum: Float = 0
            for s in samples { sum += s * s }
            self.audioLevel = sqrt(sum / max(Float(count), 1))

            onChunk(CapturedAudioChunk(
                samples: samples,
                sampleRate: targetSampleRate,
                hostTime: Self.hostTime(from: when)))
        }

        if enablePlayback {
            // Attach player for TTS output only when this client needs it.
            // Capture-only users should not add an unused source node to the
            // Voice Processing graph.
            guard let playerFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: playbackSampleRate,
                channels: 1,
                interleaved: false
            ) else { return }
            player.attach(to: engine, format: playerFormat)
        }

        do {
            try engine.start()
            if enablePlayback {
                player.startPlayback()
            }
            self.engine = engine
            microphoneState = .running
            Self.log.info(
                "Microphone started at \(targetSampleRate)Hz, playback \(self.enablePlayback ? "on at \(self.playbackSampleRate)Hz" : "off"), AEC \(self.enableAEC ? "on" : "off")")
        } catch {
            let message = enableAEC
                ? "Cannot start acoustic echo cancellation: \(error.localizedDescription)"
                : error.localizedDescription
            microphoneState = .error(message)
            throw error
        }
    }

    /// Stop microphone capture and detach the player when it was enabled.
    public func stopMicrophone() {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            if enablePlayback {
                player.detach(from: engine)
            }
            engine.stop()
        }
        engine = nil
        audioLevel = 0
        microphoneState = .stopped
    }

    deinit {
        stopMicrophone()
    }

    static func enableVoiceProcessingIfRequested(
        _ enabled: Bool,
        setEnabled: (Bool) throws -> Void
    ) throws {
        guard enabled else { return }
        try setEnabled(true)
    }

    static func hostTime(from time: AVAudioTime) -> UInt64? {
        time.isHostTimeValid ? time.hostTime : nil
    }
}
#endif
