#if canImport(AVFoundation)
import AVFoundation
import Foundation
import os

/// Microphone capture and scheduled low-latency playback on one audio engine.
///
/// Unlike ``StreamingAudioPlayer``, playback uses `AVAudioPlayerNode`. Apple's
/// Voice Processing I/O can therefore inspect scheduled output buffers and use
/// them as the acoustic-echo reference for a true full-duplex conversation.
public final class FullDuplexAudioIO: @unchecked Sendable {
    public struct Configuration: Sendable, Equatable {
        public var inputSampleRate: Int
        public var outputSampleRate: Int
        public var inputBufferFrames: UInt32
        public var playbackPrebufferFrames: Int
        public var enableAEC: Bool

        public init(
            inputSampleRate: Int = 16_000,
            outputSampleRate: Int = 22_050,
            inputBufferFrames: UInt32 = 1_024,
            playbackPrebufferFrames: Int = 3,
            enableAEC: Bool = true
        ) {
            self.inputSampleRate = inputSampleRate
            self.outputSampleRate = outputSampleRate
            self.inputBufferFrames = inputBufferFrames
            self.playbackPrebufferFrames = playbackPrebufferFrames
            self.enableAEC = enableAEC
        }

        public func validate() throws {
            guard inputSampleRate > 0 else {
                throw FullDuplexAudioError.invalidConfiguration(
                    "input sample rate must be positive")
            }
            guard outputSampleRate > 0 else {
                throw FullDuplexAudioError.invalidConfiguration(
                    "output sample rate must be positive")
            }
            guard inputBufferFrames > 0 else {
                throw FullDuplexAudioError.invalidConfiguration(
                    "input buffer size must be positive")
            }
            guard playbackPrebufferFrames > 0,
                  playbackPrebufferFrames <= 32 else {
                throw FullDuplexAudioError.invalidConfiguration(
                    "playback prebuffer must be between 1 and 32 frames")
            }
        }
    }

    public struct Statistics: Sendable, Equatable {
        public let scheduledBuffers: Int
        public let completedBuffers: Int
        public let underruns: Int
        public let microphoneLevel: Float
    }

    public enum State: Sendable, Equatable {
        case stopped
        case running
        case failed(String)
    }

    public let configuration: Configuration
    public var state: State {
        lock.lock()
        defer { lock.unlock() }
        return audioState
    }

    public var microphoneName: String {
        AVCaptureDevice.default(for: .audio)?.localizedName ?? "Default microphone"
    }

    private let lock = NSLock()
    private let playbackQueue: DispatchQueue
    private let playbackQueueKey = DispatchSpecificKey<Void>()
    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var playbackSourceFormat: AVAudioFormat?
    private var playbackFormat: AVAudioFormat?
    private var playbackConverter: AVAudioConverter?
    private var audioState: State = .stopped
    private var prebuffer: [AVAudioPCMBuffer] = []
    private var playbackPrebufferTarget = 0
    private var playbackStarted = false
    private var playbackGeneration = 0
    private var stopping = false
    private var scheduledBuffers = 0
    private var completedBuffers = 0
    private var underruns = 0
    private var microphoneLevel: Float = 0

    private static let log = Logger(
        subsystem: "audio.soniqo", category: "FullDuplexAudioIO")

    public init(configuration: Configuration = .init()) {
        self.configuration = configuration
        let queue = DispatchQueue(
            label: "audio.soniqo.full-duplex-playback",
            qos: .userInteractive)
        self.playbackQueue = queue
        queue.setSpecific(key: playbackQueueKey, value: ())
    }

    static func recoveryPrebufferFrames(initial: Int) -> Int {
        min(32, max(8, initial * 2))
    }

    /// Start capture, resampled to `configuration.inputSampleRate`.
    ///
    /// The callback executes on Core Audio's capture thread. It should only
    /// copy/enqueue samples; model inference belongs on a separate worker.
    public func start(onSamples: @escaping @Sendable ([Float]) -> Void) throws {
        try configuration.validate()
        stop()

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: configuration.enableAEC ? .voiceChat : .default,
            options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let playerNode = AVAudioPlayerNode()

        do {
            try AudioIO.enableVoiceProcessingIfRequested(
                configuration.enableAEC
            ) { enabled in
                try inputNode.setVoiceProcessingEnabled(enabled)
            }
        } catch {
            setState(.failed(
                "Cannot enable acoustic echo cancellation: \(error.localizedDescription)"))
            throw error
        }

        // Capture can be configured immediately after Voice Processing is
        // enabled. Playback is attached only after `engine.start()` below:
        // on macOS the output-side Voice Processing format settles
        // asynchronously and a pre-start player connection can retain the
        // stale hardware rate, causing AUVP initialization to fail.
        let hardwareFormat = inputNode.outputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0,
              hardwareFormat.channelCount > 0,
              let monoFormat = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: hardwareFormat.sampleRate,
                  channels: 1,
                  interleaved: false),
              let targetFormat = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: Double(configuration.inputSampleRate),
                  channels: 1,
                  interleaved: false),
              let converter = AVAudioConverter(
                  from: monoFormat, to: targetFormat),
              let playbackSourceFormat = AVAudioFormat(
                  commonFormat: .pcmFormatFloat32,
                  sampleRate: Double(configuration.outputSampleRate),
                  channels: 1,
                  interleaved: false)
        else {
            let error = FullDuplexAudioError.invalidDeviceFormat
            setState(.failed(error.description))
            throw error
        }

        inputNode.installTap(
            onBus: 0,
            bufferSize: AVAudioFrameCount(configuration.inputBufferFrames),
            format: hardwareFormat
        ) { [weak self] buffer, _ in
            guard let self,
                  let source = buffer.floatChannelData,
                  buffer.frameLength > 0,
                  let mono = AVAudioPCMBuffer(
                      pcmFormat: monoFormat,
                      frameCapacity: buffer.frameCapacity)
            else { return }

            mono.frameLength = buffer.frameLength
            memcpy(
                mono.floatChannelData![0], source[0],
                Int(buffer.frameLength) * MemoryLayout<Float>.size)

            guard let samples = AudioIO.resampleMicrophoneBuffer(
                mono, with: converter, to: targetFormat),
                !samples.isEmpty
            else { return }

            var sum: Float = 0
            for sample in samples { sum += sample * sample }
            guard self.setMicrophoneLevelIfRunning(
                sqrt(sum / Float(max(samples.count, 1))))
            else { return }
            onSamples(samples)
        }

        var playerAttached = false
        do {
            // Start the Voice Processing capture graph before adding any
            // playback source. `setVoiceProcessingEnabled` returns before the
            // macOS output client has finished moving from the hardware rate
            // to its negotiated rate; starting here is the synchronization
            // boundary that makes the live format safe to consume.
            try engine.start()

            engine.attach(playerNode)
            playerAttached = true
            engine.connect(
                playerNode, to: engine.mainMixerNode, format: nil)
            let playbackFormat = playerNode.outputFormat(forBus: 0)
            guard playbackFormat.sampleRate > 0,
                  playbackFormat.channelCount > 0,
                  let playbackConverter = AVAudioConverter(
                      from: playbackSourceFormat, to: playbackFormat)
            else {
                throw FullDuplexAudioError.invalidDeviceFormat
            }

            lock.lock()
            self.engine = engine
            self.playerNode = playerNode
            self.playbackSourceFormat = playbackSourceFormat
            self.playbackFormat = playbackFormat
            self.playbackConverter = playbackConverter
            self.prebuffer = []
            self.playbackPrebufferTarget = configuration.playbackPrebufferFrames
            self.playbackStarted = false
            self.playbackGeneration &+= 1
            self.stopping = false
            self.scheduledBuffers = 0
            self.completedBuffers = 0
            self.underruns = 0
            self.audioState = .running
            lock.unlock()
            Self.log.info(
                "Full-duplex audio started: input \(self.configuration.inputSampleRate) Hz, output \(self.configuration.outputSampleRate) Hz, AEC \(self.configuration.enableAEC ? "on" : "off")")
        } catch {
            inputNode.removeTap(onBus: 0)
            if playerAttached {
                engine.disconnectNodeOutput(playerNode)
                engine.detach(playerNode)
            }
            engine.stop()
            setState(.failed(error.localizedDescription))
            throw error
        }
    }

    /// Queue one contiguous model-audio chunk for echo-reference-aware output.
    public func schedulePlayback(_ samples: [Float]) {
        guard !samples.isEmpty else { return }
        lock.lock()
        let format = playbackSourceFormat
        let running = audioState == .running && !stopping
        lock.unlock()
        guard running, let format,
              let buffer = AVAudioPCMBuffer(
                  pcmFormat: format,
                  frameCapacity: AVAudioFrameCount(samples.count))
        else { return }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            guard let base = source.baseAddress else { return }
            buffer.floatChannelData![0].update(
                from: base, count: samples.count)
        }

        playbackQueue.async { [weak self] in
            self?.convertAndEnqueue(buffer)
        }
    }

    public func statistics() -> Statistics {
        lock.lock()
        defer { lock.unlock() }
        return Statistics(
            scheduledBuffers: scheduledBuffers,
            completedBuffers: completedBuffers,
            underruns: underruns,
            microphoneLevel: microphoneLevel)
    }

    public func stop() {
        lock.lock()
        guard engine != nil || playerNode != nil else {
            audioState = .stopped
            lock.unlock()
            return
        }
        stopping = true
        playbackGeneration &+= 1
        let engine = self.engine
        let node = playerNode
        self.engine = nil
        playerNode = nil
        playbackSourceFormat = nil
        playbackFormat = nil
        playbackConverter = nil
        audioState = .stopped
        microphoneLevel = 0
        lock.unlock()

        engine?.inputNode.removeTap(onBus: 0)
        withPlaybackQueue {
            node?.stop()
            prebuffer.removeAll(keepingCapacity: false)
            playbackPrebufferTarget = configuration.playbackPrebufferFrames
            playbackStarted = false
        }
        if let engine, let node {
            engine.disconnectNodeOutput(node)
            engine.detach(node)
        }
        engine?.stop()
    }

    deinit {
        stop()
    }

    private func convertAndEnqueue(_ source: AVAudioPCMBuffer) {
        lock.lock()
        let format = playbackFormat
        let converter = playbackConverter
        let running = audioState == .running && !stopping
        lock.unlock()
        guard running, let format, let converter,
              let buffer = Self.convertPlaybackBuffer(
                  source, with: converter, to: format),
              buffer.frameLength > 0
        else { return }
        enqueue(buffer)
    }

    /// Convert model-rate mono audio to the exact Voice Processing playback
    /// client format. Returning the PCM buffer directly preserves every
    /// output channel produced by `AVAudioConverter` (normally stereo on Mac).
    private static func convertPlaybackBuffer(
        _ inputBuffer: AVAudioPCMBuffer,
        with converter: AVAudioConverter,
        to targetFormat: AVAudioFormat
    ) -> AVAudioPCMBuffer? {
        let ratio = targetFormat.sampleRate / inputBuffer.format.sampleRate
        let capacity = AVAudioFrameCount(
            (Double(inputBuffer.frameLength) * ratio).rounded(.up) + 16)
        guard capacity > 0,
              let outputBuffer = AVAudioPCMBuffer(
                  pcmFormat: targetFormat, frameCapacity: capacity)
        else { return nil }

        var conversionError: NSError?
        var consumed = false
        converter.convert(to: outputBuffer, error: &conversionError) { _, status in
            if consumed {
                status.pointee = .noDataNow
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return inputBuffer
        }
        guard conversionError == nil else { return nil }
        return outputBuffer
    }

    private func enqueue(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        guard audioState == .running, !stopping, let node = playerNode else {
            lock.unlock()
            return
        }
        let started = playbackStarted
        let generation = playbackGeneration
        lock.unlock()

        if !started {
            prebuffer.append(buffer)
            guard prebuffer.count >= playbackPrebufferTarget else {
                return
            }
            let pending = prebuffer
            prebuffer.removeAll(keepingCapacity: true)
            for item in pending {
                schedule(item, on: node, generation: generation)
            }
            lock.lock()
            playbackStarted = true
            lock.unlock()
            node.play()
            return
        }

        schedule(buffer, on: node, generation: generation)
        if !node.isPlaying { node.play() }
    }

    private func schedule(
        _ buffer: AVAudioPCMBuffer,
        on node: AVAudioPlayerNode,
        generation: Int
    ) {
        lock.lock()
        guard generation == playbackGeneration,
              playerNode === node,
              audioState == .running,
              !stopping else {
            lock.unlock()
            return
        }
        scheduledBuffers += 1
        lock.unlock()
        node.scheduleBuffer(
            buffer,
            completionCallbackType: .dataPlayedBack
        ) { [weak self] _ in
            self?.playbackQueue.async { [weak self] in
                self?.bufferCompleted(node: node, generation: generation)
            }
        }
    }

    private func bufferCompleted(
        node: AVAudioPlayerNode,
        generation: Int
    ) {
        lock.lock()
        guard generation == playbackGeneration,
              playerNode === node else {
            lock.unlock()
            return
        }
        completedBuffers += 1
        let drained = completedBuffers == scheduledBuffers
            && audioState == .running && !stopping
        if drained {
            underruns += 1
            playbackStarted = false
            // Keep initial response latency low, but recover a starved stream
            // with enough queued audio to absorb sporadic slow MLX frames.
            // Reusing the three-frame startup cushion caused recurring audible
            // gaps whenever average generation hovered near 80 ms/frame.
            playbackPrebufferTarget = Self.recoveryPrebufferFrames(
                initial: configuration.playbackPrebufferFrames)
        }
        lock.unlock()
        if drained { node.pause() }
    }

    private func setState(_ newState: State) {
        lock.lock()
        audioState = newState
        lock.unlock()
    }

    /// Publish capture state only while this engine generation is active, so
    /// an in-flight tap callback cannot deliver samples after `stop()`.
    private func setMicrophoneLevelIfRunning(_ level: Float) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard audioState == .running, !stopping else { return false }
        microphoneLevel = level
        return true
    }

    /// Run teardown in-order without synchronously dispatching back onto the
    /// playback queue when the final strong reference is released there.
    private func withPlaybackQueue(_ body: () -> Void) {
        if DispatchQueue.getSpecific(key: playbackQueueKey) != nil {
            body()
        } else {
            playbackQueue.sync(execute: body)
        }
    }
}

public enum FullDuplexAudioError: Error, CustomStringConvertible, LocalizedError {
    case invalidConfiguration(String)
    case invalidDeviceFormat

    public var description: String {
        switch self {
        case .invalidConfiguration(let message):
            return "invalid full-duplex audio configuration: \(message)"
        case .invalidDeviceFormat:
            return "the selected audio device has no usable input/output format"
        }
    }

    public var errorDescription: String? { description }
}
#endif
