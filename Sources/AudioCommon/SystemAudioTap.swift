#if os(macOS)
import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation
import os

/// Errors thrown by `SystemAudioTap`. Every case carries the failing `OSStatus`
/// so callers can report the real Core Audio failure instead of a generic message.
public enum SystemAudioTapError: Error, LocalizedError {
    case alreadyRunning
    case processTranslationFailed(pid: pid_t, status: OSStatus)
    case tapCreationFailed(status: OSStatus)
    case formatReadFailed(status: OSStatus)
    case aggregateCreationFailed(status: OSStatus)
    case ioProcCreationFailed(status: OSStatus)
    case startFailed(status: OSStatus)
    case resamplerUnavailable(from: Double, to: Double)

    public var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "System audio capture is already running."
        case .processTranslationFailed(let pid, let status):
            return "Cannot translate pid \(pid) to a Core Audio process object (\(SystemAudioTap.describeOSStatus(status))). The process may not have registered with the audio HAL yet."
        case .tapCreationFailed(let status):
            return "Creating the system audio process tap failed (\(SystemAudioTap.describeOSStatus(status))). System audio capture requires the host app to declare NSAudioCaptureUsageDescription and the user to grant the macOS audio-recording permission."
        case .formatReadFailed(let status):
            return "Reading the tap stream format failed (\(SystemAudioTap.describeOSStatus(status)))."
        case .aggregateCreationFailed(let status):
            return "Creating the private aggregate device for the tap failed (\(SystemAudioTap.describeOSStatus(status)))."
        case .ioProcCreationFailed(let status):
            return "Creating the aggregate device IO proc failed (\(SystemAudioTap.describeOSStatus(status)))."
        case .startFailed(let status):
            return "Starting the aggregate device failed (\(SystemAudioTap.describeOSStatus(status)))."
        case .resamplerUnavailable(let from, let to):
            return "Cannot create a resampler from \(from) Hz to \(to) Hz."
        }
    }
}

/// Captures the system output mix — "what the Mac is playing" — via a Core Audio
/// process tap (macOS 14.4+ tapping API), mirroring `AudioIO`'s microphone API.
///
/// ```swift
/// let tap = SystemAudioTap()                       // excludes this process by default
/// try tap.start(targetSampleRate: 16000) { samples in
///     pipeline.pushAudio(samples)                  // mono Float32 @ 16 kHz
/// }
/// tap.stop()
/// ```
///
/// Design notes:
/// - The tap is a **mono global tap** over all processes, minus an exclusion list.
///   By default the current process is excluded so the app's own playback (for
///   example a TTS voice) is not re-captured.
/// - The tap is wrapped in a **private aggregate device that contains only the
///   tap** — no sub-devices. Including the output device's own streams can pull
///   its input channels (for example a Bluetooth headset microphone) into the
///   capture and duplicate audio.
/// - A global tap follows processes, not a specific device, so it keeps
///   capturing across default-output-device switches. The delivered sample rate
///   can change on such a switch; a property listener re-reads the tap format
///   and the internal resampler is rebuilt on the fly. `onSamples` always
///   receives mono Float32 at `targetSampleRate`.
///
/// Permission: the host app must declare `NSAudioCaptureUsageDescription`;
/// macOS prompts on first tap creation. **If the permission is denied, Core
/// Audio may still create a working tap that delivers only silence.** Callers
/// that need to fail closed should watch `framesCaptured` grow while
/// `nonSilentFrames` stays at zero and surface an explicit status.
public final class SystemAudioTap {
    /// Capture state, mirroring `AudioIO.MicrophoneState`.
    public enum CaptureState: Sendable {
        case stopped, running, error(String)
    }

    /// Current capture state.
    public private(set) var captureState: CaptureState = .stopped

    /// Whether the current process is excluded from the tap (default true).
    public let excludeCurrentProcess: Bool

    /// Additional pids excluded from the tap.
    public let excludedPIDs: [pid_t]

    /// Amplitude threshold below which a sample counts as silence.
    public static let silenceThreshold: Float = 1e-6

    private static let log = Logger(subsystem: "audio.soniqo", category: "SystemAudioTap")

    // Core Audio objects owned while running.
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private var tapDescription: CATapDescription?

    // Listener bookkeeping so the exact registered blocks can be removed.
    private var formatListener: AudioObjectPropertyListenerBlock?
    private var defaultDeviceListener: AudioObjectPropertyListenerBlock?

    // IO-block-confined resampling state (touched only on `ioQueue`).
    private var resampler: AVAudioConverter?
    private var resamplerInputRate: Double = 0
    private var targetSampleRate: Double = 16000
    private var onSamples: (([Float]) -> Void)?

    // Shared counters (IO queue writes, any thread reads).
    private var statsLock = os_unfair_lock()
    private var _tapSampleRate: Double = 0
    private var _framesCaptured: UInt64 = 0
    private var _nonSilentFrames: UInt64 = 0
    private var _audioLevel: Float = 0
    private var _deviceChanges: UInt64 = 0

    private let ioQueue = DispatchQueue(label: "audio.soniqo.system-audio-tap.io")
    private let controlQueue = DispatchQueue(label: "audio.soniqo.system-audio-tap.control")

    public init(excludeCurrentProcess: Bool = true, excludedPIDs: [pid_t] = []) {
        self.excludeCurrentProcess = excludeCurrentProcess
        self.excludedPIDs = excludedPIDs
    }

    deinit {
        stop()
    }

    // MARK: - Public capture surface

    /// Sample rate the tap currently delivers before resampling. 0 when stopped.
    public var tapSampleRate: Double {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _tapSampleRate
    }

    /// Total frames delivered by the tap since `start`. Note that the HAL may
    /// deliver no callbacks at all while every audible process is excluded from
    /// the tap (nothing contributes to the mixdown), so 0 can also mean "no
    /// non-excluded audio has played yet".
    public var framesCaptured: UInt64 {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _framesCaptured
    }

    /// Frames above the silence threshold since `start`. A capture where
    /// `framesCaptured` grows while this stays 0 usually means the audio-capture
    /// permission is missing (the tap exists but delivers silence).
    public var nonSilentFrames: UInt64 {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _nonSilentFrames
    }

    /// RMS level (0.0–1.0) of the most recent capture buffer, for UI meters.
    public var audioLevel: Float {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _audioLevel
    }

    /// Number of default-output-device changes observed while capturing.
    public var deviceChanges: UInt64 {
        os_unfair_lock_lock(&statsLock)
        defer { os_unfair_lock_unlock(&statsLock) }
        return _deviceChanges
    }

    /// Start capturing the system output mix.
    ///
    /// - Parameters:
    ///   - targetSampleRate: Output rate for `onSamples` (default 16 kHz for VAD/ASR).
    ///   - onSamples: Mono Float32 samples at `targetSampleRate`, called on the
    ///     capture queue. Keep the callback fast; long work belongs on another queue.
    public func start(
        targetSampleRate: Int = 16000,
        onSamples: @escaping ([Float]) -> Void
    ) throws {
        guard case .stopped = captureState else {
            throw SystemAudioTapError.alreadyRunning
        }

        do {
            try activate(targetSampleRate: Double(targetSampleRate), onSamples: onSamples)
            captureState = .running
            Self.log.info("System audio tap started, tap rate \(self.tapSampleRate) Hz → \(targetSampleRate) Hz")
        } catch {
            stop()
            captureState = .error(error.localizedDescription)
            throw error
        }
    }

    /// Stop capturing and destroy the tap, aggregate device, and listeners.
    /// Idempotent. Must not be called from inside `onSamples` — teardown joins
    /// the capture queue that the callback runs on.
    public func stop() {
        if let listener = formatListener, tapID != kAudioObjectUnknown {
            var address = Self.tapFormatAddress
            AudioObjectRemovePropertyListenerBlock(tapID, &address, controlQueue, listener)
            formatListener = nil
        }
        if let listener = defaultDeviceListener {
            var address = Self.defaultOutputDeviceAddress
            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject), &address, controlQueue, listener)
            defaultDeviceListener = nil
        }
        if let procID = ioProcID, aggregateID != kAudioObjectUnknown {
            AudioDeviceStop(aggregateID, procID)
            AudioDeviceDestroyIOProcID(aggregateID, procID)
        }
        ioProcID = nil
        if aggregateID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggregateID)
            aggregateID = AudioObjectID(kAudioObjectUnknown)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = AudioObjectID(kAudioObjectUnknown)
        }
        tapDescription = nil
        ioQueue.sync {
            resampler = nil
            resamplerInputRate = 0
            onSamples = nil
        }
        os_unfair_lock_lock(&statsLock)
        _tapSampleRate = 0
        _audioLevel = 0
        os_unfair_lock_unlock(&statsLock)
        captureState = .stopped
    }

    // MARK: - Capture pipeline

    private func activate(targetSampleRate: Double, onSamples: @escaping ([Float]) -> Void) throws {
        os_unfair_lock_lock(&statsLock)
        _framesCaptured = 0
        _nonSilentFrames = 0
        _deviceChanges = 0
        os_unfair_lock_unlock(&statsLock)

        ioQueue.sync {
            self.targetSampleRate = targetSampleRate
            self.onSamples = onSamples
        }

        // 1. Translate the exclusion pids to HAL process objects. Excluding a
        // process requires it to be registered with the HAL; failing closed here
        // beats silently re-capturing the caller's own playback later.
        let excludedObjects = try effectiveExcludedPIDs().map { pid -> AudioObjectID in
            try Self.processObject(for: pid)
        }

        // 2. Mono global tap over everything except the excluded processes.
        let description = CATapDescription(monoGlobalTapButExcludeProcesses: excludedObjects)
        description.name = "audio.soniqo.system-audio-tap"
        description.isPrivate = true
        description.muteBehavior = .unmuted
        self.tapDescription = description

        var newTapID = AudioObjectID(kAudioObjectUnknown)
        let tapStatus = AudioHardwareCreateProcessTap(description, &newTapID)
        guard tapStatus == noErr, newTapID != kAudioObjectUnknown else {
            throw SystemAudioTapError.tapCreationFailed(status: tapStatus)
        }
        tapID = newTapID

        try refreshTapFormat()

        // 3. Private aggregate device containing only the tap (no sub-devices).
        let composition = Self.aggregateComposition(
            tapUUID: description.uuid, aggregateUID: UUID().uuidString)
        var newAggregateID = AudioObjectID(kAudioObjectUnknown)
        let aggregateStatus = AudioHardwareCreateAggregateDevice(
            composition as CFDictionary, &newAggregateID)
        guard aggregateStatus == noErr, newAggregateID != kAudioObjectUnknown else {
            throw SystemAudioTapError.aggregateCreationFailed(status: aggregateStatus)
        }
        aggregateID = newAggregateID

        // 4. Track tap format changes (device switches can change the mix rate).
        let formatListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self else { return }
            do {
                try self.refreshTapFormat()
            } catch {
                Self.log.error("Tap format re-read failed: \(error.localizedDescription)")
            }
        }
        var formatAddress = Self.tapFormatAddress
        AudioObjectAddPropertyListenerBlock(tapID, &formatAddress, controlQueue, formatListener)
        self.formatListener = formatListener

        let deviceListener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self else { return }
            os_unfair_lock_lock(&self.statsLock)
            self._deviceChanges += 1
            os_unfair_lock_unlock(&self.statsLock)
            do {
                try self.refreshTapFormat()
            } catch {
                Self.log.error("Tap format re-read after device change failed: \(error.localizedDescription)")
            }
        }
        var deviceAddress = Self.defaultOutputDeviceAddress
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &deviceAddress, controlQueue, deviceListener)
        self.defaultDeviceListener = deviceListener

        // 5. IO proc on the aggregate: the tap arrives as the input buffer list.
        var newIOProcID: AudioDeviceIOProcID?
        let ioStatus = AudioDeviceCreateIOProcIDWithBlock(&newIOProcID, aggregateID, ioQueue) {
            [weak self] _, inInputData, _, _, _ in
            self?.handleInput(inInputData)
        }
        guard ioStatus == noErr, let procID = newIOProcID else {
            throw SystemAudioTapError.ioProcCreationFailed(status: ioStatus)
        }
        ioProcID = procID

        let startStatus = AudioDeviceStart(aggregateID, procID)
        guard startStatus == noErr else {
            throw SystemAudioTapError.startFailed(status: startStatus)
        }
    }

    /// Runs on `ioQueue`. Downmixes the tap buffers to mono, updates counters,
    /// resamples to the target rate, and delivers the batch.
    private func handleInput(_ list: UnsafePointer<AudioBufferList>) {
        let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: list))
        guard buffers.count > 0 else { return }

        var mono: [Float]
        if buffers.count == 1, buffers[0].mNumberChannels <= 1 {
            guard let data = buffers[0].mData else { return }
            let frames = Int(buffers[0].mDataByteSize) / MemoryLayout<Float>.size
            guard frames > 0 else { return }
            mono = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: frames))
        } else if buffers.count == 1 {
            guard let data = buffers[0].mData else { return }
            let channels = Int(buffers[0].mNumberChannels)
            let values = Int(buffers[0].mDataByteSize) / MemoryLayout<Float>.size
            let interleaved = Array(UnsafeBufferPointer(start: data.assumingMemoryBound(to: Float.self), count: values))
            mono = Self.monoMixdown(interleaved: interleaved, channels: channels)
        } else {
            var planar: [[Float]] = []
            planar.reserveCapacity(buffers.count)
            for buffer in buffers {
                guard let data = buffer.mData else { continue }
                let frames = Int(buffer.mDataByteSize) / MemoryLayout<Float>.size
                planar.append(Array(UnsafeBufferPointer(
                    start: data.assumingMemoryBound(to: Float.self), count: frames)))
            }
            mono = Self.monoMixdown(planar: planar)
        }
        guard !mono.isEmpty else { return }

        let nonSilent = Self.nonSilentCount(mono, threshold: Self.silenceThreshold)
        var sum: Float = 0
        for sample in mono { sum += sample * sample }
        let rms = (sum / Float(mono.count)).squareRoot()

        os_unfair_lock_lock(&statsLock)
        _framesCaptured += UInt64(mono.count)
        _nonSilentFrames += UInt64(nonSilent)
        _audioLevel = rms
        let inputRate = _tapSampleRate
        os_unfair_lock_unlock(&statsLock)

        guard inputRate > 0 else { return }
        guard let delivery = onSamples else { return }

        if inputRate == targetSampleRate {
            delivery(mono)
            return
        }
        guard let resampled = resampleOnIOQueue(mono, from: inputRate) else { return }
        if !resampled.isEmpty {
            delivery(resampled)
        }
    }

    /// Stateful streaming resample on `ioQueue`; the converter is rebuilt when the
    /// tap rate changes (for example after a default-output-device switch).
    private func resampleOnIOQueue(_ samples: [Float], from inputRate: Double) -> [Float]? {
        if resampler == nil || resamplerInputRate != inputRate {
            guard
                let inputFormat = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: inputRate,
                    channels: 1, interleaved: false),
                let outputFormat = AVAudioFormat(
                    commonFormat: .pcmFormatFloat32, sampleRate: targetSampleRate,
                    channels: 1, interleaved: false),
                let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
            else {
                Self.log.error("Resampler unavailable: \(inputRate) Hz → \(self.targetSampleRate) Hz")
                return nil
            }
            resampler = converter
            resamplerInputRate = inputRate
        }
        guard let resampler else { return nil }

        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: resampler.inputFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else { return nil }
        inputBuffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            inputBuffer.floatChannelData![0].update(from: source.baseAddress!, count: samples.count)
        }

        let capacity = AVAudioFrameCount(
            (Double(samples.count) * targetSampleRate / inputRate).rounded(.up) + 16)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: resampler.outputFormat, frameCapacity: capacity)
        else { return nil }

        var conversionError: NSError?
        var consumed = false
        resampler.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return inputBuffer
        }
        if conversionError != nil { return nil }

        let frames = Int(outputBuffer.frameLength)
        guard frames > 0, let data = outputBuffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: data[0], count: frames))
    }

    /// Re-read the tap's stream format and publish the current sample rate.
    private func refreshTapFormat() throws {
        var address = Self.tapFormatAddress
        var asbd = AudioStreamBasicDescription()
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let status = AudioObjectGetPropertyData(tapID, &address, 0, nil, &size, &asbd)
        guard status == noErr, asbd.mSampleRate > 0 else {
            throw SystemAudioTapError.formatReadFailed(status: status)
        }
        os_unfair_lock_lock(&statsLock)
        let previous = _tapSampleRate
        _tapSampleRate = asbd.mSampleRate
        os_unfair_lock_unlock(&statsLock)
        if previous != 0, previous != asbd.mSampleRate {
            Self.log.info("Tap sample rate changed: \(previous) Hz → \(asbd.mSampleRate) Hz")
        }
    }

    // MARK: - Pure helpers (unit-tested)

    /// Exclusion list actually applied to the tap: the explicit pids plus the
    /// current process (unless opted out), deduplicated, in stable order.
    func effectiveExcludedPIDs() -> [pid_t] {
        var pids = excludedPIDs
        if excludeCurrentProcess {
            pids.append(getpid())
        }
        var seen = Set<pid_t>()
        return pids.filter { seen.insert($0).inserted }
    }

    /// Composition dictionary for the tap's aggregate device. Deliberately
    /// contains **no sub-device list**: aggregating the output device alongside
    /// the tap can import that device's input streams (for example a Bluetooth
    /// headset microphone) and duplicate audio into the capture.
    static func aggregateComposition(tapUUID: UUID, aggregateUID: String) -> [String: Any] {
        [
            kAudioAggregateDeviceNameKey: "soniqo system audio tap",
            kAudioAggregateDeviceUIDKey: aggregateUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapUUID.uuidString,
                    kAudioSubTapDriftCompensationKey: true,
                ]
            ],
        ]
    }

    /// Average interleaved frames down to mono. `channels <= 1` returns the input.
    static func monoMixdown(interleaved samples: [Float], channels: Int) -> [Float] {
        guard channels > 1 else { return samples }
        let frames = samples.count / channels
        guard frames > 0 else { return [] }
        var mono = [Float](repeating: 0, count: frames)
        let scale = 1 / Float(channels)
        for frame in 0..<frames {
            var sum: Float = 0
            let base = frame * channels
            for channel in 0..<channels {
                sum += samples[base + channel]
            }
            mono[frame] = sum * scale
        }
        return mono
    }

    /// Average planar channel buffers down to mono, tolerating length mismatches.
    static func monoMixdown(planar channels: [[Float]]) -> [Float] {
        guard let longest = channels.map(\.count).max(), longest > 0 else { return [] }
        guard channels.count > 1 else { return channels.first ?? [] }
        var mono = [Float](repeating: 0, count: longest)
        for channel in channels {
            for index in 0..<channel.count {
                mono[index] += channel[index]
            }
        }
        let scale = 1 / Float(channels.count)
        for index in 0..<longest {
            mono[index] *= scale
        }
        return mono
    }

    /// Count of samples whose magnitude exceeds `threshold`.
    static func nonSilentCount(_ samples: [Float], threshold: Float) -> Int {
        var count = 0
        for sample in samples where abs(sample) > threshold {
            count += 1
        }
        return count
    }

    /// Render an `OSStatus` as its four-character code when printable
    /// (for example 1852797029 → "'nope'"), otherwise as a decimal.
    static func describeOSStatus(_ status: OSStatus) -> String {
        let value = UInt32(bitPattern: status)
        let bytes = [
            UInt8((value >> 24) & 0xFF),
            UInt8((value >> 16) & 0xFF),
            UInt8((value >> 8) & 0xFF),
            UInt8(value & 0xFF),
        ]
        let printable = bytes.allSatisfy { $0 >= 0x20 && $0 < 0x7F }
        if printable, let code = String(bytes: bytes, encoding: .ascii) {
            return "'\(code)' (\(status))"
        }
        return "\(status)"
    }

    // MARK: - Core Audio plumbing

    private static var tapFormatAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
    }

    private static var defaultOutputDeviceAddress: AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
    }

    /// Translate a pid to its HAL process object.
    static func processObject(for pid: pid_t) throws -> AudioObjectID {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var qualifier = pid
        var object = AudioObjectID(kAudioObjectUnknown)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address,
            UInt32(MemoryLayout<pid_t>.size), &qualifier, &size, &object)
        guard status == noErr, object != kAudioObjectUnknown else {
            throw SystemAudioTapError.processTranslationFailed(pid: pid, status: status)
        }
        return object
    }
}
#endif
