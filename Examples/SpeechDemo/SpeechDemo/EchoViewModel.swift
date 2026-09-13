#if os(macOS)
import AVFoundation
import Foundation
import Observation
import ParakeetASR
import Qwen3TTS
import SpeechCore
import SpeechVAD
import AudioCommon

/// Wraps the Smart Turn model so each end-of-turn decision shows up in the log.
/// The pipeline calls it on the audio thread, once per VAD pause.
private final class LoggingTurnCompletion: TurnCompletionProvider {
    private let model: SmartTurnModel
    private let onDecision: (_ probability: Float, _ turnSeconds: Double) -> Void

    init(model: SmartTurnModel, onDecision: @escaping (Float, Double) -> Void) {
        self.model = model
        self.onDecision = onDecision
    }

    func turnCompleteProbability(audio: [Float], sampleRate: Int) throws -> Float {
        let probability = try model.turnCompleteProbability(audio: audio, sampleRate: sampleRate)
        onDecision(probability, Double(audio.count) / Double(sampleRate))
        return probability
    }
}

@Observable
@MainActor
final class EchoViewModel {
    var isLoading = false
    var loadingStatus = ""
    var errorMessage: String?
    var isRunning = false
    var pipelineState: String = "idle"
    var lastTranscription: String = ""
    var lastLanguage: String = ""
    var log: [String] = []
    var vadLevel: Float = 0
    /// Confirm each VAD pause with Smart Turn before ending the user's turn, so a
    /// mid-sentence pause does not trigger a reply. Read at `startPipeline()`.
    var smartTurnEnabled = true
    /// Most recent Smart Turn answer (probability that the turn was complete).
    var lastTurnProbability: Float?

    private var vad: SileroVADModel?
    private var asr: ParakeetASRModel?
    private var tts: Qwen3TTSModel?
    private var smartTurn: SmartTurnModel?
    private var turnCompletion: LoggingTurnCompletion?
    private var pipeline: VoicePipeline?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var debugMicBuffer: [Float] = []
    private var debugTTSBuffer: [Float] = []
    private var speechStartTime: Date?
    private var isSpeaking = false

    var modelsLoaded: Bool { vad != nil && asr != nil && tts != nil }

    // MARK: - Model Loading

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        log = []

        do {
            loadingStatus = "Loading VAD (CoreML)..."
            vad = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml)
            }.value

            // Smart Turn is optional (~17 MB). If it cannot be loaded the pipeline
            // still runs and ends turns on VAD silence alone.
            loadingStatus = "Loading Smart Turn (CoreML)..."
            do {
                smartTurn = try await Task.detached {
                    let model = try await SmartTurnModel.fromPretrained()
                    try model.prewarm()
                    return model
                }.value
            } catch {
                smartTurn = nil
                appendLog("[Turn] Smart Turn unavailable, turns end on VAD silence: \(error.localizedDescription)")
            }

            loadingStatus = "Loading ASR (Parakeet CoreML)..."
            asr = try await Task.detached {
                let model = try await ParakeetASRModel.fromPretrained()
                try model.warmUp()
                return model
            }.value

            loadingStatus = "Loading TTS (Qwen3 Base)..."
            tts = try await Task.detached {
                try await Qwen3TTSModel.fromPretrained(
                    modelId: TTSModelVariant.base.rawValue
                ) { (progress: Double, status: String) in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadingStatus = status.isEmpty
                            ? "Loading TTS... \(Int(progress * 100))%"
                            : "\(status) (\(Int(progress * 100))%)"
                    }
                }
            }.value

            appendLog("All models loaded.")
            loadingStatus = ""
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }
        isLoading = false
    }

    // MARK: - Pipeline Lifecycle

    func startPipeline() {
        guard let vad, let asr, let tts else { return }
        guard !isRunning else { return }

        var config = PipelineConfig()
        config.mode = .echo
        config.allowInterruptions = false
        config.minSilenceDuration = 0.6
        config.eagerSTT = false
        config.maxResponseDuration = 15.0

        pipeline = VoicePipeline(
            stt: asr, tts: tts, vad: vad, config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )

        // End-of-turn classifier: a VAD pause only ends the turn once Smart Turn
        // agrees (probability >= turnCompletionThreshold); otherwise the pipeline
        // keeps listening until speech resumes or turnCompletionMaxSilence elapses.
        lastTurnProbability = nil
        turnCompletion = nil
        if smartTurnEnabled, let smartTurn {
            let threshold = config.turnCompletionThreshold
            let logging = LoggingTurnCompletion(model: smartTurn) { [weak self] probability, seconds in
                DispatchQueue.main.async {
                    guard let self else { return }
                    self.lastTurnProbability = probability
                    let verdict = probability >= threshold ? "complete" : "hold"
                    self.appendLog("[Turn] p=\(String(format: "%.2f", probability)) \(verdict) after \(String(format: "%.1f", seconds))s of speech")
                }
            }
            turnCompletion = logging
            pipeline?.setTurnCompletion(logging)
        }

        player.onPlaybackFinished = { [weak self] in
            guard let self, self.isRunning else { return }
            self.isSpeaking = false
            self.pipeline?.resumeListening()
            self.pipelineState = "listening"
            self.appendLog("Listening...")
        }

        pipeline?.start()
        isRunning = true
        debugMicBuffer = []
        debugTTSBuffer = []
        pipelineState = "listening"
        appendLog(turnCompletion != nil
            ? "Pipeline started with Smart Turn — pauses are confirmed by the end-of-turn classifier..."
            : "Pipeline started — speak into the mic...")
        startMicrophone()
    }

    func stopPipeline() {
        stopMicrophone()
        pipeline?.stop()
        pipeline = nil
        turnCompletion = nil
        isRunning = false
        isSpeaking = false
        pipelineState = "idle"
        saveDebugFiles()
        appendLog("Pipeline stopped.")
    }

    // MARK: - Event Handling

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            break
        case .speechStarted:
            pipelineState = "speech detected"
            speechStartTime = Date()
            appendLog("[VAD] Speech started")
        case .speechEnded:
            pipelineState = "transcribing..."
            let duration = Date().timeIntervalSince(speechStartTime ?? Date())
            if duration > 13 {
                appendLog("[VAD] Speech ended (\(String(format: "%.0f", duration))s — max duration may have cut your phrase)")
            } else {
                appendLog("[VAD] Speech ended")
            }
        case .transcriptionCompleted(let text, let language, _):
            pipelineState = "synthesizing..."
            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                isSpeaking = true
            }
            lastTranscription = text
            lastLanguage = language ?? ""
            appendLog("[STT\(language.map { " [\($0)]" } ?? "")] \(text)")
        case .responseCreated:
            pipelineState = "speaking..."
            isSpeaking = true
            player.resetGeneration()
        case .responseInterrupted:
            player.stop()
            isSpeaking = false
            pipelineState = "listening"
        case .responseAudioDelta(let samples):
            debugTTSBuffer.append(contentsOf: samples)
            try? player.play(samples: samples, sampleRate: 24000)
        case .responseDone:
            appendLog("[TTS] Done")
            player.markGenerationComplete()
        case .toolCallStarted, .toolCallCompleted:
            break
        case .error(let msg):
            pipelineState = "error"
            isSpeaking = false
            appendLog("[ERROR] \(msg)")
            pipeline?.resumeListening()
        }
    }

    // MARK: - Microphone

    private func startMicrophone() {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode

        try? inputNode.setVoiceProcessingEnabled(true)

        let vpFormat = inputNode.outputFormat(forBus: 0)
        guard vpFormat.sampleRate > 0, vpFormat.channelCount > 0 else {
            appendLog("[Mic] Invalid format")
            return
        }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: vpFormat.sampleRate,
            channels: 1, interleaved: false
        ), let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false
        ), let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else {
            appendLog("[Mic] Cannot create audio formats")
            return
        }

        var bufCount = 0
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: vpFormat) { [weak self] buffer, _ in
            guard let self, let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            guard let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else { return }
            mono.frameLength = buffer.frameLength
            memcpy(mono.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            let outCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / vpFormat.sampleRate)
            guard outCount > 0, let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCount) else { return }
            var err: NSError?
            resampler.convert(to: out, error: &err) { _, status in status.pointee = .haveData; return mono }
            guard err == nil, let outData = out.floatChannelData, out.frameLength > 0 else { return }

            let count = Int(out.frameLength)
            let samples = Array(UnsafeBufferPointer(start: outData[0], count: count))
            let rms = sqrt(samples.reduce(0) { $0 + $1 * $1 } / Float(count))

            bufCount += 1
            if bufCount <= 3 {
                DispatchQueue.main.async {
                    self.appendLog("[Mic] Buffer #\(bufCount): \(count) samples, RMS=\(String(format: "%.6f", rms))")
                }
            }
            if bufCount % 5 == 0 {
                DispatchQueue.main.async { self.vadLevel = min(rms / 0.05, 1.0) }
            }

            self.debugMicBuffer.append(contentsOf: samples)
            // Keep the C++ pipeline's audio clock continuous, but do not let
            // speaker playback become a new VAD speech turn.
            self.pipeline?.pushAudio(
                EchoMicrophoneGate.samplesToPush(samples, muted: self.isSpeaking))
        }

        do {
            try engine.start()
            audioEngine = engine

            let mixerRate = engine.mainMixerNode.outputFormat(forBus: 0).sampleRate
            guard let playerFmt = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: mixerRate,
                channels: 1, interleaved: false
            ) else { return }
            player.attach(to: engine, format: playerFmt)
            appendLog("[Mic] Started (\(Int(vpFormat.sampleRate))Hz, \(vpFormat.channelCount)ch)")
        } catch {
            appendLog("[Mic] Error: \(error.localizedDescription)")
        }
    }

    private func stopMicrophone() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        if let engine = audioEngine {
            player.detach(from: engine)
            engine.stop()
        }
        audioEngine = nil
    }

    // MARK: - Debug

    private func saveDebugFiles() {
        let tmp = FileManager.default.temporaryDirectory
        for (buf, name, sr) in [
            (debugMicBuffer, "echo_debug_mic.wav", 16000),
            (debugTTSBuffer, "echo_debug_tts.wav", 24000),
        ] where !buf.isEmpty {
            let url = tmp.appendingPathComponent(name)
            try? WAVWriter.write(samples: buf, sampleRate: sr, to: url)
            appendLog("[Debug] Saved \(name) (\(String(format: "%.1f", Double(buf.count) / Double(sr)))s)")
        }
        debugMicBuffer = []
        debugTTSBuffer = []
    }

    private static let logFileURL: URL = {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("echo_debug.log")
        try? "".write(to: url, atomically: true, encoding: .utf8)
        return url
    }()

    private func appendLog(_ message: String) {
        log.append(message)
        if log.count > 50 { log.removeFirst(log.count - 50) }
        let line = "[\(Date())] \(message)\n"
        if let handle = try? FileHandle(forWritingTo: Self.logFileURL) {
            handle.seekToEndOfFile()
            handle.write(line.data(using: .utf8)!)
            handle.closeFile()
        }
    }
}

#endif
