import AVFoundation
import CoreML
import Foundation
import os
import Observation
import KokoroTTS
import ParakeetASR
import SpeechVAD
import SpeechCore
import AudioCommon

/// Apple built-in TTS for simulator — uses AVSpeechSynthesizer.speak() which plays
/// directly through speakers. The .write() API produces empty buffers on simulator.
final class AppleTTSModel: NSObject, SpeechGenerationModel, AVSpeechSynthesizerDelegate {
    var sampleRate: Int { 24000 }
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<[Float], Error>?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func generate(text: String, language: String?) async throws -> [Float] {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language ?? "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        return try await withCheckedThrowingContinuation { cont in
            self.continuation = cont
            synthesizer.speak(utterance)
        }
    }

    private func finish() {
        // AVSpeechSynthesizer plays audio directly — return empty samples
        // since the pipeline doesn't need to play them via StreamingAudioPlayer.
        continuation?.resume(returning: [Float](repeating: 0, count: 2400))
        continuation = nil
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finish()
    }

    /// Audio session interruption / explicit cancel — also resume so the
    /// upstream pipeline doesn't stay in `isGenerating` forever.
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finish()
    }
}

enum MessageRole { case user, assistant, system }

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: MessageRole
    var text: String
    let timestamp = Date()
}

private let pipelineLog = Logger(subsystem: "audio.soniqo.iOSEchoDemo", category: "Pipeline")

@Observable
@MainActor
final class CompanionChatViewModel {
    // MARK: - UI State

    var messages: [ChatBubbleMessage] = []
    var inputText = ""
    var isLoading = false
    var isGenerating = false
    var isListening = false
    var isSpeechDetected = false
    var pipelineState = "idle"
    var audioLevel: Float = 0
    var loadProgress: Double = 0
    var loadingStatus = ""
    var errorMessage: String?
    /// Which compute backend the loaded ASR encoder is using ("ANE", "GPU",
    /// "CPU"). Updated when models load. Surfaced in the diagnostics view.
    var asrBackend = "—"

    private var _modelsLoaded = false
    var modelsLoaded: Bool { _modelsLoaded }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Private State

    private var vadModel: SileroVADModel?
    private var sttModel: ParakeetASRModel?
    private var ttsModel: (any SpeechGenerationModel)?
    private var pipeline: VoicePipeline?
    /// Brief post-playback gate to swallow any residual echo / decay in the
    /// simulator's host-audio loop. With `isGenerating` already gating the
    /// mic during TTS playback, only the speaker→mic decay tail needs covering.
    private let pipelinePostPlaybackGuard: Double = 0.5
    /// Force-cut threshold — slightly under `maxUtteranceDuration` so the
    /// recovery cooldown kicks in when VAD reports a near-MAX utterance.
    private let forceCutThreshold: Double = 4.5
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var isSpeaking = false
    private var speechStartTime: CFAbsoluteTime = 0
    private var wasForceCut = false
    private var pipelineCooldownEnd: CFAbsoluteTime = 0
    private var lastResponseAudioDuration: Double = 0
    private var responseAudioStartTime: CFAbsoluteTime = 0
    private var micRecordBuffer: [Float] = []
    private var ttsRecordBuffer: [Float] = []
    private var debugLog: [String] = []
    /// Decaying peak amplitude for the simple AGC applied before pushing
    /// audio to the pipeline. The simulator's host-mic capture varies in
    /// gain across a session — quiet trailing utterances dropped below
    /// Silero VAD's onset threshold even at `vadOnset = 0.2`. AGC tracks
    /// the loudest recent sample (with exponential decay) and scales each
    /// buffer toward a target peak, capped at 10× to avoid amplifying
    /// pure silence to noise.
    private var agcRecentPeak: Float = 0

    private func dbg(_ msg: String) {
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent().truncatingRemainder(dividingBy: 1000))
        let line = "[\(ts)] \(msg)"
        debugLog.append(line)
        pipelineLog.warning("\(line, privacy: .public)")
    }

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            loadingStatus = "Loading VAD..."
            loadProgress = 0.05
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.05 + progress * 0.15
                        if !status.isEmpty { self?.loadingStatus = "VAD: \(status)" }
                    }
                }
            }.value

            // Pre-download and compile STT model (~500MB).
            // Without this, the first speech triggers a download that blocks
            // the pipeline worker thread — no transcriptions until complete.
            loadingStatus = "Downloading ASR model..."
            loadProgress = 0.2
            sttModel = try await Task.detached {
                try await ParakeetASRModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.2 + progress * 0.4
                        if !status.isEmpty { self?.loadingStatus = "ASR: \(status)" }
                    }
                }
            }.value
            if let units = sttModel?.encoderComputeUnits {
                asrBackend = Self.computeUnitsLabel(units)
            }

            // Load TTS model
            loadingStatus = "Loading TTS..."
            loadProgress = 0.6
            #if targetEnvironment(simulator)
            // Simulator: Apple built-in TTS (CoreML models too slow on CPU-only simulator)
            ttsModel = AppleTTSModel()
            loadProgress = 0.95
            #else
            // Device: Kokoro CoreML (fast on ANE/GPU)
            ttsModel = try await Task.detached {
                try await KokoroTTSModel.fromPretrained { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.6 + progress * 0.35
                        if !status.isEmpty { self?.loadingStatus = "TTS: \(status)" }
                    }
                }
            }.value
            #endif

            loadProgress = 1.0
            loadingStatus = "Ready"
            _modelsLoaded = true
        } catch {
            errorMessage = "Load failed: \(error.localizedDescription)"
        }

        isLoading = false
    }

    // MARK: - Pipeline Start/Stop

    func startListening() {
        guard !isListening, let vad = vadModel,
              let stt = sttModel, let tts = ttsModel else { return }

        var config = PipelineConfig()
        config.mode = .echo  // ASR → TTS, no LLM
        config.allowInterruptions = false  // No AEC — can't distinguish user from speaker
        config.minSilenceDuration = 0.6
        config.maxUtteranceDuration = 5.0   // Matches iOS Parakeet encoder (5s max, single fixed shape)
        config.maxResponseDuration = 5.0   // Cap TTS output to prevent repetition loops
        // Disable eager STT — it transcribes mid-speech and discards the
        // result if a new `speechStarted` fires before STT completes. On
        // real iPhone audio (loud + Silero hysteresis), a 2-3 s utterance
        // sometimes splits into 2.0 s + 0.7 s with the gap interpreted as
        // a new turn, so the eager result for the first 2.0 s gets
        // discarded and the second 0.7 s produces an empty transcription
        // that strands the UI in "transcribing…". Latency cost is small
        // for a voice echo demo.
        config.eagerSTT = false
        config.warmupSTT = false
        // Short pre-roll — long pre-roll (2.0 s) of leading silence before
        // brief utterances flips Parakeet TDT v3's auto language detection
        // into Russian/etc. and produces phonetic transliteration. 0.3 s
        // is enough to capture the consonant onset without dominating the
        // mel input.
        config.preSpeechBufferDuration = 0.3
        config.postPlaybackGuard = 2.0  // Suppress VAD for 2s after TTS to prevent echo feedback (no AEC yet)
        // The simulator's host-mic capture can be quieter than a real
        // device. Default Silero `vadOnset = 0.5` misses utterances around
        // RMS 0.02, which surfaces as "occasional missed phrase" — the
        // user has to repeat. Lower the onset threshold so VAD triggers
        // on quieter speech; the isGenerating gate above already prevents
        // false positives during TTS playback.
        config.vadOnset = 0.2
        config.vadOffset = 0.15

        pipeline = VoicePipeline(
            stt: stt,
            tts: tts,
            vad: vad,
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )

        pipelineLog.warning("[START] echo pipeline created")

        pipeline?.start()
        isListening = true
        pipelineState = "listening"
        diagnostics.start()
        startMicrophone()
        pipelineLog.warning("[START] mic started, pipeline running")
    }

    func stopListening() {
        diagnostics.stop()
        stopMicrophone()
        pipeline?.stop()
        pipeline = nil
        isListening = false
        isGenerating = false
        isSpeechDetected = false
        isSpeaking = false
        audioLevel = 0
        pipelineState = "idle"
        saveDebugRecording()
    }

    private func saveDebugRecording() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("debug_audio")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        if !micRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("mic_debug.wav")
            writeWAV(samples: micRecordBuffer, sampleRate: 16000, to: url)
            pipelineLog.warning("DEBUG MIC: \(url.path) (\(self.micRecordBuffer.count / 16000)s)")
            micRecordBuffer.removeAll()
        }

        if !ttsRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("tts_debug.wav")
            writeWAV(samples: ttsRecordBuffer, sampleRate: 24000, to: url)
            pipelineLog.warning("DEBUG TTS: \(url.path) (\(self.ttsRecordBuffer.count / 24000)s)")
            ttsRecordBuffer.removeAll()
        }

        if !debugLog.isEmpty {
            let logUrl = dir.appendingPathComponent("pipeline_debug.log")
            try? debugLog.joined(separator: "\n").write(to: logUrl, atomically: true, encoding: .utf8)
            debugLog.removeAll()
        }
    }

    private func writeWAV(samples: [Float], sampleRate: Int, to url: URL) {
        var data = Data()
        let dataSize = samples.count * 2
        data.append(contentsOf: "RIFF".utf8)
        var fileSize = UInt32(36 + dataSize); data.append(Data(bytes: &fileSize, count: 4))
        data.append(contentsOf: "WAVE".utf8)
        data.append(contentsOf: "fmt ".utf8)
        var fmtSize: UInt32 = 16; data.append(Data(bytes: &fmtSize, count: 4))
        var fmt: UInt16 = 1; data.append(Data(bytes: &fmt, count: 2))
        var ch: UInt16 = 1; data.append(Data(bytes: &ch, count: 2))
        var sr = UInt32(sampleRate); data.append(Data(bytes: &sr, count: 4))
        var byteRate = UInt32(sampleRate * 2); data.append(Data(bytes: &byteRate, count: 4))
        var blockAlign: UInt16 = 2; data.append(Data(bytes: &blockAlign, count: 2))
        var bps: UInt16 = 16; data.append(Data(bytes: &bps, count: 2))
        data.append(contentsOf: "data".utf8)
        var dSize = UInt32(dataSize); data.append(Data(bytes: &dSize, count: 4))
        for s in samples {
            var pcm = Int16(max(-1, min(1, s)) * 32767)
            data.append(Data(bytes: &pcm, count: 2))
        }
        try? data.write(to: url)
    }

    // MARK: - Pipeline Events

    private func handleEvent(_ event: PipelineEvent) {
        switch event {
        case .sessionCreated:
            dbg("sessionCreated")

        case .speechStarted:
            // Cooldown applies to every response (force-cut or not) — gate
            // the handler uniformly. The mic-tap silence-push usually
            // prevents VAD from firing during cooldown, but if it slips
            // through (e.g. the C++ pipeline detected onset before our
            // cooldown engaged) we drop it here as a safety net.
            let now = CFAbsoluteTimeGetCurrent()
            if now < pipelineCooldownEnd {
                dbg("speechStarted IGNORED (cooldown, \(String(format: "%.1f", pipelineCooldownEnd - now))s remaining)")
                speechStartTime = now
                return
            }
            wasForceCut = false
            dbg("speechStarted")
            isSpeechDetected = true
            speechStartTime = now
            pipelineState = "listening..."

        case .speechEnded:
            // Only check force-cut if we're actually tracking speech
            guard isSpeechDetected else {
                dbg("speechEnded IGNORED (not tracking)")
                return
            }
            let elapsed = CFAbsoluteTimeGetCurrent() - speechStartTime
            wasForceCut = elapsed >= forceCutThreshold
            if wasForceCut {
                // Initial cooldown until responseDone extends it
                pipelineCooldownEnd = .greatestFiniteMagnitude
                dbg("speechEnded (MAX LENGTH after \(String(format: "%.1f", elapsed))s)")
                pipelineState = "max length reached, transcribing..."
                messages.append(ChatBubbleMessage(role: .system,
                    text: "Recording limit reached (\(Int(elapsed))s). Transcribing what was captured."))
            } else {
                dbg("speechEnded (\(String(format: "%.1f", elapsed))s)")
                pipelineState = "transcribing..."
            }
            isSpeechDetected = false

        case .transcriptionCompleted(let text, let lang, let conf):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            dbg("transcription: '\(trimmed)' lang=\(lang ?? "-") conf=\(String(format: "%.2f", conf))")
            guard !trimmed.isEmpty else {
                // Any empty transcription strands the UI in "transcribing…"
                // without a recovery path — reset state so the next
                // utterance can come through. Force-cut additionally
                // clears the cooldown lock that was set on speechEnded.
                if wasForceCut {
                    wasForceCut = false
                    pipelineCooldownEnd = 0
                    dbg("empty transcription after force-cut — cooldown cleared")
                }
                pipelineState = "listening"
                return
            }
            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            // Echo mode: transcription is sent directly to TTS by the pipeline
            pipelineState = "speaking..."
            isGenerating = true

        case .responseCreated:
            dbg("responseCreated")
            lastResponseAudioDuration = 0
            // In echo mode, the response IS the transcription — show it as assistant echo
            if let lastUser = messages.last, lastUser.role == .user {
                messages.append(ChatBubbleMessage(role: .assistant, text: "🔊 \(lastUser.text)"))
            }

        case .responseInterrupted:
            dbg("responseInterrupted")
            player.fadeOutAndStop()
            isSpeaking = false
            isGenerating = false
            pipelineState = "listening"

        case .responseAudioDelta(let samples):
            if !isSpeaking { responseAudioStartTime = CFAbsoluteTimeGetCurrent() }
            isSpeaking = true
            lastResponseAudioDuration += Double(samples.count) / 24000.0
            pipelineState = "speaking..."
            dbg("audioDelta: \(samples.count) samples (\(String(format: "%.2f", Double(samples.count)/24000))s)")
            // Capture for tts_debug.wav so we can inspect what's being
            // sent to the speaker (helps diagnose audio artifacts like
            // the end-of-synthesis click).
            ttsRecordBuffer.append(contentsOf: samples)
            do { try player.play(samples: samples, sampleRate: 24000) }
            catch { dbg("playback error: \(error)") }

        case .responseDone:
            // Guard = remaining playback time + postPlaybackGuard
            let elapsedSinceAudioStart = CFAbsoluteTimeGetCurrent() - responseAudioStartTime
            let remainingPlayback = max(0, lastResponseAudioDuration - elapsedSinceAudioStart)
            let guard_ = remainingPlayback + pipelinePostPlaybackGuard
            dbg("responseDone (audio=\(String(format: "%.1f", lastResponseAudioDuration))s, guard=\(String(format: "%.1f", guard_))s)")
            isGenerating = false
            isSpeaking = false
            player.markGenerationComplete()
            // Always apply the cooldown (not just on force-cut) so the mic
            // gate in the audio tap covers post-TTS residual echo on the
            // iOS Simulator. On a real device the cooldown is mostly
            // redundant (low acoustic coupling between speaker and mic)
            // but harmless.
            pipelineCooldownEnd = CFAbsoluteTimeGetCurrent() + guard_
            lastResponseAudioDuration = 0
            resumeAfterResponse()

        case .toolCallStarted, .toolCallCompleted:
            break

        case .error(let msg):
            dbg("ERROR: \(msg)")
            errorMessage = msg
            pipelineState = "error"
            isGenerating = false
            pipeline?.resumeListening()
        }
    }

    private func resumeAfterResponse() {
        guard isListening else { return }
        isSpeaking = false
        pipeline?.resumeListening()
        pipelineState = "listening"
    }

    // MARK: - Microphone

    private func startMicrophone() {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()

        switch AVAudioApplication.shared.recordPermission {
        case .undetermined:
            AVAudioApplication.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    if granted {
                        self?.startMicrophone()
                    } else {
                        self?.errorMessage = "Microphone permission denied"
                    }
                }
            }
            return
        case .denied:
            errorMessage = "Microphone permission denied. Enable in Settings."
            return
        case .granted:
            break
        @unknown default:
            break
        }

        do {
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
            // `.defaultToSpeaker` is honoured at category-set time but iOS
            // can quietly route .playAndRecord audio to the earpiece at
            // runtime (especially after the mic is hot). Force the route
            // to the bottom speaker so TTS playback is audible.
            try session.overrideOutputAudioPort(.speaker)
        } catch {
            errorMessage = "Mic access failed: \(error.localizedDescription)"
            return
        }
        #endif

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16000,
            channels: 1, interleaved: false
        ) else { return }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: hwFormat.sampleRate,
            channels: 1, interleaved: false
        ) else { return }

        guard let resampler = AVAudioConverter(from: monoFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: hwFormat) { [weak self] buffer, _ in
            guard let self else { return }
            guard let srcData = buffer.floatChannelData else { return }
            let frameLen = Int(buffer.frameLength)
            guard frameLen > 0 else { return }

            guard let monoBuffer = AVAudioPCMBuffer(pcmFormat: monoFormat,
                                                     frameCapacity: buffer.frameCapacity) else { return }
            monoBuffer.frameLength = buffer.frameLength
            memcpy(monoBuffer.floatChannelData![0], srcData[0], frameLen * MemoryLayout<Float>.size)

            let outFrameCount = AVAudioFrameCount(Double(frameLen) * 16000.0 / hwFormat.sampleRate)
            guard outFrameCount > 0,
                  let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat,
                                                    frameCapacity: outFrameCount) else { return }

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

            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = sqrt(sum / max(Float(count), 1))
            DispatchQueue.main.async {
                self.audioLevel = rms
                self.diagnostics.updateVAD(rms)
            }

            self.micRecordBuffer.append(contentsOf: samples)
            let maxMicSamples = 16000 * 60
            if self.micRecordBuffer.count > maxMicSamples {
                self.micRecordBuffer.removeFirst(self.micRecordBuffer.count - maxMicSamples)
            }

            // While a response is being generated/played, or during the
            // post-playback cooldown, replace the live mic samples with
            // silence before pushing to the pipeline. We can't simply
            // `return` because the C++ VAD relies on a continuous audio
            // stream — dropping a chunk creates a time discontinuity that
            // appears to leave VAD in a state where the next real speech
            // burst doesn't fire `speechStarted` until after a long delay.
            // Pushing zeros keeps the buffer aligned in real time and
            // lets VAD transition cleanly from silence to speech.
            //
            // We use `isGenerating` (set on transcriptionCompleted,
            // cleared on responseDone) rather than `isSpeaking` because
            // AVSpeechSynthesizer starts emitting audio immediately on
            // `speak()`, before the pipeline emits `responseAudioDelta` —
            // `isSpeaking` only flips true *after* didFinish, by which
            // time the relevant playback has already happened. The
            // cooldown also covers force-cut recovery (its original use).
            let now = CFAbsoluteTimeGetCurrent()
            if self.isGenerating || now < self.pipelineCooldownEnd {
                // Reset AGC tracker so the first post-cooldown buffer
                // doesn't inherit a stale peak from before TTS.
                self.agcRecentPeak = 0
                let silence = [Float](repeating: 0, count: samples.count)
                self.pipeline?.pushAudio(silence)
                return
            }
            self.pipeline?.pushAudio(self.applyAGC(to: samples))
        }

        guard let playerFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 24000,
            channels: 1, interleaved: false
        ) else { return }
        player.attach(to: engine, format: playerFormat)

        do {
            try engine.start()
            player.startPlayback()
            audioEngine = engine
        } catch {
            errorMessage = "Mic error: \(error.localizedDescription)"
        }
    }

    /// Friendly label for an `MLComputeUnits` value, used to display the
    /// ASR backend in the diagnostics view.
    private static func computeUnitsLabel(_ u: MLComputeUnits) -> String {
        switch u {
        case .cpuOnly:               return "CPU"
        case .cpuAndGPU:             return "GPU"
        case .all:                   return "ANE"
        case .cpuAndNeuralEngine:    return "ANE"
        @unknown default:            return "?"
        }
    }

    /// Per-buffer automatic gain control. Tracks a decaying peak of the
    /// loudest recent sample and scales the buffer so that peak hits a
    /// target level (~0.5). Caps at 10× to avoid amplifying silence /
    /// background noise to speech-trigger levels. The pipeline's silence
    /// gate ensures this only runs on real mic input (not TTS bleed).
    private func applyAGC(to samples: [Float]) -> [Float] {
        let targetPeak: Float = 0.5
        let decay: Float = 0.95   // peak persists ~ -0.4 dB / buffer
        let maxGain: Float = 10
        let minPeakForGain: Float = 0.005   // below this is silence — no gain

        var bufferPeak: Float = 0
        for s in samples {
            let abs = s < 0 ? -s : s
            if abs > bufferPeak { bufferPeak = abs }
        }
        agcRecentPeak = max(bufferPeak, agcRecentPeak * decay)

        guard agcRecentPeak > minPeakForGain else { return samples }
        let gain = min(targetPeak / agcRecentPeak, maxGain)
        if gain <= 1.05 { return samples }   // already loud enough
        return samples.map { $0 * gain }
    }

    private func stopMicrophone() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        if let engine = audioEngine {
            player.detach(from: engine)
            engine.stop()
        }
        audioEngine = nil
    }

    // MARK: - Text Input

    func send(_ text: String) {
        inputText = ""
        guard isListening else {
            messages.append(ChatBubbleMessage(role: .user, text: text))
            return
        }
        pipeline?.pushText(text)
    }

    func clearChat() {
        messages = []
    }
}
