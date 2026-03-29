import AVFoundation
import CoreML
import Foundation
import os
import Observation
import Qwen3Chat
import KokoroTTS
import ParakeetASR
import SpeechVAD
import SpeechCore
import AudioCommon

/// Message displayed in chat UI.
struct ChatBubbleMessage: Identifiable {
    let id = UUID()
    let role: ChatMessage.Role
    var text: String
    let timestamp = Date()
}

private let pipelineLog = Logger(subsystem: "audio.soniqo.TamagotchiDemo", category: "Pipeline")

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

    private var _modelsLoaded = false
    var modelsLoaded: Bool { _modelsLoaded }

    let diagnostics = DiagnosticsMonitor()

    // MARK: - Private State

    private var vadModel: SileroVADModel?
    private var sttModel: ParakeetASRModel?
    private var llmModel: Qwen35PipelineLLM?
    private var ttsModel: KokoroTTSModel?
    private var pipeline: VoicePipeline?
    private var audioEngine: AVAudioEngine?
    private let player = StreamingAudioPlayer()
    private var waitingForPlaybackEnd = false
    /// True while TTS is playing — used for UI state and event suppression.
    private var isSpeaking = false
    private var currentResponseText = ""
    private var currentAssistantIdx: Int?
    private var micRecordBuffer: [Float] = []
    private var ttsRecordBuffer: [Float] = []
    private var debugLog: [String] = []

    private func dbg(_ msg: String) {
        let ts = String(format: "%.3f", CFAbsoluteTimeGetCurrent().truncatingRemainder(dividingBy: 1000))
        let line = "[\(ts)] \(msg)"
        debugLog.append(line)
        pipelineLog.warning("\(line, privacy: .public)")
    }

    private let systemPrompt = "Your name is Tama. Give short direct answers. Do not explain your reasoning."

    // MARK: - Load Models

    func loadModels() async {
        isLoading = true
        errorMessage = nil
        loadProgress = 0

        do {
            // Load all models upfront with warmup — no delays during conversation.

            loadingStatus = "Loading VAD..."
            loadProgress = 0.05
            vadModel = try await Task.detached {
                try await SileroVADModel.fromPretrained(engine: .coreml) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.05 + progress * 0.1
                        if !status.isEmpty { self?.loadingStatus = "VAD: \(status)" }
                    }
                }
            }.value

            loadingStatus = "Loading ASR..."
            loadProgress = 0.15
            let asr = try await Task.detached {
                try await ParakeetASRModel.fromPretrained(
                    modelId: ParakeetASRModel.int8iOSModelId
                ) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.15 + progress * 0.2
                        if !status.isEmpty { self?.loadingStatus = "ASR: \(status)" }
                    }
                }
            }.value
            // Warmup: trigger CoreML compilation with dummy audio
            loadingStatus = "Warming up ASR..."
            loadProgress = 0.38
            // Warmup with 1s of silence — triggers CoreML compilation
            _ = asr.transcribe(audio: [Float](repeating: 0, count: 16000), sampleRate: 16000, language: nil)
            sttModel = asr

            loadingStatus = "Loading LLM..."
            loadProgress = 0.42
            #if targetEnvironment(simulator)
            // Simulator: use mock LLM (CoreML DeltaNet produces garbage without Neural Engine)
            let chatBackend: any Qwen35ChatBackend = MockChatBackend()
            #else
            let chatBackend: any Qwen35ChatBackend = try await Task.detached {
                return try await Qwen35CoreMLChat.fromPretrained(
                    quantization: .int8, computeUnits: .cpuAndNeuralEngine
                ) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.42 + progress * 0.2
                        if !status.isEmpty { self?.loadingStatus = "LLM: \(status)" }
                    }
                } as any Qwen35ChatBackend
            }.value
            #endif
            let sampling = ChatSamplingConfig(
                temperature: 0.5, topK: 30, maxTokens: 30, repetitionPenalty: 1.3)
            let llm = Qwen35PipelineLLM(model: chatBackend, systemPrompt: systemPrompt, sampling: sampling)
            llm.onToken = { [weak self] token in
                DispatchQueue.main.async { self?.appendToken(token) }
            }
            llmModel = llm

            loadingStatus = "Loading TTS..."
            loadProgress = 0.7
            #if targetEnvironment(simulator)
            let ttsUnits: MLComputeUnits = .all  // Simulator: no ANE
            #else
            let ttsUnits: MLComputeUnits = .cpuAndNeuralEngine
            #endif
            let tts = try await Task.detached {
                try await KokoroTTSModel.fromPretrained(
                    modelId: KokoroTTSModel.int8iOSModelId,
                    computeUnits: ttsUnits,
                    loadG2P: false
                ) { progress, status in
                    DispatchQueue.main.async { [weak self] in
                        self?.loadProgress = 0.7 + progress * 0.2
                        if !status.isEmpty { self?.loadingStatus = "TTS: \(status)" }
                    }
                }
            }.value
            ttsModel = tts

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
        guard !isListening, let vad = vadModel else { return }

        let prompt = systemPrompt

        // Use lazy factories so autoUnloadModels can release and re-create
        // models between phases. Re-creation from CoreML cache takes ~0.1-0.3s.
        // Peak memory = largest single model, not sum of all.
        #if targetEnvironment(simulator)
        let ttsUnits: MLComputeUnits = .all
        #else
        let ttsUnits: MLComputeUnits = .cpuAndNeuralEngine
        #endif

        var config = PipelineConfig()
        config.mode = .voicePipeline
        config.allowInterruptions = true
        config.minInterruptionDuration = 1.5  // 1.5s sustained speech to confirm barge-in
        config.minSilenceDuration = 0.8
        config.maxResponseDuration = 15.0
        config.eagerSTT = false               // Wait for confirmed end-of-speech
        config.warmupSTT = false              // Already warmed up during loadModels
        config.preSpeechBufferDuration = 1.5
        config.autoUnloadModels = true        // Unload models between phases

        pipeline = VoicePipeline(
            sttFactory: {
                try await ParakeetASRModel.fromPretrained(
                    modelId: ParakeetASRModel.int8iOSModelId)
            },
            ttsFactory: {
                try await KokoroTTSModel.fromPretrained(
                    modelId: KokoroTTSModel.int8iOSModelId,
                    computeUnits: ttsUnits, loadG2P: false)
            },
            vad: vad,
            llmFactory: { [weak self] in
                #if targetEnvironment(simulator)
                let backend: any Qwen35ChatBackend = MockChatBackend()
                #else
                let backend: any Qwen35ChatBackend = try await Qwen35CoreMLChat.fromPretrained(
                    quantization: .int8, computeUnits: .cpuAndNeuralEngine)
                #endif
                let sampling = ChatSamplingConfig(
                    temperature: 0.5, topK: 30, maxTokens: 30, repetitionPenalty: 1.3)
                let llm = Qwen35PipelineLLM(model: backend, systemPrompt: prompt, sampling: sampling)
                llm.onToken = { token in
                    DispatchQueue.main.async { self?.appendToken(token) }
                }
                return llm
            },
            config: config,
            onEvent: { [weak self] event in
                DispatchQueue.main.async { self?.handleEvent(event) }
            }
        )

        player.onPlaybackFinished = { [weak self] in
            DispatchQueue.main.async { self?.playbackDidFinish() }
        }

        // Enable SpeexDSP echo cancellation on real device only.
        // Simulator has no real speakers/mic — AEC diverges on simulated audio.
        #if !targetEnvironment(simulator)
        pipeline?.enableEchoCancellation(sampleRate: 16000)
        #endif

        // Release preloaded model references — pipeline has its own lazy factories.
        // This frees ~5GB held by the loadModels() phase.
        sttModel = nil
        llmModel = nil
        ttsModel = nil
        pipelineLog.warning("[START] pipeline created, releasing preloaded models")

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
            let dur = micRecordBuffer.count / 16000
            pipelineLog.warning("DEBUG MIC: \(url.path) (\(dur)s)")
            print("DEBUG MIC: \(url.path) (\(dur)s)")
            micRecordBuffer.removeAll()
        }

        if !ttsRecordBuffer.isEmpty {
            let url = dir.appendingPathComponent("tts_debug.wav")
            writeWAV(samples: ttsRecordBuffer, sampleRate: 24000, to: url)
            let dur = ttsRecordBuffer.count / 24000
            pipelineLog.warning("DEBUG TTS: \(url.path) (\(dur)s)")
            print("DEBUG TTS: \(url.path) (\(dur)s)")
            ttsRecordBuffer.removeAll()
        }

        // Save event log
        if !debugLog.isEmpty {
            let logUrl = dir.appendingPathComponent("pipeline_debug.log")
            let logText = debugLog.joined(separator: "\n")
            try? logText.write(to: logUrl, atomically: true, encoding: .utf8)
            print("DEBUG LOG: \(logUrl.path) (\(debugLog.count) events)")
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
            dbg("speechStarted gen=\(isGenerating) speak=\(isSpeaking)")
            isSpeechDetected = true
            pipelineState = "speech detected"
            // Don't cancel LLM/TTS here — let the pipeline handle interruption
            // based on its own config (allowInterruptions). Manual cancellation
            // caused the pipeline to get stuck with empty responses.

        case .speechEnded:
            dbg("speechEnded")
            isSpeechDetected = false
            pipelineState = "transcribing..."

        case .transcriptionCompleted(let text, let lang, let conf):
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            dbg("transcription: '\(trimmed)' lang=\(lang ?? "-") conf=\(String(format: "%.2f", conf))")
            guard !trimmed.isEmpty else {
                dbg("  → empty, skipping")
                return
            }
            // Don't unload STT — CoreML doesn't free memory on nil,
            // reloading creates new buffers that accumulate.
            messages.append(ChatBubbleMessage(role: .user, text: trimmed))
            pipelineState = "thinking..."
            isGenerating = true

        case .responseCreated:
            dbg("responseCreated")
            if currentAssistantIdx == nil {
                currentResponseText = ""
                currentAssistantIdx = messages.count
                messages.append(ChatBubbleMessage(role: .assistant, text: "..."))
                pipelineState = "generating..."
            }

        case .responseInterrupted:
            dbg("responseInterrupted")
            player.fadeOutAndStop()
            isSpeaking = false
            isGenerating = false
            currentAssistantIdx = nil
            pipelineState = "interrupted"

        case .responseAudioDelta(let samples):
            dbg("audioDelta \(samples.count) samples (\(String(format: "%.2f", Float(samples.count) / 24000))s)")
            pipelineState = "speaking..."
            isSpeaking = true
            ttsRecordBuffer.append(contentsOf: samples)
            let maxTtsSamples = 24000 * 30  // Cap at 30s
            if ttsRecordBuffer.count > maxTtsSamples {
                ttsRecordBuffer.removeFirst(ttsRecordBuffer.count - maxTtsSamples)
            }
            do { try player.play(samples: samples, sampleRate: 24000) }
            catch { dbg("playback error: \(error)") }

        case .responseDone:
            dbg("responseDone ttsBuffer=\(ttsRecordBuffer.count) samples")
            isGenerating = false
            currentAssistantIdx = nil
            if player.isPlaying {
                waitingForPlaybackEnd = true
            } else {
                resumeAfterResponse()
            }

        case .toolCallStarted(let name):
            dbg("toolCall: \(name)")

        case .toolCallCompleted:
            break

        case .error(let msg):
            dbg("ERROR: \(msg)")
            errorMessage = msg
            pipelineState = "error"
            isGenerating = false
            pipeline?.resumeListening()
        }
    }

    private func playbackDidFinish() {
        guard waitingForPlaybackEnd else { return }
        waitingForPlaybackEnd = false
        resumeAfterResponse()
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
        do {
            try session.setCategory(.playAndRecord, mode: .default,
                                    options: [.defaultToSpeaker, .allowBluetooth])
            try session.setActive(true)
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

            // Echo cancellation is handled by SpeexDSP inside the pipeline
            // (enableEchoCancellation). No need to drop frames here.

            // Resample to 16kHz mono
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

            // RMS for visual level
            var sum: Float = 0
            for s in samples { sum += s * s }
            let rms = sqrt(sum / max(Float(count), 1))
            DispatchQueue.main.async {
                self.audioLevel = rms
                self.diagnostics.updateVAD(rms)
            }

            // Debug: record mic audio (cap at 60s to prevent memory growth)
            self.micRecordBuffer.append(contentsOf: samples)
            let maxMicSamples = 16000 * 60  // 60s at 16kHz
            if self.micRecordBuffer.count > maxMicSamples {
                self.micRecordBuffer.removeFirst(self.micRecordBuffer.count - maxMicSamples)
            }

            self.pipeline?.pushAudio(samples)
        }

        // Attach TTS player at 24kHz before starting engine
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

    // MARK: - LLM Token Streaming

    func appendToken(_ token: String) {
        if currentAssistantIdx == nil {
            currentResponseText = ""
            currentAssistantIdx = messages.count
            messages.append(ChatBubbleMessage(role: .assistant, text: ""))
            pipelineState = "generating..."
            isGenerating = true
        }
        guard let idx = currentAssistantIdx, idx < messages.count else { return }
        currentResponseText += token
        messages[idx].text = currentResponseText
    }
}
