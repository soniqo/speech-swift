import CSpeechCore
import AudioCommon
import Foundation

// MARK: - Pipeline Types

/// Voice pipeline mode.
public enum PipelineMode: Int {
    /// Full voice agent: audio → VAD → STT → LLM → TTS → audio
    case voicePipeline = 0
    /// Speech-to-text only: audio → VAD → STT → text
    case transcribeOnly = 1
    /// Echo test: audio → VAD → STT → TTS → audio
    case echo = 2
}

/// Pipeline state.
public enum PipelineState: Int {
    case idle = 0
    case listening = 1
    case transcribing = 2
    case thinking = 3
    case speaking = 4
}

/// Events emitted by the voice pipeline.
public enum PipelineEvent {
    case sessionCreated
    case speechStarted
    case speechEnded
    case transcriptionCompleted(text: String, confidence: Float)
    case responseCreated
    case responseAudioDelta(samples: [Float])
    case responseDone
    case error(String)
}

/// Pipeline configuration.
public struct PipelineConfig {
    public var vadOnset: Float = 0.5
    public var vadOffset: Float = 0.35
    public var minSpeechDuration: Float = 0.25
    public var minSilenceDuration: Float = 0.1
    public var allowInterruptions: Bool = true
    public var interruptionRecoveryTimeout: Float = 0.4
    public var maxUtteranceDuration: Float = 15.0
    public var language: String = ""
    public var mode: PipelineMode = .echo

    public static let `default` = PipelineConfig()

    public init() {}
}

// MARK: - LLM Protocol

/// Protocol for language model integration with the voice pipeline.
public protocol PipelineLLM: AnyObject {
    /// Chat with streaming token output.
    func chat(messages: [(role: MessageRole, content: String)],
              onToken: @escaping (String, Bool) -> Void)
    /// Cancel in-progress generation.
    func cancel()
}

/// Message roles matching speech-core's sc_role_t.
public enum MessageRole: Int {
    case system = 0
    case user = 1
    case assistant = 2
    case tool = 3
}

// MARK: - Bridge Classes

/// Bridges a SpeechRecognitionModel to the C vtable.
private final class STTBridge {
    let model: SpeechRecognitionModel
    var lastText: [CChar] = []  // keeps C string alive between calls

    init(_ model: SpeechRecognitionModel) { self.model = model }
}

/// Bridges a SpeechGenerationModel to the C vtable.
private final class TTSBridge {
    let model: SpeechGenerationModel
    var cancelled = false

    init(_ model: SpeechGenerationModel) { self.model = model }
}

/// Bridges a StreamingVADProvider to the C vtable.
private final class VADBridge {
    let model: StreamingVADProvider

    init(_ model: StreamingVADProvider) { self.model = model }
}

/// Bridges a PipelineLLM to the C vtable.
private final class LLMBridge {
    let model: PipelineLLM

    init(_ model: PipelineLLM) { self.model = model }
}

/// Bridges the event callback.
private final class EventBridge {
    let handler: (PipelineEvent) -> Void

    init(_ handler: @escaping (PipelineEvent) -> Void) { self.handler = handler }
}

// MARK: - VoicePipeline

/// Swift wrapper around speech-core's C voice pipeline.
///
/// Connects Swift ML models (STT, TTS, VAD, LLM) to the C++ pipeline engine
/// via the C vtable FFI. The pipeline manages the full conversational loop:
///
/// ```
/// audio → VAD → STT → LLM → TTS → audio
/// ```
///
/// Usage:
/// ```swift
/// let pipeline = VoicePipeline(
///     stt: asrModel,
///     tts: ttsModel,
///     vad: sileroVAD,
///     config: .init(mode: .echo),
///     onEvent: { event in print(event) }
/// )
/// pipeline.start()
/// pipeline.pushAudio(micSamples)
/// ```
public final class VoicePipeline {

    private var handle: sc_pipeline_t?

    // Bridges must stay alive for the lifetime of the pipeline
    private let sttBridge: STTBridge
    private let ttsBridge: TTSBridge
    private let vadBridge: VADBridge
    private var llmBridge: LLMBridge?
    private let eventBridge: EventBridge

    /// Create a voice pipeline.
    ///
    /// - Parameters:
    ///   - stt: Speech recognition model (implements `SpeechRecognitionModel`)
    ///   - tts: Speech generation model (implements `SpeechGenerationModel`)
    ///   - vad: Streaming VAD model (implements `StreamingVADProvider`)
    ///   - llm: Language model (optional, required for `.voicePipeline` mode)
    ///   - config: Pipeline configuration
    ///   - onEvent: Event callback
    public init(
        stt: SpeechRecognitionModel,
        tts: SpeechGenerationModel,
        vad: StreamingVADProvider,
        llm: PipelineLLM? = nil,
        config: PipelineConfig = .default,
        onEvent: @escaping (PipelineEvent) -> Void
    ) {
        self.sttBridge = STTBridge(stt)
        self.ttsBridge = TTSBridge(tts)
        self.vadBridge = VADBridge(vad)
        self.eventBridge = EventBridge(onEvent)
        if let llm { self.llmBridge = LLMBridge(llm) }

        let sttVtable = Self.makeSTTVtable(sttBridge)
        let ttsVtable = Self.makeTTSVtable(ttsBridge)
        let vadVtable = Self.makeVADVtable(vadBridge)

        var cConfig = sc_config_default()
        cConfig.vad_onset = config.vadOnset
        cConfig.vad_offset = config.vadOffset
        cConfig.min_speech_duration = config.minSpeechDuration
        cConfig.min_silence_duration = config.minSilenceDuration
        cConfig.allow_interruptions = config.allowInterruptions
        cConfig.interruption_recovery_timeout = config.interruptionRecoveryTimeout
        cConfig.max_utterance_duration = config.maxUtteranceDuration
        cConfig.mode = sc_mode_t(rawValue: UInt32(config.mode.rawValue))

        let eventCtx = Unmanaged.passRetained(eventBridge).toOpaque()

        if let llmBridge {
            var llmVtable = Self.makeLLMVtable(llmBridge)
            config.language.withCString { langPtr in
                cConfig.language = langPtr
                handle = sc_pipeline_create(
                    sttVtable, ttsVtable, &llmVtable, vadVtable,
                    cConfig, Self.eventCallback, eventCtx)
            }
        } else {
            config.language.withCString { langPtr in
                cConfig.language = langPtr
                handle = sc_pipeline_create(
                    sttVtable, ttsVtable, nil, vadVtable,
                    cConfig, Self.eventCallback, eventCtx)
            }
        }
    }

    deinit {
        if handle != nil {
            sc_pipeline_stop(handle)
            sc_pipeline_destroy(handle)
        }
        // Balance the passRetained on bridges
        Unmanaged.passUnretained(sttBridge).release()
        Unmanaged.passUnretained(ttsBridge).release()
        Unmanaged.passUnretained(vadBridge).release()
        if let llmBridge { Unmanaged.passUnretained(llmBridge).release() }
        Unmanaged.passUnretained(eventBridge).release()
    }

    // MARK: - Public API

    public func start() {
        sc_pipeline_start(handle)
    }

    public func stop() {
        sc_pipeline_stop(handle)
    }

    /// Feed microphone audio (Float32 PCM at VAD sample rate).
    public func pushAudio(_ samples: [Float]) {
        samples.withUnsafeBufferPointer { buf in
            sc_pipeline_push_audio(handle, buf.baseAddress, buf.count)
        }
    }

    /// Inject text directly — bypasses STT, sent to LLM.
    public func pushText(_ text: String) {
        sc_pipeline_push_text(handle, text)
    }

    public var state: PipelineState {
        PipelineState(rawValue: Int(sc_pipeline_state(handle).rawValue)) ?? .idle
    }

    public var isRunning: Bool {
        sc_pipeline_is_running(handle)
    }

    // MARK: - Event Callback

    private static let eventCallback: sc_event_fn = { eventPtr, ctx in
        guard let eventPtr, let ctx else { return }
        let bridge = Unmanaged<EventBridge>.fromOpaque(ctx).takeUnretainedValue()
        let e = eventPtr.pointee

        let event: PipelineEvent
        switch e.type {
        case SC_EVENT_SESSION_CREATED:
            event = .sessionCreated
        case SC_EVENT_SPEECH_STARTED:
            event = .speechStarted
        case SC_EVENT_SPEECH_ENDED:
            event = .speechEnded
        case SC_EVENT_TRANSCRIPTION_COMPLETED:
            let text = e.text.map { String(cString: $0) } ?? ""
            event = .transcriptionCompleted(text: text, confidence: e.confidence)
        case SC_EVENT_RESPONSE_CREATED:
            event = .responseCreated
        case SC_EVENT_RESPONSE_AUDIO_DELTA:
            event = .responseAudioDelta(samples: pcm16ToFloat32(e.audio_data, count: e.audio_data_length))
        case SC_EVENT_RESPONSE_DONE:
            event = .responseDone
        case SC_EVENT_ERROR:
            let text = e.text.map { String(cString: $0) } ?? "Unknown error"
            event = .error(text)
        default:
            return
        }

        bridge.handler(event)
    }

    // MARK: - STT Vtable

    private static func makeSTTVtable(_ bridge: STTBridge) -> sc_stt_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_stt_vtable_t(
            context: ctx,
            transcribe: { ctx, audio, length, sampleRate in
                let bridge = Unmanaged<STTBridge>.fromOpaque(ctx!).takeUnretainedValue()
                let samples = Array(UnsafeBufferPointer(start: audio, count: length))
                let text = bridge.model.transcribe(
                    audio: samples,
                    sampleRate: Int(sampleRate),
                    language: nil)
                // Store as C string — valid until next transcribe call
                bridge.lastText = Array(text.utf8CString)
                return bridge.lastText.withUnsafeBufferPointer { buf in
                    sc_transcription_result_t(
                        text: buf.baseAddress,
                        confidence: 1.0,
                        start_time: 0,
                        end_time: 0)
                }
            },
            input_sample_rate: { ctx in
                let bridge = Unmanaged<STTBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.inputSampleRate)
            }
        )
    }

    // MARK: - TTS Vtable

    private static func makeTTSVtable(_ bridge: TTSBridge) -> sc_tts_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_tts_vtable_t(
            context: ctx,
            synthesize: { ctx, text, language, onChunk, chunkCtx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.cancelled = false
                let textStr = String(cString: text!)
                let langStr = language.map { String(cString: $0) } ?? ""

                // Synchronous: generate full audio and deliver as single chunk
                let semaphore = DispatchSemaphore(value: 0)
                Task {
                    defer { semaphore.signal() }
                    do {
                        let audio = try await bridge.model.generate(
                            text: textStr, language: langStr.isEmpty ? nil : langStr)
                        guard !bridge.cancelled else { return }
                        audio.withUnsafeBufferPointer { buf in
                            onChunk?(buf.baseAddress, buf.count, true, chunkCtx)
                        }
                    } catch {
                        // Deliver empty final chunk on error
                        onChunk?(nil, 0, true, chunkCtx)
                    }
                }
                semaphore.wait()
            },
            output_sample_rate: { ctx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.sampleRate)
            },
            cancel: { ctx in
                let bridge = Unmanaged<TTSBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.cancelled = true
            }
        )
    }

    // MARK: - VAD Vtable

    private static func makeVADVtable(_ bridge: VADBridge) -> sc_vad_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_vad_vtable_t(
            context: ctx,
            process_chunk: { ctx, samples, length in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                let chunk = Array(UnsafeBufferPointer(start: samples, count: length))
                return bridge.model.processChunk(chunk)
            },
            reset: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.model.resetState()
            },
            input_sample_rate: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return Int32(bridge.model.inputSampleRate)
            },
            chunk_size: { ctx in
                let bridge = Unmanaged<VADBridge>.fromOpaque(ctx!).takeUnretainedValue()
                return bridge.model.chunkSize
            }
        )
    }

    // MARK: - LLM Vtable

    private static func makeLLMVtable(_ bridge: LLMBridge) -> sc_llm_vtable_t {
        let ctx = Unmanaged.passRetained(bridge).toOpaque()
        return sc_llm_vtable_t(
            context: ctx,
            chat: { ctx, messages, count, onToken, tokenCtx in
                let bridge = Unmanaged<LLMBridge>.fromOpaque(ctx!).takeUnretainedValue()

                // Convert C messages to Swift
                var swiftMessages: [(role: MessageRole, content: String)] = []
                for i in 0..<count {
                    let msg = messages![i]
                    let role = MessageRole(rawValue: Int(msg.role.rawValue)) ?? .user
                    let content = msg.content.map { String(cString: $0) } ?? ""
                    swiftMessages.append((role: role, content: content))
                }

                bridge.model.chat(messages: swiftMessages) { token, isFinal in
                    token.withCString { cstr in
                        onToken?(cstr, isFinal, tokenCtx)
                    }
                }
            },
            cancel: { ctx in
                let bridge = Unmanaged<LLMBridge>.fromOpaque(ctx!).takeUnretainedValue()
                bridge.model.cancel()
            }
        )
    }

    // MARK: - PCM Conversion

    /// Convert PCM16-LE bytes to Float32 samples.
    private static func pcm16ToFloat32(_ data: UnsafePointer<UInt8>?, count: Int) -> [Float] {
        guard let data, count >= 2 else { return [] }
        let sampleCount = count / 2
        return (0..<sampleCount).map { i in
            let lo = UInt16(data[i * 2])
            let hi = UInt16(data[i * 2 + 1])
            let raw = Int16(bitPattern: lo | (hi << 8))
            return Float(raw) / 32768.0
        }
    }
}
