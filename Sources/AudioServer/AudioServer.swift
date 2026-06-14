import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import NIOCore
import Qwen3ASR
import Qwen3TTS
import CosyVoiceTTS
import ParakeetASR
import NemotronStreamingASR
import OmnilingualASR
import KokoroTTS
import VoxCPM2TTS
import MagpieTTS
import PersonaPlex
import SpeechEnhancement
import AudioCommon

// MARK: - Server

public struct AudioServer {
    let state: ModelState
    let host: String
    let port: Int

    public init(host: String = "127.0.0.1", port: Int = 8080, preload: Bool = false) {
        self.state = ModelState()
        self.host = host
        self.port = port
    }

    public func run() async throws {
        let router = buildRouter()
        let state = self.state
        let wsConfig = WebSocketServerConfiguration(maxFrameSize: 1 << 24)  // 16 MB max frame
        let wsServer: HTTPServerBuilder = .http1WebSocketUpgrade(configuration: wsConfig) { head, _, _ in
            let path = head.path ?? ""
            guard path == "/v1/realtime" else { return .dontUpgrade }
            return .upgrade([:]) { inbound, outbound, _ in
                try await handleRealtimeWS(inbound: inbound, outbound: outbound, state: state)
            }
        }
        let app = Application(
            router: router,
            server: wsServer,
            configuration: .init(address: .hostname(host, port: port)))
        try await app.run()
    }

    public func preloadModels() async throws {
        _ = try await state.loadASR()
        _ = try await state.loadTTS()
        _ = try await state.loadPersonaPlex()
        _ = try await state.loadEnhancer()
    }

    // MARK: - HTTP Routes

    func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()
        let state = self.state

        router.get("/health") { _, _ in
            Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: "{\"status\":\"ok\"}")))
        }

        router.post("/v1/audio/transcriptions") { request, _ in
            try await handleOpenAITranscriptions(request: request, state: state)
        }

        router.post("/transcribe") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let sampleRate = params.int("sample_rate") ?? 16000
            let model = try await state.loadASR()
            let audio = try decodeWAVData(audioData, targetSampleRate: sampleRate)
            let text = model.transcribe(audio: audio, sampleRate: sampleRate)

            return jsonResponse([
                "text": text,
                "duration": round(Double(audio.count) / Double(sampleRate) * 100) / 100
            ] as [String: Any])
        }

        router.post("/speak") { request, _ in
            let body = try await request.body.collect(upTo: 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let text = params.text else {
                return errorResponse("Missing 'text' field", status: .badRequest)
            }

            let engine = params.string("engine") ?? "cosyvoice"
            let language = params.string("language") ?? "english"

            let samples: [Float]

            if engine == "qwen3" {
                let model = try await state.loadTTS()
                samples = model.synthesize(text: text, language: language)
            } else {
                let model = try await state.loadCosyVoice()
                samples = model.synthesize(text: text, language: language)
            }

            let wavData = try encodeWAV(samples: samples, sampleRate: 24000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/respond") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let voiceName = params.string("voice") ?? "NATM0"
            let maxSteps = params.int("max_steps") ?? 200

            guard let voice = PersonaPlexVoice(rawValue: voiceName) else {
                return errorResponse("Unknown voice: \(voiceName)", status: .badRequest)
            }

            let model = try await state.loadPersonaPlex()
            let audio = try decodeWAVData(audioData, targetSampleRate: 24000)
            let result = model.respond(
                userAudio: audio,
                voice: voice,
                maxSteps: maxSteps)

            var transcript: String?
            if let dec = state.spmDecoder, !result.textTokens.isEmpty {
                transcript = dec.decode(result.textTokens)
            }

            let wavData = try encodeWAV(samples: result.audio, sampleRate: 24000)
            let duration = Double(result.audio.count) / 24000.0

            if params.string("format") == "json" {
                var json: [String: Any] = [
                    "duration": round(duration * 100) / 100,
                    "text_tokens": result.textTokens.count
                ]
                if let t = transcript { json["transcript"] = t }
                json["audio_base64"] = wavData.base64EncodedString()
                return jsonResponse(json)
            }

            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/enhance") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let enhancer = try await state.loadEnhancer()
            let audio = try decodeWAVData(audioData, targetSampleRate: 48000)
            // Auto-chunk long inputs. The body cap of 50 MB allows roughly 4-5
            // min of 48 kHz mono PCM, which can easily exceed the model's 60 s
            // single-shot cap. enhanceChunked() does its own short-input
            // fast-path so we route everything through it (bit-identical to
            // enhance() when duration ≤ 45 s).
            let enhanced = try enhancer.enhanceChunked(audio: audio, sampleRate: 48000)

            let wavData = try encodeWAV(samples: enhanced, sampleRate: 48000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        return router
    }
}

// MARK: - Lazy Model State

final class ModelState: @unchecked Sendable {
    private var asr: Qwen3ASRModel?
    private var parakeet: ParakeetASRModel?
    private var nemotron: NemotronStreamingASRModel?
    private var omnilingual: OmnilingualASRModel?
    private var tts: Qwen3TTSModel?
    private var cosyvoice: CosyVoiceTTSModel?
    private var kokoro: KokoroTTSModel?
    private var voxcpm2: VoxCPM2TTSModel?
    private var magpie: MagpieTTS?
    private var personaplex: PersonaPlexModel?
    private var enhancer: SpeechEnhancer?
    var spmDecoder: SentencePieceDecoder?

    func loadASR() async throws -> Qwen3ASRModel {
        if let m = asr { return m }
        print("[server] Loading Qwen3-ASR...")
        let m = try await Qwen3ASRModel.fromPretrained(progressHandler: logProgress)
        asr = m
        return m
    }

    func loadParakeet() async throws -> ParakeetASRModel {
        if let m = parakeet { return m }
        print("[server] Loading Parakeet TDT v3...")
        let m = try await ParakeetASRModel.fromPretrained(progressHandler: logProgress)
        parakeet = m
        return m
    }

    func loadNemotron() async throws -> NemotronStreamingASRModel {
        if let m = nemotron { return m }
        print("[server] Loading Nemotron Streaming ASR...")
        let m = try await NemotronStreamingASRModel.fromPretrained(progressHandler: logProgress)
        nemotron = m
        return m
    }

    func loadOmnilingual() async throws -> OmnilingualASRModel {
        if let m = omnilingual { return m }
        print("[server] Loading Omnilingual ASR...")
        let m = try await OmnilingualASRModel.fromPretrained(progressHandler: logProgress)
        omnilingual = m
        return m
    }

    func loadTTS() async throws -> Qwen3TTSModel {
        if let m = tts { return m }
        print("[server] Loading Qwen3-TTS...")
        let m = try await Qwen3TTSModel.fromPretrained(progressHandler: logProgress)
        tts = m
        return m
    }

    func loadCosyVoice() async throws -> CosyVoiceTTSModel {
        if let m = cosyvoice { return m }
        print("[server] Loading CosyVoice...")
        let m = try await CosyVoiceTTSModel.fromPretrained(progressHandler: logProgress)
        cosyvoice = m
        return m
    }

    func loadKokoro() async throws -> KokoroTTSModel {
        if let m = kokoro { return m }
        print("[server] Loading Kokoro-82M...")
        let m = try await KokoroTTSModel.fromPretrained(progressHandler: logProgress)
        kokoro = m
        return m
    }

    func loadVoxCPM2() async throws -> VoxCPM2TTSModel {
        if let m = voxcpm2 { return m }
        print("[server] Loading VoxCPM2...")
        let m = try await VoxCPM2TTSModel.fromPretrained(progressHandler: logProgress)
        voxcpm2 = m
        return m
    }

    func loadMagpie() async throws -> MagpieTTS {
        if let m = magpie { return m }
        print("[server] Loading Magpie-TTS Multilingual...")
        let m = try await MagpieTTS.fromPretrained()
        magpie = m
        return m
    }

    func loadPersonaPlex() async throws -> PersonaPlexModel {
        if let m = personaplex { return m }
        print("[server] Loading PersonaPlex 7B...")
        let m = try await PersonaPlexModel.fromPretrained(progressHandler: logProgress)
        personaplex = m
        do {
            // Resolve the SPM tokenizer cache dir from the LOADED model's
            // modelId — not a hardcoded 4-bit repo. Same root cause as #300:
            // 8-bit users were silently falling back to no-decoder mode
            // because the cache dir lookup pointed at the wrong directory.
            let cacheDir = try HuggingFaceDownloader.getCacheDirectory(for: m.modelId)
            let spmPath = cacheDir.appendingPathComponent("tokenizer_spm_32k_3.model").path
            if FileManager.default.fileExists(atPath: spmPath) {
                spmDecoder = try SentencePieceDecoder(modelPath: spmPath)
            }
        } catch {}
        return m
    }

    func loadEnhancer() async throws -> SpeechEnhancer {
        if let m = enhancer { return m }
        print("[server] Loading DeepFilterNet3...")
        let m = try await SpeechEnhancer.fromPretrained(progressHandler: logProgress)
        enhancer = m
        return m
    }
}

private func logProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

// MARK: - OpenAI Realtime API Handler

/// Per-connection session state for the OpenAI Realtime protocol.
///
/// ASR and TTS engines are tracked independently so a client can switch the
/// transcription backend without disturbing synthesis (and vice versa). The
/// `model` field is the canonical OpenAI-style name last set by the client;
/// it does not by itself control routing — the resolved engine in
/// `asrEngine`/`ttsEngine` does.
private final class RealtimeSession {
    /// ASR backend used by `input_audio_buffer.commit`.
    var asrEngine: String = "parakeet"
    /// TTS backend used by `response.create`.
    var ttsEngine: String = "kokoro"
    /// The canonical model name last set via session.update (or the default).
    var model: String = "kokoro"
    var language: String = "english"
    /// Optional reference audio (PCM16 24 kHz) for voice-cloning engines (VoxCPM2).
    var voiceCloneReferenceAudio: [Float]?
    /// Optional reference transcript that pairs with `voiceCloneReferenceAudio`.
    var voiceCloneReferenceText: String?
    var inputAudioBuffer = Data()
    var inputSampleRate: Int = 24000
}

/// Map an OpenAI-style model name to the ASR engine string used by AudioServer.
///
/// Returns `nil` if the name does not name a known ASR engine (the caller
/// should then try the TTS resolver before treating the name as unknown).
///
/// Bare "qwen3" maps to the ASR engine; the TTS-side variant "qwen3-speech"
/// (or "qwen3-tts*") routes through `resolveModelToTTSEngine`. Variant suffixes
/// like "parakeet-tdt-0.6b" are accepted.
func resolveModelToASREngine(_ model: String) -> String? {
    let lower = model.lowercased()
    if lower == "qwen3" || lower == "qwen3-asr" || lower.hasPrefix("qwen3-asr-") { return "qwen3" }
    if lower == "parakeet" || lower.hasPrefix("parakeet-") { return "parakeet" }
    if lower == "nemotron" || lower.hasPrefix("nemotron-") { return "nemotron" }
    if lower == "omnilingual" || lower.hasPrefix("omnilingual-") { return "omnilingual" }
    return nil
}

/// Map an OpenAI-style model name to the TTS engine string used by AudioServer.
///
/// Returns `nil` if the name does not name a known TTS engine. Bare "qwen3"
/// is treated as ASR (see `resolveModelToASREngine`); use "qwen3-speech" or
/// "qwen3-tts*" to target the TTS side explicitly.
func resolveModelToTTSEngine(_ model: String) -> String? {
    let lower = model.lowercased()
    if lower == "kokoro" || lower.hasPrefix("kokoro-") { return "kokoro" }
    if lower == "cosyvoice" || lower.hasPrefix("cosyvoice-") { return "cosyvoice" }
    if lower == "voxcpm2" || lower.hasPrefix("voxcpm2-") { return "voxcpm2" }
    if lower == "magpie" || lower.hasPrefix("magpie-") { return "magpie" }
    if lower == "qwen3-speech" || lower.hasPrefix("qwen3-speech-")
        || lower == "qwen3-tts" || lower.hasPrefix("qwen3-tts-") { return "qwen3" }
    return nil
}

/// Map an OpenAI-style model name to either the ASR or TTS engine string.
///
/// Tries ASR resolution first, then TTS. Kept for clients that want a single
/// lookup without caring which side of the pipeline the engine drives.
///
/// Variant names like "qwen3-0.6b" or "qwen3-0.6b-coreml" are accepted and
/// resolve to the base engine. Unknown names return `nil` so callers can
/// apply forward-compatibility rules without changing the active engine.
func resolveModelToEngine(_ model: String) -> String? {
    if let asr = resolveModelToASREngine(model) { return asr }
    if let tts = resolveModelToTTSEngine(model) { return tts }
    return nil
}

/// Handle /v1/realtime: OpenAI Realtime API compatible protocol.
/// All messages are JSON with a "type" field. Audio is base64-encoded PCM16 24kHz.
func handleRealtimeWS(
    inbound: WebSocketInboundStream,
    outbound: WebSocketOutboundWriter,
    state: ModelState
) async throws {
    let session = RealtimeSession()
    let sessionId = UUID().uuidString

    // session.created reflects the same fields as session.updated so clients
    // can rely on a single shape for either event.
    try await outbound.write(.text(formatJSON(sessionEnvelope(id: sessionId, session: session, type: "session.created"))))

    for try await message in inbound.messages(maxSize: 50 * 1024 * 1024) {
        guard case .text(let string) = message else { continue }
        guard let jsonData = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let eventType = json["type"] as? String else {
            try await sendRealtimeError(outbound: outbound, message: "Invalid message format")
            continue
        }

        switch eventType {

        case "session.update":
            if let sessionConfig = json["session"] as? [String: Any] {
                // Top-level `model`: resolve to ASR and/or TTS engine and
                // update whichever slot(s) match. An ambiguous name like
                // "qwen3" updates only the ASR slot (the TTS variant is
                // "qwen3-speech"); unknown names are accepted and echoed
                // but leave the active engines unchanged.
                if let modelName = sessionConfig["model"] as? String, !modelName.isEmpty {
                    session.model = modelName
                    if let asr = resolveModelToASREngine(modelName) {
                        session.asrEngine = asr
                    }
                    if let tts = resolveModelToTTSEngine(modelName) {
                        session.ttsEngine = tts
                    }
                }
                // OpenAI-standard: `input_audio_transcription.model` selects
                // the ASR backend independently of the top-level model.
                if let iat = sessionConfig["input_audio_transcription"] as? [String: Any],
                   let asrModel = iat["model"] as? String,
                   let asr = resolveModelToASREngine(asrModel) {
                    session.asrEngine = asr
                }
                // Legacy `engine` field used to control TTS dispatch only;
                // preserve that semantics so existing clients keep working.
                if let engine = sessionConfig["engine"] as? String {
                    session.ttsEngine = engine
                }
                if let lang = sessionConfig["language"] as? String {
                    session.language = lang
                }
                if let fmt = sessionConfig["input_audio_format"] as? String, fmt == "pcm16" {
                    session.inputSampleRate = 24000
                }
                // Voice-cloning reference. PCM16 24 kHz, base64-encoded.
                // Setting this routes the next response.create to VoxCPM2
                // regardless of the active TTS engine.
                if let vc = sessionConfig["voice_cloning"] as? [String: Any] {
                    if let refB64 = vc["reference_audio"] as? String,
                       let refData = Data(base64Encoded: refB64) {
                        session.voiceCloneReferenceAudio = pcm16LEToFloat(refData)
                    }
                    if let refText = vc["reference_text"] as? String {
                        session.voiceCloneReferenceText = refText
                    }
                }
            }
            try await outbound.write(.text(formatJSON(sessionEnvelope(id: sessionId, session: session, type: "session.updated"))))

        case "input_audio_buffer.append":
            guard let audioB64 = json["audio"] as? String,
                  let audioData = Data(base64Encoded: audioB64) else {
                try await sendRealtimeError(outbound: outbound, message: "Missing or invalid 'audio' field")
                continue
            }
            session.inputAudioBuffer.append(audioData)

        case "input_audio_buffer.clear":
            session.inputAudioBuffer.removeAll()
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.cleared"
            ])))

        case "input_audio_buffer.commit":
            let audioData = session.inputAudioBuffer
            session.inputAudioBuffer.removeAll()

            guard !audioData.isEmpty else {
                try await sendRealtimeError(outbound: outbound, message: "Audio buffer is empty")
                continue
            }

            let itemId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "input_audio_buffer.committed",
                "item_id": itemId
            ])))

            // Transcribe via the active ASR engine. All engines internally
            // assume 16 kHz mono Float32 input; resample once and dispatch.
            let floats = pcm16LEToFloat(audioData)
            let audio16k = resample(floats, from: session.inputSampleRate, to: 16000)
            let text: String
            switch session.asrEngine {
            case "parakeet":
                let model = try await state.loadParakeet()
                text = (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: nil)) ?? ""
            case "nemotron":
                let model = try await state.loadNemotron()
                text = (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: session.language)) ?? ""
            case "omnilingual":
                let model = try await state.loadOmnilingual()
                text = (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: session.language)) ?? ""
            case "qwen3":
                let model = try await state.loadASR()
                text = model.transcribe(audio: audio16k, sampleRate: 16000)
            default:
                try await sendRealtimeError(outbound: outbound,
                    message: "ASR engine '\(session.asrEngine)' is not enabled in this build")
                continue
            }

            let responseId = UUID().uuidString
            try await outbound.write(.text(formatJSON([
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": itemId,
                "transcript": text
            ])))

            // Also emit as a response for clients expecting response.* events
            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.delta",
                "response_id": responseId,
                "delta": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.audio_transcript.done",
                "response_id": responseId,
                "transcript": text
            ])))
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": ["id": responseId, "status": "completed"]
            ] as [String: Any])))

        case "response.create":
            let input = json["response"] as? [String: Any]
            let instructions = input?["instructions"] as? String
            let modalities = input?["modalities"] as? [String] ?? ["audio", "text"]
            let responseId = UUID().uuidString

            // If there's text to speak (from instructions or input items)
            var textToSpeak: String?

            if let instructions = instructions, !instructions.isEmpty {
                textToSpeak = instructions
            }

            // Check for input items with text content
            if textToSpeak == nil, let inputItems = input?["input"] as? [[String: Any]] {
                for item in inputItems {
                    if let content = item["content"] as? [[String: Any]] {
                        for part in content {
                            if part["type"] as? String == "input_text",
                               let text = part["text"] as? String {
                                textToSpeak = text
                            }
                        }
                    }
                }
            }

            // Also check conversation.item.create pattern — text in input
            if textToSpeak == nil, let text = input?["text"] as? String {
                textToSpeak = text
            }

            guard let text = textToSpeak else {
                try await sendRealtimeError(outbound: outbound, message: "No text to synthesize")
                continue
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.created",
                "response": ["id": responseId, "status": "in_progress"]
            ] as [String: Any])))

            // Stream TTS audio as base64 PCM16 24 kHz chunks. The per-request
            // `engine` override mirrors the session field's legacy semantics
            // (TTS only). Voice-cloning requests are routed to VoxCPM2 even
            // if the active engine is something else.
            let perRequestEngine = input?["engine"] as? String
            let language = (input?["language"] as? String) ?? session.language
            let hasCloneReference = session.voiceCloneReferenceAudio != nil
            let engine: String
            if hasCloneReference {
                engine = "voxcpm2"
            } else if let perRequestEngine, !perRequestEngine.isEmpty {
                engine = perRequestEngine
            } else {
                engine = session.ttsEngine
            }
            var totalSamples = 0

            switch engine {
            case "kokoro":
                let model = try await state.loadKokoro()
                let langCode = mapToKokoroLanguageCode(language)
                let samples = try model.synthesize(text: text, language: langCode)
                totalSamples += try await streamSamplesAsDeltas(
                    samples, outbound: outbound, responseId: responseId)
            case "qwen3":
                let model = try await state.loadTTS()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            case "cosyvoice":
                let model = try await state.loadCosyVoice()
                let stream = model.synthesizeStream(text: text, language: language)
                for try await chunk in stream {
                    if !chunk.samples.isEmpty {
                        totalSamples += chunk.samples.count
                        let pcm = floatToPCM16LE(chunk.samples)
                        try await outbound.write(.text(formatJSON([
                            "type": "response.audio.delta",
                            "response_id": responseId,
                            "delta": pcm.base64EncodedString()
                        ])))
                    }
                }
            case "voxcpm2":
                let model = try await state.loadVoxCPM2()
                let samples: [Float]
                if hasCloneReference {
                    samples = try await model.generateVoxCPM2(
                        text: text,
                        language: language,
                        refText: session.voiceCloneReferenceText,
                        refAudio: session.voiceCloneReferenceAudio)
                } else {
                    samples = try await model.generate(text: text, language: language)
                }
                // VoxCPM2 runs at 48 kHz internally; downsample to the 24 kHz
                // the Realtime protocol expects.
                let samples24k = model.outputSampleRate == 24000
                    ? samples
                    : resample(samples, from: model.outputSampleRate, to: 24000)
                totalSamples += try await streamSamplesAsDeltas(
                    samples24k, outbound: outbound, responseId: responseId)
            case "magpie":
                let model = try await state.loadMagpie()
                let magpieLang = MagpieLanguage(code: mapToMagpieLanguageCode(language))
                    ?? .english
                let samples22k = try model.synthesize(text: text, language: magpieLang)
                // Magpie emits 22.05 kHz; resample to the 24 kHz protocol rate.
                let samples24k = resample(samples22k, from: MagpieTTS.sampleRate, to: 24000)
                totalSamples += try await streamSamplesAsDeltas(
                    samples24k, outbound: outbound, responseId: responseId)
            default:
                try await sendRealtimeError(outbound: outbound,
                    message: "TTS engine '\(engine)' is not enabled in this build")
                continue
            }

            if modalities.contains("text") {
                try await outbound.write(.text(formatJSON([
                    "type": "response.audio_transcript.done",
                    "response_id": responseId,
                    "transcript": text
                ])))
            }

            try await outbound.write(.text(formatJSON([
                "type": "response.audio.done",
                "response_id": responseId
            ])))

            let duration = Double(totalSamples) / 24000.0
            try await outbound.write(.text(formatJSON([
                "type": "response.done",
                "response": [
                    "id": responseId,
                    "status": "completed",
                    "usage": [
                        "total_tokens": 0,
                        "output_tokens": 0
                    ],
                    "output": [
                        ["type": "audio", "duration": round(duration * 100) / 100, "sample_rate": 24000]
                    ]
                ]
            ] as [String: Any])))

        case "conversation.item.create":
            // Accept text items for TTS via response.create flow
            if let item = json["item"] as? [String: Any],
               let content = item["content"] as? [[String: Any]] {
                for part in content {
                    if part["type"] as? String == "input_text" || part["type"] as? String == "text",
                       let _ = part["text"] as? String {
                        try await outbound.write(.text(formatJSON([
                            "type": "conversation.item.created",
                            "item": item
                        ] as [String: Any])))
                    }
                }
            }

        default:
            try await sendRealtimeError(outbound: outbound,
                message: "Unknown event type: \(eventType)")
        }
    }
}

private func sendRealtimeError(outbound: WebSocketOutboundWriter, message: String) async throws {
    try await outbound.write(.text(formatJSON([
        "type": "error",
        "error": ["type": "invalid_request_error", "message": message]
    ] as [String: Any])))
}

/// Build a `session.created` / `session.updated` envelope.
///
/// Both event types share the same payload shape so clients can read either
/// without branching on the type. Legacy clients can still read `engine`
/// (TTS engine) without knowing about the new `asr_engine`/`tts_engine`
/// split.
private func sessionEnvelope(id: String, session: RealtimeSession, type: String) -> [String: Any] {
    return [
        "type": type,
        "session": [
            "id": id,
            "model": session.model,
            "engine": session.ttsEngine,
            "asr_engine": session.asrEngine,
            "tts_engine": session.ttsEngine,
            "language": session.language,
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16"
        ] as [String: Any]
    ]
}

/// Stream a buffer of Float32 24 kHz samples to the client as base64 PCM16
/// `response.audio.delta` chunks. Returns the total number of samples sent.
/// Used by non-streaming TTS engines (Kokoro, VoxCPM2) — chunks are sized
/// for ~200 ms of audio at 24 kHz to balance time-to-first-byte against
/// per-chunk overhead.
private func streamSamplesAsDeltas(
    _ samples: [Float],
    outbound: WebSocketOutboundWriter,
    responseId: String,
    chunkSize: Int = 4800
) async throws -> Int {
    guard !samples.isEmpty else { return 0 }
    var i = 0
    while i < samples.count {
        let end = min(i + chunkSize, samples.count)
        let pcm = floatToPCM16LE(Array(samples[i..<end]))
        try await outbound.write(.text(formatJSON([
            "type": "response.audio.delta",
            "response_id": responseId,
            "delta": pcm.base64EncodedString()
        ])))
        i = end
    }
    return samples.count
}

/// Map a user-facing language name (or ISO code) to the 2-letter code
/// Kokoro's phonemizer expects. Unknown values fall through to English.
func mapToKokoroLanguageCode(_ language: String) -> String {
    let lower = language.lowercased()
    switch lower {
    case "en", "english": return "en"
    case "zh", "cmn", "chinese", "mandarin": return "zh"
    case "ja", "japanese": return "ja"
    case "fr", "french": return "fr"
    case "es", "spanish": return "es"
    case "pt", "portuguese": return "pt"
    case "it", "italian": return "it"
    case "hi", "hindi": return "hi"
    default: return "en"
    }
}

/// Map a user-facing language name (or ISO code) to the 2-letter code
/// Magpie's `MagpieLanguage(code:)` initialiser expects. Magpie ships
/// 9 languages — anything else falls through to English.
func mapToMagpieLanguageCode(_ language: String) -> String {
    let lower = language.lowercased()
    switch lower {
    case "en", "english": return "en"
    case "es", "spanish": return "es"
    case "de", "german": return "de"
    case "fr", "french": return "fr"
    case "it", "italian": return "it"
    case "vi", "vietnamese": return "vi"
    case "zh", "cmn", "chinese", "mandarin": return "zh"
    case "hi", "hindi": return "hi"
    case "ja", "japanese": return "ja"
    default: return "en"
    }
}

/// Resample audio via AVAudioConverter (delegates to AudioFileLoader).
func resample(_ samples: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
    AudioFileLoader.resample(samples, from: sourceSR, to: targetSR)
}

// MARK: - Request Parsing

struct RequestParams {
    var audioData: Data?
    var text: String?
    var fields: [String: String] = [:]

    func string(_ key: String) -> String? { fields[key] }
    func int(_ key: String) -> Int? { fields[key].flatMap(Int.init) }

    static func parse(_ body: ByteBuffer, contentType: String?) throws -> RequestParams {
        var params = RequestParams()

        if let ct = contentType, ct.contains("application/json") {
            let data = Data(buffer: body)
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                if let text = json["text"] as? String { params.text = text }
                if let b64 = json["audio_base64"] as? String {
                    params.audioData = Data(base64Encoded: b64)
                }
                for (k, v) in json {
                    if let s = v as? String { params.fields[k] = s }
                    else if let n = v as? Int { params.fields[k] = String(n) }
                    else if let n = v as? Double { params.fields[k] = String(n) }
                }
            }
            return params
        }

        // Raw audio body (WAV)
        let data = Data(buffer: body)
        if data.count > 44 {
            params.audioData = data
        }
        return params
    }
}

// MARK: - Audio Encoding/Decoding

func decodeWAVData(_ data: Data, targetSampleRate: Int) throws -> [Float] {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try data.write(to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try AudioFileLoader.load(url: tmpURL, targetSampleRate: targetSampleRate)
}

func encodeWAV(samples: [Float], sampleRate: Int) throws -> Data {
    let tmpURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString + ".wav")
    try WAVWriter.write(samples: samples, sampleRate: sampleRate, to: tmpURL)
    defer { try? FileManager.default.removeItem(at: tmpURL) }
    return try Data(contentsOf: tmpURL)
}

// MARK: - Response Helpers

func jsonResponse(_ dict: [String: Any]) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])) ?? Data()
    return Response(
        status: .ok,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

func errorResponse(_ message: String, status: HTTPResponse.Status) -> Response {
    let data = (try? JSONSerialization.data(
        withJSONObject: ["error": message], options: [])) ?? Data()
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

// MARK: - PCM Conversion

func pcm16LEToFloat(_ data: Data) -> [Float] {
    let sampleCount = data.count / 2
    var result = [Float](repeating: 0, count: sampleCount)
    data.withUnsafeBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<sampleCount {
            result[i] = Float(Int16(littleEndian: int16s[i])) / 32768.0
        }
    }
    return result
}

func floatToPCM16LE(_ samples: [Float]) -> Data {
    var data = Data(count: samples.count * 2)
    data.withUnsafeMutableBytes { raw in
        let int16s = raw.bindMemory(to: Int16.self)
        for i in 0..<samples.count {
            let clamped = max(-1.0, min(1.0, samples[i]))
            int16s[i] = Int16(clamped * 32767.0).littleEndian
        }
    }
    return data
}

func formatJSON(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys]) else {
        return "{}"
    }
    return String(data: data, encoding: .utf8) ?? "{}"
}
