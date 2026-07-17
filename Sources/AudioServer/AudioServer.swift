import Foundation
import Hummingbird
import HummingbirdCore
import HummingbirdWebSocket
import NIOCore
import Qwen3ASR
import Qwen3TTS
import Qwen3TTSCoreML
import CosyVoiceTTS
import ParakeetASR
import ParakeetStreamingASR
import NemotronStreamingASR
import OmnilingualASR
import KokoroTTS
import VoxCPM2TTS
import IndicMioTTS
import MagpieTTS
import MagpieTTSCoreML
import VibeVoiceTTS
import PersonaPlex
import HibikiTranslate
import SpeechEnhancement
import AudioCommon

// MARK: - Server

public struct AudioServer {
    let state: ModelState
    let realtimeState: any RealtimeModelLoading
    let host: String
    let port: Int

    public init(host: String = "127.0.0.1", port: Int = 8080, preload: Bool = false) {
        let state = ModelState()
        self.state = state
        self.realtimeState = state
        self.host = host
        self.port = port
    }

    init(
        host: String = "127.0.0.1",
        port: Int = 8080,
        state: ModelState = ModelState(),
        realtimeState: any RealtimeModelLoading
    ) {
        self.state = state
        self.realtimeState = realtimeState
        self.host = host
        self.port = port
    }

    public func run() async throws {
        let router = buildRouter()
        let realtimeState = self.realtimeState
        let wsConfig = realtimeWebSocketServerConfiguration()
        let wsServer: HTTPServerBuilder = .http1WebSocketUpgrade(configuration: wsConfig) { head, _, _ in
            let path = head.path ?? ""
            guard path == "/v1/realtime" else { return .dontUpgrade }
            return .upgrade([:]) { inbound, outbound, _ in
                try await handleRealtimeWS(inbound: inbound, outbound: outbound, state: realtimeState)
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

        // Shared row builder so the two model-list endpoints can't drift
        // apart in shape.
        @Sendable
        func modelRow(_ v: ModelVariant) -> [String: Any] {
            return [
                "id": v.name,
                "object": "model",
                "engine": v.engine,
                "kind": v.kind.rawValue,
                "model_id": v.modelId,
                "aliases": v.aliases
            ]
        }

        router.get("/v1/models") { _, _ in
            // The complete model catalog — every model the server can run,
            // across every kind (ASR, TTS, S2S, enhance, music, VAD,
            // diarize, speaker, separate, SR). Clients can introspect
            // what's selectable across both the Realtime WS and the HTTP
            // routes without trying names blindly.
            return jsonResponse([
                "object": "list",
                "data": MODEL_REGISTRY.map(modelRow)
            ] as [String: Any])
        }

        router.get("/v1/realtime/models") { _, _ in
            // The Realtime-protocol subset — only kinds that the WS
            // session.update model field actually dispatches to (ASR,
            // TTS, S2S). Convenience filter for clients that only care
            // about the WS surface; same registry backs both endpoints.
            let realtime: Set<ModelVariant.Kind> = [.asr, .tts, .s2s]
            let filtered = MODEL_REGISTRY.filter { realtime.contains($0.kind) }
            return jsonResponse([
                "object": "list",
                "data": filtered.map(modelRow)
            ] as [String: Any])
        }

        router.post("/v1/audio/transcriptions") { request, _ in
            try await handleOpenAITranscriptions(request: request, state: state)
        }

        router.post("/v1/audio/speech") { request, _ in
            try await handleOpenAISpeech(request: request, state: state)
        }

        router.post("/transcribe") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            // Validate the model name BEFORE reading audio so a typo in
            // the model field returns the actual problem, not a confusing
            // "missing audio" error. Unknown names error rather than
            // silently fall back, so clients learn fast.
            let variant: ModelVariant
            if let modelName = params.string("model"), !modelName.isEmpty {
                if let v = resolveModelToASRVariant(modelName) {
                    variant = v
                } else {
                    return errorResponse(
                        "Unknown ASR model: \(modelName)",
                        status: .badRequest)
                }
            } else {
                variant = defaultVariant(forEngine: "parakeet", kind: .asr)
            }

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let sampleRate = params.int("sample_rate") ?? 16000
            let audio = try decodeWAVData(audioData, targetSampleRate: sampleRate)
            let language = params.string("language")
            let text = try await dispatchTranscribe(
                audio: audio, sampleRate: sampleRate,
                variant: variant, language: language, state: state)

            return jsonResponse([
                "text": text,
                "model": variant.name,
                "duration": round(Double(audio.count) / Double(sampleRate) * 100) / 100
            ] as [String: Any])
        }

        router.post("/speak") { request, _ in
            let body = try await request.body.collect(upTo: 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            guard let text = params.text else {
                return errorResponse("Missing 'text' field", status: .badRequest)
            }

            // Variant precedence: model > legacy engine > default Kokoro.
            // Legacy engine field kept so old callers don't break.
            let variant: ModelVariant
            if let modelName = params.string("model"), !modelName.isEmpty {
                if let v = resolveModelToTTSVariant(modelName) {
                    variant = v
                } else {
                    return errorResponse(
                        "Unknown TTS model: \(modelName)",
                        status: .badRequest)
                }
            } else if let engineName = params.string("engine"), !engineName.isEmpty {
                variant = resolveModelToTTSVariant(engineName)
                    ?? defaultVariant(forEngine: "kokoro", kind: .tts)
            } else {
                variant = defaultVariant(forEngine: "kokoro", kind: .tts)
            }
            let language = params.string("language") ?? "english"
            let samples = try await dispatchSynthesize(
                text: text, variant: variant, language: language, state: state)

            let wavData = try encodeWAV(samples: samples, sampleRate: 24000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        }

        router.post("/respond") { request, _ in
            let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
            let params = try RequestParams.parse(body, contentType: request.headers[.contentType])

            // /respond is the PersonaPlex/Hibiki entry point on HTTP — same
            // S2S surface the Realtime WS uses, just request-shaped. The
            // `model` field picks which S2S engine (and which quantization
            // bundle). Default stays PersonaPlex 4-bit for back-compat.
            //
            // Validate the model name first so a typo doesn't trip the
            // audio guard or the voice guard with a misleading error.
            let variant: ModelVariant
            if let modelName = params.string("model"), !modelName.isEmpty {
                if let v = resolveModelToS2SVariant(modelName) {
                    variant = v
                } else {
                    return errorResponse(
                        "Unknown speech-to-speech model: \(modelName)",
                        status: .badRequest)
                }
            } else {
                variant = defaultVariant(forEngine: "personaplex", kind: .s2s)
            }

            guard let audioData = params.audioData, !audioData.isEmpty else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let maxSteps = params.int("max_steps") ?? 200
            let audio = try decodeWAVData(audioData, targetSampleRate: 24000)
            let responseAudio: [Float]
            var responseTranscript: String?
            switch variant.engine {
            case "personaplex":
                // PersonaPlex consumes a `voice` preset; Hibiki ignores it.
                // Keep the voice guard inside the personaplex branch so a
                // bad `voice` value never blocks a Hibiki translate call.
                let voiceName = params.string("voice") ?? "NATM0"
                guard let voice = PersonaPlexVoice(rawValue: voiceName) else {
                    return errorResponse("Unknown voice: \(voiceName)", status: .badRequest)
                }
                let model = try await state.loadPersonaPlex()
                let result = model.respond(
                    userAudio: audio,
                    voice: voice,
                    maxSteps: maxSteps)
                responseAudio = result.audio
                if let dec = state.spmDecoder, !result.textTokens.isEmpty {
                    responseTranscript = dec.decode(result.textTokens)
                }
            case "hibiki":
                let model = try await state.loadHibiki(modelId: variant.modelId)
                let lang = params.string("language") ?? "french"
                let sourceLang = HibikiSourceLanguage(
                    rawValue: mapToHibikiSourceLanguage(lang)) ?? .fr
                let result = model.translate(sourceAudio: audio, sourceLanguage: sourceLang)
                responseAudio = result.audio
            default:
                return errorResponse(
                    "S2S engine '\(variant.engine)' not enabled in this build",
                    status: .badRequest)
            }

            let wavData = try encodeWAV(samples: responseAudio, sampleRate: 24000)
            let duration = Double(responseAudio.count) / 24000.0

            if params.string("format") == "json" {
                var json: [String: Any] = [
                    "duration": round(duration * 100) / 100,
                    "model": variant.name
                ]
                if let t = responseTranscript { json["transcript"] = t }
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

            // Variant precedence: model > default. Same pattern as the
            // other registry-driven routes — typos return 400 with a
            // specific message rather than silently using the default.
            let variant: ModelVariant
            if let modelName = params.string("model"), !modelName.isEmpty {
                if let v = resolveModelVariant(modelName), v.kind == .enhance {
                    variant = v
                } else {
                    return errorResponse(
                        "Unknown enhance model: \(modelName)",
                        status: .badRequest)
                }
            } else {
                variant = defaultVariant(forEngine: "deepfilternet3", kind: .enhance)
            }

            guard let audioData = params.audioData else {
                return errorResponse("Missing audio data", status: .badRequest)
            }

            let enhancer = try await state.loadEnhancer(modelId: variant.modelId)
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

func realtimeWebSocketServerConfiguration() -> WebSocketServerConfiguration {
    // Hummingbird's autoPing watchdog expects a payload-matching control-frame
    // pong, which URLSessionWebSocketTask doesn't reliably surface while a
    // receive is pending behind synchronous work. The session-scoped
    // `realtime.keepalive` text frame in handleRealtimeWS replaces it: it
    // probes the transport every 15s for the whole connection lifetime,
    // including idle gaps between operations.
    WebSocketServerConfiguration(
        maxFrameSize: 1 << 24,
        autoPing: .disabled)
}

// MARK: - Lazy Model State

protocol RealtimeModelLoading: Sendable {
    func loadQwen3ASR(modelId: String) async throws -> Qwen3ASRModel
    func loadParakeet(modelId: String) async throws -> ParakeetASRModel
    func loadParakeetStreaming(modelId: String) async throws -> ParakeetStreamingASRModel
    func loadNemotron(modelId: String) async throws -> NemotronStreamingASRModel
    func loadOmnilingual(modelId: String) async throws -> OmnilingualASRModel
    func loadQwen3TTS(modelId: String) async throws -> Qwen3TTSModel
    func loadCosyVoice(modelId: String) async throws -> CosyVoiceTTSModel
    func loadKokoro(modelId: String) async throws -> KokoroTTSModel
    func loadVoxCPM2(modelId: String) async throws -> VoxCPM2TTSModel
    func loadIndicMio(modelId: String) async throws -> IndicMioTTSModel
    func loadMagpie() async throws -> MagpieTTS
    func loadMagpieCoreML() async throws -> MagpieTTSCoreML
    func loadQwen3TTSCoreML(modelId: String) async throws -> Qwen3TTSCoreMLModel
    func loadVibeVoice(modelId: String) async throws -> VibeVoiceTTSModel
    func loadVibeVoice15B(modelId: String) async throws -> VibeVoice15BTTSModel
    func loadHibiki(modelId: String) async throws -> HibikiTranslateModel
    func loadPersonaPlex() async throws -> PersonaPlexModel
}

struct RealtimeModelLoadingFailure: Error, CustomStringConvertible {
    let description: String

    init(_ description: String = "forced realtime model failure") {
        self.description = description
    }
}

final class FailingRealtimeModelLoading: RealtimeModelLoading, @unchecked Sendable {
    let error: Error
    let beforeFailure: (@Sendable () -> Void)?

    init(
        error: Error = RealtimeModelLoadingFailure(),
        beforeFailure: (@Sendable () -> Void)? = nil
    ) {
        self.error = error
        self.beforeFailure = beforeFailure
    }

    private func fail<T>() async throws -> T {
        beforeFailure?()
        throw error
    }

    func loadQwen3ASR(modelId: String) async throws -> Qwen3ASRModel {
        try await fail()
    }

    func loadParakeet(modelId: String) async throws -> ParakeetASRModel {
        try await fail()
    }

    func loadParakeetStreaming(modelId: String) async throws -> ParakeetStreamingASRModel {
        try await fail()
    }

    func loadNemotron(modelId: String) async throws -> NemotronStreamingASRModel {
        try await fail()
    }

    func loadOmnilingual(modelId: String) async throws -> OmnilingualASRModel {
        try await fail()
    }

    func loadQwen3TTS(modelId: String) async throws -> Qwen3TTSModel {
        try await fail()
    }

    func loadCosyVoice(modelId: String) async throws -> CosyVoiceTTSModel {
        try await fail()
    }

    func loadKokoro(modelId: String) async throws -> KokoroTTSModel {
        try await fail()
    }

    func loadVoxCPM2(modelId: String) async throws -> VoxCPM2TTSModel {
        try await fail()
    }

    func loadIndicMio(modelId: String) async throws -> IndicMioTTSModel {
        try await fail()
    }

    func loadMagpie() async throws -> MagpieTTS {
        try await fail()
    }

    func loadMagpieCoreML() async throws -> MagpieTTSCoreML {
        try await fail()
    }

    func loadQwen3TTSCoreML(modelId: String) async throws -> Qwen3TTSCoreMLModel {
        try await fail()
    }

    func loadVibeVoice(modelId: String) async throws -> VibeVoiceTTSModel {
        try await fail()
    }

    func loadVibeVoice15B(modelId: String) async throws -> VibeVoice15BTTSModel {
        try await fail()
    }

    func loadHibiki(modelId: String) async throws -> HibikiTranslateModel {
        try await fail()
    }

    func loadPersonaPlex() async throws -> PersonaPlexModel {
        try await fail()
    }
}

final class ModelState: RealtimeModelLoading, @unchecked Sendable {
    // Per-modelId caches: switching variants of the same engine (e.g.
    // qwen3-asr-0.6b → qwen3-asr-1.7b) keeps both loaded so flipping back
    // is instant. The typical session picks one variant and sticks, but
    // multi-tenant servers benefit from holding the small set warm.
    private var qwen3ASR: [String: Qwen3ASRModel] = [:]
    private var parakeet: [String: ParakeetASRModel] = [:]
    private var parakeetStreaming: [String: ParakeetStreamingASRModel] = [:]
    private var nemotron: [String: NemotronStreamingASRModel] = [:]
    private var omnilingual: [String: OmnilingualASRModel] = [:]
    private var qwen3TTS: [String: Qwen3TTSModel] = [:]
    private var qwen3TTSCoreML: [String: Qwen3TTSCoreMLModel] = [:]
    private var cosyvoice: [String: CosyVoiceTTSModel] = [:]
    private var kokoro: [String: KokoroTTSModel] = [:]
    private var voxcpm2: [String: VoxCPM2TTSModel] = [:]
    private var indicMio: [String: IndicMioTTSModel] = [:]
    private var magpie: MagpieTTS?
    private var magpieCoreML: MagpieTTSCoreML?
    private var vibevoice: [String: VibeVoiceTTSModel] = [:]
    private var vibevoice15B: [String: VibeVoice15BTTSModel] = [:]
    private var personaplex: PersonaPlexModel?
    private var hibikiByModelId: [String: HibikiTranslateModel] = [:]
    private var enhancer: SpeechEnhancer?
    var spmDecoder: SentencePieceDecoder?

    func loadQwen3ASR(modelId: String) async throws -> Qwen3ASRModel {
        if let m = qwen3ASR[modelId] { return m }
        print("[server] Loading Qwen3-ASR (\(modelId))...")
        let m = try await Qwen3ASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        qwen3ASR[modelId] = m
        return m
    }

    /// Back-compat shim for the HTTP routes that still want the default
    /// Qwen3-ASR build without naming a variant.
    func loadASR() async throws -> Qwen3ASRModel {
        try await loadQwen3ASR(modelId: "aufklarer/Qwen3-ASR-0.6B-MLX-4bit")
    }

    func loadParakeet(modelId: String) async throws -> ParakeetASRModel {
        if let m = parakeet[modelId] { return m }
        print("[server] Loading Parakeet (\(modelId))...")
        let m = try await ParakeetASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        parakeet[modelId] = m
        return m
    }

    func loadParakeetStreaming(modelId: String) async throws -> ParakeetStreamingASRModel {
        if let m = parakeetStreaming[modelId] { return m }
        print("[server] Loading Parakeet EOU Streaming (\(modelId))...")
        let m = try await ParakeetStreamingASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        parakeetStreaming[modelId] = m
        return m
    }

    func loadNemotron(modelId: String) async throws -> NemotronStreamingASRModel {
        if let m = nemotron[modelId] { return m }
        print("[server] Loading Nemotron Streaming ASR (\(modelId))...")
        let m = try await NemotronStreamingASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        nemotron[modelId] = m
        return m
    }

    func loadOmnilingual(modelId: String) async throws -> OmnilingualASRModel {
        if let m = omnilingual[modelId] { return m }
        print("[server] Loading Omnilingual ASR (\(modelId))...")
        let m = try await OmnilingualASRModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        omnilingual[modelId] = m
        return m
    }

    func loadQwen3TTS(modelId: String) async throws -> Qwen3TTSModel {
        if let m = qwen3TTS[modelId] { return m }
        print("[server] Loading Qwen3-TTS (\(modelId))...")
        let m = try await Qwen3TTSModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        qwen3TTS[modelId] = m
        return m
    }

    /// Back-compat shim — default Qwen3-TTS bundle.
    func loadTTS() async throws -> Qwen3TTSModel {
        try await loadQwen3TTS(modelId: Qwen3TTSModel.defaultModelId)
    }

    func loadCosyVoice(modelId: String) async throws -> CosyVoiceTTSModel {
        if let m = cosyvoice[modelId] { return m }
        print("[server] Loading CosyVoice (\(modelId))...")
        let m = try await CosyVoiceTTSModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        cosyvoice[modelId] = m
        return m
    }

    /// Back-compat shim — default CosyVoice bundle for HTTP routes.
    func loadCosyVoice() async throws -> CosyVoiceTTSModel {
        try await loadCosyVoice(modelId: CosyVoiceTTSModel.defaultModelId)
    }

    func loadKokoro(modelId: String) async throws -> KokoroTTSModel {
        if let m = kokoro[modelId] { return m }
        print("[server] Loading Kokoro (\(modelId))...")
        let m = try await KokoroTTSModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        kokoro[modelId] = m
        return m
    }

    func loadVoxCPM2(modelId: String) async throws -> VoxCPM2TTSModel {
        if let m = voxcpm2[modelId] { return m }
        print("[server] Loading VoxCPM2 (\(modelId))...")
        let m = try await VoxCPM2TTSModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        voxcpm2[modelId] = m
        return m
    }

    func loadIndicMio(modelId: String) async throws -> IndicMioTTSModel {
        if let m = indicMio[modelId] { return m }
        print("[server] Loading Indic-Mio (\(modelId))...")
        let m = try await IndicMioTTSModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        indicMio[modelId] = m
        return m
    }

    /// Magpie ships as a single fixed bundle today — `fromPretrained` takes
    /// a `MagpieTTSVariant` enum, not an HF slug. The registry's modelId is
    /// informational; the loader uses the variant default.
    func loadMagpie() async throws -> MagpieTTS {
        if let m = magpie { return m }
        print("[server] Loading Magpie-TTS Multilingual...")
        let m = try await MagpieTTS.fromPretrained()
        magpie = m
        return m
    }

    /// MagpieTTSCoreML uses a fixed CoreML bundle; like MLX-Magpie, no
    /// per-modelId caching — there's one variant.
    func loadMagpieCoreML() async throws -> MagpieTTSCoreML {
        if let m = magpieCoreML { return m }
        print("[server] Loading Magpie-TTS CoreML...")
        let m = try await MagpieTTSCoreML.fromPretrained()
        magpieCoreML = m
        return m
    }

    func loadQwen3TTSCoreML(modelId: String) async throws -> Qwen3TTSCoreMLModel {
        if let m = qwen3TTSCoreML[modelId] { return m }
        print("[server] Loading Qwen3-TTS CoreML (\(modelId))...")
        let m = try await Qwen3TTSCoreMLModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        qwen3TTSCoreML[modelId] = m
        return m
    }

    func loadVibeVoice(modelId: String) async throws -> VibeVoiceTTSModel {
        if let m = vibevoice[modelId] { return m }
        print("[server] Loading VibeVoice Realtime (\(modelId))...")
        var cfg = VibeVoiceTTSModel.Configuration()
        cfg.modelId = modelId
        let m = try await VibeVoiceTTSModel.fromPretrained(configuration: cfg, progressHandler: logProgress)
        vibevoice[modelId] = m
        return m
    }

    func loadVibeVoice15B(modelId: String) async throws -> VibeVoice15BTTSModel {
        if let m = vibevoice15B[modelId] { return m }
        print("[server] Loading VibeVoice 1.5B (\(modelId))...")
        var cfg = VibeVoice15BTTSModel.Configuration()
        cfg.modelId = modelId
        let m = try await VibeVoice15BTTSModel.fromPretrained(configuration: cfg, progressHandler: logProgress)
        vibevoice15B[modelId] = m
        return m
    }

    func loadHibiki(modelId: String) async throws -> HibikiTranslateModel {
        if let m = hibikiByModelId[modelId] { return m }
        print("[server] Loading Hibiki (\(modelId))...")
        let m = try await HibikiTranslateModel.fromPretrained(modelId: modelId, progressHandler: logProgress)
        hibikiByModelId[modelId] = m
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

    private var enhancerByModelId: [String: SpeechEnhancer] = [:]

    func loadEnhancer(modelId: String) async throws -> SpeechEnhancer {
        if let m = enhancerByModelId[modelId] { return m }
        print("[server] Loading DeepFilterNet3 (\(modelId))...")
        let m = try await SpeechEnhancer.fromPretrained(modelId: modelId, progressHandler: logProgress)
        enhancerByModelId[modelId] = m
        // Mirror to the legacy slot too so back-compat callers see the
        // last-loaded enhancer.
        enhancer = m
        return m
    }

    /// Back-compat shim — default DeepFilterNet3 bundle.
    func loadEnhancer() async throws -> SpeechEnhancer {
        try await loadEnhancer(modelId: SpeechEnhancer.defaultModelId)
    }
}

private func logProgress(_ progress: Double, _ status: String) {
    print("  [\(Int(progress * 100))%] \(status)")
}

// MARK: - OpenAI Realtime API Handler

/// Per-connection session state for the OpenAI Realtime protocol.
///
/// Three engine slots tracked independently:
///   - `asrVariant` for the input transcription stage
///   - `ttsVariant` for the output synthesis stage
///   - `s2sVariant` for true speech-to-speech models (PersonaPlex, Hibiki)
///
/// When `s2sVariant` is non-nil it takes precedence — `input_audio_buffer.commit`
/// captures the audio for the S2S model and `response.create` runs the model
/// over that audio in one shot, bypassing the ASR→TTS compose path.
///
/// The `model` field is the canonical name last set by the client; it does
/// not control routing on its own — the resolved variants do.
private final class RealtimeSession {
    /// ASR variant used by `input_audio_buffer.commit` when no S2S is active.
    var asrVariant: ModelVariant
    /// TTS variant used by `response.create` when no S2S is active.
    var ttsVariant: ModelVariant
    /// Optional speech-to-speech variant. When non-nil, the S2S path is
    /// active and the ASR/TTS slots are bypassed for both events.
    var s2sVariant: ModelVariant?
    /// Canonical model name last set via session.update (or the default).
    /// Stored verbatim — may be a registered name, an alias, or a forward-
    /// compat name we accept-and-echo without dispatching.
    var model: String
    /// Legacy `engine` field — last value the client sent. Echoed back as
    /// `session.engine` for back-compat with clients that don't read the
    /// new `asr_engine` / `tts_engine` / `s2s_engine` fields.
    var legacyEngine: String?
    var language: String = "english"
    /// PersonaPlex voice preset (e.g. "NATM0"). Only consulted when the
    /// active S2S engine is PersonaPlex.
    var voice: String?
    /// Optional reference audio (PCM16 24 kHz) for voice-cloning engines
    /// (VoxCPM2). Setting this forces the next response.create to VoxCPM2.
    var voiceCloneReferenceAudio: [Float]?
    /// Optional reference transcript that pairs with the cloning audio.
    var voiceCloneReferenceText: String?
    var inputAudioBuffer = Data()
    var inputSampleRate: Int = 24000
    /// Audio captured by the last `input_audio_buffer.commit`, kept at the
    /// protocol sample rate (24 kHz mono Float32). The S2S path reads from
    /// here on the following `response.create`. Cleared after use.
    var lastCommittedAudio: [Float]?

    init() {
        let defaultASR = defaultVariant(forEngine: "parakeet", kind: .asr)
        let defaultTTS = defaultVariant(forEngine: "kokoro", kind: .tts)
        self.asrVariant = defaultASR
        self.ttsVariant = defaultTTS
        self.s2sVariant = nil
        // Canonical model defaults to the TTS variant name — that's what
        // the user hears, and matches the OpenAI convention of `model`
        // naming the user-facing output side.
        self.model = defaultTTS.name
    }

    /// Echo value for the legacy `engine` field. If the client set it
    /// explicitly we round-trip the raw string; otherwise we derive it from
    /// the active TTS variant's engine slot.
    var engineEcho: String {
        return legacyEngine ?? ttsVariant.engine
    }
}

/// Resolve a model name to its ASR variant, if any.
///
/// Convenience over `resolveAllVariants` for callers that only care about
/// one slot (e.g. the OpenAI-standard `input_audio_transcription.model`
/// field, which explicitly targets ASR).
func resolveModelToASRVariant(_ model: String) -> ModelVariant? {
    return resolveAllVariants(model).first(where: { $0.kind == .asr })
}

/// Resolve a model name to its TTS variant, if any.
func resolveModelToTTSVariant(_ model: String) -> ModelVariant? {
    return resolveAllVariants(model).first(where: { $0.kind == .tts })
}

/// Resolve a model name to its S2S variant, if any.
func resolveModelToS2SVariant(_ model: String) -> ModelVariant? {
    return resolveAllVariants(model).first(where: { $0.kind == .s2s })
}

/// Look up a name across every kind. A single name can match more than one
/// kind (e.g. "qwen3" hits both an ASR variant and a TTS variant), in which
/// case all matches are returned — used by the top-level `model` field on
/// session.update so a paired family name updates both slots in one step.
func resolveAllVariants(_ name: String) -> [ModelVariant] {
    let lower = name.lowercased()
    guard !lower.isEmpty else { return [] }
    // Exact canonical-name match short-circuits everything else.
    if let exact = MODEL_REGISTRY.first(where: { $0.name == lower }) {
        return [exact]
    }
    // Alias match — collect one per kind so paired names update every
    // slot they fit.
    var hits: [ModelVariant] = []
    for kind in [ModelVariant.Kind.asr, .tts, .s2s] {
        if let v = MODEL_REGISTRY.first(where: { $0.kind == kind && $0.aliases.contains(lower) }) {
            hits.append(v)
        }
    }
    return hits
}

// MARK: - Legacy resolver shims
//
// These keep the original `resolveModelToASREngine` / `resolveModelToTTSEngine`
// / `resolveModelToEngine` surface for tests and any external callers that
// pinned to it. The body just walks the registry.

func resolveModelToASREngine(_ model: String) -> String? {
    return resolveModelToASRVariant(model)?.engine
}

func resolveModelToTTSEngine(_ model: String) -> String? {
    return resolveModelToTTSVariant(model)?.engine
}

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
    state: any RealtimeModelLoading
) async throws {
    let session = RealtimeSession()
    let sessionId = UUID().uuidString

    // session.created reflects the same fields as session.updated so clients
    // can rely on a single shape for either event.
    try await outbound.write(.text(formatJSON(sessionEnvelope(id: sessionId, session: session, type: "session.created"))))

    // Session-scoped keepalive: emits `realtime.keepalive` every 15s for the
    // entire connection lifetime, not just while a model operation is in
    // flight. Covers idle gaps between operations that the Hummingbird
    // autoPing watchdog would otherwise have caught — see
    // realtimeWebSocketServerConfiguration for why autoPing is off.
    let keepalive = Task.detached { [outbound] in
        while !Task.isCancelled {
            do {
                try await Task.sleep(nanoseconds: realtimeKeepaliveIntervalNanoseconds)
                try Task.checkCancellation()
                try await outbound.write(.text(formatJSON([
                    "type": realtimeKeepaliveEvent
                ])))
            } catch {
                return
            }
        }
    }
    defer { keepalive.cancel() }

    for try await message in inbound.messages(maxSize: 50 * 1024 * 1024) {
        guard case .text(let string) = message else { continue }
        guard let jsonData = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
              let eventType = json["type"] as? String else {
            try await sendRealtimeError(outbound: outbound, message: "Invalid message format")
            continue
        }

        do {
            switch eventType {

            case "session.update":
                if let sessionConfig = json["session"] as? [String: Any] {
                    // Top-level `model`: walk the registry, update whichever
                    // slots match. Bare "qwen3" updates both ASR and TTS
                    // (they share the alias); "voxcpm2" updates only TTS;
                    // "hibiki" updates only S2S. Unknown names are accepted
                    // and echoed without touching any slot.
                    if let modelName = sessionConfig["model"] as? String, !modelName.isEmpty {
                        session.model = modelName
                        let variants = resolveAllVariants(modelName)
                        let pickedS2S = variants.first(where: { $0.kind == .s2s })
                        for v in variants {
                            switch v.kind {
                            case .asr: session.asrVariant = v
                            case .tts: session.ttsVariant = v
                            case .s2s: session.s2sVariant = v
                            case .enhance, .music, .vad, .diarize,
                                 .speaker, .separate, .sr:
                                // Cataloged in the registry for discovery via
                                // /v1/models, but the Realtime session protocol
                                // has no slot for these — they're routed via
                                // dedicated HTTP endpoints (/enhance, /compose,
                                // /diarize, …). No-op on session.update.
                                break
                            }
                        }
                        // S2S is exclusive — picking a recognized ASR/TTS-only
                        // model turns S2S off so the user gets the compose
                        // path back. Also drop any pending S2S input audio so
                        // the next response.create starts clean.
                        if pickedS2S == nil && !variants.isEmpty {
                            session.s2sVariant = nil
                            session.lastCommittedAudio = nil
                        }
                    }
                    // OpenAI-standard: `input_audio_transcription.model` selects
                    // the ASR backend independently of the top-level model.
                    if let iat = sessionConfig["input_audio_transcription"] as? [String: Any],
                       let asrModel = iat["model"] as? String,
                       let asr = resolveModelToASRVariant(asrModel) {
                        session.asrVariant = asr
                    }
                    // Legacy `engine` field used to control TTS dispatch only.
                    // Preserve that — store the raw string for the echo and
                    // update the TTS variant if the name resolves.
                    if let engine = sessionConfig["engine"] as? String {
                        session.legacyEngine = engine
                        if let tts = resolveModelToTTSVariant(engine) {
                            session.ttsVariant = tts
                        }
                    }
                    if let lang = sessionConfig["language"] as? String {
                        session.language = lang
                    }
                    if let v = sessionConfig["voice"] as? String, !v.isEmpty {
                        session.voice = v
                    }
                    if let fmt = sessionConfig["input_audio_format"] as? String, fmt == "pcm16" {
                        session.inputSampleRate = 24000
                    }
                    // Voice-cloning reference. PCM16 24 kHz, base64-encoded.
                    // Setting this routes the next response.create to VoxCPM2
                    // regardless of the active TTS engine. Guard against empty
                    // / malformed payloads so we never hand VoxCPM2 a zero-
                    // length reference that the model can't condition on.
                    if let vc = sessionConfig["voice_cloning"] as? [String: Any] {
                        if let refB64 = vc["reference_audio"] as? String,
                           let refData = Data(base64Encoded: refB64), !refData.isEmpty {
                            let samples = pcm16LEToFloat(refData)
                            if !samples.isEmpty {
                                session.voiceCloneReferenceAudio = samples
                            }
                        }
                        if let refText = vc["reference_text"] as? String, !refText.isEmpty {
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

                let floats24k = pcm16LEToFloat(audioData)

                // S2S precedence: when an S2S variant is active, commit stores
                // the audio at the protocol rate (24 kHz mono) for the next
                // response.create to consume. No transcription is emitted —
                // the S2S model produces text as part of generation.
                if session.s2sVariant != nil {
                    session.lastCommittedAudio = floats24k
                    continue
                }

                // Compose path: transcribe via the active ASR variant. Every
                // ASR engine expects 16 kHz mono Float32; resample once.
                let audio16k = resample(floats24k, from: session.inputSampleRate, to: 16000)
                let asr = session.asrVariant
                let text: String
                switch asr.engine {
                case "parakeet":
                    text = try await runOffloaded {
                        let model = try await state.loadParakeet(modelId: asr.modelId)
                        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: nil)) ?? ""
                    }
                case "parakeet-streaming":
                    text = try await runOffloaded {
                        let model = try await state.loadParakeetStreaming(modelId: asr.modelId)
                        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: session.language)) ?? ""
                    }
                case "nemotron":
                    text = try await runOffloaded {
                        let model = try await state.loadNemotron(modelId: asr.modelId)
                        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: session.language)) ?? ""
                    }
                case "omnilingual":
                    text = try await runOffloaded {
                        let model = try await state.loadOmnilingual(modelId: asr.modelId)
                        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: session.language)) ?? ""
                    }
                case "qwen3-asr":
                    text = try await runOffloaded {
                        let model = try await state.loadQwen3ASR(modelId: asr.modelId)
                        return model.transcribe(audio: audio16k, sampleRate: 16000)
                    }
                default:
                    try await sendRealtimeError(outbound: outbound,
                        message: "ASR engine '\(asr.engine)' is not enabled in this build")
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

                // S2S precedence: when an S2S variant is active AND a prior
                // `input_audio_buffer.commit` left audio on the session, run
                // the S2S model on that audio. Text input is ignored — S2S
                // is audio-driven and produces its own transcript as part of
                // generation.
                if let s2s = session.s2sVariant, let userAudio = session.lastCommittedAudio {
                    session.lastCommittedAudio = nil
                    try await outbound.write(.text(formatJSON([
                        "type": "response.created",
                        "response": ["id": responseId, "status": "in_progress"]
                    ] as [String: Any])))
                    var s2sTotalSamples = 0
                    switch s2s.engine {
                    case "personaplex":
                        let result = try await runOffloaded {
                            let model = try await state.loadPersonaPlex()
                            let voice = PersonaPlexVoice(rawValue: session.voice ?? "")
                                ?? .NATM0
                            return model.respond(
                                userAudio: userAudio,
                                voice: voice,
                                systemPromptTokens: nil,
                                maxSteps: 200)
                        }
                        s2sTotalSamples += try await streamSamplesAsDeltas(
                            result.audio, outbound: outbound, responseId: responseId)
                    case "hibiki":
                        let result = try await runOffloaded {
                            let model = try await state.loadHibiki(modelId: s2s.modelId)
                            let sourceLang = HibikiSourceLanguage(
                                rawValue: mapToHibikiSourceLanguage(session.language)) ?? .fr
                            return model.translate(sourceAudio: userAudio, sourceLanguage: sourceLang)
                        }
                        s2sTotalSamples += try await streamSamplesAsDeltas(
                            result.audio, outbound: outbound, responseId: responseId)
                    default:
                        try await sendRealtimeError(outbound: outbound,
                            message: "S2S engine '\(s2s.engine)' is not enabled in this build")
                        continue
                    }
                    try await outbound.write(.text(formatJSON([
                        "type": "response.audio.done",
                        "response_id": responseId
                    ])))
                    let duration = Double(s2sTotalSamples) / 24000.0
                    try await outbound.write(.text(formatJSON([
                        "type": "response.done",
                        "response": [
                            "id": responseId,
                            "status": "completed",
                            "usage": ["total_tokens": 0, "output_tokens": 0],
                            "output": [
                                ["type": "audio", "duration": round(duration * 100) / 100, "sample_rate": 24000]
                            ]
                        ]
                    ] as [String: Any])))
                    continue
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
                let ttsVariant: ModelVariant
                if hasCloneReference {
                    // Cloning forces VoxCPM2. If the session already has a
                    // VoxCPM2 variant selected (bf16, int8, etc.), keep that
                    // exact bundle — the user picked it for a reason.
                    // Otherwise fall back to the default VoxCPM2 variant.
                    if session.ttsVariant.engine == "voxcpm2" {
                        ttsVariant = session.ttsVariant
                    } else {
                        ttsVariant = resolveModelToTTSVariant("voxcpm2") ?? session.ttsVariant
                    }
                } else if let perRequestEngine, !perRequestEngine.isEmpty,
                          let v = resolveModelToTTSVariant(perRequestEngine) {
                    ttsVariant = v
                } else {
                    ttsVariant = session.ttsVariant
                }
                var totalSamples = 0

                switch ttsVariant.engine {
                case "kokoro":
                    let samples = try await runOffloaded {
                        let model = try await state.loadKokoro(modelId: ttsVariant.modelId)
                        let langCode = mapToKokoroLanguageCode(language)
                        return try model.synthesize(text: text, language: langCode)
                    }
                    totalSamples += try await streamSamplesAsDeltas(
                        samples, outbound: outbound, responseId: responseId)
                case "qwen3-tts":
                    let model = try await runOffloaded {
                        try await state.loadQwen3TTS(modelId: ttsVariant.modelId)
                    }
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
                    let model = try await runOffloaded {
                        try await state.loadCosyVoice(modelId: ttsVariant.modelId)
                    }
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
                    let (model, samples) = try await runOffloaded {
                        let model = try await state.loadVoxCPM2(modelId: ttsVariant.modelId)
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
                        return (model, samples)
                    }
                    // VoxCPM2 runs at 48 kHz internally; downsample to the 24 kHz
                    // the Realtime protocol expects.
                    let samples24k = model.outputSampleRate == 24000
                        ? samples
                        : resample(samples, from: model.outputSampleRate, to: 24000)
                    totalSamples += try await streamSamplesAsDeltas(
                        samples24k, outbound: outbound, responseId: responseId)
                case "indic-mio":
                    let samples = try await runOffloaded {
                        let model = try await state.loadIndicMio(modelId: ttsVariant.modelId)
                        if hasCloneReference {
                            return try await model.generate(
                                text: text,
                                language: language,
                                referenceAudio: session.voiceCloneReferenceAudio ?? [],
                                referenceSampleRate: 24_000)
                        }
                        return try await model.generate(text: text, language: language)
                    }
                    totalSamples += try await streamSamplesAsDeltas(
                        samples, outbound: outbound, responseId: responseId)
                case "magpie":
                    let samples22k = try await runOffloaded {
                        let model = try await state.loadMagpie()
                        let magpieLang = MagpieLanguage(code: mapToMagpieLanguageCode(language))
                            ?? .english
                        return try model.synthesize(text: text, language: magpieLang)
                    }
                    // Magpie emits 22.05 kHz; resample to the 24 kHz protocol rate.
                    let samples24k = resample(samples22k, from: MagpieTTS.sampleRate, to: 24000)
                    totalSamples += try await streamSamplesAsDeltas(
                        samples24k, outbound: outbound, responseId: responseId)
                case "magpie-coreml":
                    let samples22k = try await runOffloaded {
                        let model = try await state.loadMagpieCoreML()
                        let magpieLang = MagpieCoreMLLanguage(rawValue: mapToMagpieLanguageCode(language))
                            ?? .english
                        return try model.synthesize(text: text, language: magpieLang)
                    }
                    let samples24k = resample(samples22k, from: MagpieTTSCoreML.sampleRate, to: 24000)
                    totalSamples += try await streamSamplesAsDeltas(
                        samples24k, outbound: outbound, responseId: responseId)
                case "qwen3-tts-coreml":
                    let samples = try await runOffloaded {
                        let model = try await state.loadQwen3TTSCoreML(modelId: ttsVariant.modelId)
                        return try model.synthesize(text: text, language: language)
                    }
                    totalSamples += try await streamSamplesAsDeltas(
                        samples, outbound: outbound, responseId: responseId)
                case "vibevoice":
                    let samples = try await runOffloaded {
                        let model = try await state.loadVibeVoice(modelId: ttsVariant.modelId)
                        return try await model.generate(text: text, language: language)
                    }
                    totalSamples += try await streamSamplesAsDeltas(
                        samples, outbound: outbound, responseId: responseId)
                case "vibevoice-1.5b":
                    let samples = try await runOffloaded {
                        let model = try await state.loadVibeVoice15B(modelId: ttsVariant.modelId)
                        return try await model.generate(text: text, language: language)
                    }
                    totalSamples += try await streamSamplesAsDeltas(
                        samples, outbound: outbound, responseId: responseId)
                default:
                    try await sendRealtimeError(outbound: outbound,
                        message: "TTS engine '\(ttsVariant.engine)' is not enabled in this build")
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
        } catch let error as CancellationError {
            throw error
        } catch {
            try await sendRealtimeProcessingError(
                outbound: outbound,
                eventType: eventType,
                error: error)
        }
    }
}

private func sendRealtimeProcessingError(
    outbound: WebSocketOutboundWriter,
    eventType: String,
    error: Error
) async throws {
    let detail = realtimeErrorDescription(error)
    try await sendRealtimeError(
        outbound: outbound,
        type: "server_error",
        message: "Realtime event '\(eventType)' failed: \(detail)",
        eventType: eventType)
}

private func realtimeErrorDescription(_ error: Error) -> String {
    if let localized = error as? LocalizedError,
       let description = localized.errorDescription,
       !description.isEmpty {
        return description
    }
    return String(describing: error)
}

private func sendRealtimeError(
    outbound: WebSocketOutboundWriter,
    type: String = "invalid_request_error",
    message: String,
    eventType: String? = nil
) async throws {
    var errorBody: [String: Any] = ["type": type, "message": message]
    if let eventType {
        errorBody["event_type"] = eventType
    }
    try await outbound.write(.text(formatJSON([
        "type": "error",
        "error": errorBody
    ] as [String: Any])))
}

private let realtimeKeepaliveIntervalNanoseconds: UInt64 = 15_000_000_000
private let realtimeKeepaliveEvent = "realtime.keepalive"

/// Run `operation` on a detached task so the calling websocket handler's
/// executor stays free to deliver the session-scoped `realtime.keepalive`
/// emissions even while inference monopolizes a cooperative thread.
///
/// Cancellation is forwarded: if the awaiting task is cancelled, the
/// detached operation is cancelled too so it doesn't run unobserved.
private func runOffloaded<T: Sendable>(
    operation: @escaping @Sendable () async throws -> T
) async throws -> T {
    let task = Task.detached(operation: operation)
    return try await withTaskCancellationHandler {
        try await task.value
    } onCancel: {
        task.cancel()
    }
}

/// Build a `session.created` / `session.updated` envelope.
///
/// Both event types share the same payload shape so clients can read either
/// without branching on the type. Legacy clients can still read `engine`
/// (TTS engine) without knowing about the new `asr_engine`/`tts_engine`
/// split.
private func sessionEnvelope(id: String, session: RealtimeSession, type: String) -> [String: Any] {
    var payload: [String: Any] = [
        "id": id,
        "model": session.model,
        "engine": session.engineEcho,
        "asr_engine": session.asrVariant.engine,
        "tts_engine": session.ttsVariant.engine,
        "asr_model": session.asrVariant.name,
        "tts_model": session.ttsVariant.name,
        "language": session.language,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16"
    ]
    if let s2s = session.s2sVariant {
        payload["s2s_engine"] = s2s.engine
        payload["s2s_model"] = s2s.name
    } else {
        payload["s2s_engine"] = NSNull()
        payload["s2s_model"] = NSNull()
    }
    if let v = session.voice {
        payload["voice"] = v
    }
    return [
        "type": type,
        "session": payload
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

// MARK: - Shared dispatch (registry-driven)

/// Transcribe audio via the engine + modelId named by `variant`. Used by
/// both the WS `input_audio_buffer.commit` handler and the HTTP routes so
/// the routing is single-sourced — a new ASR variant only has to be added
/// to one place.
///
/// Returns an empty string on engine-level errors (matching the WS path's
/// `try?` shape) so the calling route can return an empty transcript
/// rather than a 500.
func dispatchTranscribe(
    audio: [Float],
    sampleRate: Int,
    variant: ModelVariant,
    language: String?,
    state: ModelState
) async throws -> String {
    let audio16k: [Float] = sampleRate == 16000
        ? audio
        : resample(audio, from: sampleRate, to: 16000)
    switch variant.engine {
    case "parakeet":
        let model = try await state.loadParakeet(modelId: variant.modelId)
        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: language)) ?? ""
    case "parakeet-streaming":
        let model = try await state.loadParakeetStreaming(modelId: variant.modelId)
        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: language)) ?? ""
    case "nemotron":
        let model = try await state.loadNemotron(modelId: variant.modelId)
        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: language)) ?? ""
    case "omnilingual":
        let model = try await state.loadOmnilingual(modelId: variant.modelId)
        return (try? model.transcribeAudio(audio16k, sampleRate: 16000, language: language)) ?? ""
    case "qwen3-asr":
        let model = try await state.loadQwen3ASR(modelId: variant.modelId)
        return model.transcribe(audio: audio16k, sampleRate: 16000)
    default:
        throw RealtimeDispatchError.engineNotEnabled(kind: "ASR", engine: variant.engine)
    }
}

/// Synthesize text into 24 kHz Float32 PCM via the engine named by
/// `variant`. Non-streaming — accumulates the full waveform. The WS path
/// keeps its own per-chunk streaming dispatch for engines that support it
/// (qwen3, cosyvoice); the HTTP routes use this helper since they return
/// a complete WAV body. Engines that emit at a different sample rate
/// (Magpie 22.05 kHz, VoxCPM2 48 kHz) are resampled here.
func dispatchSynthesize(
    text: String,
    variant: ModelVariant,
    language: String,
    state: ModelState,
    voice: String? = nil,
    speed: Float = 1.0,
    cloneReferenceAudio: [Float]? = nil,
    cloneReferenceText: String? = nil
) async throws -> [Float] {
    switch variant.engine {
    case "kokoro":
        let model = try await state.loadKokoro(modelId: variant.modelId)
        return try model.synthesize(
            text: text,
            voice: voice ?? KokoroTTSModel.defaultVoice,
            language: mapToKokoroLanguageCode(language),
            speed: speed)
    case "qwen3-tts":
        let model = try await state.loadQwen3TTS(modelId: variant.modelId)
        return model.synthesize(text: text, language: language, speaker: voice)
    case "qwen3-tts-coreml":
        let model = try await state.loadQwen3TTSCoreML(modelId: variant.modelId)
        return try model.synthesize(text: text, language: language)
    case "cosyvoice":
        let model = try await state.loadCosyVoice(modelId: variant.modelId)
        return model.synthesize(text: text, language: language)
    case "voxcpm2":
        let model = try await state.loadVoxCPM2(modelId: variant.modelId)
        let samples: [Float]
        if let refAudio = cloneReferenceAudio, !refAudio.isEmpty {
            samples = try await model.generateVoxCPM2(
                text: text, language: language,
                refText: cloneReferenceText, refAudio: refAudio)
        } else {
            samples = try await model.generate(text: text, language: language)
        }
        return model.outputSampleRate == 24000
            ? samples
            : resample(samples, from: model.outputSampleRate, to: 24000)
    case "indic-mio":
        let model = try await state.loadIndicMio(modelId: variant.modelId)
        if let refAudio = cloneReferenceAudio, !refAudio.isEmpty {
            return try await model.generate(
                text: text,
                language: language,
                referenceAudio: refAudio,
                referenceSampleRate: 24_000)
        }
        return try await model.generate(text: text, language: language)
    case "magpie":
        let model = try await state.loadMagpie()
        let lang = MagpieLanguage(code: mapToMagpieLanguageCode(language)) ?? .english
        let samples22k = try model.synthesize(text: text, language: lang)
        return resample(samples22k, from: MagpieTTS.sampleRate, to: 24000)
    case "magpie-coreml":
        let model = try await state.loadMagpieCoreML()
        let lang = MagpieCoreMLLanguage(rawValue: mapToMagpieLanguageCode(language)) ?? .english
        let samples22k = try model.synthesize(text: text, language: lang)
        return resample(samples22k, from: MagpieTTSCoreML.sampleRate, to: 24000)
    case "vibevoice":
        let model = try await state.loadVibeVoice(modelId: variant.modelId)
        return try await model.generate(text: text, language: language)
    case "vibevoice-1.5b":
        let model = try await state.loadVibeVoice15B(modelId: variant.modelId)
        return try await model.generate(text: text, language: language)
    default:
        throw RealtimeDispatchError.engineNotEnabled(kind: "TTS", engine: variant.engine)
    }
}

/// Errors raised by the registry-driven dispatchers when an engine is
/// named but not wired into the build.
enum RealtimeDispatchError: Error, CustomStringConvertible {
    case engineNotEnabled(kind: String, engine: String)

    var description: String {
        switch self {
        case .engineNotEnabled(let kind, let engine):
            return "\(kind) engine '\(engine)' is not enabled in this build"
        }
    }
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

/// Map a language name to the 2-letter Hibiki source-language code.
/// Hibiki Zero-3B supports French / Spanish / Portuguese / German as
/// source. Unknown values default to French (the model's strongest
/// language); Hibiki auto-detects in practice so the code is a hint
/// rather than a hard constraint.
func mapToHibikiSourceLanguage(_ language: String) -> String {
    let lower = language.lowercased()
    switch lower {
    case "fr", "french": return "fr"
    case "es", "spanish": return "es"
    case "pt", "portuguese": return "pt"
    case "de", "german": return "de"
    default: return "fr"
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
