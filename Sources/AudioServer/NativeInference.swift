import AudioCommon
import Foundation
import Hummingbird
import NIOCore
import MLX
import Qwen3Chat
import SpeechVAD
import VoxCPM2TTS
import WhisperASR

/// Explicit local directories: this service never downloads or substitutes models.
public struct NativeInferenceConfiguration: Codable, Sendable {
    public let runtimeRevision: String
    public let gemmaDirectory: String
    public let agentDirectory: String?
    public let voxDirectory: String
    public let whisperDirectory: String
    public let vadDirectory: String
    public let assets: ChatJSONValue?
    public let failureCaptureDirectory: String?

    public static func load(from url: URL) throws -> Self {
        try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }
}

struct NativeInferenceRequest: Decodable, Sendable {
    enum Operation: String, Decodable, Sendable { case chat, speech, transcribe, vad }
    let operation: Operation
    let requestID: String
    let model: String
    var messages: [Gemma4ConversationMessage]?
    var tools: [Gemma4Tool]?
    var maxTokens: Int?
    var temperature: Float?
    var text: String?
    var voiceInstruction: String?
    var audioBase64: String?
    var sampleRate: Int?
    var stream: Bool?
    var streamID: String?
    var reset: Bool?
    var seed: UInt64?
    var endStream: Bool?

    func validate() throws {
        guard !requestID.isEmpty, requestID.count <= 128 else { throw HTTPError(.badRequest, message: "Invalid requestID") }
        let expected: String
        switch operation {
        case .chat: expected = model == "gemma4-12b-it" ? "gemma4-12b-it" : "gemma4-e4b-it"
        case .speech: expected = "voxcpm2"
        case .transcribe: expected = "whisper-large-v3"
        case .vad: expected = "silero-vad"
        }
        guard model == expected else { throw HTTPError(.badRequest, message: "Model unavailable for operation: \(model)") }
        if let maxTokens, !(1...4096).contains(maxTokens) { throw HTTPError(.badRequest, message: "maxTokens must be 1...4096") }
        if let temperature, !temperature.isFinite || !(0...2).contains(temperature) { throw HTTPError(.badRequest, message: "Invalid temperature") }
        switch operation {
        case .chat:
            guard let messages, !messages.isEmpty else { throw HTTPError(.badRequest, message: "messages required") }
        case .speech:
            guard let text, !text.isEmpty, text.count <= 8000 else { throw HTTPError(.badRequest, message: "text must contain 1...8000 characters") }
        case .transcribe, .vad:
            guard let audioBase64, let data = Data(base64Encoded: audioBase64), !data.isEmpty, data.count.isMultiple(of: 2),
                  let sampleRate, (8000...48000).contains(sampleRate), data.count <= sampleRate * 2 * 60 else {
                throw HTTPError(.badRequest, message: "Expected at most 60 seconds of mono PCM16 audio")
            }
            if operation == .vad {
                guard sampleRate == 16000, (data.count / 2).isMultiple(of: 512),
                      let streamID, !streamID.isEmpty, streamID.count <= 128 else {
                    throw HTTPError(.badRequest, message: "VAD requires streamID and complete 512-sample frames at 16 kHz")
                }
            }
        }
    }

    func audio() -> [Float] {
        let bytes = [UInt8](Data(base64Encoded: audioBase64!)!)
        return stride(from: 0, to: bytes.count, by: 2).map {
            Float(Int16(bitPattern: UInt16(bytes[$0]) | UInt16(bytes[$0 + 1]) << 8)) / 32768
        }
    }
}

/// Holds a permit across suspension points so model loads and inference cannot overlap.
actor NativeInferenceGate {
    private var busy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if busy { await withCheckedContinuation { waiters.append($0) } }
        else { busy = true }
    }

    func release() {
        if waiters.isEmpty { busy = false }
        else { waiters.removeFirst().resume() }
    }
}

final class NativeInferenceCancellations: @unchecked Sendable {
    private let lock = NSLock()
    private var tokens: [String: InferenceCancellationToken] = [:]

    func insert(_ id: String) throws -> InferenceCancellationToken {
        try lock.withLock {
            guard tokens[id] == nil else { throw HTTPError(.conflict, message: "Duplicate active requestID") }
            let token = InferenceCancellationToken()
            tokens[id] = token
            return token
        }
    }

    func remove(_ id: String) { _ = lock.withLock { tokens.removeValue(forKey: id) } }
    func cancel(_ id: String) -> Bool {
        lock.withLock {
            guard let token = tokens[id] else { return false }
            token.cancel()
            return true
        }
    }
}

private actor NativeVADStreams {
    let model: SileroVADModel
    var streams: [String: SileroVADModel] = [:]
    init(model: SileroVADModel) { self.model = model }

    func predict(_ request: NativeInferenceRequest) throws -> [Float] {
        let id = request.streamID!
        if request.reset == true { streams.removeValue(forKey: id) }
        if streams[id] == nil {
            guard streams.count < 64 else { throw HTTPError(.tooManyRequests, message: "Too many VAD streams; reset an existing stream") }
            streams[id] = model.makeIndependentStream()
        }
        let stream = streams[id]!
        let samples = request.audio()
        let probabilities = try stride(from: 0, to: samples.count, by: 512).map {
            try stream.processChunkThrowing(Array(samples[$0..<$0 + 512]))
        }
        if request.endStream == true { streams.removeValue(forKey: id) }
        return probabilities
    }
}

/// Local chat, speech, transcription, and stateful VAD with explicit model identity.
/// The owner serializes heavy inference; VAD maintains independent lightweight state.
public final class NativeInferenceService: @unchecked Sendable {
    private let config: NativeInferenceConfiguration
    private let gemma: Gemma4Chat
    private let agent: Gemma4UnifiedChat?
    private let vox: VoxCPM2TTSModel
    private let whisper: WhisperASRModel
    private let vad: NativeVADStreams
    private let gate = NativeInferenceGate()
    private let cancellations = NativeInferenceCancellations()
    private let failures: NativeInferenceFailures?

    public init(configuration: NativeInferenceConfiguration) async throws {
        self.config = configuration
        self.failures = try configuration.failureCaptureDirectory.map {
            try NativeInferenceFailures(directory: URL(fileURLWithPath: $0))
        }
        Memory.cacheLimit = 128 * 1024 * 1024
        Memory.memoryLimit = 20 * 1024 * 1024 * 1024
        Memory.clearCache()
        let gemmaFolder = URL(fileURLWithPath: configuration.gemmaDirectory)
        let gemmaJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: gemmaFolder.appendingPathComponent("config.json"))) as? [String: Any]
        let quantization = gemmaJSON?["quantization"] as? [String: Any]
        let textConfig = gemmaJSON?["text_config"] as? [String: Any]
        guard quantization?["bits"] as? Int == 4,
              textConfig?["num_hidden_layers"] as? Int == 42,
              textConfig?["hidden_size"] as? Int == 2560 else {
            throw HTTPError(.badRequest, message: "Gemma requires the E4B four-bit checkpoint")
        }
        self.gemma = try Gemma4Chat.fromDirectory(gemmaFolder)
        if let directory = configuration.agentDirectory {
            self.agent = try Gemma4UnifiedChat.fromDirectory(URL(fileURLWithPath: directory))
        } else { self.agent = nil }
        self.vox = try await VoxCPM2TTSModel.fromPretrained(
            cacheDir: URL(fileURLWithPath: configuration.voxDirectory),
            tokenizerDirectory: URL(fileURLWithPath: configuration.voxDirectory),
            offlineMode: true, weightPrecision: .float16)
        let whisperFolder = URL(fileURLWithPath: configuration.whisperDirectory)
        let whisperJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: whisperFolder.appendingPathComponent("config.json"))) as? [String: Any]
        guard whisperJSON?["decoder_layers"] as? Int == 32 else {
            throw HTTPError(.badRequest, message: "Full Whisper Large-v3 requires 32 decoder layers")
        }
        self.whisper = try WhisperASRModel.fromDirectory(whisperFolder, modelId: "whisper-large-v3")
        self.vad = NativeVADStreams(model: try await SileroVADModel.fromPretrained(
            engine: .coreml, cacheDir: URL(fileURLWithPath: configuration.vadDirectory), offlineMode: true))
    }

    var capabilities: [String: Any] {
        func model(_ id: String, _ engine: String, _ precision: String?, _ features: [String]) -> [String: Any] {
            var row: [String: Any] = ["model_id": id, "engine": "speech-swift/" + engine, "features": features]
            if let precision { row["precision"] = precision }
            return row
        }
        var models: [String: Any] = [
            "gemma4-e4b-it": model("gemma4-e4b-it", "gemma4", "int4", ["tool_calls", "tool_history", "json_output", "cancel"]),
            "voxcpm2": model("voxcpm2", "voxcpm2", "fp16", ["streaming", "cancel"]),
            "whisper-large-v3": model("whisper-large-v3", "whisper", "fp16", []),
            "silero-vad": model("silero-vad", "silero", nil, ["streaming"]),
        ]
        if let agent {
            var features = ["tool_calls", "tool_history", "cancel"]
            if agent.prefixCacheBudgetBytes > 0 { features.append("system_prefix_cache") }
            models["gemma4-12b-it"] = model("gemma4-12b-it", "gemma4-unified", "int4", features)
        }
        return ["schema_version": 1, "runtime": "speech-swift", "revision": config.runtimeRevision, "models": models]
    }

    private static func pcm(_ samples: [Float]) throws -> String {
        guard samples.allSatisfy(\.isFinite) else { throw HTTPError(.internalServerError, message: "Non-finite generated audio") }
        var bytes = Data(capacity: samples.count * 2)
        for sample in samples {
            var value = Int16(max(-32768, min(32767, sample * 32768))).littleEndian
            withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
        }
        return bytes.base64EncodedString()
    }

    private func perform(_ request: NativeInferenceRequest, token: InferenceCancellationToken,
                         audioChunk: (@Sendable ([Float]) -> Void)? = nil,
                         onGeneration: (@Sendable (Gemma4GenerationTrace) -> Void)? = nil) async throws -> Data {
        if request.operation == .vad {
            return try JSONSerialization.data(withJSONObject: ["probabilities": try await vad.predict(request)])
        }
        await gate.acquire()
        do {
            try token.checkCancellation()
            let result: Data
            switch request.operation {
            case .chat:
                let sampling = ChatSamplingConfig(temperature: request.temperature ?? 0,
                                                  maxTokens: request.maxTokens ?? 768, repetitionPenalty: 1)
                let response: Gemma4Response
                if request.model == "gemma4-12b-it" {
                    guard let agent else { throw HTTPError(.badRequest, message: "Gemma 4 12B is not loaded") }
                    response = try agent.generateResponse(messages: request.messages!, tools: request.tools ?? [],
                                                          sampling: sampling, cancellation: token, onGeneration: onGeneration)
                } else {
                    response = try gemma.generateResponse(messages: request.messages!, tools: request.tools ?? [],
                                                          sampling: sampling, cancellation: token)
                }
                let encoded = try JSONEncoder().encode(response)
                if request.model == "gemma4-12b-it", let agent {
                    guard var payload = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
                        throw HTTPError(.internalServerError, message: "Invalid native chat response encoding")
                    }
                    payload["cachedPrefixTokens"] = agent.lastCachedPrefixTokens
                    result = try JSONSerialization.data(withJSONObject: payload)
                } else {
                    result = encoded
                }
            case .speech:
                let samples = try await vox.generateVoxCPM2(
                    text: request.text!, maxTokens: request.maxTokens ?? 1500,
                    instruct: request.voiceInstruction, seed: request.seed ?? 42, cancellation: token, onAudioChunk: audioChunk)
                result = try JSONSerialization.data(withJSONObject: ["audioBase64": Self.pcm(samples), "sampleRate": 48000])
            case .transcribe:
                let text = try await whisper.transcribeAudio(request.audio(), sampleRate: request.sampleRate!, language: "en")
                try token.checkCancellation()
                result = try JSONSerialization.data(withJSONObject: ["text": text])
            case .vad: fatalError("VAD handled before heavy inference")
            }
            Memory.clearCache()
            await gate.release()
            return result
        } catch {
            Memory.clearCache()
            await gate.release()
            throw error
        }
    }

    func registerRoutes(on router: Router<BasicRequestContext>) {
        router.get("/v1/inference/capabilities") { _, _ in jsonResponse(self.capabilities) }
        router.get("/v1/inference/provenance") { _, _ in
            let encoded = try JSONEncoder().encode(self.config)
            guard var configuration = try JSONSerialization.jsonObject(with: encoded) as? [String: Any] else {
                throw HTTPError(.internalServerError, message: "Invalid native configuration encoding")
            }
            configuration["gemma4_12b_prefix_cache_budget_bytes"] = self.agent?.prefixCacheBudgetBytes ?? 0
            return jsonResponse([
                "configuration": configuration,
                "pid": ProcessInfo.processInfo.processIdentifier,
                "voxcpm2_parameter_elements": self.vox.parameterDtypes,
                "voxcpm2_audio_vae": "float32",
                "voxcpm2_compute": "mixed; FP16 network inputs and FP32 AudioVAE",
                "mlx_active_bytes": Memory.activeMemory,
                "mlx_cache_bytes": Memory.cacheMemory,
                "mlx_peak_bytes": Memory.peakMemory,
                "mlx_memory_limit_bytes": Memory.memoryLimit,
            ])
        }
        router.post("/v1/inference/cancel") { request, _ in
            struct Cancel: Decodable { let requestID: String }
            let body = try await request.body.collect(upTo: 1024)
            let cancel = try JSONDecoder().decode(Cancel.self, from: Data(body.readableBytesView))
            return jsonResponse(["cancelled": self.cancellations.cancel(cancel.requestID)])
        }
        router.post("/v1/inference") { request, context in
            let body = try await request.body.collect(upTo: 16 * 1024 * 1024)
            let input: NativeInferenceRequest
            do { input = try JSONDecoder().decode(NativeInferenceRequest.self, from: Data(body.readableBytesView)) }
            catch { throw HTTPError(.badRequest, message: "Invalid native inference request: \(error.localizedDescription)") }
            try input.validate()
            let token = try self.cancellations.insert(input.requestID)
            if input.operation == .speech && input.stream == true {
                let stream = AsyncThrowingStream<ByteBuffer, Error> { continuation in
                    let task = Task {
                        defer { self.cancellations.remove(input.requestID) }
                        do {
                            _ = try await self.perform(input, token: token, audioChunk: { samples in
                                do {
                                    let data = try JSONSerialization.data(withJSONObject: [
                                        "event": "audio", "audioBase64": Self.pcm(samples), "sampleRate": 48000])
                                    var buffer = ByteBuffer(data: data); buffer.writeString("\n")
                                    continuation.yield(buffer)
                                } catch { token.cancel(); continuation.finish(throwing: error) }
                            })
                            continuation.yield(ByteBuffer(string: "{\"event\":\"done\"}\n"))
                            continuation.finish()
                        } catch {
                            let data = try? JSONSerialization.data(withJSONObject: ["event": "error", "message": String(describing: error)])
                            if let data { var buffer = ByteBuffer(data: data); buffer.writeString("\n"); continuation.yield(buffer) }
                            continuation.finish()
                        }
                    }
                    continuation.onTermination = { _ in token.cancel(); task.cancel() }
                }
                return Response(status: .ok, headers: [.contentType: "application/x-ndjson"], body: .init(asyncSequence: stream))
            }
            defer { self.cancellations.remove(input.requestID) }
            let capture = self.failures != nil && input.operation == .chat ? NativeInferenceGenerationCapture() : nil
            do {
                let observer: (@Sendable (Gemma4GenerationTrace) -> Void)? = capture.map { capture in
                    { trace in capture.store(trace) }
                }
                let data = try await self.perform(input, token: token, onGeneration: observer)
                return Response(status: .ok, headers: [.contentType: "application/json"], body: .init(byteBuffer: .init(data: data)))
            } catch {
                if input.operation == .chat, let failures = self.failures {
                    do {
                        let file = try failures.write(requestBody: Data(body.readableBytesView),
                                                      requestID: input.requestID, model: input.model,
                                                      runtimeRevision: self.config.runtimeRevision,
                                                      error: error, generation: capture?.trace)
                        context.logger.error("Native chat failure captured", metadata: ["file": .string(file.path)])
                    } catch {
                        context.logger.error("Native chat failure capture could not be written", metadata: [
                            "requestID": .string(input.requestID), "error": .string(String(describing: error)),
                        ])
                    }
                }
                context.logger.error("Native inference failed", metadata: [
                    "requestID": .string(input.requestID),
                    "operation": .string(input.operation.rawValue),
                    "error": .string(String(describing: error)),
                ])
                throw error
            }
        }
    }
}
