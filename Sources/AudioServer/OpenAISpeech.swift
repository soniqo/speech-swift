import Foundation
import Hummingbird
import HummingbirdCore
import NIOCore
import KokoroTTS

// MARK: - OpenAI-compatible /v1/audio/speech

enum OpenAISpeechResponseFormat: String, Equatable, Sendable {
    case wav
    case pcm
}

struct OpenAISpeechRequest: Equatable, Sendable {
    static let maximumInputCharacters = 4096

    let model: String
    let input: String
    let voice: String
    let responseFormat: OpenAISpeechResponseFormat
    let speed: Float
    let language: String

    private struct Payload: Decodable {
        let model: String?
        let input: String?
        let voice: String?
        let responseFormat: String?
        let speed: Float?
        let language: String?

        enum CodingKeys: String, CodingKey {
            case model
            case input
            case voice
            case responseFormat = "response_format"
            case speed
            case language
        }
    }

    static func parse(_ data: Data) throws -> OpenAISpeechRequest {
        let payload: Payload
        do {
            payload = try JSONDecoder().decode(Payload.self, from: data)
        } catch {
            throw OpenAISpeechRequestError.invalidJSON
        }

        guard let model = payload.model?.trimmingCharacters(in: .whitespacesAndNewlines),
              !model.isEmpty
        else {
            throw OpenAISpeechRequestError.missingField("model")
        }
        guard let input = payload.input,
              !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw OpenAISpeechRequestError.missingField("input")
        }
        guard input.count <= maximumInputCharacters else {
            throw OpenAISpeechRequestError.inputTooLong(maximumInputCharacters)
        }
        guard let voice = payload.voice?.trimmingCharacters(in: .whitespacesAndNewlines),
              !voice.isEmpty
        else {
            throw OpenAISpeechRequestError.missingField("voice")
        }

        let formatName = payload.responseFormat?.lowercased() ?? OpenAISpeechResponseFormat.wav.rawValue
        guard let responseFormat = OpenAISpeechResponseFormat(rawValue: formatName) else {
            throw OpenAISpeechRequestError.unsupportedResponseFormat(formatName)
        }

        let speed = payload.speed ?? 1.0
        guard speed >= 0.25, speed <= 4.0 else {
            throw OpenAISpeechRequestError.speedOutOfRange(speed)
        }

        let language = payload.language?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedLanguage = language.flatMap { $0.isEmpty ? nil : $0 } ?? "english"
        return OpenAISpeechRequest(
            model: model,
            input: input,
            voice: voice,
            responseFormat: responseFormat,
            speed: speed,
            language: resolvedLanguage)
    }
}

enum OpenAISpeechRequestError: LocalizedError, Equatable {
    case invalidJSON
    case missingField(String)
    case inputTooLong(Int)
    case unsupportedResponseFormat(String)
    case speedOutOfRange(Float)

    var errorDescription: String? {
        switch self {
        case .invalidJSON:
            "Invalid JSON request body"
        case .missingField(let field):
            "Missing '\(field)' field"
        case .inputTooLong(let limit):
            "The 'input' field must not exceed \(limit) characters"
        case .unsupportedResponseFormat(let format):
            "Unsupported response format '\(format)'; use 'wav' or 'pcm'"
        case .speedOutOfRange(let speed):
            "Speed \(speed) is outside the supported range 0.25...4.0"
        }
    }
}

/// Handle the OpenAI speech synthesis request shape.
///
/// The local server emits WAV by default because its synthesis engines produce
/// 24 kHz PCM directly. Callers that need headerless PCM can request `pcm`.
/// Compressed OpenAI formats (`mp3`, `opus`, `aac`, `flac`) are rejected rather
/// than returning bytes under the wrong content type.
func handleOpenAISpeech(
    request: Request,
    state: ModelState
) async throws -> Response {
    let body = try await request.body.collect(upTo: 1024 * 1024)

    let speechRequest: OpenAISpeechRequest
    do {
        speechRequest = try OpenAISpeechRequest.parse(Data(buffer: body))
    } catch {
        return openAIErrorResponse(
            error.localizedDescription,
            status: .badRequest)
    }

    guard let variant = resolveModelToTTSVariant(speechRequest.model) else {
        return openAIErrorResponse(
            "Unknown TTS model: \(speechRequest.model)",
            status: .badRequest)
    }
    guard variant.engine == "kokoro" || speechRequest.speed == 1.0 else {
        return openAIErrorResponse(
            "The 'speed' field is supported only by Kokoro models",
            status: .badRequest)
    }

    let voice = resolvedLocalVoice(
        speechRequest.voice,
        engine: variant.engine)

    let samples: [Float]
    do {
        samples = try await dispatchSynthesize(
            text: speechRequest.input,
            variant: variant,
            language: speechRequest.language,
            state: state,
            voice: voice,
            speed: speechRequest.speed)
    } catch {
        return openAIErrorResponse(
            "Speech synthesis failed: \(error.localizedDescription)",
            status: .internalServerError)
    }

    do {
        switch speechRequest.responseFormat {
        case .wav:
            let wavData = try encodeWAV(samples: samples, sampleRate: 24_000)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/wav"],
                body: .init(byteBuffer: .init(data: wavData)))
        case .pcm:
            let pcmData = floatToPCM16LE(samples)
            return Response(
                status: .ok,
                headers: [.contentType: "audio/pcm"],
                body: .init(byteBuffer: .init(data: pcmData)))
        }
    } catch {
        return openAIErrorResponse(
            "Audio encoding failed: \(error.localizedDescription)",
            status: .internalServerError)
    }
}

private let openAIVoiceNames: Set<String> = [
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
    "juniper", "marin", "nova", "onyx", "sage", "shimmer", "verse",
]

/// Generic OpenAI voice names have no matching embeddings in local models.
/// Treat them as a request for the selected engine's default voice. Native
/// model voice names (for example Kokoro's `af_heart`) pass through unchanged.
func resolvedLocalVoice(_ voice: String, engine: String) -> String? {
    let normalized = voice.lowercased()
    if openAIVoiceNames.contains(normalized) {
        return engine == "kokoro" ? KokoroTTSModel.defaultVoice : nil
    }
    switch engine {
    case "kokoro", "qwen3-tts":
        return voice
    default:
        return nil
    }
}
