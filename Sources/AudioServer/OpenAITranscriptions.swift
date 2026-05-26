import Foundation
import Hummingbird
import HummingbirdCore
import NIOCore
import AudioCommon

// MARK: - OpenAI-compatible /v1/audio/transcriptions

/// Handle POST /v1/audio/transcriptions.
///
/// Accepts multipart/form-data with the standard OpenAI Whisper fields
/// (`file`, `model`, `language`, `response_format`, `prompt`, `temperature`).
/// JSON bodies with `audio_base64` are also accepted for parity with
/// `/transcribe`, since some clients prefer JSON over multipart.
///
/// Like the `/v1/realtime` handler, transcription always runs on the single
/// local ASR backend (Qwen3-ASR). The `model` field (`whisper-1`,
/// `gpt-4o-transcribe`, …) is accepted for client compatibility but does not
/// select a backend — there is only one.
func handleOpenAITranscriptions(
    request: Request,
    state: ModelState
) async throws -> Response {
    let body = try await request.body.collect(upTo: 50 * 1024 * 1024)
    let contentType = request.headers[.contentType] ?? ""

    let audioData: Data
    let language: String?
    let responseFormat: String

    if contentType.contains("multipart/form-data") {
        guard let boundary = parseBoundary(contentType) else {
            return openAIErrorResponse("Missing multipart boundary", status: .badRequest)
        }
        let parts = MultipartParser.parse(Data(buffer: body), boundary: boundary)
        let formFields = Dictionary(
            uniqueKeysWithValues: parts.compactMap { part -> (String, MultipartPart)? in
                guard let name = part.name else { return nil }
                return (name, part)
            })

        guard let filePart = formFields["file"], !filePart.body.isEmpty else {
            return openAIErrorResponse("Missing 'file' field", status: .badRequest)
        }
        audioData = filePart.body
        language = formFields["language"]?.stringValue
        responseFormat = formFields["response_format"]?.stringValue ?? "json"
    } else {
        // Fallback: reuse the existing JSON / raw-WAV parsing used by /transcribe.
        let params = try RequestParams.parse(body, contentType: contentType)
        guard let data = params.audioData else {
            return openAIErrorResponse("Missing 'file' field", status: .badRequest)
        }
        audioData = data
        language = params.string("language")
        responseFormat = params.string("response_format") ?? "json"
    }

    let asr = try await state.loadASR()
    let sampleRate = asr.inputSampleRate
    let audio: [Float]
    do {
        audio = try decodeWAVData(audioData, targetSampleRate: sampleRate)
    } catch {
        return openAIErrorResponse(
            "Could not decode audio: \(error.localizedDescription)",
            status: .badRequest)
    }

    let result = asr.transcribeWithLanguage(
        audio: audio, sampleRate: sampleRate, language: language)
    let duration = Double(audio.count) / Double(sampleRate)
    let durationRounded = round(duration * 100) / 100

    switch responseFormat {
    case "text":
        return Response(
            status: .ok,
            headers: [.contentType: "text/plain; charset=utf-8"],
            body: .init(byteBuffer: .init(string: result.text)))

    case "verbose_json":
        // Minimal verbose_json: a single segment covering the whole utterance.
        // Word-level timestamps and proper segmentation are a follow-up.
        let segment: [String: Any] = [
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": durationRounded,
            "text": result.text,
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": 0.0,
            "compression_ratio": 0.0,
            "no_speech_prob": 0.0
        ]
        var json: [String: Any] = [
            "task": "transcribe",
            "language": result.language ?? language ?? "english",
            "duration": durationRounded,
            "text": result.text,
            "segments": [segment]
        ]
        if (json["language"] as? String)?.isEmpty ?? true { json["language"] = "english" }
        return jsonResponse(json)

    case "srt":
        let srt = formatSRT(text: result.text, duration: duration)
        return Response(
            status: .ok,
            headers: [.contentType: "application/x-subrip; charset=utf-8"],
            body: .init(byteBuffer: .init(string: srt)))

    case "vtt":
        let vtt = formatVTT(text: result.text, duration: duration)
        return Response(
            status: .ok,
            headers: [.contentType: "text/vtt; charset=utf-8"],
            body: .init(byteBuffer: .init(string: vtt)))

    default:  // "json" and any unrecognized value
        // OpenAI's plain `json` returns only {"text": "..."} — keep the shape
        // strict so typed deserializers in the openai-* SDKs don't reject it.
        // Rich metadata (duration, language, segments) belongs in verbose_json.
        return jsonResponse(["text": result.text])
    }
}

/// OpenAI-shaped error envelope:
/// `{"error": {"message": "...", "type": "invalid_request_error", "code": null, "param": null}}`.
/// The official openai-python / openai-node SDKs parse `error.message` as a
/// nested field; returning a bare `{"error": "string"}` crashes their error
/// handling. Kept local to this file so the shared `errorResponse` used by
/// other endpoints (e.g. `/transcribe`) keeps its existing shape.
private func openAIErrorResponse(
    _ message: String,
    status: HTTPResponse.Status
) -> Response {
    let type: String
    switch status {
    case .unauthorized, .forbidden:
        type = "authentication_error"
    case .internalServerError, .badGateway, .serviceUnavailable:
        type = "server_error"
    default:
        type = "invalid_request_error"
    }
    let envelope: [String: Any] = [
        "error": [
            "message": message,
            "type": type,
            "code": NSNull(),
            "param": NSNull()
        ]
    ]
    let data = (try? JSONSerialization.data(
        withJSONObject: envelope, options: [])) ?? Data()
    return Response(
        status: status,
        headers: [.contentType: "application/json"],
        body: .init(byteBuffer: .init(data: data)))
}

private func parseBoundary(_ contentType: String) -> String? {
    // Content-Type: multipart/form-data; boundary=----WebKitFormBoundary...
    for part in contentType.split(separator: ";") {
        let trimmed = part.trimmingCharacters(in: .whitespaces)
        if trimmed.lowercased().hasPrefix("boundary=") {
            var value = String(trimmed.dropFirst("boundary=".count))
            if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value.isEmpty ? nil : value
        }
    }
    return nil
}

private func formatTimestamp(_ seconds: Double, comma: Bool) -> String {
    let total = max(0, seconds)
    let h = Int(total) / 3600
    let m = (Int(total) % 3600) / 60
    let s = Int(total) % 60
    let ms = Int((total - floor(total)) * 1000)
    let sep = comma ? "," : "."
    return String(format: "%02d:%02d:%02d\(sep)%03d", h, m, s, ms)
}

private func formatSRT(text: String, duration: Double) -> String {
    "1\n\(formatTimestamp(0, comma: true)) --> \(formatTimestamp(duration, comma: true))\n\(text)\n"
}

private func formatVTT(text: String, duration: Double) -> String {
    "WEBVTT\n\n\(formatTimestamp(0, comma: false)) --> \(formatTimestamp(duration, comma: false))\n\(text)\n"
}

// MARK: - Minimal multipart/form-data parser (RFC 7578)

struct MultipartPart {
    let headers: [String: String]
    let body: Data

    var name: String? {
        guard let cd = headers["content-disposition"] else { return nil }
        return parseDispositionParam(cd, key: "name")
    }

    var filename: String? {
        guard let cd = headers["content-disposition"] else { return nil }
        return parseDispositionParam(cd, key: "filename")
    }

    var stringValue: String? {
        String(data: body, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

private func parseDispositionParam(_ header: String, key: String) -> String? {
    for part in header.split(separator: ";") {
        let trimmed = part.trimmingCharacters(in: .whitespaces)
        let prefix = "\(key)="
        if trimmed.lowercased().hasPrefix(prefix) {
            var value = String(trimmed.dropFirst(prefix.count))
            if value.hasPrefix("\"") && value.hasSuffix("\"") && value.count >= 2 {
                value = String(value.dropFirst().dropLast())
            }
            return value
        }
    }
    return nil
}

enum MultipartParser {
    static func parse(_ data: Data, boundary: String) -> [MultipartPart] {
        let dashBoundary = Data("--\(boundary)".utf8)
        let headerSep = Data([0x0D, 0x0A, 0x0D, 0x0A])
        var parts: [MultipartPart] = []

        var markers: [Int] = []
        var search = data.startIndex
        while let range = data.range(of: dashBoundary, in: search..<data.endIndex) {
            let after = range.upperBound
            // A valid boundary must be followed by CRLF (next part) or "--"
            // (closing). This guards against `--boundary` byte sequences that
            // happen to appear inside a binary payload.
            let validMarker: Bool
            if after + 1 < data.endIndex,
               data[after] == 0x0D, data[after + 1] == 0x0A {
                validMarker = true
            } else if after + 1 < data.endIndex,
                      data[after] == 0x2D, data[after + 1] == 0x2D {
                validMarker = true
            } else if after == data.endIndex {
                validMarker = true
            } else {
                validMarker = false
            }
            if validMarker {
                markers.append(range.lowerBound)
            }
            search = range.upperBound
        }
        guard markers.count >= 2 else { return parts }

        for i in 0..<(markers.count - 1) {
            var cursor = markers[i] + dashBoundary.count
            let blockEnd = markers[i + 1]
            guard cursor < blockEnd else { continue }

            // Skip CRLF that follows the opening boundary marker.
            if cursor + 1 < data.endIndex,
               data[cursor] == 0x0D, data[cursor + 1] == 0x0A {
                cursor += 2
            }

            guard let sep = data.range(of: headerSep, in: cursor..<blockEnd) else {
                continue
            }
            let headerBytes = data[cursor..<sep.lowerBound]

            // Body ends just before the CRLF that precedes the next boundary.
            var bodyEnd = blockEnd
            if bodyEnd - 2 >= sep.upperBound,
               data[bodyEnd - 2] == 0x0D, data[bodyEnd - 1] == 0x0A {
                bodyEnd -= 2
            }
            let bodyBytes = data[sep.upperBound..<bodyEnd]

            let headersString = String(data: headerBytes, encoding: .utf8) ?? ""
            var headers: [String: String] = [:]
            for line in headersString.components(separatedBy: "\r\n") {
                guard let colon = line.firstIndex(of: ":") else { continue }
                let key = line[..<colon]
                    .trimmingCharacters(in: .whitespaces).lowercased()
                let value = line[line.index(after: colon)...]
                    .trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
            parts.append(MultipartPart(headers: headers, body: Data(bodyBytes)))
        }
        return parts
    }
}
