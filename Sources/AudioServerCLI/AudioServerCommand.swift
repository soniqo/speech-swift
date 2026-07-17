import Foundation
import ArgumentParser
import AudioServer

@main
struct AudioServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "speech-server",
        abstract: "HTTP API server for speech models on Apple Silicon"
    )

    @Option(name: .long, help: "Host to bind (default: 127.0.0.1)")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Port to bind (default: 8080)")
    var port: Int = 8080

    @Flag(name: .long, help: "Load all models on startup (slower start, faster first request)")
    var preload: Bool = false

    func run() async throws {
        if let argv0 = CommandLine.arguments.first,
           (argv0 as NSString).lastPathComponent == "audio-server" {
            FileHandle.standardError.write(Data(
                "warning: `audio-server` is a deprecated alias and will be removed in a future release — use `speech-server` instead.\n".utf8
            ))
        }

        let server = AudioServer(host: host, port: port)

        if preload {
            print("Preloading models...")
            try await server.preloadModels()
            print("All models loaded.")
        }

        print("Starting server on http://\(host):\(port)")
        print("Endpoints:")
        print("  POST /transcribe     - Speech-to-text (WAV body or JSON with audio_base64)")
        print("  POST /v1/audio/transcriptions - OpenAI-compatible STT (multipart/form-data)")
        print("  POST /v1/audio/speech - OpenAI-compatible TTS (JSON body)")
        print("  POST /speak          - Text-to-speech (JSON: {text, engine?, language?})")
        print("  POST /respond        - Speech-to-speech (WAV body, voice/max_steps via query)")
        print("  POST /enhance        - Speech enhancement (WAV body)")
        print("  GET  /health         - Health check")
        print("  WS   /v1/realtime    - OpenAI Realtime API (JSON events, base64 PCM16 audio)")

        try await server.run()
    }
}
