import ArgumentParser

public struct AudioCLI: ParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "speech",
        abstract: "AI speech models for Apple Silicon",
        subcommands: [
            TranscribeCommand.self,
            TranscribeBatchCommand.self,
            AlignCommand.self,
            SpeakCommand.self,
            RespondCommand.self,
            AudioTranslateCommand.self,
            VadCommand.self,
            VadStreamCommand.self,
            DiarizeCommand.self,
            EmbedSpeakerCommand.self,
            DenoiseCommand.self,
            RestoreCommand.self,
            SeparateCommand.self,
            ComposeCommand.self,
            SA3ParityCommand.self,
            UpsampleCommand.self,
            KokoroCommand.self,
            Qwen3TTSCoreMLCommand.self,
            VibeVoiceCommand.self,
            VibeVoiceEncodeCommand.self,
            TranslateCommand.self,
            WakeCommand.self,
        ]
    )

    public init() {}
}
