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
            CSMCommand.self,
            RespondCommand.self,
            VoiceChatCommand.self,
            AudioTranslateCommand.self,
            VadCommand.self,
            VadStreamCommand.self,
            TurnCommand.self,
            DiarizeCommand.self,
            EmbedSpeakerCommand.self,
            LanguageIDCommand.self,
            DenoiseCommand.self,
            RestoreCommand.self,
            SeparateCommand.self,
            AnalyzeMusicCommand.self,
            PracticeCommand.self,
            ComposeCommand.self,
            SA3ParityCommand.self,
            UpsampleCommand.self,
            KokoroCommand.self,
            Qwen3TTSCoreMLCommand.self,
            VibeVoiceCommand.self,
            VibeVoiceEncodeCommand.self,
            TranslateCommand.self,
            GLiNERCommand.self,
            ClefCommand.self,
            InklingCommand.self,
            WakeCommand.self,
            AvatarMotionCommand.self,
        ]
    )

    public init() {}
}
