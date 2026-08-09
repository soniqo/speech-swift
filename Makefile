.PHONY: build debug test clean

CONFIG ?= release

build:
	swift build -c release --disable-sandbox
	./scripts/build_mlx_metallib.sh release

debug:
	swift build -c debug --disable-sandbox
	./scripts/build_mlx_metallib.sh debug

test: debug
	swift test --skip E2E --filter "WAVParsingSecurityTests|DownloadSecurityTests|MetallibScriptTests|MakefileTestSelectionTests|DERScoringTests|SpectralClusteringTests|Qwen3TTSConfigTests|CodePredictorFrameTests|CosyVoiceTTSConfigTests|SamplingTests|PersonaPlexTests[.]PersonaPlexTests/|ForcedAlignerTests/testText|ForcedAlignerTests/testTimestamp|ForcedAlignerTests/testLIS|SileroVADTests/testSilero|SileroVADTests/testReflection|SileroVADTests/testProcess|SileroVADTests/testReset|SileroVADTests/testDetect|SileroVADTests/testStreaming|SileroVADTests/testVADEvent|MemoryManagementTests|CosyVoiceMemoryTests|SpeakerEncoderUnitTests|PCMConversionTests|ResampleTests|FormatJSONTests|RealtimeAPITests|MultipartParserTests"

clean:
	swift package clean
