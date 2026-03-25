# Voice Pipeline

## Overview

C++ pipeline engine (`speech-core` xcframework) orchestrating VAD, STT, LLM, and TTS with Swift model bridges. The pipeline manages the full voice interaction loop: detecting when the user speaks, transcribing their speech, optionally generating a response via LLM, and synthesizing audio output.

## State Machine

```
Idle --> [push_audio + VAD onset] --> Listening
Listening --> [VAD offset + silence confirmed] --> Transcribing
Transcribing --> [STT complete, echo mode] --> Speaking
Transcribing --> [STT complete, pipeline mode] --> Thinking
Thinking --> [LLM complete] --> Speaking
Speaking --> [resume_listening()] --> Idle
Any --> [user interruption] --> Listening (TTS cancelled)
```

- **Idle**: No audio processing. Waiting for `push_audio` calls.
- **Listening**: VAD has detected speech onset. Audio is accumulating for STT.
- **Transcribing**: VAD detected speech offset and silence was confirmed. STT is running on the accumulated audio.
- **Thinking**: STT produced a transcription, now waiting for LLM to generate a response (pipeline mode only).
- **Speaking**: TTS is synthesizing and playing audio. Transitions back to Idle when playback completes and `resume_listening()` is called.

User interruption from any state cancels the current TTS output (if playing) and returns to Listening.

## Event Flow

The pipeline emits events through the event bridge as the state machine progresses:

| Event | Fired when |
|-------|-----------|
| `sessionCreated` | Pipeline initialized and ready |
| `speechStarted` | VAD onset detected |
| `speechEnded` | VAD offset confirmed (silence threshold met) |
| `transcriptionCompleted` | STT finished, transcription text available |
| `responseCreated` | LLM response generated (pipeline mode) |
| `responseInterrupted` | User interrupted during TTS playback |
| `responseAudioDelta` | TTS audio chunk available for playback |
| `responseDone` | TTS playback completed |
| `toolCallStarted` | LLM invoked a tool (pipeline mode with tools) |
| `toolCallCompleted` | Tool execution finished, result available |
| `error` | Pipeline error (STT failure, TTS failure, etc.) |

## Modes

### Echo

STT transcription is echoed back via TTS — the pipeline speaks back what the user said. No LLM is involved. Useful for testing the full audio loop (mic → VAD → STT → TTS → speaker) without requiring an LLM model.

### TranscribeOnly

STT only. The pipeline transcribes user speech and emits `transcriptionCompleted` but does not generate a TTS response. Useful for dictation or transcription-only applications.

### Pipeline

Full voice agent: STT → LLM → TTS. The user's transcription is sent to the LLM, the LLM response is synthesized via TTS, and the audio is played back. This is the primary mode for conversational AI.

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vadOnset` | 0.5 | VAD speech onset threshold (0-1) |
| `vadOffset` | 0.35 | VAD speech offset threshold (0-1) |
| `minSilenceDuration` | 0.5s | Silence duration to confirm end of utterance |
| `allowInterruptions` | true | Whether user speech cancels TTS playback |
| `minInterruptionDuration` | 0.3s | Minimum speech duration to trigger interruption |
| `postPlaybackGuard` | 0.5s | Ignore VAD onsets for this duration after TTS finishes (prevents self-triggering) |
| `eagerSTT` | false | Start STT while user is still speaking (lower latency, risk of partial input) |
| `maxResponseDuration` | 30s | Maximum TTS playback duration before auto-stop |
| `maxUtteranceDuration` | 30s | Maximum recording duration before forced STT |
| `warmupSTT` | true | Run a warmup inference on pipeline init to precompile CoreML/MLX |

## Architecture

### Worker Thread

The pipeline processes utterances on a dedicated worker thread. Each utterance flows through the pipeline sequentially:

```
Worker thread:
  1. STT inference (accumulated audio → text)
  2. Process utterance (echo text back / send to LLM)
  3. TTS synthesis (text → audio chunks → speaker)
  4. Wait for playback to complete
```

Utterances are processed one at a time. If multiple utterances arrive while one is being processed, they queue up and are handled in order.

### Audio Flow

`push_audio` is called on the microphone thread with raw PCM samples. Inside `push_audio`, the pipeline mutex is held briefly to run VAD on the incoming audio. If VAD detects speech, the audio is appended to the utterance buffer.

STT runs on the worker thread **without** the pipeline mutex held. This means audio continues flowing into the VAD during STT inference — if the user starts speaking again during transcription, the new speech is detected immediately.

### TTS Blocking

TTS synthesis blocks the worker thread via the synthesize callback. The TTS bridge calls back into Swift to run the TTS model, and audio chunks are fed to the audio player as they are generated. The worker thread does not proceed to the next utterance until TTS playback completes. Concurrent TTS is not supported — it would cause interleaved audio from multiple utterances.

### Interruptions

When the user speaks during TTS playback (and `allowInterruptions` is true):

1. TTS synthesis is cancelled (callback returns early)
2. LLM generation is cancelled (if running)
3. Speech queue is cleared
4. Audio player fades out and stops
5. Turn detector resets
6. State transitions to Listening for the new utterance

## Swift Integration (VoicePipeline.swift)

The Swift layer bridges the C++ pipeline engine with Swift model implementations via FFI:

| Bridge | Purpose |
|--------|---------|
| `STTBridge` | Wraps Qwen3-ASR or Parakeet for speech-to-text |
| `TTSBridge` | Wraps Qwen3-TTS, CosyVoice, or Kokoro for text-to-speech |
| `VADBridge` | Wraps Silero VAD for voice activity detection |
| `LLMBridge` | Wraps Qwen3-Chat or external LLM for response generation |
| `EventBridge` | Receives C++ events and dispatches to Swift handlers |

Events from the C++ pipeline are dispatched to the main queue for UI updates:

```swift
eventBridge.onEvent = { event in
    DispatchQueue.main.async {
        self.handlePipelineEvent(event)
    }
}
```

## Known Limitations

- **STT blocks worker for 2-3s** (Parakeet CoreML on Neural Engine). Phrases spoken during STT inference queue up and are processed in the next cycle. Enabling `eagerSTT` can reduce perceived latency but risks cutting off the user mid-sentence.

- **TTS blocks worker** — concurrent TTS is not supported. If a second utterance arrives while TTS is still playing, it waits in the queue. This prevents interleaved audio but adds latency for rapid back-and-forth conversations.

- **AEC quality depends on Apple Voice Processing** — echo cancellation is imperfect, especially on long playback or with external speakers. The `postPlaybackGuard` parameter helps prevent self-triggering but does not eliminate all echo artifacts. See [docs/audio/playback.md](playback.md) for details on the PlayerNode vs SourceNode tradeoff.
