# Streaming Audio Playback

## The Buffer Underflow Problem

TTS generates audio in chunks. Each chunk takes variable time to generate (~1s for 2s of audio with Qwen3-TTS). If chunks are scheduled to `AVAudioPlayerNode` as they arrive, gaps between chunks cause silence and pops — **buffer underflow**.

```
Without pre-buffer:
  TTS:    [chunk1]---wait 870ms---[chunk2]---wait---[chunk3]
  Player: [play][silence/pop][play][silence/pop][play]
```

## Pre-Buffer Solution

`StreamingAudioPlayer` accumulates TTS samples in a pre-buffer before starting playback. Once the buffer reaches `preBufferDuration` seconds, playback begins from the accumulated audio. Since TTS generation is faster than real-time (RTF < 1.0), the buffer level only grows — no underflow.

```
With pre-buffer (2s):
  TTS:    [chunk1][chunk2][chunk3][chunk4]...
               ↓
  Buffer: [accumulate 2s]→[continuous drain]
               ↓ start playback here
  Player: [continuous audio, no gaps]
```

## Latency vs Quality Tradeoff

| `preBufferDuration` | First-audio latency | Use case |
|---------------------|--------------------:|----------|
| 0 | ~130ms | Kokoro (single-pass, all audio at once) |
| 1.0 | ~1.1s | Fast TTS, low-latency voice assistant |
| 2.0 | ~2.1s | Streaming TTS (recommended for Qwen3-TTS) |
| 3.0 | ~3.1s | High jitter, Bluetooth audio |

Choose based on your TTS engine's RTF and the audio output route.

## Usage

```swift
let player = StreamingAudioPlayer()
player.preBufferDuration = 2.0  // 2s pre-buffer for streaming TTS
try player.start(sampleRate: 24000)

// TTS generates chunks asynchronously
for try await chunk in ttsStream {
    try player.play(samples: chunk.samples, sampleRate: 24000)
}

// Signal end of stream — flushes remaining buffer + sentinel
player.markGenerationComplete()

// Callback fires after ALL audio has played through the speaker
player.onPlaybackFinished = {
    print("Playback complete")
}
```

For single-pass TTS (Kokoro):

```swift
let player = StreamingAudioPlayer()
player.preBufferDuration = 0  // No pre-buffer needed
try player.start(sampleRate: 24000)
try player.play(samples: allSamples, sampleRate: 24000)
player.markGenerationComplete()
```

## End-of-Stream: Sentinel Buffer

`markGenerationComplete()` schedules a 50ms silent sentinel buffer after the last real chunk. `AVAudioPlayerNode` plays buffers in FIFO order, so when the sentinel's `.dataPlayedBack` callback fires, all preceding real audio has been physically played through the speaker. This eliminates the ~400ms hardware pipeline latency that causes the last few words to be cut off.

## Generation Timing Reference

| TTS Engine | RTF | Chunk size | Chunk interval | Recommended pre-buffer |
|------------|-----|-----------|----------------|----------------------:|
| Qwen3-TTS 0.6B (4-bit) | 0.53 | 2.0s | ~1.07s | 2.0s |
| CosyVoice3 (4-bit) | 0.59 | ~150ms | ~100ms | 1.0s |
| Kokoro-82M | N/A | all at once | ~45ms | 0s |
