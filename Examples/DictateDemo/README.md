# DictateDemo

macOS menu bar dictation app powered by Parakeet EOU 120M streaming ASR. Speak into your microphone and see real-time partial transcripts, then paste into any app.

## Features

- Menu bar app with `Cmd+Shift+D` hotkey
- Live streaming transcription (partial results update as you speak)
- End-of-utterance detection (auto-commits sentences)
- Paste to frontmost app with `Cmd+Shift+V`
- Floating HUD with audio level indicator
- 18x real-time on Apple Silicon (RTF 0.056)

## Build

```bash
cd Examples/DictateDemo
swift build
.build/debug/DictateDemo
```

## Usage

1. Launch — model downloads automatically on first run (~112 MB)
2. Click the mic icon in the menu bar
3. Click "Start Dictation" or press `Cmd+Shift+D`
4. Speak — partial transcripts appear in real time
5. Click "Paste to App" to insert text into the active application
